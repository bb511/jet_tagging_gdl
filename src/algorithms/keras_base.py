"""The base of the models trained quantisation-aware with HGQ2.

We use lightning for the training loop, the data, validation, testing and the invariance
probe; Keras on JAX implements the network, its optimiser and its step.
"""

import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.algorithms import DenormalisedModule
from src.algorithms.components.hgq_port import (
    KEEP_KIF,
    WORD_F_RANGE,
    KerasNetwork,
    Quantisation,
    calibrate,
    configure_jax,
    export_port,
    quantiser_variables,
    require_cpu_jax,
    seed_datalane_bits,
    seed_weight_bits,
    seed_word_bits,
    set_beta,
    total_ebops,
    word_range,
)
from src.algorithms.utils.flops import count_flops_per_jet
from src.callbacks.invariance import InvarianceMetricCallback
from src.utils.pylogger import RankedLogger

# isort: split
# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.layers import QDense, QEinsum, QEinsumDense, QMultiply, QSum, Quantizer
from hgq.utils.sugar import PieceWiseSchedule
from keras import ops

log = RankedLogger(__name__)

# Jets per Keras call on a whole split.
CHUNK = 65536
# Jets per batch of HGQ2's trace, unless a row bounds it.
CALIBRATION_BATCH = 16384
VERSIONED = ("keras", "hgq2", "jax")


class KerasQuantisedModule(DenormalisedModule):
    """A row trained quantisation-aware with HGQ2, for synthesis by alkaid.

    ``net`` is a ``KerasNetwork`` without torch parameters.
    Training is manual: each batch is one Keras ``train_on_batch`` and one step of the
    mock optimiser that Lightning puts in place of a torch one.
    That step advances ``global_step``, without which ``ModelCheckpoint`` never saves.
    The Keras weights then go inside the Lightning checkpoint.
    The optimiser always stays separate.

    At test time the integer bits of every wrapping quantiser are traced on the record's
    whole training split; the test, the probe and the export all read the traced port,
    and ``ebops_total`` holds its EBOPs=NaN until then. Testing runs with JAX on CPU.
    """

    def _keras_net(
        self,
        quantisation: dict[str, Any] | None,
        floatx: str,
        jax_platform: str,
        jax_devices: int | list[int],
    ) -> KerasNetwork:
        """Configure JAX, register the de-normalisation and build the port.

        :param quantisation: the fields of ``Quantisation`` that the config sets.
        """
        configure_jax(jax_platform, jax_devices, floatx)
        # Keras draws the initial weights from a generator of its own, seeded from torch.
        keras.utils.set_random_seed(torch.initial_seed() % 2**32)
        self._register_denorm_scales()
        self.automatic_optimization = False
        self.quantisation = Quantisation(**(quantisation or {}))
        model, words = self._build_port()
        return KerasNetwork(model, words)

    def _build_port(self) -> tuple[keras.Model, keras.Model]:
        """The port and the model of its input words."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_port()."
        )

    @property
    def q(self) -> Quantisation:
        return self.quantisation

    @property
    def nconstituents(self) -> int:
        return self.hparams.nconstituents

    @property
    def units(self) -> float:
        return self.hparams.units

    def _dense(self, units: int, name: str, activation=None, **kwargs):
        """A dense layer on the last axis, its input, kernel and bias each quantised."""
        return QDense(
            units,
            activation=activation,
            iq_conf=self.q.datalane(),
            kq_conf=self.q.weight(),
            bq_conf=self.q.bias(),
            name=name,
            **kwargs,
        )

    def _einsum_dense(self, equation: str, shape: tuple[int, ...], name: str):
        """A bias-free linear map of *equation*, its input and kernel quantised."""
        q = self.q
        return QEinsumDense(
            equation, shape, iq_conf=q.datalane(), kq_conf=q.weight(), name=name
        )

    def _einsum(self, equation: str, name: str, *inputs):
        """The einsum of two tensors, each read through a datalane quantiser."""
        return QEinsum(equation, iq_confs=self.q.datalane(), name=name)(list(inputs))

    def _sum(self, name: str, x, keepdims: bool = False, iq_conf=None):
        """The sum over the slots, axis 1, of *x* read through *iq_conf* (a
        datalane)."""
        iq_conf = iq_conf or self.q.datalane()
        return QSum(iq_conf=iq_conf, axes=1, keepdims=keepdims, name=name)(x)

    def _keep_multiply(self, name: str, x, keep):
        """Zero the padded slots of *x*, the 0/1 flag *keep* read through one fixed
        bit."""
        confs = [self.q.datalane(), self.q.fixed(KEEP_KIF)]
        return QMultiply(iq_confs=confs, name=name)([x, keep])

    def _words(self, x):
        """Quantise the first three columns of rows (jets, N, F) to p_T words (jets, N)
        and angle words (jets, N, 2).

        The words wrap: alkaid traces nothing but WRAP on a model input, so the caller
        clamps the rows to the words' ranges. Trained words have one width per slot, the
        two angles of a slot sharing theirs, so that a rotation of the (eta, phi) plane
        acts on words of equal precision.
        """
        q = self.q
        pt = Quantizer(q.word(q.pt_kif, (0,)), name="pt_word")(x[:, :, 0])
        angles = Quantizer(q.word(q.angle_kif, (0, 2)), name="angle_word")(x[:, :, 1:3])
        return pt, angles

    def _word_rows(self, pt, angles, offset):
        """The words the port reads, (jets, N, 3) [p_T, eta, phi]; under the axis offset
        (jets, N, 5), the two offset words in row 0 and zeros, which nothing reads,
        below them."""
        columns = [ops.expand_dims(pt, -1), angles]
        if offset is not None:
            below = [[0, 0], [0, self.nconstituents - 1], [0, 0]]
            columns.append(ops.pad(ops.expand_dims(offset, 1), below))
        return ops.concatenate(columns, axis=-1)

    def _p_keep(self, pt):
        """Return p, the p_T words (jets, N) in ``units``, and the keep flag (jets, N,
        1), one on a real constituent and zero on a padded one."""
        # A division rather than the product by 1 / units: Keras marks the product of a
        # tensor and a Python scalar sparse, which the float32 build fails to trace.
        # alkaid replays both as the same shift.
        p = pt / self.units
        # Exact: a real p_T is a multiple of its word's LSB, at least 2**-f with f the
        # word's fraction bits or a trained word's greatest, and a cast of a bool has no
        # alkaid handler.
        f = WORD_F_RANGE[1] if self.q.trained_words else self.q.pt_kif[2]
        keep = ops.expand_dims(ops.minimum(pt * 2.0**f, 1.0), -1)
        return p, keep

    def _lifts(self, p, angles):
        """Return the unit lift's tail (eta, phi, eta^2 + phi^2) (jets, N, 3) and the lifts
        u = p (1, tail) (jets, N, 4)."""
        r2 = self._einsum("bnk,bnk->bn", "lift_r2", angles, angles)
        tail = ops.concatenate([angles, ops.expand_dims(r2, -1)], axis=-1)
        u_tail = self._einsum("bn,bnk->bnk", "lift_u", p, tail)
        return tail, ops.concatenate([ops.expand_dims(p, -1), u_tail], axis=-1)

    def _float_twin(self) -> nn.Module:
        """The float network of the HGQ architecture, on which the FLOPs are counted."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _float_twin()."
        )

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the trained float network that ``init_from`` names into the port."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _copy_float_network()."
        )

    def _calibration_batch(self) -> int:
        """Jets per batch of HGQ2's trace."""
        return CALIBRATION_BATCH

    def _record_costs(self, nconstituents: int, nbr_feats: int) -> None:
        """Price the FLOPs on the float network of this architecture."""
        twin = self._float_twin()
        self.flops_total = count_flops_per_jet(twin, nconstituents, nbr_feats)
        self.ebops_total = self.ebops_unpriced = math.nan

    def _clamp_words(self, pt: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Concatenate the (jets, N, 1) p_T and the (jets, N, A) angles.

        Each of these is clamped to the range of the input quantisation: the inputs that
        go beyond the range are wrapped around, supported in alkaid.
        """
        q = self.quantisation
        pt = pt.clamp(*word_range(q.pt_kif))
        return torch.cat([pt, angles.clamp(*word_range(q.angle_kif))], dim=-1)

    def _keras_inputs(self, x: torch.Tensor) -> np.ndarray:
        """The prepared rows of a batch or a whole split, in Keras's float type."""
        # The buffers' device: during setup() Lightning already reports the
        # accelerator as self.device, before it moves the module there.
        device = self.denorm_scales.device
        chunks = (self._network_inputs(c.to(device)) for c in x.split(CHUNK))
        rows = np.concatenate([chunk.cpu().numpy() for chunk in chunks])
        return rows.astype(self.net.keras_model.inputs[0].dtype)

    def setup(self, stage: str) -> None:
        """Resolve the feature layout and, before a fit, initialise the port.

        :raises ValueError: under the SCL, which the Keras step does not compute.
        """
        super().setup(stage)
        datamodule = getattr(self._trainer, "datamodule", None)
        if stage != "fit" or datamodule is None:
            return
        if self.symm_loss is not None:
            raise ValueError(
                f"The Keras step of {self.hparams.name} has no symmetry-consistency "
                "loss; train it under data.method=augmentation."
            )
        self._initialise(datamodule)

    def _initialise(self, datamodule) -> None:
        """Copy the float network ``init_from`` names, if any, then seed the bits of the
        weights and, on the first ``calibration_jets`` training jets, of the
        datalanes."""
        model, q = self.net.keras_model, self.quantisation
        if self.hparams.init_from:
            log.info(f"Copying the float network of {self.hparams.init_from}")
            self._copy_float_network(model)
        seed_weight_bits(model, q.weight_bits)
        jets = datamodule.data_train.X[: self.hparams.calibration_jets]
        rows, batch = self._keras_inputs(jets), self._calibration_batch()
        ebops = seed_datalane_bits(model, rows, q.data_bits, batch)
        if q.trained_words:
            ebops = seed_word_bits(model, q, rows, batch)
        log.info(f"{ebops} EBOPs at {q.data_bits} data and {q.weight_bits} weight bits")
        log.info(f"beta x EBOPs {self._beta_at(0) * ebops:.4g} in the first epoch")

    def configure_optimizers(self) -> None:
        """Compile the port with the Keras optimiser and the learning-rate schedule of
        the config, stepped per batch over the whole run.

        :return: None, for which Lightning substitutes its mock optimiser.
        """
        total = int(self.trainer.estimated_stepping_batches)
        warm = int(self.hparams.warmup_fraction * total)
        schedule = self._scheduler(decay_steps=max(total - warm, 1), warmup_steps=warm)
        optimizer = self._optimizer(learning_rate=schedule)
        model = self.net.keras_model
        optimizer.exclude_from_weight_decay(var_list=quantiser_variables(model))
        self.net.compile_keras(optimizer)
        return None

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """One Keras step, logging the epoch means of the cross entropy, of Keras's
        (micro) accuracy and of the loss with the EBOPs and bit penalties."""
        x, y = batch
        rows = self._keras_inputs(x)
        logs = self.net.train_on_batch(rows, y.cpu().numpy().astype(rows.dtype))
        # The mock optimiser's step does nothing but advance global_step.
        self.optimizers().step()
        # Keras resets its metrics at every train_on_batch, so acc is the batch's own.
        self._epoch_hits += logs["acc"] * len(x)
        self._epoch_jets += len(x)
        means = {"train/loss": logs["ce"], "train/acc": logs["acc"]}
        means["train/loss_total"] = logs["loss"]
        self.log_dict(means, on_step=False, on_epoch=True, batch_size=len(x))

    def _beta_at(self, epoch: int) -> float:
        """The constant beta, or the value at *epoch* of HGQ2's PieceWiseSchedule over
        the (epoch, beta, interpolation) points of ``beta_schedule``."""
        points = self.hparams.beta_schedule
        if not points:
            return float(self.hparams.beta)
        return PieceWiseSchedule([tuple(point) for point in points])(epoch)

    def on_train_epoch_start(self) -> None:
        """Give every quantised layer this epoch's beta and start counting the epoch's
        correct predictions."""
        set_beta(self.net.keras_model, self._beta_at(self.current_epoch))
        self._epoch_hits, self._epoch_jets = 0.0, 0

    def on_train_epoch_end(self) -> None:
        """Log the EBOPs of the epoch's last step, its beta, the next learning rate and
        the epoch's Keras accuracy, the mean over its jets.

        The base's summary would read ``train_acc``, which this row never updates. The
        campaign scripts parse the INFO line, so its format is an interface.
        """
        epoch, ebops = self.current_epoch, total_ebops(self.net.keras_model)
        beta, lr = self._beta_at(epoch), self.net.learning_rate()
        acc = self._epoch_hits / max(self._epoch_jets, 1)
        self.log_dict({"train/ebops": float(ebops), "train/beta": beta, "train/lr": lr})
        log.info(
            f"[epoch {epoch:>4d}] EBOPs {ebops}, beta {beta:.3g}, lr {lr:.3g}, "
            f"train acc {acc:.4f}"
        )

    def on_fit_start(self) -> None:
        """Send the count of Keras variables to the loggers; the torch count is zero."""
        count = self.net.keras_model.count_params()
        for logger in self.loggers:
            logger.log_hyperparams({"algorithm/keras_params": count})

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Keep the Keras weights beside the torch state."""
        checkpoint["keras_weights"] = self.net.keras_model.get_weights()

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore the Keras weights, cast to this port's float type, which may differ
        from the training's."""
        variables = self.net.keras_model.weights
        weights = zip(checkpoint["keras_weights"], variables, strict=True)
        self.net.keras_model.set_weights([np.asarray(w, v.dtype) for w, v in weights])

    def on_test_start(self) -> None:
        """Trace the integer bits of every wrapping quantiser on the record's whole
        training split, the range the hardware is built for, and price the traced port.

        The callbacks' hooks run first and read the feature layout alone, so every test
        batch, the probe's included, meets the traced port.

        :raises ValueError: when JAX does not run on the CPU.
        """
        require_cpu_jax()
        start = time.perf_counter()
        jets, _ = self.trainer.datamodule.load_split("train")
        rows = self._keras_inputs(jets)
        ebops = calibrate(self.net.keras_model, rows, self._calibration_batch())
        self.ebops_total, seconds = float(ebops), time.perf_counter() - start
        log.info(f"Traced {len(jets)} training jets in {seconds:.0f} s: {ebops} EBOPs")
        self._test_inputs: list[np.ndarray] = []

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """The base's test step, keeping the quantised words the port read, in the
        probe's jet order."""
        out = super().test_step(batch, batch_idx)
        words = self.net.quantised_inputs(self._keras_inputs(batch[0]))
        # Exact for words of at most 24 significant bits, float32's; synthesise_vn.py's
        # check of the reloaded port on them would show any loss.
        self._test_inputs.append(words.astype(np.float32))
        return out

    def on_test_epoch_end(self) -> None:
        """Log the traced EBOPs and, when the probe writes outputs (under eval.py),
        write the traced port, its test inputs and its metadata beside them, for
        synthesis."""
        self.log("test/ebops", self.ebops_total)
        label = self._outputs_label()
        if label is None:
            return
        directory = Path(self.trainer.default_root_dir) / "outputs" / f"{label}_hgq"
        inputs = np.concatenate(self._test_inputs)
        meta = self._export_meta(len(inputs))
        export_port(directory, self.net.keras_model, inputs, meta)
        log.info(f"Exported the traced port and {len(inputs)} test jets to {directory}")

    def _outputs_label(self) -> str | None:
        """The label under which the invariance callback writes this pass's outputs,
        None when no callback writes any."""
        callbacks = self.trainer.callbacks
        probes = [cb for cb in callbacks if isinstance(cb, InvarianceMetricCallback)]
        return probes[0].subdir if probes else None

    def _export_meta(self, n_test: int) -> dict[str, Any]:
        """The run behind an export, as scripts/synthesis/synthesise_vn.py reads it; the
        port's name is the top module of its design."""
        keys = ("nconstituents", "beta", "beta_schedule", "units", "floatx")
        return {
            "row": self.hparams.name,
            "port": self.net.keras_model.name,
            **{key: self.hparams[key] for key in keys},
            "ebops": int(self.ebops_total),
            "quantisation": asdict(self.quantisation),
            "n_test": n_test,
            "ckpt_path": self.trainer.ckpt_path,
            "test_acc": float(self.test_acc.compute()),
            "versions": {name: version(name) for name in VERSIONED},
        }
