"""The base of the rows trained quantisation-aware with HGQ2: a Keras port inside the
shared training loop, as `src.algorithms` is the base of the torch rows.

Lightning keeps the loop, the data, validation, testing and the invariance probe; Keras
on JAX owns the network, its optimiser and its training step. A row builds its port in
``build_net`` through ``_keras_net`` and supplies its float twin, the copy of a trained
float network and the preparation of its rows; everything else is shared.
"""

import math
import time
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.algorithms import DenormalisedModule
from src.algorithms.components.hgq_port import (
    KerasNetwork,
    Quantisation,
    calibrate,
    configure_jax,
    export_port,
    quantiser_variables,
    require_cpu_jax,
    seed_datalane_bits,
    seed_weight_bits,
    set_beta,
    total_ebops,
    word_range,
)
from src.algorithms.utils.flops import count_flops_per_jet
from src.callbacks.invariance import InvarianceMetricCallback
from src.utils.pylogger import RankedLogger

# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.utils.sugar import PieceWiseSchedule

log = RankedLogger(__name__)

# Jets per Keras call on a whole split, so that the torch intermediates of one chunk
# alone are held at a time.
CHUNK = 65536
# Jets per batch of HGQ2's trace, unless a row bounds it.
CALIBRATION_BATCH = 16384
VERSIONED = ("keras", "hgq2", "jax")


class KerasQuantisedModule(DenormalisedModule):
    """A row trained quantisation-aware with HGQ2, for synthesis by alkaid.

    ``net`` is a ``KerasNetwork`` without torch parameters, so the base class's forward
    pass, validation, test and probe run unchanged on it. Training is manual: each batch
    is one Keras ``train_on_batch`` and one step of the mock optimiser that Lightning
    puts in place of a torch one. That step advances ``global_step``, without which
    ``ModelCheckpoint`` never saves. The Keras weights travel inside the Lightning
    checkpoint and the Keras optimiser state does not, so a run cannot resume mid-way.

    At test time the integer bits of every wrapping quantiser are traced on the record's
    whole training split; the test, the probe and the export all read the traced port,
    and ``ebops_total`` holds its EBOPs, NaN until then. Testing runs with JAX on the
    CPU alone, where the float64 port computes the fixed-point values of the hardware
    exactly; on a GPU, XLA's float64 2**f is off by an ulp for some f, so the stored
    logits would not be bit-exact against alkaid.

    A row's ``build_net`` calls ``_keras_net``; the row also implements ``_float_twin``,
    ``_copy_float_network`` and ``_prepare_inputs``, which clamps through
    ``_clamp_words``. This class defines no ``__init__``, since
    ``JetTaggingModule.__init__`` collects the network's arguments from the caller's
    frame, and no ``required_features``, which each row sets.
    """

    def _keras_net(
        self,
        build_port: Callable[..., tuple[keras.Model, keras.Model]],
        quantisation: dict[str, Any] | None,
        floatx: str,
        jax_platform: str,
        jax_devices: int | list[int],
        **port_kwargs: Any,
    ) -> KerasNetwork:
        """Configure JAX, register the de-normalisation and build the port.

        :param build_port: the row's builder, called with N, then ``n_class``,
            ``quantisation`` and *port_kwargs* by keyword, returning the network and the
            model of its words.
        :param quantisation: the fields of ``Quantisation`` that the config sets.
        """
        configure_jax(jax_platform, jax_devices, floatx)
        # Keras draws the initial weights from a generator of its own, seeded from torch.
        keras.utils.set_random_seed(torch.initial_seed() % 2**32)
        self._register_denorm_scales()
        self.automatic_optimization = False
        self.quantisation = q = Quantisation(**(quantisation or {}))
        n, n_class = self.hparams.nconstituents, self.hparams.output_dim
        model, words = build_port(n, n_class=n_class, quantisation=q, **port_kwargs)
        return KerasNetwork(model, words)

    def _float_twin(self) -> nn.Module:
        """The float network of this port's architecture, on which the FLOPs are
        counted."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _float_twin()."
        )

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the trained float network that ``init_from`` names into the port
        *model*."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _copy_float_network()."
        )

    def _calibration_batch(self) -> int:
        """Jets per batch of HGQ2's trace."""
        return CALIBRATION_BATCH

    def _record_costs(self, nconstituents: int, nbr_feats: int) -> None:
        """Price the FLOPs on the float network of this architecture, as its float row
        is priced; the EBOPs are NaN until the trace at test time."""
        twin = self._float_twin()
        self.flops_total = count_flops_per_jet(twin, nconstituents, nbr_feats)
        self.ebops_total = self.ebops_unpriced = math.nan

    def _clamp_words(self, pt: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """Concatenate the (jets, N, 1) p_T and the (jets, N, A) angles, each clamped to
        the range of its word: the words wrap, since alkaid traces no other input
        quantiser."""
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
        """Give every quantised layer this epoch's beta."""
        set_beta(self.net.keras_model, self._beta_at(self.current_epoch))

    def on_train_epoch_end(self) -> None:
        """Log the EBOPs of the epoch's last step, its beta and the next learning rate.

        The base's summary would read ``train_acc``, which this row never updates.
        """
        epoch, ebops = self.current_epoch, total_ebops(self.net.keras_model)
        beta, lr = self._beta_at(epoch), self.net.learning_rate()
        self.log_dict({"train/ebops": float(ebops), "train/beta": beta, "train/lr": lr})
        log.info(f"[epoch {epoch:>3d}] EBOPs {ebops}, beta {beta:.3g}, lr {lr:.3g}")

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
        # Exact: a word holds 14 bits at most.
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
