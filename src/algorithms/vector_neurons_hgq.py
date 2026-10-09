"""Quantisation-aware vector neurons: the HGQ2 port inside the shared training loop.

Lightning keeps the loop, the data, validation, testing and the invariance probe; Keras
on JAX owns the network, its optimiser and its training step. The port reads the rows
[p_T GeV, eta, phi] that the trigger provides, sorted by p_T and measured from the
leading constituent, which ``_prepare_inputs`` computes in torch.
"""

import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

from src.algorithms.components.vector_neurons import (
    VectorNeuronsClassifier,
    _relative_angles,
    _sort_by_pt,
)
from src.algorithms.components.vector_neurons_hgq import (
    KerasNetwork,
    Quantisation,
    build_port,
    calibrate,
    configure_jax,
    export_port,
    quantiser_variables,
    seed_datalane_bits,
    seed_weight_bits,
    set_beta,
    total_ebops,
)
from src.algorithms.utils.flops import count_flops_per_jet
from src.algorithms.utils.vector_neurons_port import (
    architecture,
    copy_float_weights,
    load_float_network,
)
from src.algorithms.vector_neurons import VectorNeurons
from src.callbacks.invariance import InvarianceMetricCallback
from src.utils.pylogger import RankedLogger

# isort: split

# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.utils.sugar import PieceWiseSchedule

log = RankedLogger(__name__)

# Jets per Keras call on a whole split, so that the torch intermediates of one chunk
# alone are held at a time.
CHUNK = 65536
VERSIONED = ("keras", "hgq2", "jax")


def _word_range(kif: Sequence[int]) -> tuple[float, float]:
    """The least and the greatest value of a fixed-point word of (keep_negative, integer
    bits, fraction bits)."""
    k, i, f = kif
    return -k * 2.0**i, 2.0**i - 2.0**-f


def _outputs_label(callbacks: list) -> str | None:
    """The label under which the invariance callback writes this pass's outputs, None
    when no callback writes any."""
    probes = [cb for cb in callbacks if isinstance(cb, InvarianceMetricCallback)]
    return probes[0].subdir if probes else None


def _require_cpu_jax() -> None:
    """Refuse to evaluate unless JAX runs on the CPU.

    :raises ValueError: on any other backend, where XLA's float64 2**f is off by an ulp
        for some f, so that the port would round some jets otherwise than alkaid.
    """
    backend = jax.default_backend()
    if backend != "cpu":
        raise ValueError(
            f"JAX runs on {backend!r}, where XLA's float64 2**f is inexact, so the "
            "exported port and its stored logits would not match alkaid bit for bit. "
            "Evaluate vn_hgq with algorithm.jax_platform=cpu."
        )


class VectorNeuronsHGQ(VectorNeurons):
    """The vector neurons trained quantisation-aware with HGQ2, for synthesis by alkaid.

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
    """

    def build_net(
        self,
        head_layers: list[int],
        num_blocks: int = 1,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        tensor_invariants: int = 2,
        stream_channels: int = 16,
        mlp_layers: int = 2,
        gate: bool = True,
        dropout: float = 0.0,
        units: float = 16.0,
        quantisation: dict[str, Any] | None = None,
        beta: float = 1e-7,
        beta_schedule: list[list] | None = None,
        init_from: str | None = None,
        calibration_jets: int = 100_000,
        floatx: str = "float64",
        jax_platform: str = "cpu",
        jax_devices: int | list[int] = 1,
        warmup_fraction: float = 0.0,
    ) -> KerasNetwork:
        """Build the port; the arguments after ``units`` are described in
        configs/algorithm/vn_hgq.yaml, and the training ones are read from the
        hyperparameters when they are needed."""
        configure_jax(jax_platform, jax_devices, floatx)
        # Keras draws the initial weights from a generator of its own, seeded from torch.
        keras.utils.set_random_seed(torch.initial_seed() % 2**32)
        self._register_denorm_scales()
        self.automatic_optimization = False
        self.quantisation = Quantisation(**(quantisation or {}))
        model, inputs_model = build_port(
            nconstituents=self.hparams.nconstituents,
            head_layers=head_layers,
            num_blocks=num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            tensor_invariants=tensor_invariants,
            stream_channels=stream_channels,
            mlp_layers=mlp_layers,
            gate=gate,
            dropout=dropout,
            units=units,
            n_class=self.hparams.output_dim,
            quantisation=self.quantisation,
        )
        return KerasNetwork(model, inputs_model)

    def _record_costs(self, nconstituents: int, nbr_feats: int) -> None:
        """Price the FLOPs on the float network of this architecture, as the ``vn`` row
        is priced; the EBOPs are NaN until the trace at test time."""
        n_class = self.hparams.output_dim
        twin = VectorNeuronsClassifier(
            nconstituents, n_class=n_class, **architecture(self.hparams)
        )
        self.flops_total = count_flops_per_jet(twin, nconstituents, nbr_feats)
        self.ebops_total = self.ebops_unpriced = math.nan

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3) [p_T GeV, eta, phi].

        They are sorted by decreasing p_T, the angles measured from the leading
        constituent and zero on a padded row, and each column is clamped to the range of
        its trigger word: the words wrap, since alkaid traces no other input quantiser.
        """
        x = _sort_by_pt(super()._prepare_inputs(x))
        pt, q = x[..., :1], self.quantisation
        angles = torch.stack(_relative_angles(x), dim=-1) * (pt > 0)
        pt = pt.clamp(*_word_range(q.pt_kif))
        return torch.cat([pt, angles.clamp(*_word_range(q.angle_kif))], dim=-1)

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
                "The Keras step of vn_hgq has no symmetry-consistency loss; train it "
                "under data.method=augmentation."
            )
        self._initialise(datamodule)

    def _initialise(self, datamodule) -> None:
        """Copy the float network ``init_from`` names, if any, then seed the bits of the
        weights and, on the first ``calibration_jets`` training jets, of the
        datalanes."""
        model, q = self.net.keras_model, self.quantisation
        if self.hparams.init_from:
            net = load_float_network(self.hparams.init_from, like=self.hparams)
            log.info(f"Copying the float network of {self.hparams.init_from}")
            copy_float_weights(model, net, self.hparams.units, fold_norm=q.fold_norm)
        seed_weight_bits(model, q.weight_bits)
        jets = datamodule.data_train.X[: self.hparams.calibration_jets]
        ebops = seed_datalane_bits(model, self._keras_inputs(jets), q.data_bits)
        log.info(f"{ebops} EBOPs at {q.data_bits} data and {q.weight_bits} weight bits")

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
        _require_cpu_jax()
        start = time.perf_counter()
        jets, _ = self.trainer.datamodule.load_split("train")
        ebops = calibrate(self.net.keras_model, self._keras_inputs(jets))
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
        label = _outputs_label(self.trainer.callbacks)
        if label is None:
            return
        directory = Path(self.trainer.default_root_dir) / "outputs" / f"{label}_hgq"
        inputs = np.concatenate(self._test_inputs)
        meta = self._export_meta(len(inputs))
        export_port(directory, self.net.keras_model, inputs, meta)
        log.info(f"Exported the traced port and {len(inputs)} test jets to {directory}")

    def _export_meta(self, n_test: int) -> dict[str, Any]:
        """The run behind an export, as scripts/synthesis/synthesise_vn.py reads it."""
        keys = ("nconstituents", "beta", "beta_schedule", "units", "floatx")
        return {
            "row": self.hparams.name,
            **{key: self.hparams[key] for key in keys},
            "ebops": int(self.ebops_total),
            "quantisation": asdict(self.quantisation),
            "n_test": n_test,
            "ckpt_path": self.trainer.ckpt_path,
            "test_acc": float(self.test_acc.compute()),
            "versions": {name: version(name) for name in VERSIONED},
        }
