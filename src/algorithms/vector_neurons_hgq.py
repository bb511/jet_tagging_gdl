"""Quantisation-aware vector neurons: the HGQ2 port inside the shared training loop.

The port reads the rows [p_T GeV, eta, phi] that the trigger provides, sorted by p_T and
measured from the leading constituent, which ``_prepare_inputs`` computes in torch. The
training, the trace and the export are those of every quantised row (keras_base.py).
"""

from typing import Any

import torch

from src.algorithms.components.hgq_port import KerasNetwork
from src.algorithms.components.vector_neurons import (
    VectorNeuronsClassifier,
    _relative_angles,
    _sort_by_pt,
)
from src.algorithms.components.vector_neurons_hgq import build_port
from src.algorithms.keras_base import KerasQuantisedModule
from src.algorithms.utils.vector_neurons_port import (
    architecture,
    copy_float_weights,
    load_float_network,
)
from src.algorithms.vector_neurons import VectorNeurons

# isort: split

# After the component, which selects Keras's JAX backend before the first import.
import keras


class VectorNeuronsHGQ(KerasQuantisedModule, VectorNeurons):
    """The vector neurons trained quantisation-aware with HGQ2, for synthesis by alkaid.

    Their ``required_features`` are those of ``VectorNeurons``.
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
        axis_offset: bool = False,
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
        if axis_offset:
            # As in VectorNeurons.build_net: meaningful under full_axis=True alone.
            self.required_features = [*self.required_features, "etarel", "phirel"]
        return self._keras_net(
            build_port,
            quantisation,
            floatx,
            jax_platform,
            jax_devices,
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
            axis_offset=axis_offset,
        )

    def _float_twin(self) -> VectorNeuronsClassifier:
        """The float vn of this architecture, as the ``vn`` row is priced."""
        n, n_class = self.hparams.nconstituents, self.hparams.output_dim
        return VectorNeuronsClassifier(n, n_class=n_class, **architecture(self.hparams))

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the float vn of ``init_from``, its units folded into the port's."""
        net = load_float_network(self.hparams.init_from, like=self.hparams)
        fold_norm = self.quantisation.fold_norm
        copy_float_weights(model, net, self.hparams.units, fold_norm=fold_norm)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3) [p_T GeV, eta, phi], and under
        the axis offset (jets, N, 5), with etarel and phirel behind them.

        They are sorted by decreasing p_T, eta and phi measured from the leading
        constituent and zero on a padded row, and each column is clamped to the range of
        its trigger word.
        """
        x = _sort_by_pt(super()._prepare_inputs(x))
        pt = x[..., :1]
        relative = torch.stack(_relative_angles(x), dim=-1) * (pt > 0)
        # x[..., 3:] is empty without the axis offset.
        angles = torch.cat([relative, x[..., 3:]], dim=-1)
        return self._clamp_words(pt, angles)
