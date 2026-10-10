"""Quantisation-aware JEDI-linear, with or without the jet moments.

The port reads the record's rows [p_T GeV, etarel, phirel] sorted by p_T, the angles
measured outside the block from the axis of the kept constituents, as the published
network reads them; ``_prepare_inputs`` de-normalises, sorts and clamps them in torch.
"""

from typing import Any

import torch

from src.algorithms.components.deepsets import MOMENT_FEATURES, JEDILinearNet
from src.algorithms.components.hgq_port import KerasNetwork
from src.algorithms.components.jedi_linear_hgq import build_port
from src.algorithms.components.vector_neurons import _sort_by_pt
from src.algorithms.keras_base import CALIBRATION_BATCH, KerasQuantisedModule
from src.algorithms.utils.jedi_linear_port import (
    architecture,
    copy_float_weights,
    float_network,
    load_float_network,
)

# After the component, which selects Keras's JAX backend before the first import.
import keras


def _check_folded(quantisation: dict[str, Any] | None) -> None:
    """Refuse quantisers that do not fold the batch normalisation.

    :raises ValueError: unless ``fold_norm`` is true, which export.json then records.
    """
    if not (quantisation or {}).get("fold_norm", False):
        raise ValueError(
            "The JEDI-linear port has no batch normalisation: the copy folds the "
            "moments' into the head's first layer. Set quantisation.fold_norm true."
        )


class JEDILinearHGQ(KerasQuantisedModule):
    """JEDI-linear trained quantisation-aware with HGQ2, for synthesis by alkaid.

    Its head reads the jet moments under ``moments`` uniform or rankcut; under none it
    is the control without them.
    """

    required_features = list(MOMENT_FEATURES)

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        aggregation: str = "sum",
        moments: str = "uniform",
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
        configs/algorithm/jedi_linear_hgq.yaml, and the training ones are read from the
        hyperparameters when they are needed."""
        _check_folded(quantisation)
        return self._keras_net(
            build_port,
            quantisation,
            floatx,
            jax_platform,
            jax_devices,
            hidden=hidden,
            head_layers=head_layers,
            aggregation=aggregation,
            moments=moments,
            units=units,
        )

    def _float_twin(self) -> JEDILinearNet:
        """The float JEDI-linear of this architecture, as its float row is priced."""
        return float_network(architecture(self.hparams), self.hparams.output_dim)

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the float JEDI-linear of ``init_from``, read on the record's scales."""
        net = load_float_network(self.hparams.init_from, like=self.hparams)
        copy_float_weights(model, net, self.hparams.units, self.denorm_scales)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3) [p_T GeV, etarel, phirel],
        sorted by decreasing p_T, each column clamped to the range of its word."""
        x = _sort_by_pt(super()._prepare_inputs(x))
        return self._clamp_words(x[..., :1], x[..., 1:])

    def _calibration_batch(self) -> int:
        """Fewer jets per trace batch at larger N."""
        # The trace holds (jets, N, hidden) float64 intermediates.
        return max(1024, CALIBRATION_BATCH * 16 // self.hparams.nconstituents)
