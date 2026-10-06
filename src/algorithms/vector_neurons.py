"""Vector neurons, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.vector_neurons import VectorNeuronsClassifier
from src.data.components.augmentation import get_norm_scale


class VectorNeurons(JetTaggingModule):
    """Jet tagger built on vector neurons (arXiv:2104.12229).

    The inputs are lifted from (p_T, eta, phi) to own 4-vectors, exact under the
    Euclidean group of the (eta, phi) plane.
    """

    # Absolute angles: the network measures them from its leading constituent itself.
    required_features = ["pt", "eta", "phi"]

    def build_net(
        self,
        head_layers: list[int],
        num_blocks: int = 1,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        tensor_invariants: int = 2,
        stream_channels: int = 16,
        dropout: float = 0.0,
        units: float = 20.0,
    ) -> nn.Module:
        """Build the network; the arguments are in components/vector_neurons.py."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return VectorNeuronsClassifier(
            nconstituents=self.hparams.nconstituents,
            head_layers=head_layers,
            num_blocks=num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            tensor_invariants=tensor_invariants,
            stream_channels=stream_channels,
            dropout=dropout,
            units=units,
            n_class=self.hparams.output_dim,
        )

    def _on_features_resolved(self) -> None:
        """Cache the returning [pt, eta, phi] to GeV and radians, in that order."""
        scales = [
            get_norm_scale(self.norm_scales, name) for name in self.required_features
        ]
        self.denorm_scales.copy_(torch.tensor(scales))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch.

        :param x: (jets, constituents, 3), the columns [pt, eta, phi] as stored.
        :return: the same columns in GeV and radians.
        """
        return x * self.denorm_scales
