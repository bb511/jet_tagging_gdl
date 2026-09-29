"""Vector neurons, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.vector_neurons import VectorNeuronsClassifier
from src.data.components.augmentation import get_norm_scale


class VectorNeurons(JetTaggingModule):
    """Jet tagger built on vector neurons (arXiv:2104.12229) whose weights are shared
    within blocks of the four-momentum, which makes it exact under a subgroup of the
    Lorentz group."""

    # The network needs only the full four-momentum.
    required_features = ["px", "py", "pz", "e"]

    def build_net(
        self,
        head_layers: list[int],
        symmetry: str = "z",
        num_blocks: int = 3,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        dropout: float = 0.0,
        units: float = 20.0,
    ) -> nn.Module:
        """Build the network; the arguments are in components/vector_neurons.py."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return VectorNeuronsClassifier(
            nconstituents=self.hparams.nconstituents,
            head_layers=head_layers,
            symmetry=symmetry,
            num_blocks=num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            dropout=dropout,
            units=units,
            n_class=self.hparams.output_dim,
        )

    def _on_features_resolved(self) -> None:
        """Cache the factors returning [px, py, pz, e] to GeV, in that column order."""
        scales = [
            get_norm_scale(self.norm_scales, name) for name in self.required_features
        ]
        self.denorm_scales.copy_(torch.tensor(scales))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Undo the normalisation of a batch and hand it over as four-momenta.

        :param x: (jets, constituents, 4), the columns [px, py, pz, e] as stored.
        :return: (jets, constituents, 4), the columns [E, px, py, pz] in GeV.
        """
        return (x * self.denorm_scales)[..., [3, 0, 1, 2]]
