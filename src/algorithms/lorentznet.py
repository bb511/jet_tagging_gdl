"""LorentzNet, wired into the shared training loop."""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.lorentznet import LorentzNetGraph
from src.data.components.augmentation import get_norm_scale


class LorentzNet(JetTaggingModule):
    """Jet tagger built on Lorentz Group Equivariant Blocks (arXiv:2201.08187)."""

    # The network needs only the full four-momentum.
    required_features = ["px", "py", "pz", "e"]

    def build_net(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the graph network; the arguments are in components/lorentznet.py."""
        self.register_buffer("denorm_scales", torch.ones(len(self.required_features)))
        return LorentzNetGraph(
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_class=self.hparams.output_dim,
            c_weight=c_weight,
            dropout=dropout,
            nconstituents=self.hparams.nconstituents,
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
