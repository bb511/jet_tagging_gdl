"""LorentzNet, wired into the shared training loop."""

from torch import nn

from src.algorithms import FourMomentumModule
from src.algorithms.components.lorentznet import LorentzNetGraph


class LorentzNet(FourMomentumModule):
    """Jet tagger built on Lorentz Group Equivariant Blocks (arXiv:2201.08187)."""

    def build_net(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the graph network; the arguments are in components/lorentznet.py."""
        self._register_denorm_scales()
        return LorentzNetGraph(
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_class=self.hparams.output_dim,
            c_weight=c_weight,
            dropout=dropout,
            nconstituents=self.hparams.nconstituents,
        )
