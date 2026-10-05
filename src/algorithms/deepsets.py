"""DeepSets jet taggers: DeepSets itself and JEDI-linear, which adds one context layer.

Both networks are invariant under a permutation of the constituents by construction.
"""

from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.deepsets import DeepSetsNet, JEDILinearNet


class DeepSets(JetTaggingModule):
    """Jet tagger built from a per-constituent network, a pooling and a head."""

    def build_net(
        self,
        phi_layers: list[int],
        rho_layers: list[int],
        aggregation: str = "sum",
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the network; the arguments are documented in components/deepsets.py."""
        return DeepSetsNet(
            in_dim=self.hparams.nbr_feats,
            phi_layers=phi_layers,
            rho_layers=rho_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            dropout=dropout,
        )


class JEDILinear(JetTaggingModule):
    """JEDI-linear in float: DeepSets with one permutation-equivariant context layer, a
    grid row and the comparator from the FPGA literature (arXiv:2508.15468)."""

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        aggregation: str = "sum",
    ) -> nn.Module:
        """Build the network; the arguments are documented in components/deepsets.py."""
        return JEDILinearNet(
            in_dim=self.hparams.nbr_feats,
            hidden=hidden,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
        )
