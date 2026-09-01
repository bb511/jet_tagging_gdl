"""DeepSets jet tagger.

The network is invariant under a permutation of the constituents by construction, but under nothing
else, so the rotations and boosts have to be learnt from the data. That is what makes it the
reference architecture for comparing augmentation with the SCL.
"""

from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.deepsets import DeepSetsNet


class DeepSets(JetTaggingModule):
    """Jet tagger built from a per-constituent network, a pooling and a head."""

    def build_net(
        self,
        phi_layers: list[int],
        rho_layers: list[int],
        activation: str = "relu",
        aggregation: str = "mean",
        dropout: float = 0.0,
        mask: bool = True,
    ) -> nn.Module:
        """Build the network; the arguments are documented on :class:`DeepSetsNet`."""
        return DeepSetsNet(
            in_dim=self.hparams.nbr_feats,
            phi_layers=phi_layers,
            rho_layers=rho_layers,
            out_dim=self.hparams.output_dim,
            activation=activation,
            aggregation=aggregation,
            dropout=dropout,
            mask=mask,
        )
