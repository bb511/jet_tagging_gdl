"""DeepSets jet taggers: DeepSets itself and JEDI-linear, which adds one context layer.

Both networks are invariant under a permutation of the constituents by construction.
Their moment rows also read the jet moments of [pt, etarel, phirel], to which they are
pinned.
"""

from torch import nn

from src.algorithms import JetTaggingModule, denorm_factors
from src.algorithms.components.deepsets import (
    MOMENT_FEATURES,
    DeepSetsNet,
    JEDILinearNet,
)


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
    """JEDI-linear in float: DeepSets with one permutation-equivariant context layer,

    For more details, see (arXiv:2508.15468).
    """

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


def _moment_set(moments: str) -> str:
    """Return *moments*, refusing ``none``, under which a moment row is its parent."""
    if moments == "none":
        raise ValueError("A moment row needs a moment set, got 'none'.")
    return moments


class _MomentsRow(JetTaggingModule):
    """A row whose network reads `JetMoments`, pinned to the columns they read."""

    required_features = list(MOMENT_FEATURES)

    def _on_features_resolved(self) -> None:
        """Cache the factors returning [pt, etarel, phirel] to GeV and radians."""
        factors = denorm_factors(self.norm_scales, MOMENT_FEATURES)
        self.net.moments.denorm_scales.copy_(factors)


class DeepSetsMoments(_MomentsRow):
    """DeepSets whose rho also reads the jet moments."""

    def build_net(
        self,
        rho_layers: list[int],
        moments: str,
        phi_layers: list[int] | None = None,
        aggregation: str = "sum",
        dropout: float = 0.0,
        moments_only: bool = False,
    ) -> nn.Module:
        """Build the network; the arguments are documented in components/deepsets.py."""
        return DeepSetsNet(
            in_dim=self.hparams.nbr_feats,
            phi_layers=phi_layers,
            rho_layers=rho_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            dropout=dropout,
            moments=_moment_set(moments),
            moments_only=moments_only,
        )


class JEDILinearMoments(_MomentsRow):
    """JEDI-linear whose head also reads the jet moments."""

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        moments: str,
        aggregation: str = "sum",
    ) -> nn.Module:
        """Build the network; the arguments are documented in components/deepsets.py."""
        return JEDILinearNet(
            in_dim=self.hparams.nbr_feats,
            hidden=hidden,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            moments=_moment_set(moments),
        )
