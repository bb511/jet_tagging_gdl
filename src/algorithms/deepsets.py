"""Permutation-invariant DeepSets classifiers for jets (arXiv:1703.06114).

``DeepSetsNet`` is the configurable DeepSets row. ``JEDILinearNet`` (jedi_linear.py) is
JEDI-linear (arXiv:2508.15468) in float: DeepSets with one permutation-equivariant
context layer in its per-constituent network.

Either network can also read ``JetMoments`` as jet-level inputs beside its pooled
vector: the shift-invariant angular moments that the vector-neuron row multiplies.
"""

import torch
from torch import nn

from src.algorithms import JetTaggingModule, denorm_factors
from src.algorithms.components.mlp import _dense
from src.algorithms.components.moments import MOMENT_FEATURES, MOMENT_SETS, JetMoments
from src.algorithms.components.pooling import AGGREGATIONS, _check_choice, _masked_pool


def _check_deepsets_options(
    aggregation: str, moments: str, moments_only: bool, phi_layers: list[int] | None
) -> None:
    """Refuse an unknown pooling or moment set, a moments-only network without moments,
    and phi widths given to a network without phi or missing from one with it."""
    _check_choice("aggregation", aggregation, AGGREGATIONS)
    _check_choice("moments", moments, MOMENT_SETS)
    if moments_only and moments == "none":
        raise ValueError("moments_only needs a moment set, but moments is 'none'.")
    if (phi_layers is None) != moments_only:
        raise ValueError(
            "phi_layers must be given exactly when moments_only is false, got "
            f"phi_layers={phi_layers!r} with moments_only={moments_only}."
        )


class DeepSetsNet(nn.Module):
    """Apply phi to every constituent, pool over the jet, then read the pool with rho.

    :param phi_layers: widths of the per-constituent network, the last entry being the
        width of the pooled constituent embedding; None exactly under ``moments_only``.
    :param rho_layers: hidden widths of the network applied to the pooled embedding.
    :param aggregation: pooling over the real constituents, one of `AGGREGATIONS`.
    :param moments: one of `MOMENT_SETS`; unless ``none``, rho also reads the
        `JetMoments` of that set, which need the inputs [p_T, etarel, phirel].
    :param moments_only: build no phi: rho reads the moments and the multiplicity over
        the slot count alone.
    """

    def __init__(
        self,
        in_dim: int,
        phi_layers: list[int] | None,
        rho_layers: list[int],
        out_dim: int,
        aggregation: str = "sum",
        dropout: float = 0.0,
        moments: str = "none",
        moments_only: bool = False,
    ) -> None:
        super().__init__()
        _check_deepsets_options(aggregation, moments, moments_only, phi_layers)
        self.aggregation = aggregation
        self.moments_only = moments_only
        # A plain attribute under "none", so that the network keeps the state-dict keys
        # and the random draws it had before the moments existed.
        self.moments = None if moments == "none" else JetMoments(moments)
        if not moments_only:
            self.phi = _dense([in_dim, *phi_layers], final_activation=True)
        width = self._rho_width(phi_layers)
        self.rho = _dense([width, *rho_layers, out_dim], False, dropout)

    def _rho_width(self, phi_layers: list[int] | None) -> int:
        """The width rho reads: the pool, the moments beside it, or the moments and the
        multiplicity alone."""
        if self.moments is None:
            return phi_layers[-1]
        if self.moments_only:
            return self.moments.width + 1
        return phi_layers[-1] + self.moments.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents,
        features)."""
        # A padded constituent is an all-zero row.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        if self.moments_only:
            multiplicity = real.sum(dim=1).to(x.dtype) / x.shape[1]
            return self.rho(torch.cat([self.moments(x), multiplicity], dim=-1))
        pooled = _masked_pool(self.phi(x), real, self.aggregation)
        if self.moments is not None:
            pooled = torch.cat([pooled, self.moments(x)], dim=-1)
        return self.rho(pooled)


class DeepSets(JetTaggingModule):
    """Jet tagger built from a per-constituent network, a pooling and a head."""

    def build_net(
        self,
        phi_layers: list[int],
        rho_layers: list[int],
        aggregation: str = "sum",
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the network; the arguments are documented in :class:`DeepSetsNet`."""
        return DeepSetsNet(
            in_dim=self.hparams.nbr_feats,
            phi_layers=phi_layers,
            rho_layers=rho_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            dropout=dropout,
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
        """Build the network; the arguments are documented in :class:`DeepSetsNet`."""
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
