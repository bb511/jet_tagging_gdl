"""Permutation-invariant DeepSets classifiers for jets (arXiv:1703.06114).

``DeepSetsNet`` is the configurable DeepSets row. ``JEDILinearNet`` is JEDI-linear
(arXiv:2508.15468) in float: DeepSets with one permutation-equivariant context layer in
its per-constituent network.

Either network can also read ``JetMoments`` as jet-level inputs beside its pooled
vector: the shift-invariant angular moments that the vector-neuron row multiplies.
"""

import math

import torch
from torch import nn

from src.algorithms.components.vector_neurons import (
    _form_matrix,
    _sort_by_pt,
    _unit_lift,
)

AGGREGATIONS = ("sum", "max", "sum_over_n")
MOMENT_SETS = ("none", "uniform", "rankcut")
# The columns the moments read, in this order, as data.features names them.
MOMENT_FEATURES = ("pt", "etarel", "phirel")
# Top-k cuts by p_T of each moment set, None keeping every real constituent.
RANK_CUTS = {"uniform": (None,), "rankcut": (None, 1, 2, 4, 8)}
# sum_p, pair, chain, pair4 and triangle, and the distinct constituents each needs to be
# nonzero.
MOMENTS_PER_WEIGHTING = 5
MOMENT_SUPPORT = (1, 2, 2, 2, 3)
# GeV per unit of p, as in the vector-neuron row.
PT_UNIT_GEV = 20.0


def _masked_pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`. ``sum_over_n`` divides the masked sum by
        the slot count N, the same for every jet, so unlike the mean it needs no per-jet
        division on an FPGA.
    """
    if aggregation == "max":
        # The Linformer's tokens can be negative, so a padded slot must lose every
        # comparison rather than enter the maximum as zero.
        return h.masked_fill(~real, float("-inf")).amax(dim=1)
    total = (h * real).sum(dim=1)
    if aggregation == "sum":
        return total
    return total / h.shape[1]


def _dense(
    dims: list[int], final_activation: bool, dropout: float = 0.0
) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers, with dropout in front of each and a ReLU
    between them and, if asked, after the last."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if final_activation or i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _check_choices(aggregation: str) -> None:
    """Refuse a pooling this module does not provide, by name."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
        )


def _check_moments(moments: str) -> None:
    """Refuse a moment set this module does not provide, by name."""
    if moments not in MOMENT_SETS:
        raise ValueError(
            f"moments must be one of {list(MOMENT_SETS)}, got {moments!r}."
        )


def _check_deepsets_options(
    aggregation: str, moments: str, moments_only: bool, phi_layers: list[int] | None
) -> None:
    """Refuse an unknown pooling or moment set, a moments-only network without moments,
    and phi widths given to a network without phi or missing from one with it."""
    _check_choices(aggregation)
    _check_moments(moments)
    if moments_only and moments == "none":
        raise ValueError("moments_only needs a moment set, but moments is 'none'.")
    if (phi_layers is None) != moments_only:
        raise ValueError(
            "phi_layers must be given exactly when moments_only is false, got "
            f"phi_layers={phi_layers!r} with moments_only={moments_only}."
        )


def _rank_weights(real: torch.Tensor, cuts: tuple[int | None, ...]) -> torch.Tensor:
    """Return the (jets, K, N) 0/1 weights of K cuts from the (jets, N) mask of rows
    sorted by decreasing p_T: the real rows among the first *cut*, or all for None."""
    rank = torch.arange(real.shape[1], device=real.device)
    keep = [real if cut is None else real & (rank < cut) for cut in cuts]
    return torch.stack(keep, dim=1)


def _support_mask(cuts: tuple[int | None, ...]) -> torch.Tensor:
    """Return the (K, 5) mask of the moments a cut leaves nonzero.

    Under the top-1 cut every moment but sum_p, and under the top-2 cut the triangle,
    vanish identically, five of the 25 rankcut columns. In float32 they come out as
    roundoff of terms of order p^3, which the batch normalisation would scale up to unit
    variance; masked to exact zeros, they leave it as zeros.
    """
    kept = torch.tensor([math.inf if cut is None else cut for cut in cuts])
    return (kept[:, None] >= torch.tensor(MOMENT_SUPPORT)).float()


def _weighted_moments(
    u: torch.Tensor, unit: torch.Tensor, w: torch.Tensor, metric: torch.Tensor
) -> torch.Tensor:
    """Return the (jets, K, 5) moments sum_p, pair, chain, pair4 and triangle of the
    (jets, N, 4) lifts *u* and unit lifts *unit* under the (jets, K, N) weights *w*, in
    O(N) through U and M."""
    big_u = torch.einsum("bkn,bni->bki", w, u)
    # G M, (jets, K, 4, 4); G is symmetric, so U G is the vector G U.
    gm = metric @ torch.einsum("bkn,bni,bnj->bkij", w, u, unit)
    gu = big_u @ metric
    pair = (big_u * gu).sum(dim=-1)
    chain = torch.einsum("bki,bkij,bkj->bk", big_u, gm, gu)
    pair4 = torch.einsum("bkij,bkji->bk", gm, gm)
    triangle = torch.einsum("bkij,bkjl,bkli->bk", gm, gm, gm)
    return torch.stack([big_u[..., 0], pair, chain, pair4, triangle], dim=-1)


class JetMoments(nn.Module):
    """Batch-normalised angular moments of a jet in [p_T, etarel, phirel].

    With the rows sorted by decreasing p_T, p_i = p_T,i / 20 GeV, the unit lift
    n_i = (1, eta_i, phi_i, eta_i^2 + phi_i^2), u_i = p_i n_i and G the form of the
    vector-neuron row, n_i^T G n_j = dR_ij^2. For each rank weighting w, with
    U = sum_i w_i u_i and M = sum_i w_i u_i n_i^T, the five moments are

    - sum_p = sum_i w_i p_i,
    - pair = U^T G U = sum_ij w_i w_j p_i p_j dR_ij^2,
    - chain = U^T G M G U = sum_ijl w_i w_j w_l p_i p_j p_l dR_ij^2 dR_jl^2,
    - pair4 = tr(G M G M) = sum_ij w_i w_j p_i p_j dR_ij^4,
    - triangle = tr(G M G M G M) = sum_ijl w_i w_j w_l p_i p_j p_l dR_ij^2 dR_jl^2
      dR_li^2,

    each invariant under shifts and rotations of the (eta, phi) plane and permutations
    of the constituents. The output is (jets, 5K), weighting by weighting.

    The input arrives normalised; ``denorm_scales``, filled by the algorithm once the
    datamodule is known, returns it to GeV and radians.

    :param weighting: a key of `RANK_CUTS`.
    """

    def __init__(self, weighting: str) -> None:
        super().__init__()
        self.cuts = RANK_CUTS[weighting]
        self.width = MOMENTS_PER_WEIGHTING * len(self.cuts)
        self.register_buffer("denorm_scales", torch.ones(len(MOMENT_FEATURES)))
        self.register_buffer("metric", _form_matrix(), persistent=False)
        self.register_buffer("support", _support_mask(self.cuts), persistent=False)
        self.norm = nn.BatchNorm1d(self.width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalised (jets, N, 3) rows to (jets, width) normalised moments."""
        return self.norm(self.raw(x))

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (jets, width) moments of normalised (jets, N, 3) rows, in at least
        float32."""
        # The moments are cubic in p, which reaches about 50: fp16 under 16-mixed would
        # overflow.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.promote_types(x.dtype, torch.float32)) * self.denorm_scales
            x = _sort_by_pt(x)
            p = x[..., 0] / PT_UNIT_GEV
            unit = _unit_lift(x[..., 1], x[..., 2])
            w = _rank_weights(p > 0, self.cuts).to(p.dtype)
            moments = _weighted_moments(p.unsqueeze(-1) * unit, unit, w, self.metric)
            return (moments * self.support).flatten(1)


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


class JEDILinearNet(nn.Module):
    """DeepSets whose per-constituent network carries one permutation-equivariant layer
    (Zaheer et al., arXiv:1703.06114): JEDI-linear (arXiv:2508.15468) in float.

    With s the ReLU, the context layer maps the embedding h_i of each constituent i to
    s(W1 h_i + b1) + s(W2 pool_j h_j + b2); a shared layer follows, and the head reads
    the pool of the result over the jet. Both pools use one aggregation over the real
    constituents, chosen among those the DeepSets row searches, so that the two rows
    differ in the context layer alone; padding changes nothing, and only the sum carries
    the multiplicity to the head. The published point is the sum. The released code
    scales each sum by a constant, which the next dense layer absorbs, and sums the
    padded rows too. Its BatchNorm is left out: it folds into the dense layers at
    inference and served the quantiser in training.

    :param hidden: width of the projection, the two context branches and the shared
        layer.
    :param head_layers: hidden widths of the MLP that reads the pooled vector.
    :param aggregation: one of `AGGREGATIONS`, for the context and the readout alike.
    :param moments: one of `MOMENT_SETS`; unless ``none``, the head also reads the
        `JetMoments` of that set, which need the inputs [p_T, etarel, phirel].
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        head_layers: list[int],
        out_dim: int,
        aggregation: str = "sum",
        moments: str = "none",
    ) -> None:
        super().__init__()
        _check_choices(aggregation)
        _check_moments(moments)
        self.aggregation = aggregation
        # A plain attribute under "none", as in DeepSetsNet.
        self.moments = None if moments == "none" else JetMoments(moments)
        self.embed = _dense([in_dim, hidden], True)
        self.local = _dense([hidden, hidden], True)
        self.context = _dense([hidden, hidden], True)
        self.shared = _dense([hidden, hidden], True)
        width = hidden if self.moments is None else hidden + self.moments.width
        self.head = _dense([width, *head_layers, out_dim], False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pooled(x)
        if self.moments is not None:
            pooled = torch.cat([pooled, self.moments(x)], dim=-1)
        return self.head(pooled)

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Return the jet vector the head reads, (jets, hidden), from a batch of shape
        (jets, constituents, features)."""
        h = self.embed(x)
        # A padded slot is an all-zero row but embeds to the activation of the first
        # bias, so the mask, not the input, keeps it out of both pools.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        context = self.context(_masked_pool(h, real, self.aggregation))
        h = self.local(h) + context.unsqueeze(1)
        return _masked_pool(self.shared(h), real, self.aggregation)
