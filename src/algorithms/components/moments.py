import math

import torch
from torch import nn

from src.algorithms.components.geometry import _form_matrix, _sort_by_pt, _unit_lift

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
    r"""Batch-normalised angular moments of a jet in [p_T, etarel, phirel].

    With the rows sorted by decreasing :math:`p_T`, :math:`G` the form of the
    vector-neuron row and :math:`w` a rank weighting,

    .. math::

        p_i = \frac{p_{T,i}}{20\,\mathrm{GeV}}, \quad
        n_i = (1, \eta_i, \phi_i, \eta_i^2 + \phi_i^2), \quad
        u_i = p_i n_i, \quad
        n_i^\top G n_j = \Delta R_{ij}^2, \quad
        U = \sum_i w_i u_i, \quad
        M = \sum_i w_i u_i n_i^\top,

    the five moments are

    .. math::

        \mathrm{sum\_p} &= \sum_i w_i p_i, \\
        \mathrm{pair} &= U^\top G U = \sum_{ij} w_i w_j p_i p_j \Delta R_{ij}^2, \\
        \mathrm{chain} &= U^\top G M G U
            = \sum_{ijl} w_i w_j w_l p_i p_j p_l \Delta R_{ij}^2 \Delta R_{jl}^2, \\
        \mathrm{pair4} &= \operatorname{tr}(GMGM)
            = \sum_{ij} w_i w_j p_i p_j \Delta R_{ij}^4, \\
        \mathrm{triangle} &= \operatorname{tr}(GMGMGM)
            = \sum_{ijl} w_i w_j w_l p_i p_j p_l
              \Delta R_{ij}^2 \Delta R_{jl}^2 \Delta R_{li}^2,

    each invariant under shifts and rotations of the :math:`(\eta, \phi)` plane and
    permutations of the constituents. The output is (jets, 5K), weighting by weighting.

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
        # The moments are cubic in p: fp16 under 16-mixed overflows.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(torch.promote_types(x.dtype, torch.float32)) * self.denorm_scales
            x = _sort_by_pt(x)
            p = x[..., 0] / PT_UNIT_GEV
            unit = _unit_lift(x[..., 1], x[..., 2])
            w = _rank_weights(p > 0, self.cuts).to(p.dtype)
            moments = _weighted_moments(p.unsqueeze(-1) * unit, unit, w, self.metric)
            return (moments * self.support).flatten(1)
