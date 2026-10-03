"""Vector neurons (arXiv:2104.12229) on the lifts of (p_T, eta, phi), exact under the
Euclidean group of the (eta, phi) plane.

Each constituent is lifted to u = p (1, eta, phi, eta^2 + phi^2), with p its p_T in
units and the angles measured from the leading constituent. A translation of the plane
acts on the lift by a fixed matrix T, a rotation about the origin by an orthogonal
matrix on (eta, phi), and both preserve the form B(u, v) = u^T G v, where G[0, 3] =
G[3, 0] = 1 and G[1, 1] = G[2, 2] = -2; on two constituents B(u_i, u_j) = p_i p_j
dR_ij^2. Every layer mixes the vector channels with one weight matrix shared by the four
components, and the only other operations on a vector are B and a scalar gate per
channel, so the forms and the first components, weighted sums of p, are invariant.
Batches must be de-normalised before they reach this module.

Symmetry. The logits are exact under shifts of eta and phi, the reflections eta -> -eta
and phi -> -phi, rotations of the plane about any point and any permutation of the
constituents. A rotation about the beam shifts every phi, and a boost along it every eta
of a massless constituent, by the same amount, so the probe's so2 and lorentz groups are
exact and so3 and lorentz_iso are measured. The logits do not depend on the reference of
the angles; the leader is taken because it moves with the jet, so that the angles stay
small: the phi wrap stays away from the jet, and B is spared the cancellation between
large terms that absolute angles would cost in float32.

The tensor invariants are three-point correlators built in O(N) from rank-2 tensors, as
energy flow moments are (arXiv:1911.04491): the beta = 2 energy flow polynomials of the
two-edge path, the doubled edge and the triangle, with learned weights per rank in place
of the uniform sum.

The network is meant for an FPGA, so it holds only operations computed there without a
lookup table or a per-jet division: multiply-adds, ReLU and comparisons. The gates are
therefore hard sigmoids, no invariant is compressed by a logarithm, and the multiplicity
is divided by the constant N.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# The first block's scalars per rank, p_i and B(u_i, U), and per jet, the sum of p and
# B(U, U), with U the jet's summed lift.
RANK_SCALARS = 2
JET_SCALARS = 2


def _form_matrix() -> torch.Tensor:
    """Return the (4, 4) matrix G of the invariant form B(u, v) = u^T G v."""
    metric = torch.zeros(4, 4)
    metric[0, 3] = metric[3, 0] = 1.0
    metric[1, 1] = metric[2, 2] = -2.0
    return metric


def _form(u: torch.Tensor, v: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """B(u, v) over the last axis, of length 4, elementwise over the leading ones."""
    return (u * (v @ metric)).sum(dim=-1)


def _unit_lift(eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Lift (jets, N) angles to (jets, N, 4) rows (1, eta, phi, eta^2 + phi^2)."""
    return torch.stack([torch.ones_like(eta), eta, phi, eta**2 + phi**2], dim=-1)


def _sort_by_pt(x: torch.Tensor) -> torch.Tensor:
    """Order (jets, N, 3) rows [p_T, eta, phi] by decreasing p_T, padded rows last."""
    order = x[..., 0].argsort(dim=1, descending=True)
    return x.gather(1, order.unsqueeze(-1).expand_as(x))


def _relative_angles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eta and phi of sorted (jets, N, 3) rows measured from the leading row's,
    each (jets, N), phi wrapped into [-pi, pi)."""
    eta = x[..., 1] - x[:, :1, 1]
    phi = (x[..., 2] - x[:, :1, 2] + math.pi) % (2 * math.pi) - math.pi
    return eta, phi


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers, ReLU between them and dropout in front of
    each."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class SplitLinear(nn.Module):
    """Mix the vector channels with one weight matrix shared by the four components of
    the lift, which form a single split.

    No bias, since a constant vector would not transform.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        bound = c_in**-0.5
        self.weight = nn.Parameter(torch.empty(c_out, c_in).uniform_(-bound, bound))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Map (jets, c_in, 4) vectors to (jets, c_out, 4)."""
        return torch.einsum("oc,bcm->bom", self.weight, v)


class FormInvariants(nn.Module):
    """Invariant forms of pairs of learned combinations of the vector channels, and the
    first component, a weighted sum of p, of every channel."""

    def __init__(self, channels: int, invariants: int) -> None:
        super().__init__()
        self.left = SplitLinear(channels, invariants)
        self.right = SplitLinear(channels, invariants)
        self.register_buffer("metric", _form_matrix(), persistent=False)
        self.width = invariants + channels

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map (jets, channels, 4) vectors to (jets, invariants + channels) scalars."""
        # Elementwise, not a matmul: form k pairs left and right channel k alone.
        forms = _form(self.left(u), self.right(u), self.metric)
        return torch.cat([forms, u[..., 0]], dim=-1)


class SplitGate(nn.Module):
    """Scale each vector channel, its four components alike, by a hard sigmoid of the
    scalar state."""

    def __init__(self, scalars: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(scalars, channels)
        # Every gate starts at exactly 1/2. One started near zero would stay there: the
        # first block has no residual and the forms after it are quadratic in the
        # vectors, so their gradient vanishes with the vectors.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Gate the (jets, channels, 4) vectors *u* by the (jets, scalars) state *s*."""
        return u * F.hardsigmoid(self.linear(s)).unsqueeze(-1)


class EquivariantBlock(nn.Module):
    """One update of the vector and scalar states of a jet.

    The block mixes the vectors, reads invariants off them, updates the scalars from
    those invariants, the previous scalars and the context, and gates the mixed vectors
    with the new scalars. The batch normalisation in front of the scalar MLP folds into
    its first layer at inference.

    :param c_in: input vector channels, the constituent slots in the first block.
    :param s_in: input scalar channels.
    :param invariants: forms read off the mixed vectors.
    :param dropout: dropout rate in front of each linear layer of the scalar MLP.
    :param residual: add the input to both states. Every block but the first does; the
        input of the first is the constituents by rank and their scalars, not a state of
        the same space.
    """

    def __init__(
        self,
        c_in: int,
        s_in: int,
        vector_channels: int,
        scalar_channels: int,
        invariants: int,
        dropout: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.linear = SplitLinear(c_in, vector_channels)
        self.invariants = FormInvariants(vector_channels, invariants)
        width = s_in + self.invariants.width + 1
        self.norm = nn.BatchNorm1d(width)
        self.mlp = _mlp([width, scalar_channels, scalar_channels], dropout)
        self.gate = SplitGate(scalar_channels, vector_channels)
        self.residual = residual

    def forward(
        self, v: torch.Tensor, s: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated ``(v, s)``.

        :param v: (jets, c_in, 4) vector state.
        :param s: (jets, s_in) scalar state.
        :param c: (jets, 1) context, the multiplicity over the slot count.
        :return: (jets, vector_channels, 4) vectors and (jets, scalar_channels) scalars.
        """
        u = self.linear(v)
        s_new = self.mlp(self.norm(torch.cat([s, self.invariants(u), c], dim=-1)))
        if self.residual:
            s_new = s_new + s
        v_new = self.gate(s_new, u)
        return (v_new + v if self.residual else v_new), s_new


class Stream(nn.Module):
    """JEDI-linear's per-constituent network (arXiv:2508.15468) with one context layer,
    over the scalars of each rank.

    Both pools are sums over the real ranks, which need no per-jet division.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(RANK_SCALARS)
        self.embed = nn.Linear(RANK_SCALARS, channels)
        self.local = nn.Linear(channels, channels)
        self.context = nn.Linear(channels, channels)

    def forward(self, r: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """Map the (jets, N, RANK_SCALARS) scalars of the ranks to (jets, channels).

        :param keep: (jets, N, 1), one on a real rank and zero on a padded one.
        """
        normed = self.norm(r.flatten(0, 1)).unflatten(0, r.shape[:2])
        e = keep * F.relu(self.embed(normed))
        pooled = e.sum(dim=1, keepdim=True)
        return (keep * F.relu(self.local(e) + self.context(pooled))).sum(dim=1)


class TensorInvariants(nn.Module):
    """Three kinds of three-point correlators from the rank-2 tensors t_i = u_i n_i^T,
    n_i the lift at unit p_T, in O(N).

    For each of K learned weightings of the ranks, the vectors L = sum_i A_i u_i and
    R = sum_i B_i u_i and the tensors M, M', M'' = sum_i (C_i, D_i, E_i) t_i give

    - chain, L^T G M G R = sum_ijl A_i C_j B_l p_i p_j p_l dR_ij^2 dR_jl^2,
    - pair4, tr(G M G M') = sum_ij C_i D_j p_i p_j dR_ij^4,
    - triangle, tr(G M G M' G M'') = sum_ijl C_i D_j E_l p_i p_j p_l dR_ij^2 dR_jl^2
      dR_li^2.

    A translation maps t_i to T t_i T^T, and T^T G T = G, so all three are invariant.
    """

    def __init__(self, nconstituents: int, invariants: int) -> None:
        super().__init__()
        self.left = SplitLinear(nconstituents, invariants)
        self.right = SplitLinear(nconstituents, invariants)
        bound = nconstituents**-0.5
        # C, D and E stacked: (3, K, N).
        weight = torch.empty(3, invariants, nconstituents).uniform_(-bound, bound)
        self.weight = nn.Parameter(weight)
        self.register_buffer("metric", _form_matrix(), persistent=False)

    def forward(self, u: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
        """Map the (jets, N, 4) lifts *u* and unit lifts *unit* to (jets, 3K) scalars,
        chain, pair4 and triangle in that order."""
        g = self.metric
        t = u.unsqueeze(-1) * unit.unsqueeze(-2)
        # G M, G M' and G M'': (3, jets, K, 4, 4).
        gm = g @ torch.einsum("wkn,bnij->wbkij", self.weight, t)
        chain = torch.einsum("bki,bkij,bkj->bk", self.left(u), gm[0], self.right(u) @ g)
        pair4 = torch.einsum("bkij,bkji->bk", gm[0], gm[1])
        triangle = torch.einsum("bkij,bkjl,bkli->bk", gm[0], gm[1], gm[2])
        return torch.cat([chain, pair4, triangle], dim=-1)


def _first_width(nconstituents: int, stream_channels: int, tensors: int) -> int:
    """Width of the first block's input scalars: every rank's and the jet's, the stream
    and the tensor invariants."""
    ranks = nconstituents * RANK_SCALARS
    return ranks + JET_SCALARS + stream_channels + 3 * tensors


def _blocks(
    nconstituents: int, s_first: int, num_blocks: int, **block_kwargs
) -> nn.ModuleList:
    """Stack *num_blocks* blocks, the first reading the constituents by rank and their
    scalars."""
    first = (nconstituents, s_first, False)
    later = (block_kwargs["vector_channels"], block_kwargs["scalar_channels"], True)
    return nn.ModuleList(
        EquivariantBlock(c_in, s_in, residual=residual, **block_kwargs)
        for c_in, s_in, residual in [first] + [later] * (num_blocks - 1)
    )


class VectorNeuronsClassifier(nn.Module):
    """Vector neurons over a fixed-width batch of jets in (p_T, eta, phi).

    The p_T-ordered lifts are the input vector channels of the first block, and their
    scalars, the stream and the tensor invariants its input scalars; from there on the
    state is per jet. The context, the multiplicity over the slot count, enters every
    block and the head.

    :param nconstituents: constituent slots per jet, the input vector channels of the
        first block.
    :param head_layers: hidden widths of the readout MLP.
    :param vector_channels: lifted vector channels of every block.
    :param scalar_channels: scalar channels of every block.
    :param invariants: forms read in every block and in the readout.
    :param tensor_invariants: correlators of each of the three kinds the first block
        reads.
    :param stream_channels: width of the per-constituent stream.
    :param dropout: dropout rate in front of each linear layer of the scalar MLPs and of
        the head; it never touches a vector.
    :param units: GeV per unit of p_T, 20 as in the L-GATr-slim and LLoCa rows.
    :param n_class: classes, read off the scalars, the readout invariants and the
        context.
    """

    def __init__(
        self,
        nconstituents: int,
        head_layers: list[int],
        num_blocks: int = 1,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        tensor_invariants: int = 2,
        stream_channels: int = 16,
        dropout: float = 0.0,
        units: float = 20.0,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        self.nconstituents, self.units = nconstituents, units
        self.register_buffer("metric", _form_matrix(), persistent=False)
        self.stream = Stream(stream_channels)
        self.tensors = TensorInvariants(nconstituents, tensor_invariants)
        self.blocks = _blocks(
            nconstituents,
            _first_width(nconstituents, stream_channels, tensor_invariants),
            num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            dropout=dropout,
        )
        self.readout = FormInvariants(vector_channels, invariants)
        width = scalar_channels + self.readout.width + 1
        self.head_norm = nn.BatchNorm1d(width)
        self.head = _mlp([width, *head_layers, n_class], dropout)

    def _first_scalars(
        self, u: torch.Tensor, unit: torch.Tensor, p: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        """Return the (jets, s_first) input scalars of the first block.

        :param u: (jets, N, 4) lifts, sorted.
        :param unit: (jets, N, 4) the lifts at unit p_T.
        :param p: (jets, N) p_T in units.
        :param keep: (jets, N, 1), one on a real rank and zero on a padded one.
        """
        jet = u.sum(dim=1, keepdim=True)
        r = torch.stack([p, _form(u, jet, self.metric)], dim=-1)
        jet_forms = _form(jet, jet, self.metric)
        jet_scalars = torch.cat([p.sum(dim=1, keepdim=True), jet_forms], dim=-1)
        extras = [self.stream(r, keep), self.tensors(u, unit)]
        return torch.cat([r.flatten(1), jet_scalars, *extras], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param x: (jets, constituents, 3) rows [p_T GeV, eta, phi], with padded
            constituents as all-zero rows.
        :return: (jets, n_class) logits.
        """
        x = _sort_by_pt(x)
        p = x[..., 0] / self.units
        unit = _unit_lift(*_relative_angles(x))
        u = p.unsqueeze(-1) * unit
        # p_T alone marks a padded row: its lift is zero whatever its angles.
        keep = (p != 0.0).unsqueeze(-1).to(p.dtype)
        c = keep.sum(dim=1) / self.nconstituents
        v, s = u, self._first_scalars(u, unit, p, keep)
        for block in self.blocks:
            v, s = block(v, s, c)
        return self.head(self.head_norm(torch.cat([s, self.readout(v), c], dim=-1)))
