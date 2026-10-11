"""Vector neurons (arXiv:2104.12229) on the lifts of (p_T, eta, phi).

Exact under the Euclidean group of the (eta, phi) plane.

Each constituent is lifted to u = p (1, eta, phi, eta^2 + phi^2).
The p is p_T in GeV and the angles measured from the leading constituent.
A translation of the plane acts on the lift by a fixed matrix T, a rotation about the
origin by an orthogonal matrix on (eta, phi).
Every layer mixes the vector channels with one weight matrix shared by the four
components, and the only other operations on a vector are B and a scalar gate per
channel. The forms and the first components, weighted sums of p, are invariant.

Batches must be de-normalised before they reach this module!!

Symmetry. The logits are exact under shifts of eta and phi, the reflections eta -> -eta
and phi -> -phi, rotations of the plane about any point and any permutation of the
constituents. A rotation about the beam shifts every phi, and a boost along it every eta
of a massless constituent.
Under the axis offset the leader's eta and phi from the full jet's axis are two more
scalars: a shift of the jet and its axis together leaves them alone, a rotation of the
plane does not.

The tensor invariants are three-point correlators built in O(N) from rank-2 tensors, as
energy flow moments are (arXiv:1911.04491): the beta = 2 energy flow polynomials of the
two-edge path, the doubled edge and the triangle, with learned weights per rank in place
of the uniform sum.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import DenormalisedModule
from src.algorithms.components.geometry import (
    _form,
    _form_matrix,
    _relative_angles,
    _sort_by_pt,
    _unit_lift,
)
from src.algorithms.components.mlp import _dense

RANK_SCALARS = 2


class ChannelLinear(nn.Module):
    """Mix the vector channels with one weight matrix shared by the four components.

    The four components pertain to the lifted inputs. No bias, since a constant vector
    would not transform.
    """

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        bound = c_in**-0.5
        self.weight = nn.Parameter(torch.empty(c_out, c_in).uniform_(-bound, bound))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Map (jets, c_in, 4) vectors to (jets, c_out, 4)."""
        return torch.einsum("oc,bcm->bom", self.weight, v)


class FormInvariants(nn.Module):
    """Invariant forms of pairs of learned combinations of the vector channels."""

    def __init__(self, channels: int, invariants: int) -> None:
        super().__init__()
        self.left = ChannelLinear(channels, invariants)
        self.right = ChannelLinear(channels, invariants)
        self.register_buffer("metric", _form_matrix(), persistent=False)
        self.width = invariants + channels

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map (jets, channels, 4) vectors to (jets, invariants + channels) scalars."""
        # Elementwise: form k pairs left and right channel k.
        forms = _form(self.left(u), self.right(u), self.metric)
        return torch.cat([forms, u[..., 0]], dim=-1)


class ChannelGate(nn.Module):
    """Scale each vector channel, its four components alike, by a hard sigmoid."""

    def __init__(self, scalars: int, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(scalars, channels)
        # Every gate starts at exactly 1/2. If one starts near zero then it stays there.
        # The first block has no residual and the forms after it are quadratic in the
        # vectors: their gradient vanishes with the vectors.
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
    :param mlp_layers: linear layers of the scalar MLP, ReLU between them and none after
        the last.
    :param gate: gate the mixed vectors. Without the gate they pass unscaled, so that on
        an FPGA the vector path does not wait for the scalar MLP.
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
        mlp_layers: int = 2,
        gate: bool = True,
    ) -> None:
        super().__init__()
        self.linear = ChannelLinear(c_in, vector_channels)
        self.invariants = FormInvariants(vector_channels, invariants)
        width = s_in + self.invariants.width + 1
        self.norm = nn.BatchNorm1d(width)
        self.mlp = _dense([width, *[scalar_channels] * mlp_layers], False, dropout)
        self.gate = ChannelGate(scalar_channels, vector_channels) if gate else None
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
        if self.gate is not None:
            u = self.gate(s_new, u)
        return (u + v if self.residual else u), s_new


class Stream(nn.Module):
    """JEDI-linear's per-constituent network (arXiv:2508.15468) with one context layer.

    This acts over the scalars. Both pools are sums over the real ranks, which need no
    per-jet division.
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
    """Three kinds of three-point correlators from the rank-2 tensors."""

    def __init__(self, nconstituents: int, invariants: int) -> None:
        super().__init__()
        self.left = ChannelLinear(nconstituents, invariants)
        self.right = ChannelLinear(nconstituents, invariants)
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


def _tensor_invariants(nconstituents: int, invariants: int) -> TensorInvariants | None:
    return TensorInvariants(nconstituents, invariants) if invariants > 0 else None


def _first_width(
    nconstituents: int, stream_channels: int, tensor_invariants: int, axis_offset: bool
) -> int:
    # Every rank's scalars, the jet's two, the stream, three kinds of tensor
    # invariant and the leader's offset, in the order of _first_scalars.
    return (
        nconstituents * RANK_SCALARS
        + 2
        + stream_channels
        + 3 * tensor_invariants
        + (2 if axis_offset else 0)
    )


def _blocks(
    nconstituents: int, s_first: int, num_blocks: int, **block_kwargs
) -> nn.ModuleList:
    """Stack *num_blocks* blocks, the first reading the constituents by rank."""
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
        reads; 0 builds no tensor invariants.
    :param stream_channels: width of the per-constituent stream; 0 builds no stream.
    :param mlp_layers: linear layers of the scalar MLP of every block.
    :param gate: gate the mixed vectors in every block.
    :param dropout: dropout rate in front of each linear layer of the scalar MLPs and of
        the head; it never touches a vector.
    :param units: GeV per unit of p_T, 20 as in the L-GATr-slim and LLoCa rows.
    :param axis_offset: read two more columns, eta and phi measured from the full jet's
        axis, and give the first block the leader's two as scalars.
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
        mlp_layers: int = 2,
        gate: bool = True,
        dropout: float = 0.0,
        units: float = 20.0,
        axis_offset: bool = False,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        self.nconstituents, self.units = nconstituents, units
        self.axis_offset = axis_offset
        self.register_buffer("metric", _form_matrix(), persistent=False)
        self.stream = Stream(stream_channels) if stream_channels > 0 else None
        self.tensors = _tensor_invariants(nconstituents, tensor_invariants)
        self.blocks = _blocks(
            nconstituents,
            _first_width(
                nconstituents, stream_channels, tensor_invariants, axis_offset
            ),
            num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            dropout=dropout,
            mlp_layers=mlp_layers,
            gate=gate,
        )
        self.readout = FormInvariants(vector_channels, invariants)
        width = scalar_channels + self.readout.width + 1
        self.head_norm = nn.BatchNorm1d(width)
        self.head = _dense([width, *head_layers, n_class], False, dropout)

    def _first_scalars(
        self,
        u: torch.Tensor,
        unit: torch.Tensor,
        p: torch.Tensor,
        keep: torch.Tensor,
        offset: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return the (jets, s_first) input scalars of the first block.

        :param u: (jets, N, 4) lifts, sorted.
        :param unit: (jets, N, 4) the lifts at unit p_T.
        :param p: (jets, N) p_T in units.
        :param keep: (jets, N, 1), one on a real rank and zero on a padded one.
        :param offset: (jets, 2) the leader's eta and phi from the full jet's axis, None
            without the axis offset.
        """
        jet = u.sum(dim=1, keepdim=True)
        r = torch.stack([p, _form(u, jet, self.metric)], dim=-1)
        jet_scalars = [p.sum(dim=1, keepdim=True), _form(jet, jet, self.metric)]
        stream = [] if self.stream is None else [self.stream(r, keep)]
        tensors = [] if self.tensors is None else [self.tensors(u, unit)]
        axis = [] if offset is None else [offset]
        return torch.cat([r.flatten(1), *jet_scalars, *stream, *tensors, *axis], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param x: (jets, constituents, 3) rows [p_T GeV, eta, phi], with padded
            constituents as all-zero rows. Under the axis offset the rows are 5 wide,
            with eta and phi from the full jet's axis behind them.
        :return: (jets, n_class) logits.
        """
        x = _sort_by_pt(x)
        p = x[..., 0] / self.units
        unit = _unit_lift(*_relative_angles(x))
        u = p.unsqueeze(-1) * unit
        # p_T alone marks a padded row: its lift is zero whatever its angles.
        keep = (p != 0.0).unsqueeze(-1).to(p.dtype)
        c = keep.sum(dim=1) / self.nconstituents
        offset = x[:, 0, 3:5] if self.axis_offset else None
        v, s = u, self._first_scalars(u, unit, p, keep, offset)
        for block in self.blocks:
            v, s = block(v, s, c)
        return self.head(self.head_norm(torch.cat([s, self.readout(v), c], dim=-1)))


class VectorNeurons(DenormalisedModule):
    """Jet tagger built on vector neurons (arXiv:2104.12229).

    The inputs are lifted from (p_T, eta, phi) to own 4-vectors, exact under the
    Euclidean group of the (eta, phi) plane.
    """

    # Absolute angles: the network measures them from its leading constituent itself.
    required_features = ["pt", "eta", "phi"]

    def build_net(
        self,
        head_layers: list[int],
        num_blocks: int = 1,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        tensor_invariants: int = 2,
        stream_channels: int = 16,
        mlp_layers: int = 2,
        gate: bool = True,
        dropout: float = 0.0,
        units: float = 20.0,
        axis_offset: bool = False,
    ) -> nn.Module:
        """Build the network; the arguments are in :class:`VectorNeuronsClassifier`."""
        if axis_offset:
            self.required_features = [*self.required_features, "etarel", "phirel"]

        self._register_denorm_scales()
        return VectorNeuronsClassifier(
            nconstituents=self.hparams.nconstituents,
            head_layers=head_layers,
            num_blocks=num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            tensor_invariants=tensor_invariants,
            stream_channels=stream_channels,
            mlp_layers=mlp_layers,
            gate=gate,
            dropout=dropout,
            units=units,
            axis_offset=axis_offset,
            n_class=self.hparams.output_dim,
        )
