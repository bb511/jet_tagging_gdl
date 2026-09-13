"""PELICAN, the permutation-equivariant Lorentz aggregator of arXiv:2307.16506.

A jet enters as the array of Minkowski products of every pair of its constituents, and
the network is a stack of the 15 linear maps that take one such array to another
equivariantly (Maron et al., arXiv:1812.09902), interleaved with channel mixes.

Shape conventions. ``pmu (B, N, 4)`` holds :math:`[E, p_x, p_y, p_z]` in GeV with padded
constituents as all-zero rows, and N counts the two beams once they are prepended;
``real (B, N)`` is ``pmu[..., 0] != 0``; ``pair_mask (B, N, N, 1)`` is ``real_i &
real_j``; ``nobj (B, 1)`` is the real count, beams included. The pair tensor ``x (B, N,
N, C)`` is zero on every padded pair, which is what lets every reduction sum over all
slots and divide by ``nobj``; each layer re-masks its output, since a bias would
otherwise fill the padding. No dtype and no device is hard-coded, so the module runs in
float64.

Symmetry. With the beams on, the exact invariance is SO(2) about the beam axis together
with :math:`p_z \\to -p_z`, which merely swaps two beams carrying the same label. It is
not the boosts along the beam: the beams are fixed null vectors, a boost of rapidity y
along z rescales :math:`E \\mp p_z` by :math:`e^{\\mp y}`, so the beam-constituent
products move and the logits with them. With the beams off, every product the network
reads is a Lorentz scalar and the logits are invariant under the full group.

Batches must be de-normalised before they reach this module.
"""

import torch
import torch.nn.functional as F
from torch import nn


def pair_dots(p: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """Minkowski products of every pair, ``(B, N, N, 1)``, metric (+, -, -, -).

    One ``bmm``, so the FLOP counter sees it and no ``(B, N, N, 4)`` intermediate is
    built.
    """
    return torch.matmul(p, (p * metric).transpose(1, 2)).unsqueeze(-1)


def pair_reductions(x: torch.Tensor, nobj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The compact sources of the basis: ``(diag, rows, cols, trace, total)``.

    The diagonal and the row and column means are ``(B, N, C)``, the trace and the total
    ``(B, 1, C)``. Padded pairs of *x* are zero, so a sum over every slot divided by
    *nobj* is the mean over the real ones.
    """
    n = nobj.unsqueeze(-1)
    diag = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
    rows, cols = x.sum(2) / n, x.sum(1) / n
    trace = diag.sum(1, keepdim=True) / n
    total = rows.sum(1, keepdim=True) / n
    return diag, rows, cols, trace, total


def scaled_mix(v: torch.Tensor, scale: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mix the channels of a compact source, each rescaled by its own power of N.

    The factor :math:`(N/\\bar N)^\\alpha` is one per input channel and the mix is a sum
    over input channels, so rescaling here is the same as rescaling the basis element
    the result is broadcast into.
    """
    return (v * scale) @ w


class InputEmbedding(nn.Module):
    """Spread one Minkowski product over *n_embed* channels of compressed magnitude.

    Channel c returns :math:`\\mathrm{sign}(y)((1+|y|)^a - 1)/a` of the scaled product y,
    with exponents a spanning [0.0025, 0.25]: the products run over several orders of
    magnitude in GeV squared and the powers pick out different decades. The signed form
    is the reference's ``slog`` mode, and it is mandatory here: the plain
    :math:`(1+y)^a` is NaN on a negative argument, which the random normals of the FLOP
    counting pass produce. On data the products of future-directed momenta are
    non-negative and the two agree.
    """

    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_embed))
        self.alphas = nn.Parameter(torch.linspace(0.05, 0.5, n_embed))

    def forward(self, dots: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, N, N, 1)`` products into ``(B, N, N, n_embed)``.

        f(0) = 0, so the padded pairs stay zero and no mask is needed.
        """
        a = 1e-6 + self.alphas**2
        y = dots * self.scale
        return torch.sign(y) * ((1.0 + y.abs()).pow(a) - 1.0) / a


class LinEq2to2(nn.Module):
    """The equivariant linear map ``(B, N, N, in_dim) -> (B, N, N, out_dim)``.

    The 15 basis elements are the reference's ``eops_2_to_2``, in its order: 0 x[i,j], 1
    x[j,i], 2 delta_ij d_i, 3 d_j, 4 d_i, 5 delta_ij c_i, 6 c_j, 7 c_i, 8 delta_ij r_i,
    9 r_j, 10 r_i, 11 t, 12 delta_ij t, 13 s, 14 delta_ij s, with d the diagonal, r and
    c the row and column means, t the trace and s the total. Elements 5 to 14 are the
    ones whose value depends on the multiplicity, and each carries a learnable exponent
    on :math:`N/\\bar N`, which is what lets a network trained at one multiplicity read
    another.

    Written on the compact sources rather than as the reference's ``(B, C, 15, N, N)``
    stack, which holds 15 ``in_dim``-channel arrays per layer where this holds two.

    :param nbar: the mean multiplicity the exponents are measured against.
    """

    def __init__(self, in_dim: int, out_dim: int, nbar: float = 49.0) -> None:
        super().__init__()
        self.nbar = nbar
        std = (2.0 / (15 * in_dim)) ** 0.5
        self.weight = nn.Parameter(torch.normal(0.0, std, (in_dim, 15, out_dim)))
        self.alphas = nn.Parameter(torch.rand(in_dim, 10))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.diag_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self, x: torch.Tensor, pair_mask: torch.Tensor, nobj: torch.Tensor
    ) -> torch.Tensor:
        """Return the re-masked, LeakyReLU image of the pair tensor *x*."""
        scale = (nobj / self.nbar).view(-1, 1, 1, 1) ** self.alphas
        diag, rows, cols, trace, total = pair_reductions(x, nobj)
        by_row, by_col = self._broadcast_terms(diag, rows, cols, scale)
        out = self._array_terms(x) + self._scalar_terms(trace, total, scale) + self.bias
        out += by_row.unsqueeze(2)
        out += by_col.unsqueeze(1)
        # In place on the diagonal of a fresh sum: legal for autograd, whose backward
        # through those adds and matmuls reads none of their outputs. Nothing may come
        # between the terms and this add: an activation first would leave the diagonal
        # with an incomplete basis.
        on_diagonal = self._diagonal_terms(diag, rows, cols, trace, total, scale)
        out.diagonal(dim1=1, dim2=2).add_(on_diagonal.transpose(1, 2))
        return torch.where(pair_mask, F.leaky_relu(out, 0.01), 0.0)

    def _array_terms(self, x: torch.Tensor) -> torch.Tensor:
        """Elements 0 and 1, the two mixes of the full array."""
        return x @ self.weight[:, 0] + (x @ self.weight[:, 1]).transpose(1, 2)

    def _scalar_terms(
        self, trace: torch.Tensor, total: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Elements 11 and 13, one value per jet spread over the whole array."""
        mixed = scaled_mix(trace, scale[..., 6], self.weight[:, 11])
        mixed = mixed + scaled_mix(total, scale[..., 8], self.weight[:, 13])
        return mixed.unsqueeze(1)

    def _broadcast_terms(
        self,
        diag: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Elements 4, 7, 10 (constant along a row) and 3, 6, 9 (down a column)."""
        by_row = diag @ self.weight[:, 4]
        by_row = by_row + scaled_mix(cols, scale[..., 2], self.weight[:, 7])
        by_row = by_row + scaled_mix(rows, scale[..., 5], self.weight[:, 10])
        by_col = diag @ self.weight[:, 3]
        by_col = by_col + scaled_mix(cols, scale[..., 1], self.weight[:, 6])
        by_col = by_col + scaled_mix(rows, scale[..., 4], self.weight[:, 9])
        return by_row, by_col

    def _diagonal_terms(
        self,
        diag: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        trace: torch.Tensor,
        total: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Elements 2, 5, 8, 12, 14, the ones living on the diagonal, and its bias."""
        out = diag @ self.weight[:, 2]
        out = out + scaled_mix(cols, scale[..., 0], self.weight[:, 5])
        out = out + scaled_mix(rows, scale[..., 3], self.weight[:, 8])
        out = out + scaled_mix(trace, scale[..., 7], self.weight[:, 12])
        out = out + scaled_mix(total, scale[..., 9], self.weight[:, 14])
        return out + self.diag_bias


class LinEq2to0(nn.Module):
    """The invariant map ``(B, N, N, in_dim) -> (B, out_dim)``, two basis elements.

    The total and the trace, each with its own learnable exponent on :math:`N/\\bar N`.
    The activation sits on the aggregated values and not on the output, as in the
    reference's ``activate_agg_out``.
    """

    def __init__(self, in_dim: int, out_dim: int, nbar: float = 49.0) -> None:
        super().__init__()
        self.nbar = nbar
        std = (4.0 / (2 * in_dim)) ** 0.5
        self.weight = nn.Parameter(torch.normal(0.0, std, (in_dim, 2, out_dim)))
        self.alphas = nn.Parameter(torch.rand(in_dim, 2))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, nobj: torch.Tensor) -> torch.Tensor:
        """Aggregate the pair tensor over both particle axes."""
        diag = torch.diagonal(x, dim1=1, dim2=2)
        aggs = torch.stack([x.sum((1, 2)) / nobj**2, diag.sum(-1) / nobj], dim=-1)
        aggs = F.leaky_relu(aggs * (nobj / self.nbar).view(-1, 1, 1) ** self.alphas)
        return torch.einsum("bcg,cgs->bs", aggs, self.weight) + self.bias


class MaskedBatchNorm(nn.Module):
    """Batch normalisation of a ``(B, N, N, C)`` pair tensor over its real pairs.

    The statistics run over the batch and both particle axes, and the count is the
    number of real pairs rather than of slots, so a padded jet does not drag the mean
    towards zero. The padded pairs must be zero on entry, which every layer here
    guarantees. The running estimates are registered buffers, since the FLOP-counting
    pass snapshots and restores buffers alone.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, momentum: float = 0.1
    ) -> None:
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Normalise *x* channel by channel, then re-mask it."""
        if self.training:
            mean, var = self._batch_statistics(x, pair_mask)
        else:
            mean, var = self.running_mean, self.running_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        return torch.where(pair_mask, x * scale + (self.bias - mean * scale), 0.0)

    def _batch_statistics(
        self, x: torch.Tensor, pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and biased variance of the real pairs, folded into the running
        estimates."""
        n = pair_mask.sum()
        mean = x.sum((0, 1, 2)) / n
        var = x.pow(2).sum((0, 1, 2)) / n - mean.pow(2)
        self._track(mean.detach(), var.detach() * n / (n - 1).clamp(min=1))
        return mean, var

    @torch.no_grad()
    def _track(self, mean: torch.Tensor, var: torch.Tensor) -> None:
        """Fold one batch into the estimates that the evaluation mode reads."""
        self.num_batches_tracked += 1
        self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
        self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)


class MessageNet(nn.Module):
    """The channel mix between two equivariant layers: linear, LeakyReLU, batch norm."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = MaskedBatchNorm(out_dim)

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        """Mix the channels of every pair, independently of the particle indices."""
        x = F.leaky_relu(self.linear(x), 0.01, inplace=True)
        return self.norm(torch.where(pair_mask, x, 0.0), pair_mask)


class Eq2to2Block(nn.Module):
    """One PELICAN block: a channel mix, dropout, then the equivariant aggregation.

    With ``n_message`` channels the mix is an MLP whose batch normalisation ends it, as
    in the paper. With ``n_message = 0`` the block keeps a bare normalisation of its
    input, so that a deep stack without messaging is still normalised.
    """

    def __init__(
        self, n_hidden: int, n_message: int, dropout: float, nbar: float
    ) -> None:
        super().__init__()
        self.message = MessageNet(n_hidden, n_message) if n_message else None
        self.norm = None if n_message else MaskedBatchNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.aggregate = LinEq2to2(n_message or n_hidden, n_hidden, nbar)

    def forward(
        self, x: torch.Tensor, pair_mask: torch.Tensor, nobj: torch.Tensor
    ) -> torch.Tensor:
        """Take the pair tensor through the block, at unchanged width."""
        if self.message is not None:
            x = self.message(x, pair_mask)
        else:
            x = self.norm(x, pair_mask)
        return self.aggregate(self.dropout(x), pair_mask, nobj)


class PELICANClassifier(nn.Module):
    """PELICAN over a batch of jets that are a fixed-width four-momentum tensor.

    The logits are invariant under any permutation of the constituents and, with the
    beams off, under the full Lorentz group; with them on, under SO(2) about the beam
    axis and the reflection of the beam axis (see the module docstring). The FLOP count
    this network reports is for ``nconstituents + 2`` fully occupied particles, since
    the beams are prepended inside ``forward``.

    :param n_hidden: channels of the pair tensor between blocks.
    :param n_message: channels of the messaging MLP inside a block, 0 to drop it.
    :param nbar: mean multiplicity, the beams included, that the aggregators scale by.
    :param add_beams: prepend the two beam four-vectors and their label.
    """

    def __init__(
        self,
        n_hidden: int = 25,
        n_message: int = 15,
        n_blocks: int = 5,
        dropout: float = 0.025,
        nbar: float = 49.0,
        add_beams: bool = True,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        self.add_beams = add_beams
        metric = torch.tensor([1.0, -1.0, -1.0, -1.0])
        beams = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]])
        self.register_buffer("metric", metric, persistent=False)
        self.register_buffer("beams", beams, persistent=False)
        # The beam label reaches the pair tensor as the two channels s_i and s_j.
        self.embedding = InputEmbedding(n_hidden - 2 if add_beams else n_hidden)
        self.blocks = nn.ModuleList(
            [Eq2to2Block(n_hidden, n_message, dropout, nbar) for _ in range(n_blocks)]
        )
        self.message = MessageNet(n_hidden, n_message) if n_message else None
        self.aggregate = LinEq2to0(n_message or n_hidden, n_hidden, nbar)
        self.decoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_hidden, n_class))

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits, unchanged by how many slots the padding fills.
        """
        pmu, is_beam = self._with_beams(pmu)
        real = pmu[..., 0] != 0.0
        pair_mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(-1)
        nobj = real.sum(-1, keepdim=True).clamp(min=1).to(pmu.dtype)
        x = self._pair_inputs(pmu, is_beam, real)
        for block in self.blocks:
            x = block(x, pair_mask, nobj)
        if self.message is not None:
            x = self.message(x, pair_mask)
        return self.decoder(self.aggregate(x, nobj))

    def _with_beams(
        self, pmu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepend the two beam four-vectors, and flag them for the pair channels.

        The beams are the null vectors (1, 0, 0, +-1), carrying one binary label between
        them: the network cannot tell the forward beam from the backward one.
        """
        if not self.add_beams:
            return pmu, None
        beams = self.beams.to(pmu.dtype).expand(pmu.shape[0], 2, 4)
        is_beam = pmu.new_zeros(pmu.shape[0], pmu.shape[1] + 2)
        is_beam[:, :2] = 1.0
        return torch.cat([beams, pmu], dim=1), is_beam

    def _pair_inputs(
        self, pmu: torch.Tensor, is_beam: torch.Tensor | None, real: torch.Tensor
    ) -> torch.Tensor:
        """The pair tensor: the embedded invariants, plus the beam labels of i and j.

        The label is promoted the way an Eq1to2 layer would promote it, as the pair of
        channels s_i = is_beam_i real_j and s_j = is_beam_j real_i, both of which vanish
        outside the pair mask.
        """
        x = self.embedding(pair_dots(pmu, self.metric))
        if is_beam is None:
            return x
        s_i = is_beam.unsqueeze(-1) * real.unsqueeze(1)
        return torch.cat([x, s_i.unsqueeze(-1), s_i.transpose(1, 2).unsqueeze(-1)], -1)
