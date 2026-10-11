"""PELICAN, the permutation-equivariant Lorentz aggregator of arXiv:2307.16506.

A jet is converted to the Minkowski products of its constituents. The network is a stack
of 15 linear maps that take this set of products to another, interleaved with channel
mixes, as described in (Maron et al., arXiv:1812.09902).

Shape conventions. Every jet is specified by its constituents, and every constituent is
specified by its momentum 4-vector in GeV. The number of constituents is padded to N for
every jet. Two 4-vector describing the beam directions are appended to each jet.

Symmetry. With the beams on, the exact invariance is SO(2) about the beam axis together
with `p_z -> -p_z`, which merely swaps two beams with the same label. However, all the
logits are Lorentz invariant, given only the constituents as input.

Batches must be de-normalised before they reach this module.
"""

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms import FourMomentumModule


def pair_dots(p: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """Minkowski products of every pair, ``(B, N, N, 1)``, metric (+, -, -, -)."""
    return torch.matmul(p, (p * metric).transpose(1, 2)).unsqueeze(-1)


def pair_reductions(x: torch.Tensor, nobj: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """The compact sources of the basis: ``(diag, rows, cols, trace, total)``."""
    n = nobj.unsqueeze(-1)
    diag = torch.diagonal(x, dim1=1, dim2=2).transpose(1, 2)
    rows, cols = x.sum(2) / n, x.sum(1) / n
    trace = diag.sum(1, keepdim=True) / n
    total = rows.sum(1, keepdim=True) / n
    return diag, rows, cols, trace, total


def scaled_mix(v: torch.Tensor, scale: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mix the channels of a compact source, each rescaled by its own power of N."""
    return (v * scale) @ w


def real_width(real: torch.Tensor) -> torch.Tensor:
    """Slots up to and including the last real one of every jet, ``(B,)``."""
    # From the mask, not the count: the record's loader leads with the real slots, but
    # nothing here may rely on that ordering.
    slots = torch.arange(1, real.shape[1] + 1, device=real.device)
    return (real * slots).amax(dim=1)


def chunk_widths(widths: torch.Tensor, pieces: list[torch.Tensor]) -> list[int]:
    """The chunking partitioning, with global syncing."""
    return torch.stack([widths[piece].amax() for piece in pieces]).clamp(min=1).tolist()


def invert_permutation(order: torch.Tensor) -> torch.Tensor:
    """Indices putting a batch sorted by *order* back into the order it arrived in."""
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=order.device)
    return inverse


class InputEmbedding(nn.Module):
    """Spread one Minkowski product over *n_embed* channels of compressed magnitude.

    This embedding is used as a form of normalisation: soft constituent 4-vector dot
    products become of comparable magnitude as hard constituent dot products. This is
    done by applying a Box-Cox transform with a trainable parameter. The network learns
    the best embedding during its training. The different channels then separate the
    different momentum scales in the data.
    """

    def __init__(self, n_embed: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_embed))
        self.alphas = nn.Parameter(torch.linspace(0.05, 0.5, n_embed))

    def forward(self, dots: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, N, N, 1)`` products into ``(B, N, N, n_embed)``.

        f(0) = 0, so the padded pairs stay zero without a mask.
        """
        a = 1e-6 + self.alphas**2
        y = dots * self.scale
        return torch.sign(y) * ((1.0 + y.abs()).pow(a) - 1.0) / a


class LinEq2to2(nn.Module):
    """The equivariant linear map ``(B, N, N, in_dim) -> (B, N, N, out_dim)``.

    Maps one 4-vector matrix to another using a mix of the 15 operations that leave this
    matrix permutation equivariant. The matrix is made of Lorentz invariants.
    This module is a layer in the PELICAN network. Weight slot k holds basis element k;
    elements 0 to 9 are averages over the multiplicity and each carries a learned
    exponent, ``alphas[:, k]``, that rescales it by ``(N / nbar) ** alpha``:

    ====  ==========  ==================
    k     source      placed
    ====  ==========  ==================
    0     col means   on the diagonal
    1     col means   down column j
    2     col means   along row i
    3     row means   on the diagonal
    4     row means   down column j
    5     row means   along row i
    6     trace       everywhere
    7     trace       on the diagonal
    8     total       everywhere
    9     total       on the diagonal
    10    array       as is
    11    array       transposed
    12    diagonal    on the diagonal
    13    diagonal    down column j
    14    diagonal    along row i
    ====  ==========  ==================

    :param nbar: the mean multiplicity the exponents are measured against.
    """

    def __init__(self, in_dim: int, out_dim: int, nbar: float = 49.0) -> None:
        super().__init__()
        self.nbar = nbar
        # Weight initialisation, He (Kaiming) style.
        # This is done for stability reasons when stacking multiple layers.
        std = (2.0 / (15 * in_dim)) ** 0.5
        self.weight = nn.Parameter(torch.normal(0.0, std, (in_dim, 15, out_dim)))
        self.alphas = nn.Parameter(torch.rand(in_dim, 10))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.diag_bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self, x: torch.Tensor, pair_mask: torch.Tensor, nobj: torch.Tensor
    ) -> torch.Tensor:
        """Return the re-masked, LeakyReLU image of the pair tensor *x*.

        The 10 aggregation operations are scaled by *scale*, which is a learned scale
        through *alpha*. When *alpha=0*, the aggregation operations are means, and when
        *alpha=1* the aggregation operations are effectively sums.
        """
        scale = (nobj / self.nbar).view(-1, 1, 1, 1) ** self.alphas
        diag, rows, cols, trace, total = pair_reductions(x, nobj)
        by_row, by_col = self._broadcast_terms(diag, rows, cols, scale)
        by_row = by_row + self._scalar_terms(trace, total, scale) + self.bias

        out = self._array_terms(x) + by_row.unsqueeze(2)
        out += by_col.unsqueeze(1)
        on_diagonal = self._diagonal_terms(diag, rows, cols, trace, total, scale)
        out.diagonal(dim1=1, dim2=2).add_(on_diagonal.transpose(1, 2))
        return torch.where(pair_mask, F.leaky_relu(out, 0.01), 0.0)

    def _array_terms(self, x: torch.Tensor) -> torch.Tensor:
        """Elements 10 and 11, the two mixes of the full array."""
        return x @ self.weight[:, 10] + (x @ self.weight[:, 11]).transpose(1, 2)

    def _scalar_terms(
        self, trace: torch.Tensor, total: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Elements 6 and 8, one value per jet, ``(B, 1, out_dim)``."""
        mixed = scaled_mix(trace, scale[..., 6], self.weight[:, 6])
        return mixed + scaled_mix(total, scale[..., 8], self.weight[:, 8])

    def _broadcast_terms(
        self,
        diag: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Elements 14, 2, 5 (constant along a row) and 13, 1, 4 (down a column)."""
        by_row = diag @ self.weight[:, 14]
        by_row = by_row + scaled_mix(cols, scale[..., 2], self.weight[:, 2])
        by_row = by_row + scaled_mix(rows, scale[..., 5], self.weight[:, 5])
        by_col = diag @ self.weight[:, 13]
        by_col = by_col + scaled_mix(cols, scale[..., 1], self.weight[:, 1])
        by_col = by_col + scaled_mix(rows, scale[..., 4], self.weight[:, 4])
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
        """Elements 12, 0, 3, 7, 9, the ones living on the diagonal, and its bias."""
        out = diag @ self.weight[:, 12]
        out = out + scaled_mix(cols, scale[..., 0], self.weight[:, 0])
        out = out + scaled_mix(rows, scale[..., 3], self.weight[:, 3])
        out = out + scaled_mix(trace, scale[..., 7], self.weight[:, 7])
        out = out + scaled_mix(total, scale[..., 9], self.weight[:, 9])
        return out + self.diag_bias


class LinEq2to0(nn.Module):
    """The invariant map ``(B, N, N, in_dim) -> (B, out_dim)``, two basis elements.

    The total and the trace, each with its own learnable exponent. These are the only
    two operations which are permutation equivariant. After this aggregation,
    permutation equivariance becomes permutation invariance.
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

    This is such that a padded jet does not artificially have almost 0 mean.
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

    def forward(
        self, xs: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Normalise every chunk channel by channel, then re-mask it."""
        if self.training:
            mean, var = self._batch_statistics(xs, masks)
        else:
            mean, var = self.running_mean, self.running_var
        scale = self.weight * torch.rsqrt(var + self.eps)
        shift = self.bias - mean * scale
        return [torch.where(m, x * scale + shift, 0.0) for x, m in zip(xs, masks)]

    def _batch_statistics(
        self, xs: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and biased variance of the real pairs, into the running estimates."""
        n = sum(mask.sum() for mask in masks)
        mean = sum(x.sum((0, 1, 2)) for x in xs) / n
        var = sum(x.pow(2).sum((0, 1, 2)) for x in xs) / n - mean.pow(2)
        # Once per forward, whatever the chunk count, or the running estimates would
        # decay n_chunks times faster than the training-mode statistics.
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

    def forward(
        self, xs: list[torch.Tensor], masks: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Mix the channels of every pair, independently of the particle indices."""
        xs = [F.leaky_relu(self.linear(x), 0.01, inplace=True) for x in xs]
        return self.norm([torch.where(m, x, 0.0) for x, m in zip(xs, masks)], masks)


class Eq2to2Block(nn.Module):
    """One PELICAN block: a channel mix, dropout, then the equivariant aggregation."""

    def __init__(
        self, n_hidden: int, n_message: int, dropout: float, nbar: float
    ) -> None:
        super().__init__()
        self.message = MessageNet(n_hidden, n_message) if n_message else None
        self.norm = None if n_message else MaskedBatchNorm(n_hidden)
        self.dropout = nn.Dropout(dropout)
        self.aggregate = LinEq2to2(n_message or n_hidden, n_hidden, nbar)

    def forward(
        self,
        xs: list[torch.Tensor],
        masks: list[torch.Tensor],
        nobjs: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Take every chunk of the pair tensor through the block, at unchanged width."""
        if self.message is not None:
            xs = self.message(xs, masks)
        else:
            xs = self.norm(xs, masks)
        pairs = zip(xs, masks, nobjs)
        return [self.aggregate(self.dropout(x), m, n) for x, m, n in pairs]


def _blocks(
    n_blocks: int, n_hidden: int, n_message: int, dropout: float, nbar: float
) -> nn.ModuleList:
    return nn.ModuleList(
        [Eq2to2Block(n_hidden, n_message, dropout, nbar) for _ in range(n_blocks)]
    )


class PELICANClassifier(nn.Module):
    """PELICAN over a batch of jets that are a fixed-width four-momentum tensor.

    The logits are invariant under any permutation of the constituents and, with the
    beams off, under the full Lorentz group; with them on, under SO(2) about the beam
    axis and the reflection of the beam axis.

    :param n_hidden: channels of the pair tensor between blocks.
    :param n_message: channels of the messaging MLP inside a block, 0 to drop it.
    :param nbar: mean multiplicity, the beams included, that the aggregators scale by.
    :param add_beams: prepend the two beam four-vectors and their label.
    :param n_chunks: multiplicity-sorted chunks the batch is cut into, each trimmed to
        its own widest jet; a speed knob, exact up to the order of the summations.
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
        n_chunks: int = 1,
    ) -> None:
        super().__init__()
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be at least 1, got {n_chunks}")
        self.add_beams = add_beams
        self.n_chunks = n_chunks
        metric = torch.tensor([1.0, -1.0, -1.0, -1.0])
        beams = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]])
        self.register_buffer("metric", metric, persistent=False)
        self.register_buffer("beams", beams, persistent=False)
        # The beam label reaches the pair tensor as the two channels s_i and s_j.
        self.embedding = InputEmbedding(n_hidden - 2 if add_beams else n_hidden)
        self.blocks = _blocks(n_blocks, n_hidden, n_message, dropout, nbar)
        self.message = MessageNet(n_hidden, n_message) if n_message else None
        self.aggregate = LinEq2to0(n_message or n_hidden, n_hidden, nbar)
        self.decoder = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_hidden, n_class))

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits, unchanged by how many slots the padding fills.
        """
        order, chunks = self._split(pmu)
        chunk_inputs = (self._chunk_inputs(chunk) for chunk in chunks)
        xs, masks, nobjs = map(list, zip(*chunk_inputs))
        for block in self.blocks:
            xs = block(xs, masks, nobjs)
        if self.message is not None:
            xs = self.message(xs, masks)
        pooled = torch.cat([self.aggregate(x, n) for x, n in zip(xs, nobjs)])
        # The labels of a step are never permuted, so the sort has to be undone here.
        return self.decoder(pooled[invert_permutation(order)])

    def _split(self, pmu: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Sort the jets by multiplicity and cut them into chunks.

        Every chunk is trimmed to the last real slot of its own widest jet, which is
        where the padded pair slots are saved.

        :return: the sorting permutation and the chunks, in its order.
        """
        real = pmu[..., 0] != 0.0
        order = torch.arange(pmu.shape[0], device=pmu.device)
        if self.n_chunks > 1:
            order = real.sum(-1).argsort()
        # A batch shorter than the chunk count leaves empty pieces behind.
        split = torch.tensor_split(order, self.n_chunks)
        pieces = [piece for piece in split if piece.numel()]
        widths = chunk_widths(real_width(real), pieces)
        return order, [pmu[p][:, :width] for p, width in zip(pieces, widths)]

    def _chunk_inputs(
        self, pmu: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The pair tensor, the pair mask and the real count of one chunk."""
        pmu, is_beam = self._with_beams(pmu)
        real = pmu[..., 0] != 0.0
        pair_mask = (real.unsqueeze(2) & real.unsqueeze(1)).unsqueeze(-1)
        nobj = real.sum(-1, keepdim=True).clamp(min=1).to(pmu.dtype)
        return self._pair_inputs(pmu, is_beam, real), pair_mask, nobj

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


class PELICAN(FourMomentumModule):
    """Jet tagger built on permutation-equivariant aggregators of the Minkowski products
    of the constituents (arXiv:2307.16506)."""

    def build_net(
        self,
        n_hidden: int = 25,
        n_message: int = 15,
        n_blocks: int = 5,
        dropout: float = 0.025,
        nbar: float = 49.0,
        add_beams: bool = True,
        n_chunks: int = 1,
    ) -> nn.Module:
        """Build the aggregator network; the arguments are in
        :class:`PELICANClassifier`."""
        self._register_denorm_scales()
        return PELICANClassifier(
            n_hidden=n_hidden,
            n_message=n_message,
            n_blocks=n_blocks,
            dropout=dropout,
            nbar=nbar,
            add_beams=add_beams,
            n_chunks=n_chunks,
            n_class=self.hparams.output_dim,
        )
