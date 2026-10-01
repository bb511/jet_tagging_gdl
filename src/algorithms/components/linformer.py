"""Single-head Linformer jet tagger in float, after Laatu et al., "Sub-microsecond
Transformers for Jet Tagging on FPGAs" (arXiv:2510.24784).

The paper fixes one head and no positional encoding and gives no widths. The block
layout follows ``get_llformer`` in github.com/calad0i/HGQ2-examples
(``jsc150/model.py``): a dense embedding, one attention and one feed-forward sub-layer,
each with a residual, a global average pool and a small head, with BatchNorm, which
folds into the dense layers at inference. Its point in float (width 16, k = 4, head 16)
is the config default. The attention is the Linformer of Wang et al. (arXiv:2006.04768)
as ``QLinformerAttention`` of github.com/calad0i/HGQ2
(``src/hgq/layers/attn/linformer.py``) implements it. Four departures from that code
are deliberate: the network is float only, with plain dense layers where the code has
LUT-based ones and no quantisation; padded slots are masked, where the code has no
mask; the readout pool is one of `AGGREGATIONS`, where the code averages over all N
slots, which ``sum_over_n`` restricts to the real constituents; and BatchNorm is a
switch.

The network is not permutation-invariant. The projections of the keys and values over
the slot axis carry one weight per slot, which is one per p_T rank because the record is
sorted by p_T, so the parameter count grows with N by 2Nk. The slots are read in the
record's order, with no sort inside the network, as the MLP row reads them.
"""

import torch
from torch import nn

from src.algorithms.components.deepsets import _masked_pool
from src.algorithms.components.jetformer import SeqNorm

AGGREGATIONS = ("sum", "max", "sum_over_n")


def _check_aggregation(aggregation: str) -> None:
    """Refuse a readout pool this module does not provide, by name."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
        )


def _pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`. ``sum_over_n`` divides the masked sum by
        the slot count N, the same for every jet: the reference's global average pool,
        restricted to the real constituents.
    """
    if aggregation == "sum_over_n":
        return _masked_pool(h, real, "sum") / h.shape[1]
    return _masked_pool(h, real, aggregation)


def _slot_projection(nconstituents: int, kv_proj_dim: int) -> nn.Parameter:
    """Return an (N, k) projection over the constituent slots, Glorot-uniform as the
    reference's ``glorot_uniform``."""
    weight = torch.empty(nconstituents, kv_proj_dim)
    return nn.Parameter(nn.init.xavier_uniform_(weight))


class LinformerAttention(nn.Module):
    """Single-head attention whose keys and values are read from k learned mixtures of
    the N constituent slots.

    Before its dense layer, the key input is projected over the slot axis by E_k, a
    bias-free (N, k) matrix, and the value input by E_v, a second one; the key dense
    layer has no bias. Padded slots are zeroed before both projections. Their queries
    are still computed, and the tokens they give enter nothing but the batch statistics
    of the feed-forward BatchNorm.
    """

    def __init__(self, dim: int, nconstituents: int, kv_proj_dim: int) -> None:
        super().__init__()
        self.scale = dim**-0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.e_k = _slot_projection(nconstituents, kv_proj_dim)
        self.e_v = _slot_projection(nconstituents, kv_proj_dim)

    def forward(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Return the attention output for tokens of shape (jets, constituents, dim),
        without the residual.

        :param real: (jets, constituents, 1), True for a real constituent.
        """
        masked = h * real
        # (jets, k, dim): the N slots mixed down to k.
        keys = self.k(torch.einsum("bnc,nk->bkc", masked, self.e_k))
        values = self.v(torch.einsum("bnc,nk->bkc", masked, self.e_v))
        scores = torch.einsum("bnc,bkc->bnk", self.q(h), keys) * self.scale
        attention = torch.softmax(scores, dim=-1)
        return self.out(torch.einsum("bnk,bkc->bnc", attention, values))


def _token_dense(in_dim: int, out_dim: int, batchnorm: bool) -> nn.Sequential:
    """Return Linear, BatchNorm if asked, then ReLU, for tokens of shape (jets,
    constituents, channels)."""
    # The BatchNorm statistics include the padded slots, as in the reference, which has
    # no mask, and in JetFormer's batch option.
    norm = SeqNorm(out_dim, "batch") if batchnorm else nn.Identity()
    return nn.Sequential(nn.Linear(in_dim, out_dim), norm, nn.ReLU())


def _head(widths: list[int], out_dim: int, batchnorm: bool) -> nn.Sequential:
    """Stack Linear, BatchNorm if asked and ReLU for each step of *widths*, then a
    linear layer to the logits."""
    layers: list[nn.Module] = []
    for width_in, width_out in zip(widths, widths[1:]):
        norm = nn.BatchNorm1d(width_out) if batchnorm else nn.Identity()
        layers += [nn.Linear(width_in, width_out), norm, nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(widths[-1], out_dim))


class LinformerNet(nn.Module):
    """Embed the constituents, run one Linformer block and read the pool of its tokens
    with a small head.

    With h the embedded tokens, the block adds attention(h) to h and then adds the
    output of the feed-forward sub-layer, W2 ReLU(BN(W1 h + b1)) + b2, to the result.
    The head is Linear, BN and ReLU for each width of ``head_layers``, then a linear
    layer to the logits. ReLU is the only activation.

    :param nconstituents: N, the slot count, which fixes the shape of E_k and E_v.
    :param kv_proj_dim: k, the number of slots the keys and values are projected onto.
    :param ff_dim: hidden width of the feed-forward sub-layer.
    :param head_layers: hidden widths of the head.
    :param aggregation: readout pool over the real constituents, one of `AGGREGATIONS`.
    :param batchnorm: put a BatchNorm before every ReLU.
    """

    def __init__(
        self,
        in_dim: int,
        nconstituents: int,
        embed_dim: int,
        kv_proj_dim: int,
        ff_dim: int,
        head_layers: list[int],
        out_dim: int,
        aggregation: str = "sum_over_n",
        batchnorm: bool = True,
    ) -> None:
        super().__init__()
        _check_aggregation(aggregation)
        self.aggregation = aggregation
        self.embed = _token_dense(in_dim, embed_dim, batchnorm)
        self.attention = LinformerAttention(embed_dim, nconstituents, kv_proj_dim)
        self.feed_forward = nn.Sequential(
            _token_dense(embed_dim, ff_dim, batchnorm), nn.Linear(ff_dim, embed_dim)
        )
        self.head = _head([embed_dim, *head_layers], out_dim, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents,
        features)."""
        # A padded slot is an all-zero row but does not embed to zero, since the first
        # bias survives the dense layer, so the mask, not the input, keeps it out of
        # the keys, the values and the pool.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        h = self.embed(x)
        h = h + self.attention(h, real)
        h = h + self.feed_forward(h)
        return self.head(_pool(h, real, self.aggregation))
