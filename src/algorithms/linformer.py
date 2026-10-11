"""Single-head Linformer jet tagger in float (Laatu et al., arXiv:2510.24784).

The attention component is the Linformer of Wang et al. (arXiv:2006.04768) as
``QLinformerAttention`` of github.com/calad0i/HGQ2 (``src/hgq/layers/attn/linformer.py``)
implements it.

We change that code in four places:
* the network is float only, with plain dense layers where the code has LUT-based ones
* padded slots are masked, where the code has no mask
* the readout pool is one of `AGGREGATIONS`, where the code averages over all N slots
* BatchNorm is a switch

The network is not permutation-invariant. The projections of the keys and values over
the slot axis have one weight per slot, which is one per p_T rank.
We sort the data by pT, since we can do this easily in the trigger.
The parameter count grows with N by 2Nk.
The slots are read in the record's order; there's no sort in the record.
"""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.mlp import dense_stack
from src.algorithms.components.norm import SeqNorm
from src.algorithms.components.pooling import AGGREGATIONS, _check_choice, _masked_pool


def _slot_projection(nconstituents: int, kv_proj_dim: int) -> nn.Parameter:
    """Return an (N, k) projection over the constituent slots, Glorot-uniform as the
    reference's ``glorot_uniform``."""
    weight = torch.empty(nconstituents, kv_proj_dim)
    return nn.Parameter(nn.init.xavier_uniform_(weight))


class LinformerAttention(nn.Module):
    """Single-head attention whose keys and values are read from k learned mixtures.

    Before its dense layer, the key input is projected over the slot axis by E_k, a
    bias-free (N, k) matrix, and the value input by E_v, a second one; the key dense
    layer has no bias. Padded slots are zeroed before both projections. Their queries
    are still computed, and the tokens they give enter the batch statistics of the
    BatchNorm in the feed-forward sub-layer.
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
        """Return the attention output for tokens of shape (jets, constituents, dim).

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
    """Return Linear, BatchNorm if asked, then ReLU."""
    # The BatchNorm statistics include the padded slots, as in the reference, which has
    # no mask, and in JetFormer's batch option.
    norm = SeqNorm(out_dim, "batch") if batchnorm else nn.Identity()
    return nn.Sequential(nn.Linear(in_dim, out_dim), norm, nn.ReLU())


class LinformerNet(nn.Module):
    """Embed the constituents, run one Linformer block and read the pool of its tokens.

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
        _check_choice("aggregation", aggregation, AGGREGATIONS)
        self.aggregation = aggregation
        self.embed = _token_dense(in_dim, embed_dim, batchnorm)
        self.attention = LinformerAttention(embed_dim, nconstituents, kv_proj_dim)
        self.feed_forward = nn.Sequential(
            _token_dense(embed_dim, ff_dim, batchnorm), nn.Linear(ff_dim, embed_dim)
        )
        self.head = dense_stack(embed_dim, head_layers, out_dim, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits."""
        # A padded slot is an all-zero row but does not embed to zero, since the first
        # bias survives the dense layer. The mask is keeping it out of the pool.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        h = self.embed(x)
        h = h + self.attention(h, real)
        h = h + self.feed_forward(h)
        return self.head(_masked_pool(h, real, self.aggregation))


class Linformer(JetTaggingModule):
    """Jet tagger reading the constituents as tokens with one Linformer block."""

    def build_net(
        self,
        embed_dim: int,
        kv_proj_dim: int,
        ff_dim: int,
        head_layers: list[int],
        aggregation: str = "sum_over_n",
        batchnorm: bool = True,
    ) -> nn.Module:
        """Build the network; the arguments are in :class:`LinformerNet`."""
        return LinformerNet(
            in_dim=self.hparams.nbr_feats,
            nconstituents=self.hparams.nconstituents,
            embed_dim=embed_dim,
            kv_proj_dim=kv_proj_dim,
            ff_dim=ff_dim,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            batchnorm=batchnorm,
        )
