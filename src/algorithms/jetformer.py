"""Custom transformer matching the reference JetFormer (arXiv:2601.17215).

Unlike a standard Transformer, this has bias-free Q/K/V projs, a pre-activation FFN
(norm -> activation -> linear, not linear -> activation) with bias-free linears and a
x2 expansion, dropout on the residual stream after each block rather than on the branch,
no padding mask, no norm after the embedding, and a learned CLS token read out through
one norm and one linear layer.

Divergences from the reference:

* The reference registers a BatchNorm1d or LayerNorm over the flattened (N + 1)^2
  attention-score matrix that is never actually used. We leave it out.
* The network returns logits, where the reference returns log-probabilities; the
  cross-entropy loss applies the same log-softmax, so the objective is the same.
* Every norm is a layer norm where the reference's search fixed batch norm, and the
  search space differs.
"""

import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.norm import SeqNorm


def _qkv_projections(dim: int) -> tuple[nn.Linear, ...]:
    return tuple(nn.Linear(dim, dim, bias=False) for _ in range(3))


class JetFormerAttention(nn.Module):
    """Pre-norm multi-head self-attention, no padding mask."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = SeqNorm(dim, "layer")
        self.q, self.k, self.v = _qkv_projections(dim)
        self.out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over the tokens of a (jets, tokens, channels) batch and add result."""
        B, N, _ = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, N, self.num_heads, self.head_dim)
        k = self.k(h).view(B, N, self.num_heads, self.head_dim)
        v = self.v(h).view(B, N, self.num_heads, self.head_dim)

        scores = torch.einsum("bqhc,bkhc->bhqk", q, k) / (self.head_dim**0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhc->bqhc", attn, v).reshape(B, N, -1)
        return x + self.out(out)


class JetFormerBlock(nn.Module):
    """Self-attention sub-layer followed by a x2-expansion pre-activation FFN."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = JetFormerAttention(dim, num_heads)

        self.norm_0 = SeqNorm(dim, "layer")
        self.activ_0 = nn.ReLU()
        self.linear_0 = nn.Linear(dim, dim * 2, bias=False)
        self.norm_1 = SeqNorm(dim * 2, "layer")
        self.activ_1 = nn.ReLU()
        self.linear_1 = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attention, then the feed-forward sub-layer, each residual; dropout on the
        sum."""
        x = self.attn(x)
        h = self.linear_0(self.activ_0(self.norm_0(x)))
        h = self.linear_1(self.activ_1(self.norm_1(h)))
        return self.dropout(x + h)


class JetFormerNet(nn.Module):
    """Embed the constituents, prepend a CLS token, run the blocks, read token 0.

    :param num_heads: attention heads per block, which must divide ``embed_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_heads: int,
        num_transformers: int,
        out_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        block_args = (embed_dim, num_heads, dropout)
        self.blocks = nn.ModuleList(
            [JetFormerBlock(*block_args) for _ in range(num_transformers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents,
        features)."""
        x = self.embedding(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # (jets, 1 + constituents, embed_dim)
        for block in self.blocks:
            x = block(x)
        return self.classifier(self.norm(x[:, 0, :]))


class JetFormer(JetTaggingModule):
    """Jet tagger reading the constituents as a token sequence with a CLS token."""

    def build_net(
        self,
        embed_dim: int,
        num_heads: int,
        num_transformers: int,
        normalization: str = "layer",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the network; the arguments are documented on :class:`JetFormerNet`.

        ``normalization`` must be ``layer`` and ``activation`` ``relu``; the arguments
        stay because the checkpoints save them as hyperparameters.
        """
        if (normalization, activation) != ("layer", "relu"):
            raise ValueError(f"Unsupported options: {normalization}, {activation}")
        return JetFormerNet(
            in_dim=self.hparams.nbr_feats,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformers=num_transformers,
            out_dim=self.hparams.output_dim,
            dropout=dropout,
        )
