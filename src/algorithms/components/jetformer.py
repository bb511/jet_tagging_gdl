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
* The config picks layer norm where the reference's search fixed batch norm, and the
  search space differs.
"""

import torch
from torch import nn


class SeqNorm(nn.Module):
    """BatchNorm1d or LayerNorm over the channel axis of a (jets, tokens, channels)."""

    def __init__(self, dim: int, normalization: str = "batch") -> None:
        super().__init__()
        self.normalization = normalization
        self.norm = (
            nn.BatchNorm1d(dim) if normalization == "batch" else nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise over the channel axis, transposing for BatchNorm1d."""
        if self.normalization == "batch":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class JetFormerAttention(nn.Module):
    """Pre-norm multi-head self-attention, no padding mask."""

    def __init__(self, dim: int, num_heads: int, normalization: str = "batch") -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = SeqNorm(dim, normalization)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
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

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.attn = JetFormerAttention(dim, num_heads, normalization)

        act_cls = nn.ReLU if activation == "relu" else nn.SiLU
        self.norm_0 = SeqNorm(dim, normalization)
        self.activ_0 = act_cls()
        self.linear_0 = nn.Linear(dim, dim * 2, bias=False)
        self.norm_1 = SeqNorm(dim * 2, normalization)
        self.activ_1 = act_cls()
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
    :param normalization: either ``batch`` or ``layer``, used by every norm here.
    :param activation: either ``relu`` or ``silu``, used in the feed-forward sub-layers.
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_heads: int,
        num_transformers: int,
        out_dim: int,
        normalization: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        block_args = (embed_dim, num_heads, dropout, activation, normalization)
        self.blocks = nn.ModuleList(
            [JetFormerBlock(*block_args) for _ in range(num_transformers)]
        )
        self.norm = (
            nn.BatchNorm1d(embed_dim)
            if normalization == "batch"
            else nn.LayerNorm(embed_dim)
        )
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
