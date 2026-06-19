"""Custom transformer block matching the reference JetFormer architecture (arXiv:2601.17215).

Differs from a standard nn.TransformerEncoderLayer in three ways: bias-free Q/K/V
projections, a pre-activation FFN (norm -> activation -> linear, not the usual
linear -> activation), and a learnable normalization layer over the full
(num_tokens x num_tokens) attention-score matrix that is registered (so it
contributes to the parameter count) but never applied in forward — matching the
reference implementation, where this layer is defined but commented out.
"""

import torch
import torch.nn as nn


class SeqNorm(nn.Module):
    """BatchNorm1d or LayerNorm over the channel dim of a (B, N, C) sequence."""

    def __init__(self, dim: int, normalization: str = "batch") -> None:
        super().__init__()
        self.normalization = normalization
        self.norm = nn.BatchNorm1d(dim) if normalization == "batch" else nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalization == "batch":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class JetFormerAttention(nn.Module):
    """Pre-norm multi-head self-attention, no padding mask, with a dead score-norm layer."""

    def __init__(
        self, dim: int, num_heads: int, num_tokens: int, normalization: str = "batch"
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = SeqNorm(dim, normalization)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=True)

        # Registered but unused in forward — kept only so the parameter count
        # matches the reference (it has the same dead layer).
        score_dim = num_tokens * num_tokens
        self.pre_exp_norm = (
            nn.BatchNorm1d(score_dim) if normalization == "batch" else nn.LayerNorm(score_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.norm(x)

        q = self.q(h).view(B, N, self.num_heads, self.head_dim)
        k = self.k(h).view(B, N, self.num_heads, self.head_dim)
        v = self.v(h).view(B, N, self.num_heads, self.head_dim)

        scores = torch.einsum("bqhc,bkhc->bhqk", q, k) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bkhc->bqhc", attn, v).reshape(B, N, -1)
        return x + self.out(out)


class JetFormerBlock(nn.Module):
    """Self-attention sub-layer followed by a x2-expansion pre-activation FFN, both residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_tokens: int,
        dropout: float = 0.0,
        activation: str = "relu",
        normalization: str = "batch",
    ) -> None:
        super().__init__()
        self.attn = JetFormerAttention(dim, num_heads, num_tokens, normalization)

        act_cls = nn.ReLU if activation == "relu" else nn.SiLU
        self.norm_0 = SeqNorm(dim, normalization)
        self.activ_0 = act_cls()
        self.linear_0 = nn.Linear(dim, dim * 2, bias=False)
        self.norm_1 = SeqNorm(dim * 2, normalization)
        self.activ_1 = act_cls()
        self.linear_1 = nn.Linear(dim * 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        h = self.linear_0(self.activ_0(self.norm_0(x)))
        h = self.linear_1(self.activ_1(self.norm_1(h)))
        return x + self.dropout(h)
