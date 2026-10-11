import torch
from torch import nn


def _norm_layer(dim: int, normalization: str) -> nn.Module:
    return nn.BatchNorm1d(dim) if normalization == "batch" else nn.LayerNorm(dim)


class SeqNorm(nn.Module):
    """BatchNorm1d or LayerNorm over the channel axis of a (jets, tokens, channels)."""

    def __init__(self, dim: int, normalization: str = "batch") -> None:
        super().__init__()
        self.normalization = normalization
        self.norm = _norm_layer(dim, normalization)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise over the channel axis, transposing for BatchNorm1d."""
        if self.normalization == "batch":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)
