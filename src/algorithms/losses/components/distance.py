"""Distances between the logits on ``x`` and on ``T(x)``, for the symmetry-consistency loss."""

import torch
import torch.nn.functional as F
from torch import nn


class LogitDistance(nn.Module):
    """Mean per-jet distance between two logit vectors.

    ``l1`` and ``l2`` are the norms of the difference, ``mse`` its mean square per element,
    and ``cosine`` one minus the cosine of the angle between the two vectors.
    """

    METRICS = ("l1", "l2", "mse", "cosine")

    def __init__(self, metric: str = "l2") -> None:
        super().__init__()
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {list(self.METRICS)}, got {metric!r}.")
        self.metric = metric

    def forward(self, logits: torch.Tensor, logits_sym: torch.Tensor) -> torch.Tensor:
        """Return the scalar distance between two (jets, classes) tensors."""
        if self.metric == "l1":
            return (logits - logits_sym).abs().sum(dim=1).mean()
        if self.metric == "l2":
            return (logits - logits_sym).norm(dim=1).mean()
        if self.metric == "mse":
            return F.mse_loss(logits, logits_sym)
        return (1.0 - F.cosine_similarity(logits, logits_sym, dim=1)).mean()
