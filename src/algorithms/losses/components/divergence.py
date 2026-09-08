"""Divergences between the class distributions on ``x`` and on ``T(x)``, for the SCL."""

import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxDivergence(nn.Module):
    """Symmetrised divergence between the softmax logit vectors, mean over the jets.

    ``js`` is the Jensen-Shannon divergence and ``kl`` the mean of the two
    Kullback-Leibler divergences.
    """

    METRICS = ("js", "kl")

    def __init__(self, metric: str = "js", eps: float = 1e-8) -> None:
        super().__init__()
        if metric not in self.METRICS:
            raise ValueError(f"metric must be one of {list(self.METRICS)}, got {metric!r}.")
        self.metric = metric
        self.eps = eps

    def forward(self, logits: torch.Tensor, logits_sym: torch.Tensor) -> torch.Tensor:
        """Return the scalar divergence between two (jets, classes) tensors."""
        p, q = F.softmax(logits, dim=-1), F.softmax(logits_sym, dim=-1)
        if self.metric == "kl":
            kl_pq = F.kl_div(p.add(self.eps).log(), q, reduction="none").sum(dim=1)
            kl_qp = F.kl_div(q.add(self.eps).log(), p, reduction="none").sum(dim=1)
            return 0.5 * (kl_pq + kl_qp).mean()
        mix = 0.5 * (p + q)
        js = 0.5 * (self._relative_entropy(p, mix) + self._relative_entropy(q, mix))
        return js.clamp_min(0.0).mean()

    def _relative_entropy(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Per-jet :math:`\\sum_c p_c \\log(p_c / q_c)`."""
        return (p * (p.add(self.eps).log() - q.add(self.eps).log())).sum(dim=1)
