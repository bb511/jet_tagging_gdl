"""Symmetry-consistency loss (SCL).

The training objective is the classification loss plus w times d(f(x), f(T x)), with `T`
drawn independently per jet from one of `src.symmetry.GROUPS`, and `d` the L2 distance
between the logits, averaged over the jets. The transforms come from
`src.data.components.augmentation`.
"""

import torch
from torch import nn

from src.symmetry import GROUPS


class SymmetryConsistencyLoss(nn.Module):
    """Add a weighted symmetry penalty to a classification loss.

    :param group: the group `T` is drawn from, one of `src.symmetry.GROUPS`. It follows
        ``data.augmentation_scheme``.
    :param weight: the coefficient `w` multiplying the symmetry term.
    """

    def __init__(self, group: str, weight: float = 0.1) -> None:
        super().__init__()
        if group not in GROUPS:
            raise ValueError(
                f"method=symm_loss needs a group in {list(GROUPS)}, got {group!r}. The group "
                "follows data.augmentation_scheme, so set that (e.g. "
                "data.augmentation_scheme=so3) or override algorithm.group."
            )
        self.group = group
        self.weight = float(weight)

    def forward(
        self, cls_loss: torch.Tensor, logits: torch.Tensor, logits_sym: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, symmetry_term)``.

        *logits* and *logits_sym* describe the same jets in the same order.
        """
        if logits.shape != logits_sym.shape:
            raise ValueError(
                "logits and logits_sym must have identical shape, got "
                f"{tuple(logits.shape)} vs {tuple(logits_sym.shape)}."
            )
        sym = (logits - logits_sym).norm(dim=1).mean()
        return cls_loss + self.weight * sym, sym
