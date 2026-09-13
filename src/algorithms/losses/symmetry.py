"""Symmetry-consistency loss (SCL).

The training objective is

.. math:: L = L_\\mathrm{cls} + w \\, d\\bigl(f(x), f(T x)\\bigr),

with `T` drawn independently per jet from one of `src.symmetry.GROUPS`, and `d` a distance
between the logits (``components/distance.py``) or a divergence between the class
distributions (``components/divergence.py``).
The transforms come from `src.data.components.augmentation`.
"""

from collections.abc import Sequence

import torch
from torch import nn

from src.algorithms.losses.components.distance import LogitDistance
from src.algorithms.losses.components.divergence import SoftmaxDivergence
from src.data.components.augmentation import make_transform
from src.symmetry import GROUPS

SUPPORTED_METRICS = LogitDistance.METRICS + SoftmaxDivergence.METRICS


class SymmetryConsistencyLoss(nn.Module):
    """Add a weighted symmetry penalty to a classification loss.

    :param group: the group `T` is drawn from, one of `src.symmetry.GROUPS`. It follows
        ``data.augmentation_scheme``.
    :param weight: the coefficient `w` multiplying the symmetry term.
    :param metric: one of `SUPPORTED_METRICS`.
    :param eps: floor on the probabilities of the divergence metrics.
    """

    def __init__(
        self,
        group: str | None,
        weight: float = 0.1,
        metric: str = "l2",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if group is None or str(group).lower() not in GROUPS:
            raise ValueError(
                f"method=symm_loss needs a group in {list(GROUPS)}, got {group!r}. The group "
                "follows data.augmentation_scheme, so set that (e.g. "
                "data.augmentation_scheme=so3) or override algorithm.group."
            )
        self.group = str(group).lower()
        self.weight = float(weight)
        metric = metric.lower()
        if metric in SoftmaxDivergence.METRICS:
            self.distance = SoftmaxDivergence(metric, eps)
        elif metric in LogitDistance.METRICS:
            self.distance = LogitDistance(metric)
        else:
            raise ValueError(
                f"metric must be one of {list(SUPPORTED_METRICS)}, got {metric!r}."
            )

    def transform(
        self,
        x: torch.Tensor,
        feature_names: Sequence[str],
        norm_scales: dict[str, float] | None = None,
        **transform_kwargs: float,
    ) -> torch.Tensor:
        """Draw one transformation of the group per jet and return ``T(x)``.

        :param x: batch of shape (jets, constituents, features), whose columns
            *feature_names* lists in order.
        :param norm_scales: per-column factors returning the stored values to GeV.
        :param transform_kwargs: additional args for ``make_transform``.
        :return: the transformed batch, of the shape and normalisation of *x*.
        """
        return make_transform(
            self.group, feature_names, norm_scales, **transform_kwargs
        )(x)

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
        sym = self.distance(logits, logits_sym)
        return cls_loss + self.weight * sym, sym
