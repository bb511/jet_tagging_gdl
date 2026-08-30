# Symmetry-consistency loss: the penalty on how far a classifier's logits move under a
# transformation of its input that should leave the label unchanged.
"""Symmetry-consistency loss (SCL).

The training objective is

.. math:: L = L_\\mathrm{cls} + w \\, d\\bigl(f(x), f(T x)\\bigr),

with :math:`T` drawn independently per jet from one of SO(2), SO(3) or the Lorentz
boosts, and :math:`d` one of the distances in :func:`symmetry_distance`. The transforms
themselves come from :mod:`src.data.components.augmentation`, which is the single source
of truth shared with the online augmentation and the invariance probe.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.data.components.augmentation import make_transform

SUPPORTED_GROUPS = ("so2", "so3", "lorentz")
SUPPORTED_METRICS = ("l1", "l2", "mse", "cosine", "js", "kl")


def validate_symmetry_group(method: str, group: str | None) -> None:
    """Reject an SCL run whose symmetry group names no transform.

    ``algorithm.group`` is interpolated from ``data.augmentation_scheme``, whose domain also holds
    ``none``. That value is legitimate for an augmentation run, where the group is never read, so
    it is rejected only under ``method="symm_loss"``, and at construction time rather than at the
    first training step.

    :param method: either ``augmentation`` or ``symm_loss``.
    :param group: the symmetry group, or ``None``.
    :raises ValueError: under ``symm_loss`` with a group outside :data:`SUPPORTED_GROUPS`.
    """
    if method != "symm_loss":
        return
    if group is None or str(group).lower() not in SUPPORTED_GROUPS:
        raise ValueError(
            f"method='symm_loss' requires a group in {list(SUPPORTED_GROUPS)}, got "
            f"{group!r}. The group follows data.augmentation_scheme, so set that "
            "(e.g. data.augmentation_scheme=so3) or override algorithm.group."
        )


def build_symmetric_batch(
    x: torch.Tensor,
    group: str,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
    *,
    lorentz_eta_max: float = 2.0,
    so3_apply_filter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw one transformation per jet and return the transformed partner batch.

    :param x: batch of shape (jets, constituents, features).
    :param group: one of :data:`SUPPORTED_GROUPS`.
    :param feature_names: column names of *x*, in order.
    :param norm_scales: per-column factors returning the stored values to GeV.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts.
    :param so3_apply_filter: drop jets rotated outside detector acceptance.
    :return: the pair ``(x_sym, keep_mask)``. Under filtered SO(3), ``x_sym`` holds fewer jets than
        *x*, and ``keep_mask``, of the input batch length, selects the rows of any batch-aligned
        tensor that survive; it is ``None`` for the groups that drop none.
    """
    transform = make_transform(
        group, feature_names, norm_scales, lorentz_eta_max, so3_apply_filter
    )
    return transform(x)


def symmetry_distance(
    logits: torch.Tensor,
    logits_sym: torch.Tensor,
    metric: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean per-jet discrepancy between the outputs on ``x`` and on ``T(x)``.

    ``l1``, ``l2`` and ``mse`` act on the logits directly; ``cosine`` on their direction; ``js``
    and ``kl`` on the softmax probabilities, symmetrised, with *eps* keeping the logarithms finite
    where a probability underflows.

    :param metric: one of :data:`SUPPORTED_METRICS`.
    :raises ValueError: on a shape mismatch or an unknown metric.
    """
    if logits.shape != logits_sym.shape:
        raise ValueError(
            "logits and logits_sym must have identical shape, got "
            f"{tuple(logits.shape)} vs {tuple(logits_sym.shape)}."
        )

    m = metric.lower()
    if m == "l1":
        return (logits - logits_sym).abs().sum(dim=1).mean()
    if m == "l2":
        return (logits - logits_sym).norm(dim=1).mean()
    if m == "mse":
        return F.mse_loss(logits, logits_sym)
    if m == "cosine":
        return (1.0 - F.cosine_similarity(logits, logits_sym, dim=1)).mean()
    if m in ("js", "kl"):
        return _divergence(logits, logits_sym, m, eps)

    raise ValueError(f"metric must be one of {list(SUPPORTED_METRICS)}, got {metric!r}.")


def _divergence(
    logits: torch.Tensor, logits_sym: torch.Tensor, metric: str, eps: float
) -> torch.Tensor:
    """Symmetrised Jensen-Shannon or Kullback-Leibler divergence of the two softmaxes."""
    p = F.softmax(logits, dim=-1)
    q = F.softmax(logits_sym, dim=-1)
    if metric == "js":
        mix = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.add(eps).log() - mix.add(eps).log())).sum(dim=1)
            + (q * (q.add(eps).log() - mix.add(eps).log())).sum(dim=1)
        )
        return js.clamp_min(0.0).mean()

    kl_pq = F.kl_div(p.add(eps).log(), q, reduction="none").sum(dim=1)
    kl_qp = F.kl_div(q.add(eps).log(), p, reduction="none").sum(dim=1)
    return 0.5 * (kl_pq + kl_qp).mean()


class SymmetryConsistencyLoss(nn.Module):
    """Add a weighted symmetry penalty to a classification loss."""

    def __init__(self, weight: float = 0.1, metric: str = "l2", eps: float = 1e-8) -> None:
        """
        :param weight: the coefficient :math:`w` multiplying the symmetry term.
        :param metric: one of :data:`SUPPORTED_METRICS`.
        :param eps: floor on the probabilities of the divergence metrics.
        :raises ValueError: on an unknown metric.
        """
        super().__init__()
        if metric.lower() not in SUPPORTED_METRICS:
            raise ValueError(f"metric must be one of {list(SUPPORTED_METRICS)}, got {metric!r}.")
        self.weight = float(weight)
        self.metric = metric.lower()
        self.eps = eps

    def forward(
        self, cls_loss: torch.Tensor, logits: torch.Tensor, logits_sym: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(total_loss, symmetry_term)``.

        *logits* and *logits_sym* must already describe the same jets: under filtered SO(3) the
        caller subsets *logits* with the mask from :func:`build_symmetric_batch`.
        """
        sym = symmetry_distance(logits, logits_sym, self.metric, self.eps)
        return cls_loss + self.weight * sym, sym
