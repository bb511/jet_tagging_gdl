"""Symmetry-consistency loss (SCL).

The training objective is

.. math:: L = L_\\mathrm{cls} + w \\, d\\bigl(f(x), f(T x)\\bigr),

with :math:`T` drawn independently per jet from one of SO(2), SO(3), the boosts along the beam
(``lorentz``) or the boosts of isotropic direction (``lorentz_iso``), and :math:`d` one of the
distances in :func:`symmetry_distance`. Only the beam boost is a symmetry of a pp collision. The
transforms themselves come from :mod:`src.data.components.augmentation`, which is the single
source of truth shared with the online augmentation and the invariance probe.
"""

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.data.components.augmentation import make_transform

SUPPORTED_GROUPS = ("so2", "so3", "lorentz", "lorentz_iso")


def validate_symmetry_group(method: str, group: str | None) -> None:
    """Reject an SCL run whose symmetry group names no transform.

    ``algorithm.group`` is interpolated from ``data.augmentation_scheme``, whose domain also holds
    ``none``; that value is legitimate for an augmentation run, where the group is never read, so
    it is rejected only under ``method="symm_loss"``, and at construction rather than at the first
    training step.
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
    lorentz_eta_max: float = 1.0,
    so3_apply_filter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Draw one transformation per jet and return the transformed partner batch.

    :param x: batch of shape (jets, constituents, features), whose columns *feature_names* lists in
        order.
    :param norm_scales: per-column factors returning the stored values to GeV.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts, along the beam
        for ``lorentz``, in a random direction for ``lorentz_iso``.
    :param so3_apply_filter: drop jets rotated outside detector acceptance; off by default, so
        SO(3) is learned under the Haar measure.
    :return: the pair ``(x_sym, keep_mask)``. Under filtered SO(3), ``x_sym`` holds fewer jets than
        *x*, and ``keep_mask``, of the input batch length, selects the rows of any batch-aligned
        tensor that survive; it is ``None`` whenever no jet is dropped.
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

    ``l1``, ``l2`` and ``mse`` act on the logits directly, ``cosine`` on their direction, and
    ``js`` and ``kl`` on the softmax probabilities, symmetrised, with *eps* keeping the logarithms
    finite where a probability underflows.
    """
    if logits.shape != logits_sym.shape:
        raise ValueError(
            "logits and logits_sym must have identical shape, got "
            f"{tuple(logits.shape)} vs {tuple(logits_sym.shape)}."
        )

    distance = _METRICS.get(metric.lower())
    if distance is None:
        raise ValueError(f"metric must be one of {list(SUPPORTED_METRICS)}, got {metric!r}.")
    return distance(logits, logits_sym, eps)


def _relative_entropy(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-jet :math:`\\sum_c p_c \\log(p_c / q_c)`, with *eps* keeping both logarithms finite."""
    return (p * (p.add(eps).log() - q.add(eps).log())).sum(dim=1)


def _jensen_shannon(logits: torch.Tensor, logits_sym: torch.Tensor, eps: float) -> torch.Tensor:
    """Jensen-Shannon divergence between the two softmaxes.

    Clamped at zero, which the sum of two relative entropies can undercut by a rounding once *eps*
    has shifted the probabilities.
    """
    p = F.softmax(logits, dim=-1)
    q = F.softmax(logits_sym, dim=-1)
    mix = 0.5 * (p + q)
    js = 0.5 * (_relative_entropy(p, mix, eps) + _relative_entropy(q, mix, eps))
    return js.clamp_min(0.0).mean()


def _symmetrised_kl(logits: torch.Tensor, logits_sym: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean of the two Kullback-Leibler divergences between the two softmaxes."""
    p = F.softmax(logits, dim=-1)
    q = F.softmax(logits_sym, dim=-1)
    kl_pq = F.kl_div(p.add(eps).log(), q, reduction="none").sum(dim=1)
    kl_qp = F.kl_div(q.add(eps).log(), p, reduction="none").sum(dim=1)
    return 0.5 * (kl_pq + kl_qp).mean()


# Every distance takes (logits, logits_sym, eps) and returns a scalar, whether or not it has a
# logarithm to keep finite, so that symmetry_distance can dispatch on the name alone.
_METRICS: dict[str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = {
    "l1": lambda a, b, eps: (a - b).abs().sum(dim=1).mean(),
    "l2": lambda a, b, eps: (a - b).norm(dim=1).mean(),
    "mse": lambda a, b, eps: F.mse_loss(a, b),
    "cosine": lambda a, b, eps: (1.0 - F.cosine_similarity(a, b, dim=1)).mean(),
    "js": _jensen_shannon,
    "kl": _symmetrised_kl,
}
SUPPORTED_METRICS = tuple(_METRICS)


class SymmetryConsistencyLoss(nn.Module):
    """Add a weighted symmetry penalty to a classification loss."""

    def __init__(self, weight: float = 0.1, metric: str = "l2", eps: float = 1e-8) -> None:
        """
        :param weight: the coefficient :math:`w` multiplying the symmetry term.
        :param metric: one of :data:`SUPPORTED_METRICS`.
        :param eps: floor on the probabilities of the divergence metrics.
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
