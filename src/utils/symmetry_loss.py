"""Helpers for symmetry-regularized training.

This module provides a minimal API for symmetry loss workflows:

1. Build transformed partner batches (SO(2), SO(3), Lorentz) — delegated to
   ``src.utils.augmentation``, the single source of truth for transform code.
2. Measure output discrepancy between original and transformed batches.
3. Combine classification and symmetry losses.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from src.utils.augmentation import (
    GeneratorLike,
    apply_random_lorentz,
    apply_random_so2,
    apply_random_so3,
)

# ---------------------------------------------------------------------------
# Lightweight diagnostic timing — accumulated across calls, reset on demand.
# Keys: "to_cpu_ms", "loop_ms", "to_gpu_ms"
# ---------------------------------------------------------------------------
_TIMING: dict[str, float] = {"to_cpu_ms": 0.0, "loop_ms": 0.0, "to_gpu_ms": 0.0}


def get_and_reset_timings() -> dict[str, float]:
    """Return accumulated transform timings (ms) and reset counters to zero."""
    global _TIMING
    snapshot = dict(_TIMING)
    _TIMING = {"to_cpu_ms": 0.0, "loop_ms": 0.0, "to_gpu_ms": 0.0}
    return snapshot


def build_symmetric_batch(
    x: torch.Tensor,
    group: str = "so2",
    feature_names: Optional[Sequence[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    e_index: Optional[int] = None,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    eta_max: float = 2.0,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[GeneratorLike] = None,
    norm_scales: Optional[dict] = None,
    apply_filter: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Create transformed partners ``T(x)`` for symmetry regularization.

    Each jet receives an independently sampled transformation.

    Returns:
        ``(x_transformed, transform_meta)``, where ``transform_meta`` is:
        - SO(2): sampled angle tensor
        - SO(3): rotation matrices
        - Lorentz: boost matrices
    """
    g = group.lower()
    if g == "so2":
        return apply_random_so2(
            x=x,
            feature_names=feature_names,
            px_index=px_index,
            py_index=py_index,
            min_angle=min_angle,
            max_angle=max_angle,
            generator=generator,
            norm_scales=norm_scales,
        )

    if g == "so3":
        return apply_random_so3(
            x=x,
            feature_names=feature_names,
            px_index=px_index,
            py_index=py_index,
            pz_index=pz_index,
            generator=generator,
            norm_scales=norm_scales,
            apply_filter=apply_filter,
        )

    if g == "lorentz":
        return apply_random_lorentz(
            x=x,
            feature_names=feature_names,
            e_index=e_index,
            px_index=px_index,
            py_index=py_index,
            pz_index=pz_index,
            eta_max=eta_max,
            preserve_jet_phi=preserve_jet_phi,
            require_positive_energy=require_positive_energy,
            generator=generator,
            norm_scales=norm_scales,
        )

    raise NotImplementedError(
        f"Group '{group}' is not implemented. Choose one of ['so2', 'so3', 'lorentz']."
    )


def symmetry_distance(
    logits: torch.Tensor,
    logits_sym: torch.Tensor,
    metric: str = "l2",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute discrepancy between model outputs on ``x`` and ``T(x)``.

    Supported metrics:
        - ``l1``: mean L1 norm on logits (sum of absolute differences per sample)
        - ``l2``: mean L2 norm on logits
        - ``mse``: mean squared error on logits
        - ``cosine``: mean cosine distance
        - ``js``: mean Jensen-Shannon divergence
        - ``kl``: mean symmetric KL divergence
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

    p = F.softmax(logits, dim=-1)
    q = F.softmax(logits_sym, dim=-1)

    if m == "js":
        m_mix = 0.5 * (p + q)
        js = 0.5 * (
            (p * (p.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
            + (q * (q.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
        )
        return js.clamp_min(0.0).mean()

    if m == "kl":
        log_p = p.add(eps).log()
        log_q = q.add(eps).log()
        kl_pq = F.kl_div(log_p, q, reduction="none").sum(dim=1)
        kl_qp = F.kl_div(log_q, p, reduction="none").sum(dim=1)
        return 0.5 * (kl_pq + kl_qp).mean()

    raise ValueError(
        f"Unknown symmetry metric '{metric}'. "
        "Choose one of ['l1', 'l2', 'mse', 'cosine', 'js', 'kl']."
    )


def combine_with_symmetry_loss(
    cls_loss: torch.Tensor,
    logits: torch.Tensor,
    logits_sym: torch.Tensor,
    weight: float,
    metric: str = "l2",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total_loss, sym_loss)`` for ``L = L_cls + weight * L_sym``."""
    sym_loss = symmetry_distance(logits, logits_sym, metric=metric, eps=eps)
    total_loss = cls_loss + float(weight) * sym_loss
    return total_loss, sym_loss
