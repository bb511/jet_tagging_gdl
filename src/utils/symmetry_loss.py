"""Helpers for symmetry-regularized training.

This module provides a minimal API for symmetry loss workflows:

1. Build transformed partner batches (SO(2), SO(3), Lorentz).
2. Measure output discrepancy between original and transformed batches.
3. Combine classification and symmetry losses.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.augmentation import (
    apply_boost_preserve_phi,
    apply_so2_single_rotation,
    apply_so3_single_rotation,
    create_lorentz_boosts,
    create_so3_rot_matrices,
)


GeneratorLike = Union[np.random.Generator, torch.Generator]


def _as_numpy_rng(generator: Optional[GeneratorLike] = None) -> np.random.Generator:
    """Normalise RNG input to a numpy Generator."""
    if generator is None:
        return np.random.default_rng()
    if isinstance(generator, np.random.Generator):
        return generator
    if isinstance(generator, torch.Generator):
        seed = int(
            torch.randint(
                low=0,
                high=2**31 - 1,
                size=(1,),
                generator=generator,
            ).item()
        )
        return np.random.default_rng(seed)
    raise TypeError(
        "generator must be None, np.random.Generator or torch.Generator, "
        f"got {type(generator)}."
    )


def _to_numpy_batch(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy()


def _from_numpy_batch(x_np: np.ndarray, ref: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(x_np).to(device=ref.device, dtype=ref.dtype)


def _require_feature_names(feature_names: Optional[Sequence[str]]) -> list[str]:
    if feature_names is None:
        raise ValueError(
            "feature_names must be provided to use augmentation.py-exact transforms."
        )
    return list(feature_names)


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    per_sample: bool = True,
    generator: Optional[GeneratorLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(2) rotations using the exact augmentation backend.

    Transformation logic is delegated to
    ``src.utils.augmentation.apply_so2_single_rotation``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    # Kept for API compatibility; exact transformation uses feature_names.
    _ = (px_index, py_index)
    names = _require_feature_names(feature_names)
    rng = _as_numpy_rng(generator)
    x_np = _to_numpy_batch(x)

    batch_size = x_np.shape[0]
    if per_sample:
        theta = rng.uniform(min_angle, max_angle, size=batch_size).astype(np.float32)
        x_rot_np = np.empty_like(x_np)
        for i, t in enumerate(theta):
            c, s = np.cos(t), np.sin(t)
            rot = np.array(
                [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            x_rot_np[i : i + 1] = apply_so2_single_rotation(
                x_np[i : i + 1], rot, names
            )
    else:
        theta = np.array([rng.uniform(min_angle, max_angle)], dtype=np.float32)
        c, s = np.cos(theta[0]), np.sin(theta[0])
        rot = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        x_rot_np = apply_so2_single_rotation(x_np, rot, names)

    x_rot = _from_numpy_batch(x_rot_np, ref=x)
    theta_t = torch.from_numpy(theta).to(device=x.device, dtype=x.dtype)
    return x_rot, theta_t


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    per_sample: bool = True,
    generator: Optional[GeneratorLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(3) rotations using the exact augmentation backend.

    Transformation logic is delegated to
    ``src.utils.augmentation.apply_so3_single_rotation``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    # Kept for API compatibility; exact transformation uses feature_names.
    _ = (px_index, py_index, pz_index)
    names = _require_feature_names(feature_names)
    rng = _as_numpy_rng(generator)
    x_np = _to_numpy_batch(x)

    num_mats = x_np.shape[0] if per_sample else 1
    rot_mats_np = create_so3_rot_matrices(num_mats, rng)

    if per_sample:
        x_rot_np = np.empty_like(x_np)
        for i, rot in enumerate(rot_mats_np):
            x_rot_np[i : i + 1] = apply_so3_single_rotation(
                x_np[i : i + 1], rot, names
            )
    else:
        x_rot_np = apply_so3_single_rotation(x_np, rot_mats_np[0], names)

    x_rot = _from_numpy_batch(x_rot_np, ref=x)
    rot_mats = torch.from_numpy(rot_mats_np).to(device=x.device, dtype=x.dtype)
    return x_rot, rot_mats


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    e_index: Optional[int] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    eta_max: float = 2.0,
    per_sample: bool = True,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[GeneratorLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random Lorentz boosts using the exact augmentation backend.

    Transformation logic is delegated to
    ``src.utils.augmentation.apply_boost_preserve_phi`` and therefore expects
    the same required feature set used by ``apply_augmentation(..., 'lorentz')``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    # Kept for API compatibility; exact Lorentz backend ignores these knobs.
    _ = (e_index, px_index, py_index, pz_index, preserve_jet_phi, require_positive_energy)
    names = _require_feature_names(feature_names)
    rng = _as_numpy_rng(generator)
    x_np = _to_numpy_batch(x)

    required = {
        "j1_px",
        "j1_py",
        "j1_pz",
        "j1_e",
        "j1_pt",
        "j1_eta",
        "j1_phi",
        "j1_etarel",
        "j1_phirel",
        "j1_deltaR",
        "j1_ptrel",
    }
    missing = [name for name in required if name not in names]
    if missing:
        print(
            "Lorentz augmentation requested but missing required "
            f"features: {missing}. Returning unaugmented data."
        )
        empty = torch.empty((0, 4, 4), device=x.device, dtype=x.dtype)
        return x.clone(), empty

    num_mats = x_np.shape[0] if per_sample else 1
    boosts_np = create_lorentz_boosts(num_mats, eta_max, rng)

    if per_sample:
        x_rot_np = np.empty_like(x_np)
        for i, l_mat in enumerate(boosts_np):
            x_rot_np[i : i + 1] = apply_boost_preserve_phi(
                x_np[i : i + 1], l_mat, names
            )
    else:
        x_rot_np = apply_boost_preserve_phi(x_np, boosts_np[0], names)

    x_rot = _from_numpy_batch(x_rot_np, ref=x)
    boosts = torch.from_numpy(boosts_np).to(device=x.device, dtype=x.dtype)
    return x_rot, boosts


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
    per_sample: bool = True,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[GeneratorLike] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Create transformed partners ``T(x)`` for symmetry regularization.

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
            per_sample=per_sample,
            generator=generator,
        )

    if g == "so3":
        return apply_random_so3(
            x=x,
            feature_names=feature_names,
            px_index=px_index,
            py_index=py_index,
            pz_index=pz_index,
            per_sample=per_sample,
            generator=generator,
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
            per_sample=per_sample,
            preserve_jet_phi=preserve_jet_phi,
            require_positive_energy=require_positive_energy,
            generator=generator,
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
        - ``l2``: mean normalised L2 on logits
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

    if m == "l2":
        diff = (logits - logits_sym).norm(dim=1)
        base = logits.norm(dim=1).clamp_min(eps)
        return (diff / base).mean()

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
        "Choose one of ['l2', 'mse', 'cosine', 'js', 'kl']."
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
