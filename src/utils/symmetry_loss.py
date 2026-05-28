"""Helpers for symmetry-regularized training.

This module provides a minimal API for symmetry loss workflows:

1. Build transformed partner batches (SO(2), SO(3), Lorentz).
2. Measure output discrepancy between original and transformed batches.
3. Combine classification and symmetry losses.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

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

from src.utils.augmentation import (
    apply_boost_preserve_phi,
    apply_so2_single_rotation,
    apply_so3_single_rotation,
    compute_eta,
    compute_pt,
    compute_phi,
    create_lorentz_boosts,
    create_so2_rot_matrices,
    create_so3_rot_matrices,
    feature_index,
    get_norm_scale,
    sample_haar_so3_rot_matrices,
    so3_jet_filter,
    wrap_phi,
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


# ── GPU-vectorized batched transform helpers ──────────────────────────────────
#
# These replace the per-sample Python loops used by the legacy symm_loss path.
# Each helper applies a *different* transformation matrix to every jet using a
# single batched einsum:  "bij,bcj->bci"  (B jets, C constituents, K components)
# — zero Python loops, all computation on whatever device x lives on.


def _apply_batched_so2(
    x: torch.Tensor,
    mats: torch.Tensor,
    names: list,
    norm_scales,
) -> torch.Tensor:
    """Apply a different SO(2) rotation to each jet.  mats: (B, 3, 3), x: (B, C, F)."""
    px_idx  = feature_index(names, "j1_px",  "px")
    py_idx  = feature_index(names, "j1_py",  "py")
    pz_idx  = feature_index(names, "j1_pz",  "pz")
    pt_idx  = feature_index(names, "j1_pt",  "pt")
    phi_idx = feature_index(names, "j1_phi", "phi")
    if None in (px_idx, py_idx, pz_idx):
        return x.clone()

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    vec = torch.stack([
        x[:, :, px_idx] * s_px,
        x[:, :, py_idx] * s_py,
        x[:, :, pz_idx] * s_pz,
    ], dim=-1)                                          # (B, C, 3)
    vec_rot = torch.einsum("bij,bcj->bci", mats, vec)  # (B, C, 3)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]
    pt_r  = compute_pt(px_r, py_r)
    phi_r = compute_phi(px_r, py_r)

    n_feats = x.shape[2]
    x_new = x.clone()
    for idx, arr, feat in zip(
        [px_idx, py_idx, pz_idx, pt_idx, phi_idx],
        [px_r,   py_r,   pz_r,   pt_r,   phi_r],
        ["px",   "py",   "pz",   "pt",   "phi"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = arr / get_norm_scale(norm_scales, feat)
    return x_new


def _apply_random_so2_gpu(
    x: torch.Tensor,
    names: list,
    min_angle: float,
    max_angle: float,
    norm_scales,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample B independent SO(2) angles on GPU and apply them in one einsum."""
    B = x.shape[0]
    theta = torch.rand(B, device=x.device, dtype=x.dtype) * (max_angle - min_angle) + min_angle
    mats = torch.zeros(B, 3, 3, device=x.device, dtype=x.dtype)
    mats[:, 0, 0] =  torch.cos(theta)
    mats[:, 0, 1] = -torch.sin(theta)
    mats[:, 1, 0] =  torch.sin(theta)
    mats[:, 1, 1] =  torch.cos(theta)
    mats[:, 2, 2] = 1.0
    return _apply_batched_so2(x, mats, names, norm_scales), theta


def _apply_batched_so3(
    x: torch.Tensor,
    mats: torch.Tensor,
    names: list,
    norm_scales,
    apply_filter: bool,
) -> torch.Tensor:
    """Apply a different SO(3) rotation to each jet.  mats: (B, 3, 3), x: (B, C, F)."""
    px_idx      = feature_index(names, "j1_px",     "px")
    py_idx      = feature_index(names, "j1_py",     "py")
    pz_idx      = feature_index(names, "j1_pz",     "pz")
    pt_idx      = feature_index(names, "j1_pt",     "pt")
    eta_idx     = feature_index(names, "j1_eta",    "eta")
    phi_idx     = feature_index(names, "j1_phi",    "phi")
    etarel_idx  = feature_index(names, "j1_etarel", "etarel")
    phirel_idx  = feature_index(names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(names, "j1_deltaR", "deltaR")
    ptrel_idx   = feature_index(names, "j1_ptrel",  "ptrel")
    if None in (px_idx, py_idx, pz_idx):
        return x.clone()

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")
    n_feats = x.shape[2]

    px_phys = x[:, :, px_idx] * s_px
    py_phys = x[:, :, py_idx] * s_py
    pz_phys = x[:, :, pz_idx] * s_pz
    pad_mask = (px_phys ** 2 + py_phys ** 2) > 0   # (B, C)

    vec = torch.stack([px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    vec_rot = torch.einsum("bij,bcj->bci", mats, vec)  # (B, C, 3)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]

    pt_r  = compute_pt(px_r, py_r)
    eta_r = compute_eta(px_r, py_r, pz_r)
    phi_r = compute_phi(px_r, py_r)

    jet_px = (px_r * pad_mask).sum(dim=1, keepdim=True)
    jet_py = (py_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pz = (pz_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pt  = compute_pt(jet_px, jet_py)
    jet_eta = compute_eta(jet_px, jet_py, jet_pz)
    jet_phi = compute_phi(jet_px, jet_py)

    eta_rel = eta_r - jet_eta
    phi_rel = wrap_phi(phi_r - jet_phi)
    delta_r = torch.sqrt(eta_rel ** 2 + phi_rel ** 2)
    pt_rel  = pt_r / jet_pt.clamp(min=1e-8)

    x_new = x.clone()
    for idx, arr, feat in zip(
        [px_idx, py_idx, pz_idx, pt_idx, eta_idx, phi_idx,
         etarel_idx, phirel_idx, delta_r_idx, ptrel_idx],
        [px_r, py_r, pz_r, pt_r, eta_r, phi_r,
         eta_rel, phi_rel, delta_r, pt_rel],
        ["px", "py", "pz", "pt", "eta", "phi",
         "etarel", "phirel", "deltaR", "ptrel"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = torch.where(
                pad_mask,
                arr / get_norm_scale(norm_scales, feat),
                x[:, :, idx],
            )

    if apply_filter:
        ok = so3_jet_filter(x_new, names, norm_scales=norm_scales)  # (B,)
        failed = ~ok
        if failed.any():
            x_new[failed] = x[failed]
    return x_new


def _apply_random_so3_gpu(
    x: torch.Tensor,
    names: list,
    norm_scales,
    apply_filter: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample B Haar-random SO(3) matrices on GPU and apply them in one einsum."""
    B = x.shape[0]
    mats = sample_haar_so3_rot_matrices(B, device=x.device, dtype=x.dtype)
    return _apply_batched_so3(x, mats, names, norm_scales, apply_filter), mats


def _apply_batched_lorentz(
    x: torch.Tensor,
    boosts: torch.Tensor,
    names: list,
    norm_scales,
) -> torch.Tensor:
    """Apply a different Lorentz boost to each jet.  boosts: (B, 4, 4), x: (B, C, F)."""
    px_idx      = feature_index(names, "j1_px",     "px")
    py_idx      = feature_index(names, "j1_py",     "py")
    pz_idx      = feature_index(names, "j1_pz",     "pz")
    e_idx       = feature_index(names, "j1_e",      "e")
    pt_idx      = feature_index(names, "j1_pt",     "pt")
    eta_idx     = feature_index(names, "j1_eta",    "eta")
    phi_idx     = feature_index(names, "j1_phi",    "phi")
    etarel_idx  = feature_index(names, "j1_etarel", "etarel")
    phirel_idx  = feature_index(names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(names, "j1_deltaR", "deltaR")
    ptrel_idx   = feature_index(names, "j1_ptrel",  "ptrel")
    if None in (px_idx, py_idx, pz_idx):
        return x.clone()

    n_feats = x.shape[2]
    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    px_phys = x[:, :, px_idx] * s_px
    py_phys = x[:, :, py_idx] * s_py
    pz_phys = x[:, :, pz_idx] * s_pz
    pad_mask = (px_phys ** 2 + py_phys ** 2) > 0

    if e_idx is not None and e_idx < n_feats:
        e_in = x[:, :, e_idx] * get_norm_scale(norm_scales, "e")
    else:
        e_in = torch.sqrt(px_phys ** 2 + py_phys ** 2 + pz_phys ** 2)

    p4  = torch.stack([e_in, px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    p4b = torch.einsum("bij,bcj->bci", boosts, p4)  # (B, C, 4)

    e_b  = p4b[:, :, 0]
    px_b = p4b[:, :, 1]
    py_b = p4b[:, :, 2]
    pz_b = p4b[:, :, 3]

    # Compensating SO(2) rotation to restore original jet φ after boost
    jet_phi_orig  = torch.atan2(
        (py_phys * pad_mask).sum(dim=1, keepdim=True),
        (px_phys * pad_mask).sum(dim=1, keepdim=True),
    )
    jet_phi_boost = torch.atan2(
        (py_b * pad_mask).sum(dim=1, keepdim=True),
        (px_b * pad_mask).sum(dim=1, keepdim=True),
    )
    theta = jet_phi_orig - jet_phi_boost
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    px_r = (cos_t * px_b - sin_t * py_b) * pad_mask
    py_r = (sin_t * px_b + cos_t * py_b) * pad_mask
    pz_r = pz_b * pad_mask
    e_r  = e_b  * pad_mask
    energy_mask = (e_r > 0) & pad_mask

    pt  = compute_pt(px_r, py_r)
    phi = compute_phi(px_r, py_r)
    eta = compute_eta(px_r, py_r, pz_r)

    jet_px  = (px_r * pad_mask).sum(dim=1, keepdim=True)
    jet_py  = (py_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pz  = (pz_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pt  = compute_pt(jet_px, jet_py)
    jet_phi = compute_phi(jet_px, jet_py)
    jet_eta = compute_eta(jet_px, jet_py, jet_pz)

    eta_rel = eta - jet_eta
    phi_rel = wrap_phi(phi - jet_phi)
    delta_r = torch.sqrt(eta_rel ** 2 + phi_rel ** 2)
    pt_rel  = pt / jet_pt.clamp(min=1e-8)

    x_new = x.clone()
    for idx, arr, feat in zip(
        [px_idx, py_idx, pz_idx, e_idx, pt_idx, eta_idx, phi_idx,
         etarel_idx, phirel_idx, delta_r_idx, ptrel_idx],
        [px_r,   py_r,   pz_r,   e_r,   pt,     eta,     phi,
         eta_rel, phi_rel, delta_r, pt_rel],
        ["px", "py", "pz", "e", "pt", "eta", "phi",
         "etarel", "phirel", "deltaR", "ptrel"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = torch.where(
                energy_mask,
                arr / get_norm_scale(norm_scales, feat),
                x[:, :, idx],
            )
    return x_new


def _apply_random_lorentz_gpu(
    x: torch.Tensor,
    names: list,
    eta_max: float,
    norm_scales,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample B Lorentz boosts on GPU and apply them in one einsum — no CPU transfer."""
    B = x.shape[0]
    direction = torch.randn(B, 3, device=x.device, dtype=x.dtype)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)

    eta   = (torch.rand(B, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * eta_max
    beta  = torch.tanh(eta).unsqueeze(1) * direction       # (B, 3)
    beta2 = (beta ** 2).sum(dim=1)                         # (B,)
    gamma = 1.0 / torch.sqrt((1.0 - beta2).clamp(min=1e-8))
    gb    = (gamma - 1.0) / beta2.clamp(min=1e-12)

    L = torch.eye(4, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1).clone()
    L[:, 0, 0]  = gamma
    L[:, 0, 1:] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 0] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 1:] += gb.view(B, 1, 1) * torch.bmm(beta.unsqueeze(2), beta.unsqueeze(1))
    near_zero = beta2 < 1e-12
    if near_zero.any():
        L[near_zero] = torch.eye(4, dtype=x.dtype, device=x.device)

    return _apply_batched_lorentz(x, L, names, norm_scales), L


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    generator: Optional[GeneratorLike] = None,
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(2) rotations using the exact augmentation backend.

    Each jet in the batch receives an independently sampled rotation angle.
    Transformation logic is delegated to
    ``src.utils.augmentation.apply_so2_single_rotation``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    _ = (px_index, py_index)
    names = _require_feature_names(feature_names)

    # GPU path — fully vectorized, no CPU round-trip
    if x.device.type == "cuda":
        return _apply_random_so2_gpu(x, names, min_angle, max_angle, norm_scales)

    # ---- Legacy CPU/numpy path (kept for reference) ----
    # rng = _as_numpy_rng(generator)
    # _t0 = time.perf_counter()
    # x_np = _to_numpy_batch(x)
    # _TIMING["to_cpu_ms"] += (time.perf_counter() - _t0) * 1000
    # batch_size = x_np.shape[0]
    # theta = rng.uniform(min_angle, max_angle, size=batch_size).astype(np.float32)
    # x_rot_np = np.empty_like(x_np)
    # _t1 = time.perf_counter()
    # for i, t in enumerate(theta):
    #     c, s = np.cos(t), np.sin(t)
    #     rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    #     x_rot_np[i:i+1] = apply_so2_single_rotation(x_np[i:i+1], rot, names, norm_scales=norm_scales)
    # _TIMING["loop_ms"] += (time.perf_counter() - _t1) * 1000
    # _t2 = time.perf_counter()
    # x_rot = _from_numpy_batch(x_rot_np, ref=x)
    # _TIMING["to_gpu_ms"] += (time.perf_counter() - _t2) * 1000
    # theta_t = torch.from_numpy(theta).to(device=x.device, dtype=x.dtype)
    # return x_rot, theta_t
    rng = _as_numpy_rng(generator)
    mats = create_so2_rot_matrices(x.shape[0], rng, device=x.device, dtype=x.dtype)
    theta_t = torch.atan2(mats[:, 1, 0], mats[:, 0, 0])
    return _apply_batched_so2(x, mats, names, norm_scales), theta_t


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    generator: Optional[GeneratorLike] = None,
    norm_scales: Optional[dict] = None,
    apply_filter: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(3) rotations using the exact augmentation backend.

    Each jet in the batch receives an independently sampled rotation matrix
    drawn from the Haar measure on SO(3).
    Transformation logic is delegated to
    ``src.utils.augmentation.apply_so3_single_rotation``.

    When *apply_filter* is True, jets rotated outside detector acceptance
    (|ηrel| > 0.5, |φrel| > 0.5 in physical space) fall back to the original
    sample, contributing zero to the symmetry loss for that jet.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    _ = (px_index, py_index, pz_index)
    names = _require_feature_names(feature_names)

    # GPU path — fully vectorized, no CPU round-trip
    if x.device.type == "cuda":
        return _apply_random_so3_gpu(x, names, norm_scales, apply_filter)

    # ---- Legacy CPU/numpy path (kept for reference) ----
    # rng = _as_numpy_rng(generator)
    # _t0 = time.perf_counter()
    # x_np = _to_numpy_batch(x)
    # _TIMING["to_cpu_ms"] += (time.perf_counter() - _t0) * 1000
    # rot_mats_np = create_so3_rot_matrices(x_np.shape[0], rng)
    # x_rot_np = np.empty_like(x_np)
    # _t1 = time.perf_counter()
    # for i, rot in enumerate(rot_mats_np):
    #     x_rot_i = apply_so3_single_rotation(x_np[i:i+1], rot, names, norm_scales=norm_scales)
    #     if apply_filter:
    #         ok = so3_jet_filter(x_rot_i, names, norm_scales=norm_scales)
    #         if not ok[0]:
    #             x_rot_i = x_np[i:i+1]
    #     x_rot_np[i:i+1] = x_rot_i
    # _TIMING["loop_ms"] += (time.perf_counter() - _t1) * 1000
    # _t2 = time.perf_counter()
    # x_rot = _from_numpy_batch(x_rot_np, ref=x)
    # _TIMING["to_gpu_ms"] += (time.perf_counter() - _t2) * 1000
    # rot_mats = torch.from_numpy(rot_mats_np).to(device=x.device, dtype=x.dtype)
    # return x_rot, rot_mats
    rng = _as_numpy_rng(generator)
    rot_mats = create_so3_rot_matrices(x.shape[0], rng, device=x.device, dtype=x.dtype)
    return _apply_batched_so3(x, rot_mats, names, norm_scales, apply_filter), rot_mats


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    e_index: Optional[int] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    eta_max: float = 2.0,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[GeneratorLike] = None,
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random Lorentz boosts using the exact augmentation backend.

    Each jet in the batch receives an independently sampled boost.
    Transformation logic is delegated to
    ``src.utils.augmentation.apply_boost_preserve_phi`` and therefore expects
    the same required feature set used by ``apply_augmentation(..., 'lorentz')``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    _ = (e_index, px_index, py_index, pz_index, preserve_jet_phi, require_positive_energy)
    names = _require_feature_names(feature_names)

    if None in (
        feature_index(names, "j1_px", "px"),
        feature_index(names, "j1_py", "py"),
        feature_index(names, "j1_pz", "pz"),
    ):
        print("Lorentz augmentation: missing required px/py/pz features. Returning unaugmented data.")
        return x.clone(), torch.empty((0, 4, 4), device=x.device, dtype=x.dtype)

    # GPU path — fully vectorized, no CPU round-trip
    if x.device.type == "cuda":
        return _apply_random_lorentz_gpu(x, names, eta_max, norm_scales)

    # ---- Legacy CPU/numpy path (kept for reference) ----
    # rng = _as_numpy_rng(generator)
    # _t0 = time.perf_counter()
    # x_np = _to_numpy_batch(x)
    # _TIMING["to_cpu_ms"] += (time.perf_counter() - _t0) * 1000
    # boosts_np = create_lorentz_boosts(x.shape[0], eta_max, rng)
    # x_rot_np = np.empty_like(x_np)
    # _t1 = time.perf_counter()
    # for i, l_mat in enumerate(boosts_np):
    #     x_rot_np[i:i+1] = apply_boost_preserve_phi(x_np[i:i+1], l_mat, names, norm_scales=norm_scales)
    # _TIMING["loop_ms"] += (time.perf_counter() - _t1) * 1000
    # _t2 = time.perf_counter()
    # x_rot = _from_numpy_batch(x_rot_np, ref=x)
    # _TIMING["to_gpu_ms"] += (time.perf_counter() - _t2) * 1000
    # boosts = torch.from_numpy(boosts_np).to(device=x.device, dtype=x.dtype)
    # return x_rot, boosts
    rng = _as_numpy_rng(generator)
    boosts = create_lorentz_boosts(x.shape[0], eta_max, rng, device=x.device, dtype=x.dtype)
    return _apply_batched_lorentz(x, boosts, names, norm_scales), boosts


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
