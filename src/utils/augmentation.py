"""Augmentation utilities for jet classification data.

Provides SO(2), SO(3) rotation and Lorentz-boost augmentation schemes
applied to particle-level 4-vectors.  All functions operate on torch
tensors and keep all computation on the input device (GPU or CPU).

Sampling is torch-native throughout: there is a single implementation per
group, used on whatever device ``x`` lives on. Matrices are built in float32
and cast to the input dtype at the end.

Note: transform sampling currently draws from the *global* torch RNG and is
therefore not independently seedable. Making it reproducible means threading a
device-matched ``torch.Generator`` through the three samplers below — a
single-point change now that the duplicate numpy path is gone.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# ── Numeric helpers (torch) ──────────────────────────────────────────


def compute_pt(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(px ** 2 + py ** 2)


def compute_phi(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.atan2(py, px)


def compute_eta(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p = torch.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return 0.5 * torch.log((p + pz + eps) / (p - pz + eps))


def wrap_phi(dphi: torch.Tensor) -> torch.Tensor:
    return (dphi + math.pi) % (2 * math.pi) - math.pi


def px_py_pz_to_pt_eta_phi(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py)


# ── Feature-index helper ───────────────────────────────────────────


def feature_index(feature_names: List[str], *candidate_names: str) -> Optional[int]:
    """Return the column index for the first matching feature name.

    Tries each candidate as-is and with a ``j1_`` prefix.
    """
    index_map = {name: idx for idx, name in enumerate(feature_names)}
    for name in candidate_names:
        if name in index_map:
            return index_map[name]
        prefixed = f"j1_{name}"
        if prefixed in index_map:
            return index_map[prefixed]
    return None


def get_norm_scale(norm_scales: Optional[dict], name: str) -> float:
    """Return the normalisation scale for *name*, or 1.0 if unavailable."""
    if norm_scales is None:
        return 1.0
    return float(norm_scales.get(name, norm_scales.get(f"j1_{name}", 1.0)))


# ── Rotation matrix generators (vectorized, torch) ──────────────────


def sample_so2_rot_matrices(
    nbr_rot: int,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample *nbr_rot* SO(2) rotation matrices (3×3) about the beam (z) axis.

    Returns ``(mats, thetas)``; angles are drawn uniformly on
    ``[min_angle, max_angle)``.
    """
    if nbr_rot <= 0:
        return (
            torch.empty((0, 3, 3), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )
    thetas = (
        torch.rand(nbr_rot, device=device, dtype=torch.float32)
        * (max_angle - min_angle)
        + min_angle
    )
    mats = torch.zeros(nbr_rot, 3, 3, dtype=torch.float32, device=device)
    mats[:, 0, 0] = torch.cos(thetas)
    mats[:, 0, 1] = -torch.sin(thetas)
    mats[:, 1, 0] = torch.sin(thetas)
    mats[:, 1, 1] = torch.cos(thetas)
    mats[:, 2, 2] = 1.0
    return mats.to(dtype), thetas.to(dtype)


def sample_haar_so3_rot_matrices(
    num_rotations: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create Haar-random SO(3) matrices from unit quaternions on *device*."""
    if num_rotations <= 0:
        return torch.empty((0, 3, 3), dtype=dtype, device=device)

    quat = torch.randn(num_rotations, 4, device=device, dtype=torch.float32)
    quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mats = torch.empty(num_rotations, 3, 3, device=device, dtype=torch.float32)
    mats[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    mats[:, 0, 1] = 2.0 * (xy - wz)
    mats[:, 0, 2] = 2.0 * (xz + wy)
    mats[:, 1, 0] = 2.0 * (xy + wz)
    mats[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    mats[:, 1, 2] = 2.0 * (yz - wx)
    mats[:, 2, 0] = 2.0 * (xz - wy)
    mats[:, 2, 1] = 2.0 * (yz + wx)
    mats[:, 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mats.to(dtype)


# ── SO(3) physical acceptance filter ────────────────────────────────


def so3_jet_filter(
    x_aug: torch.Tensor,
    feature_names: List[str],
    pt_max: float = 1600.0,
    eta_max: float = 3.0,
    etarel_max: float = 0.5,
    phirel_max: float = 0.5,
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Return boolean mask (N,) of jets passing SO(3) detector-acceptance cuts."""
    px_idx     = feature_index(feature_names, "j1_px",     "px")
    py_idx     = feature_index(feature_names, "j1_py",     "py")
    pt_idx     = feature_index(feature_names, "j1_pt",     "pt")
    eta_idx    = feature_index(feature_names, "j1_eta",    "eta")
    etarel_idx = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx = feature_index(feature_names, "j1_phirel", "phirel")

    n_feats = x_aug.shape[2]

    if pt_idx is not None and pt_idx < n_feats:
        active = x_aug[:, :, pt_idx] > 0
    elif px_idx is not None and py_idx is not None:
        active = (x_aug[:, :, px_idx] ** 2 + x_aug[:, :, py_idx] ** 2) > 0
    else:
        active = torch.ones(x_aug.shape[:2], dtype=torch.bool, device=x_aug.device)

    _etarel_max = etarel_max / get_norm_scale(norm_scales, "etarel")
    _phirel_max = phirel_max / get_norm_scale(norm_scales, "phirel")

    ok = torch.ones(x_aug.shape[:2], dtype=torch.bool, device=x_aug.device)
    if pt_idx is not None and pt_idx < n_feats:
        pt = x_aug[:, :, pt_idx]
        ok &= (pt >= 0) & (pt <= pt_max)
    if eta_idx is not None and eta_idx < n_feats:
        eta = x_aug[:, :, eta_idx]
        ok &= (eta >= -eta_max) & (eta <= eta_max)
    if etarel_idx is not None and etarel_idx < n_feats:
        etarel = x_aug[:, :, etarel_idx]
        ok &= (etarel >= -_etarel_max) & (etarel <= _etarel_max)
    if phirel_idx is not None and phirel_idx < n_feats:
        phirel = x_aug[:, :, phirel_idx]
        ok &= (phirel >= -_phirel_max) & (phirel <= _phirel_max)

    constituent_ok = ~active | ok
    return constituent_ok.all(dim=1)


# ── Lorentz boost generation (vectorized, torch) ────────────────────


def sample_lorentz_boosts(
    nbr_rot: int,
    eta_max: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample *nbr_rot* random Lorentz boost matrices (4×4) on *device*.

    The boost direction is isotropic in 3D and the rapidity is uniform on
    ``[-eta_max, eta_max]``. Fully vectorized — no Python loop over boosts.

    Note: an isotropic 3D boost is not a symmetry of a pp collision (only
    longitudinal boosts along the beam are). This is a deliberate probe of a
    *model* property, not a physical symmetry of the data.
    """
    if nbr_rot <= 0:
        return torch.empty((0, 4, 4), dtype=dtype, device=device)

    direction = torch.randn(nbr_rot, 3, dtype=torch.float32, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)

    eta   = (torch.rand(nbr_rot, dtype=torch.float32, device=device) * 2.0 - 1.0) * eta_max
    beta  = torch.tanh(eta).unsqueeze(1) * direction      # (B, 3)
    beta2 = (beta ** 2).sum(dim=1)                        # (B,)
    gamma = 1.0 / torch.sqrt((1.0 - beta2).clamp(min=1e-8))
    gb    = (gamma - 1.0) / beta2.clamp(min=1e-12)

    L = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).expand(nbr_rot, -1, -1).clone()
    L[:, 0, 0]  = gamma
    L[:, 0, 1:] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 0] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 1:] += gb.view(nbr_rot, 1, 1) * torch.bmm(beta.unsqueeze(2), beta.unsqueeze(1))

    near_zero = beta2 < 1e-12
    if near_zero.any():
        L[near_zero] = torch.eye(4, dtype=torch.float32, device=device)

    return L.to(dtype)


def _require_feature_names(feature_names: Optional[Sequence[str]]) -> List[str]:
    if feature_names is None:
        raise ValueError("feature_names must be provided to apply a transform.")
    return list(feature_names)


# ── Batched online transforms (one independent matrix per jet) ──────
#
# Unlike the single-matrix appliers above, these apply a *different*
# transformation matrix to every jet in the batch via one batched einsum:
# "bij,bcj->bci" (B jets, C constituents, K components) — zero Python loops,
# all computation on whatever device x lives on. This is the backend for
# per-batch ("online") augmentation, as opposed to baking fixed copies into
# the on-disk dataset.


def apply_so2_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Apply a different SO(2) rotation to each jet. mats: (B, 3, 3), x: (B, C, F)."""
    px_idx  = feature_index(feature_names, "j1_px",  "px")
    py_idx  = feature_index(feature_names, "j1_py",  "py")
    pz_idx  = feature_index(feature_names, "j1_pz",  "pz")
    pt_idx  = feature_index(feature_names, "j1_pt",  "pt")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")
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


def apply_so3_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
    apply_filter: bool = True,
) -> torch.Tensor:
    """Apply a different SO(3) rotation to each jet. mats: (B, 3, 3), x: (B, C, F)."""
    px_idx      = feature_index(feature_names, "j1_px",     "px")
    py_idx      = feature_index(feature_names, "j1_py",     "py")
    pz_idx      = feature_index(feature_names, "j1_pz",     "pz")
    pt_idx      = feature_index(feature_names, "j1_pt",     "pt")
    eta_idx     = feature_index(feature_names, "j1_eta",    "eta")
    phi_idx     = feature_index(feature_names, "j1_phi",    "phi")
    etarel_idx  = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx  = feature_index(feature_names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(feature_names, "j1_deltaR", "deltaR")
    ptrel_idx   = feature_index(feature_names, "j1_ptrel",  "ptrel")
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
        ok = so3_jet_filter(x_new, feature_names, norm_scales=norm_scales)  # (B,)
        failed = ~ok
        if failed.any():
            x_new[failed] = x[failed]
    return x_new


def apply_lorentz_batched(
    x: torch.Tensor,
    boosts: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Apply a different Lorentz boost to each jet (no compensating rotation —
    jet φ is left as the boost leaves it). boosts: (B, 4, 4), x: (B, C, F)."""
    px_idx      = feature_index(feature_names, "j1_px",     "px")
    py_idx      = feature_index(feature_names, "j1_py",     "py")
    pz_idx      = feature_index(feature_names, "j1_pz",     "pz")
    e_idx       = feature_index(feature_names, "j1_e",      "e")
    pt_idx      = feature_index(feature_names, "j1_pt",     "pt")
    eta_idx     = feature_index(feature_names, "j1_eta",    "eta")
    phi_idx     = feature_index(feature_names, "j1_phi",    "phi")
    etarel_idx  = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx  = feature_index(feature_names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(feature_names, "j1_deltaR", "deltaR")
    ptrel_idx   = feature_index(feature_names, "j1_ptrel",  "ptrel")
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

    px_r = px_b * pad_mask
    py_r = py_b * pad_mask
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


# ── Random online transforms (public entry points) ──────────────────
#
# Each jet in the batch receives an independently sampled transformation.
# Matrices are sampled with torch on whatever device x lives on (no CPU
# round-trip on GPU), then applied with a single batched einsum.


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled SO(2) rotation to each jet in the batch.

    Returns ``(x_rotated, thetas)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    mats, thetas = sample_so2_rot_matrices(
        x.shape[0], min_angle=min_angle, max_angle=max_angle,
        device=x.device, dtype=x.dtype,
    )
    return apply_so2_batched(x, mats, names, norm_scales), thetas


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    norm_scales: Optional[dict] = None,
    apply_filter: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Haar-random SO(3) rotation to each jet.

    When *apply_filter* is True, jets rotated outside detector acceptance
    (|ηrel| > 0.5, |φrel| > 0.5 in physical space) fall back to the original
    sample.

    Returns ``(x_rotated, rotation_matrices)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    mats = sample_haar_so3_rot_matrices(x.shape[0], device=x.device, dtype=x.dtype)
    return apply_so3_batched(x, mats, names, norm_scales, apply_filter), mats


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: Optional[Sequence[str]] = None,
    eta_max: float = 2.0,
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Lorentz boost to each jet in the batch.

    No compensating rotation is applied — jet φ is left as the boost leaves it.

    Returns ``(x_boosted, boost_matrices)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)

    if None in (
        feature_index(names, "j1_px", "px"),
        feature_index(names, "j1_py", "py"),
        feature_index(names, "j1_pz", "pz"),
    ):
        print("Lorentz augmentation: missing required px/py/pz features. Returning unaugmented data.")
        return x.clone(), torch.empty((0, 4, 4), device=x.device, dtype=x.dtype)

    boosts = sample_lorentz_boosts(x.shape[0], eta_max, device=x.device, dtype=x.dtype)
    return apply_lorentz_batched(x, boosts, names, norm_scales), boosts


# ── Permutation augmentation ────────────────────────────────────────


def permute_constituents(
    x_data: torch.Tensor, rng: Optional[np.random.Generator] = None
) -> torch.Tensor:
    """Randomly permute constituent order within each jet.

    With *rng* given, permutations are drawn from that numpy Generator (used by
    ``InvarianceMetricCallback``, which owns a seeded one). With *rng=None* they
    are drawn torch-natively on the input's device, matching how so2/so3/lorentz
    sample -- this is the path used for online per-batch augmentation.
    """
    n_jets, n_const = x_data.shape[:2]
    if rng is None:
        rand_vals = torch.rand(n_jets, n_const, dtype=torch.float32)
        if isinstance(x_data, torch.Tensor):
            rand_vals = rand_vals.to(x_data.device)
    else:
        rand_vals = torch.as_tensor(
            rng.random((n_jets, n_const)), dtype=torch.float32,
            device=x_data.device if isinstance(x_data, torch.Tensor) else None,
        )
    perms = torch.argsort(rand_vals, dim=1)
    idx = torch.arange(n_jets)[:, None]
    if isinstance(x_data, torch.Tensor):
        idx = idx.to(x_data.device)
    return x_data[idx, perms]
