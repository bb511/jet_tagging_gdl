"""Augmentation utilities for jet classification data.

Provides SO(2), SO(3) rotation and Lorentz-boost augmentation schemes
applied to particle-level 4-vectors.  All functions operate on torch
tensors and keep all computation on the input device (GPU or CPU).
"""

import math
from typing import List, Optional, Tuple

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


def create_so2_rot_matrices(
    nbr_rot: int,
    rng: np.random.Generator,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create *nbr_rot* random SO(2) rotation matrices (3×3) on *device*."""
    if nbr_rot <= 0:
        return torch.empty((0, 3, 3), dtype=dtype, device=device)
    thetas = torch.as_tensor(
        rng.uniform(0.0, 2.0 * math.pi, nbr_rot), dtype=dtype, device=device
    )
    mats = torch.zeros(nbr_rot, 3, 3, dtype=dtype, device=device)
    mats[:, 0, 0] = torch.cos(thetas)
    mats[:, 0, 1] = -torch.sin(thetas)
    mats[:, 1, 0] = torch.sin(thetas)
    mats[:, 1, 1] = torch.cos(thetas)
    mats[:, 2, 2] = 1.0
    return mats


def create_so3_rot_matrices(
    num_rotations: int,
    rng: np.random.Generator,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create *num_rotations* Haar-random SO(3) matrices via batched QR on *device*."""
    if num_rotations <= 0:
        return torch.empty((0, 3, 3), dtype=dtype, device=device)
    M = torch.randn(num_rotations, 3, 3, device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    # Fix sign convention (match numpy: column *= sign of diagonal of R)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))  # (B, 3)
    Q = Q * signs.unsqueeze(1)
    # Ensure det = +1 (not a reflection)
    neg = torch.linalg.det(Q) < 0
    if neg.any():
        Q[neg, :, 0] *= -1
    return Q.to(dtype)


# ── SO(2) augmentation ─────────────────────────────────────────────


def apply_so2_single_rotation(
    x_data: torch.Tensor,
    rotation_matrix: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Apply one SO(2) rotation matrix to all jets in *x_data*.

    All computation stays on the device of *x_data*.
    """
    px_idx  = feature_index(feature_names, "j1_px",  "px")
    py_idx  = feature_index(feature_names, "j1_py",  "py")
    pz_idx  = feature_index(feature_names, "j1_pz",  "pz")
    pt_idx  = feature_index(feature_names, "j1_pt",  "pt")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")

    if None in (px_idx, py_idx, pz_idx):
        print("SO(2) augmentation: missing px/py/pz features. Returning unaugmented data.")
        return x_data

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    rot = rotation_matrix.to(device=x_data.device, dtype=x_data.dtype)
    vec = torch.stack(
        [x_data[:, :, px_idx] * s_px,
         x_data[:, :, py_idx] * s_py,
         x_data[:, :, pz_idx] * s_pz],
        dim=-1,
    )
    vec_rot = torch.einsum("ij,ncj->nci", rot, vec)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]
    pt_r  = compute_pt(px_r, py_r)
    phi_r = compute_phi(px_r, py_r)

    x_new = x_data.clone()
    n_feats = x_new.shape[2]
    for idx, arr, feat in zip(
        [px_idx, py_idx, pz_idx, pt_idx, phi_idx],
        [px_r,   py_r,   pz_r,   pt_r,   phi_r],
        ["px",   "py",   "pz",   "pt",   "phi"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = arr / get_norm_scale(norm_scales, feat)
    return x_new


# ── SO(3) augmentation ─────────────────────────────────────────────


def apply_so3_single_rotation(
    x_data: torch.Tensor,
    rotation_matrix: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Apply one SO(3) rotation matrix to all jets in *x_data* (padding-aware).

    All computation stays on the device of *x_data*.
    """
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
        print("SO(3) augmentation: missing px/py/pz features. Returning unaugmented data.")
        return x_data

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")
    n_feats = x_data.shape[2]

    px_phys = x_data[:, :, px_idx] * s_px
    py_phys = x_data[:, :, py_idx] * s_py
    pz_phys = x_data[:, :, pz_idx] * s_pz
    pad_mask = (px_phys ** 2 + py_phys ** 2) > 0  # (N, C)

    rot = rotation_matrix.to(device=x_data.device, dtype=x_data.dtype)
    vec = torch.stack([px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    vec_rot = torch.einsum("ij,ncj->nci", rot, vec)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]

    pt_r, eta_r, phi_r = px_py_pz_to_pt_eta_phi(px_r, py_r, pz_r)
    jet_px = (px_r * pad_mask).sum(dim=1, keepdim=True)
    jet_py = (py_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pz = (pz_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pt, jet_eta, jet_phi = px_py_pz_to_pt_eta_phi(jet_px, jet_py, jet_pz)

    eta_rel = eta_r - jet_eta
    phi_rel = wrap_phi(phi_r - jet_phi)
    delta_r = torch.sqrt(eta_rel ** 2 + phi_rel ** 2)
    pt_rel  = pt_r / jet_pt.clamp(min=1e-8)

    x_new = x_data.clone()
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
                x_data[:, :, idx],
            )
    return x_new


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


# ── Shared rotation augmentation loop ──────────────────────────────


def augment_with_rotations(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    rotation_mats: torch.Tensor,
    space: str,
    shuffle: bool,
    rng: np.random.Generator,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multiple rotation matrices and concatenate with originals."""
    x_all = [x_data]
    y_all = [y_data]

    apply_fn = apply_so2_single_rotation if space == "so2" else apply_so3_single_rotation

    n_total_aug = 0
    n_kept_aug  = 0

    for rotation_matrix in rotation_mats:
        x_aug = apply_fn(x_data, rotation_matrix, feature_names, norm_scales)

        if space == "so3":
            mask = so3_jet_filter(x_aug, feature_names, norm_scales=norm_scales)
            n_total_aug += x_aug.shape[0]
            n_kept_aug  += int(mask.sum())
            x_aug = x_aug[mask]
            y_aug = y_data[mask]
        else:
            y_aug = y_data

        x_all.append(x_aug)
        y_all.append(y_aug)

    if space == "so3":
        print(
            f"SO(3) filter: kept {n_kept_aug}/{n_total_aug} augmented jets "
            f"({100 * n_kept_aug / max(n_total_aug, 1):.1f}%)"
        )

    x_out = torch.cat(x_all, dim=0)
    y_out = torch.cat(y_all, dim=0)
    if shuffle:
        indices = torch.randperm(x_out.shape[0], device=x_out.device)
        x_out = x_out[indices]
        y_out = y_out[indices]
    return x_out, y_out


# ── Lorentz boost generation (vectorized, torch) ────────────────────


def create_lorentz_boosts(
    nbr_rot: int,
    eta_max: float,
    rng: np.random.Generator,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate *nbr_rot* random Lorentz boost matrices (4×4) on *device*.

    Fully vectorized — no Python loop over individual boosts.
    """
    if nbr_rot <= 0:
        return torch.empty((0, 4, 4), dtype=dtype, device=device)

    direction = torch.randn(nbr_rot, 3, dtype=torch.float32, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)

    eta  = torch.as_tensor(
        rng.uniform(-eta_max, eta_max, nbr_rot), dtype=torch.float32, device=device
    )
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


# ── Lorentz boost application (with frame restoration) ──────────────


def apply_boost_preserve_phi(
    x_data: torch.Tensor,
    L: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> torch.Tensor:
    """Boost 4-vectors and apply a compensating SO(2) rotation to preserve jet φ.

    All computation stays on the device of *x_data*.
    Energy approximated as |p| when the energy feature is absent.
    """
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
        print("Lorentz augmentation: missing px/py/pz features. Returning unaugmented data.")
        return x_data

    n_feats = x_data.shape[2]
    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    px_phys = x_data[:, :, px_idx] * s_px
    py_phys = x_data[:, :, py_idx] * s_py
    pz_phys = x_data[:, :, pz_idx] * s_pz
    pad_mask = (px_phys ** 2 + py_phys ** 2) > 0

    if e_idx is not None and e_idx < n_feats:
        e_in = x_data[:, :, e_idx] * get_norm_scale(norm_scales, "e")
    else:
        e_in = torch.sqrt(px_phys ** 2 + py_phys ** 2 + pz_phys ** 2)

    p4  = torch.stack([e_in, px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    boost = L.to(device=x_data.device, dtype=x_data.dtype)
    p4b = torch.einsum("ij,ncj->nci", boost, p4)

    e_b  = p4b[:, :, 0]; px_b = p4b[:, :, 1]
    py_b = p4b[:, :, 2]; pz_b = p4b[:, :, 3]

    jet_phi_orig  = torch.atan2(
        (py_phys * pad_mask).sum(dim=1, keepdim=True),
        (px_phys * pad_mask).sum(dim=1, keepdim=True),
    )
    jet_phi_boost = torch.atan2(
        (py_b * pad_mask).sum(dim=1, keepdim=True),
        (px_b * pad_mask).sum(dim=1, keepdim=True),
    )

    theta = jet_phi_orig - jet_phi_boost
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)

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

    x_new = x_data.clone()
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
                x_data[:, :, idx],
            )
    return x_new


def augment_with_lorentz_boosts(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    lorentz_mats: torch.Tensor,
    feature_names: List[str],
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply multiple Lorentz boosts and concatenate with originals."""
    x_all = [x_data]
    y_all = [y_data]
    for L in lorentz_mats:
        x_all.append(apply_boost_preserve_phi(x_data, L, feature_names, norm_scales))
        y_all.append(y_data)
    return torch.cat(x_all, dim=0), torch.cat(y_all, dim=0)


# ── Permutation augmentation ────────────────────────────────────────


def permute_constituents(
    x_data: torch.Tensor, rng: np.random.Generator
) -> torch.Tensor:
    """Randomly permute constituent order within each jet."""
    n_jets, n_const = x_data.shape[:2]
    rand_vals = torch.as_tensor(
        rng.random((n_jets, n_const)), dtype=torch.float32, device=x_data.device
    )
    perms = torch.argsort(rand_vals, dim=1)
    return x_data[torch.arange(n_jets, device=x_data.device)[:, None], perms]


# ── Public dispatcher ───────────────────────────────────────────────


def apply_augmentation(
    scheme: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    feature_names: List[str],
    rng: np.random.Generator,
    rotation_nbr_rot: int = 8,
    rotation_shuffle: bool = True,
    lorentz_nbr_rot: int = 8,
    lorentz_eta_max: float = 2.0,
    norm_scales: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the requested augmentation scheme to training data."""
    scheme = scheme.lower()

    if scheme == "none":
        return x_train, y_train

    if scheme == "so2":
        rot_mats = create_so2_rot_matrices(
            rotation_nbr_rot, rng, device=x_train.device, dtype=x_train.dtype
        )
        return augment_with_rotations(
            x_train, y_train, rot_mats, "so2",
            rotation_shuffle, rng, feature_names, norm_scales,
        )

    if scheme == "so3":
        rot_mats = create_so3_rot_matrices(
            rotation_nbr_rot, rng, device=x_train.device, dtype=x_train.dtype
        )
        return augment_with_rotations(
            x_train, y_train, rot_mats, "so3",
            rotation_shuffle, rng, feature_names, norm_scales,
        )

    if scheme == "lorentz":
        required = {
            "j1_px", "j1_py", "j1_pz", "j1_e", "j1_pt", "j1_eta",
            "j1_phi", "j1_etarel", "j1_phirel", "j1_deltaR", "j1_ptrel",
        }
        missing = [name for name in required if name not in feature_names]
        if missing:
            print(f"Lorentz augmentation: missing features {missing}. Returning unaugmented data.")
            return x_train, y_train
        lorentz_mats = create_lorentz_boosts(
            lorentz_nbr_rot, lorentz_eta_max, rng,
            device=x_train.device, dtype=x_train.dtype,
        )
        return augment_with_lorentz_boosts(
            x_train, y_train, lorentz_mats, feature_names, norm_scales,
        )

    raise ValueError(f"Unknown augmentation scheme: {scheme}")
