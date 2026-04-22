"""Augmentation utilities for jet classification data.

Provides SO(2), SO(3) rotation and Lorentz-boost augmentation schemes
applied to particle-level 4-vectors.  All functions are pure (no class
state) and operate on numpy arrays.  The Lorentz boost includes a
frame-restoration step that preserves the jet azimuthal angle.
"""

from typing import List, Optional, Tuple

import numpy as np


# ── Numeric helpers ─────────────────────────────────────────────────


def compute_pt(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.sqrt(px ** 2 + py ** 2)


def compute_phi(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    return np.arctan2(py, px)


def compute_eta(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    return 0.5 * np.log((p + pz + eps) / (p - pz + eps))


def wrap_phi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def px_py_pz_to_pt_eta_phi(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# ── Rotation matrix generators ──────────────────────────────────────


def create_so2_rot_matrices(nbr_rot: int) -> np.ndarray:
    """Create *nbr_rot* evenly-spaced SO(2) rotation matrices (3×3)."""
    if nbr_rot <= 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    thetas = np.linspace(0.0, 2.0 * np.pi, nbr_rot, endpoint=False)
    rot_mats = np.zeros((nbr_rot, 3, 3), dtype=np.float32)
    rot_mats[:, 0, 0] = np.cos(thetas)
    rot_mats[:, 0, 1] = -np.sin(thetas)
    rot_mats[:, 1, 0] = np.sin(thetas)
    rot_mats[:, 1, 1] = np.cos(thetas)
    rot_mats[:, 2, 2] = 1.0
    return rot_mats


def create_so3_rot_matrices(
    num_rotations: int, rng: np.random.Generator
) -> np.ndarray:
    """Create *num_rotations* random SO(3) rotation matrices (3×3).

    Uses QR decomposition of random Gaussian matrices to produce
    uniformly distributed rotation matrices (Haar measure on SO(3)).
    """
    if num_rotations <= 0:
        return np.empty((0, 3, 3), dtype=np.float32)

    matrices = np.empty((num_rotations, 3, 3), dtype=np.float32)
    for i in range(num_rotations):
        M = rng.standard_normal((3, 3))
        Q, R = np.linalg.qr(M)
        # Ensure unique QR by fixing sign of diagonal of R
        Q = Q @ np.diag(np.sign(np.diag(R)))
        # Ensure proper rotation (det = +1, not reflection)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        matrices[i] = Q.astype(np.float32)

    return matrices


# ── SO(2) augmentation ─────────────────────────────────────────────


def apply_so2_single_rotation(
    x_data: np.ndarray,
    rotation_matrix: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Apply one SO(2) rotation to all jets."""
    px_idx = feature_index(feature_names, "j1_px", "px")
    py_idx = feature_index(feature_names, "j1_py", "py")
    pz_idx = feature_index(feature_names, "j1_pz", "pz")
    pt_idx = feature_index(feature_names, "j1_pt", "pt")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")

    if None in (px_idx, py_idx, pz_idx):
        print(
            "SO(2) augmentation requested but required px/py/pz "
            "features were not found. Returning unaugmented data."
        )
        return x_data

    x_new = x_data.copy()
    vec = np.stack(
        [x_data[:, :, px_idx], x_data[:, :, py_idx], x_data[:, :, pz_idx]],
        axis=-1,
    )
    vec_rot = np.einsum("ij,ncj->nci", rotation_matrix, vec, optimize=True)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]
    pt_r, _, phi_r = px_py_pz_to_pt_eta_phi(px_r, py_r, pz_r)

    # Only update features if their indices are valid and within bounds
    n_feats = x_new.shape[2]
    for idx, arr in zip([px_idx, py_idx, pz_idx, pt_idx, phi_idx], [px_r, py_r, pz_r, pt_r, phi_r]):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = arr
    return x_new


# ── SO(3) augmentation ─────────────────────────────────────────────


def apply_so3_single_rotation(
    x_data: np.ndarray,
    rotation_matrix: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Apply one SO(3) rotation to all jets (padding-aware)."""
    px_idx = feature_index(feature_names, "j1_px", "px")
    py_idx = feature_index(feature_names, "j1_py", "py")
    pz_idx = feature_index(feature_names, "j1_pz", "pz")
    pt_idx = feature_index(feature_names, "j1_pt", "pt")
    eta_idx = feature_index(feature_names, "j1_eta", "eta")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")
    etarel_idx = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx = feature_index(feature_names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(feature_names, "j1_deltaR", "deltaR")
    ptrel_idx = feature_index(feature_names, "j1_ptrel", "ptrel")

    if None in (px_idx, py_idx, pz_idx):
        print(
            "SO(3) augmentation requested but required px/py/pz "
            "features were not found. Returning unaugmented data."
        )
        return x_data

    n_feats = x_data.shape[2]

    # Derive padding mask from pt column when present; otherwise use pt=sqrt(px²+py²).
    if pt_idx is not None and pt_idx < n_feats:
        pad_mask = x_data[:, :, pt_idx] > 0
    else:
        pad_mask = (x_data[:, :, px_idx] ** 2 + x_data[:, :, py_idx] ** 2) > 0

    x_new = x_data.copy()
    vec = np.stack(
        [x_data[:, :, px_idx], x_data[:, :, py_idx], x_data[:, :, pz_idx]],
        axis=-1,
    )
    vec = vec * pad_mask[:, :, None]
    vec_rot = np.einsum("ij,ncj->nci", rotation_matrix, vec, optimize=True)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]

    pt_r, eta_r, phi_r = px_py_pz_to_pt_eta_phi(px_r, py_r, pz_r)
    jet_px = (px_r * pad_mask).sum(axis=1, keepdims=True)
    jet_py = (py_r * pad_mask).sum(axis=1, keepdims=True)
    jet_pz = (pz_r * pad_mask).sum(axis=1, keepdims=True)
    jet_pt, jet_eta, jet_phi = px_py_pz_to_pt_eta_phi(jet_px, jet_py, jet_pz)

    eta_rel = eta_r - jet_eta
    phi_rel = wrap_phi(phi_r - jet_phi)
    delta_r = np.sqrt(eta_rel ** 2 + phi_rel ** 2)
    pt_rel = pt_r / np.where(jet_pt > 0, jet_pt, 1.0)

    for idx, arr in zip(
        [px_idx, py_idx, pz_idx, pt_idx, eta_idx, phi_idx,
         etarel_idx, phirel_idx, delta_r_idx, ptrel_idx],
        [px_r, py_r, pz_r, pt_r, eta_r, phi_r,
         eta_rel, phi_rel, delta_r, pt_rel],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = np.where(pad_mask, arr, x_data[:, :, idx])
    return x_new


# ── SO(3) physical acceptance filter ────────────────────────────────


def _so3_jet_filter(
    x_aug: np.ndarray,
    feature_names: List[str],
    pt_max: float = 1600.0,
    eta_max: float = 3.0,
    etarel_max: float = 0.5,
    phirel_max: float = 0.5,
) -> np.ndarray:
    """Return boolean mask of jets passing SO(3) detector-acceptance cuts.

    A jet passes if **every** non-padded constituent satisfies:
        pT ∈ [0, pt_max],  η ∈ [-eta_max, eta_max],
        ηrel ∈ [-etarel_max, etarel_max],  φrel ∈ [-phirel_max, phirel_max].
    """
    pt_idx = feature_index(feature_names, "j1_pt", "pt")
    eta_idx = feature_index(feature_names, "j1_eta", "eta")
    etarel_idx = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx = feature_index(feature_names, "j1_phirel", "phirel")

    pt = x_aug[:, :, pt_idx]
    eta = x_aug[:, :, eta_idx]
    etarel = x_aug[:, :, etarel_idx]
    phirel = x_aug[:, :, phirel_idx]

    active = pt > 0  # non-padded constituents

    ok = (
        (pt >= 0) & (pt <= pt_max)
        & (eta >= -eta_max) & (eta <= eta_max)
        & (etarel >= -etarel_max) & (etarel <= etarel_max)
        & (phirel >= -phirel_max) & (phirel <= phirel_max)
    )

    # Padded constituents automatically pass
    constituent_ok = ~active | ok
    return constituent_ok.all(axis=1)  # shape (N,)


# ── Shared rotation augmentation loop ──────────────────────────────


def augment_with_rotations(
    x_data: np.ndarray,
    y_data: np.ndarray,
    rotation_mats: np.ndarray,
    space: str,
    shuffle: bool,
    rng: np.random.Generator,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple rotation matrices and concatenate with originals."""
    x_all = [x_data]
    y_all = [y_data]

    apply_fn = apply_so2_single_rotation if space == "so2" else apply_so3_single_rotation

    n_total_aug = 0
    n_kept_aug = 0

    for rotation_matrix in rotation_mats:
        x_aug = apply_fn(x_data, rotation_matrix, feature_names)

        if space == "so3":
            mask = _so3_jet_filter(x_aug, feature_names)
            n_total_aug += x_aug.shape[0]
            n_kept_aug += mask.sum()
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

    x_out = np.concatenate(x_all, axis=0)
    y_out = np.concatenate(y_all, axis=0)
    if shuffle:
        indices = rng.permutation(x_out.shape[0])
        x_out = x_out[indices]
        y_out = y_out[indices]
    return x_out, y_out


# ── Lorentz boost generation ───────────────────────────────────────


def create_lorentz_boosts(
    nbr_rot: int,
    eta_max: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random pure Lorentz boosts acting on (E, px, py, pz)."""
    if nbr_rot <= 0:
        return np.empty((0, 4, 4), dtype=np.float32)

    mats = np.zeros((nbr_rot, 4, 4), dtype=np.float32)

    for i in range(nbr_rot):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        eta = rng.uniform(-eta_max, eta_max)
        beta = np.tanh(eta) * direction
        bx, by, bz = beta
        beta2 = bx * bx + by * by + bz * bz

        if beta2 < 1e-12:
            mats[i] = np.eye(4, dtype=np.float32)
            continue

        gamma = 1.0 / np.sqrt(1.0 - beta2)
        gb = (gamma - 1.0) / beta2

        L = np.eye(4, dtype=np.float32)
        L[0, 0] = gamma
        L[0, 1:] = -gamma * beta
        L[1:, 0] = -gamma * beta
        L[1:, 1:] += gb * np.outer(beta, beta)
        mats[i] = L

    return mats


# ── Lorentz boost application (with frame restoration) ──────────────


def apply_boost_preserve_phi(
    x_data: np.ndarray,
    L: np.ndarray,
    feature_names: List[str],
) -> np.ndarray:
    """Boost 4-vectors and apply a compensating SO(2) rotation to preserve jet φ.

    When the energy feature is absent (partial feature sets used in symm_loss),
    energy is approximated as |p| = sqrt(px²+py²+pz²), which is exact for
    massless particles and a good approximation for LHC jet constituents.
    """
    x_new = x_data.copy()
    n_feats = x_data.shape[2]

    px_idx     = feature_index(feature_names, "j1_px",     "px")
    py_idx     = feature_index(feature_names, "j1_py",     "py")
    pz_idx     = feature_index(feature_names, "j1_pz",     "pz")
    e_idx      = feature_index(feature_names, "j1_e",      "e")
    pt_idx     = feature_index(feature_names, "j1_pt",     "pt")
    eta_idx    = feature_index(feature_names, "j1_eta",    "eta")
    phi_idx    = feature_index(feature_names, "j1_phi",    "phi")
    etarel_idx = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx = feature_index(feature_names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(feature_names, "j1_deltaR", "deltaR")
    ptrel_idx   = feature_index(feature_names, "j1_ptrel",  "ptrel")

    if None in (px_idx, py_idx, pz_idx):
        print(
            "Lorentz augmentation requested but required px/py/pz "
            "features were not found. Returning unaugmented data."
        )
        return x_data

    # Derive padding mask: prefer pt column; fall back to sqrt(px²+py²).
    if pt_idx is not None and pt_idx < n_feats:
        pad_mask = x_data[:, :, pt_idx] > 0
    else:
        pad_mask = (x_data[:, :, px_idx] ** 2 + x_data[:, :, py_idx] ** 2) > 0

    # Build 4-vectors: use stored energy if available, otherwise approximate
    # E ≈ |p| = sqrt(px²+py²+pz²), which is exact for massless particles.
    if e_idx is not None and e_idx < n_feats:
        e_in = x_data[:, :, e_idx]
    else:
        e_in = np.sqrt(
            x_data[:, :, px_idx] ** 2
            + x_data[:, :, py_idx] ** 2
            + x_data[:, :, pz_idx] ** 2
        )

    p4 = np.stack(
        [e_in, x_data[:, :, px_idx], x_data[:, :, py_idx], x_data[:, :, pz_idx]],
        axis=-1,
    )
    p4 = p4 * pad_mask[:, :, None]

    p4b = np.einsum("ij,ncj->nci", L, p4, optimize=True)

    e_b  = p4b[:, :, 0]
    px_b = p4b[:, :, 1]
    py_b = p4b[:, :, 2]
    pz_b = p4b[:, :, 3]

    px = x_data[:, :, px_idx] * pad_mask
    py = x_data[:, :, py_idx] * pad_mask

    jet_px_orig  = px.sum(axis=1, keepdims=True)
    jet_py_orig  = py.sum(axis=1, keepdims=True)
    jet_phi_orig = np.arctan2(jet_py_orig, jet_px_orig)

    jet_px_b    = (px_b * pad_mask).sum(axis=1, keepdims=True)
    jet_py_b    = (py_b * pad_mask).sum(axis=1, keepdims=True)
    jet_phi_boost = np.arctan2(jet_py_b, jet_px_b)

    theta = jet_phi_orig - jet_phi_boost
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    px_r =  cos_t * px_b - sin_t * py_b
    py_r =  sin_t * px_b + cos_t * py_b
    pz_r =  pz_b
    e_r  =  e_b

    px_r *= pad_mask
    py_r *= pad_mask
    pz_r *= pad_mask
    e_r  *= pad_mask

    energy_mask = (e_r > 0) & pad_mask

    pt  = compute_pt(px_r, py_r)
    phi = compute_phi(px_r, py_r)
    eta = compute_eta(px_r, py_r, pz_r)

    jet_px  = (px_r * pad_mask).sum(axis=1, keepdims=True)
    jet_py  = (py_r * pad_mask).sum(axis=1, keepdims=True)
    jet_pz  = (pz_r * pad_mask).sum(axis=1, keepdims=True)
    jet_pt  = compute_pt(jet_px, jet_py)
    jet_phi = compute_phi(jet_px, jet_py)
    jet_eta = compute_eta(jet_px, jet_py, jet_pz)

    eta_rel = eta - jet_eta
    phi_rel = wrap_phi(phi - jet_phi)
    delta_r = np.sqrt(eta_rel ** 2 + phi_rel ** 2)
    pt_rel  = pt / np.where(jet_pt > 0, jet_pt, 1.0)

    for idx, arr in zip(
        [px_idx, py_idx, pz_idx, e_idx, pt_idx, eta_idx, phi_idx,
         etarel_idx, phirel_idx, delta_r_idx, ptrel_idx],
        [px_r,   py_r,   pz_r,   e_r,   pt,     eta,     phi,
         eta_rel, phi_rel, delta_r, pt_rel],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = np.where(energy_mask, arr, x_data[:, :, idx])

    return x_new


def augment_with_lorentz_boosts(
    x_data: np.ndarray,
    y_data: np.ndarray,
    lorentz_mats: np.ndarray,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply multiple Lorentz boosts and concatenate with originals."""
    x_all = [x_data]
    y_all = [y_data]

    for L in lorentz_mats:
        x_aug = apply_boost_preserve_phi(x_data, L, feature_names)
        x_all.append(x_aug)
        y_all.append(y_data)

    return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)


# ── Public dispatcher ───────────────────────────────────────────────


def apply_augmentation(
    scheme: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    rng: np.random.Generator,
    rotation_nbr_rot: int = 8,
    rotation_shuffle: bool = True,
    lorentz_nbr_rot: int = 8,
    lorentz_eta_max: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the requested augmentation scheme to training data.

    Args:
        scheme: One of ``"none"``, ``"so2"``, ``"so3"``, ``"lorentz"``.
        x_train: ``(N, nconst, nfeats)`` particle features.
        y_train: ``(N, nclasses)`` one-hot labels.
        feature_names: Column names matching the last axis of *x_train*.
        rng: Numpy random generator for reproducibility.
        rotation_nbr_rot: Number of rotation copies (SO2/SO3).
        rotation_shuffle: Whether to shuffle after rotation augmentation.
        lorentz_nbr_rot: Number of Lorentz boost copies.
        lorentz_eta_max: Maximum rapidity for Lorentz boosts.

    Returns:
        Tuple of augmented ``(x_train, y_train)`` arrays.
    """
    scheme = scheme.lower()

    if scheme == "none":
        return x_train, y_train

    if scheme == "so2":
        rot_mats = create_so2_rot_matrices(rotation_nbr_rot)
        return augment_with_rotations(
            x_train, y_train, rot_mats, "so2",
            rotation_shuffle, rng, feature_names,
        )

    if scheme == "so3":
        rot_mats = create_so3_rot_matrices(rotation_nbr_rot, rng)
        return augment_with_rotations(
            x_train, y_train, rot_mats, "so3",
            rotation_shuffle, rng, feature_names,
        )

    if scheme == "lorentz":
        required = {
            "j1_px", "j1_py", "j1_pz", "j1_e", "j1_pt", "j1_eta",
            "j1_phi", "j1_etarel", "j1_phirel", "j1_deltaR", "j1_ptrel",
        }
        missing = [name for name in required if name not in feature_names]
        if missing:
            print(
                f"Lorentz augmentation requested but missing required "
                f"features: {missing}. Returning unaugmented data."
            )
            return x_train, y_train

        lorentz_mats = create_lorentz_boosts(lorentz_nbr_rot, lorentz_eta_max, rng)
        return augment_with_lorentz_boosts(
            x_train, y_train, lorentz_mats, feature_names,
        )

    raise ValueError(f"Unknown augmentation scheme: {scheme}")
