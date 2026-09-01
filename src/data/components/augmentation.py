"""Transforms of the constituent four-vectors of a jet.

Functions take and return ``(batch, constituents, features)`` tensors on the device of the input,
matrices built in float32 and cast back at the end. One matrix is drawn per jet and applied through
a batched einsum, ``"bij,bcj->bci"`` over jets, constituents and momentum components.

Feature columns are addressed by name, bare or with the record's ``j1_`` prefix, and every
transform first undoes the normalisation with ``norm_scales``, the per-column factors back to GeV,
since the acceptance cuts and the boost rapidities are defined in physical units (GeV, radians).

Transforms draw from the global torch RNG, so ``lightning.seed_everything`` fixes them.
"""

import math
from collections.abc import Callable, Sequence

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

Transform = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]


def compute_pt(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(px**2 + py**2)


def compute_phi(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    return torch.atan2(py, px)


def compute_eta(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    p = torch.sqrt(px**2 + py**2 + pz**2)
    # eps keeps a purely longitudinal constituent, and a padded all-zero row, off the log
    # singularity: both go to eta = 0 rather than to +/- inf.
    return 0.5 * torch.log((p + pz + eps) / (p - pz + eps))


def wrap_phi(dphi: torch.Tensor) -> torch.Tensor:
    return (dphi + math.pi) % (2 * math.pi) - math.pi


def feature_index(feature_names: list[str], *candidate_names: str) -> int | None:
    """Return the column of the first candidate present, each tried bare and ``j1_``-prefixed."""
    index_map = {name: idx for idx, name in enumerate(feature_names)}
    for name in candidate_names:
        if name in index_map:
            return index_map[name]
        prefixed = f"j1_{name}"
        if prefixed in index_map:
            return index_map[prefixed]
    return None


def get_norm_scale(norm_scales: dict[str, float] | None, name: str) -> float:
    """Return the normalisation scale for *name*, or 1.0 if unavailable."""
    if norm_scales is None:
        return 1.0
    return float(norm_scales.get(name, norm_scales.get(f"j1_{name}", 1.0)))


def sample_so2_rot_matrices(
    nbr_rot: int,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample *nbr_rot* SO(2) rotation matrices about the beam (z) axis, one per jet.

    :param min_angle: lower end of the uniform angle range, in radians.
    :param max_angle: upper end of that range, in radians.
    :return: matrices of shape (nbr_rot, 3, 3) and the angles they turn through, (nbr_rot,).
    """
    if nbr_rot <= 0:
        return (
            torch.empty((0, 3, 3), dtype=dtype, device=device),
            torch.empty((0,), dtype=dtype, device=device),
        )
    thetas = (
        torch.rand(nbr_rot, device=device, dtype=torch.float32) * (max_angle - min_angle)
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
    """Create Haar-random SO(3) matrices from unit quaternions, shape (num_rotations, 3, 3)."""
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


def so3_jet_filter(
    pt: torch.Tensor,
    eta: torch.Tensor,
    etarel: torch.Tensor,
    phirel: torch.Tensor,
    active: torch.Tensor,
    pt_max: float = 1600.0,
    eta_max: float = 3.0,
    etarel_max: float = 0.5,
    phirel_max: float = 0.5,
) -> torch.Tensor:
    """Mark the jets whose constituents all pass the SO(3) detector-acceptance cuts.

    The constituent quantities are of shape (jets, constituents) and, like the ``*_max``
    thresholds, in physical un-normalised units (GeV, radians), so no threshold ever has to be
    rescaled by a norm scale. *etarel* and *phirel* are measured from the jet axis, *phirel*
    wrapped to [-pi, pi).

    :param active: True for a real constituent; a padded slot never fails a cut.
    :return: mask of shape (jets,), True where every real constituent passes.
    """
    ok = (
        (pt >= 0)
        & (pt <= pt_max)
        & (eta >= -eta_max)
        & (eta <= eta_max)
        & (etarel >= -etarel_max)
        & (etarel <= etarel_max)
        & (phirel >= -phirel_max)
        & (phirel <= phirel_max)
    )
    constituent_ok = ~active | ok
    return constituent_ok.all(dim=1)


def sample_lorentz_boosts(
    nbr_rot: int,
    eta_max: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample *nbr_rot* random Lorentz boost matrices, one per jet.

    The direction is isotropic and the rapidity uniform on ``[-eta_max, eta_max]``. An isotropic
    boost is not a symmetry of a pp collision, where only boosts along the beam are: it probes a
    property of the model, not one of the data.

    :return: matrices of shape (nbr_rot, 4, 4), acting on [E, px, py, pz].
    """
    if nbr_rot <= 0:
        return torch.empty((0, 4, 4), dtype=dtype, device=device)

    direction = torch.randn(nbr_rot, 3, dtype=torch.float32, device=device)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)

    eta = (torch.rand(nbr_rot, dtype=torch.float32, device=device) * 2.0 - 1.0) * eta_max
    beta = torch.tanh(eta).unsqueeze(1) * direction  # (B, 3)
    beta2 = (beta**2).sum(dim=1)  # (B,)
    gamma = 1.0 / torch.sqrt((1.0 - beta2).clamp(min=1e-8))
    gb = (gamma - 1.0) / beta2.clamp(min=1e-12)

    eye = torch.eye(4, dtype=torch.float32, device=device)
    L = eye.unsqueeze(0).expand(nbr_rot, -1, -1).clone()
    L[:, 0, 0] = gamma
    L[:, 0, 1:] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 0] = -gamma.unsqueeze(1) * beta
    L[:, 1:, 1:] += gb.view(nbr_rot, 1, 1) * torch.bmm(beta.unsqueeze(2), beta.unsqueeze(1))

    near_zero = beta2 < 1e-12
    if near_zero.any():
        L[near_zero] = eye

    return L.to(dtype)


def _require_feature_names(feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        raise ValueError("feature_names must be provided to apply a transform.")
    return list(feature_names)


def apply_so2_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Rotate every jet of the batch about the beam axis by its own matrix.

    :param mats: one rotation per jet, shape (jets, 3, 3).
    :return: the rotated batch, of the shape and the normalisation of *x*. Whichever of px, py, pz,
        pt and phi it carries are rewritten; every other column is left alone.
    """
    px_idx = feature_index(feature_names, "j1_px", "px")
    py_idx = feature_index(feature_names, "j1_py", "py")
    pz_idx = feature_index(feature_names, "j1_pz", "pz")
    pt_idx = feature_index(feature_names, "j1_pt", "pt")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")
    if None in (px_idx, py_idx, pz_idx):
        return x.clone()

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    vec = torch.stack(
        [
            x[:, :, px_idx] * s_px,
            x[:, :, py_idx] * s_py,
            x[:, :, pz_idx] * s_pz,
        ],
        dim=-1,
    )  # (B, C, 3)
    vec_rot = torch.einsum("bij,bcj->bci", mats, vec)  # (B, C, 3)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]
    pt_r = compute_pt(px_r, py_r)
    phi_r = compute_phi(px_r, py_r)

    n_feats = x.shape[2]
    x_new = x.clone()
    for idx, arr, feat in zip(
        [px_idx, py_idx, pz_idx, pt_idx, phi_idx],
        [px_r, py_r, pz_r, pt_r, phi_r],
        ["px", "py", "pz", "pt", "phi"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = arr / get_norm_scale(norm_scales, feat)
    return x_new


def apply_so3_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None = None,
    apply_filter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate every jet of the batch in three dimensions by its own matrix.

    :param mats: one rotation per jet, shape (jets, 3, 3).
    :param apply_filter: drop the jets a rotation carries outside detector acceptance rather than
        revert them, since reverting would make T(x) equal x for them and deflate every invariance
        metric measured against the rotated batch.
    :return: the surviving jets alone, and a mask over the input batch, so the caller can subset
        the labels, and anything else aligned with that batch, the same way.
    """
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
        return x.clone(), torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")
    n_feats = x.shape[2]

    px_phys = x[:, :, px_idx] * s_px
    py_phys = x[:, :, py_idx] * s_py
    pz_phys = x[:, :, pz_idx] * s_pz
    pad_mask = (px_phys**2 + py_phys**2) > 0  # (B, C)

    vec = torch.stack([px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    vec_rot = torch.einsum("bij,bcj->bci", mats, vec)  # (B, C, 3)

    px_r = vec_rot[:, :, 0]
    py_r = vec_rot[:, :, 1]
    pz_r = vec_rot[:, :, 2]

    pt_r = compute_pt(px_r, py_r)
    eta_r = compute_eta(px_r, py_r, pz_r)
    phi_r = compute_phi(px_r, py_r)

    jet_px = (px_r * pad_mask).sum(dim=1, keepdim=True)
    jet_py = (py_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pz = (pz_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pt = compute_pt(jet_px, jet_py)
    jet_eta = compute_eta(jet_px, jet_py, jet_pz)
    jet_phi = compute_phi(jet_px, jet_py)

    eta_rel = eta_r - jet_eta
    phi_rel = wrap_phi(phi_r - jet_phi)
    delta_r = torch.sqrt(eta_rel**2 + phi_rel**2)
    pt_rel = pt_r / jet_pt.clamp(min=1e-8)

    if apply_filter:
        keep_mask = so3_jet_filter(pt_r, eta_r, eta_rel, phi_rel, active=pad_mask)
    else:
        keep_mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    x_new = x.clone()
    for idx, arr, feat in zip(
        [
            px_idx,
            py_idx,
            pz_idx,
            pt_idx,
            eta_idx,
            phi_idx,
            etarel_idx,
            phirel_idx,
            delta_r_idx,
            ptrel_idx,
        ],
        [px_r, py_r, pz_r, pt_r, eta_r, phi_r, eta_rel, phi_rel, delta_r, pt_rel],
        ["px", "py", "pz", "pt", "eta", "phi", "etarel", "phirel", "deltaR", "ptrel"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = torch.where(
                pad_mask,
                arr / get_norm_scale(norm_scales, feat),
                x[:, :, idx],
            )

    return x_new[keep_mask], keep_mask


def apply_lorentz_batched(
    x: torch.Tensor,
    boosts: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Boost every jet of the batch by its own matrix.

    No compensating rotation follows, so the jet azimuth is left where the boost puts it.

    :param boosts: one boost per jet, shape (jets, 4, 4), acting on [E, px, py, pz].
    :return: the boosted batch, of the shape and the normalisation of *x*. Constituents whose
        boosted energy is not positive keep their stored values.
    """
    px_idx = feature_index(feature_names, "j1_px", "px")
    py_idx = feature_index(feature_names, "j1_py", "py")
    pz_idx = feature_index(feature_names, "j1_pz", "pz")
    e_idx = feature_index(feature_names, "j1_e", "e")
    pt_idx = feature_index(feature_names, "j1_pt", "pt")
    eta_idx = feature_index(feature_names, "j1_eta", "eta")
    phi_idx = feature_index(feature_names, "j1_phi", "phi")
    etarel_idx = feature_index(feature_names, "j1_etarel", "etarel")
    phirel_idx = feature_index(feature_names, "j1_phirel", "phirel")
    delta_r_idx = feature_index(feature_names, "j1_deltaR", "deltaR")
    ptrel_idx = feature_index(feature_names, "j1_ptrel", "ptrel")
    if None in (px_idx, py_idx, pz_idx):
        return x.clone()

    n_feats = x.shape[2]
    s_px = get_norm_scale(norm_scales, "px")
    s_py = get_norm_scale(norm_scales, "py")
    s_pz = get_norm_scale(norm_scales, "pz")

    px_phys = x[:, :, px_idx] * s_px
    py_phys = x[:, :, py_idx] * s_py
    pz_phys = x[:, :, pz_idx] * s_pz
    pad_mask = (px_phys**2 + py_phys**2) > 0

    if e_idx is not None and e_idx < n_feats:
        e_in = x[:, :, e_idx] * get_norm_scale(norm_scales, "e")
    else:
        e_in = torch.sqrt(px_phys**2 + py_phys**2 + pz_phys**2)

    p4 = torch.stack([e_in, px_phys, py_phys, pz_phys], dim=-1) * pad_mask.unsqueeze(-1)
    p4b = torch.einsum("bij,bcj->bci", boosts, p4)  # (B, C, 4)

    e_b = p4b[:, :, 0]
    px_b = p4b[:, :, 1]
    py_b = p4b[:, :, 2]
    pz_b = p4b[:, :, 3]

    px_r = px_b * pad_mask
    py_r = py_b * pad_mask
    pz_r = pz_b * pad_mask
    e_r = e_b * pad_mask
    energy_mask = (e_r > 0) & pad_mask

    pt = compute_pt(px_r, py_r)
    phi = compute_phi(px_r, py_r)
    eta = compute_eta(px_r, py_r, pz_r)

    jet_px = (px_r * pad_mask).sum(dim=1, keepdim=True)
    jet_py = (py_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pz = (pz_r * pad_mask).sum(dim=1, keepdim=True)
    jet_pt = compute_pt(jet_px, jet_py)
    jet_phi = compute_phi(jet_px, jet_py)
    jet_eta = compute_eta(jet_px, jet_py, jet_pz)

    eta_rel = eta - jet_eta
    phi_rel = wrap_phi(phi - jet_phi)
    delta_r = torch.sqrt(eta_rel**2 + phi_rel**2)
    pt_rel = pt / jet_pt.clamp(min=1e-8)

    x_new = x.clone()
    for idx, arr, feat in zip(
        [
            px_idx,
            py_idx,
            pz_idx,
            e_idx,
            pt_idx,
            eta_idx,
            phi_idx,
            etarel_idx,
            phirel_idx,
            delta_r_idx,
            ptrel_idx,
        ],
        [px_r, py_r, pz_r, e_r, pt, eta, phi, eta_rel, phi_rel, delta_r, pt_rel],
        ["px", "py", "pz", "e", "pt", "eta", "phi", "etarel", "phirel", "deltaR", "ptrel"],
    ):
        if idx is not None and 0 <= idx < n_feats:
            x_new[:, :, idx] = torch.where(
                energy_mask,
                arr / get_norm_scale(norm_scales, feat),
                x[:, :, idx],
            )
    return x_new


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled SO(2) rotation to each jet of the batch.

    :param min_angle: lower end of the uniform angle range, in radians.
    :param max_angle: upper end of that range, in radians.
    :return: the rotated batch and the angles it was turned through, shape (jets,).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    mats, thetas = sample_so2_rot_matrices(
        x.shape[0],
        min_angle=min_angle,
        max_angle=max_angle,
        device=x.device,
        dtype=x.dtype,
    )
    return apply_so2_batched(x, mats, names, norm_scales), thetas


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    norm_scales: dict[str, float] | None = None,
    apply_filter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Haar-random SO(3) rotation to each jet of the batch.

    :param apply_filter: drop the jets a rotation carries outside the acceptance bounds of
        :func:`so3_jet_filter`, rather than revert them to their unrotated values.
    :return: the rotated batch and the matrices it was rotated by, both already subset to the
        surviving jets, and the acceptance mask over the input batch, so the caller can subset the
        labels, and anything else aligned with that batch, the same way.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    mats = sample_haar_so3_rot_matrices(x.shape[0], device=x.device, dtype=x.dtype)
    x_new, keep_mask = apply_so3_batched(x, mats, names, norm_scales, apply_filter)
    return x_new, mats[keep_mask], keep_mask


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    eta_max: float = 2.0,
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Lorentz boost to each jet of the batch.

    No compensating rotation follows, so the jet azimuth is left where the boost puts it. A batch
    whose columns hold no momentum is returned unchanged, with a warning.

    :param eta_max: half-width of the uniform rapidity range the boosts are drawn from.
    :return: the boosted batch and the matrices it was boosted by, shape (jets, 4, 4).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)

    if None in (
        feature_index(names, "j1_px", "px"),
        feature_index(names, "j1_py", "py"),
        feature_index(names, "j1_pz", "pz"),
    ):
        log.warning(
            "Lorentz transform: px/py/pz are absent from the feature list, so the batch "
            "is returned untransformed."
        )
        return x.clone(), torch.empty((0, 4, 4), device=x.device, dtype=x.dtype)

    boosts = sample_lorentz_boosts(x.shape[0], eta_max, device=x.device, dtype=x.dtype)
    return apply_lorentz_batched(x, boosts, names, norm_scales), boosts


def make_transform(
    scheme: str,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 2.0,
    so3_apply_filter: bool = True,
) -> Transform:
    """Build the online transform named by *scheme*.

    The datamodule, the invariance probe and the symmetry-consistency loss all take
    their transforms from here, so a change to one group reaches all three.

    :param scheme: ``so2``, ``so3`` or ``lorentz``. ``none`` names no transform and is
        rejected: the caller decides whether to transform at all.
    :param feature_names: column names of the batches the transform will receive, in order.
    :param norm_scales: None when the batch is already physical.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts.
    :return: a callable from a ``(batch, constituents, features)`` tensor to
        ``(transformed, keep_mask)``. ``keep_mask`` is ``None`` for the groups that drop
        no jets; for filtered SO(3) it marks, over the input batch, the jets that
        survive, and the transformed batch is already subset by it.
    """
    names = list(feature_names)
    if scheme == "so2":
        return lambda x: (apply_random_so2(x, names, norm_scales=norm_scales)[0], None)
    if scheme == "lorentz":
        return lambda x: (
            apply_random_lorentz(x, names, lorentz_eta_max, norm_scales)[0],
            None,
        )
    if scheme != "so3":
        raise ValueError(f"scheme must be one of ['so2', 'so3', 'lorentz'], got {scheme!r}.")

    def so3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_sym, _, keep_mask = apply_random_so3(x, names, norm_scales, so3_apply_filter)
        return x_sym, keep_mask

    return so3
