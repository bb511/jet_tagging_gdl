"""Transforms of the constituent four-vectors of a jet.

Functions take and return ``(jets, constituents, features)`` tensors on the device of the input.
Matrices are built in float32 and cast back at the end, one drawn per jet and applied through a
batched einsum, ``"bij,bcj->bci"`` over jets, constituents and momentum components. Four-vectors
are ordered [E, px, py, pz] wherever one appears, and the beam is the z axis.

Feature columns are addressed by name, bare or with the record's ``j1_`` prefix, and every
transform first undoes the normalisation with ``norm_scales``, the per-column factors back to GeV,
since the acceptance cuts and the boost rapidities are defined in physical units (GeV, radians).
Padded constituents are all-zero rows, which every transform leaves at zero.

The transforms rewrite px, py, pz, e and whichever of pt, eta, phi, etarel, phirel, deltaR and
ptrel a batch carries. They leave erel, etarot, phirot, costheta and costhetarel as they found
them, so a feature list holding those columns would carry stale values after a transform.

Transforms draw from the global torch RNG, so ``lightning.seed_everything`` fixes them.
"""

import math
from collections.abc import Callable, Sequence

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

Transform = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]]

# Detector acceptance the SO(3) filter enforces, in the un-normalised units the constituent
# columns carry: GeV for the transverse momentum, radians for phirel, and the dimensionless
# pseudorapidity for eta and etarel.
SO3_PT_MAX = 1600.0
SO3_ETA_MAX = 3.0
SO3_ETAREL_MAX = 0.5
SO3_PHIREL_MAX = 0.5

# The momentum columns every transform rewrites, and without which it cannot run at all.
MOMENTUM = ("px", "py", "pz")


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


def _column_index(feature_names: list[str], name: str) -> int | None:
    """Column of *name*, the record's ``j1_``-prefixed spelling preferred over the bare one."""
    return feature_index(feature_names, f"j1_{name}", name)


def _matrices_from_rows(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    """Stack nine per-sample entries, row by row, into matrices of shape (samples, 3, 3)."""
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def sample_so2_rot_matrices(
    nbr_rot: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample *nbr_rot* SO(2) rotation matrices about the beam (z) axis, one per jet.

    The angle is uniform on the full turn, [0, 2*pi), which is the only range the experiments draw
    from: the azimuth of a pp collision carries no physics.

    :return: matrices of shape (nbr_rot, 3, 3) and the angles they turn through, (nbr_rot,), in
        radians.
    """
    # A count of zero or below leaves thetas empty, hence matrices of shape (0, 3, 3).
    thetas = torch.rand(max(nbr_rot, 0), device=device, dtype=torch.float32) * (2.0 * math.pi)
    cos, sin = torch.cos(thetas), torch.sin(thetas)
    zero, one = torch.zeros_like(thetas), torch.ones_like(thetas)
    rows = [[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]

    return _matrices_from_rows(rows).to(dtype), thetas.to(dtype)


def _quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrix of each unit quaternion, shape (quaternions, 3, 3).

    :param quat: unit quaternions of shape (quaternions, 4), ordered (x, y, z, w).
    """
    x, y, z, w = quat.unbind(dim=1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rows = [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]

    return _matrices_from_rows(rows)


def sample_haar_so3_rot_matrices(
    num_rotations: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create Haar-random SO(3) matrices from unit quaternions, shape (num_rotations, 3, 3)."""
    if num_rotations <= 0:
        return torch.empty((0, 3, 3), dtype=dtype, device=device)

    # A Gaussian in R^4 is isotropic, so the normalised draws are uniform on the unit sphere,
    # which is the Haar measure on SO(3). The clamp guards the all-zero draw.
    quat = torch.randn(num_rotations, 4, device=device, dtype=torch.float32)
    quat = quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return _quaternion_to_matrix(quat).to(dtype)


def so3_jet_filter(
    pt: torch.Tensor,
    eta: torch.Tensor,
    etarel: torch.Tensor,
    phirel: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    """Mark the jets whose constituents all pass the SO(3) detector-acceptance cuts.

    The constituent quantities are of shape (jets, constituents) and, like the ``SO3_*`` thresholds
    above, in physical un-normalised units, so no threshold ever has to be rescaled by a norm
    scale. *etarel* and *phirel* are measured from the jet axis, *phirel* wrapped to [-pi, pi).

    :param active: True for a real constituent; a padded slot never fails a cut.
    :return: mask of shape (jets,), True where every real constituent passes.
    """
    passes = (
        (pt >= 0)
        & (pt <= SO3_PT_MAX)
        & (eta.abs() <= SO3_ETA_MAX)
        & (etarel.abs() <= SO3_ETAREL_MAX)
        & (phirel.abs() <= SO3_PHIREL_MAX)
    )

    return (~active | passes).all(dim=1)


def sample_beam_boosts(
    nbr_rot: int,
    eta_max: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample *nbr_rot* Lorentz boosts along the beam (z) axis, one per jet.

    The rapidity is uniform on ``[-eta_max, eta_max]``, drawn per jet and independently of it. This
    is the symmetry of a pp collision, whose longitudinal boost is unknown event by event: for a
    massless constituent it shifts the pseudorapidity by ``-xi`` and leaves pt and phi where they
    were.

    :return: matrices of shape (nbr_rot, 4, 4), acting on [E, px, py, pz].
    """
    if nbr_rot <= 0:
        return torch.empty((0, 4, 4), dtype=dtype, device=device)

    xi = (torch.rand(nbr_rot, dtype=torch.float32, device=device) * 2.0 - 1.0) * eta_max
    boosts = torch.eye(4, dtype=torch.float32, device=device).repeat(nbr_rot, 1, 1)
    boosts[:, 0, 0] = torch.cosh(xi)
    boosts[:, 3, 3] = torch.cosh(xi)
    boosts[:, 0, 3] = -torch.sinh(xi)
    boosts[:, 3, 0] = -torch.sinh(xi)

    return boosts.to(dtype)


def _boost_matrices(beta: torch.Tensor) -> torch.Tensor:
    """Return the boost of each velocity, shape (boosts, 4, 4), acting on [E, px, py, pz].

    :param beta: velocities in units of c, shape (boosts, 3).
    """
    beta2 = (beta**2).sum(dim=1)
    # 1e-8 keeps gamma finite as |beta| approaches 1; 1e-12 keeps the (gamma - 1) / beta^2 factor
    # of the spatial block finite at rest, where the identity is written over it below anyway.
    gamma = 1.0 / torch.sqrt((1.0 - beta2).clamp(min=1e-8))
    spatial = ((gamma - 1.0) / beta2.clamp(min=1e-12)).view(-1, 1, 1)
    eye = torch.eye(4, dtype=beta.dtype, device=beta.device)
    boosts = eye.repeat(beta.shape[0], 1, 1)
    boosts[:, 0, 0] = gamma
    boosts[:, 0, 1:] = -gamma.unsqueeze(1) * beta
    boosts[:, 1:, 0] = -gamma.unsqueeze(1) * beta
    boosts[:, 1:, 1:] += spatial * torch.bmm(beta[:, :, None], beta[:, None, :])
    boosts[beta2 < 1e-12] = eye

    return boosts


def sample_isotropic_boosts(
    nbr_rot: int,
    eta_max: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample *nbr_rot* Lorentz boosts of isotropic direction, one per jet.

    The rapidity is uniform on ``[-eta_max, eta_max]``, as for the beam boost. An isotropic boost
    is not a symmetry of a pp collision, where only boosts along the beam are: it probes the full
    group, that is a property of the model rather than one of the data.

    :return: matrices of shape (nbr_rot, 4, 4), acting on [E, px, py, pz].
    """
    if nbr_rot <= 0:
        return torch.empty((0, 4, 4), dtype=dtype, device=device)

    direction = torch.randn(nbr_rot, 3, dtype=torch.float32, device=device)
    # The clamp guards the all-zero draw, whose direction is undefined.
    direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-8)
    eta = (torch.rand(nbr_rot, dtype=torch.float32, device=device) * 2.0 - 1.0) * eta_max

    return _boost_matrices(torch.tanh(eta).unsqueeze(1) * direction).to(dtype)


def _require_feature_names(feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is None:
        raise ValueError("feature_names must be provided to apply a transform.")
    return list(feature_names)


def _has_momentum(feature_names: list[str]) -> bool:
    """Whether the batch carries px, py and pz, without which no transform can run."""
    return all(_column_index(feature_names, name) is not None for name in MOMENTUM)


def _physical_momenta(
    x: torch.Tensor, feature_names: list[str], norm_scales: dict[str, float] | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return px, py and pz in GeV, each of shape (jets, constituents).

    The caller must have checked :func:`_has_momentum` first.
    """
    px, py, pz = (
        x[:, :, _column_index(feature_names, name)] * get_norm_scale(norm_scales, name)
        for name in MOMENTUM
    )

    return px, py, pz


def _active_mask(px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    """True for a constituent carrying transverse momentum; a padded slot is an all-zero row."""
    return (px**2 + py**2) > 0


def _all_jets(x: torch.Tensor) -> torch.Tensor:
    """An acceptance mask of shape (jets,) that keeps every jet of the batch."""
    return torch.ones(x.shape[0], dtype=torch.bool, device=x.device)


def _jet_columns(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, active: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Every column a rotation or a boost implies, from the momenta it left behind, in GeV.

    The jet axis is the sum over the real constituents, so *active* has to exclude the padded
    slots; phirel is wrapped to [-pi, pi) and ptrel is the constituent's share of the jet pt.
    """
    pt, eta, phi = compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py)
    jet_px, jet_py, jet_pz = ((p * active).sum(dim=1, keepdim=True) for p in (px, py, pz))
    jet_pt = compute_pt(jet_px, jet_py)
    etarel = eta - compute_eta(jet_px, jet_py, jet_pz)
    phirel = wrap_phi(phi - compute_phi(jet_px, jet_py))
    columns = {"pt": pt, "eta": eta, "phi": phi, "etarel": etarel, "phirel": phirel}
    columns["deltaR"] = torch.sqrt(etarel**2 + phirel**2)
    # A jet of padded slots alone has no transverse momentum to share out; the clamp is in GeV.
    columns["ptrel"] = pt / jet_pt.clamp(min=1e-8)

    return columns


def _write_columns(
    x: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None,
    columns: dict[str, torch.Tensor],
    where: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return *x* with each named column replaced by its value, renormalised.

    A column the batch does not carry is skipped, so a caller may offer every quantity it derived
    and let the feature list decide which of them are stored.

    :param columns: physical values in GeV or radians, keyed by bare column name, each of shape
        (jets, constituents).
    :param where: True where the new value is written, the stored value standing elsewhere; None
        writes everywhere.
    """
    x_new = x.clone()
    for name, values in columns.items():
        index = _column_index(feature_names, name)
        if index is None or index >= x.shape[2]:
            continue
        scaled = values / get_norm_scale(norm_scales, name)
        stored = x[:, :, index]
        x_new[:, :, index] = scaled if where is None else torch.where(where, scaled, stored)

    return x_new


def apply_so2_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Rotate every jet of the batch about the beam axis by its own matrix.

    :param mats: one rotation per jet, shape (jets, 3, 3).
    :return: the rotated batch, of the shape and the normalisation of *x*. Whichever of px, py, pz,
        pt and phi it carries are rewritten; every other column is left alone, since a turn about
        the beam moves neither the pseudorapidities nor the quantities relative to the jet axis.
    """
    if not _has_momentum(feature_names):
        return x.clone()

    vec = torch.stack(_physical_momenta(x, feature_names, norm_scales), dim=-1)
    px, py, pz = torch.einsum("bij,bcj->bci", mats, vec).unbind(dim=-1)
    columns = {"px": px, "py": py, "pz": pz}
    columns |= {"pt": compute_pt(px, py), "phi": compute_phi(px, py)}

    return _write_columns(x, feature_names, norm_scales, columns)


def _acceptance_mask(
    x: torch.Tensor, columns: dict[str, torch.Tensor], active: torch.Tensor, apply_filter: bool
) -> torch.Tensor:
    """The jets an SO(3) rotation leaves in acceptance, or all of them with the filter off."""
    if not apply_filter:
        return _all_jets(x)

    return so3_jet_filter(
        columns["pt"], columns["eta"], columns["etarel"], columns["phirel"], active
    )


def apply_so3_batched(
    x: torch.Tensor,
    mats: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None = None,
    apply_filter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate every jet of the batch in three dimensions by its own matrix.

    :param mats: one rotation per jet, shape (jets, 3, 3).
    :param apply_filter: off by default, so that SO(3) is learned and measured under the Haar
        measure. Switched on, it drops the jets a rotation carries outside detector acceptance
        rather than reverting them, since reverting would make T(x) equal x for them and deflate
        every invariance metric measured against the rotated batch.
    :return: the surviving jets alone, and a mask over the input batch, so the caller can subset
        the labels, and anything else aligned with that batch, the same way.
    """
    if not _has_momentum(feature_names):
        return x.clone(), _all_jets(x)

    momenta = _physical_momenta(x, feature_names, norm_scales)
    active = _active_mask(momenta[0], momenta[1])
    vec = torch.stack(momenta, dim=-1) * active.unsqueeze(-1)
    px, py, pz = torch.einsum("bij,bcj->bci", mats, vec).unbind(dim=-1)
    columns = {"px": px, "py": py, "pz": pz} | _jet_columns(px, py, pz, active)
    keep_mask = _acceptance_mask(x, columns, active, apply_filter)

    return _write_columns(x, feature_names, norm_scales, columns, active)[keep_mask], keep_mask


def _energy(
    x: torch.Tensor,
    feature_names: list[str],
    norm_scales: dict[str, float] | None,
    momenta: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """The energy column in GeV, or the massless momentum magnitude if the batch carries none."""
    index = _column_index(feature_names, "e")
    if index is None or index >= x.shape[2]:
        return torch.sqrt(sum(p**2 for p in momenta))

    return x[:, :, index] * get_norm_scale(norm_scales, "e")


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
    if not _has_momentum(feature_names):
        return x.clone()

    momenta = _physical_momenta(x, feature_names, norm_scales)
    active = _active_mask(momenta[0], momenta[1])
    p4 = torch.stack([_energy(x, feature_names, norm_scales, momenta), *momenta], dim=-1)
    boosted = torch.einsum("bij,bcj->bci", boosts, p4 * active.unsqueeze(-1))
    e, px, py, pz = (component * active for component in boosted.unbind(dim=-1))
    columns = {"px": px, "py": py, "pz": pz, "e": e} | _jet_columns(px, py, pz, active)

    return _write_columns(x, feature_names, norm_scales, columns, (e > 0) & active)


def apply_random_so2(
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    norm_scales: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled SO(2) rotation to each jet of the batch.

    :return: the rotated batch and the angles it was turned through, shape (jets,), in radians.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    mats, thetas = sample_so2_rot_matrices(x.shape[0], device=x.device, dtype=x.dtype)

    return apply_so2_batched(x, mats, names, norm_scales), thetas


def apply_random_so3(
    x: torch.Tensor,
    feature_names: Sequence[str] | None = None,
    norm_scales: dict[str, float] | None = None,
    apply_filter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Haar-random SO(3) rotation to each jet of the batch.

    :param apply_filter: off by default. Switched on, it drops the jets a rotation carries outside
        the acceptance bounds of :func:`so3_jet_filter`, rather than reverting them to their
        unrotated values.
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
    eta_max: float = 1.0,
    norm_scales: dict[str, float] | None = None,
    isotropic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply an independently sampled Lorentz boost to each jet of the batch.

    A batch whose columns hold no momentum is returned unchanged, with a warning.

    :param eta_max: half-width of the uniform rapidity range of the boosts, along the beam for
        ``lorentz``, in a random direction for ``lorentz_iso``.
    :param isotropic: boost in a random direction rather than along the beam. No compensating
        rotation follows, so the jet azimuth is left where such a boost puts it.
    :return: the boosted batch and the matrices it was boosted by, shape (jets, 4, 4).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    names = _require_feature_names(feature_names)
    if not _has_momentum(names):
        log.warning("Lorentz transform: px/py/pz are absent, so the batch is not transformed.")
        return x.clone(), torch.empty((0, 4, 4), device=x.device, dtype=x.dtype)

    sampler = sample_isotropic_boosts if isotropic else sample_beam_boosts
    boosts = sampler(x.shape[0], eta_max, device=x.device, dtype=x.dtype)

    return apply_lorentz_batched(x, boosts, names, norm_scales), boosts


def _lorentz_transform(
    names: list[str],
    norm_scales: dict[str, float] | None,
    eta_max: float,
    isotropic: bool,
) -> Transform:
    """A boost per jet, along the beam or in a random direction, dropping no jet."""
    return lambda x: (apply_random_lorentz(x, names, eta_max, norm_scales, isotropic)[0], None)


def _so3_transform(
    names: list[str], norm_scales: dict[str, float] | None, apply_filter: bool
) -> Transform:
    """A Haar-random rotation per jet, with the acceptance filter optional."""

    def so3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_sym, _, keep_mask = apply_random_so3(x, names, norm_scales, apply_filter)
        # Unfiltered, every jet survives; None then says so, and spares the callers a no-op
        # subset and the probe a constant-zero rejection fraction.
        return x_sym, keep_mask if apply_filter else None

    return so3


def make_transform(
    scheme: str,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 1.0,
    so3_apply_filter: bool = False,
) -> Transform:
    """Build the online transform named by *scheme*.

    The datamodule, the invariance probe and the symmetry-consistency loss all take their
    transforms from here, so a change to one group reaches all three.

    :param scheme:``so2``, ``so3``, ``lorentz`` (along the beam) or ``lorentz_iso`` (isotropic
        direction). ``none`` names no transform and is rejected: the caller decides whether to
        transform at all.
    :param feature_names: column names of the batches the transform will receive, in order.
    :param norm_scales: None when the batch is already physical.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts, along the beam
        for ``lorentz``, in a random direction for ``lorentz_iso``.
    :param so3_apply_filter: off by default, so no scheme drops a jet.
    :return: a callable from a ``(jets, constituents, features)`` tensor to ``(transformed,
        keep_mask)``. ``keep_mask`` is ``None`` unless the SO(3) filter is on, in which case it
        marks, over the input batch, the jets that survive, and the transformed batch is already
        subset by it.
    """
    names = list(feature_names)
    if scheme == "so2":
        return lambda x: (apply_random_so2(x, names, norm_scales)[0], None)
    if scheme == "so3":
        return _so3_transform(names, norm_scales, so3_apply_filter)
    if scheme in ("lorentz", "lorentz_iso"):
        return _lorentz_transform(names, norm_scales, lorentz_eta_max, scheme == "lorentz_iso")
    raise ValueError(
        f"scheme must be one of ['so2', 'so3', 'lorentz', 'lorentz_iso'], got {scheme!r}."
    )
