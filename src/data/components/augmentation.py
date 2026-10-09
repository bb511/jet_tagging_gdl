"""Transforms of the constituent four-vectors of a jet using a group action.

A transform takes a ``(jets, constituents, features)`` tensor and returns one of the
same shape and normalisation: the columns are taken back to GeV by ``norm_scales``,
acted on by one matrix per jet, and written back. Four-vectors are ordered [E, px, py,
pz] and the beam is the z axis.

A batch must contain px, py, pz and e, bare or with the record's ``j1_`` prefix.
Whichever of pt, eta, phi, etarel, phirel, deltaR and ptrel it has are rewritten,
and erel, etarot, phirot, costheta and costhetarel are left as found. Therefore, a
feature list holding those would have stale values. Padded constituents are all-zero
rows: the transform does not apply to these.

Of the rewritten columns, etarel, phirel, deltaR and ptrel are measured from the jet
axis. This is computed based on the constituent values present in the data.
"""

import math
from collections.abc import Callable, Sequence
from functools import partial

import torch

from src.symmetry import GROUPS

Transform = Callable[[torch.Tensor], torch.Tensor]

# The columns measured from the jet axis.
AXIS_COLUMNS = ("etarel", "phirel", "deltaR", "ptrel")


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


def feature_index(feature_names: Sequence[str], name: str) -> int | None:
    """Return the feature of *name*, bare or ``j1_``-prefixed, or None."""
    names = list(feature_names)
    return next((names.index(c) for c in (name, f"j1_{name}") if c in names), None)


def resolve_feature_indices(
    model_features: list[str], feature_names: list[str]
) -> list[int]:
    """Return the index of each of *model_features* in *feature_names*, matched by
    `feature_index`.

    :raises ValueError: if a feature is absent, bare and prefixed.
    """
    indices = [feature_index(feature_names, name) for name in model_features]
    missing = [name for name, index in zip(model_features, indices) if index is None]
    if missing:
        raise ValueError(
            f"Feature {missing[0]!r} is absent from the data columns {feature_names}. "
            f"data.features must contain {', '.join(model_features)}."
        )
    return indices


def get_norm_scale(norm_scales: dict[str, float] | None, name: str) -> float:
    """Return the factor taking the stored feature of *name* back to GeV."""
    scales = norm_scales or {}
    return float(scales.get(name, scales.get(f"j1_{name}", 1.0)))


def _matrices(rows: list[list[torch.Tensor]]) -> torch.Tensor:
    """Stack nine per-sample entries, row by row, into matrices."""
    return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=-2)


def sample_so2_rot_matrices(
    nbr_rot: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw *nbr_rot* rotations about the beam, shape (nbr_rot, 3, 3), and their angles.

    The angle is uniform on the full turn, [0, 2 pi) since the azimuth of a pp collision
    does not contain physics information.
    """
    thetas = torch.rand(max(nbr_rot, 0), device=device) * (2.0 * math.pi)
    cos, sin = torch.cos(thetas), torch.sin(thetas)
    zero, one = torch.zeros_like(thetas), torch.ones_like(thetas)
    return _matrices([[cos, -sin, zero], [sin, cos, zero], [zero, zero, one]]), thetas


def sample_haar_so3_rot_matrices(
    num_rotations: int, device: torch.device | None = None
) -> torch.Tensor:
    """Draw *num_rotations* Haar-random rotations, shape (num_rotations, 3, 3).

    A Gaussian in R^4 is isotropic: the normalised draws are uniform on the unit
    quaternions. This is just the Haar measure on SO(3). The clamp forbids the all-zero
    draw.

    The draw is not restricted to the detector: about 3.5 % of the rotated jets have a
    constituent beyond ``|eta| = 3``, accepted because SO(3) is the control for a group
    that is not a symmetry of the data.
    """
    quat = torch.randn(max(num_rotations, 0), 4, device=device)
    x, y, z, w = (quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-8)).unbind(dim=1)
    rows = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return _matrices(rows)


def _sample_boosts(
    nbr_rot: int, device: torch.device | None, eta_max: float, isotropic: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one boost per jet, shape (nbr_rot, 4, 4), acting on [E, px, py, pz].

    The rapidity is drawn uniform on [-eta_max, eta_max]. Along the beam this is the
    symmetry of a pp collision. We also draw isotropic Lorentz rotations, which are not
    a symmetry of the data but are used later to test models that claim to be fully
    Lorentz invariant or equivariant. Boosting without caring about the detector bounds
    creates 0.2% jets that would not be seen by the detector normally.

    :return: the boosts and the draw: the rapidity, shape (nbr_rot,), or under
        *isotropic* the rapidity followed by the unit axis, shape (nbr_rot, 4).
    """
    beam = torch.eye(3, device=device)[2].expand(nbr_rot, 3)
    axis = torch.randn(nbr_rot, 3, device=device) if isotropic else beam

    # The clamp guards an all-zero draw.
    axis = axis / axis.norm(dim=1, keepdim=True).clamp(min=1e-8)
    xi = (torch.rand(nbr_rot, device=device) * 2.0 - 1.0) * eta_max
    boosts = torch.eye(4, device=device).repeat(nbr_rot, 1, 1)
    boosts[:, 0, 0] = torch.cosh(xi)
    boosts[:, 0, 1:] = -torch.sinh(xi).unsqueeze(1) * axis
    boosts[:, 1:, 0] = -torch.sinh(xi).unsqueeze(1) * axis
    boosts[:, 1:, 1:] += (
        (torch.cosh(xi) - 1)[:, None, None] * axis[:, :, None] * axis[:, None, :]
    )

    return boosts, torch.cat([xi[:, None], axis], dim=1) if isotropic else xi


def _read_four_momenta(
    x: torch.Tensor, names: Sequence[str], norm_scales: dict[str, float] | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [E, px, py, pz] in GeV and the real-constituent mask.

    :raises ValueError: if the batch lacks one of the four required features.
    """
    indices = {key: feature_index(names, key) for key in ("e", "px", "py", "pz")}
    if None in indices.values():
        raise ValueError(
            f"A transform needs the columns {list(indices)}, got {list(names)}."
        )
    p4 = [
        x[..., index] * get_norm_scale(norm_scales, key)
        for key, index in indices.items()
    ]
    return torch.stack(p4, dim=-1), compute_pt(p4[1], p4[2]) > 0


def _derived_columns(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, active: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The features that follow from the momenta, in GeV and radians.

    The jet axis is the sum over the real constituents; phirel is wrapped to [-pi, pi)
    and ptrel is the constituent's share of the jet pt.
    """
    pt, eta, phi = compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py)
    jet_px, jet_py, jet_pz = (
        (p * active).sum(dim=1, keepdim=True) for p in (px, py, pz)
    )
    etarel = eta - compute_eta(jet_px, jet_py, jet_pz)
    phirel = (phi - compute_phi(jet_px, jet_py) + math.pi) % (2 * math.pi) - math.pi
    # A jet of padded slots has no pt to share out; the clamp is in GeV.
    ptrel = pt / compute_pt(jet_px, jet_py).clamp(min=1e-8)
    columns = {"pt": pt, "eta": eta, "phi": phi, "etarel": etarel, "phirel": phirel}
    return columns | {"deltaR": torch.sqrt(etarel**2 + phirel**2), "ptrel": ptrel}


def _write(
    x: torch.Tensor,
    names: Sequence[str],
    norm_scales: dict[str, float] | None,
    columns: dict[str, torch.Tensor],
    where: torch.Tensor,
) -> torch.Tensor:
    """Return *x* with each feature in *columns* replaced where *where* is True."""
    x_new = x.clone()
    for name, values in columns.items():
        index = feature_index(names, name)
        if index is not None:
            scaled = values / get_norm_scale(norm_scales, name)
            x_new[..., index] = torch.where(where, scaled, x[..., index])
    return x_new


def _apply(
    x: torch.Tensor,
    mats: torch.Tensor,
    names: Sequence[str],
    norm_scales: dict[str, float] | None,
) -> torch.Tensor:
    """Act on every jet with its own matrix and rewrite the columns that follow from it.

    :param mats: shape (jets, 4, 4) on [E, px, py, pz], or (jets, 3, 3) on the momenta
        alone.
    :return: the new batch. A padded slot keeps its zeros, and so does a constituent
        left with no positive energy.
    """
    if mats.shape[-1] == 3:
        # A rotation leaves the energy alone: pad it to the block-diagonal 4 x 4 form.
        mats = torch.nn.functional.pad(mats, (1, 0, 1, 0))
        mats[:, 0, 0] = 1.0
    p4, active = _read_four_momenta(x, names, norm_scales)
    e, px, py, pz = torch.einsum("bij,bcj->bci", mats.to(p4.dtype), p4).unbind(dim=-1)
    columns = {"e": e, "px": px, "py": py, "pz": pz} | _derived_columns(
        px, py, pz, active
    )
    return _write(x, names, norm_scales, columns, active & (e > 0))


def rebase_on_kept_axis(
    x: torch.Tensor,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
) -> torch.Tensor:
    """Return *x* with `AXIS_COLUMNS` measured from the present constituents.

    The record stores those four against the untruncated jet's axis.
    A transform recomputes them from the present constituents.
    Everything else, the momenta and the absolute pt, eta and phi among them,
    is returned untouched.
    """
    p4, active = _read_four_momenta(x, feature_names, norm_scales)
    _, px, py, pz = p4.unbind(dim=-1)
    columns = _derived_columns(px, py, pz, active)
    kept = {name: columns[name] for name in AXIS_COLUMNS}
    return _write(x, feature_names, norm_scales, kept, active)


# Per group of src.symmetry.GROUPS, (jets, device, eta_max) -> the matrices acting on
# each jet and the draws behind them; eta_max reaches the boosts alone. A rotation of
# SO(3) is stored as its own matrix.
_SAMPLERS = {
    "so2": lambda n, device, eta_max: sample_so2_rot_matrices(n, device),
    "so3": lambda n, device, eta_max: (sample_haar_so3_rot_matrices(n, device),) * 2,
    "lorentz": partial(_sample_boosts, isotropic=False),
    "lorentz_iso": partial(_sample_boosts, isotropic=True),
}


def make_drawing_transform(
    scheme: str,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 1.0,
) -> Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Build the transform named by *scheme*, returning its draws beside the batch.

    The invariance probe calls this, so that every transformed jet can be traced back to
    its draw: the angle about the beam on [0, 2 pi), shape (jets,), for ``so2``; the
    rotation matrix, shape (jets, 3, 3), for ``so3``; the rapidity, shape (jets,), for
    ``lorentz``; the rapidity and the unit axis, shape (jets, 4), for ``lorentz_iso``.
    The arguments are those of `make_transform`, which draws the same numbers.
    """
    if scheme not in _SAMPLERS:
        raise ValueError(
            f"Transformation scheme must be one of {list(GROUPS)}, got {scheme!r}..."
        )
    names, sample = list(feature_names), _SAMPLERS[scheme]

    def drawing(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mats, draw = sample(x.shape[0], x.device, lorentz_eta_max)
        return _apply(x, mats, names, norm_scales), draw

    return drawing


def make_transform(
    scheme: str,
    feature_names: Sequence[str],
    norm_scales: dict[str, float] | None = None,
    lorentz_eta_max: float = 1.0,
) -> Transform:
    """Build the online transform named by *scheme*.

    The datamodule and the symmetry-consistency loss call this method for applying
    transformations; the invariance probe calls `make_drawing_transform`, which draws
    the same numbers and returns them too.

    :param scheme: one of `src.symmetry.GROUPS`, the boosts being along the beam
        (``lorentz``) or in a random direction (``lorentz_iso``). ``none`` is rejected.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts.
    :return: a callable from a ``(jets, constituents, features)`` tensor to its
        transformed batch of the same shape and normalisation.
    """
    drawing = make_drawing_transform(
        scheme, feature_names, norm_scales, lorentz_eta_max
    )
    return lambda x: drawing(x)[0]
