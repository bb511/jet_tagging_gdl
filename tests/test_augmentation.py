import math

import pytest
import torch

from src.data.components.augmentation import (
    AXIS_COLUMNS,
    _apply,
    apply_random_lorentz,
    apply_random_so2,
    apply_random_so3,
    compute_eta,
    compute_phi,
    compute_pt,
    feature_index,
    make_transform,
    rebase_on_kept_axis,
    sample_haar_so3_rot_matrices,
    sample_so2_rot_matrices,
)
from src.symmetry import GROUPS
from tests.conftest import FOUR_MOMENTUM, MASS, RECORD_COLUMNS, jets

# jets() fills the leading slots of every jet and pads the rest.
N_REAL = 5
AXIS_INDICES = [feature_index(RECORD_COLUMNS, name) for name in AXIS_COLUMNS]
UNTOUCHED = [i for i in range(len(RECORD_COLUMNS)) if i not in AXIS_INDICES]


def _mass_squared(batch: torch.Tensor) -> torch.Tensor:
    return batch[..., 3] ** 2 - (batch[..., :3] ** 2).sum(-1)


def _wrapped(delta: torch.Tensor) -> torch.Tensor:
    return (delta + math.pi) % (2 * math.pi) - math.pi


def _real(batch: torch.Tensor, name: str) -> torch.Tensor:
    return batch[:, :N_REAL, feature_index(RECORD_COLUMNS, name)]


def _axis_columns(
    px: torch.Tensor,
    py: torch.Tensor,
    pz: torch.Tensor,
    d_eta: float = 0.0,
    d_phi: float = 0.0,
    pt_factor: float = 1.0,
) -> dict[int, torch.Tensor]:
    # The four jet-relative columns in radians and GeV, from the axis of the real slots.
    # The offsets displace that axis, standing in for the record's untruncated one.
    jet = [p[:, :N_REAL].sum(dim=1, keepdim=True) for p in (px, py, pz)]
    etarel = compute_eta(px, py, pz) - compute_eta(*jet) - d_eta
    phirel = _wrapped(compute_phi(px, py) - compute_phi(*jet[:2]) - d_phi)
    ptrel = compute_pt(px, py) / (pt_factor * compute_pt(*jet[:2]))
    deltaR = torch.sqrt(etarel**2 + phirel**2)
    return dict(zip(AXIS_INDICES, (etarel, phirel, deltaR, ptrel)))


def _record_batch(
    seed: int = 0, mass: float = MASS
) -> tuple[torch.Tensor, dict[str, float]]:
    # A batch in the record's layout: the cylindrical columns filled from the momenta, the
    # jet-relative four from a displaced axis, and every column divided by a scale of its
    # own, as the record stores them.
    batch = jets(n_columns=len(RECORD_COLUMNS), mass=mass, seed=seed)
    px, py, pz = batch[..., 0], batch[..., 1], batch[..., 2]
    columns = {
        feature_index(RECORD_COLUMNS, "pt"): compute_pt(px, py),
        feature_index(RECORD_COLUMNS, "eta"): compute_eta(px, py, pz),
        feature_index(RECORD_COLUMNS, "phi"): compute_phi(px, py),
    }
    columns |= _axis_columns(px, py, pz, d_eta=0.3, d_phi=0.2, pt_factor=1.1)
    for index, values in columns.items():
        batch[..., index] = values

    scales = {name: 1.5 + 0.5 * i for i, name in enumerate(RECORD_COLUMNS)}
    batch /= torch.tensor([scales[name] for name in RECORD_COLUMNS])
    batch[:, N_REAL:] = 0.0
    return batch, scales


def test_an_so2_rotation_turns_the_jet_about_the_beam() -> None:
    batch = jets()
    rotated, _ = apply_random_so2(batch, FOUR_MOMENTUM)
    matrices, thetas = sample_so2_rot_matrices(64)

    before = compute_pt(batch[..., 0], batch[..., 1])
    after = compute_pt(rotated[..., 0], rotated[..., 1])

    assert torch.allclose(before, after, atol=1e-3)
    assert torch.allclose(batch[..., 2], rotated[..., 2], atol=1e-4)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)
    assert thetas.min() >= 0.0 and thetas.max() < 2 * math.pi


def test_an_so3_rotation_leaves_the_momentum_magnitude_alone() -> None:
    batch = jets()
    rotated, _ = apply_random_so3(batch, FOUR_MOMENTUM)
    matrices = sample_haar_so3_rot_matrices(64)

    before = batch[..., :3].norm(dim=-1)

    assert torch.allclose(before, rotated[..., :3].norm(dim=-1), atol=1e-3)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)


def test_a_beam_boost_shifts_eta_rigidly_and_leaves_pt_and_phi_alone() -> None:
    # Only for a massless constituent is a beam boost a rigid shift in pseudorapidity.
    torch.manual_seed(0)
    batch = jets(mass=0.0)
    boosted, xi = apply_random_lorentz(batch, FOUR_MOMENTUM, eta_max=1.0)

    real = slice(None, 5)
    d_eta = (
        compute_eta(boosted[..., 0], boosted[..., 1], boosted[..., 2])
        - compute_eta(batch[..., 0], batch[..., 1], batch[..., 2])
    )[:, real]

    assert torch.allclose(
        compute_pt(boosted[..., 0], boosted[..., 1]),
        compute_pt(batch[..., 0], batch[..., 1]),
    )
    assert torch.allclose(
        compute_phi(boosted[..., 0], boosted[..., 1]),
        compute_phi(batch[..., 0], batch[..., 1]),
    )
    assert torch.allclose(d_eta, -xi.unsqueeze(1).expand_as(d_eta), atol=1e-4)
    assert xi.abs().max() <= 1.0
    # Drawn per jet, so two jets are not shifted by the same amount.
    assert (xi - xi[0]).abs().max() > 1e-3


@pytest.mark.parametrize("isotropic", [False, True])
def test_a_lorentz_boost_leaves_the_masses_alone_and_the_padding_at_zero(
    isotropic: bool,
) -> None:
    batch = jets(n_slots=8, n_real=5)
    boosted, _ = apply_random_lorentz(batch, FOUR_MOMENTUM, 2.0, None, isotropic)

    assert torch.allclose(
        _mass_squared(boosted)[:, :5], _mass_squared(batch)[:, :5], rtol=1e-4
    )
    # A padded slot that acquired an energy would be read as a constituent downstream.
    assert torch.equal(boosted[:, 5:, :], torch.zeros_like(boosted[:, 5:, :]))


def test_the_factory_builds_one_transform_per_group_and_refuses_none() -> None:
    batch = jets()
    for scheme in GROUPS:
        assert make_transform(scheme, FOUR_MOMENTUM)(batch).shape == batch.shape

    # 'none' is a legitimate value of data.augmentation_scheme, but not a transform.
    with pytest.raises(ValueError, match="lorentz_iso"):
        make_transform("none", FOUR_MOMENTUM)


def test_the_returned_draws_are_the_ones_the_batch_was_transformed_by() -> None:
    batch = jets()
    boosted, draw = apply_random_lorentz(batch, FOUR_MOMENTUM, 1.0, None, True)

    # The boost of rapidity xi along the unit axis n, rebuilt from the stored pair.
    xi, n = draw[:, 0], draw[:, 1:]
    rebuilt = torch.eye(4).repeat(6, 1, 1)
    rebuilt[:, 0, 0] = torch.cosh(xi)
    rebuilt[:, 0, 1:] = -torch.sinh(xi)[:, None] * n
    rebuilt[:, 1:, 0] = -torch.sinh(xi)[:, None] * n
    outer = n[:, :, None] * n[:, None, :]
    rebuilt[:, 1:, 1:] += (torch.cosh(xi) - 1)[:, None, None] * outer
    replayed = _apply(batch, rebuilt, FOUR_MOMENTUM, None)
    assert torch.allclose(replayed, boosted, atol=1e-4)


def test_the_rebase_moves_the_axis_columns_onto_the_kept_constituents() -> None:
    batch, scales = _record_batch()
    px, py, pz = (batch[..., i] * scales[RECORD_COLUMNS[i]] for i in (0, 1, 2))
    expected = _axis_columns(px, py, pz)

    rebased = rebase_on_kept_axis(batch, RECORD_COLUMNS, scales)

    for index, values in expected.items():
        scaled = values[:, :N_REAL] / scales[RECORD_COLUMNS[index]]
        assert torch.allclose(rebased[:, :N_REAL, index], scaled, atol=1e-5)
    assert torch.equal(rebased[..., UNTOUCHED], batch[..., UNTOUCHED])
    assert torch.equal(rebased[:, N_REAL:], batch[:, N_REAL:])
    # Idempotent: the second pass finds the axis the columns already refer to.
    assert torch.equal(rebase_on_kept_axis(rebased, RECORD_COLUMNS, scales), rebased)


def test_after_the_rebase_the_identity_reproduces_every_column_it_rewrites() -> None:
    # Without it the two definitions of the jet axis disagree: by 0.3 in eta here, and on
    # 70 to 85 % of the entries of the record at 16 constituents.
    batch, scales = _record_batch()
    rebased = rebase_on_kept_axis(batch, RECORD_COLUMNS, scales)
    identity = torch.eye(4).expand(rebased.shape[0], 4, 4)

    applied = _apply(rebased, identity, RECORD_COLUMNS, scales)

    assert torch.allclose(applied, rebased, atol=1e-5)


@pytest.mark.parametrize(
    "scheme, mass, unchanged, moved",
    [
        ("so2", MASS, ("pt", "eta", "etarel", "phirel"), ("phi",)),
        # A beam boost shifts the pseudorapidity rigidly only for a massless constituent,
        # and the axis etarel is measured from is massive either way, so etarel moves too.
        ("lorentz", 0.0, ("pt", "phi", "phirel"), ("eta",)),
    ],
)
def test_the_rebased_columns_move_under_a_group_only_where_they_must(
    scheme: str, mass: float, unchanged: tuple[str, ...], moved: tuple[str, ...]
) -> None:
    torch.manual_seed(0)
    batch, scales = _record_batch(mass=mass)
    rebased = rebase_on_kept_axis(batch, RECORD_COLUMNS, scales)

    transformed = make_transform(scheme, RECORD_COLUMNS, scales)(rebased)

    for name in unchanged:
        before, after = _real(rebased, name), _real(transformed, name)
        assert torch.allclose(before, after, atol=1e-4), name
    for name in moved:
        shift = (_real(transformed, name) - _real(rebased, name)).abs().max()
        assert shift > 1e-3, name
