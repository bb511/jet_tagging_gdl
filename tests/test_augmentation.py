import math

import pytest
import torch

from src.data.components.augmentation import (
    apply_random_lorentz,
    apply_random_so2,
    apply_random_so3,
    compute_eta,
    compute_phi,
    compute_pt,
    feature_index,
    get_norm_scale,
    make_transform,
    sample_haar_so3_rot_matrices,
    sample_so2_rot_matrices,
)
from src.symmetry import GROUPS
from tests.conftest import FOUR_MOMENTUM, RECORD_COLUMNS, jets


def _mass_squared(batch: torch.Tensor) -> torch.Tensor:
    return batch[..., 3] ** 2 - (batch[..., :3] ** 2).sum(-1)


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
    boosted, boosts = apply_random_lorentz(batch, FOUR_MOMENTUM, eta_max=1.0)
    # cosh(xi) sits on the (E, E) entry and -sinh(xi) on (E, pz).
    xi = torch.atanh(-boosts[:, 0, 3] / boosts[:, 0, 0])

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


def test_a_column_is_found_by_either_name_and_scaled_by_its_own_factor() -> None:
    assert feature_index(RECORD_COLUMNS, "px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "j1_px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "nonesuch") is None
    # A batch built by hand, and the record's own before it is prepared, is already in GeV.
    assert get_norm_scale(None, "px") == 1.0
    assert get_norm_scale({"j1_px": 3.0}, "px") == 3.0


def test_the_factory_builds_one_transform_per_group_and_refuses_none() -> None:
    batch = jets()
    for scheme in GROUPS:
        assert make_transform(scheme, FOUR_MOMENTUM)(batch).shape == batch.shape

    # 'none' is a legitimate value of data.augmentation_scheme, but not a transform.
    with pytest.raises(ValueError, match="lorentz_iso"):
        make_transform("none", FOUR_MOMENTUM)
