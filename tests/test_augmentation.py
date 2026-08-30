"""The transforms must move the jets the way the symmetry says, and nothing else.

Every check is a conserved quantity of the group being applied: the transverse momentum and the
longitudinal component under SO(2), the momentum magnitude under SO(3), the invariant mass under a
boost. A transform that quietly wrote the wrong column would break one of them, which no accuracy
number ever would.
"""

import math

import pytest
import torch

from src.data.components.augmentation import (
    apply_random_lorentz,
    apply_random_so2,
    apply_random_so3,
    compute_pt,
    feature_index,
    get_norm_scale,
    make_transform,
    sample_haar_so3_rot_matrices,
    sample_so2_rot_matrices,
)

# The record's own column order; only the four-momentum is addressed by the transforms.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
FOUR_MOMENTUM = ["px", "py", "pz", "e"]

# Constituents of about 20 GeV per component carrying a 20 GeV mass. A massless
# constituent would put m^2 = E^2 - |p|^2 at the level of the float32 rounding of E^2, so
# the relative tolerances below would measure the arithmetic rather than the transform.
MOMENTUM_SCALE = 20.0
MASS = 20.0


def _jets(n_jets: int = 6, n_slots: int = 8, n_real: int = 5, seed: int = 0) -> torch.Tensor:
    """Massive constituents in GeV, columns [px, py, pz, e], padded slots all-zero.

    :return: shape (jets, constituents, 4).
    """
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * MOMENTUM_SCALE
    energy = torch.sqrt((momenta**2).sum(-1, keepdim=True) + MASS**2)
    jets = torch.cat([momenta, energy], dim=-1)
    jets[:, n_real:, :] = 0.0

    return jets


def _collimated_jets(n_jets: int = 64, seed: int = 0) -> torch.Tensor:
    """Narrow jets of about 200 GeV, filling all 16 record columns, five constituents each.

    The SO(3) acceptance cuts are on the constituents' spread in eta and phi about the jet axis, so
    an isotropic spray of constituents is rejected outright and says nothing about the filter. A
    collimated jet passes unless the rotation lands it near the beam, where the same opening angle
    covers a much larger rapidity range.

    :return: shape (jets, 8, 16), the last three slots padding.
    """
    generator = torch.Generator().manual_seed(seed)
    axis = torch.randn(n_jets, 1, 3, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True) * 200.0
    momenta = axis / 5.0 + torch.randn(n_jets, 5, 3, generator=generator) * 2.0
    jets = torch.zeros(n_jets, 8, len(RECORD_COLUMNS))
    jets[:, :5, :3] = momenta
    jets[:, :5, 3] = momenta.norm(dim=-1)

    return jets


def _mass_squared(jets: torch.Tensor) -> torch.Tensor:
    """Minkowski norm of each constituent, metric (+, -, -, -), from columns [px, py, pz, e]."""
    return jets[..., 3] ** 2 - (jets[..., :3] ** 2).sum(-1)


def test_an_so2_rotation_leaves_the_transverse_momentum_and_pz_alone() -> None:
    """SO(2) turns the jet about the beam, which is the symmetry of a pp collision."""
    jets = _jets()
    rotated, _ = apply_random_so2(jets, FOUR_MOMENTUM)

    before = compute_pt(jets[..., 0], jets[..., 1])
    after = compute_pt(rotated[..., 0], rotated[..., 1])

    assert torch.allclose(before, after, atol=1e-3)
    assert torch.allclose(jets[..., 2], rotated[..., 2], atol=1e-4)


def test_the_sampled_so2_matrices_are_rotations() -> None:
    """A determinant of -1 would mirror the jet, which is not the group being probed."""
    matrices, thetas = sample_so2_rot_matrices(64)

    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)
    assert thetas.min() >= 0.0 and thetas.max() < 2 * math.pi


def test_an_so3_rotation_leaves_the_momentum_magnitude_alone() -> None:
    """The rotated jets keep the momentum magnitude per constituent, padding included."""
    jets = _jets()
    rotated, _, keep_mask = apply_random_so3(jets, FOUR_MOMENTUM, apply_filter=False)

    before = jets[..., :3].norm(dim=-1)[keep_mask]
    after = rotated[..., :3].norm(dim=-1)

    assert torch.allclose(before, after, atol=1e-3)


def test_the_sampled_so3_matrices_are_rotations() -> None:
    """Haar-random over SO(3), so no orientation of the jet is favoured."""
    matrices = sample_haar_so3_rot_matrices(64)

    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)


def test_the_so3_acceptance_mask_covers_the_input_batch() -> None:
    """The labels are subset with this mask, so it must be indexed by the jets going in."""
    jets = _collimated_jets(32)
    torch.manual_seed(0)
    rotated, _, keep_mask = apply_random_so3(jets, RECORD_COLUMNS, apply_filter=True)

    assert keep_mask.shape == (32,)
    assert keep_mask.dtype == torch.bool
    assert rotated.shape[0] == int(keep_mask.sum())


def test_a_rejected_jet_is_absent_from_the_rotated_batch() -> None:
    """Reverting a rejected jet instead would make T(x) == x and deflate every metric."""
    jets = _collimated_jets(64)
    torch.manual_seed(0)
    rotated, _, keep_mask = apply_random_so3(jets, RECORD_COLUMNS, apply_filter=True)
    rejected = int((~keep_mask).sum())

    assert 0 < rejected < 64, "the acceptance cuts left nothing to check"
    assert rotated.shape[0] == 64 - rejected
    # The survivors were rotated, not passed through.
    assert not torch.allclose(jets[keep_mask][..., :3], rotated[..., :3])


def test_a_lorentz_boost_leaves_the_constituent_masses_alone() -> None:
    """The invariant mass is what a boost may not touch, whatever it does to E and p."""
    jets = _jets()
    boosted, _ = apply_random_lorentz(jets, FOUR_MOMENTUM, eta_max=2.0)

    before = _mass_squared(jets)[:, :5]
    after = _mass_squared(boosted)[:, :5]

    assert torch.allclose(after, before, rtol=1e-4)


def test_a_lorentz_boost_leaves_the_padded_rows_at_zero() -> None:
    """A padded slot that acquired an energy would be read as a constituent downstream."""
    jets = _jets(n_slots=8, n_real=5)
    boosted, _ = apply_random_lorentz(jets, FOUR_MOMENTUM, eta_max=2.0)

    assert torch.equal(boosted[:, 5:, :], torch.zeros_like(boosted[:, 5:, :]))


def test_a_prefixed_and_a_bare_feature_name_find_the_same_column() -> None:
    """Configs list the features bare; the record names them with a ``j1_`` prefix."""
    assert feature_index(RECORD_COLUMNS, "px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "j1_px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "nonesuch") is None


def test_a_missing_normalisation_scale_falls_back_to_a_physical_batch() -> None:
    """Batches built by hand, and the record's own before it has been prepared, are in GeV."""
    assert get_norm_scale(None, "px") == 1.0
    assert get_norm_scale({"j1_px": 3.0}, "px") == 3.0


@pytest.mark.parametrize("scheme", ["so2", "so3", "lorentz"])
def test_the_factory_builds_one_transform_per_group(scheme: str) -> None:
    """Datamodule, SCL and invariance probe all draw from here, so they cannot drift."""
    jets = _jets()
    transformed, keep_mask = make_transform(scheme, FOUR_MOMENTUM)(jets)

    assert transformed.shape[1:] == jets.shape[1:]
    assert (keep_mask is None) == (scheme != "so3")


def test_the_factory_refuses_the_scheme_that_names_no_transform() -> None:
    """'none' is a legitimate value of data.augmentation_scheme; the caller must not ask."""
    with pytest.raises(ValueError, match="so2"):
        make_transform("none", FOUR_MOMENTUM)
