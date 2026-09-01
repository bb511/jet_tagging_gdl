"""The transforms must move the jets the way the symmetry says, and nothing else.

Every check is a conserved quantity of the group applied: pt and pz under SO(2), the momentum
magnitude under SO(3), the invariant mass under a boost. A transform that quietly wrote the wrong
column would break one of them, which no accuracy number ever would.
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

# A massless constituent would put m^2 = E^2 - |p|^2 at the level of the float32 rounding of E^2,
# so the tolerances below would measure the arithmetic rather than the transform. In GeV.
MOMENTUM_SCALE = 20.0
MASS = 20.0


def _jets(n_jets: int = 6, n_slots: int = 8, n_real: int = 5, seed: int = 0) -> torch.Tensor:
    """Massive constituents in GeV, shape (jets, constituents, 4) over [px, py, pz, e]."""
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * MOMENTUM_SCALE
    energy = torch.sqrt((momenta**2).sum(-1, keepdim=True) + MASS**2)
    jets = torch.cat([momenta, energy], dim=-1)
    jets[:, n_real:, :] = 0.0

    return jets


def _collimated_jets(n_jets: int = 64, seed: int = 0) -> torch.Tensor:
    """Narrow jets of about 200 GeV, shape (jets, 8, 16), five constituents and three pad slots.

    The SO(3) acceptance cuts are on the spread in eta and phi about the jet axis, so an isotropic
    spray is rejected outright and says nothing about the filter.
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


def test_an_so2_rotation_turns_the_jet_about_the_beam() -> None:
    """The symmetry of a pp collision: pt and pz are kept, and a determinant of -1 would mirror."""
    jets = _jets()
    rotated, _ = apply_random_so2(jets, FOUR_MOMENTUM)
    matrices, thetas = sample_so2_rot_matrices(64)

    before = compute_pt(jets[..., 0], jets[..., 1])
    after = compute_pt(rotated[..., 0], rotated[..., 1])

    assert torch.allclose(before, after, atol=1e-3)
    assert torch.allclose(jets[..., 2], rotated[..., 2], atol=1e-4)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)
    assert thetas.min() >= 0.0 and thetas.max() < 2 * math.pi


def test_an_so3_rotation_leaves_the_momentum_magnitude_alone() -> None:
    """Haar-random over SO(3), so no orientation is favoured and the magnitude is unchanged."""
    jets = _jets()
    rotated, _, keep_mask = apply_random_so3(jets, FOUR_MOMENTUM, apply_filter=False)
    matrices = sample_haar_so3_rot_matrices(64)

    before = jets[..., :3].norm(dim=-1)[keep_mask]

    assert torch.allclose(before, rotated[..., :3].norm(dim=-1), atol=1e-3)
    assert torch.allclose(torch.linalg.det(matrices), torch.ones(64), atol=1e-5)
    assert torch.allclose(matrices @ matrices.transpose(1, 2), torch.eye(3), atol=1e-5)


def test_the_so3_filter_drops_the_rejected_jets_rather_than_reverting_them() -> None:
    """Reverting a rejected jet would make T(x) == x and deflate every metric."""
    jets = _collimated_jets(64)
    torch.manual_seed(0)
    rotated, _, keep_mask = apply_random_so3(jets, RECORD_COLUMNS, apply_filter=True)
    rejected = int((~keep_mask).sum())

    # The labels are subset with the mask, so it is indexed by the jets going in.
    assert keep_mask.shape == (64,) and keep_mask.dtype == torch.bool
    assert 0 < rejected < 64, "the acceptance cuts left nothing to check"
    assert rotated.shape[0] == 64 - rejected
    # The survivors were rotated, not passed through.
    assert not torch.allclose(jets[keep_mask][..., :3], rotated[..., :3])


def test_a_lorentz_boost_leaves_the_masses_alone_and_the_padding_at_zero() -> None:
    """The invariant mass is what a boost may not touch, whatever it does to E and p."""
    jets = _jets(n_slots=8, n_real=5)
    boosted, _ = apply_random_lorentz(jets, FOUR_MOMENTUM, eta_max=2.0)

    assert torch.allclose(_mass_squared(boosted)[:, :5], _mass_squared(jets)[:, :5], rtol=1e-4)
    # A padded slot that acquired an energy would be read as a constituent downstream.
    assert torch.equal(boosted[:, 5:, :], torch.zeros_like(boosted[:, 5:, :]))


def test_a_column_is_found_by_either_name_and_scaled_by_its_own_factor() -> None:
    """Configs list the features bare, the record prefixes them ``j1_``, and both must resolve."""
    assert feature_index(RECORD_COLUMNS, "px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "j1_px") == RECORD_COLUMNS.index("j1_px")
    assert feature_index(RECORD_COLUMNS, "nonesuch") is None
    # A batch built by hand, and the record's own before it is prepared, is already in GeV.
    assert get_norm_scale(None, "px") == 1.0
    assert get_norm_scale({"j1_px": 3.0}, "px") == 3.0


def test_the_factory_builds_one_transform_per_group_and_refuses_none() -> None:
    """Datamodule, SCL and invariance probe all draw from here, so they cannot drift; 'none' is a
    legitimate value of data.augmentation_scheme, but the caller must not ask for it."""
    jets = _jets()
    for scheme in ("so2", "so3", "lorentz"):
        transformed, keep_mask = make_transform(scheme, FOUR_MOMENTUM)(jets)

        assert transformed.shape[1:] == jets.shape[1:]
        assert (keep_mask is None) == (scheme != "so3")

    with pytest.raises(ValueError, match="so2"):
        make_transform("none", FOUR_MOMENTUM)
