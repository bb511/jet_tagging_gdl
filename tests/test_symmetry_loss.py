"""The symmetry-consistency loss: the partner batch, the distance and the total.

The SCL is one of the two training methods being compared, so its arithmetic is pinned here rather
than inferred from a training curve.
"""

import pytest
import torch

from src.algorithms.losses.symmetry import (
    SUPPORTED_METRICS,
    SymmetryConsistencyLoss,
    build_symmetric_batch,
    symmetry_distance,
    validate_symmetry_group,
)

FOUR_MOMENTUM = ["px", "py", "pz", "e"]


def _jets(n_jets: int = 16, n_slots: int = 8, n_real: int = 5) -> torch.Tensor:
    """Massive constituents in GeV, shape (jets, constituents, 4) over [px, py, pz, e]."""
    generator = torch.Generator().manual_seed(0)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * 20.0
    energy = torch.sqrt((momenta**2).sum(-1, keepdim=True) + 400.0)
    jets = torch.cat([momenta, energy], dim=-1)
    jets[:, n_real:, :] = 0.0

    return jets


def test_the_partner_batch_keeps_the_constituent_grid() -> None:
    """The model runs on the partner exactly as it runs on the original."""
    jets = _jets()
    for group in ("so2", "so3", "lorentz"):
        partner, keep_mask = build_symmetric_batch(jets, group, FOUR_MOMENTUM)

        assert partner.shape[1:] == jets.shape[1:]
        assert partner.dtype == jets.dtype
        assert (keep_mask is None) == (group != "so3")


def test_the_so3_mask_says_which_logits_the_partner_can_be_compared_with() -> None:
    """Filtered jets have no partner, so the caller subsets its own logits with this mask."""
    jets = _jets(n_jets=32)
    filtered, keep_mask = build_symmetric_batch(jets, "so3", FOUR_MOMENTUM, so3_apply_filter=True)
    # The probe measures the model, so it may ask for the unfiltered orbit instead.
    whole, all_kept = build_symmetric_batch(jets, "so3", FOUR_MOMENTUM, so3_apply_filter=False)

    assert keep_mask.shape == (32,)
    assert filtered.shape[0] == int(keep_mask.sum())
    assert bool(all_kept.all()) and whole.shape[0] == 32


def test_the_distance_vanishes_when_the_logits_agree_and_enters_the_total_weighted() -> None:
    """A non-zero floor would penalise an already invariant model; the weight is what is tuned."""
    torch.manual_seed(0)
    logits, other = torch.randn(32, 5), torch.randn(32, 5)
    for metric in SUPPORTED_METRICS:
        total, sym = SymmetryConsistencyLoss(weight=0.25, metric=metric)(
            torch.tensor(1.5), logits, other
        )

        assert symmetry_distance(logits, logits.clone(), metric) == pytest.approx(0.0, abs=1e-6)
        assert symmetry_distance(logits, other, metric) > 0.0
        assert sym == pytest.approx(symmetry_distance(logits, other, metric).item())
        assert total.item() == pytest.approx(1.5 + 0.25 * sym.item())


def test_a_mismatched_shape_or_an_unknown_metric_is_refused() -> None:
    """The two logit sets part company under filtered SO(3), and a typo must not fall through."""
    with pytest.raises(ValueError, match="identical shape"):
        symmetry_distance(torch.randn(8, 5), torch.randn(6, 5))
    with pytest.raises(ValueError, match="metric must be one of"):
        symmetry_distance(torch.randn(8, 5), torch.randn(8, 5), "euclidean")
    with pytest.raises(ValueError, match="metric must be one of"):
        SymmetryConsistencyLoss(metric="euclidean")


def test_an_scl_run_without_a_symmetry_group_is_refused() -> None:
    """algorithm.group follows data.augmentation_scheme, whose domain includes 'none'."""
    for group in ("none", None):
        with pytest.raises(ValueError, match="symm_loss"):
            validate_symmetry_group("symm_loss", group)
    # Under augmentation the group is the datamodule's business and is never read here.
    for group in ("so2", "so3", "lorentz", "none", None):
        validate_symmetry_group("augmentation", group)
