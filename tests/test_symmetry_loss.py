import pytest
import torch

from src.algorithms.losses.symmetry import (
    SUPPORTED_METRICS,
    SymmetryConsistencyLoss,
    build_symmetric_batch,
    symmetry_distance,
    validate_symmetry_group,
)
from tests.conftest import FOUR_MOMENTUM, jets


def test_the_partner_batch_keeps_the_constituent_grid() -> None:
    batch = jets(16)
    for group in ("so2", "so3", "lorentz", "lorentz_iso"):
        partner, keep_mask = build_symmetric_batch(batch, group, FOUR_MOMENTUM)

        assert partner.shape == batch.shape
        assert partner.dtype == batch.dtype
        # The SO(3) acceptance filter is opt-in, so by default no group drops a jet.
        assert keep_mask is None


def test_the_so3_mask_says_which_logits_the_partner_can_be_compared_with() -> None:
    batch = jets(32)
    filtered, keep_mask = build_symmetric_batch(batch, "so3", FOUR_MOMENTUM, so3_apply_filter=True)

    assert keep_mask.shape == (32,)
    assert filtered.shape[0] == int(keep_mask.sum())


def test_the_distance_vanishes_when_the_logits_agree_and_enters_the_total_weighted() -> None:
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
    with pytest.raises(ValueError, match="identical shape"):
        symmetry_distance(torch.randn(8, 5), torch.randn(6, 5))
    with pytest.raises(ValueError, match="metric must be one of"):
        symmetry_distance(torch.randn(8, 5), torch.randn(8, 5), "euclidean")
    with pytest.raises(ValueError, match="metric must be one of"):
        SymmetryConsistencyLoss(metric="euclidean")


def test_an_scl_run_without_a_symmetry_group_is_refused() -> None:
    # algorithm.group follows data.augmentation_scheme, whose domain includes 'none'.
    for group in ("none", None):
        with pytest.raises(ValueError, match="symm_loss"):
            validate_symmetry_group("symm_loss", group)
    for group in ("so2", "so3", "lorentz", "lorentz_iso", "none", None):
        validate_symmetry_group("augmentation", group)
