import pytest
import torch

from src.algorithms.losses.symmetry import SUPPORTED_METRICS, SymmetryConsistencyLoss
from src.symmetry import GROUPS
from tests.conftest import FOUR_MOMENTUM, jets


def test_the_partner_batch_keeps_the_constituent_grid() -> None:
    batch = jets(16)
    for group in GROUPS:
        partner = SymmetryConsistencyLoss(group).transform(batch, FOUR_MOMENTUM)

        assert partner.shape == batch.shape
        assert partner.dtype == batch.dtype


def test_the_distance_vanishes_when_the_logits_agree_and_enters_the_total_weighted() -> (
    None
):
    torch.manual_seed(0)
    logits, other = torch.randn(32, 5), torch.randn(32, 5)
    for metric in SUPPORTED_METRICS:
        loss = SymmetryConsistencyLoss("so2", weight=0.25, metric=metric)
        total, sym = loss(torch.tensor(1.5), logits, other)

        assert loss.distance(logits, logits.clone()) == pytest.approx(0.0, abs=1e-6)
        assert sym > 0.0
        assert sym == pytest.approx(loss.distance(logits, other).item())
        assert total.item() == pytest.approx(1.5 + 0.25 * sym.item())


def test_a_mismatched_shape_or_an_unknown_metric_is_refused() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        SymmetryConsistencyLoss("so2")(
            torch.tensor(0.0), torch.randn(8, 5), torch.randn(6, 5)
        )
    with pytest.raises(ValueError, match="metric must be one of"):
        SymmetryConsistencyLoss("so2", metric="euclidean")


def test_an_scl_run_without_a_symmetry_group_is_refused() -> None:
    # algorithm.group follows data.augmentation_scheme, whose domain includes 'none'.
    for group in ("none", None):
        with pytest.raises(ValueError, match="symm_loss"):
            SymmetryConsistencyLoss(group)
