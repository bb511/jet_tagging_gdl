import pytest
import torch

from src.algorithms.losses.symmetry import SymmetryConsistencyLoss


def test_the_distance_vanishes_when_the_logits_agree_and_enters_the_total_weighted() -> (
    None
):
    torch.manual_seed(0)
    logits, other = torch.randn(32, 5), torch.randn(32, 5)
    loss = SymmetryConsistencyLoss("so2", weight=0.25)

    total, sym = loss(torch.tensor(1.5), logits, other)
    _, agreed = loss(torch.tensor(1.5), logits, logits.clone())

    assert agreed.item() == pytest.approx(0.0, abs=1e-6)
    assert sym.item() == pytest.approx((logits - other).norm(dim=1).mean().item())
    assert total.item() == pytest.approx(1.5 + 0.25 * sym.item())


def test_a_mismatched_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="identical shape"):
        SymmetryConsistencyLoss("so2")(
            torch.tensor(0.0), torch.randn(8, 5), torch.randn(6, 5)
        )


def test_an_scl_run_without_a_symmetry_group_is_refused() -> None:
    # algorithm.group follows data.augmentation_scheme, whose domain includes 'none'.
    for group in ("none", None):
        with pytest.raises(ValueError, match="symm_loss"):
            SymmetryConsistencyLoss(group)
