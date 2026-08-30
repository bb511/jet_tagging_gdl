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
    """Massive constituents in GeV, columns [px, py, pz, e], padded slots all-zero."""
    generator = torch.Generator().manual_seed(0)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * 20.0
    energy = torch.sqrt((momenta**2).sum(-1, keepdim=True) + 400.0)
    jets = torch.cat([momenta, energy], dim=-1)
    jets[:, n_real:, :] = 0.0

    return jets


@pytest.mark.parametrize("group", ["so2", "so3", "lorentz"])
def test_the_partner_batch_keeps_the_constituent_grid(group: str) -> None:
    """The model runs on the partner exactly as it runs on the original."""
    jets = _jets()
    partner, keep_mask = build_symmetric_batch(jets, group, FOUR_MOMENTUM)

    assert partner.shape[1:] == jets.shape[1:]
    assert partner.dtype == jets.dtype
    assert (keep_mask is None) == (group != "so3")


def test_the_so3_mask_says_which_logits_the_partner_can_be_compared_with() -> None:
    """Filtered jets have no partner, so the caller subsets its own logits with this."""
    jets = _jets(n_jets=32)
    partner, keep_mask = build_symmetric_batch(jets, "so3", FOUR_MOMENTUM, so3_apply_filter=True)

    assert keep_mask.shape == (32,)
    assert partner.shape[0] == int(keep_mask.sum())


def test_dropping_the_so3_filter_keeps_every_jet() -> None:
    """The probe measures the model, so it may ask for the unfiltered orbit."""
    jets = _jets(n_jets=32)
    partner, keep_mask = build_symmetric_batch(jets, "so3", FOUR_MOMENTUM, so3_apply_filter=False)

    assert bool(keep_mask.all())
    assert partner.shape[0] == 32


@pytest.mark.parametrize("metric", SUPPORTED_METRICS)
def test_the_distance_vanishes_exactly_when_the_logits_agree(metric: str) -> None:
    """A non-zero floor would make the SCL penalise an already invariant model."""
    torch.manual_seed(0)
    logits = torch.randn(32, 5)

    assert symmetry_distance(logits, logits.clone(), metric) == pytest.approx(0.0, abs=1e-6)
    assert symmetry_distance(logits, torch.randn(32, 5), metric) > 0.0


def test_the_distance_refuses_two_differently_shaped_logit_sets() -> None:
    """Under filtered SO(3) the two sets part company unless the caller applies the mask."""
    with pytest.raises(ValueError, match="identical shape"):
        symmetry_distance(torch.randn(8, 5), torch.randn(6, 5))


def test_an_unknown_distance_is_refused() -> None:
    """A typo in symm_loss_metric must not fall through to a default."""
    with pytest.raises(ValueError, match="metric must be one of"):
        symmetry_distance(torch.randn(8, 5), torch.randn(8, 5), "euclidean")
    with pytest.raises(ValueError, match="metric must be one of"):
        SymmetryConsistencyLoss(metric="euclidean")


def test_the_total_loss_is_the_classification_loss_plus_the_weighted_distance() -> None:
    """The weight is the one hyperparameter the SCL searches add to the space."""
    torch.manual_seed(0)
    loss_fn = SymmetryConsistencyLoss(weight=0.25, metric="l2")
    cls_loss = torch.tensor(1.5)
    logits, logits_sym = torch.randn(32, 5), torch.randn(32, 5)

    total, sym = loss_fn(cls_loss, logits, logits_sym)

    assert sym == pytest.approx(symmetry_distance(logits, logits_sym, "l2").item())
    assert total.item() == pytest.approx(1.5 + 0.25 * sym.item())


def test_an_scl_run_without_a_symmetry_group_is_refused() -> None:
    """algorithm.group follows data.augmentation_scheme, whose domain includes 'none'."""
    with pytest.raises(ValueError, match="symm_loss"):
        validate_symmetry_group("symm_loss", "none")
    with pytest.raises(ValueError, match="symm_loss"):
        validate_symmetry_group("symm_loss", None)


@pytest.mark.parametrize("group", ["so2", "so3", "lorentz", "none", None])
def test_an_augmentation_run_may_leave_the_group_unset(group: str | None) -> None:
    """Under augmentation the group is the datamodule's business and is never read here."""
    validate_symmetry_group("augmentation", group)
