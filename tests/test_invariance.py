"""The two invariance families the probe reports, computed from state fed in by hand.

The flip breakdown counts label transitions and the Wasserstein family compares two logit
distributions. Both feed the results table, and neither can be checked against a training run, so
the counters and the distances are exercised directly.
"""

import math

import pytest
import torch

from src.callbacks.invariance import InvarianceMetricCallback

CLASS_NAMES = ("g", "q", "W", "Z", "t")


def test_the_flip_breakdown_splits_the_transitions_by_what_they_did() -> None:
    """A symmetry that repairs wrong predictions and one that breaks right ones both move the flip
    rate, and only this breakdown tells them apart."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._flip_counts = {
        "so2": {"wrong_before": 10, "recovered": 4, "correct_before": 90, "stable": 81}
    }

    metrics = callback._flip_breakdown_metrics()

    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(0.9)
    assert metrics["invariance/so2_flip_degrade"] == pytest.approx(0.1)


def test_a_rate_with_no_jets_behind_it_is_reported_as_nan() -> None:
    """A perfect classifier gets no wrong predictions to recover, which is not a zero rate."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._flip_counts = {
        "so2": {"wrong_before": 0, "recovered": 0, "correct_before": 50, "stable": 50}
    }

    metrics = callback._flip_breakdown_metrics()

    assert math.isnan(metrics["invariance/so2_flip_recover"])
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(1.0)
    assert metrics["invariance/so2_flip_degrade"] == pytest.approx(0.0)


def test_each_group_is_counted_on_its_own() -> None:
    """The three groups share a test pass, and the results table reads them separately."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._flip_counts = {
        "so2": {"wrong_before": 20, "recovered": 10, "correct_before": 80, "stable": 70},
        "lorentz": {"wrong_before": 20, "recovered": 0, "correct_before": 80, "stable": 80},
    }

    metrics = callback._flip_breakdown_metrics()

    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.5)
    assert metrics["invariance/lorentz_flip_recover"] == pytest.approx(0.0)
    assert metrics["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_a_probe_that_never_ran_reports_nothing() -> None:
    """A datamodule that does not name its columns disables the probe, and no key is logged."""
    assert InvarianceMetricCallback()._flip_breakdown_metrics() == {}
    assert InvarianceMetricCallback()._wasserstein_metrics() == {}


def test_the_same_logits_in_another_order_are_the_same_distribution() -> None:
    """The Wasserstein family is unpaired on purpose: it must not see the reshuffling."""
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
    logits = torch.randn(2000, 5)
    callback._orig_logits = [logits]
    callback._aug_logits = {"so2": [logits[torch.randperm(logits.shape[0])]]}

    metrics = callback._wasserstein_metrics()

    for name in CLASS_NAMES:
        assert metrics[f"invariance/so2_wasserstein_{name}"] < 1e-6
    assert metrics["invariance/so2_wasserstein"] < 1e-6


def test_a_constant_shift_in_one_class_is_measured_at_its_own_size() -> None:
    """Wasserstein-1 between two distributions offset by a constant is that constant."""
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
    logits = torch.randn(5000, 5)
    callback._orig_logits = [logits]
    callback._aug_logits = {"lorentz": [logits + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0])]}

    metrics = callback._wasserstein_metrics()

    assert metrics["invariance/lorentz_wasserstein_g"] == pytest.approx(3.0, abs=0.05)
    for name in CLASS_NAMES[1:]:
        assert metrics[f"invariance/lorentz_wasserstein_{name}"] < 0.05
    # The reported number is the mean over the five class channels.
    assert metrics["invariance/lorentz_wasserstein"] == pytest.approx(0.6, abs=0.05)


def test_the_logits_of_every_test_batch_are_pooled_before_the_comparison() -> None:
    """The distributions are compared over the whole test set, not batch by batch."""
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
    batches = [torch.randn(100, 5) for _ in range(3)]
    callback._orig_logits = batches
    callback._aug_logits = {"so2": [batch + 1.0 for batch in batches], "so3": list(batches)}

    metrics = callback._wasserstein_metrics()

    assert metrics["invariance/so2_wasserstein"] > metrics["invariance/so3_wasserstein"]
    assert metrics["invariance/so3_wasserstein"] < 1e-6
