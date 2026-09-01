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


def test_the_flip_breakdown_splits_the_transitions_by_group_and_by_what_they_did() -> None:
    """Only this breakdown tells a repaired wrong prediction from a broken right one."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._flip_counts = {
        "so2": {"wrong_before": 10, "recovered": 4, "correct_before": 90, "stable": 81},
        "lorentz": {"wrong_before": 0, "recovered": 0, "correct_before": 50, "stable": 50},
    }

    metrics = callback._flip_breakdown_metrics()

    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(0.9)
    assert metrics["invariance/so2_flip_degrade"] == pytest.approx(0.1)
    # A perfect classifier gets no wrong predictions to recover, which is not a zero rate.
    assert math.isnan(metrics["invariance/lorentz_flip_recover"])
    assert metrics["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_a_probe_that_never_ran_reports_nothing() -> None:
    """A datamodule that does not name its columns disables the probe, and no key is logged."""
    assert InvarianceMetricCallback()._flip_breakdown_metrics() == {}
    assert InvarianceMetricCallback()._wasserstein_metrics() == {}


def test_a_constant_shift_is_measured_at_its_own_size_and_a_reshuffling_is_not_seen() -> None:
    """Wasserstein-1 between two distributions offset by a constant is that constant."""
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
    # Three batches, since the distributions are compared over the whole test set at once.
    batches = [torch.randn(2000, 5) for _ in range(3)]
    callback._orig_logits = batches
    callback._aug_logits = {
        "lorentz": [batch + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]) for batch in batches],
        "so2": [batch[torch.randperm(batch.shape[0])] for batch in batches],
    }

    metrics = callback._wasserstein_metrics()

    assert metrics["invariance/lorentz_wasserstein_g"] == pytest.approx(3.0, abs=0.05)
    for name in CLASS_NAMES[1:]:
        assert metrics[f"invariance/lorentz_wasserstein_{name}"] < 0.05
    # The reported number is the mean over the five class channels.
    assert metrics["invariance/lorentz_wasserstein"] == pytest.approx(0.6, abs=0.05)
    # The family is unpaired on purpose, so a permutation of the same logits reads as zero.
    assert metrics["invariance/so2_wasserstein"] < 1e-6
