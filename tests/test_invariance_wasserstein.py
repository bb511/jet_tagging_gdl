import pytest
import torch

from src.callbacks.invariance_metric_callback import InvarianceMetricCallback


def _make_callback() -> InvarianceMetricCallback:
    cb = InvarianceMetricCallback(compute_wasserstein=True, class_names=("g", "q", "W", "Z", "t"))
    return cb


def test_wasserstein_identical_distributions_is_zero() -> None:
    """Same samples, just reshuffled across the batch dimension -> distance ~ 0."""
    torch.manual_seed(0)
    cb = _make_callback()

    y = torch.randn(2000, 5)
    y_shuffled = y[torch.randperm(y.shape[0])]

    cb._orig_logits = [y]
    cb._aug_logits = {"so2": [y_shuffled]}

    results = cb._wasserstein_metrics()

    for name in cb.class_names:
        assert results[f"invariance/so2_wasserstein_{name}"] < 1e-6
    assert results["invariance/so2_wasserstein"] < 1e-6


def test_wasserstein_constant_shift_matches_shift_magnitude() -> None:
    """A constant additive shift in one class channel should be picked up exactly."""
    torch.manual_seed(0)
    cb = _make_callback()

    y_orig = torch.randn(5000, 5)
    shift = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0])
    y_aug = y_orig + shift

    cb._orig_logits = [y_orig]
    cb._aug_logits = {"lorentz": [y_aug]}

    results = cb._wasserstein_metrics()

    assert results["invariance/lorentz_wasserstein_g"] == pytest.approx(3.0, abs=0.05)
    for name in cb.class_names[1:]:
        assert results[f"invariance/lorentz_wasserstein_{name}"] < 0.05
    assert results["invariance/lorentz_wasserstein"] == pytest.approx(0.6, abs=0.05)


def test_wasserstein_multiple_groups_and_batches() -> None:
    """Logits accumulated across several batches, multiple augmentation groups."""
    torch.manual_seed(0)
    cb = _make_callback()

    batches = [torch.randn(100, 5) for _ in range(3)]
    cb._orig_logits = batches
    cb._aug_logits = {
        "so2": [b + 1.0 for b in batches],
        "so3": [b for b in batches],
    }

    results = cb._wasserstein_metrics()

    assert results["invariance/so2_wasserstein"] > results["invariance/so3_wasserstein"]
    assert results["invariance/so3_wasserstein"] < 1e-6


def test_wasserstein_no_logits_returns_empty() -> None:
    cb = _make_callback()
    assert cb._wasserstein_metrics() == {}
