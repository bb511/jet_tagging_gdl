import math

import pytest

from src.callbacks.invariance_metric_callback import InvarianceMetricCallback


def _make_callback() -> InvarianceMetricCallback:
    return InvarianceMetricCallback(compute_flip_breakdown=True)


def test_flip_breakdown_basic_counts() -> None:
    cb = _make_callback()
    cb._flip_counts = {"so2": {"wrong_before": 10, "recovered": 4, "correct_before": 90, "stable": 81}}

    r = cb._flip_breakdown_metrics()

    assert r["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert r["invariance/so2_flip_stable"] == pytest.approx(0.9)
    assert r["invariance/so2_flip_degrade"] == pytest.approx(0.1)


def test_flip_breakdown_zero_denominator_is_nan() -> None:
    cb = _make_callback()
    cb._flip_counts = {"so2": {"wrong_before": 0, "recovered": 0, "correct_before": 50, "stable": 50}}

    r = cb._flip_breakdown_metrics()

    assert math.isnan(r["invariance/so2_flip_recover"])
    assert r["invariance/so2_flip_stable"] == pytest.approx(1.0)
    assert r["invariance/so2_flip_degrade"] == pytest.approx(0.0)


def test_flip_breakdown_multiple_groups() -> None:
    cb = _make_callback()
    cb._flip_counts = {
        "so2": {"wrong_before": 20, "recovered": 10, "correct_before": 80, "stable": 70},
        "lorentz": {"wrong_before": 20, "recovered": 0, "correct_before": 80, "stable": 80},
    }

    r = cb._flip_breakdown_metrics()

    assert r["invariance/so2_flip_recover"] == pytest.approx(0.5)
    assert r["invariance/lorentz_flip_recover"] == pytest.approx(0.0)
    assert r["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_flip_breakdown_no_counts_returns_empty() -> None:
    cb = _make_callback()
    assert cb._flip_breakdown_metrics() == {}
