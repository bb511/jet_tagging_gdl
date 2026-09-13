import math

import pytest

from src.utils.bootstrap import bootstrap_ci, bootstrap_pvalue

FOLDS = [0.91, 0.89, 0.93, 0.90, 0.92]


def test_the_interval_brackets_the_mean_and_needs_two_folds_to_exist() -> None:
    lower, upper = bootstrap_ci(FOLDS, n_boot=2000)
    mean = sum(FOLDS) / len(FOLDS)

    assert lower < mean < upper
    assert lower == pytest.approx(mean, abs=0.02)
    assert bootstrap_ci([0.5] * 5, n_boot=2000) == (
        pytest.approx(0.5),
        pytest.approx(0.5),
    )
    assert all(math.isnan(end) for end in bootstrap_ci([0.9], n_boot=100))
    assert all(math.isnan(end) for end in bootstrap_ci([], n_boot=100))


def test_the_p_value_follows_whether_the_folds_agree_on_the_shift() -> None:
    assert bootstrap_pvalue([0.050, 0.051, 0.049, 0.050, 0.050], n_boot=2000) < 0.05
    # The same mean under a spread ten times the effect.
    assert bootstrap_pvalue([0.55, -0.45, 0.55, -0.45, 0.05], n_boot=2000) > 0.05
    assert bootstrap_pvalue([0.0] * 5, n_boot=2000) == 1.0
    assert bootstrap_pvalue([0.25, -0.25, 0.5, -0.5, 0.0], n_boot=2000) == 1.0
    assert math.isnan(bootstrap_pvalue([0.05], n_boot=100))
    assert math.isnan(bootstrap_pvalue([], n_boot=100))
