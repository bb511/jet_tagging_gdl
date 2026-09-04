"""The resampling behind the significance columns of the results tables.

Five folds is little to resample from, so the two functions are checked on cases whose answer is
known in advance: an interval that must contain the sample mean, a p-value of exactly one where the
deltas cancel, and a p-value below 0.05 where every fold moved the same way.
"""

import math

import pytest

from src.utils.bootstrap import bootstrap_ci, bootstrap_pvalue

FOLDS = [0.91, 0.89, 0.93, 0.90, 0.92]


def test_the_interval_brackets_the_mean_of_the_folds_it_was_drawn_from() -> None:
    """A percentile interval of resampled means cannot exclude the mean it resamples around."""
    lower, upper = bootstrap_ci(FOLDS, n_boot=2000)
    mean = sum(FOLDS) / len(FOLDS)

    assert lower < mean < upper
    assert lower == pytest.approx(mean, abs=0.02)
    # The interval narrows as the folds agree, so a constant series collapses onto its value.
    assert bootstrap_ci([0.5] * 5, n_boot=2000) == (pytest.approx(0.5), pytest.approx(0.5))


def test_a_difference_of_exactly_nothing_is_never_called_significant() -> None:
    """The deltas centred under H0 are the deltas themselves, so no resample can beat zero."""
    assert bootstrap_pvalue([0.0] * 5, n_boot=2000) == 1.0
    assert bootstrap_pvalue([0.25, -0.25, 0.5, -0.5, 0.0], n_boot=2000) == 1.0


def test_a_shift_every_fold_agrees_on_is_called_significant() -> None:
    """Five folds all improving by 0.05 is the case the paper's columns must pick out."""
    deltas = [0.050, 0.051, 0.049, 0.050, 0.050]

    assert bootstrap_pvalue(deltas, n_boot=2000) < 0.05
    # The same mean with a spread ten times the effect is not evidence of anything.
    assert bootstrap_pvalue([0.55, -0.45, 0.55, -0.45, 0.05], n_boot=2000) > 0.05


def test_a_single_fold_yields_no_interval_and_no_test() -> None:
    """A one-fold search run must report NaN rather than an interval of width zero."""
    assert all(math.isnan(end) for end in bootstrap_ci([0.9], n_boot=100))
    assert all(math.isnan(end) for end in bootstrap_ci([], n_boot=100))
    assert math.isnan(bootstrap_pvalue([0.05], n_boot=100))
    assert math.isnan(bootstrap_pvalue([], n_boot=100))


def test_the_seed_fixes_the_resampling_so_a_table_can_be_rebuilt() -> None:
    """A row recomputed from the same fold values must carry the same interval and p-value."""
    deltas = [0.01, 0.02, -0.01, 0.03, 0.00]

    assert bootstrap_ci(FOLDS, n_boot=500) == bootstrap_ci(FOLDS, n_boot=500)
    assert bootstrap_pvalue(deltas, n_boot=500) == bootstrap_pvalue(deltas, n_boot=500)
