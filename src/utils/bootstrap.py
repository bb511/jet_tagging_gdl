"""Bootstrap the spread of a quantity measured once per cross-validation fold.

The results table uses this for two things: a percentile interval on a fold-averaged quantity, and
a two-sided test on a per-fold difference between a run and its baseline. The difference is paired,
so a single series of deltas is all the test needs: ``kfold_seed`` fixes the splits and the test
split itself never moves, so fold i of two regimes is fitted on the same jets and scored on the
same jets, and the fold-to-fold variation the pairing removes is that of the data rather than of
the regime.

With the five folds the experiments run, the power of that test is low, roughly 0.3 to 0.5 for the
effect sizes seen here, so a p-value above 0.05 is weak evidence of no difference. Read it beside
the effect size, never on its own.
"""

from collections.abc import Sequence

import numpy as np


def _resample_means(sample: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of *n_boot* draws with replacement, one row of the index array per draw."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, sample.size, size=(n_boot, sample.size))
    return sample[draws].mean(axis=1)


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 10_000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile confidence interval of the mean of *values*.

    :param ci: coverage of the interval, 0.95 for the 2.5 % and 97.5 % quantiles of the resampled
        means.
    :param seed: fixes the resampling alone, so a table rewritten from the same fold values carries
        the same interval.
    :return: the lower and upper ends, a pair of NaN below the two values a resampling needs.
    """
    sample = np.asarray(values, dtype=float)
    if sample.size < 2:
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lower, upper = np.quantile(_resample_means(sample, n_boot, seed), [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def bootstrap_pvalue(deltas: Sequence[float], n_boot: int = 10_000, seed: int = 0) -> float:
    """Two-sided paired bootstrap p-value on a series of per-fold differences.

    Each element of *deltas* is one fold's value for a run minus the same fold's value for its
    baseline. The deltas are shifted to a mean of zero, which is the null hypothesis, and the
    p-value is the fraction of resampled means lying at least as far from zero as the observed mean
    does.

    :return: 1.0 at an observed mean of exactly zero, NaN below the two folds the resampling needs.
    """
    sample = np.asarray(deltas, dtype=float)
    if sample.size < 2:
        return float("nan")
    observed = float(sample.mean())
    if observed == 0.0:
        return 1.0
    means = _resample_means(sample - observed, n_boot, seed)
    return float(np.mean(np.abs(means) >= abs(observed)))
