#!/usr/bin/env python3
"""Compare a results table with a baseline results table, fold by fold.

A results table row holds one run's own numbers and nothing else, so the run a regime is measured
against is chosen here, when the tables are joined, rather than at evaluation time. Rows are paired
on ``model`` and ``nconstituents``; a run row with no baseline row of that pair is reported and
left out rather than dropped in silence.

The per-fold difference is the unit. ``kfold_seed`` fixes the splits and the test split itself
never moves, so fold i of two regimes is fitted on the same jets and scored on the same jets, and
the pairing removes the fold-to-fold variation of the data. A pair whose ``_folds`` cells disagree
on the number of folds is refused for the same reason. The difference series is what
:mod:`src.utils.bootstrap` is given, the two functions the results table called for its own delta
columns.

**Positive always means the run is the better model.** The difference is the run minus its baseline
for ``accuracy_transformed``, ``flip_recover``, ``flip_stable`` and ``test_acc``, and the baseline
minus the run for ``flip`` and the invariance metric (``wasserstein``), both being quantities
where less is better.

The output is long, one row per (run row, group, metric)::

    model, nconstituents, method, augmentation_scheme, group, metric,
    delta_mean, delta_std, delta_pm, delta_folds,
    ci_lower, ci_upper, bootstrap_pvalue, significant_boot_005

The ``test_acc`` row leaves the group empty, being a property of the run and not of a group. The
metrics are read off the ``{group}_{metric}_folds`` columns the tables carry, so a table written
under another ``invariance_metric`` is covered without an edit here.

Unlike the results tables it reads, which are appended to, the deltas table is rewritten from its
two inputs on every run::

    python3 scripts/analysis/compare_to_baseline.py \\
        results/px_py_pz_e/results_table_augmentation.csv \\
        results/px_py_pz_e/results_table_baseline.csv

writes ``results/px_py_pz_e/deltas_augmentation_vs_baseline.csv``, named after both inputs so a
second reference does not overwrite the first.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import common

# common puts the repository root on the import path, which the two src imports below need.
from src.symmetry import FLIP_FAMILIES, TABULATED_GROUPS
from src.utils.bootstrap import bootstrap_ci, bootstrap_pvalue

# The metrics whose difference is the run minus its baseline; every other one is read the other way
# round, so that positive is the better model throughout.
HIGHER_IS_BETTER = ("accuracy_transformed", "flip_recover", "flip_stable", "test_acc")

# The columns naming the run, copied from its results row onto every one of its deltas rows.
HEAD = ("model", "nconstituents", "method", "augmentation_scheme")

COLUMNS = [
    *HEAD,
    "group",
    "metric",
    "delta_mean",
    "delta_std",
    "delta_pm",
    "delta_folds",
    "ci_lower",
    "ci_upper",
    "bootstrap_pvalue",
    "significant_boot_005",
]


def _pair_key(row: dict[str, str]) -> tuple[str, str]:
    """The architecture and constituent count a row is paired on, and nothing else."""
    return str(row.get("model", "")).strip(), str(row.get("nconstituents", "")).strip()


def _label(row: dict[str, str]) -> str:
    """Name a row in a message: enough of it to find the run in the table."""
    model, nconstituents = _pair_key(row)
    scheme = row.get("augmentation_scheme", "")
    return f"model={model} nconstituents={nconstituents} scheme={scheme}"


def _fold_pair(
    run: dict[str, str], baseline: dict[str, str], column: str
) -> tuple[list[float], list[float]] | None:
    """The two tables' fold values for one column, or None when no fold of it can be
    paired.

    Both sides keep their gaps through the fold-count check, so that a fold neither
    table could report costs that fold alone. A column no fold reported on either side
    drops out entirely.
    """
    ours = common.parse_folds(run.get(column, ""), keep_nan=True)
    theirs = common.parse_folds(baseline.get(column, ""), keep_nan=True)
    if ours is None or theirs is None:
        return None
    if len(ours) != len(theirs):
        raise ValueError(
            f"{_label(run)}: {column} holds {len(ours)} folds against the baseline's "
            f"{len(theirs)}. The comparison is paired fold by fold, so the two must agree; "
            "re-evaluate one of the runs, or compare it with a baseline of the same fold count."
        )
    paired = [
        (a, b) for a, b in zip(ours, theirs) if not (math.isnan(a) or math.isnan(b))
    ]
    if not paired:
        return None
    return [a for a, _ in paired], [b for _, b in paired]


def _difference(
    run_folds: list[float], baseline_folds: list[float], metric: str
) -> list[float]:
    """The per-fold difference, signed so that positive is the better model."""
    if metric in HIGHER_IS_BETTER:
        return [ours - theirs for ours, theirs in zip(run_folds, baseline_folds)]
    return [theirs - ours for ours, theirs in zip(run_folds, baseline_folds)]


def _delta_columns(deltas: list[float]) -> dict[str, object]:
    """The eight columns of one difference series: the summary, the interval, the paired
    test.

    The test is paired, so the null hypothesis is a mean of zero; a single fold leaves
    it undecided rather than answered.
    """
    nan = float("nan")
    mean, std = common.stats(deltas)
    lower, upper = bootstrap_ci(deltas) if len(deltas) > 1 else (nan, nan)
    pvalue = bootstrap_pvalue(deltas) if len(deltas) > 1 else nan
    # Undecided is not no difference.
    significant = "nan" if math.isnan(pvalue) else ("yes" if pvalue < 0.05 else "no")
    return {
        "delta_mean": round(mean, 6),
        "delta_std": round(std, 6),
        "delta_pm": f"{mean:.4f} +/- {std:.4f}",
        "delta_folds": ";".join(f"{delta:.6f}" for delta in deltas),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "bootstrap_pvalue": round(pvalue, 6),
        "significant_boot_005": significant,
    }


def metrics_covered(rows: list[dict[str, str]]) -> tuple[str, ...]:
    """The per-group metrics a deltas table covers: the flip families and the invariance
    one.

    The invariance metric is read off the columns rather than assumed. A table written
    with ``invariance_metric=flip`` names none, that family being tabulated for every
    group already.
    """
    metric = common.detect_inv_metric_suffix(rows)
    return tuple(dict.fromkeys(FLIP_FAMILIES + ((metric,) if metric else ())))


def delta_rows(
    run: dict[str, str], baseline: dict[str, str], metrics: tuple[str, ...]
) -> list[dict[str, object]]:
    """Every long-format row of one pair: the run's own accuracy, then each group and
    metric."""
    head = {name: run.get(name, "") for name in HEAD}
    wanted = [("", "test_acc")] + [
        (group, m) for group in TABULATED_GROUPS for m in metrics
    ]
    rows: list[dict[str, object]] = []
    for group, metric in wanted:
        column = f"{group}_{metric}_folds" if group else f"{metric}_folds"
        folds = _fold_pair(run, baseline, column)
        if folds is None:
            continue
        columns = _delta_columns(_difference(*folds, metric))
        rows.append(head | {"group": group, "metric": metric} | columns)
    return rows


def index_baselines(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    """(model, nconstituents) -> baseline row; the tables are appended to, so the last
    one wins."""
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = _pair_key(row)
        if key in index:
            print(
                f"[note] {key[0]} N={key[1]}: several baseline rows, the last of them is used"
            )
        index[key] = row
    return index


def compare(
    run_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    metrics: tuple[str, ...],
) -> list[dict[str, object]]:
    """The deltas rows of every run row that has a baseline row; the others are
    reported."""
    baselines = index_baselines(baseline_rows)
    rows: list[dict[str, object]] = []
    for run in run_rows:
        baseline = baselines.get(_pair_key(run))
        if baseline is None:
            print(
                f"[skip] {_label(run)}: no baseline row of that model and constituent count"
            )
            continue
        rows += delta_rows(run, baseline, metrics)
    return rows


def write_deltas(rows: list[dict[str, object]], output_path: Path) -> None:
    """Write the deltas table, replacing whatever stood there: it is derived from its
    inputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _trimmed_stem(path: Path) -> str:
    """A table's stem without the ``results_table_`` the callback puts on every one of
    them."""
    prefix = "results_table_"
    return path.stem[len(prefix) :] if path.stem.startswith(prefix) else path.stem


def default_output(run_table: Path, baseline_table: Path) -> Path:
    """``deltas_<run>_vs_<baseline>.csv`` beside the run table, named after both
    inputs."""
    name = f"deltas_{_trimmed_stem(run_table)}_vs_{_trimmed_stem(baseline_table)}.csv"
    return run_table.parent / name


def _parse_args() -> argparse.Namespace:
    """Command line: the run table, the baseline it is measured against, where the
    deltas go."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "run_table", type=Path, help="results table whose runs are compared"
    )
    parser.add_argument(
        "baseline_table", type=Path, help="results table they are measured against"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Where the deltas table goes (default: deltas_<run>_vs_<baseline>.csv beside the run "
        "table).",
    )
    return parser.parse_args()


def main() -> None:
    """Write one deltas table from one results table and one baseline table."""
    args = _parse_args()
    run_rows = common.read_csv(args.run_table)
    baseline_rows = common.read_csv(args.baseline_table)
    if not run_rows or not baseline_rows:
        sys.exit(
            f"nothing to compare: {args.run_table} or {args.baseline_table} holds no rows"
        )
    rows = compare(run_rows, baseline_rows, metrics_covered(run_rows + baseline_rows))
    if not rows:
        sys.exit(
            "no run row could be paired with a baseline row, so nothing was written"
        )
    output_path = args.out or default_output(args.run_table, args.baseline_table)
    write_deltas(rows, output_path)
    print(f"[write] {len(rows)} delta rows -> {output_path}")


if __name__ == "__main__":
    main()
