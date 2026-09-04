"""Append the summary row of a run to the results table of its regime.

One row per run rather than per fold: the callback stays attached across the k folds and writes the
row once the last fold has been tested. Tables are split by the input feature list and by the
regime, so ``results/px_py_pz_e/results_table_symm_loss.csv`` holds every symmetry-consistency run
on the four-momentum features, whatever the architecture or the constituent count. Most quantities
occupy four columns (the mean over the folds, its standard deviation, the two joined as ``mean +/-
std``, and the fold values joined by ';'), in the column order the tables already on disk carry, so
a new row can be appended to them.

A run evaluated against a baseline (``eval.baseline_checkpoint_manifest_path``) carries the
baseline's own accuracy as well, and one eight-column family per delta the invariance probe logged:
the four summary columns, the percentile interval of the mean over the folds, and the paired
bootstrap test of :mod:`src.utils.bootstrap` against a true difference of zero.
"""

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lightning import Callback, LightningModule, Trainer

from src.utils.bootstrap import bootstrap_ci, bootstrap_pvalue
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

GROUPS = ("so2", "so3", "lorentz")

# The delta families reported whatever `invariance_metric` selects.
FLIP_DELTA_SUFFIXES = ("flip", "flip_recover", "flip_stable", "flip_degrade")


def _delta_suffixes(invariance_metric: str) -> tuple[str, ...]:
    """The delta families a row carries, in column order and without repeats.

    ``invariance_metric`` may already name one of the fixed families, ``flip`` for instance.
    """
    return tuple(dict.fromkeys(("accuracy_transformed", invariance_metric, *FLIP_DELTA_SUFFIXES)))


def _group_keys(suffixes: Sequence[str]) -> tuple[str, ...]:
    """``invariance/<group>_<suffix>`` for every suffix and group, suffix-major."""
    return tuple(f"invariance/{group}_{suffix}" for suffix in suffixes for group in GROUPS)


def _finite(values: Sequence[float]) -> list[float]:
    """The values the folds actually reported, the NaNs of the others dropped."""
    return [value for value in values if not math.isnan(value)]


def _display_regime(method: str, augmentation_scheme: str) -> str:
    """Name the regime as the tables report it: a run with no symmetry is the baseline."""
    if method == "augmentation" and augmentation_scheme == "none":
        return "baseline"
    return method


def _table_path(base: Path, features_label: str | None, display_method: str) -> Path:
    """Weave the feature list and the regime into the table name.

    ``results/results_table.csv`` becomes ``results/px_py_pz_e/results_table_baseline.csv``.
    """
    directory = base.parent / features_label if features_label else base.parent
    return directory / f"{base.stem}_{display_method}{base.suffix}"


def _join(values: Sequence[float]) -> str:
    """Join the per-fold values into one cell, six decimals each."""
    return ";".join("nan" if math.isnan(value) else f"{value:.6f}" for value in values)


def _check_header(path: Path, fieldnames: list[str]) -> None:
    """Refuse to append under a header with other columns, which would misalign the row."""
    with open(path, newline="") as table:
        header = next(csv.reader(table), None)
    if header is None or header == fieldnames:
        return
    raise ValueError(
        f"{path} carries the columns {header}, and this run writes {fieldnames}. Appending "
        "would misalign every value, so move the old table aside or set another output_path."
    )


class ResultsTableCallback(Callback):
    """Aggregate the folds of one run into a single row of the results table.

    :param output_path: base name of the table; :func:`_table_path` weaves the regime and the
        feature list into it.
    :param n_folds: folds to wait for before the row is written.
    :param model_name: short architecture label, ``algorithm.name``.
    :param features: the input feature list, which names the subdirectory.
    :param selected_metric_mode: with ``scheme`` the run is judged on the group it was trained on,
        and on the mean over the three groups when it was trained on none; ``mean_all`` always
        reports that mean.
    :param invariance_metric: which invariance family fills the selected and per-group columns,
        e.g. ``wasserstein`` reads ``invariance/so2_wasserstein``.
    """

    # Reported whatever `invariance_metric` selects, being counts of label transitions
    # rather than a distance between logits.
    FLIP_BREAKDOWN_SUFFIXES = ("flip_recover", "flip_stable", "flip_degrade")

    def __init__(
        self,
        output_path: str,
        n_folds: int = 1,
        method: str = "augmentation",
        model_name: str = "model",
        augmentation_scheme: str = "none",
        nconstituents: int = 0,
        features: Sequence[str] | None = None,
        selected_metric_mode: str = "scheme",
        invariance_metric: str = "wasserstein",
    ) -> None:
        super().__init__()
        self.n_folds = int(n_folds)
        self.run_method_raw = method
        self.model_name = model_name
        self.augmentation_scheme = str(augmentation_scheme).lower()
        self.nconstituents = int(nconstituents)
        self.selected_metric_mode = selected_metric_mode
        self._resolve_output(output_path, features)
        self._resolve_keys(invariance_metric)
        self._reset_folds()

    def _resolve_output(self, output_path: str, features: Sequence[str] | None) -> None:
        """Name the table this run appends to, from its regime and its feature list."""
        self.display_method = _display_regime(self.run_method_raw, self.augmentation_scheme)
        self.features_label = "_".join(str(f) for f in features) if features else None
        self.output_path = _table_path(Path(output_path), self.features_label, self.display_method)

    def _resolve_keys(self, invariance_metric: str) -> None:
        """The ``trainer.callback_metrics`` keys the columns of a fold are read from."""
        self.invariance_metric = invariance_metric
        self.INVARIANCE_KEYS = _group_keys([invariance_metric])
        self._FLIP_BREAKDOWN_KEYS = _group_keys(self.FLIP_BREAKDOWN_SUFFIXES)
        self._DELTA_SUFFIXES = _delta_suffixes(invariance_metric)
        self._DELTA_KEYS = _group_keys([f"delta_{suffix}" for suffix in self._DELTA_SUFFIXES])

    def _reset_folds(self) -> None:
        """Empty the per-fold accumulators and re-arm the row written once they are full."""
        self._fold_invariance: list[dict[str, float]] = []
        self._fold_flops: list[float] = []
        self._fold_test_acc: list[float] = []
        self._fold_baseline_test_acc: list[float] = []
        self._fold_delta_test_acc: list[float] = []
        self._summary_written = False

    @staticmethod
    def _to_float(value) -> float:
        """Read a metric out of Lightning's dicts, whether it is a tensor or a number."""
        if value is None:
            return float("nan")
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        """Mean over the folds that reported the metric; NaN when none did."""
        finite = np.asarray(values, dtype=float)
        finite = finite[~np.isnan(finite)]
        return float(finite.mean()) if finite.size else float("nan")

    @staticmethod
    def _safe_std(values: Sequence[float]) -> float:
        """Sample standard deviation over the folds; zero for a single fold, NaN for none."""
        finite = np.asarray(values, dtype=float)
        finite = finite[~np.isnan(finite)]
        if finite.size == 0:
            return float("nan")
        return float(finite.std(ddof=1)) if finite.size > 1 else 0.0

    @staticmethod
    def _format_pm(mean_val: float, std_val: float) -> str:
        """Render a mean and its spread as one cell for a paper table."""
        if math.isnan(mean_val):
            return "nan"
        return f"{mean_val:.4f} +/- {std_val:.4f}"

    def _summary(self, prefix: str, values: Sequence[float]) -> dict[str, object]:
        """The three summary columns of one series: mean, standard deviation, both."""
        mean, std = self._safe_mean(values), self._safe_std(values)
        return {
            f"{prefix}_mean": round(mean, 6),
            f"{prefix}_std": round(std, 6),
            f"{prefix}_pm": self._format_pm(mean, std),
        }

    def _ci_columns(self, prefix: str, values: Sequence[float]) -> dict[str, object]:
        """The percentile interval of the mean, NaN below the two folds a resampling needs."""
        finite = _finite(values)
        nan = (float("nan"), float("nan"))
        lower, upper = bootstrap_ci(finite) if len(finite) > 1 else nan
        return {f"{prefix}_ci_lower": round(lower, 6), f"{prefix}_ci_upper": round(upper, 6)}

    def _delta_family(self, prefix: str, values: Sequence[float]) -> dict[str, object]:
        """The eight columns of one delta series: the summary, the folds, the interval, the test.

        The test is paired: every value is already the difference between one fold of the run and
        the same fold of its baseline, so the null hypothesis is a true mean of zero.
        """
        finite = _finite(values)
        pvalue = bootstrap_pvalue(finite) if len(finite) > 1 else float("nan")
        columns = self._summary(prefix, values)
        columns[f"{prefix}_folds"] = _join(values)
        columns.update(self._ci_columns(prefix, values))
        columns[f"{prefix}_bootstrap_pvalue"] = round(pvalue, 6)
        columns[f"{prefix}_significant_boot_005"] = "yes" if pvalue < 0.05 else "no"
        return columns

    def _extract_invariance(self, metrics: dict[str, object]) -> dict[str, float]:
        """Pull this fold's invariance metrics out of ``trainer.callback_metrics``."""
        keys = self.INVARIANCE_KEYS + self._FLIP_BREAKDOWN_KEYS + self._DELTA_KEYS
        return {key: self._to_float(metrics.get(key)) for key in keys}

    def _group_series(
        self, inv_folds: list[dict[str, float]], suffix: str
    ) -> dict[str, list[float]]:
        """One series per group: the folds' values of ``invariance/<group>_<suffix>``."""
        nan = float("nan")
        return {
            group: [inv.get(f"invariance/{group}_{suffix}", nan) for inv in inv_folds]
            for group in GROUPS
        }

    def _selected_metric_for_fold(self, inv: dict[str, float]) -> tuple[str, float]:
        """The invariance value the run is judged on, and the key naming it.

        A run trained on one group is judged on it; a baseline belongs to none, so the three groups
        are averaged instead.
        """
        metric = self.invariance_metric
        if self.selected_metric_mode == "scheme" and self.augmentation_scheme in GROUPS:
            key = f"invariance/{self.augmentation_scheme}_{metric}"
            return key, inv.get(key, float("nan"))
        values = [inv.get(key, float("nan")) for key in self.INVARIANCE_KEYS]
        return f"invariance/mean_{metric}", self._safe_mean(values)

    def _selected_group_value(self, inv: dict[str, float], suffix: str) -> float:
        """As :meth:`_selected_metric_for_fold`, for a fixed suffix such as ``flip_recover``."""
        if self.selected_metric_mode == "scheme" and self.augmentation_scheme in GROUPS:
            return inv.get(f"invariance/{self.augmentation_scheme}_{suffix}", float("nan"))
        keys = [f"invariance/{group}_{suffix}" for group in GROUPS]
        return self._safe_mean([inv.get(key, float("nan")) for key in keys])

    def _flip_columns(self, inv_folds: list[dict[str, float]]) -> dict[str, object]:
        """The flip breakdown of the group the run is judged on."""
        columns: dict[str, object] = {}
        for suffix in self.FLIP_BREAKDOWN_SUFFIXES:
            values = [self._selected_group_value(inv, suffix) for inv in inv_folds]
            columns.update(self._summary(suffix, values))
            columns[f"{suffix}_folds"] = _join(values)
        return columns

    def _group_flip_columns(self, inv_folds: list[dict[str, float]]) -> dict[str, object]:
        """The flip breakdown of every group, whichever the run was trained on.

        A baseline's selected columns collapse to a mean over the three groups, so only these let
        it be compared with an SO(2)-trained run on SO(2) alone.
        """
        columns: dict[str, object] = {}
        for suffix in self.FLIP_BREAKDOWN_SUFFIXES:
            for group, values in self._group_series(inv_folds, suffix).items():
                columns.update(self._summary(f"{group}_{suffix}", values))
                columns[f"{group}_{suffix}_folds"] = _join(values)
        return columns

    def _trained_group_delta(self, inv: dict[str, float], suffix: str) -> float:
        """One fold's delta for the group the run was trained on; NaN when it was trained on none.

        A baseline has nothing to be compared against, so the invariance probe logs no ``delta_*``
        key for one and the whole family stays blank.
        """
        if self.augmentation_scheme not in GROUPS:
            return float("nan")
        return inv.get(f"invariance/{self.augmentation_scheme}_delta_{suffix}", float("nan"))

    def _baseline_columns(
        self, baseline_folds: list[float], delta_folds: list[float]
    ) -> dict[str, object]:
        """The baseline's own accuracy and the gain over it, all NaN when there was no baseline."""
        columns = self._summary("baseline_test_acc", baseline_folds)
        columns["baseline_test_acc_folds"] = _join(baseline_folds)
        columns.update(self._delta_family("delta_test_acc", delta_folds))
        return columns

    def _delta_columns(self, inv_folds: list[dict[str, float]]) -> dict[str, object]:
        """One family per delta the probe logged, each read on the group the run was trained on."""
        columns: dict[str, object] = {}
        for suffix in self._DELTA_SUFFIXES:
            values = [self._trained_group_delta(inv, suffix) for inv in inv_folds]
            columns.update(self._delta_family(f"delta_{suffix}", values))
        return columns

    def _group_metric_columns(self, inv_folds: list[dict[str, float]]) -> dict[str, object]:
        """Every group's invariance metric: the summary columns first, then the fold values."""
        metric = self.invariance_metric
        per_group = self._group_series(inv_folds, metric)
        columns: dict[str, object] = {}
        for group, values in per_group.items():
            columns.update(self._summary(f"{group}_{metric}", values))
        for group, values in per_group.items():
            columns[f"{group}_{metric}_folds"] = _join(values)
        return columns

    def _tail_columns(
        self, test_folds: list[float], selected: list[float], flops_folds: list[float]
    ) -> dict[str, object]:
        """The FLOP count and the per-fold values the tables carry last."""
        flops = self._safe_mean(flops_folds)
        return {
            "flops_mean": round(flops, 3) if not math.isnan(flops) else float("nan"),
            "test_acc_folds": _join(test_folds),
            "selected_inv_folds": _join(selected),
        }

    def _head_columns(
        self, test_folds: list[float], selected: list[float], selected_key: str
    ) -> dict[str, object]:
        """The columns naming the run, its test accuracy and the invariance value judged on."""
        row: dict[str, object] = {
            "method": self.display_method,
            "model": self.model_name,
            "augmentation_scheme": self.augmentation_scheme,
            "nconstituents": self.nconstituents,
            "folds_used": len(test_folds),
        }
        row |= self._summary("test_acc", test_folds) | self._ci_columns("test_acc", test_folds)
        row["selected_inv_key"] = selected_key

        return row | self._summary("selected_inv", selected)

    def _build_row(
        self,
        inv_folds: list[dict[str, float]],
        test_folds: list[float],
        flops_folds: list[float],
        baseline_folds: list[float],
        delta_folds: list[float],
    ) -> dict[str, object]:
        """Assemble the row, in the column order the tables on disk already carry."""
        selected_key = self._selected_metric_for_fold(inv_folds[0])[0]
        selected = [self._selected_metric_for_fold(inv)[1] for inv in inv_folds]
        row = self._head_columns(test_folds, selected, selected_key)
        row.update(self._group_metric_columns(inv_folds))
        row.update(self._flip_columns(inv_folds))
        row.update(self._group_flip_columns(inv_folds))
        row.update(self._baseline_columns(baseline_folds, delta_folds))
        row.update(self._delta_columns(inv_folds))
        row.update(self._tail_columns(test_folds, selected, flops_folds))
        return row

    def _folds_ready(self) -> int:
        """How many folds are complete, and so how many the row would average over.

        An eval-only pass never fires ``on_fit_end``, so the invariance metrics and the FLOP count
        come from ``on_test_end`` instead and may lag the accuracy by a fold; the row is still
        written when only the accuracy is in.
        """
        n_test = len(self._fold_test_acc)
        if self._fold_invariance and self._fold_flops:
            return min(n_test, len(self._fold_invariance), len(self._fold_flops))
        return n_test

    def _write_row(self, row: dict[str, object]) -> None:
        """Append the row to the table, writing the header if the table is new."""
        fieldnames = list(row)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.output_path.exists() or self.output_path.stat().st_size == 0
        if not write_header:
            _check_header(self.output_path, fieldnames)
        with open(self.output_path, "a", newline="") as table:
            writer = csv.DictWriter(table, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _first_folds(self, n_ready: int) -> tuple[list, list, list, list, list]:
        """The first *n_ready* entries of every accumulator, in the order _build_row reads them."""
        return (
            self._fold_invariance[:n_ready],
            self._fold_test_acc[:n_ready],
            self._fold_flops[:n_ready],
            self._fold_baseline_test_acc[:n_ready],
            self._fold_delta_test_acc[:n_ready],
        )

    def _log_row(self, row: dict[str, object]) -> None:
        """Announce the appended row, so a queue's log says what landed in which table."""
        log.info(
            f"Results row appended to {self.output_path}: method={self.display_method}, "
            f"model={self.model_name}, group={self.augmentation_scheme}, "
            f"nconst={self.nconstituents}, folds={row['folds_used']}, "
            f"test/acc={row['test_acc_pm']}, {row['selected_inv_key']}="
            f"{row['selected_inv_pm']}, flops={row['flops_mean']}"
        )

    def _maybe_write_summary(self) -> None:
        """Write the row once every fold has been tested; a no-op before then."""
        n_ready = self._folds_ready()
        if self._summary_written or n_ready < max(self.n_folds, 1):
            return
        row = self._build_row(*self._first_folds(n_ready))
        self._write_row(row)
        self._summary_written = True
        self._log_row(row)

    @staticmethod
    def _flops(pl_module: LightningModule) -> float:
        """The forward FLOP count ``record_flops`` left on the algorithm."""
        return ResultsTableCallback._to_float(getattr(pl_module, "flops_total", float("nan")))

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close one fold's training: the invariance metrics and the FLOPs are in."""
        inv = self._extract_invariance(trainer.callback_metrics)
        self._fold_invariance.append(inv)
        self._fold_flops.append(self._flops(pl_module))
        key, value = self._selected_metric_for_fold(inv)
        log.info(
            f"Fold {len(self._fold_invariance)}/{self.n_folds} | {key}={value:.4f} | "
            f"FLOPs={self._fold_flops[-1]:.0f}"
        )

    def _record_test_fold(self, metrics: dict[str, object], fold: int) -> None:
        """Take this fold's invariance metrics, whether or not a fit already recorded them."""
        inv = self._extract_invariance(metrics)
        if len(self._fold_invariance) < fold:
            self._fold_invariance.append(inv)
        elif any(not math.isnan(value) for value in inv.values()):
            # A fit in this same process recorded the fold before the probe had run, since
            # the probe runs on the test set alone. Its values are the ones that count.
            self._fold_invariance[fold - 1] = inv

    def _record_fold(self, metrics: dict[str, object], pl_module: LightningModule) -> int:
        """Take one fold's accuracies, and whatever a fit in this process did not already record.

        :return: the number of folds recorded so far, this one included.
        """
        self._fold_test_acc.append(self._to_float(metrics.get("test/acc")))
        self._fold_baseline_test_acc.append(self._to_float(metrics.get("baseline/test_acc")))
        self._fold_delta_test_acc.append(self._to_float(metrics.get("delta_test_acc")))
        fold = len(self._fold_test_acc)
        self._record_test_fold(metrics, fold)
        if len(self._fold_flops) < fold:
            self._fold_flops.append(self._flops(pl_module))
        return fold

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close one fold's test pass, and write the row when it was the last fold."""
        # An eval-only pass has no fit behind it, so this is where the invariance metrics
        # (logged by InvarianceMetricCallback) and the FLOP count are collected.
        fold = self._record_fold(trainer.callback_metrics, pl_module)
        log.info(f"Fold {fold}/{self.n_folds} | test/acc={self._fold_test_acc[-1]:.4f}")
        self._maybe_write_summary()
