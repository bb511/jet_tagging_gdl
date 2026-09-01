"""Append the summary row of a run to the results table of its regime.

One row per run rather than per fold: the callback stays attached across the k folds and writes the
row once the last fold has been tested. Tables are split by the input feature list and by the
regime, so ``results/px_py_pz_e/results_table_symm_loss.csv`` holds every symmetry-consistency run
on the four-momentum features, whatever the architecture or the constituent count. Most quantities
occupy four columns (the mean over the folds, its standard deviation, the two joined as
``mean +/- std``, and the fold values joined by ';'), in the column order the tables already on
disk carry, so a new row can be appended to them.
"""

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from lightning import Callback, LightningModule, Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

GROUPS = ("so2", "so3", "lorentz")


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
        self.display_method = _display_regime(method, self.augmentation_scheme)
        self.nconstituents = int(nconstituents)
        self.features_label = "_".join(str(f) for f in features) if features else None
        self.output_path = _table_path(Path(output_path), self.features_label, self.display_method)
        self.selected_metric_mode = selected_metric_mode
        self.invariance_metric = invariance_metric
        self.INVARIANCE_KEYS = tuple(f"invariance/{g}_{invariance_metric}" for g in GROUPS)
        self._FLIP_BREAKDOWN_KEYS = tuple(
            f"invariance/{g}_{suffix}" for suffix in self.FLIP_BREAKDOWN_SUFFIXES for g in GROUPS
        )
        self._fold_invariance: list[dict[str, float]] = []
        self._fold_flops: list[float] = []
        self._fold_test_acc: list[float] = []
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
        """Sample standard deviation over the folds; zero for a single fold."""
        finite = np.asarray(values, dtype=float)
        finite = finite[~np.isnan(finite)]
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

    def _extract_invariance(self, metrics: dict[str, object]) -> dict[str, float]:
        """Pull this fold's invariance metrics out of ``trainer.callback_metrics``."""
        keys = self.INVARIANCE_KEYS + self._FLIP_BREAKDOWN_KEYS
        return {key: self._to_float(metrics.get(key, None)) for key in keys}

    def _selected_metric_for_fold(self, inv: dict[str, float]) -> tuple[str, float]:
        """The invariance value the run is judged on, and the key naming it.

        A run trained on one group is judged on it; a baseline belongs to none, so the three
        groups are averaged instead.
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
            for group in GROUPS:
                key = f"invariance/{group}_{suffix}"
                values = [inv.get(key, float("nan")) for inv in inv_folds]
                columns.update(self._summary(f"{group}_{suffix}", values))
                columns[f"{group}_{suffix}_folds"] = _join(values)
        return columns

    def _build_row(
        self,
        inv_folds: list[dict[str, float]],
        test_folds: list[float],
        flops_folds: list[float],
    ) -> dict[str, object]:
        """Assemble the row, in the column order the tables on disk already carry."""
        metric = self.invariance_metric
        selected_key = self._selected_metric_for_fold(inv_folds[0])[0]
        selected = [self._selected_metric_for_fold(inv)[1] for inv in inv_folds]
        per_group = {
            group: [inv.get(f"invariance/{group}_{metric}", float("nan")) for inv in inv_folds]
            for group in GROUPS
        }
        row: dict[str, object] = {
            "method": self.display_method,
            "model": self.model_name,
            "augmentation_scheme": self.augmentation_scheme,
            "nconstituents": self.nconstituents,
            "folds_used": len(test_folds),
            **self._summary("test_acc", test_folds),
            "selected_inv_key": selected_key,
            **self._summary("selected_inv", selected),
        }
        for group in GROUPS:
            row.update(self._summary(f"{group}_{metric}", per_group[group]))
        for group in GROUPS:
            row[f"{group}_{metric}_folds"] = _join(per_group[group])
        row.update(self._flip_columns(inv_folds))
        row.update(self._group_flip_columns(inv_folds))
        flops = self._safe_mean(flops_folds)
        row["flops_mean"] = round(flops, 3) if not math.isnan(flops) else float("nan")
        row["test_acc_folds"] = _join(test_folds)
        row["selected_inv_folds"] = _join(selected)
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

    def _maybe_write_summary(self) -> None:
        """Write the row once every fold has been tested; a no-op before then."""
        n_ready = self._folds_ready()
        if self._summary_written or n_ready < max(self.n_folds, 1):
            return
        row = self._build_row(
            self._fold_invariance[:n_ready],
            self._fold_test_acc[:n_ready],
            self._fold_flops[:n_ready],
        )
        self._write_row(row)
        self._summary_written = True
        log.info(
            f"Results row appended to {self.output_path}: method={self.display_method}, "
            f"model={self.model_name}, group={self.augmentation_scheme}, "
            f"nconst={self.nconstituents}, folds={row['folds_used']}, "
            f"test/acc={row['test_acc_pm']}, {row['selected_inv_key']}="
            f"{row['selected_inv_pm']}, flops={row['flops_mean']}"
        )

    @staticmethod
    def _flops(pl_module: LightningModule) -> float:
        """The forward FLOP count the algorithm recorded in its hyperparameters."""
        hparams = pl_module.hparams
        return ResultsTableCallback._to_float(
            hparams.get("flops_total", hparams.get("flops", float("nan")))
        )

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

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close one fold's test pass, and write the row when it was the last fold."""
        metrics = trainer.callback_metrics
        self._fold_test_acc.append(self._to_float(metrics.get("test/acc", None)))
        fold = len(self._fold_test_acc)
        log.info(f"Fold {fold}/{self.n_folds} | test/acc={self._fold_test_acc[-1]:.4f}")
        # An eval-only pass has no fit behind it, so this is where the invariance metrics
        # (logged by InvarianceMetricCallback) and the FLOP count are collected.
        inv = self._extract_invariance(metrics)
        if len(self._fold_invariance) < fold:
            self._fold_invariance.append(inv)
        elif any(not math.isnan(value) for value in inv.values()):
            # A fit in this same process recorded the fold before the probe had run, since
            # the probe runs on the test set alone. Its values are the ones that count.
            self._fold_invariance[fold - 1] = inv
        if len(self._fold_flops) < fold:
            self._fold_flops.append(self._flops(pl_module))
        self._maybe_write_summary()
