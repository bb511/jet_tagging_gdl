"""Append the summary row of an evaluated training to the results table of its regime.

One row per training with the following columns per run:
* the test accuracy
* FLOP count
* invariance metrics

These quantities are averaged over the folds. The per-fold values are also saved.
The evaluation calls this method once it completed running inference over each fold.

The generated tables are partitioned depending on the features consumed by the model
as well as the methodology used to learn symmetry invariances.
For the SCL and augmentation methods, we save the results table in a folder whose name
is the features ingested by that model. For example,
```
results/px_py_pz_e/results_table_symm_loss.csv
```
which contains all results for the MLP, DS, JetFormer across different constituents.
For the architectures that have symmetry embedded into them, the results go into the
``results/embedded`` folder.
"""

import csv
import math
import statistics
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig

from src.symmetry import FLIP_FAMILIES, TABULATED_GROUPS, key
from src.utils.bootstrap import bootstrap_ci
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ResultsTable:
    """Turn the folds of one evaluated training into a single row of its regime's table.

    :param output_path: base name of the table; `_table_path` weaves the regime and the
        subdirectory into it.
    :param model_name: short architecture label, ``algorithm.name``.
    :param features: the input feature list, labels the subdirectory unless ``subdir``.
    :param subdir: directory under the table's parent in place of the feature list.
    :param selected_metric_mode: with ``scheme`` the run is judged on the group it was trained on,
        and on the mean over the three groups when it was trained on none; ``mean_all`` always
        reports that mean.
    :param invariance_metric: which invariance family fills the selected and per-group columns,
        e.g., ``wasserstein`` reads ``invariance/so2_wasserstein``.
    """

    def __init__(
        self,
        output_path: str,
        method: str = "augmentation",
        model_name: str = "model",
        augmentation_scheme: str = "none",
        nconstituents: int = 0,
        features: Sequence[str] | None = None,
        subdir: str | None = None,
        selected_metric_mode: str = "scheme",
        invariance_metric: str = "wasserstein",
    ) -> None:
        self.model_name = model_name
        self.augmentation_scheme = str(augmentation_scheme).lower()
        self.nconstituents = int(nconstituents)
        self.invariance_metric = invariance_metric
        self.judged_group = _judged_group(selected_metric_mode, self.augmentation_scheme)
        # A selected metric that is itself tabulated would fill two identical column families.
        self.suffixes = tuple(s for s in FLIP_FAMILIES if s != invariance_metric)
        self.display_method = _display_regime(method, self.augmentation_scheme)
        label = subdir or ("_".join(str(f) for f in features) if features else None)
        self.output_path = _table_path(Path(output_path), label, self.display_method)

    def write(self, folds: list[dict[str, object]]) -> None:
        """Append one row summarising *folds*, each one fold's metrics and its ``flops_total``."""
        if not folds:
            log.warning("No fold was evaluated, so no results row is written.")
            return
        row = self._row(folds)
        _append(self.output_path, row)
        self._log(row)

    def _values(
        self, folds: list[dict[str, object]], suffix: str, group: str | None = None
    ) -> list[float]:
        """One value per fold of *suffix*, on *group* or on the group the run is judged on.

        A run judged on no group reads the mean over the three groups instead.
        """
        group = group or self.judged_group
        if group:
            return [_to_float(fold.get(key(group, suffix))) for fold in folds]
        per_group = [self._values(folds, suffix, other) for other in TABULATED_GROUPS]
        return [_mean(fold_values) for fold_values in zip(*per_group)]

    def _metric_columns(self, folds: list[dict[str, object]]) -> dict[str, object]:
        """The families that vary by group: the selected metric on each group, then every tabulated
        suffix on the judged group and once more on each."""
        metric = self.invariance_metric
        columns: dict[str, object] = {}
        for group in TABULATED_GROUPS:
            columns |= _family(f"{group}_{metric}", self._values(folds, metric, group))
        for suffix in self.suffixes:
            columns |= _family(suffix, self._values(folds, suffix))
        for suffix in self.suffixes:
            for group in TABULATED_GROUPS:
                columns |= _family(f"{group}_{suffix}", self._values(folds, suffix, group))
        return columns

    def _row(self, folds: list[dict[str, object]]) -> dict[str, object]:
        """Assemble the row, in the column order the tables on disk carry."""
        metric = self.invariance_metric
        test_acc = [_to_float(fold.get("test/acc")) for fold in folds]
        selected = self._values(folds, metric)
        flops = _mean([_to_float(fold.get("flops_total")) for fold in folds])
        row: dict[str, object] = {
            "method": self.display_method,
            "model": self.model_name,
            "augmentation_scheme": self.augmentation_scheme,
            "nconstituents": self.nconstituents,
            "folds_used": len(folds),
        }
        row |= _summary("test_acc", test_acc) | _ci("test_acc", test_acc)
        row["selected_inv_key"] = key(self.judged_group or "mean", metric)
        row |= _summary("selected_inv", selected) | self._metric_columns(folds)
        row["flops_mean"] = round(flops, 3) if not math.isnan(flops) else float("nan")
        row["test_acc_folds"] = _join(test_acc)
        row["selected_inv_folds"] = _join(selected)
        return row

    def _log(self, row: dict[str, object]) -> None:
        """Announce the appended row, so a queue's log says what landed in which table."""
        log.info(
            f"Results row appended to {self.output_path}: method={self.display_method}, "
            f"model={self.model_name}, group={self.augmentation_scheme}, "
            f"nconst={self.nconstituents}, folds={row['folds_used']}, "
            f"test/acc={row['test_acc_pm']}, {row['selected_inv_key']}="
            f"{row['selected_inv_pm']}, flops={row['flops_mean']}"
        )


def write_results_row(cfg: DictConfig, folds: list[dict[str, object]]) -> None:
    """Summarise *folds* into one row of the table ``cfg.results_table`` names.

    A composition without that node writes nothing, which is how ``debug`` keeps a throwaway
    evaluation out of ``results/``.
    """
    if not cfg.get("results_table"):
        log.info("No results_table config found, so no row is written.")
        return
    ResultsTable(
        output_path=cfg.results_table.output_path,
        method=cfg.data.method,
        model_name=cfg.algorithm.name,
        augmentation_scheme=cfg.data.augmentation_scheme,
        nconstituents=cfg.data.nconstituents,
        features=cfg.data.features,
        subdir=cfg.results_table.subdir,
        selected_metric_mode=cfg.results_table.selected_metric_mode,
        invariance_metric=cfg.results_table.invariance_metric,
    ).write(folds)


def _judged_group(selected_metric_mode: str, augmentation_scheme: str) -> str | None:
    """The group the run is judged on, None when it belongs to none and the three are averaged."""
    judged = selected_metric_mode == "scheme" and augmentation_scheme in TABULATED_GROUPS
    return augmentation_scheme if judged else None


def _display_regime(method: str, augmentation_scheme: str) -> str:
    """Name the regime as the tables report it: a run with no symmetry is the baseline."""
    if method == "augmentation" and augmentation_scheme == "none":
        return "baseline"
    return method


def _table_path(base: Path, subdir: str | None, display_method: str) -> Path:
    """Weave the subdirectory and the regime into the table name.

    ``results/results_table.csv`` becomes ``results/px_py_pz_e/results_table_baseline.csv``.
    """
    directory = base.parent / subdir if subdir else base.parent
    return directory / f"{base.stem}_{display_method}{base.suffix}"


def _to_float(value) -> float:
    """Read a metric out of Lightning's dicts, whether it is a tensor, a number or absent."""
    if value is None:
        return float("nan")
    return float(value.item()) if hasattr(value, "item") else float(value)


def _mean(values: Sequence[float]) -> float:
    """Mean over the folds that reported the metric; NaN when none did."""
    finite = [value for value in values if not math.isnan(value)]
    return statistics.fmean(finite) if finite else float("nan")


def _join(values: Sequence[float]) -> str:
    """Join the per-fold values into one cell, six decimals each."""
    return ";".join("nan" if math.isnan(value) else f"{value:.6f}" for value in values)


def _summary(prefix: str, values: Sequence[float]) -> dict[str, object]:
    """The mean over the folds, its sample deviation, and the two rendered as one cell."""
    finite = [value for value in values if not math.isnan(value)]
    mean = _mean(values)
    # One fold has no spread to report, which is not the same as no fold having reported.
    std = statistics.stdev(finite) if len(finite) > 1 else (0.0 if finite else float("nan"))
    return {
        f"{prefix}_mean": round(mean, 6),
        f"{prefix}_std": round(std, 6),
        f"{prefix}_pm": "nan" if math.isnan(mean) else f"{mean:.4f} +/- {std:.4f}",
    }


def _family(prefix: str, values: Sequence[float]) -> dict[str, object]:
    """The four columns of one series: the summary, then the folds behind it."""
    return _summary(prefix, values) | {f"{prefix}_folds": _join(values)}


def _ci(prefix: str, values: Sequence[float]) -> dict[str, object]:
    """The percentile interval of the mean, NaN below the two folds a resampling needs."""
    finite = [value for value in values if not math.isnan(value)]
    nan = (float("nan"), float("nan"))
    lower, upper = bootstrap_ci(finite) if len(finite) > 1 else nan
    return {f"{prefix}_ci_lower": round(lower, 6), f"{prefix}_ci_upper": round(upper, 6)}


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


def _append(path: Path, row: dict[str, object]) -> None:
    """Append the row to the table, writing the header if the table is new."""
    fieldnames = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        _check_header(path, fieldnames)
    with open(path, "a", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
