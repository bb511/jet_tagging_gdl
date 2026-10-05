"""Append the summary row of an evaluated training to the results table of its regime.

One row per training with the following columns per run:
* the test accuracy
* the clean working points and the one-vs-rest AUC, as macro means over the classes
* the FLOP and eBOP counts per jet
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

from src.symmetry import (
    PERFORMANCE_FAMILIES,
    TABULATED_FAMILIES,
    TABULATED_GROUPS,
    TOTAL_VARIATION,
    key,
)
from src.utils.bootstrap import bootstrap_ci
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _fold_values(folds: list[dict[str, object]], name: str) -> list[float]:
    return [_to_float(fold.get(name)) for fold in folds]


def _performance_columns(folds: list[dict[str, object]]) -> dict[str, object]:
    """The groupless families: the clean working points and the AUC, as macro means."""
    columns: dict[str, object] = {}
    for family in PERFORMANCE_FAMILIES:
        columns |= _family(family, _fold_values(folds, key(None, family)))
    return columns


def _cost_columns(folds: list[dict[str, object]]) -> dict[str, object]:
    """The FLOPs and the eBOPs per jet, means over the folds that report them."""
    return {
        f"{cost}_mean": round(_mean(_fold_values(folds, f"{cost}_total")), 3)
        for cost in ("flops", "ebops")
    }


class ResultsTable:
    """Turn the folds of one evaluated training into a single row of its regime's table.

    :param output_path: base name of the table; `_table_path` weaves the regime and the
        subdirectory into it.
    :param model_name: short architecture label, ``algorithm.name``.
    :param augmentation_scheme: the group the run was trained on and is judged on; a run
        trained on none of `src.symmetry.TABULATED_GROUPS` is judged on the mean over
        the three.
    :param features: the input feature list, labels the subdirectory unless ``subdir``.
    :param subdir: directory under the table's parent in place of the feature list.
    :param invariance_metric: which invariance family fills the selected and per-group
        columns, e.g., ``tv`` reads ``invariance/so2_tv``.
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
        invariance_metric: str = TOTAL_VARIATION,
    ) -> None:
        self.model_name = model_name
        self.augmentation_scheme = str(augmentation_scheme).lower()
        self.nconstituents = int(nconstituents)
        self.invariance_metric = invariance_metric
        self.judged_group = _judged_group(self.augmentation_scheme)
        # A selected metric that is itself tabulated would fill two identical column families.
        self.suffixes = tuple(s for s in TABULATED_FAMILIES if s != invariance_metric)
        self.display_method = _display_regime(method, self.augmentation_scheme)
        label = subdir or ("_".join(str(f) for f in features) if features else None)
        self.output_path = _table_path(Path(output_path), label, self.display_method)

    def write(self, folds: list[dict[str, object]]) -> None:
        """Append one row summarising *folds*, each one fold's metrics with its
        ``flops_total`` and ``ebops_total``."""
        if not folds:
            log.warning("No fold was evaluated, so no results row is written.")
            return
        row = self._row(folds)
        _append(self.output_path, row)
        self._log(row)

    def _values(
        self, folds: list[dict[str, object]], suffix: str, group: str | None = None
    ) -> list[float]:
        """One value per fold of *suffix*, on *group* or on the group the run is judged
        on.

        A run judged on no group reads the mean over the three groups instead.
        """
        group = group or self.judged_group
        if group:
            return _fold_values(folds, key(group, suffix))
        per_group = [self._values(folds, suffix, other) for other in TABULATED_GROUPS]
        return [_mean(fold_values) for fold_values in zip(*per_group)]

    def _metric_columns(self, folds: list[dict[str, object]]) -> dict[str, object]:
        """The families that vary by group: the selected metric on each group, then
        every tabulated suffix on the judged group and once more on each."""
        metric = self.invariance_metric
        columns: dict[str, object] = {}
        for group in TABULATED_GROUPS:
            columns |= _family(f"{group}_{metric}", self._values(folds, metric, group))
        for suffix in self.suffixes:
            columns |= _family(suffix, self._values(folds, suffix))
        for suffix in self.suffixes:
            for group in TABULATED_GROUPS:
                columns |= _family(
                    f"{group}_{suffix}", self._values(folds, suffix, group)
                )
        return columns

    def _head(self, folds: list[dict[str, object]]) -> dict[str, object]:
        """The columns naming the run, then its test accuracy."""
        test_acc = _fold_values(folds, "test/acc")
        row: dict[str, object] = {
            "method": self.display_method,
            "model": self.model_name,
            "augmentation_scheme": self.augmentation_scheme,
            "nconstituents": self.nconstituents,
            "folds_used": len(folds),
        }
        return row | _summary("test_acc", test_acc) | _ci("test_acc", test_acc)

    def _row(self, folds: list[dict[str, object]]) -> dict[str, object]:
        """Assemble the row, in the column order the tables on disk carry."""
        metric = self.invariance_metric
        selected = self._values(folds, metric)
        row = self._head(folds) | _performance_columns(folds)
        row["selected_inv_key"] = key(self.judged_group or "mean", metric)
        row |= _summary("selected_inv", selected) | self._metric_columns(folds)
        row |= _cost_columns(folds)
        row["test_acc_folds"] = _join(_fold_values(folds, "test/acc"))
        row["selected_inv_folds"] = _join(selected)
        return row

    def _log(self, row: dict[str, object]) -> None:
        """Announce the appended row, so a queue's log says what landed in which
        table."""
        log.info(
            f"Results row appended to {self.output_path}: method={self.display_method}, "
            f"model={self.model_name}, group={self.augmentation_scheme}, "
            f"nconst={self.nconstituents}, folds={row['folds_used']}, "
            f"test/acc={row['test_acc_pm']}, {row['selected_inv_key']}="
            f"{row['selected_inv_pm']}, flops={row['flops_mean']}, "
            f"ebops={row['ebops_mean']}"
        )


def write_results_row(cfg: DictConfig, folds: list[dict[str, object]]) -> None:
    """Summarise *folds* into one row of the table ``cfg.results_table`` names.

    A composition without that node writes nothing, which is how ``debug`` keeps a
    throwaway evaluation out of ``results/``.
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
        invariance_metric=cfg.results_table.invariance_metric,
    ).write(folds)


def _judged_group(augmentation_scheme: str) -> str | None:
    """The group the run is judged on, None when it belongs to none and the three are
    averaged."""
    return augmentation_scheme if augmentation_scheme in TABULATED_GROUPS else None


def _display_regime(method: str, augmentation_scheme: str) -> str:
    """Name the regime as the tables report it: a run with no symmetry is the
    baseline."""
    if method == "augmentation" and augmentation_scheme == "none":
        return "baseline"
    return method


def _table_path(base: Path, subdir: str | None, display_method: str) -> Path:
    """Weave the subdirectory and the regime into the table name.

    ``results/results_table.csv`` becomes
    ``results/px_py_pz_e/results_table_baseline.csv``.
    """
    directory = base.parent / subdir if subdir else base.parent
    return directory / f"{base.stem}_{display_method}{base.suffix}"


def _to_float(value) -> float:
    """Read a metric out of Lightning's dicts, whether it is a tensor, a number or
    absent."""
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
    """The mean over the folds, its sample deviation, and the two rendered as one
    cell."""
    finite = [value for value in values if not math.isnan(value)]
    mean = _mean(values)
    # One fold has no spread to report, which is not the same as no fold having reported.
    std = (
        statistics.stdev(finite)
        if len(finite) > 1
        else (0.0 if finite else float("nan"))
    )
    return {
        f"{prefix}_mean": round(mean, 6),
        f"{prefix}_std": round(std, 6),
        f"{prefix}_pm": "nan" if math.isnan(mean) else f"{mean:.4f} +/- {std:.4f}",
    }


def _family(prefix: str, values: Sequence[float]) -> dict[str, object]:
    """The four columns of one series: the summary, then the folds behind it."""
    return _summary(prefix, values) | {f"{prefix}_folds": _join(values)}


def _ci(prefix: str, values: Sequence[float]) -> dict[str, object]:
    """The percentile interval of the mean, NaN below the two folds a resampling
    needs."""
    finite = [value for value in values if not math.isnan(value)]
    nan = (float("nan"), float("nan"))
    lower, upper = bootstrap_ci(finite) if len(finite) > 1 else nan
    return {
        f"{prefix}_ci_lower": round(lower, 6),
        f"{prefix}_ci_upper": round(upper, 6),
    }


def _check_header(path: Path, fieldnames: list[str]) -> None:
    """Refuse to append under a header with other columns, which would misalign the
    row."""
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
