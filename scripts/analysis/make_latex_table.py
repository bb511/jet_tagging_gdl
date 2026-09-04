#!/usr/bin/env python3
"""Render a results table CSV as a standalone LaTeX table.

Formatting only, no statistics. ``src/eval.py`` scores a run against the baseline run named by
``baseline_checkpoint_manifest_path`` and the results table callback writes every delta,
confidence-interval and p-value column at that point, so nothing is computed here: whatever
columns a table already holds are laid out, and the ones it does not hold print as a dash.

Usage, one ``.tex`` written beside each input::

    python3 scripts/analysis/make_latex_table.py results/px_py_pz_e/results_table_*.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a results table into one dict per row, keyed by column name.

    Short rows are padded rather than truncated, so a table whose header grew between runs still
    yields every column the header names.

    :param path: a ``results_table_<regime>.csv`` written by the results table callback.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    return [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows]


# The flip breakdown is written under the same per-group prefix as the invariance metric, so
# these three are never what callbacks.results_table.invariance_metric named.
_BREAKDOWN_FOLDS = {"flip_recover", "flip_stable", "flip_degrade"}


def _detect_inv_metric_suffix(rows: list[dict[str, str]]) -> str | None:
    """Infer the invariance metric (``wasserstein``, ``flip``) from a ``so2_<metric>_folds``
    column."""
    for row in rows:
        for column in row:
            if column.startswith("so2_") and column.endswith("_folds"):
                metric = column[len("so2_") : -len("_folds")]
                if metric not in _BREAKDOWN_FOLDS:
                    return metric
    return None


AUG_ORDER = {"none": 0, "lorentz": 1, "so2": 2, "so3": 3}
AUG_DISPLAY = {"none": "None", "lorentz": "Lorentz", "so2": "SO(2)", "so3": "SO(3)"}

# The model column holds algorithm.name for tables written since the refactor and a full class
# path for the ones written before it, so both are matched. Transformer is JetFormer's old name.
_MODEL_DISPLAY = {
    "deepsets": "DeepSets",
    "mlp": "MLP",
    "jetformer": "JetFormer",
    "lorentznet": "LorentzNet",
    "transformer": "JetFormer",
}

_METRIC_DISPLAY = {
    "wasserstein": "Wasserstein",
    "flip": "Flip rate",
    "accuracy_transformed": "Accuracy (transformed)",
}

# Sentinels the results table writes in place of a number when a row has no baseline to compare
# with: the baseline rows themselves, and LorentzNet, which is trained in that regime alone.
_NO_VALUE = {"", "nan", "n/a", "baseline", "no_baseline", "no_folds", "—"}


def _model_display(model_raw: str) -> str:
    """Map a short algorithm name or a full class path to the label the tables print."""
    key = model_raw.lower().split(".")[-1]
    if key in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[key]
    for name, label in _MODEL_DISPLAY.items():
        if name in key:
            return label
    return model_raw.split(".")[-1]


def _metric_display(metric: str | None) -> str:
    """The column heading for an invariance metric suffix."""
    if not metric:
        return "Metric"
    return _METRIC_DISPLAY.get(metric, metric.capitalize())


def _safe_fmt(value, digits: int) -> str:
    """Fixed-point format, ``N/A`` for anything that is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if math.isnan(number) else f"{number:.{digits}f}"


def _fmt_inv_cell(row: dict, mean_key: str, std_key: str, digits: int = 3) -> str:
    """Format an invariance cell as ``mean \\pm std``, or the mean alone when the std is absent."""
    mean = _safe_fmt(row.get(mean_key, "nan"), digits)
    std = _safe_fmt(row.get(std_key, "nan"), digits)
    return f"{mean} \\pm {std}" if std != "N/A" else mean


def _pval_cell(pvalue: str | None) -> str:
    """Format a p-value, collapsing anything below the bootstrap's resolution to ``<0.001``."""
    if pvalue is None or str(pvalue).strip().lower() in _NO_VALUE:
        return "—"
    try:
        p = float(pvalue)
    except (TypeError, ValueError):
        return str(pvalue)
    return "N/A" if math.isnan(p) else (r"$<$0.001" if p < 0.001 else f"{p:.3f}")


def _fmt_delta_cell(row: dict, metric: str, digits: int = 3) -> str:
    """Format ``delta_<metric>`` as ``mean \\pm std``, starred when the bootstrap p-value clears
    0.05.

    A dash marks a row with no baseline to be measured against: a baseline row itself, or
    LorentzNet, which is trained in the baseline regime alone.
    """
    if str(row.get(f"delta_{metric}_mean", "")).strip().lower() in _NO_VALUE:
        return "—"
    mean = _safe_fmt(row.get(f"delta_{metric}_mean"), digits)
    std = _safe_fmt(row.get(f"delta_{metric}_std"), digits)
    if mean == "N/A":
        return "N/A"
    marker = r"^{*}" if row.get(f"delta_{metric}_significant_boot_005") == "yes" else ""
    return f"${mean} \\pm {std}{marker}$"


def _ci_cell(
    row: dict,
    digits: int = 1,
    lo_key: str = "test_acc_ci_lower",
    hi_key: str = "test_acc_ci_upper",
) -> str:
    """Format a bootstrap confidence interval as ``[lo, hi]`` in percent.

    An accuracy is written as a proportion by the callback and as a percentage by the MLflow
    export, so a bound below 2 is taken to be the former and scaled.
    """
    try:
        lo, hi = float(row.get(lo_key, "nan")), float(row.get(hi_key, "nan"))
    except (TypeError, ValueError):
        return "N/A"
    if math.isnan(lo) or math.isnan(hi):
        return "N/A"
    scale = 100.0 if lo < 2.0 else 1.0
    return f"[{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def _acc_cell(row: dict) -> str:
    """Format the test accuracy as ``mean \\pm std`` in percent, whatever units the row used."""
    try:
        mean, std = float(row.get("test_acc_mean", "nan")), float(row.get("test_acc_std", "nan"))
    except (TypeError, ValueError):
        return "N/A"
    scale = 100.0 if not math.isnan(mean) and mean < 2.0 else 1.0
    return f"{_safe_fmt(mean * scale, 2)} \\pm {_safe_fmt(std * scale, 2)}"


def _flops_display(row: dict) -> str:
    """FLOPs in thousands, the unit the published tables quote."""
    try:
        flops = float(row.get("flops_mean", "nan"))
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if math.isnan(flops) else f"{int(flops / 1000)}K"


def _sort_key(row: dict) -> tuple[str, int, int]:
    """Model, then constituent count, then the group order the tables report."""
    try:
        nconst = int(row.get("nconstituents", 0) or 0)
    except (ValueError, TypeError):
        nconst = 0
    return (row.get("model", ""), nconst, AUG_ORDER.get(_group_of(row), 99))


def _group_of(row: dict) -> str:
    """The row's augmentation scheme, lowercased."""
    return str(row.get("augmentation_scheme", "")).lower()


def _multirow(span: int, body: str) -> str:
    """A cell spanning ``span`` rows of the group it opens."""
    return f"\\multirow{{{span}}}{{*}}{{{body}}}"


def _spans(rows: list[dict]) -> tuple[Counter, Counter]:
    """How many rows each spanning cell covers: one count per model, one per (model, count)."""
    return (
        Counter(row.get("model", "") for row in rows),
        Counter((row.get("model", ""), row.get("nconstituents", "")) for row in rows),
    )


def _rule_before(key: tuple[str, str], previous: tuple[str, str] | None) -> list[str]:
    """The rule separating a row from the one above: heavy between models, light between counts."""
    if previous is None or key == previous:
        return []
    if key[0] != previous[0]:
        return [r"    \specialrule{.1em}{.5em}{.5em}"]
    return [r"    \cmidrule{2-10}"]


def _spanning_cells(
    row: dict,
    key: tuple[str, str],
    previous: tuple[str, str] | None,
    spans: tuple[Counter, Counter],
) -> tuple[str, str, str]:
    """The model, constituent-count and FLOPs cells, blank on a row an open ``multirow`` covers."""
    model_span, nconst_span = spans
    if key == previous:
        return "", "", ""
    new_model = previous is None or key[0] != previous[0]
    label = f"\\textbf{{{_model_display(key[0])}}}"
    model_cell = _multirow(model_span[key[0]], label) if new_model else ""
    nconst_cell = _multirow(nconst_span[key], key[1])
    return model_cell, nconst_cell, _multirow(nconst_span[key], _flops_display(row))


def _table_row(row: dict, metric_key: str, cells: tuple[str, str, str]) -> str:
    """One tabular line: the spanning cells, then this row's own numbers."""
    model_cell, nconst_cell, flops_cell = cells
    return (
        f"    {model_cell} & {nconst_cell} & "
        f"{AUG_DISPLAY.get(_group_of(row), _group_of(row))} & {flops_cell} & "
        f"{_acc_cell(row)} & {_ci_cell(row)} & {_fmt_delta_cell(row, metric_key)} & "
        f"{_fmt_inv_cell(row, 'flip_recover_mean', 'flip_recover_std')} & "
        f"{_fmt_inv_cell(row, 'flip_stable_mean', 'flip_stable_std')} & "
        f"{_fmt_inv_cell(row, 'flip_degrade_mean', 'flip_degrade_std')} & "
        f"{_pval_cell(row.get('delta_test_acc_bootstrap_pvalue'))} \\\\"
    )


def _body_lines(rows: list[dict], metric_key: str) -> list[str]:
    """Every tabular line of the body, rules included, over rows already sorted."""
    spans = _spans(rows)
    lines: list[str] = []
    previous: tuple[str, str] | None = None
    for row in rows:
        key = (row.get("model", ""), row.get("nconstituents", ""))
        cells = _spanning_cells(row, key, previous, spans)
        lines += _rule_before(key, previous) + [_table_row(row, metric_key, cells)]
        previous = key
    return lines


def _preamble(metric_label: str) -> list[str]:
    """The standalone document, the tabular declaration and the header row."""
    return [
        r"\documentclass[preview,border=1pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs, siunitx, adjustbox, multirow}",
        r"\sisetup{separate-uncertainty=true, bracket-numbers=false}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\scriptsize",
        r"\begin{adjustbox}{max width=\textwidth}",
        (
            r"\begin{tabular}{ll l c S[table-format=2.2(2)] l l "
            r"S[table-format=1.3(3)] S[table-format=1.3(3)] S[table-format=1.3(3)] r}"
        ),
        r"    \toprule",
        (
            r"    \textbf{Model} & \textbf{\# Const} & \textbf{Group} & \textbf{FLOPs} & "
            r"{\textbf{Acc. (\%)}} & {\textbf{95\% CI}} & "
            f"\\textbf{{$\\Delta$ {metric_label}}} & "
            r"{\textbf{Flip recover}} & {\textbf{Flip stable}} & {\textbf{Flip degrade}} & "
            r"\textbf{$p$ (boot)} \\"
        ),
        r"    \midrule",
    ]


def _epilogue() -> list[str]:
    """The closing rules and the footnote explaining the significance star."""
    return [
        r"    \bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"",
        r"\vspace{2pt}",
        r"{\scriptsize $^{*}$ significant at bootstrap $p<0.05$ against the baseline run of the "
        r"same architecture and constituent count.}",
        r"\end{document}",
    ]


def write_latex_table(rows: list[dict], output_path: Path, inv_metric: str | None = None) -> None:
    """Write ``rows`` as a standalone LaTeX table.

    :param inv_metric: which ``delta_<metric>`` column becomes the delta column, ``flip`` when
        nothing is given.
    """
    if not rows:
        return
    sorted_rows = sorted(rows, key=_sort_key)
    lines = _preamble(_metric_display(inv_metric))
    lines += _body_lines(sorted_rows, inv_metric or "flip")
    lines += _epilogue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write_latex] {len(sorted_rows)}-row table -> {output_path}")


def _parse_args() -> argparse.Namespace:
    """Command line: the results tables to render, and which delta column to show."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csvs", nargs="+", type=Path, help="results table CSV(s) to render")
    parser.add_argument(
        "--inv-metric",
        default=None,
        help="Invariance metric shown as the delta column ('wasserstein', 'flip'). "
        "Read off the '<group>_<metric>_folds' columns when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    """Render each table named on the command line as a ``.tex`` beside it."""
    args = _parse_args()
    for csv_path in args.csvs:
        rows = read_csv(csv_path) if csv_path.exists() else []
        if not rows:
            print(f"[skip] {csv_path} is missing or empty")
            continue
        metric = args.inv_metric or _detect_inv_metric_suffix(rows)
        write_latex_table(rows, csv_path.parent / f"{csv_path.stem}.tex", inv_metric=metric)


if __name__ == "__main__":
    main()
