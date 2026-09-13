#!/usr/bin/env python3
"""Render a results table CSV as a standalone LaTeX table.

Formatting only, no statistics. A results row holds one run's own numbers, and the delta column and
the p-value column come from the deltas tables ``compare_to_baseline.py`` writes, named with
``--deltas``: whatever the tables already hold is laid out, and what they do not hold prints as a
dash. Deltas rows are matched on the method, the model, the constituent count and the group, so
several deltas tables can be named at once without their rows being confused.

Usage, one ``.tex`` written beside each input::

    python3 scripts/analysis/compare_to_baseline.py \\
        results/px_py_pz_e/results_table_augmentation.csv \\
        results/px_py_pz_e/results_table_baseline.csv
    python3 scripts/analysis/make_latex_table.py results/px_py_pz_e/results_table_*.csv \\
        --deltas results/px_py_pz_e/deltas_*_vs_baseline.csv
"""

from __future__ import annotations

import argparse
import math
from itertools import groupby
from pathlib import Path

from common import (
    AUG_DISPLAY,
    AUG_ORDER,
    NO_VALUE,
    delta_key,
    detect_inv_metric_suffix,
    index_deltas,
    metric_display,
    model_display,
    read_csv,
    read_deltas,
    safe_fmt,
)


def _group_of(row: dict) -> str:
    """The row's augmentation scheme, lowercased."""
    return str(row.get("augmentation_scheme", "")).lower()


def _sort_key(row: dict) -> tuple[str, int, int]:
    """Model, then constituent count, then the group order the tables report."""
    try:
        nconst = int(row.get("nconstituents", 0) or 0)
    except (ValueError, TypeError):
        nconst = 0
    return (row.get("model", ""), nconst, AUG_ORDER.get(_group_of(row), 99))


def _percent_pair(
    row: dict, first_key: str, second_key: str
) -> tuple[float, float] | None:
    """Two accuracy columns as percentages, None when they do not read as numbers.

    An accuracy is written as a proportion by the callback and as a percentage by the MLflow
    export, so a first value below 2 is taken to be the former and both are scaled.
    """
    try:
        first, second = float(row.get(first_key, "nan")), float(
            row.get(second_key, "nan")
        )
    except (TypeError, ValueError):
        return None
    scale = 100.0 if not math.isnan(first) and first < 2.0 else 1.0
    return first * scale, second * scale


def _acc_cell(row: dict) -> str:
    """The test accuracy as ``mean \\pm std`` in percent, whatever units the row used."""
    pair = _percent_pair(row, "test_acc_mean", "test_acc_std")
    return (
        "N/A" if pair is None else f"{safe_fmt(pair[0], 2)} \\pm {safe_fmt(pair[1], 2)}"
    )


def _ci_cell(row: dict) -> str:
    """The bootstrap confidence interval of the accuracy as ``[lo, hi]`` in percent."""
    pair = _percent_pair(row, "test_acc_ci_lower", "test_acc_ci_upper")
    if pair is None or any(math.isnan(bound) for bound in pair):
        return "N/A"
    return f"[{pair[0]:.1f}, {pair[1]:.1f}]"


def _pm_cell(row: dict, prefix: str, digits: int = 3) -> str:
    """The ``<prefix>_mean`` and ``<prefix>_std`` pair, the mean alone when the std is absent."""
    mean = safe_fmt(row.get(f"{prefix}_mean", "nan"), digits)
    std = safe_fmt(row.get(f"{prefix}_std", "nan"), digits)
    return f"{mean} \\pm {std}" if std != "N/A" else mean


def _pval_cell(pvalue: str | None) -> str:
    """A p-value, collapsing anything below the bootstrap's resolution to ``<0.001``."""
    if pvalue is None or str(pvalue).strip().lower() in NO_VALUE:
        return "—"
    try:
        p = float(pvalue)
    except (TypeError, ValueError):
        return str(pvalue)
    return "N/A" if math.isnan(p) else (r"$<$0.001" if p < 0.001 else f"{p:.3f}")


def _delta_cell(delta: dict | None, digits: int = 3) -> str:
    """One deltas row as ``mean \\pm std``, starred when its bootstrap p-value clears 0.05.

    A dash marks a row the deltas tables hold nothing for: a baseline row, which is the reference
    rather than a run measured against it, or LorentzNet, trained in the baseline regime alone.
    """
    if delta is None or str(delta.get("delta_mean", "")).strip().lower() in NO_VALUE:
        return "—"
    mean = safe_fmt(delta.get("delta_mean"), digits)
    if mean == "N/A":
        return "N/A"
    marker = r"^{*}" if delta.get("significant_boot_005") == "yes" else ""
    return f"${mean} \\pm {safe_fmt(delta.get('delta_std'), digits)}{marker}$"


def _flops_cell(row: dict) -> str:
    """FLOPs in thousands, the unit the published tables quote."""
    try:
        flops = float(row.get("flops_mean", "nan"))
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if math.isnan(flops) else f"{int(flops / 1000)}K"


def _multirow(span: int, body: str) -> str:
    """A cell spanning the ``span`` rows of the block it opens."""
    return f"\\multirow{{{span}}}{{*}}{{{body}}}"


def _blocks(rows: list[dict], column: str) -> list[list[dict]]:
    """Sorted rows cut into the runs of equal *column*, one run per spanning cell and per rule."""
    return [
        list(block)
        for _, block in groupby(rows, key=lambda row: str(row.get(column, "")))
    ]


def _table_row(
    row: dict, cells: tuple[str, str, str], deltas: dict, metric: str
) -> str:
    """One tabular line: the spanning cells, this row's own numbers, then its two deltas."""
    model_cell, nconst_cell, flops_cell = cells
    inv_delta = deltas.get(delta_key(row, _group_of(row), metric))
    acc_delta = deltas.get(delta_key(row, "", "test_acc"))
    return (
        f"    {model_cell} & {nconst_cell} & "
        f"{AUG_DISPLAY.get(_group_of(row), _group_of(row))} & {flops_cell} & "
        f"{_acc_cell(row)} & {_ci_cell(row)} & {_delta_cell(inv_delta)} & "
        f"{_pm_cell(row, 'flip_recover')} & {_pm_cell(row, 'flip_stable')} & "
        f"{_pval_cell(acc_delta.get('bootstrap_pvalue') if acc_delta else None)} \\\\"
    )


def _count_lines(
    block: list[dict], model_cell: str, deltas: dict, metric: str
) -> list[str]:
    """One constituent count of one model: its first line opens the spans, the rest are blank."""
    opening = (
        model_cell,
        _multirow(len(block), str(block[0].get("nconstituents", ""))),
        _multirow(len(block), _flops_cell(block[0])),
    )
    return [
        _table_row(row, opening if index == 0 else ("", "", ""), deltas, metric)
        for index, row in enumerate(block)
    ]


def _body_lines(rows: list[dict], metric: str, deltas: dict) -> list[str]:
    """Every tabular line of the body, over rows already sorted.

    The blocks the rows fall into are both the multirow spans and the rules: a heavy rule opens
    every model but the first, a light one every constituent count but the first of its model.
    """
    lines: list[str] = []
    for model_rows in _blocks(rows, "model"):
        if lines:
            lines.append(r"    \specialrule{.1em}{.5em}{.5em}")
        label = f"\\textbf{{{model_display(str(model_rows[0].get('model', '')))}}}"
        model_cell = _multirow(len(model_rows), label)
        for index, block in enumerate(_blocks(model_rows, "nconstituents")):
            lines += [r"    \cmidrule{2-9}"] if index else []
            lines += _count_lines(
                block, model_cell if index == 0 else "", deltas, metric
            )
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
            r"S[table-format=1.3(3)] S[table-format=1.3(3)] r}"
        ),
        r"    \toprule",
        (
            r"    \textbf{Model} & \textbf{\# Const} & \textbf{Group} & \textbf{FLOPs} & "
            r"{\textbf{Acc. (\%)}} & {\textbf{95\% CI}} & "
            f"\\textbf{{$\\Delta$ {metric_label}}} & "
            r"{\textbf{Flip recover}} & {\textbf{Flip stable}} & "
            r"\textbf{$p$ (boot)} \\"
        ),
        r"    \midrule",
    ]


# The closing rules and the footnote explaining the significance star.
EPILOGUE = [
    r"    \bottomrule",
    r"\end{tabular}",
    r"\end{adjustbox}",
    r"",
    r"\vspace{2pt}",
    r"{\scriptsize $^{*}$ significant at bootstrap $p<0.05$ against the baseline run of the "
    r"same architecture and constituent count.}",
    r"\end{document}",
]


def write_latex_table(
    rows: list[dict],
    output_path: Path,
    inv_metric: str | None = None,
    deltas: dict | None = None,
) -> None:
    """Write ``rows`` as a standalone LaTeX table.

    :param inv_metric: the metric whose delta becomes the delta column, ``flip`` when nothing is
        given.
    :param deltas: the index of :func:`common.index_deltas`; the delta and p-value columns are
        dashes without it.
    """
    if not rows:
        return
    sorted_rows = sorted(rows, key=_sort_key)
    lines = _preamble(metric_display(inv_metric))
    lines += _body_lines(sorted_rows, inv_metric or "flip", deltas or {})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines + EPILOGUE), encoding="utf-8")
    print(f"[write_latex] {len(sorted_rows)}-row table -> {output_path}")


def _parse_args() -> argparse.Namespace:
    """Command line: the results tables to render, and which delta column to show."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "csvs", nargs="+", type=Path, help="results table CSV(s) to render"
    )
    parser.add_argument(
        "--deltas",
        nargs="*",
        type=Path,
        default=[],
        help="Deltas table(s) written by compare_to_baseline.py, holding the delta column and the "
        "p-value column. Without them both print as a dash.",
    )
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
    deltas = index_deltas(read_deltas(args.deltas))
    if not deltas:
        print(
            "[note] no deltas table given: the delta and p-value columns will print as dashes"
        )
    for csv_path in args.csvs:
        rows = read_csv(csv_path) if csv_path.exists() else []
        if not rows:
            print(f"[skip] {csv_path} is missing or empty")
            continue
        metric = args.inv_metric or detect_inv_metric_suffix(rows)
        output_path = csv_path.parent / f"{csv_path.stem}.tex"
        write_latex_table(rows, output_path, inv_metric=metric, deltas=deltas)


if __name__ == "__main__":
    main()
