#!/usr/bin/env python3
"""Add paired t-test p-values to results table CSVs.

For each (model, nconstituents) group the row with augmentation_scheme=none
is used as the baseline.  A two-sided paired t-test (scipy.stats.ttest_rel)
compares every other row against that baseline using the per-fold test
accuracies stored in the test_acc_folds column.

Why paired: KFold uses random_state=kfold_seed (default 42) and the test set
is a fixed held-out split, so fold i of condition A and fold i of condition B
are trained/tested on identical data — pairing is valid and more powerful than
an independent-samples test.

Caveat: with n=5 folds the test has low power (~0.3 for small effects).
Interpret p-values qualitatively; consider them alongside effect sizes.

Usage
-----
Single table (baseline rows must be present):
    python src/utils/add_pvalues.py results/results_table_augmentation.csv

Cross-table comparison (e.g. symm_loss vs augmentation baseline):
    python src/utils/add_pvalues.py results/results_table_symm_loss.csv \
        --baseline results/results_table_augmentation.csv

Multiple tables at once:
    python src/utils/add_pvalues.py results/results_table_*.csv

Outputs are written next to each input CSV with a _pvalues suffix, and a
corresponding _pvalues.tex LaTeX table is also written.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy import stats as scipy_stats
except ImportError:
    sys.exit("scipy is required: pip install scipy")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _parse_folds(raw: str) -> Optional[List[float]]:
    """Parse semicolon-separated fold accuracies; return None if unparseable."""
    if not raw or raw.strip().lower() in {"", "nan"}:
        return None
    try:
        vals = [float(v) for v in raw.strip().split(";")]
        if any(math.isnan(v) for v in vals):
            return None
        return vals
    except ValueError:
        return None


_FLIP_METRICS = ("so2_flip", "so3_flip", "lorentz_flip")
_FLIP_SUFFIXES = ("mean", "std", "pm")
_FLIP_FULL_COLS = [f"{m}_{s}" for m in _FLIP_METRICS for s in _FLIP_SUFFIXES]
_FLIP_MEAN_ONLY = [f"{m}_mean" for m in _FLIP_METRICS]


def _fix_header(header: List[str], n_data_cols: int) -> List[str]:
    """Expand a header that lists only flip means to one with mean/std/pm.

    Some CSVs (e.g. symm_loss) were written with headers that declare only
    so2_flip_mean, so3_flip_mean, lorentz_flip_mean but the data rows have
    mean+std+pm for each metric (6 extra columns).  This causes DictReader to
    map column names to the wrong data positions.
    """
    if n_data_cols <= len(header):
        return header
    # Check if the three flip-mean-only names are present but the full set isn't
    has_mean_only = all(c in header for c in _FLIP_MEAN_ONLY)
    has_full = all(c in header for c in _FLIP_FULL_COLS)
    if not (has_mean_only and not has_full):
        return header
    expanded = []
    for col in header:
        expanded.append(col)
        if col in _FLIP_MEAN_ONLY:
            metric = col[: col.rfind("_mean")]  # e.g. "so2_flip"
            expanded.append(f"{metric}_std")
            expanded.append(f"{metric}_pm")
    return expanded


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        rows_raw = list(reader)

    if not rows_raw:
        return []

    max_cols = max(len(r) for r in rows_raw)
    header = _fix_header(raw_header, max_cols)

    result = []
    for row in rows_raw:
        # Pad short rows so zip doesn't silently drop columns
        padded = row + [""] * (len(header) - len(row))
        result.append(dict(zip(header, padded)))
    return result


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write_csv] {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Baseline index
# ---------------------------------------------------------------------------

def build_baseline_index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], List[float]]:
    """Return {(model, nconstituents): fold_accs} for all none-augmentation rows."""
    index: Dict[Tuple[str, str], List[float]] = {}
    for r in rows:
        if str(r.get("augmentation_scheme", "")).lower() != "none":
            continue
        key = (r.get("model", ""), str(r.get("nconstituents", "")))
        folds = _parse_folds(r.get("test_acc_folds", ""))
        if folds is not None:
            index[key] = folds
    return index


# ---------------------------------------------------------------------------
# p-value computation
# ---------------------------------------------------------------------------

def compute_pvalue(aug_folds: List[float], baseline_folds: List[float]) -> float:
    """Two-sided paired t-test; returns NaN on failure."""
    if len(aug_folds) != len(baseline_folds) or len(aug_folds) < 2:
        return float("nan")
    try:
        result = scipy_stats.ttest_rel(aug_folds, baseline_folds)
        return float(result.pvalue)
    except Exception:
        return float("nan")


def annotate_rows(
    rows: List[Dict[str, str]],
    baseline_index: Dict[Tuple[str, str], List[float]],
) -> List[Dict]:
    """Return new row dicts with p_value and significant_005 columns added."""
    out = []
    for r in rows:
        row = dict(r)
        aug_scheme = str(row.get("augmentation_scheme", "")).lower()
        key = (row.get("model", ""), str(row.get("nconstituents", "")))

        if aug_scheme == "none":
            # Baseline row — no comparison needed
            row["p_value"] = "baseline"
            row["significant_005"] = "—"
        else:
            baseline_folds = baseline_index.get(key)
            aug_folds = _parse_folds(row.get("test_acc_folds", ""))

            if baseline_folds is None:
                row["p_value"] = "no_baseline"
                row["significant_005"] = "—"
            elif aug_folds is None:
                row["p_value"] = "no_folds"
                row["significant_005"] = "—"
            else:
                pval = compute_pvalue(aug_folds, baseline_folds)
                row["p_value"] = f"{pval:.4f}" if not math.isnan(pval) else "nan"
                row["significant_005"] = (
                    "yes" if (not math.isnan(pval) and pval < 0.05) else "no"
                )
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

AUG_ORDER = {"none": 0, "lorentz": 1, "so2": 2, "so3": 3, "permute": 4}
AUG_DISPLAY = {
    "none": "None", "lorentz": "Lorentz",
    "so2": "SO(2)", "so3": "SO(3)", "permute": "Permutation",
}


def _model_display(model_raw: str) -> str:
    """Convert full class path or short name to a human-readable label."""
    if "DeepSets" in model_raw:
        return "DeepSets"
    if "MLP" in model_raw or model_raw == "mlp":
        return "MLP"
    if "Transformer" in model_raw or model_raw == "transformer":
        return "Transformer"
    if "Pelican" in model_raw or model_raw == "pelican":
        return "PELICAN"
    # Fallback: last component of dotted path
    return model_raw.split(".")[-1]


def _safe_fmt(val, digits: int) -> str:
    try:
        f = float(val)
        if math.isnan(f):
            return "N/A"
        return f"{f:.{digits}f}"
    except Exception:
        return "N/A"


def _fmt_inv_cell(r: Dict, mean_key: str, std_key: str, digits: int = 3) -> str:
    """Format an invariance cell as 'mean \\pm std' or just 'mean' if std missing."""
    mean_str = _safe_fmt(r.get(mean_key, "nan"), digits)
    std_raw = r.get(std_key, "")
    if std_raw and std_raw not in {"", "nan"}:
        std_str = _safe_fmt(std_raw, digits)
        if std_str != "N/A":
            return f"{mean_str} \\pm {std_str}"
    return mean_str


def _pval_cell(pval_str: str) -> str:
    if pval_str in {"baseline", "—", "no_baseline", "no_folds"}:
        return "—"
    try:
        p = float(pval_str)
        if math.isnan(p):
            return "N/A"
        if p < 0.001:
            return r"$<$0.001"
        return f"{p:.3f}"
    except ValueError:
        return pval_str


def write_latex_table(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return

    sorted_rows = sorted(rows, key=lambda x: (
        x.get("model", ""),
        int(x.get("nconstituents", 0)),
        AUG_ORDER.get(str(x.get("augmentation_scheme", "")).lower(), 99),
    ))

    from collections import Counter
    model_span = Counter(r["model"] for r in sorted_rows)
    nconst_span = Counter((r["model"], r["nconstituents"]) for r in sorted_rows)

    lines = [
        r"\documentclass[preview,border=1pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs, siunitx, adjustbox, multirow}",
        r"\sisetup{separate-uncertainty=true, bracket-numbers=false}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\scriptsize",
        r"\begin{tabular}{ll l c S[table-format=2.2(2)] S[table-format=1.3(3)] S[table-format=1.3(3)] S[table-format=1.3(3)] r}",
        r"    \toprule",
        (
            r"    \textbf{Model} & \textbf{\# Const} & \textbf{Group} & \textbf{FLOPs} & "
            r"{\textbf{Acc. (\%)}} & {\textbf{Lorentz flip}} & {\textbf{SO(2) flip}} & "
            r"{\textbf{SO(3) flip}} & \textbf{$p$-value} \\"
        ),
        r"    \midrule",
    ]

    prev_model, prev_n = None, None
    for r in sorted_rows:
        model_raw = r.get("model", "")
        model_display = _model_display(model_raw)
        n = r.get("nconstituents", "")
        group = str(r.get("augmentation_scheme", "")).lower()
        aug_display = AUG_DISPLAY.get(group, group)

        new_model = prev_model is not None and model_raw != prev_model
        new_n = prev_n is not None and n != prev_n

        if new_model:
            lines.append(r"    \specialrule{.1em}{.5em}{.5em}")
        elif new_n:
            lines.append(r"    \cmidrule{2-9}")

        n_rows_model = model_span[model_raw]
        n_rows_nconst = nconst_span[(model_raw, n)]

        model_str = (
            f"\\multirow{{{n_rows_model}}}{{*}}{{\\textbf{{{model_display}}}}}"
            if model_raw != prev_model else ""
        )
        n_str = (
            f"\\multirow{{{n_rows_nconst}}}{{*}}{{{n}}}"
            if (n != prev_n or model_raw != prev_model) else ""
        )

        flops_val = r.get("flops_mean", float("nan"))
        try:
            flops_display = (
                f"{int(float(flops_val) / 1000)}K"
                if not math.isnan(float(flops_val)) else "N/A"
            )
        except Exception:
            flops_display = "N/A"
        flops_str = (
            f"\\multirow{{{n_rows_nconst}}}{{*}}{{{flops_display}}}"
            if (n != prev_n or model_raw != prev_model) else ""
        )

        # acc is stored as proportion in the callback CSV, percentage in mlflow CSV
        acc_mean = float(r.get("test_acc_mean", "nan"))
        acc_std = float(r.get("test_acc_std", "nan"))
        # normalise to percentage if stored as proportion (< 2.0)
        if not math.isnan(acc_mean) and acc_mean < 2.0:
            acc_mean *= 100
            acc_std *= 100

        lines.append(
            f"    {model_str} & {n_str} & {aug_display} & {flops_str} & "
            f"{_safe_fmt(acc_mean, 2)} \\pm {_safe_fmt(acc_std, 2)} & "
            f"{_fmt_inv_cell(r, 'lorentz_flip_mean', 'lorentz_flip_std')} & "
            f"{_fmt_inv_cell(r, 'so2_flip_mean', 'so2_flip_std')} & "
            f"{_fmt_inv_cell(r, 'so3_flip_mean', 'so3_flip_std')} & "
            f"{_pval_cell(r.get('p_value', '—'))} \\\\"
        )
        prev_model, prev_n = model_raw, n

    lines += [r"    \bottomrule", r"\end{tabular}", r"\end{document}"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write_latex] {len(sorted_rows)}-row table → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csvs", nargs="+", type=Path, help="Results table CSV(s) to process")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="External baseline CSV containing none-augmentation rows (for cross-table comparisons). "
             "When omitted the baseline rows are taken from within each input CSV.",
    )
    args = parser.parse_args()

    # Build a shared baseline index if an external baseline CSV is provided
    external_baseline: Optional[Dict[Tuple[str, str], List[float]]] = None
    if args.baseline is not None:
        if not args.baseline.exists():
            sys.exit(f"Baseline CSV not found: {args.baseline}")
        external_baseline = build_baseline_index(read_csv(args.baseline))
        print(f"[baseline] Loaded {len(external_baseline)} baseline entries from {args.baseline}")

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue

        rows = read_csv(csv_path)
        if not rows:
            print(f"[skip] {csv_path} is empty")
            continue

        baseline_index = external_baseline if external_baseline is not None else build_baseline_index(rows)

        if not baseline_index:
            print(
                f"[warn] No baseline (augmentation_scheme=none) rows found for {csv_path}. "
                "Pass --baseline <path> to provide an external baseline CSV."
            )

        annotated = annotate_rows(rows, baseline_index)

        out_csv = csv_path.parent / (csv_path.stem + "_pvalues.csv")
        out_tex = csv_path.parent / (csv_path.stem + "_pvalues.tex")

        write_csv(annotated, out_csv)
        write_latex_table(annotated, out_tex)

        # Print a quick summary to stdout
        print(f"\n{'─' * 60}")
        print(f"Table: {csv_path.name}")
        print(f"{'Model':<12} {'nconst':>7} {'Aug':<10} {'Acc (%)':>8} {'p-value':>10} {'sig':>5}")
        print(f"{'─' * 60}")
        for r in annotated:
            acc = float(r.get("test_acc_mean", "nan"))
            if not math.isnan(acc) and acc < 2.0:
                acc *= 100
            print(
                f"{_model_display(r.get('model', '')):<12} "
                f"{r.get('nconstituents', ''):>7} "
                f"{r.get('augmentation_scheme', ''):<10} "
                f"{acc:>8.2f} "
                f"{r.get('p_value', '—'):>10} "
                f"{r.get('significant_005', '—'):>5}"
            )
        print()


if __name__ == "__main__":
    main()
