#!/usr/bin/env python3
"""Add bootstrap p-values and confidence intervals to results table CSVs.

For each (model, nconstituents) group the row with augmentation_scheme=none
is used as the baseline.  A paired bootstrap p-value is computed: resamples
the per-fold differences under H0 (true mean difference = 0) and counts how
often abs(bootstrap mean) is at least abs(observed mean).  Makes no normality assumption;
better calibrated at n=5 folds than a t-test.

Bootstrap 95% CI for the mean accuracy is computed for every row by resampling
the per-fold values with replacement (--n-boot iterations, default 10 000).

Why paired: KFold uses random_state=kfold_seed (default 42) and the test set
is a fixed held-out split, so fold i of condition A and fold i of condition B
are trained/tested on identical data.

Caveat: with n=5 folds statistical power is inherently limited (~0.3–0.5).
Interpret p-values qualitatively alongside effect sizes.

Usage
-----
Results tables live under results/<features>/, split by the literal input
feature list used for training. Baseline (augmentation_scheme=none) rows only
ever exist in the pt_etarel_phirel bucket, so when --baseline is omitted a
sibling results_table_baseline.csv is auto-discovered (same directory first,
then any results/*/results_table_baseline.csv) and used.

    python3 scripts/analysis/add_pvalues.py results/*/results_table_*.csv
    python3 scripts/analysis/add_pvalues.py results/pt_etarel_phirel/results_table_symm_loss.csv \
        --baseline results/pt_etarel_phirel/results_table_augmentation.csv

Outputs are written next to each input CSV with a _pvalues suffix, and a
corresponding _pvalues.tex LaTeX table is also written.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _parse_folds(raw: str) -> list[float] | None:
    """Parse semicolon-separated fold accuracies; return None if it cannot be parsed."""
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


def _fix_header(header: list[str], n_data_cols: int) -> list[str]:
    """Expand a header that lists only flip means to one with mean/std/pm.

    Some CSVs (e.g. symm_loss) were written with headers that declare only so2_flip_mean,
    so3_flip_mean, lorentz_flip_mean but the data rows have mean+std+pm for each metric (6 extra
    columns).  This causes DictReader to map column names to the wrong data positions.
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


def read_csv(path: Path) -> list[dict[str, str]]:
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
        if not any(cell.strip() for cell in row):  # skip blank/empty rows
            continue
        # Pad short rows so zip doesn't silently drop columns
        padded = row + [""] * (len(header) - len(row))
        result.append(dict(zip(header, padded)))
    return result


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rows may come from CSVs with different schemas (e.g. a wasserstein-metric
    # baseline concatenated with a flip-metric table): union the keys instead
    # of trusting rows[0] alone, or DictWriter raises on the extra columns.
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write_csv] {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Baseline index
# ---------------------------------------------------------------------------


def find_default_baseline(csv_path: Path) -> Path | None:
    """Locate a sibling results_table_baseline.csv for auto baseline lookup.

    Checks the input's own directory first, then any results/<features>/ directory next to it.
    Baseline (augmentation_scheme=none) rows only ever live in the 3-feature (pt_etarel_phirel)
    bucket, so augmentation and symm_loss tables in other feature buckets need to cross into it.
    """
    same_dir = csv_path.parent / "results_table_baseline.csv"
    if same_dir.exists() and same_dir.resolve() != csv_path.resolve():
        return same_dir
    results_root = csv_path.parent.parent
    for candidate in sorted(results_root.glob("*/results_table_baseline.csv")):
        return candidate
    return None


def build_baseline_index(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[float]]:
    """Return {(model, nconstituents): fold_accs} for all none-augmentation rows."""
    index: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        if str(r.get("augmentation_scheme", "")).lower() != "none":
            continue
        key = (r.get("model", ""), str(r.get("nconstituents", "")))
        folds = _parse_folds(r.get("test_acc_folds", ""))
        if folds is not None:
            index[key] = folds
    return index


def _detect_inv_metric_suffix(rows: list[dict[str, str]]) -> str | None:
    """Infer the invariance-metric suffix (e.g. 'wasserstein', 'flip') from a 'so2_<m>_folds'
    column present in the CSV header."""
    if not rows:
        return None
    for col in rows[0]:
        if col.startswith("so2_") and col.endswith("_folds"):
            return col[len("so2_") : -len("_folds")]
    return None


def build_metric_baseline_index(
    rows: list[dict[str, str]], metric: str
) -> dict[tuple[str, str, str], list[float]]:
    """Return {(model, nconstituents, group): fold_values} for so2/so3/lorentz per-fold <metric>
    columns of all none-augmentation rows."""
    index: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        if str(r.get("augmentation_scheme", "")).lower() != "none":
            continue
        model_key = (r.get("model", ""), str(r.get("nconstituents", "")))
        for group in ("so2", "so3", "lorentz"):
            folds = _parse_folds(r.get(f"{group}_{metric}_folds", ""))
            if folds is not None:
                index[(*model_key, group)] = folds
    return index


def _acc_after_transform_folds(
    acc_folds: list[float] | None,
    stab_folds: list[float] | None,
    rec_folds: list[float] | None,
) -> list[float] | None:
    """Per-fold accuracy on the *transformed* test set, reconstructed from the
    flip breakdown via total probability over correct/wrong-before:

        acc_after = acc_before * stable_frac + (1 - acc_before) * recover_frac

    acc_before is P(correct before transform) = test_acc_folds; stable_frac
    and recover_frac are the model's own P(correct after | correct before)
    and P(correct after | wrong before) for that group. All three are
    proportions (0-1) here, matching this module's own units for
    test_acc_folds/flip_*_folds.
    """
    if acc_folds is None or stab_folds is None or rec_folds is None:
        return None
    if not (len(acc_folds) == len(stab_folds) == len(rec_folds)):
        return None
    return [a * s + (1.0 - a) * r for a, s, r in zip(acc_folds, stab_folds, rec_folds)]


def build_transformed_acc_baseline_index(
    rows: list[dict[str, str]]
) -> dict[tuple[str, str, str], list[float]]:
    """Return {(model, nconstituents, group): fold_values} of the *baseline* ("none") model's own
    accuracy on the so2/so3/lorentz-transformed test set, reconstructed per group from its
    test_acc_folds + flip breakdown."""
    index: dict[tuple[str, str, str], list[float]] = {}
    for r in rows:
        if str(r.get("augmentation_scheme", "")).lower() != "none":
            continue
        model_key = (r.get("model", ""), str(r.get("nconstituents", "")))
        acc_folds = _parse_folds(r.get("test_acc_folds", ""))
        for group in ("so2", "so3", "lorentz"):
            stab_folds = _parse_folds(r.get(f"{group}_flip_stable_folds", ""))
            rec_folds = _parse_folds(r.get(f"{group}_flip_recover_folds", ""))
            folds = _acc_after_transform_folds(acc_folds, stab_folds, rec_folds)
            if folds is not None:
                index[(*model_key, group)] = folds
    return index


# ---------------------------------------------------------------------------
# p-value computation
# ---------------------------------------------------------------------------


def bootstrap_ci(
    folds: list[float],
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap percentile CI for the mean of fold values.

    Resamples the fold list with replacement n_boot times and returns the (alpha/2, 1-alpha/2)
    quantiles of the bootstrap distribution of the mean.
    """
    if len(folds) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(folds)
    boot_means = sorted(sum(rng.choices(folds, k=n)) / n for _ in range(n_boot))
    alpha = (1.0 - ci) / 2.0
    lo = boot_means[int(alpha * n_boot)]
    hi = boot_means[min(int((1.0 - alpha) * n_boot), n_boot - 1)]
    return (lo, hi)


def bootstrap_pvalue(
    aug_folds: list[float],
    baseline_folds: list[float],
    n_boot: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided paired bootstrap p-value.

    Computes per-fold differences d_i = aug[i] - baseline[i], then centers
    them under H0 (true mean = 0) and counts the fraction of bootstrap means
    of the centered differences whose absolute value meets or exceeds the
    observed abs(mean(d)).
    """
    if len(aug_folds) != len(baseline_folds) or len(aug_folds) < 2:
        return float("nan")
    n = len(aug_folds)
    diffs = [a - b for a, b in zip(aug_folds, baseline_folds)]
    obs_mean = sum(diffs) / n
    if obs_mean == 0.0:
        return 1.0
    # Shift diffs to satisfy H0: E[d] = 0
    centered = [d - obs_mean for d in diffs]
    rng = random.Random(seed)
    count = sum(
        1 for _ in range(n_boot) if abs(sum(rng.choices(centered, k=n)) / n) >= abs(obs_mean)
    )
    return count / n_boot


def compute_delta_for_row(
    row: dict[str, str],
    metric_baseline_index: dict[tuple[str, str, str], list[float]],
    metric: str,
    n_boot: int = 10_000,
) -> dict[str, str]:
    """Compute delta_<metric> = baseline_value - augmented_value for the row's
    own augmentation_scheme group, plus a bootstrap CI/p-value on the delta.

    Sign convention: positive delta means the augmented/symm_loss-trained
    model is MORE invariant (smaller metric value) than the baseline.
    Scope: only rows with augmentation_scheme in {so2, so3, lorentz} get a
    delta; none/permute rows get blank placeholders.
    """
    mean_key = f"delta_{metric}_mean"
    std_key = f"delta_{metric}_std"
    pm_key = f"delta_{metric}_pm"
    ci_lo_key = f"delta_{metric}_ci_lower"
    ci_hi_key = f"delta_{metric}_ci_upper"
    pval_key = f"delta_{metric}_bootstrap_pvalue"
    sig_key = f"delta_{metric}_significant_boot_005"

    scheme = str(row.get("augmentation_scheme", "")).lower()
    if scheme not in {"so2", "so3", "lorentz"}:
        return {
            mean_key: "nan",
            std_key: "nan",
            pm_key: "n/a",
            ci_lo_key: "nan",
            ci_hi_key: "nan",
            pval_key: "—",
            sig_key: "—",
        }

    model_key = (row.get("model", ""), str(row.get("nconstituents", "")))
    baseline_folds = metric_baseline_index.get((*model_key, scheme))
    aug_folds = _parse_folds(row.get(f"{scheme}_{metric}_folds", ""))

    if baseline_folds is None or aug_folds is None or len(aug_folds) != len(baseline_folds):
        return {
            mean_key: "nan",
            std_key: "nan",
            pm_key: "nan",
            ci_lo_key: "nan",
            ci_hi_key: "nan",
            pval_key: "no_baseline" if baseline_folds is None else "no_folds",
            sig_key: "—",
        }

    deltas = [b - a for b, a in zip(baseline_folds, aug_folds)]
    n = len(deltas)
    d_mean = sum(deltas) / n
    d_std = math.sqrt(sum((d - d_mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(deltas, n_boot=n_boot)
    # H0: true mean delta = 0 -> bootstrap_pvalue's internal diffs = deltas - 0 = deltas
    bpval = bootstrap_pvalue(deltas, [0.0] * n, n_boot=n_boot)

    return {
        mean_key: f"{d_mean:.6f}",
        std_key: f"{d_std:.6f}",
        pm_key: f"{d_mean:.4f} +/- {d_std:.4f}",
        ci_lo_key: f"{ci_lo:.6f}" if not math.isnan(ci_lo) else "nan",
        ci_hi_key: f"{ci_hi:.6f}" if not math.isnan(ci_hi) else "nan",
        pval_key: f"{bpval:.4f}" if not math.isnan(bpval) else "nan",
        sig_key: "yes" if (not math.isnan(bpval) and bpval < 0.05) else "no",
    }


def compute_delta_acc_transformed_for_row(
    row: dict[str, str],
    transformed_acc_baseline_index: dict[tuple[str, str, str], list[float]],
    n_boot: int = 10_000,
) -> dict[str, str]:
    """Compute delta_accuracy_transformed = augmented_acc_on_T(x) minus
    baseline_acc_on_T(x), for the row's own augmentation_scheme group, both
    reconstructed via _acc_after_transform_folds so this compares the trained
    and baseline models under the *same* transform -- apples to apples with
    delta_<inv_metric> and the flip-breakdown deltas, unlike plain delta
    accuracy (which compares clean-data accuracy and usually reads as the
    *cost* of invariance training rather than its benefit).

    Sign convention: positive delta means the trained model is MORE accurate
    on transformed inputs than the baseline (accuracy is "higher is better",
    the opposite convention from delta_<inv_metric>'s "baseline - augmented").
    Scope: only rows with augmentation_scheme in {so2, so3, lorentz} get a
    delta; none/permute rows get blank placeholders.
    """
    mean_key = "delta_accuracy_transformed_mean"
    std_key = "delta_accuracy_transformed_std"
    pm_key = "delta_accuracy_transformed_pm"
    ci_lo_key = "delta_accuracy_transformed_ci_lower"
    ci_hi_key = "delta_accuracy_transformed_ci_upper"
    pval_key = "delta_accuracy_transformed_bootstrap_pvalue"
    sig_key = "delta_accuracy_transformed_significant_boot_005"

    scheme = str(row.get("augmentation_scheme", "")).lower()
    if scheme not in {"so2", "so3", "lorentz"}:
        return {
            mean_key: "nan",
            std_key: "nan",
            pm_key: "n/a",
            ci_lo_key: "nan",
            ci_hi_key: "nan",
            pval_key: "—",
            sig_key: "—",
        }

    model_key = (row.get("model", ""), str(row.get("nconstituents", "")))
    baseline_folds = transformed_acc_baseline_index.get((*model_key, scheme))
    acc_folds = _parse_folds(row.get("test_acc_folds", ""))
    stab_folds = _parse_folds(row.get(f"{scheme}_flip_stable_folds", ""))
    rec_folds = _parse_folds(row.get(f"{scheme}_flip_recover_folds", ""))
    aug_folds = _acc_after_transform_folds(acc_folds, stab_folds, rec_folds)

    if baseline_folds is None or aug_folds is None or len(aug_folds) != len(baseline_folds):
        return {
            mean_key: "nan",
            std_key: "nan",
            pm_key: "nan",
            ci_lo_key: "nan",
            ci_hi_key: "nan",
            pval_key: "no_baseline" if baseline_folds is None else "no_folds",
            sig_key: "—",
        }

    deltas = [a - b for a, b in zip(aug_folds, baseline_folds)]
    n = len(deltas)
    d_mean = sum(deltas) / n
    d_std = math.sqrt(sum((d - d_mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else 0.0
    ci_lo, ci_hi = bootstrap_ci(deltas, n_boot=n_boot)
    bpval = bootstrap_pvalue(deltas, [0.0] * n, n_boot=n_boot)

    return {
        mean_key: f"{d_mean:.6f}",
        std_key: f"{d_std:.6f}",
        pm_key: f"{d_mean:.4f} +/- {d_std:.4f}",
        ci_lo_key: f"{ci_lo:.6f}" if not math.isnan(ci_lo) else "nan",
        ci_hi_key: f"{ci_hi:.6f}" if not math.isnan(ci_hi) else "nan",
        pval_key: f"{bpval:.4f}" if not math.isnan(bpval) else "nan",
        sig_key: "yes" if (not math.isnan(bpval) and bpval < 0.05) else "no",
    }


def annotate_rows(
    rows: list[dict[str, str]],
    baseline_index: dict[tuple[str, str], list[float]],
    n_boot: int = 10_000,
    metric_baseline_index: dict[tuple[str, str, str], list[float]] | None = None,
    inv_metric: str | None = None,
    transformed_acc_baseline_index: dict[tuple[str, str, str], list[float]] | None = None,
) -> list[dict]:
    """Return new row dicts with p-value, bootstrap p-value, and CI columns added."""
    out = []
    for r in rows:
        row = dict(r)
        aug_scheme = str(row.get("augmentation_scheme", "")).lower()
        key = (row.get("model", ""), str(row.get("nconstituents", "")))
        aug_folds = _parse_folds(row.get("test_acc_folds", ""))

        # Bootstrap CI for this row's mean (applies to every row with valid folds)
        if aug_folds is not None:
            ci_lo, ci_hi = bootstrap_ci(aug_folds, n_boot=n_boot)
            row["ci_lower"] = f"{ci_lo:.6f}" if not math.isnan(ci_lo) else "nan"
            row["ci_upper"] = f"{ci_hi:.6f}" if not math.isnan(ci_hi) else "nan"
        else:
            row["ci_lower"] = "nan"
            row["ci_upper"] = "nan"

        if aug_scheme == "none":
            row["bootstrap_pvalue"] = "baseline"
            row["significant_boot_005"] = "—"
        else:
            baseline_folds = baseline_index.get(key)

            if baseline_folds is None:
                row["bootstrap_pvalue"] = "no_baseline"
                row["significant_boot_005"] = "—"
            elif aug_folds is None:
                row["bootstrap_pvalue"] = "no_folds"
                row["significant_boot_005"] = "—"
            else:
                bpval = bootstrap_pvalue(aug_folds, baseline_folds, n_boot=n_boot)
                row["bootstrap_pvalue"] = f"{bpval:.4f}" if not math.isnan(bpval) else "nan"
                row["significant_boot_005"] = (
                    "yes" if (not math.isnan(bpval) and bpval < 0.05) else "no"
                )

        if inv_metric is not None and metric_baseline_index is not None:
            row.update(
                compute_delta_for_row(row, metric_baseline_index, inv_metric, n_boot=n_boot)
            )

        if transformed_acc_baseline_index is not None:
            row.update(
                compute_delta_acc_transformed_for_row(
                    row, transformed_acc_baseline_index, n_boot=n_boot
                )
            )

        out.append(row)
    return out


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

AUG_ORDER = {"none": 0, "lorentz": 1, "so2": 2, "so3": 3, "permute": 4}
AUG_DISPLAY = {
    "none": "None",
    "lorentz": "Lorentz",
    "so2": "SO(2)",
    "so3": "SO(3)",
    "permute": "Permutation",
}


# The model column holds algorithm.name for tables written since the refactor
# and a full class path for the ones written before it, so both are mapped.
_MODEL_DISPLAY = {
    "deepsets": "DeepSets",
    "mlp": "MLP",
    "jetformer": "JetFormer",
    "lorentznet": "LorentzNet",
    "pelican": "PELICAN",
}


def _model_display(model_raw: str) -> str:
    """Convert a short name or a full class path to a human-readable label."""
    if model_raw.lower() in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[model_raw.lower()]
    if "DeepSets" in model_raw:
        return "DeepSets"
    if "MLP" in model_raw:
        return "MLP"
    if "JetFormer" in model_raw or "Transformer" in model_raw:
        return "JetFormer"
    if "LorentzNet" in model_raw:
        return "LorentzNet"
    if "Pelican" in model_raw:
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


def _fmt_inv_cell(r: dict, mean_key: str, std_key: str, digits: int = 3) -> str:
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


_METRIC_DISPLAY = {
    "wasserstein": "Wasserstein",
    "flip": "Flip rate",
    "l2": "L2",
    "cosine": "Cosine",
    "js": "JS",
    "orbit_var": "Orbit var",
}


def _metric_display(metric: str | None) -> str:
    if not metric:
        return "Metric"
    return _METRIC_DISPLAY.get(metric, metric.capitalize())


def _fmt_delta_cell(r: dict, metric: str, digits: int = 3) -> str:
    """Format delta_<metric> as 'mean \\pm std' with a '*' marker if the bootstrap p-value is
    significant at 0.05.

    '—' when out of scope (None/ permute rows) or no comparable baseline/fold data was found.
    """
    mean_raw = r.get(f"delta_{metric}_mean", "")
    if mean_raw in ("", "nan", None):
        return "—"
    try:
        mean_v = float(mean_raw)
        std_v = float(r.get(f"delta_{metric}_std", "nan") or "nan")
    except (TypeError, ValueError):
        return "N/A"
    sig = r.get(f"delta_{metric}_significant_boot_005", "")
    marker = r"^{*}" if sig == "yes" else ""
    return f"${mean_v:.{digits}f} \\pm {std_v:.{digits}f}{marker}$"


def _ci_cell(r: dict, digits: int = 1) -> str:
    """Format bootstrap CI as '[lo, hi]' in percentage units."""
    lo_raw = r.get("ci_lower", "nan")
    hi_raw = r.get("ci_upper", "nan")
    try:
        lo = float(lo_raw)
        hi = float(hi_raw)
        if math.isnan(lo) or math.isnan(hi):
            return "N/A"
        # Convert proportion → percentage if stored as proportion
        if lo < 2.0:
            lo *= 100
            hi *= 100
        return f"[{lo:.{digits}f}, {hi:.{digits}f}]"
    except (ValueError, TypeError):
        return "N/A"


def write_latex_table(rows: list[dict], output_path: Path, inv_metric: str | None = None) -> None:
    if not rows:
        return

    metric_label = _metric_display(inv_metric)
    metric_key = inv_metric or "flip"

    def _sort_key(x):
        try:
            nconst = int(x.get("nconstituents", 0) or 0)
        except (ValueError, TypeError):
            nconst = 0
        return (
            x.get("model", ""),
            nconst,
            AUG_ORDER.get(str(x.get("augmentation_scheme", "")).lower(), 99),
        )

    sorted_rows = sorted(rows, key=_sort_key)

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
            lines.append(r"    \cmidrule{2-10}")

        n_rows_model = model_span[model_raw]
        n_rows_nconst = nconst_span[(model_raw, n)]

        model_str = (
            f"\\multirow{{{n_rows_model}}}{{*}}{{\\textbf{{{model_display}}}}}"
            if model_raw != prev_model
            else ""
        )
        n_str = (
            f"\\multirow{{{n_rows_nconst}}}{{*}}{{{n}}}"
            if (n != prev_n or model_raw != prev_model)
            else ""
        )

        flops_val = r.get("flops_mean", float("nan"))
        try:
            flops_display = (
                f"{int(float(flops_val) / 1000)}K" if not math.isnan(float(flops_val)) else "N/A"
            )
        except Exception:
            flops_display = "N/A"
        flops_str = (
            f"\\multirow{{{n_rows_nconst}}}{{*}}{{{flops_display}}}"
            if (n != prev_n or model_raw != prev_model)
            else ""
        )

        # acc is stored as proportion in the callback CSV, percentage in mlflow CSV
        acc_mean = float(r.get("test_acc_mean", "nan") or "nan")
        acc_std = float(r.get("test_acc_std", "nan") or "nan")
        # normalise to percentage if stored as proportion (< 2.0)
        if not math.isnan(acc_mean) and acc_mean < 2.0:
            acc_mean *= 100
            acc_std *= 100

        lines.append(
            f"    {model_str} & {n_str} & {aug_display} & {flops_str} & "
            f"{_safe_fmt(acc_mean, 2)} \\pm {_safe_fmt(acc_std, 2)} & "
            f"{_ci_cell(r)} & "
            f"{_fmt_delta_cell(r, metric_key)} & "
            f"{_fmt_inv_cell(r, 'flip_recover_mean', 'flip_recover_std')} & "
            f"{_fmt_inv_cell(r, 'flip_stable_mean', 'flip_stable_std')} & "
            f"{_fmt_inv_cell(r, 'flip_degrade_mean', 'flip_degrade_std')} & "
            f"{_pval_cell(r.get('bootstrap_pvalue', '—'))} \\\\"
        )
        prev_model, prev_n = model_raw, n

    lines += [
        r"    \bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"",
        r"\vspace{2pt}",
        r"{\scriptsize $^{*}$ significant at bootstrap $p<0.05$ (baseline vs.\ this row's invariance metric).}",
        r"\end{document}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write_latex] {len(sorted_rows)}-row table → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csvs", nargs="+", type=Path, help="Results table CSV(s) to process")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="External baseline CSV containing none-augmentation rows (for cross-table comparisons). "
        "When omitted the baseline rows are taken from within each input CSV.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples for CI and p-value estimation (default: 10000).",
    )
    parser.add_argument(
        "--inv-metric",
        type=str,
        default=None,
        help="Invariance metric suffix for delta computation (e.g. 'wasserstein', 'flip'). "
        "Auto-detected from '<group>_<metric>_folds' columns when omitted.",
    )
    args = parser.parse_args()

    # Build a shared baseline index if an external baseline CSV is provided
    external_baseline: dict[tuple[str, str], list[float]] | None = None
    external_metric_baseline: dict[tuple[str, str, str], list[float]] | None = None
    external_transformed_acc_baseline: dict[tuple[str, str, str], list[float]] | None = None
    external_inv_metric: str | None = None
    if args.baseline is not None:
        if not args.baseline.exists():
            sys.exit(f"Baseline CSV not found: {args.baseline}")
        baseline_rows_raw = read_csv(args.baseline)
        external_baseline = build_baseline_index(baseline_rows_raw)
        external_inv_metric = args.inv_metric or _detect_inv_metric_suffix(baseline_rows_raw)
        if external_inv_metric is not None:
            external_metric_baseline = build_metric_baseline_index(
                baseline_rows_raw, external_inv_metric
            )
        external_transformed_acc_baseline = build_transformed_acc_baseline_index(baseline_rows_raw)
        print(f"[baseline] Loaded {len(external_baseline)} baseline entries from {args.baseline}")

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found")
            continue

        rows = read_csv(csv_path)
        if not rows:
            print(f"[skip] {csv_path} is empty")
            continue

        # baseline_path tracks whichever CSV supplied the none-rows used below,
        # so they can be re-read and included in the output as reference rows.
        baseline_path = args.baseline
        inv_metric, metric_baseline_index = external_inv_metric, external_metric_baseline
        transformed_acc_baseline_index = external_transformed_acc_baseline

        if external_baseline is not None:
            baseline_index = external_baseline
        else:
            baseline_index = build_baseline_index(rows)
            if baseline_index:
                inv_metric = args.inv_metric or _detect_inv_metric_suffix(rows)
                metric_baseline_index = (
                    build_metric_baseline_index(rows, inv_metric) if inv_metric else None
                )
                transformed_acc_baseline_index = build_transformed_acc_baseline_index(rows)
            else:
                # This table has no augmentation_scheme=none rows of its own
                # (e.g. a lorentz-only results/px_py_pz_etarel_phirel/ table),
                # so auto-discover a sibling results_table_baseline.csv.
                auto_baseline = find_default_baseline(csv_path)
                if auto_baseline is not None:
                    baseline_path = auto_baseline
                    baseline_rows_raw = read_csv(auto_baseline)
                    baseline_index = build_baseline_index(baseline_rows_raw)
                    inv_metric = args.inv_metric or _detect_inv_metric_suffix(baseline_rows_raw)
                    metric_baseline_index = (
                        build_metric_baseline_index(baseline_rows_raw, inv_metric)
                        if inv_metric
                        else None
                    )
                    transformed_acc_baseline_index = build_transformed_acc_baseline_index(
                        baseline_rows_raw
                    )
                    print(
                        f"[baseline] Auto-detected baseline for {csv_path}: "
                        f"{auto_baseline} ({len(baseline_index)} entries)"
                    )

        if not baseline_index:
            print(
                f"[warn] No baseline (augmentation_scheme=none) rows found for {csv_path}. "
                "Pass --baseline <path> to provide an external baseline CSV."
            )

        annotated = annotate_rows(
            rows,
            baseline_index,
            n_boot=args.n_boot,
            metric_baseline_index=metric_baseline_index,
            inv_metric=inv_metric,
            transformed_acc_baseline_index=transformed_acc_baseline_index,
        )

        # When the baseline came from another file (explicit --baseline or
        # auto-discovered), include its none rows in the output so the LaTeX
        # table has a reference row for each (model, nconstituents).
        if baseline_path is not None:
            baseline_rows = [
                r
                for r in read_csv(baseline_path)
                if str(r.get("augmentation_scheme", "")).lower() == "none"
            ]
            baseline_annotated = annotate_rows(
                baseline_rows,
                baseline_index,
                n_boot=args.n_boot,
                metric_baseline_index=metric_baseline_index,
                inv_metric=inv_metric,
                transformed_acc_baseline_index=transformed_acc_baseline_index,
            )
            annotated = baseline_annotated + annotated

        out_csv = csv_path.parent / (csv_path.stem + "_pvalues.csv")
        out_tex = csv_path.parent / (csv_path.stem + "_pvalues.tex")

        write_csv(annotated, out_csv)
        write_latex_table(annotated, out_tex, inv_metric=inv_metric)

        # Print a quick summary to stdout
        delta_mean_key = f"delta_{inv_metric}_mean" if inv_metric else None
        delta_pval_key = f"delta_{inv_metric}_bootstrap_pvalue" if inv_metric else None
        delta_header = f" {'delta_' + inv_metric:>14} {'delta p':>9}" if inv_metric else ""
        print(f"\n{'─' * 76}")
        print(f"Table: {csv_path.name}")
        print(
            f"{'Model':<12} {'nconst':>7} {'Aug':<10} {'Acc (%)':>8} {'95% CI':>16} {'p (boot)':>10} {'sig*':>5}{delta_header}"
        )
        print(f"{'─' * 76}")
        for r in annotated:
            acc = float(r.get("test_acc_mean", "nan") or "nan")
            if not math.isnan(acc) and acc < 2.0:
                acc *= 100
            delta_str = (
                f" {r.get(delta_mean_key, ''):>14} {r.get(delta_pval_key, ''):>9}"
                if inv_metric
                else ""
            )
            print(
                f"{_model_display(r.get('model', '')):<12} "
                f"{r.get('nconstituents', ''):>7} "
                f"{r.get('augmentation_scheme', ''):<10} "
                f"{acc:>8.2f} "
                f"{_ci_cell(r):>16} "
                f"{r.get('bootstrap_pvalue', '—'):>10} "
                f"{r.get('significant_boot_005', '—'):>5}"
                f"{delta_str}"
            )
        print(f"{'─' * 76}")
        print("  * significant_boot_005: bootstrap p < 0.05")
        print()


if __name__ == "__main__":
    main()
