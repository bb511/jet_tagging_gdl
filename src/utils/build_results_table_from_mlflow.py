#!/usr/bin/env python3
"""Build a compact summary table from MLflow fold runs.

The script expects MLflow runs logged by this repo where each fold run has:
- `params/fold_idx` containing the fold index (0-based)
- `params/model._target_`
- `params/data.augmentation_scheme`
- `params/data.nconstituents`
- `metrics/test.acc`
- `metrics/invariance.so2_flip` etc.
- optional `metrics/model.flops`

MLflow flattens nested keys with dots on disk:
    logged key  "model/_target_"   → file  params/model._target_
    logged key  "test/acc"         → file  metrics/test.acc
    logged key  "invariance/so2_flip" → file metrics/invariance.so2_flip

For each `(model, nconstituents, augmentation_scheme, fold_idx)` it keeps the
latest run (by meta.yaml end_time) and then aggregates across folds.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


INVARIANCE_KEYS = (
    "invariance/so2_flip",
    "invariance/so3_flip",
    "invariance/lorentz_flip",
)


@dataclass
class FoldRecord:
    model_family: str
    model_target: str
    nconstituents: int
    augmentation_scheme: str
    method: str
    fold_idx: int
    run_id: str
    end_time: int
    test_acc: float
    so2_flip: float
    so3_flip: float
    lorentz_flip: float
    flops: float
    # None means legacy (single combined run); "train" or "eval" in separated mode
    run_stage: Optional[str] = None

    @property
    def selected_flip_value(self) -> float:
        if self.augmentation_scheme == "so2":
            return self.so2_flip
        if self.augmentation_scheme == "so3":
            return self.so3_flip
        if self.augmentation_scheme == "lorentz":
            return self.lorentz_flip
        return safe_mean([self.so2_flip, self.so3_flip, self.lorentz_flip])


def safe_float(raw: Optional[str]) -> float:
    if raw is None:
        return float("nan")
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return float("nan")


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def read_metric_last(metric_path: Path) -> float:
    """Read the final logged value from an MLflow metric file.

    MLflow metric files have lines:  <timestamp> <value> <step>
    """
    if not metric_path.exists():
        return float("nan")
    last_line = None
    with metric_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return float("nan")
    parts = last_line.split()
    if len(parts) < 2:
        return float("nan")
    return safe_float(parts[1])


def read_param(run_dir: Path, logged_key: str) -> Optional[str]:
    slash_path = run_dir / "params" / logged_key   # ← slash first
    if slash_path.exists():
        return read_text(slash_path)
    dot_path = run_dir / "params" / logged_key.replace("/", ".")
    if dot_path.exists():
        return read_text(dot_path)
    return None


def read_metric(run_dir: Path, logged_key: str) -> float:
    """Read an MLflow metric, handling both dot-flattened and slash-nested paths.

    MLflow flattens "test/acc" → metrics/test.acc
    """
    # Try dot-flattened path first
    dot_path = run_dir / "metrics" / logged_key.replace("/", ".")
    if dot_path.exists():
        return read_metric_last(dot_path)
    # Fallback: slash-nested path
    slash_path = run_dir / "metrics" / logged_key
    if slash_path.exists():
        return read_metric_last(slash_path)
    return float("nan")


def read_tag(run_dir: Path, tag_name: str) -> Optional[str]:
    """Read an MLflow tag from run_dir/tags/<tag_name>."""
    return read_text(run_dir / "tags" / tag_name)


def parse_end_time(meta_path: Path) -> int:
    raw = read_text(meta_path)
    if raw is None:
        return -1
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("end_time:"):
            val = line.split(":", 1)[1].strip()
            try:
                return int(val)
            except ValueError:
                return -1
    return -1


def parse_fold_idx_from_str(raw: Optional[str]) -> Optional[int]:
    """Extract fold index from a string like 'fold 3/5', 'fold_3', '3', or 'main'."""
    if raw is None:
        return None
    raw = raw.strip()
    # "main" is the label used when n_folds=1 (single run, no K-fold)
    if raw.lower() == "main":
        return 0
    # Try plain integer first
    try:
        return int(raw)
    except ValueError:
        pass
    # Try "fold N/M" or "fold_N"
    match = re.search(r"fold[_\s]+(\d+)", raw, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def model_family_from_target(target: str) -> str:
    if "DeepSets" in target:
        return "deepsets"
    if "MLPClassifier" in target:
        return "mlp"
    if ".Transformer" in target or target.endswith("Transformer"):
        return "transformer"
    return target.split(".")[-1].lower()


def safe_mean(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def safe_std(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    n = len(valid)
    if n <= 1:
        return 0.0
    mean = sum(valid) / n
    var = sum((v - mean) ** 2 for v in valid) / (n - 1)
    return float(math.sqrt(var))


def fmt_pm(mean_val: float, std_val: float) -> str:
    if math.isnan(mean_val):
        return "nan"
    return f"{mean_val:.4f} +/- {std_val:.4f}"


def collect_fold_records(mlflow_root: Path) -> List[FoldRecord]:
    """Walk all MLflow run directories and collect FoldRecord objects.

    A run is included if it has a readable fold_idx (logged as param or tag),
    a model target, augmentation scheme, nconstituents, and a non-NaN test/acc.
    """
    records: List[FoldRecord] = []
    skipped_no_fold = 0
    skipped_no_acc = 0
    skipped_no_meta = 0

    # MLflow layout: mlruns/<experiment_id>/<run_id>/
    for meta_path in sorted(mlflow_root.glob("*/*/meta.yaml")):
        run_dir = meta_path.parent
        run_id  = run_dir.name

        # Skip the experiment-level meta.yaml (it sits one level up)
        if not (run_dir / "params").exists() and not (run_dir / "metrics").exists():
            skipped_no_meta += 1
            continue

        # --- Fold index ---
        # Try param first (logged via log_hyperparameters with fold_idx in object_dict)
        fold_raw = read_param(run_dir, "fold_idx")
        if fold_raw is None:
            # Fallback: check tags (mlflow.runName often contains fold info)
            fold_raw = read_tag(run_dir, "mlflow.runName")
        fold_idx = parse_fold_idx_from_str(fold_raw)
        if fold_idx is None:
            skipped_no_fold += 1
            continue

        # --- Model / data params ---
        model_target = read_param(run_dir, "model/_target_")
        aug          = read_param(run_dir, "data/augmentation_scheme")
        nconst_raw   = read_param(run_dir, "data/nconstituents")
        method_raw   = read_param(run_dir, "data/method")

        if model_target is None or aug is None or nconst_raw is None:
            continue

        nconst = int(safe_float(nconst_raw))
        method = method_raw.strip().lower() if method_raw else "augmentation"

        # --- Run stage (None for legacy runs without this param) ---
        run_stage = read_param(run_dir, "run_stage")

        # --- Metrics ---
        test_acc = read_metric(run_dir, "test/acc")
        # Train-stage runs in separated mode have no test/acc — that is expected.
        # Only skip if there is no test/acc AND this is NOT a pure train-stage run.
        if math.isnan(test_acc) and run_stage != "train":
            skipped_no_acc += 1
            continue

        so2_flip     = read_metric(run_dir, "test/invariance/so2_flip")
        so3_flip     = read_metric(run_dir, "test/invariance/so3_flip")
        lorentz_flip = read_metric(run_dir, "test/invariance/lorentz_flip")

        # Fallback: validation-time invariance metrics (without test/ prefix)
        if math.isnan(so2_flip):
            so2_flip = read_metric(run_dir, "invariance/so2_flip")
        if math.isnan(so3_flip):
            so3_flip = read_metric(run_dir, "invariance/so3_flip")
        if math.isnan(lorentz_flip):
            lorentz_flip = read_metric(run_dir, "invariance/lorentz_flip")

        # FLOPs — try metric first, then all param variants
        flops = read_metric(run_dir, "model/flops")
        if math.isnan(flops):
            flops = safe_float(read_param(run_dir, "model/flops"))
        if math.isnan(flops):
            flops = safe_float(read_param(run_dir, "model/flops_total"))
        if math.isnan(flops):
            flops = safe_float(read_param(run_dir, "model/flops_mult_adds"))

        records.append(
            FoldRecord(
                model_family=model_family_from_target(model_target),
                model_target=model_target,
                nconstituents=nconst,
                augmentation_scheme=aug.strip().lower(),
                method=method,
                fold_idx=fold_idx,
                run_id=run_id,
                end_time=parse_end_time(meta_path),
                test_acc=test_acc,
                so2_flip=so2_flip,
                so3_flip=so3_flip,
                lorentz_flip=lorentz_flip,
                flops=flops,
                run_stage=run_stage,
            )
        )

    print(
        f"[collect] Found {len(records)} valid fold records. "
        f"Skipped: {skipped_no_fold} (no fold_idx), "
        f"{skipped_no_acc} (no test/acc), "
        f"{skipped_no_meta} (no params/metrics dir)."
    )
    return records


def keep_latest_per_fold(records: List[FoldRecord]) -> List[FoldRecord]:
    latest: Dict[Tuple[str, str, int, str, str, int], FoldRecord] = {}
    for rec in records:
        key = (
            rec.model_family,
            rec.model_target,
            rec.nconstituents,
            rec.augmentation_scheme,
            rec.method,
            rec.fold_idx,
        )
        prev = latest.get(key)
        if prev is None or rec.end_time > prev.end_time:
            latest[key] = rec
    return list(latest.values())


def join_train_eval_records(records: List[FoldRecord]) -> List[FoldRecord]:
    """Merge separated-mode train and eval records into unified FoldRecords.

    In separated mode training emits run_stage=train (invariance + FLOPs, no
    test_acc) and eval emits run_stage=eval (test_acc, no invariance).  This
    function joins them by (model_family, model_target, nconstituents,
    augmentation_scheme, method, fold_idx) so downstream aggregation sees one
    complete record per fold.  Legacy records (run_stage=None) pass through
    unchanged.
    """
    train_records = [r for r in records if r.run_stage == "train"]
    eval_records  = [r for r in records if r.run_stage == "eval"]
    legacy        = [r for r in records if r.run_stage is None]

    if not train_records or not eval_records:
        # Nothing to join — return as-is (handles legacy-only or partial runs)
        return records

    FoldKey = Tuple[str, str, int, str, str, int]

    def _key(r: FoldRecord) -> FoldKey:
        return (r.model_family, r.model_target, r.nconstituents,
                r.augmentation_scheme, r.method, r.fold_idx)

    train_by_key: Dict[FoldKey, FoldRecord] = {_key(r): r for r in train_records}

    merged: List[FoldRecord] = []
    for eval_rec in eval_records:
        k = _key(eval_rec)
        train_rec = train_by_key.get(k)
        if train_rec:
            merged.append(FoldRecord(
                model_family=eval_rec.model_family,
                model_target=eval_rec.model_target,
                nconstituents=eval_rec.nconstituents,
                augmentation_scheme=eval_rec.augmentation_scheme,
                method=eval_rec.method,
                fold_idx=eval_rec.fold_idx,
                run_id=eval_rec.run_id,
                end_time=eval_rec.end_time,
                test_acc=eval_rec.test_acc,
                so2_flip=train_rec.so2_flip,
                so3_flip=train_rec.so3_flip,
                lorentz_flip=train_rec.lorentz_flip,
                flops=train_rec.flops,
                run_stage=None,  # merged; no longer stage-specific
            ))
        else:
            print(f"[join] No matching train record for eval fold {k} — keeping eval-only.")
            merged.append(eval_rec)

    unmatched_train = [r for r in train_records if _key(r) not in {_key(e) for e in eval_records}]
    if unmatched_train:
        print(f"[join] {len(unmatched_train)} train record(s) had no matching eval fold — dropped.")

    return legacy + merged


def aggregate_rows(records: List[FoldRecord], min_folds: int) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int, str, str], List[FoldRecord]] = {}
    for rec in records:
        key = (rec.model_family, rec.model_target, rec.nconstituents, rec.augmentation_scheme, rec.method)
        grouped.setdefault(key, []).append(rec)

    rows: List[Dict] = []
    for key, folds in grouped.items():
        if len(folds) < min_folds:
            print(
                f"[aggregate] Skipping {key} — only {len(folds)} folds "
                f"(need {min_folds})"
            )
            continue

        test_vals    = [r.test_acc    for r in folds]
        so2_vals     = [r.so2_flip    for r in folds]
        so3_vals     = [r.so3_flip    for r in folds]
        lor_vals     = [r.lorentz_flip for r in folds]
        flops_vals   = [r.flops       for r in folds]

        test_mean  = safe_mean(test_vals)
        test_std   = safe_std(test_vals)
        flops_mean = safe_mean(flops_vals)

        rows.append({
            "model":               key[0],
            "model_target":        key[1],
            "nconstituents":       key[2],
            "augmentation_scheme": key[3],
            "method":              key[4],
            "folds_used":          len(folds),
            "test_acc_mean":       round(test_mean * 100, 4),
            "test_acc_std":        round(test_std  * 100, 4),
            "test_acc_pm":         fmt_pm(test_mean * 100, test_std * 100),
            "so2_flip_mean":       round(safe_mean(so2_vals),  6),
            "so2_flip_std":        round(safe_std(so2_vals),   6),
            "so2_flip_pm":         fmt_pm(safe_mean(so2_vals), safe_std(so2_vals)),
            "so3_flip_mean":       round(safe_mean(so3_vals),  6),
            "so3_flip_std":        round(safe_std(so3_vals),   6),
            "so3_flip_pm":         fmt_pm(safe_mean(so3_vals), safe_std(so3_vals)),
            "lorentz_flip_mean":   round(safe_mean(lor_vals),  6),
            "lorentz_flip_std":    round(safe_std(lor_vals),   6),
            "lorentz_flip_pm":     fmt_pm(safe_mean(lor_vals), safe_std(lor_vals)),
            "flops_mean":          round(flops_mean, 0) if not math.isnan(flops_mean) else float("nan"),
            "fold_ids":            ",".join(str(r.fold_idx) for r in folds),
            "run_ids":             ",".join(r.run_id for r in folds),
        })
    return rows


def debug_one_run(mlflow_root: Path) -> None:
    """Print the layout of the first run found — helps diagnose path issues."""
    for meta_path in sorted(mlflow_root.glob("*/*/meta.yaml"))[:1]:
        run_dir = meta_path.parent
        print(f"\n=== DEBUG: run_dir = {run_dir} ===")
        for section in ("params", "metrics", "tags"):
            section_dir = run_dir / section
            if section_dir.exists():
                files = sorted(section_dir.rglob("*"))
                print(f"\n  {section}/")
                for f in files[:30]:
                    val = f.read_text(encoding="utf-8").strip()[:60] if f.is_file() else "<dir>"
                    print(f"    {f.relative_to(section_dir)}  →  {val}")
            else:
                print(f"\n  {section}/  ← MISSING")


def write_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        print("[write_csv] No rows to write — table is empty.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write_csv] Wrote {len(rows)} rows to {output_path}")


def _latex_table_body(rows: List[Dict]) -> List[str]:
    """Return the lines between (exclusive) \\toprule and \\bottomrule for one method."""
    AUG_ORDER   = {"none": 0, "lorentz": 1, "so2": 2, "so3": 3, "permute": 4}
    AUG_DISPLAY = {"none": "None", "lorentz": "Lorentz", "so2": "SO(2)", "so3": "SO(3)", "permute": "Permutation"}

    sorted_rows = sorted(rows, key=lambda x: (
        x["model"],
        x["nconstituents"],
        AUG_ORDER.get(str(x["augmentation_scheme"]).lower(), 99),
    ))

    def safe_fmt(val, digits):
        try:
            if math.isnan(float(val)):
                return "N/A"
            return f"{float(val):.{digits}f}"
        except Exception:
            return "N/A"

    model_span  = Counter(r["model"] for r in sorted_rows)
    nconst_span = Counter((r["model"], r["nconstituents"]) for r in sorted_rows)

    lines = []
    prev_model, prev_n = None, None
    for r in sorted_rows:
        model_raw     = r["model"]
        model_display = "MLP" if model_raw == "mlp" else model_raw.capitalize()
        n             = r["nconstituents"]
        group         = str(r["augmentation_scheme"]).lower()
        aug_display   = AUG_DISPLAY.get(group, group)

        new_model = prev_model is not None and model_raw != prev_model
        new_n     = prev_n is not None and n != prev_n

        if new_model:
            lines.append(r"    \specialrule{.1em}{.5em}{.5em}")
        elif new_n:
            lines.append(r"    \cmidrule{2-8}")

        n_rows_model  = model_span[model_raw]
        n_rows_nconst = nconst_span[(model_raw, n)]

        model_str = f"\\multirow{{{n_rows_model}}}{{*}}{{\\textbf{{{model_display}}}}}" \
                    if model_raw != prev_model else ""
        n_str     = f"\\multirow{{{n_rows_nconst}}}{{*}}{{{n}}}" \
                    if (n != prev_n or model_raw != prev_model) else ""

        flops_val = r.get("flops_mean", float("nan"))
        try:
            flops_display = f"{int(float(flops_val)/1000)}K" if not math.isnan(float(flops_val)) else "N/A"
        except Exception:
            flops_display = "N/A"
        flops_str = f"\\multirow{{{n_rows_nconst}}}{{*}}{{{flops_display}}}" \
                    if (n != prev_n or model_raw != prev_model) else ""

        lines.append(
            f"    {model_str} & {n_str} & {aug_display} & {flops_str} & "
            f"{safe_fmt(r.get('test_acc_mean', float('nan')), 2)} \\pm "
            f"{safe_fmt(r.get('test_acc_std',  float('nan')), 2)} & "
            f"{safe_fmt(r.get('lorentz_flip_mean', float('nan')), 3)} \\pm "
            f"{safe_fmt(r.get('lorentz_flip_std',  float('nan')), 3)} & "
            f"{safe_fmt(r.get('so2_flip_mean', float('nan')), 3)} \\pm "
            f"{safe_fmt(r.get('so2_flip_std',  float('nan')), 3)} & "
            f"{safe_fmt(r.get('so3_flip_mean', float('nan')), 3)} \\pm "
            f"{safe_fmt(r.get('so3_flip_std',  float('nan')), 3)} \\\\"
        )
        prev_model, prev_n = model_raw, n

    return lines


def write_latex_table(rows: List[Dict], output_path: Path) -> None:
    """Write one .tex file per method, each as a standalone LaTeX document."""
    if not rows:
        print("[write_latex] No rows — skipping LaTeX output.")
        return

    METHOD_DISPLAY = {"augmentation": "Augmentation", "symm_loss": "Symmetry Loss"}

    # Group rows by method
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(str(r.get("method", "augmentation")).lower(), []).append(r)

    HEADER = [
        r"\documentclass[preview,border=1pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs, siunitx, adjustbox, multirow}",
        r"\sisetup{separate-uncertainty=true, bracket-numbers=false}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\scriptsize",
    ]
    COL_SPEC = r"\begin{tabular}{ll l c S[table-format=2.2(2)] S[table-format=1.3(3)] S[table-format=1.3(3)] S[table-format=1.3(3)]}"
    COL_HEAD = (
        r"    \textbf{Model} & \textbf{\# Const} & \textbf{Group} & \textbf{FLOPs} & "
        r"{\textbf{Acc. (\%)}} & {\textbf{Lorentz flip}} & {\textbf{SO(2) flip}} & {\textbf{SO(3) flip}} \\"
    )

    stem = output_path.with_suffix("")
    for method, method_rows in sorted(by_method.items()):
        method_path = Path(f"{stem}_{method}.tex")
        method_label = METHOD_DISPLAY.get(method, method)
        lines = HEADER + [
            f"% Method: {method_label}",
            COL_SPEC,
            r"    \toprule",
            COL_HEAD,
            r"    \midrule",
        ]
        lines += _latex_table_body(method_rows)
        lines += [r"    \bottomrule", r"\end{tabular}", r"\end{document}"]
        method_path.parent.mkdir(parents=True, exist_ok=True)
        method_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[write_latex] Wrote {len(method_rows)}-row table ({method_label}) to {method_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-dir",  type=Path, default=Path("logs/mlflow/mlruns"))
    parser.add_argument("--output",      type=Path, default=Path("results/summary_results_table.csv"))
    parser.add_argument("--min-folds",   type=int,  default=5)
    parser.add_argument("--method",      type=str,  default=None,
                        help="Only write results for this method (e.g. 'augmentation', "
                             "'symm_loss', 'equivariance'). Writes all methods when omitted.")
    parser.add_argument("--debug",       action="store_true",
                        help="Print layout of first run and exit.")
    args = parser.parse_args()

    if args.debug:
        debug_one_run(args.mlflow_dir)
        return

    all_records    = collect_fold_records(args.mlflow_dir)
    joined_records = join_train_eval_records(all_records)
    latest_records = keep_latest_per_fold(joined_records)
    rows           = aggregate_rows(latest_records, min_folds=args.min_folds)

    if args.method is not None:
        method_filter = args.method.strip().lower()
        rows = [r for r in rows if str(r.get("method", "")).lower() == method_filter]
        print(f"[filter] Keeping only method='{method_filter}' — {len(rows)} rows remaining.")

    write_csv(rows, args.output)
    write_latex_table(rows, args.output.with_suffix(".tex"))


if __name__ == "__main__":
    main()