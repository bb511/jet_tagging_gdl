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
from collections import Counter, defaultdict
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
    fold_idx: int
    run_id: str
    end_time: int
    test_acc: float
    so2_flip: float
    so3_flip: float
    lorentz_flip: float
    flops: float

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
    """Extract fold index from a string like 'fold 3/5' or 'fold_3' or just '3'."""
    if raw is None:
        return None
    raw = raw.strip()
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

        if model_target is None or aug is None or nconst_raw is None:
            continue

        nconst = int(safe_float(nconst_raw))

        # --- Metrics ---
        test_acc = read_metric(run_dir, "test/acc")
        if math.isnan(test_acc):
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

        # FLOPs — try metric first, then param
        flops = read_metric(run_dir, "model/flops")
        if math.isnan(flops):
            flops = safe_float(read_param(run_dir, "model/flops"))
        if math.isnan(flops):
            flops = safe_float(read_param(run_dir, "model/flops_total"))

        records.append(
            FoldRecord(
                model_family=model_family_from_target(model_target),
                model_target=model_target,
                nconstituents=nconst,
                augmentation_scheme=aug.strip().lower(),
                fold_idx=fold_idx,
                run_id=run_id,
                end_time=parse_end_time(meta_path),
                test_acc=test_acc,
                so2_flip=so2_flip,
                so3_flip=so3_flip,
                lorentz_flip=lorentz_flip,
                flops=flops,
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
    latest: Dict[Tuple[str, str, int, str, int], FoldRecord] = {}
    for rec in records:
        key = (
            rec.model_family,
            rec.model_target,
            rec.nconstituents,
            rec.augmentation_scheme,
            rec.fold_idx,
        )
        prev = latest.get(key)
        if prev is None or rec.end_time > prev.end_time:
            latest[key] = rec
    return list(latest.values())


def aggregate_rows(records: List[FoldRecord], min_folds: int) -> List[Dict]:
    grouped: Dict[Tuple[str, str, int, str], List[FoldRecord]] = {}
    for rec in records:
        key = (rec.model_family, rec.model_target, rec.nconstituents, rec.augmentation_scheme)
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


def write_latex_table(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        print("[write_latex] No rows — skipping LaTeX output.")
        return

    sorted_rows = sorted(rows, key=lambda x: (x["model"], x["nconstituents"]))

    aug_map = {"none": "None", "lorentz": "Lorentz", "so2": "SO(2)", "so3": "SO(3)"}

    def safe_fmt(val, digits):
        try:
            if math.isnan(float(val)):
                return "N/A"
            return f"{float(val):.{digits}f}"
        except Exception:
            return "N/A"

    lines = [
        r"\documentclass[preview,border=1pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs, siunitx, adjustbox, multirow}",
        r"\sisetup{separate-uncertainty=true, bracket-numbers=false}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\scriptsize",
        r"\begin{tabular}{ll c c S[table-format=2.2(2)] S[table-format=1.3(3)] S[table-format=1.3(3)] S[table-format=1.3(3)]}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{\# Const} & \textbf{Aug.} & \textbf{FLOPs} & "
        r"{\textbf{Acc. (\%)}} & {\textbf{Lorentz flip}} & {\textbf{SO(2) flip}} & {\textbf{SO(3) flip}} \\",
        r"    \midrule",
    ]

    prev_model, prev_n = None, None
    for r in sorted_rows:
        model_raw     = r["model"]
        model_display = "MLP" if model_raw == "mlp" else model_raw.capitalize()
        n             = r["nconstituents"]
        aug           = aug_map.get(str(r["augmentation_scheme"]).lower(), str(r["augmentation_scheme"]))

        if prev_model is not None and model_raw != prev_model:
            lines.append(r"    \specialrule{.1em}{.5em}{.5em}")
        elif prev_n is not None and n != prev_n:
            lines.append(r"    \cmidrule{2-8}")

        model_str = f"\\multirow{{12}}{{*}}{{\\textbf{{{model_display}}}}}" if model_raw != prev_model else ""
        n_str     = f"\\multirow{{4}}{{*}}{{{n}}}"                         if (n != prev_n or model_raw != prev_model) else ""

        flops_val = r.get("flops_mean", float("nan"))
        try:
            flops_display = f"{int(float(flops_val)/1000)}K" if not math.isnan(float(flops_val)) else "N/A"
        except Exception:
            flops_display = "N/A"
        flops_str = f"\\multirow{{4}}{{*}}{{{flops_display}}}" if (n != prev_n or model_raw != prev_model) else ""

        lines.append(
            f"    {model_str} & {n_str} & {aug} & {flops_str} & "
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

    lines.extend([r"    \bottomrule", r"\end{tabular}", r"\end{document}"])
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[write_latex] Wrote LaTeX table to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-dir",  type=Path, default=Path("logs/mlflow/mlruns"))
    parser.add_argument("--output",      type=Path, default=Path("results/summary_results_table.csv"))
    parser.add_argument("--min-folds",   type=int,  default=5)
    parser.add_argument("--debug",       action="store_true",
                        help="Print layout of first run and exit.")
    args = parser.parse_args()

    if args.debug:
        debug_one_run(args.mlflow_dir)
        return

    all_records    = collect_fold_records(args.mlflow_dir)
    latest_records = keep_latest_per_fold(all_records)
    rows           = aggregate_rows(latest_records, min_folds=args.min_folds)

    write_csv(rows, args.output)
    write_latex_table(rows, args.output.with_suffix(".tex"))


if __name__ == "__main__":
    main()