#!/usr/bin/env python3
"""Build a compact summary table from MLflow fold runs.

The script expects MLflow runs logged by this repo where each fold run has:
- `params/tags` containing `fold_<idx>`
- `params/model/_target_`
- `params/data/augmentation_scheme`
- `params/data/nconstituents`
- `metrics/test/acc`
- `metrics/invariance/{so2,so3,lorentz}_flip`
- optional `metrics/model/flops`

For each `(model, nconstituents, augmentation_scheme, fold_idx)` it keeps the
latest run (by `meta.yaml:end_time`) and then aggregates across folds.
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

from pdflatex import PDFLaTeX


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
    def selected_flip_key(self) -> str:
        if self.augmentation_scheme in {"so2", "so3", "lorentz"}:
            return f"invariance/{self.augmentation_scheme}_flip"
        return "invariance/mean_flip"

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
        return float(raw)
    except ValueError:
        return float("nan")


def read_text(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def read_metric_last(metric_path: Path) -> float:
    """Read the final logged value from an MLflow metric file."""
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


def parse_end_time(meta_path: Path) -> int:
    raw = read_text(meta_path)
    if raw is None:
        return -1
    for line in raw.splitlines():
        if line.startswith("end_time:"):
            return int(line.split(":", 1)[1].strip())
    return -1


def parse_fold_idx(tags_raw: Optional[str]) -> Optional[int]:
    if not tags_raw:
        return None
    match = re.search(r"fold_(\d+)", tags_raw)
    if not match:
        return None
    return int(match.group(1))


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
    records: List[FoldRecord] = []
    for meta_path in mlflow_root.glob("*/*/meta.yaml"):
        run_dir = meta_path.parent
        run_id = run_dir.name

        tags_raw = read_text(run_dir / "params" / "tags")
        fold_idx = parse_fold_idx(tags_raw)
        if fold_idx is None:
            continue

        model_target = read_text(run_dir / "params" / "model" / "_target_")
        aug = read_text(run_dir / "params" / "data" / "augmentation_scheme")
        nconst_raw = read_text(run_dir / "params" / "data" / "nconstituents")
        if model_target is None or aug is None or nconst_raw is None:
            continue

        nconst = int(safe_float(nconst_raw))
        test_acc = read_metric_last(run_dir / "metrics" / "test" / "acc")
        if math.isnan(test_acc):
            continue

        so2_flip = read_metric_last(run_dir / "metrics" / "invariance" / "so2_flip")
        so3_flip = read_metric_last(run_dir / "metrics" / "invariance" / "so3_flip")
        lorentz_flip = read_metric_last(run_dir / "metrics" / "invariance" / "lorentz_flip")
        flops = read_metric_last(run_dir / "metrics" / "model" / "flops")

        if math.isnan(flops):
            flops = safe_float(read_text(run_dir / "params" / "model" / "flops"))

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


def aggregate_simple_rows(records: List[FoldRecord], min_folds: int) -> List[Dict[str, object]]:
    # Group by (model_family, model_target, nconstituents, augmentation_scheme)
    grouped: Dict[Tuple[str, str, int, str], List[FoldRecord]] = {}
    for rec in records:
        key = (
            rec.model_family,
            rec.model_target,
            rec.nconstituents,
            rec.augmentation_scheme,
        )
        grouped.setdefault(key, []).append(rec)

    rows: List[Dict[str, object]] = []
    for key, folds in grouped.items():
        if len(folds) < min_folds:
            continue
        test_vals = [r.test_acc for r in folds]
        flops_vals = [r.flops for r in folds]
        test_mean = safe_mean(test_vals)
        test_std = safe_std(test_vals)
        flops_mean = safe_mean(flops_vals)
        # Invariance metrics
        so2_vals = [r.so2_flip for r in folds]
        so3_vals = [r.so3_flip for r in folds]
        lorentz_vals = [r.lorentz_flip for r in folds]
        row = {
            "model": key[0],
            "model_target": key[1],
            "nconstituents": key[2],
            "augmentation_scheme": key[3],
            "folds_used": len(folds),
            "test_acc_mean": round(test_mean * 100, 4), 
            "test_acc_std": round(test_std * 100, 4),       
            "so2_flip_mean": round(safe_mean(so2_vals), 6),
            "so2_flip_std": round(safe_std(so2_vals), 6),
            "so3_flip_mean": round(safe_mean(so3_vals), 6),
            "so3_flip_std": round(safe_std(so3_vals), 6),
            "lorentz_flip_mean": round(safe_mean(lorentz_vals), 6),
            "lorentz_flip_std": round(safe_std(lorentz_vals), 6),
            "flops_mean": round(flops_mean, 3) if not math.isnan(flops_mean) else float("nan"),
        }
        rows.append(row)
    return rows


def load_flops_lookup_from_slurm(slurm_dir: Path) -> Dict[Tuple[str, int], float]:
    """Build (model_family, nconstituents) -> FLOPs lookup from SLURM logs."""
    if not slurm_dir.exists():
        return {}

    flops_by_key: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    nconst_re = re.compile(r"nconst(\d+)")
    flops_re = re.compile(r"Total FLOPs:\s*([0-9.]+)\s*([KMG]?)")
    unit_scale = {"": 1.0, "K": 1e3, "M": 1e6, "G": 1e9}

    for out_file in slurm_dir.rglob("*.out"):
        name = out_file.name
        nconst_match = nconst_re.search(name)
        if not nconst_match:
            continue
        nconst = int(nconst_match.group(1))

        if name.startswith("deepsets_"):
            model_family = "deepsets"
        elif name.startswith("mlp_"):
            model_family = "mlp"
        elif name.startswith("transformer_"):
            model_family = "transformer"
        else:
            continue

        text = out_file.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = flops_re.search(line)
            if not match:
                continue
            raw = float(match.group(1))
            unit = match.group(2)
            flops_by_key[(model_family, nconst)].append(raw * unit_scale[unit])

    lookup: Dict[Tuple[str, int], float] = {}
    for key, values in flops_by_key.items():
        counts = Counter(values)
        lookup[key] = counts.most_common(1)[0][0]
    return lookup


def fill_missing_flops(
    rows: List[Dict[str, object]], flops_lookup: Dict[Tuple[str, int], float]
) -> None:
    for row in rows:
        key = (str(row["model"]), int(row["nconstituents"]))
        if key in flops_lookup:
            row["flops_mean"] = round(float(flops_lookup[key]), 3)


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_target",
        "nconstituents",
        "augmentation_scheme",
        "folds_used",
        "test_acc_mean",
        "test_acc_std",
        "test_acc_pm",
        "selected_flip_key",
        "selected_flip_mean",
        "selected_flip_std",
        "selected_flip_pm",
        "so2_flip_mean",
        "so3_flip_mean",
        "lorentz_flip_mean",

        "fold_ids",
        "run_ids",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_cross_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    # Collect all fieldnames from all rows
    fieldnames = set()
    for row in rows:
        fieldnames.update(row.keys())
    fieldnames = sorted(fieldnames)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_markdown(rows: List[Dict[str, object]]) -> None:
    headers = [
        "model",
        "nconst",
        "aug",
        "folds",
        "test acc (mean +/- std)",
        "selected flip (mean +/- std)",
        "FLOPs",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| "
            f"{r['model']} | "
            f"{r['nconstituents']} | "
            f"{r['augmentation_scheme']} | "
            f"{r['folds_used']} | "
            f"{r['test_acc_pm']} | "
            f"{r['selected_flip_pm']} | "
            f"{r['flops_mean']} |"
        )

def write_latex_table(rows: List[Dict[str, object]], output_path: Path):
    # Sort: Model (alphabetical), then N (small to large)
    sorted_rows = sorted(rows, key=lambda x: (x['model'], x['nconstituents']))

    latex_lines = [
        r"\documentclass[preview,border=1pt]{standalone}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage{booktabs, siunitx, adjustbox, multirow}",
        r"\sisetup{separate-uncertainty=true, bracket-numbers=false}",
        r"\begin{document}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\scriptsize",
        # Acc: 2 digits.2 digits(2 digits uncertainty) | flip
        # : 1 digit.3 digits(3 digits uncertainty)
        r"\begin{tabular}{ll c c S[table-format=2.2(2)] S[table-format=1.3(3)] S[table-format=1.3(3)] S[table-format=1.3(3)]}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{\# Const} & \textbf{Aug.} & \textbf{FLOPs} & {\textbf{Acc. (\%)}} & {\textbf{Lorentz $J_S$}} & {\textbf{SO(2) $J_S$}} & {\textbf{SO(3) $J_S$}} \\",
        r"    \midrule"
    ]

    prev_model = None
    prev_n = None

    for i, r in enumerate(sorted_rows):
        model_raw = r['model']
        model_display = "MLP" if model_raw == 'mlp' else model_raw.capitalize()
        n = r['nconstituents']
        # Clean up augmentation names (e.g., so2 -> SO(2))
        aug_map = {"none": "None", "lorentz": "Lorentz", "so2": "SO(2)", "so3": "SO(3)"}
        aug = aug_map.get(r['augmentation_scheme'].lower(), r['augmentation_scheme'].capitalize())
        
        # 1. Handle Model Column (Grouped)
        # Note: adjust the multiplier (12) if you have more/fewer experiments per model
        model_str = f"\\multirow{{12}}{{*}}{{\\textbf{{{model_display}}}}}" if model_raw != prev_model else ""
        
        if prev_model is not None and model_raw != prev_model:
            latex_lines.append(r"    \specialrule{.1em}{.5em}{.5em}")

        # 2. Handle N Column (Grouped)
        n_str = f"\\multirow{{4}}{{*}}{{{n}}}" if (n != prev_n or model_raw != prev_model) else ""
        
        if prev_n is not None and n != prev_n and model_raw == prev_model:
            latex_lines.append(r"    \cmidrule{2-8}")

        # 3. Handle FLOPs (Grouped)
        flops_val = r.get("flops_mean", 0)
        flops_display = f"{int(flops_val/1000)}K" if not math.isnan(flops_val) else "N/A"
        flops_str = f"\\multirow{{4}}{{*}}{{{flops_display}}}" if (n != prev_n or model_raw != prev_model) else ""

        # Construct the row with uncertainty for ALL metrics
        row = (
            f"    {model_str} & {n_str} & {aug} & {flops_str} & "
            f"{r['test_acc_mean']:.2f} \\pm {r['test_acc_std']:.2f} & "
            f"{r['lorentz_flip_mean']:.3f} \\pm {r['lorentz_flip_std']:.3f} & "
            f"{r['so2_flip_mean']:.3f} \\pm {r['so2_flip_std']:.3f} & "
            f"{r['so3_flip_mean']:.3f} \\pm {r['so3_flip_std']:.3f} \\\\"
        )
        latex_lines.append(row)

        prev_model = model_raw
        prev_n = n

    latex_lines.extend([
        r"    \bottomrule",
        r"\end{tabular}",
        r"\end{document}"
    ])

    output_path.write_text("\n".join(latex_lines))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow-dir",
        type=Path,
        default=Path("logs/mlflow/mlruns"),
        help="Path to MLflow mlruns directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/summary_results_table_flip.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--min-folds",
        type=int,
        default=5,
        help="Keep only model/aug groups with at least this many folds.",
    )
    parser.add_argument(
        "--slurm-dir",
        type=Path,
        default=Path("logs/slurm"),
        help="Optional SLURM logs directory to fill missing FLOPs.",
    )
    args = parser.parse_args()

    all_records = collect_fold_records(args.mlflow_dir)
    latest_records = keep_latest_per_fold(all_records)
    rows = aggregate_simple_rows(latest_records, min_folds=args.min_folds)
    fill_missing_flops(rows, load_flops_lookup_from_slurm(args.slurm_dir))

    write_cross_csv(rows, args.output)
    # print(f"Table written to {args.output}")
    # Optionally, print a markdown summary for a few key fields
    # print_markdown(rows)

    # 2. Save the LaTeX version
    latex_path = args.output.with_suffix(".tex")
    write_latex_table(rows, latex_path)
    
    print(f"Wrote CSV to {args.output}")
    print(f"Wrote LaTeX table to {latex_path}")


if __name__ == "__main__":
    main()