#!/usr/bin/env python3
"""Render the thesis results figures directly from the results table CSVs.

One figure per training regime (Augmentation, SCL), each a 3-row x N-model grid
of diverging bar charts, all expressed as a delta against the unaugmented (None)
baseline for that same model/constituent count -- positive is always "better":
  row 1  Delta accuracy            (pp, accuracy(group) - accuracy(None))
  row 2  Delta <invariance metric> (baseline - augmented; already computed by
                                     add_pvalues.py, e.g. Delta Wasserstein)
  row 3  Delta stable fraction     (1 - flip rate, group - None)

Columns are the architectures (DeepSets / Transformer / MLP / ...), populated
from whatever regimes and models are actually present in the results tables.
Bars span SO(2) / SO(3) / Lorentz only -- None is the zero line by construction.

Data source
-----------
Reads results/<features>/results_table_{baseline,augmentation,symm_loss}_pvalues.csv.
These _pvalues.csv files carry the bootstrap p-values, delta-invariance-metric and
flip-breakdown columns that add_pvalues.py adds on top of the raw ResultsTableCallback
output, so it must be run first:

    python src/utils/add_pvalues.py results/*/results_table_*.csv
    python src/utils/make_figures.py --features pt_eta_phi

By default every results/<features>/ bucket is merged (matching on (regime, model,
nconstituents, augmentation_scheme)), which only makes sense when buckets are
complementary (e.g. so2/so3 in one bucket, lorentz in another, no overlapping keys).
If two buckets define the same key the later one silently wins; pass --features to
pin the run to exactly one bucket and skip merging entirely -- do this whenever
buckets overlap (e.g. after re-running a full sweep into a new bucket).

Delta accuracy and delta stable-fraction are paired per-fold (KFold uses a fixed
random_state, so fold i is identical data across conditions) directly from the
test_acc_folds / flip_stable_folds columns already in the _pvalues.csv files --
same mean-of-paired-differences approach add_pvalues.py uses for delta_<metric>.

Figures are written as PNGs (fig_augmentation.svg / fig_symm_loss.svg) under
results/<features>/ when --features is given, else under results/ (default).
PNG, not PDF, because matplotlib's PDF backend renders mathtext as Type 3
fonts, which PDF.js (and so every VSCode PDF-viewer extension) tends to
garble -- a raster PNG has no such font-rendering path to break.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.utils.add_pvalues import _metric_display, _model_display  # noqa: E402

# ----------------------------------------------------------------------
# style: match the thesis body font (Latin Modern Roman == Computer Modern)
# ----------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "font.size": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 120,
})

# colorblind-safe palette (davidmathlogic.com/colorblind, IBM/Wong 4-color set)
COL = {
    "None":    "#004D40",
    "SO(2)":   "#1E88E5",
    "SO(3)":   "#FFC107",
    "Lorentz": "#D81B60",
}
AUG_DISPLAY = {"none": "None", "so2": "SO(2)", "so3": "SO(3)", "lorentz": "Lorentz"}

# preferred column order; models present in the data but not listed here are
# appended alphabetically, so a new architecture (e.g. PELICAN) shows up
# automatically once its results tables exist.
MODEL_ORDER = ["DeepSets", "Transformer", "MLP", "PELICAN"]

# regime label -> results_table_<stem>_pvalues.csv filename stem
REGIME_FILE_STEM = {"Augmentation": "augmentation", "SCL": "symm_loss"}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _parse_folds(raw) -> Optional[List[float]]:
    """Parse a semicolon-separated fold list; None if empty/unparseable."""
    if not raw:
        return None
    try:
        vals = [float(v) for v in str(raw).strip().split(";") if v != ""]
        if not vals or any(math.isnan(v) for v in vals):
            return None
        return vals
    except ValueError:
        return None


def _acc_folds_pct(row: Dict[str, str]) -> Optional[List[float]]:
    """Per-fold test accuracy as percentages, whatever units the CSV used."""
    folds = _parse_folds(row.get("test_acc_folds", ""))
    if folds is None:
        return None
    scale = 100.0 if max(folds) < 2.0 else 1.0
    return [v * scale for v in folds]


def _paired_delta(a_folds: Optional[List[float]], b_folds: Optional[List[float]]) -> Tuple[Optional[float], Optional[float]]:
    """Mean/std of (a - b) over paired per-fold values (matches add_pvalues.py's
    compute_delta_for_row: same fixed KFold split -> fold i is identical data
    across conditions, so the per-fold difference is the right unit to average)."""
    if a_folds is None or b_folds is None or len(a_folds) != len(b_folds) or len(a_folds) < 1:
        return None, None
    diffs = [a - b for a, b in zip(a_folds, b_folds)]
    n = len(diffs)
    mean = sum(diffs) / n
    std = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
    return mean, std


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _discover_rows(results_dir: Path, filename_stem: str, features: Optional[str] = None) -> List[Dict[str, str]]:
    """Collect rows from results/<features>/results_table_<stem>_pvalues.csv.

    With `features` given, reads only that one feature bucket. Otherwise globs
    every results/*/ bucket and merges them -- rows are tagged with a "_bucket"
    key (their source subdirectory) so build_dataset() can warn on collisions,
    since two buckets defining the same (model, n, group) silently clobber
    each other otherwise.
    """
    rows: List[Dict[str, str]] = []
    pattern = f"{features}/results_table_{filename_stem}_pvalues.csv" if features \
        else f"*/results_table_{filename_stem}_pvalues.csv"
    for csv_path in sorted(results_dir.glob(pattern)):
        found = _read_csv_rows(csv_path)
        for row in found:
            row["_bucket"] = csv_path.parent.name
        rows.extend(found)
        print(f"[load] {len(found):3d} rows <- {csv_path.relative_to(REPO_ROOT) if REPO_ROOT in csv_path.parents else csv_path}")
    return rows


def _detect_inv_metric(rows: List[Dict[str, str]]) -> Optional[str]:
    for row in rows:
        for col in row:
            if col.startswith("delta_") and col.endswith("_mean"):
                return col[len("delta_"):-len("_mean")]
    return None


def _opt_float(row: Dict[str, str], key: str) -> Optional[float]:
    v = _parse_float(row.get(key, "nan"))
    return None if math.isnan(v) else v


def _fmt_flops(v: Optional[float]) -> str:
    if v is None or math.isnan(v):
        return "N/A"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _extract_entry(row: Dict[str, str], inv_metric: str) -> Dict[str, Optional[float]]:
    scheme = str(row.get("augmentation_scheme", "")).lower()
    trained = scheme in {"so2", "so3", "lorentz"}
    # Per-group flip folds (so2_flip_stable_folds etc.), unconditional on this
    # row's own augmentation_scheme -- lets a baseline row be paired against an
    # SO(2)-trained row on the SO(2) value specifically. Only present once
    # results_table_callback.py has been re-run with the group_flip_cols change;
    # entries missing here fall back to the legacy "selected" columns below.
    stabfolds_by_group: Dict[str, List[float]] = {}
    recfolds_by_group: Dict[str, List[float]] = {}
    for g in ("so2", "so3", "lorentz"):
        sf = _parse_folds(row.get(f"{g}_flip_stable_folds", ""))
        rf = _parse_folds(row.get(f"{g}_flip_recover_folds", ""))
        if sf is not None:
            stabfolds_by_group[g] = sf
        if rf is not None:
            recfolds_by_group[g] = rf
    return {
        "accfolds": _acc_folds_pct(row),
        # Legacy columns: for a trained row these already equal the matching
        # group's value; for the baseline row they're a mean over so2/so3/lorentz
        # (see ResultsTableCallback._selected_group_value) -- kept only as a
        # fallback for CSVs predating the per-group columns above.
        "stabfolds": _parse_folds(row.get("flip_stable_folds", "")),
        "recfolds": _parse_folds(row.get("flip_recover_folds", "")),
        "stabfolds_by_group": stabfolds_by_group,
        "recfolds_by_group": recfolds_by_group,
        "flops": _opt_float(row, "flops_mean"),
        "dW": _opt_float(row, f"delta_{inv_metric}_mean") if trained else None,
        "dWsd": _opt_float(row, f"delta_{inv_metric}_std") if trained else None,
        # filled in by build_dataset() once the None-baseline entry for the
        # same (regime, model, n) is known: dAcc/dStab/dRec = group - baseline,
        # paired per fold (see _paired_delta)
        "dAcc": None, "dAccSd": None,
        "dStab": None, "dStabSd": None,
        "dRec": None, "dRecSd": None,
    }


def build_dataset(results_dir: Path, features: Optional[str] = None):
    """Discover results tables and build the (regime, model, n, group) -> entry map.

    With `features` given, only that results/<features>/ bucket is read. With
    features=None, every bucket under results/*/ is merged -- when two buckets
    define the same (regime, model, n, group), the later one (alphabetically)
    silently wins unless flagged, so collisions are detected and warned about.
    """
    regime_rows = {label: _discover_rows(results_dir, stem, features) for label, stem in REGIME_FILE_STEM.items()}

    # Baseline (augmentation_scheme=none) rows normally live in their own
    # results_table_baseline_pvalues.csv, but add_pvalues.py also happily builds
    # its baseline index from "none" rows inside the augmentation/symm_loss file
    # itself when run standalone -- so if a bucket has no separate baseline
    # table, fall back to the "none" rows already sitting in its regime tables.
    baseline_rows = _discover_rows(results_dir, "baseline", features)
    if not baseline_rows:
        for rows in regime_rows.values():
            baseline_rows += [r for r in rows if str(r.get("augmentation_scheme", "")).lower() == "none"]
    if not baseline_rows:
        where = f"{results_dir}/{features}/" if features else f"{results_dir}/*/"
        sys.exit(
            f"No baseline (augmentation_scheme=none) rows found under {where} -- neither a "
            "results_table_baseline_pvalues.csv nor any 'none' rows in the augmentation/symm_loss "
            "tables. Run add_pvalues.py first, e.g.:\n"
            "    python src/utils/add_pvalues.py results/*/results_table_*.csv"
        )

    all_rows = baseline_rows + [r for rows in regime_rows.values() for r in rows]
    inv_metric = _detect_inv_metric(all_rows) or "wasserstein"

    D: Dict[Tuple[str, str, int, str], Dict[str, Optional[float]]] = {}
    D_bucket: Dict[Tuple[str, str, int, str], str] = {}
    models_seen: set = set()
    ns_seen: set = set()

    def _set(key: Tuple[str, str, int, str], row: Dict[str, str]) -> None:
        bucket = row.get("_bucket", "?")
        prev_bucket = D_bucket.get(key)
        if prev_bucket is not None and prev_bucket != bucket:
            print(f"[warn] {key} defined in both '{prev_bucket}' and '{bucket}' -- using '{bucket}' "
                  "(pass --features to pin one bucket and avoid this)")
        D[key] = _extract_entry(row, inv_metric)
        D_bucket[key] = bucket

    for regime_label, rows in regime_rows.items():
        if not rows:
            continue

        for row in baseline_rows:
            if str(row.get("augmentation_scheme", "")).lower() != "none":
                continue
            model, n = _model_display(row.get("model", "")), int(row.get("nconstituents", 0))
            _set((regime_label, model, n, "None"), row)
            models_seen.add(model)
            ns_seen.add(n)

        for row in rows:
            scheme = str(row.get("augmentation_scheme", "")).lower()
            if scheme not in AUG_DISPLAY or scheme == "none":
                continue
            model, n = _model_display(row.get("model", "")), int(row.get("nconstituents", 0))
            _set((regime_label, model, n, AUG_DISPLAY[scheme]), row)
            models_seen.add(model)
            ns_seen.add(n)

    # Delta accuracy / delta stable-fraction against the None baseline for the
    # same (regime, model, n), paired per fold.
    scheme_of = {v: k for k, v in AUG_DISPLAY.items()}  # "SO(2)" -> "so2", etc.
    for (regime_label, model, n, group), entry in D.items():
        if group == "None":
            continue
        base = D.get((regime_label, model, n, "None"))
        if base is None:
            continue
        entry["dAcc"], entry["dAccSd"] = _paired_delta(entry["accfolds"], base["accfolds"])

        scheme = scheme_of.get(group)
        entry_stab = entry["stabfolds_by_group"].get(scheme) if scheme else None
        base_stab = base["stabfolds_by_group"].get(scheme) if scheme else None
        entry["dStab"], entry["dStabSd"] = _paired_delta(
            entry_stab if entry_stab is not None else entry["stabfolds"],
            base_stab if base_stab is not None else base["stabfolds"],
        )

        entry_rec = entry["recfolds_by_group"].get(scheme) if scheme else None
        base_rec = base["recfolds_by_group"].get(scheme) if scheme else None
        entry["dRec"], entry["dRecSd"] = _paired_delta(
            entry_rec if entry_rec is not None else entry["recfolds"],
            base_rec if base_rec is not None else base["recfolds"],
        )

    models = [m for m in MODEL_ORDER if m in models_seen] + sorted(models_seen - set(MODEL_ORDER))
    ns = sorted(ns_seen)
    regimes = [r for r in REGIME_FILE_STEM if regime_rows[r]]
    return D, models, ns, regimes, inv_metric


# ---------------------------------------------------------------------------
# scaffold shared by the per-regime figure: R x C grid + row/col labels
# ---------------------------------------------------------------------------

def scaffold(row_labels: List[str], models: List[str], figsize, sharey="row", height_ratios=None):
    gridspec_kw = {"height_ratios": height_ratios} if height_ratios else None
    fig, axes = plt.subplots(len(row_labels), len(models), figsize=figsize,
                              sharey=sharey, squeeze=False, gridspec_kw=gridspec_kw)
    for j, mod in enumerate(models):
        axes[0, j].set_title(mod, fontweight="bold", pad=8)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#000000", alpha=0.06, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
    return fig, axes


def group_legend(fig, groups, ncol, y=0.005, fontsize=10):
    handles = [Patch(facecolor=COL[g], edgecolor="none", label=g) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               frameon=False, bbox_to_anchor=(0.5, y), handlelength=1.5,
               columnspacing=1.8, handletextpad=0.6, fontsize=fontsize)


def _save(fig, out_dir: Path, name: str, missing: List[Tuple]) -> None:
    # PNG, not PDF: matplotlib's PDF backend emits mathtext (the $\Delta$ etc.
    # labels) as Type 3 fonts, which PDF.js -- and so every VSCode PDF-viewer
    # extension -- has long-standing bugs rendering, showing garbled glyphs.
    # A raster PNG sidesteps that entirely and just works everywhere.
    fig.savefig(out_dir / f"{name}.svg", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    if missing:
        print(f"[warn] {name}: {len(missing)} (regime, model, n, group) combos missing from the results tables, left blank:")
        for key in missing:
            print(f"        {key}")


# ---------------------------------------------------------------------------
# per-regime figure: 3 stacked delta-bar rows (accuracy / invariance / self-
# consistency) x one column per architecture. Every row is a diverging bar
# chart against the None baseline, so bars can extend below zero -- higher is
# always better here (more accurate, more invariant, more self-consistent).
# ---------------------------------------------------------------------------

# (data key, std key, row label, y-axis label) for the 3 stacked delta-bar rows
def _row_specs(inv_metric: str):
    return [
        ("dAcc", "dAccSd", "Accuracy", "$\\Delta$ accuracy (pp)"),
        ("dW", "dWsd", _metric_display(inv_metric), f"$\\Delta$ {_metric_display(inv_metric)}"),
        ("dStab", "dStabSd", "Self-consistency", "$\\Delta$ stable fraction $1-\\mathrm{FR}$"),
    ]

RECOVER_COLOR = "#1a1a1a"


def fig_regime(D, models, ns, regime: str, out_dir: Path, inv_metric: str, features: Optional[str] = None):
    import numpy as np
    specs = _row_specs(inv_metric)
    row_labels = [label for _, _, label, _ in specs] + ["FLOPs"]
    n_bar_rows = len(specs)
    # FLOPs row only holds a centered text annotation (no bars), so it's given
    # a much shorter height than the bar rows -- otherwise a plain subplots
    # grid gives it equal height, leaving it mostly empty and pushing the
    # legend far below the actual plots.
    flops_ratio = 0.16
    height_ratios = [1.0] * n_bar_rows + [flops_ratio]
    fig, axes = scaffold(row_labels, models,
                          (3.8 * len(models), 2.7 * n_bar_rows + 2.7 * flops_ratio + 0.5),
                          height_ratios=height_ratios)

    wgroups = ["SO(2)", "SO(3)", "Lorentz"]
    x = np.arange(len(ns))
    w = 0.26
    off = {"SO(2)": -w, "SO(3)": 0.0, "Lorentz": w}
    missing = []

    for i, (key, sdkey, _, ylabel) in enumerate(specs):
        row_vals = []
        for j, mod in enumerate(models):
            ax = axes[i, j]
            for g in wgroups:
                xs, ys, es, recs = [], [], [], []
                for k, n in enumerate(ns):
                    entry = D.get((regime, mod, n, g))
                    val = entry.get(key) if entry else None
                    if entry is None or val is None:
                        missing.append((regime, mod, n, g, key))
                        continue
                    xs.append(x[k] + off[g])
                    ys.append(val)
                    es.append(entry.get(sdkey) or 0.0)
                    recs.append(entry.get("dRec"))
                    row_vals.append(val)
                    if key == "dStab" and entry.get("dRec") is not None:
                        row_vals.append(entry["dRec"])
                if xs:
                    ax.bar(xs, ys, width=w, color=COL[g], edgecolor="white", lw=0.5, zorder=2)
                    ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor="#333",
                                elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
                    # self-consistency row: overlay the recover rate (P(correct
                    # after | wrong before), a different conditional than the
                    # dStab bar) as a thick band rather than a thin tick mark
                    if key == "dStab":
                        for xk, rk in zip(xs, recs):
                            if rk is None:
                                continue
                            band = ax.hlines(rk, xk - w * 0.42, xk + w * 0.42, colors=RECOVER_COLOR,
                                              linewidth=6.0, capstyle="butt", zorder=4, alpha=0.9)
                            band.set_path_effects([
                                patheffects.withStroke(linewidth=8.0, foreground="white")
                            ])
            ax.axhline(0, color="#444444", lw=0.8, zorder=1)
            ax.set_xticks(x), ax.set_xticklabels(ns)
            ax.set_xlim(-0.6, len(ns) - 0.4)
            if i == n_bar_rows - 1:
                ax.set_xlabel("constituents $N$")
        axes[i, 0].set_ylabel(ylabel)
        if row_vals:
            lo, hi = min(row_vals), max(row_vals)
            pad = 0.15 * max(hi - lo, abs(hi), abs(lo), 1e-6)
            axes[i, 0].set_ylim(min(0, lo - pad), max(0, hi + pad))

    # FLOPs footer row: one static (regime-independent) number per (model, n),
    # so architecture cost is readable at a glance alongside the deltas above.
    flops_axes = axes[n_bar_rows]
    for j, mod in enumerate(models):
        ax = flops_axes[j]
        for k, n in enumerate(ns):
            entry = D.get((regime, mod, n, "None"))
            flops = entry.get("flops") if entry else None
            ax.text(x[k], 0.5, _fmt_flops(flops), transform=ax.get_xaxis_transform(),
                     ha="center", va="center", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0",
                                edgecolor="#bbbbbb", linewidth=0.6))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlim(-0.6, len(ns) - 0.4)
    flops_axes[0].set_ylabel("FLOPs")

    handles = [Patch(facecolor=COL[g], edgecolor="none", label=g) for g in wgroups]
    handles.append(Line2D([0], [0], color=RECOVER_COLOR, lw=5, label="recover rate"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 0.005), handlelength=1.5, columnspacing=1.8,
               handletextpad=0.6, fontsize=13)
    fig.tight_layout(rect=[0.0, 0.035, 1, 0.975], h_pad=0.4)
    name = f"fig_{REGIME_FILE_STEM[regime]}" + (f"_{features}" if features else "")
    _save(fig, out_dir, name, missing)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "results",
        help="Root directory containing results/<features>/results_table_*_pvalues.csv (default: results/).",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help="Restrict to one results/<features>/ bucket (e.g. pt_eta_phi), instead of merging every "
             "bucket under results/*/. Figures are then written to results/<features>/ by default -- "
             "pass this whenever more than one feature bucket defines the same (model, group, "
             "nconstituents), so buckets don't silently overwrite each other.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write fig_augmentation.png / fig_symm_loss.png "
             "(default: results/<features>/ if --features is given, else results/).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or (
        args.results_dir / args.features if args.features else args.results_dir
    )

    D, models, ns, regimes, inv_metric = build_dataset(args.results_dir, args.features)
    if not regimes:
        where = f"{args.results_dir}/{args.features}/" if args.features else f"{args.results_dir}/*/"
        sys.exit(f"No augmentation/symm_loss _pvalues.csv tables found under {where} -- run add_pvalues.py first.")
    if not models or not ns:
        sys.exit("No usable rows found (empty model/nconstituents set) -- check the results tables.")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] features={args.features or 'ALL (merged)'} regimes={regimes} models={models} "
          f"n_constituents={ns} inv_metric={inv_metric}")

    names = []
    for regime in regimes:
        fig_regime(D, models, ns, regime, out_dir, inv_metric, args.features)
        names.append(f"fig_{REGIME_FILE_STEM[regime]}" + (f"_{args.features}" if args.features else ""))
    print(f"\ndone: {', '.join(names)} (png) -> {out_dir}")


if __name__ == "__main__":
    main()
