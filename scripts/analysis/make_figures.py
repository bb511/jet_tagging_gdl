#!/usr/bin/env python3
"""Draw the per-regime comparison figures from the results table CSVs.

One figure per training regime (augmentation, SCL), a grid of one bar row per metric by one column
per architecture. Every bar is a delta against the unaugmented baseline of the same model and
constituent count, so positive is better in every row::

    accuracy (clean)        pp, the deltas table's test_acc rows
    accuracy (transformed)  pp, its accuracy_transformed rows
    <invariance metric>     its wasserstein rows, the distance between the clean and the
                            transformed logits
    self-consistency        pp, its flip rows, the unconditional flip rate, positive being
                            fewer flips; the recover rate overlaid as a band
    FLOPs                   a text footer, one regime-independent number per (model, N)

Bars cover SO(2), SO(3) and Lorentz; the unaugmented run is the zero line by construction.

``--features`` names the one ``results/<features>/`` directory to read, since the directories hold
different input feature lists and their rows are not comparable. A results row holds one run's own
numbers, so every delta comes from the tables ``compare_to_baseline.py`` writes, named with
``--deltas``: which run is the reference is chosen there. Their rows are matched on the method as
well as the model, so both regimes are named on one command line::

    python3 scripts/analysis/compare_to_baseline.py \\
        results/px_py_pz_e/results_table_augmentation.csv \\
        results/px_py_pz_e/results_table_baseline.csv
    python3 scripts/analysis/compare_to_baseline.py \\
        results/px_py_pz_e/results_table_symm_loss.csv \\
        results/px_py_pz_e/results_table_baseline.csv
    python3 scripts/analysis/make_figures.py --features px_py_pz_e \\
        --deltas results/px_py_pz_e/deltas_augmentation_vs_baseline.csv \\
                 results/px_py_pz_e/deltas_symm_loss_vs_baseline.csv

Every delta is paired per fold, the unit ``compare_to_baseline.py`` averages over, and signed there
so that positive is the better model, which is what puts every row of the figure the same way up.

The figures are SVG rather than PDF because matplotlib's PDF backend emits mathtext as Type 3
fonts, which PDF.js and the VSCode viewers built on it garble.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# common.py puts the repository root on the import path, so it is imported before src.symmetry.
from common import (
    AUG_DISPLAY,
    GROUP_LABEL,
    REPO_ROOT,
    delta_key,
    detect_inv_metric_suffix,
    index_deltas,
    metric_display,
    model_display,
    opt_float,
    read_csv,
    read_deltas,
    shorten,
)
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.symmetry import TABULATED_GROUPS

# Match the thesis body font: Latin Modern Roman is Computer Modern.
mpl.rcParams.update(
    {
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
    }
)

# Colourblind-safe palette (davidmathlogic.com/colorblind, the IBM/Wong four-colour set).
COL = {"None": "#004D40", "SO(2)": "#1E88E5", "SO(3)": "#FFC107", "Lorentz": "#D81B60"}
RECOVER_COLOR = "#1a1a1a"

# Display label -> augmentation_scheme, the groups a figure draws bars for. Which groups, and in
# what order, is TABULATED_GROUPS, so a group the study adds raises here rather than going missing
# from the figures; it needs a colour in COL and an offset in OFFSET as well.
GROUPS = {GROUP_LABEL[group]: group for group in TABULATED_GROUPS}

WIDTH = 0.26
OFFSET = {"SO(2)": -WIDTH, "SO(3)": 0.0, "Lorentz": WIDTH}

# The FLOPs footer carries one line of text and no bars. At the height of a bar row it would sit
# almost empty and push the legend far below the panels.
FLOPS_RATIO = 0.16
BOX = dict(
    boxstyle="round,pad=0.25", facecolor="#f0f0f0", edgecolor="#bbbbbb", linewidth=0.6
)

# The style of the error whiskers, and of the recover-rate band. The recover rate is
# P(correct after | wrong before), a different conditional from the unconditional flip rate the
# bar carries, so it is drawn as a band across the bar rather than as a tick on it.
WHISKER = dict(
    fmt="none", ecolor="#333", elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3
)
BAND = dict(colors=RECOVER_COLOR, linewidth=6.0, capstyle="butt", zorder=4, alpha=0.9)

# Preferred column order; a model present in the data but not listed here is appended
# alphabetically, so a re-added architecture (see ARCHITECTURES.md) needs no edit here.
MODEL_ORDER = ["DeepSets", "MLP", "JetFormer", "LorentzNet", "PELICAN"]

# Regime label -> results_table_<stem>.csv filename stem.
REGIME_FILE_STEM = {"Augmentation": "augmentation", "SCL": "symm_loss"}

# One results row as the panels read it: a (mean, deviation) pair per metric, None where the deltas
# tables hold no row for it, and the FLOPs count.
Entry = dict[str, tuple[float, float] | float | None]


def _delta(deltas: dict, key: tuple, scale: float = 1.0) -> tuple[float, float] | None:
    """One deltas row as (mean, deviation), or None when the tables hold no such row."""
    row = deltas.get(key)
    mean = opt_float(row, "delta_mean") if row else None
    if mean is None:
        return None
    return mean * scale, (opt_float(row, "delta_std") or 0.0) * scale


def _fmt_flops(value: float | None) -> str:
    """FLOPs at the magnitude the footer prints them in."""
    if value is None or math.isnan(value):
        return "N/A"
    if value >= 1e6:
        return f"{value / 1e6:.0f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:.0f}"


def _scheme_of(row: dict[str, str]) -> str:
    """The row's augmentation scheme, lowercased."""
    return str(row.get("augmentation_scheme", "")).lower()


def _discover_rows(results_dir: Path, stem: str, features: str) -> list[dict[str, str]]:
    """Read results/<features>/results_table_<stem>.csv, the one directory --features
    names."""
    rows: list[dict[str, str]] = []
    for path in sorted(results_dir.glob(f"{features}/results_table_{stem}.csv")):
        found = read_csv(path)
        rows.extend(found)
        print(f"[load] {len(found):3d} rows <- {shorten(path)}")
    return rows


def _extract_entry(row: dict[str, str], deltas: dict, inv_metric: str) -> Entry:
    """One results row as the entry a panel reads, its deltas looked up rather than
    computed.

    Every lookup is keyed on the row's own method, model, constituent count and group, so an
    unaugmented row finds nothing: it is the zero line the others are drawn against. The two
    accuracies, the flip rate and the recover rate are scaled to percentage points, the
    units the results tables write them in being proportions.
    """
    group = _scheme_of(row)
    return {
        "flops": opt_float(row, "flops_mean"),
        "acc": _delta(deltas, delta_key(row, "", "test_acc"), 100.0),
        "acc_t": _delta(deltas, delta_key(row, group, "accuracy_transformed"), 100.0),
        "inv": _delta(deltas, delta_key(row, group, inv_metric)),
        "flip": _delta(deltas, delta_key(row, group, "flip"), 100.0),
        "recover": _delta(deltas, delta_key(row, group, "flip_recover"), 100.0),
    }


def _baseline_rows(
    results_dir: Path, features: str, regime_rows: dict
) -> list[dict[str, str]]:
    """The unaugmented rows, from their own table or from the regime tables.

    A regime catalogue evaluated on its own writes its ``none`` row into that regime's
    table, so the rows can be present while results_table_baseline.csv is not.
    """
    rows = _discover_rows(results_dir, "baseline", features)
    return rows or [
        r for found in regime_rows.values() for r in found if _scheme_of(r) == "none"
    ]


def _index_rows(
    regime_rows: dict, baselines: list[dict], deltas: dict, metric: str
) -> dict:
    """(regime, model, N, group) -> entry, over the regimes that have a table.

    The unaugmented rows are indexed again under every regime: each regime's bars are deltas
    against the unaugmented run of the same model and constituent count, whose row carries the
    FLOPs and holds the zero line. A row of a scheme the figures draw no bars for is left out.
    """
    zero = [row for row in baselines if _scheme_of(row) == "none"]
    index: dict[tuple[str, str, int, str], Entry] = {}
    for regime, rows in regime_rows.items():
        if not rows:
            continue
        drawn = [r for r in rows if _scheme_of(r) in GROUPS.values()]
        for row in zero + drawn:
            model, n = model_display(row.get("model", "")), int(
                row.get("nconstituents", 0)
            )
            group = AUG_DISPLAY[_scheme_of(row)]
            index[(regime, model, n, group)] = _extract_entry(row, deltas, metric)
    return index


def build_dataset(results_dir: Path, features: str, deltas: dict):
    """Read one results directory and index it for the figures.

    :param features: the ``results/<features>/`` directory name.
    :param deltas: the index of ``common.index_deltas``, every bar of every figure.
    :returns: the (regime, model, N, group) index, the models and constituent counts it
        covers, the regimes that have a table, and the invariance metric those tables
        carry.
    """
    regime_rows = {
        label: _discover_rows(results_dir, stem, features)
        for label, stem in REGIME_FILE_STEM.items()
    }
    baselines = _baseline_rows(results_dir, features, regime_rows)
    if not baselines:
        sys.exit(
            f"No unaugmented (augmentation_scheme=none) rows under {results_dir}/{features}/, "
            "neither a results_table_baseline.csv nor a 'none' row in the regime tables. "
            "Evaluate the baseline runs first, e.g.:\n"
            "    python3 src/eval.py checkpoint_manifest_path=<baseline>/checkpoint_manifest.json"
        )
    every_row = baselines + [row for rows in regime_rows.values() for row in rows]
    metric = detect_inv_metric_suffix(every_row) or "wasserstein"
    index = _index_rows(regime_rows, baselines, deltas, metric)
    # MODEL_ORDER first, then whatever else the tables held, alphabetically.
    seen = {model for _, model, _, _ in index}
    models = [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))
    regimes = [label for label in REGIME_FILE_STEM if regime_rows[label]]
    return index, models, sorted({n for _, _, n, _ in index}), regimes, metric


def scaffold(row_labels: list[str], models: list[str]):
    """The grid both figures use: one bar row per label, a FLOPs footer under them.

    Every panel loses its top and right spine and gains a faint horizontal grid behind
    the bars.

    :param row_labels: the bar rows only; the footer row is added here.
    :returns: the figure and its (len(row_labels) + 1, len(models)) array of axes.
    """
    ratios = [1.0] * len(row_labels) + [FLOPS_RATIO]
    figsize = (3.8 * len(models), 2.7 * len(row_labels) + 2.7 * FLOPS_RATIO + 0.5)
    fig, axes = plt.subplots(
        len(ratios),
        len(models),
        figsize=figsize,
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": ratios},
    )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#000000", alpha=0.06, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)
    for ax, model in zip(axes[0], models):
        ax.set_title(model, fontweight="bold", pad=8)
    return fig, axes


def _draw_group(ax, xs, ys, es, colour, bands=()):
    """One group's bar cluster, its error bars and any recover-rate bands.

    :param bands: (x, y) points to band, empty on a row that carries no recover rate.
    :returns: the vertical extent drawn, whiskers and bands included, so a row's limits
        do not clip them.
    """
    if not xs:
        return []
    ax.bar(xs, ys, width=WIDTH, color=colour, edgecolor="white", lw=0.5, zorder=2)
    ax.errorbar(xs, ys, yerr=es, **WHISKER)
    for x, y in bands:
        band = ax.hlines(y, x - WIDTH * 0.42, x + WIDTH * 0.42, **BAND)
        band.set_path_effects(
            [patheffects.withStroke(linewidth=8.0, foreground="white")]
        )
    whiskers = [y + e for y, e in zip(ys, es)] + [y - e for y, e in zip(ys, es)]
    return whiskers + [y for _, y in bands]


def _finish_row(
    axes_row, ylabel: str, extent: list[float], ns, last: bool, headroom=0.0
) -> None:
    """Label and scale one row of panels; sharey='row' spreads the limits across it.

    The limits fit everything drawn and keep zero in view, the rows diverging about it.

    :param headroom: extra fraction of the span left above, for a row carrying an
        annotation.
    """
    for ax in axes_row:
        ax.axhline(0, color="#444444", lw=0.8, zorder=1)
        ax.set_xticks(list(range(len(ns))))
        ax.set_xticklabels(ns)
        ax.set_xlim(-0.6, len(ns) - 0.4)
        if last:
            ax.set_xlabel("constituents $N$")
    axes_row[0].set_ylabel(ylabel)
    if not extent:
        return
    lo, hi = min(extent), max(extent)
    pad = 0.15 * max(hi - lo, abs(hi), abs(lo), 1e-6)
    lo, hi = min(0, lo - pad), max(0, hi + pad)
    axes_row[0].set_ylim(lo, hi + headroom * (hi - lo))


def _flops_footer(axes_row, flops_rows: list[list[float | None]], ns) -> None:
    """The FLOPs footer: one number per (model, N), so architecture cost reads at a
    glance.

    The footer carries bare text, so each of its axes is stripped of bars, ticks and
    frame.

    :param flops_rows: one list of values per model, in the order of ``ns``.
    """
    for ax, flops in zip(axes_row, flops_rows):
        for k, value in enumerate(flops):
            text, where = _fmt_flops(value), ax.get_xaxis_transform()
            ax.text(
                k,
                0.5,
                text,
                transform=where,
                ha="center",
                va="center",
                fontsize=10,
                bbox=BOX,
            )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(ns) - 0.4)
        ax.grid(False)
        ax.spines[:].set_visible(False)
    axes_row[0].set_ylabel("FLOPs")


def _legend(fig, ncol: int) -> None:
    """The group swatches and the recover-rate band, centred under the panels."""
    handles = [Patch(facecolor=COL[g], edgecolor="none", label=g) for g in GROUPS]
    handles.append(Line2D([0], [0], color=RECOVER_COLOR, lw=5, label="recover rate"))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=1.5,
        columnspacing=1.8,
        handletextpad=0.6,
        fontsize=13,
    )


def _save(fig, out_dir: Path, name: str, missing: list[tuple]) -> None:
    """Write the figure as ``<name>.svg`` and report the cells that had no row."""
    fig.savefig(out_dir / f"{name}.svg", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    if not missing:
        return
    print(
        f"[warn] {name}: {len(missing)} combos absent from the results tables, left blank:"
    )
    for key in missing:
        print(f"        {key}")


def _row_specs(inv_metric: str) -> list[tuple[str, str, str]]:
    """(entry key, row label, y-axis label) for each bar row, top to bottom."""
    name = metric_display(inv_metric)
    return [
        ("acc", "Accuracy (clean)", "$\\Delta$ accuracy, clean (pp)"),
        ("acc_t", "Accuracy (transformed)", "$\\Delta$ accuracy, transformed (pp)"),
        ("inv", name, f"$\\Delta$ {name}"),
        ("flip", "Self-consistency", "$\\Delta$ flip rate (pp), $+$ = fewer flips"),
    ]


def _series(index, regime: str, model: str, ns, group: str, key: str, missing):
    """Bar positions, heights, deviations and recover-rate bands for one group of one
    panel.

    Every (regime, model, N, group) with no row, or no value under the key, is appended
    to ``missing`` and left out of the panel. The recover rate is a different
    conditional from the flip rate, so only that row is banded with it.
    """
    found = []
    for k, n in enumerate(ns):
        entry = index.get((regime, model, n, group)) or {}
        if entry.get(key) is None:
            missing.append((regime, model, n, group, key))
            continue
        recover = entry.get("recover")
        found.append((k + OFFSET[group], *entry[key], recover[0] if recover else None))
    xs, ys, es, recs = zip(*found) if found else ((), (), (), ())
    bands = [(x, r) for x, r in zip(xs, recs) if r is not None] if key == "flip" else []
    return xs, ys, es, bands


def _draw_rows(axes, specs, index, regime: str, models, ns) -> list[tuple]:
    """The bar rows of one regime figure; returns the cells that had no row to draw."""
    missing: list[tuple] = []
    for i, (key, _, ylabel) in enumerate(specs):
        extent: list[float] = []
        for ax, model in zip(axes[i], models):
            for group in GROUPS:
                xs, ys, es, bands = _series(
                    index, regime, model, ns, group, key, missing
                )
                extent += _draw_group(ax, xs, ys, es, COL[group], bands)
        _finish_row(axes[i], ylabel, extent, ns, last=i == len(specs) - 1)
    return missing


def _flops_of(index, regime: str, model: str, n: int) -> float | None:
    """FLOPs are a property of the architecture, so they are read off the unaugmented
    entry."""
    entry = index.get((regime, model, n, "None"))
    return entry.get("flops") if entry else None


def fig_regime(index, models, ns, regime: str, out_dir: Path, inv_metric: str) -> None:
    """Draw one regime's figure and write it as ``fig_<stem>.svg`` under ``out_dir``."""
    specs = _row_specs(inv_metric)
    fig, axes = scaffold([label for _, label, _ in specs], models)
    missing = _draw_rows(axes, specs, index, regime, models, ns)
    flops = [[_flops_of(index, regime, model, n) for n in ns] for model in models]
    _flops_footer(axes[len(specs)], flops, ns)
    _legend(fig, ncol=len(GROUPS) + 1)
    fig.tight_layout(rect=[0.0, 0.035, 1, 0.975], h_pad=0.4)
    _save(fig, out_dir, f"fig_{REGIME_FILE_STEM[regime]}", missing)


def _parse_args() -> argparse.Namespace:
    """Command line: which results directory to read and where the figures go."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Root directory holding the results/<features>/ directories (default: results/).",
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="The one results/<features>/ directory to read, e.g. px_py_pz_e. Required: results/ "
        "holds one per input feature list and their rows are not comparable, so it is named "
        "rather than guessed. The figures go there unless --out-dir says otherwise.",
    )
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=Path,
        required=True,
        help="The deltas tables written by compare_to_baseline.py, one per regime. Required: "
        "every bar of these figures is a delta, and which run is the reference is chosen there "
        "rather than here.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write fig_augmentation.svg and fig_symm_loss.svg "
        "(default: results/<features>/).",
    )
    return parser.parse_args()


def _require_rows(regimes: list[str], models: list[str], ns: list[int], args) -> None:
    """Exit with what to run when the tables hold nothing to draw."""
    if not regimes:
        sys.exit(
            f"No augmentation or symm_loss results table under {args.results_dir}/"
            f"{args.features}/. Run the evaluation catalogues first."
        )
    if not models or not ns:
        sys.exit(
            "No usable rows found (empty model or nconstituents set). Check the tables."
        )


def main() -> None:
    """Draw one figure per regime the results directory holds a table for."""
    args = _parse_args()
    rows = read_deltas(args.deltas)
    if not rows:
        sys.exit(
            f"No delta rows under {[str(p) for p in args.deltas]}. Write them first, e.g.:\n"
            "    python3 scripts/analysis/compare_to_baseline.py "
            "results/<features>/results_table_augmentation.csv "
            "results/<features>/results_table_baseline.csv"
        )
    deltas = index_deltas(rows)
    index, models, ns, regimes, inv_metric = build_dataset(
        args.results_dir, args.features, deltas
    )
    _require_rows(regimes, models, ns, args)
    out_dir = args.out_dir or args.results_dir / args.features
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[dataset] features={args.features} regimes={regimes} models={models} "
        f"n_constituents={ns} inv_metric={inv_metric}"
    )
    for regime in regimes:
        fig_regime(index, models, ns, regime, out_dir, inv_metric)
    names = ", ".join(f"fig_{REGIME_FILE_STEM[regime]}" for regime in regimes)
    print(f"\ndone: {names} (svg) -> {out_dir}")


if __name__ == "__main__":
    main()
