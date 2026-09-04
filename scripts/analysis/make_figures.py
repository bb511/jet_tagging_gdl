#!/usr/bin/env python3
"""Draw the per-regime comparison figures from the results table CSVs.

One figure per training regime (augmentation, SCL), a grid of one bar row per metric by one column
per architecture. Every bar is a delta against the unaugmented baseline of the same model and
constituent count, so positive is better in every row::

    accuracy (clean)        pp, group - baseline
    accuracy (transformed)  pp, from delta_accuracy_transformed_*
    <invariance metric>     from delta_<metric>_*, e.g. the Wasserstein distance between the
                            clean and the transformed logits
    self-consistency        stable fraction 1 - FR, the recover rate overlaid as a band
    FLOPs                   a text footer, one regime-independent number per (model, N)

Bars cover SO(2), SO(3) and Lorentz; the unaugmented run is the zero line by construction.

``--features`` names the one ``results/<features>/`` directory to read, since the directories hold
different input feature lists and their rows are not comparable. The delta columns are written at
evaluation time, so a row carries them only if its eval run was given a
``baseline_checkpoint_manifest_path``::

    python3 scripts/analysis/make_figures.py --features px_py_pz_e

The accuracy and stable-fraction deltas are paired per fold, the unit every ``delta_<metric>``
column of the results table averages over: KFold splits on a fixed seed, so fold i is the same
data in every condition.

The figures are SVG rather than PDF because matplotlib's PDF backend emits mathtext as Type 3
fonts, which PDF.js and the VSCode viewers built on it garble.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]

# make_latex_table.py is loaded by path, not imported: scripts/ is not a package, so there is no
# module name to import it under, and anchoring on the sibling file keeps the two in step wherever
# the directory is copied to.
_SPEC = importlib.util.spec_from_file_location(
    "make_latex_table", Path(__file__).resolve().parent / "make_latex_table.py"
)
_make_latex_table = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_make_latex_table)
_read_csv = _make_latex_table.read_csv
_metric_display = _make_latex_table._metric_display
_model_display = _make_latex_table._model_display
_detect_inv_metric = _make_latex_table._detect_inv_metric_suffix

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
COL = {
    "None": "#004D40",
    "SO(2)": "#1E88E5",
    "SO(3)": "#FFC107",
    "Lorentz": "#D81B60",
}
RECOVER_COLOR = "#1a1a1a"

# Display label -> augmentation_scheme, the three groups a figure draws bars for.
GROUPS = {"SO(2)": "so2", "SO(3)": "so3", "Lorentz": "lorentz"}
AUG_DISPLAY = {"none": "None", "so2": "SO(2)", "so3": "SO(3)", "lorentz": "Lorentz"}

WIDTH = 0.26
OFFSET = {"SO(2)": -WIDTH, "SO(3)": 0.0, "Lorentz": WIDTH}

# The FLOPs footer carries one line of text and no bars. At the height of a bar row it would sit
# almost empty and push the legend far below the panels.
FLOPS_RATIO = 0.16
BOX = dict(boxstyle="round,pad=0.25", facecolor="#f0f0f0", edgecolor="#bbbbbb", linewidth=0.6)

# Preferred column order; a model present in the data but not listed here is appended
# alphabetically, so a re-added architecture (see ARCHITECTURES.md) needs no edit here.
MODEL_ORDER = ["DeepSets", "MLP", "JetFormer", "LorentzNet"]

# Regime label -> results_table_<stem>.csv filename stem.
REGIME_FILE_STEM = {"Augmentation": "augmentation", "SCL": "symm_loss"}


def _opt_float(row: dict[str, str], key: str) -> float | None:
    """A numeric column, or None when it is absent, blank or NaN."""
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _parse_folds(raw) -> list[float] | None:
    """Parse a semicolon-separated per-fold column; None if it is empty or unparsable."""
    if not raw:
        return None
    try:
        values = [float(v) for v in str(raw).strip().split(";") if v != ""]
    except ValueError:
        return None
    return None if not values or any(math.isnan(v) for v in values) else values


def _stats(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; the deviation is zero for a single value."""
    return sum(values) / len(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def _acc_folds_pct(row: dict[str, str]) -> list[float] | None:
    """Per-fold test accuracy in percent, whatever units the CSV used."""
    folds = _parse_folds(row.get("test_acc_folds", ""))
    if folds is None:
        return None
    scale = 100.0 if max(folds) < 2.0 else 1.0
    return [v * scale for v in folds]


def _paired_delta(
    a_folds: list[float] | None, b_folds: list[float] | None
) -> tuple[float | None, float | None]:
    """Mean and deviation of (a - b) over paired per-fold values.

    KFold splits on a fixed seed, so fold i is the same data in both conditions and the per-fold
    difference is the unit to average, as it is for every ``delta_<metric>`` column of the results
    table.
    """
    if a_folds is None or b_folds is None or len(a_folds) != len(b_folds) or not a_folds:
        return None, None
    return _stats([a - b for a, b in zip(a_folds, b_folds)])


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


def _cell(row: dict[str, str]) -> tuple[str, int]:
    """The (model, constituent count) the row belongs to."""
    return _model_display(row.get("model", "")), int(row.get("nconstituents", 0))


def _discover_rows(results_dir: Path, filename_stem: str, features: str) -> list[dict[str, str]]:
    """Read results/<features>/results_table_<stem>.csv, the one directory --features names."""
    rows: list[dict[str, str]] = []
    for csv_path in sorted(results_dir.glob(f"{features}/results_table_{filename_stem}.csv")):
        found = _read_csv(csv_path)
        rows.extend(found)
        shown = csv_path.relative_to(REPO_ROOT) if REPO_ROOT in csv_path.parents else csv_path
        print(f"[load] {len(found):3d} rows <- {shown}")
    return rows


def _group_folds(row: dict[str, str], suffix: str) -> dict[str, list[float]]:
    """The per-group fold columns (``so2_flip_stable_folds`` and the like), keyed by scheme.

    Read per group rather than off the row's own scheme, so an unaugmented row can be paired
    against an SO(2)-trained one on the SO(2) value specifically.
    """
    folds = {g: _parse_folds(row.get(f"{g}_{suffix}", "")) for g in GROUPS.values()}
    return {g: v for g, v in folds.items() if v is not None}


def _delta_pp(row: dict[str, str], key: str, trained: bool) -> float | None:
    """A delta column in percentage points; the callback writes accuracies as proportions."""
    value = _opt_float(row, key) if trained else None
    return None if value is None else value * 100.0


def _extract_entry(row: dict[str, str], inv_metric: str) -> dict[str, float | None]:
    """One results row as the entry a panel reads.

    The ``dAcc``, ``dStab`` and ``dRec`` keys stay empty here: they are paired against the row's
    own baseline, which is known only once every row is indexed (see :func:`_fill_deltas`).
    """
    trained = _scheme_of(row) in set(GROUPS.values())
    return {
        "accfolds": _acc_folds_pct(row),
        "stabfolds_by_group": _group_folds(row, "flip_stable_folds"),
        "recfolds_by_group": _group_folds(row, "flip_recover_folds"),
        "flops": _opt_float(row, "flops_mean"),
        "dW": _opt_float(row, f"delta_{inv_metric}_mean") if trained else None,
        "dWsd": _opt_float(row, f"delta_{inv_metric}_std") if trained else None,
        "dAccTrans": _delta_pp(row, "delta_accuracy_transformed_mean", trained),
        "dAccTransSd": _delta_pp(row, "delta_accuracy_transformed_std", trained),
        "dAcc": None,
        "dAccSd": None,
        "dStab": None,
        "dStabSd": None,
        "dRec": None,
        "dRecSd": None,
    }


def _baseline_rows(results_dir: Path, features: str, regime_rows: dict) -> list[dict[str, str]]:
    """The unaugmented rows, from their own table or from the regime tables.

    A regime catalogue evaluated on its own writes its ``none`` row into that regime's table, so
    the rows can be present while results_table_baseline.csv is not.
    """
    rows = _discover_rows(results_dir, "baseline", features)
    if rows:
        return rows
    return [r for found in regime_rows.values() for r in found if _scheme_of(r) == "none"]


def _index_regime(D: dict, regime: str, rows: list[dict], baselines: list[dict], metric: str):
    """Index one regime's rows, and the unaugmented rows again under that regime.

    Every bar is a delta against the unaugmented run of the same model and constituent count, so
    each regime needs its own copy of that row to be measured against.
    """
    for row in baselines:
        if _scheme_of(row) == "none":
            D[(regime, *_cell(row), "None")] = _extract_entry(row, metric)
    for row in rows:
        group = AUG_DISPLAY.get(_scheme_of(row), "None")
        if group != "None":
            D[(regime, *_cell(row), group)] = _extract_entry(row, metric)


def _index_rows(regime_rows: dict, baselines: list[dict], metric: str) -> dict:
    """(regime, model, N, group) -> entry, over the regimes that have a table."""
    D: dict[tuple[str, str, int, str], dict[str, float | None]] = {}
    for regime, rows in regime_rows.items():
        if rows:
            _index_regime(D, regime, rows, baselines, metric)
    return D


def _fill_deltas(D: dict) -> None:
    """Fill each transformed group's delta keys against the unaugmented entry of the same cell."""
    for (regime, model, n, group), entry in D.items():
        base = D.get((regime, model, n, "None"))
        if group == "None" or base is None:
            continue
        scheme = GROUPS[group]
        entry["dAcc"], entry["dAccSd"] = _paired_delta(entry["accfolds"], base["accfolds"])
        entry["dStab"], entry["dStabSd"] = _paired_delta(
            entry["stabfolds_by_group"].get(scheme), base["stabfolds_by_group"].get(scheme)
        )
        entry["dRec"], entry["dRecSd"] = _paired_delta(
            entry["recfolds_by_group"].get(scheme), base["recfolds_by_group"].get(scheme)
        )


def _ordered_models(D: dict) -> list[str]:
    """MODEL_ORDER first, then whatever else the tables held, alphabetically."""
    seen = {model for _, model, _, _ in D}
    return [m for m in MODEL_ORDER if m in seen] + sorted(seen - set(MODEL_ORDER))


def build_dataset(results_dir: Path, features: str):
    """Read one results directory and index it for the figures.

    :param features: the ``results/<features>/`` directory name.
    :returns: the (regime, model, N, group) index, the models and constituent counts it covers, the
        regimes that have a table, and the invariance metric those tables carry.
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
    every_row = baselines + [r for rows in regime_rows.values() for r in rows]
    metric = _detect_inv_metric(every_row) or "wasserstein"
    D = _index_rows(regime_rows, baselines, metric)
    _fill_deltas(D)
    regimes = [r for r in REGIME_FILE_STEM if regime_rows[r]]
    return D, _ordered_models(D), sorted({n for _, _, n, _ in D}), regimes, metric


def scaffold(row_labels: list[str], models: list[str]):
    """The grid both figures use: one bar row per label, a FLOPs footer under them.

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
    for ax, model in zip(axes[0], models):
        ax.set_title(model, fontweight="bold", pad=8)
    _style_panels(axes)
    return fig, axes


def _style_panels(axes) -> None:
    """No top or right spine, and a faint horizontal grid behind the bars."""
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#000000", alpha=0.06, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=3)


def _band(ax, x: float, y: float) -> None:
    """Overlay the recover rate, P(correct after | wrong before).

    A different conditional from the stable fraction the bar carries, so it is drawn as a band and
    not as a tick on the bar.
    """
    band = ax.hlines(
        y,
        x - WIDTH * 0.42,
        x + WIDTH * 0.42,
        colors=RECOVER_COLOR,
        linewidth=6.0,
        capstyle="butt",
        zorder=4,
        alpha=0.9,
    )
    band.set_path_effects([patheffects.withStroke(linewidth=8.0, foreground="white")])


def _draw_group(ax, xs: list[float], ys: list[float], es: list[float], colour, bands=()):
    """One group's bar cluster, its error bars and any recover-rate bands.

    :param bands: (x, y) points to band, empty on a row that carries no recover rate.
    :returns: the vertical extent drawn, whiskers and bands included, so a row's limits do not clip
        them.
    """
    if not xs:
        return []
    ax.bar(xs, ys, width=WIDTH, color=colour, edgecolor="white", lw=0.5, zorder=2)
    ax.errorbar(
        xs,
        ys,
        yerr=es,
        fmt="none",
        ecolor="#333",
        elinewidth=0.9,
        capsize=2.2,
        capthick=0.9,
        zorder=3,
    )
    for x, y in bands:
        _band(ax, x, y)
    return (
        [y + e for y, e in zip(ys, es)] + [y - e for y, e in zip(ys, es)] + [y for _, y in bands]
    )


def _set_ylim(ax, extent: list[float], headroom: float = 0.0) -> None:
    """Fit the limits to everything drawn, keeping zero in view: the rows diverge about it.

    :param headroom: extra fraction of the span left above, for a row carrying an annotation.
    """
    if not extent:
        return
    lo, hi = min(extent), max(extent)
    pad = 0.15 * max(hi - lo, abs(hi), abs(lo), 1e-6)
    lo, hi = min(0, lo - pad), max(0, hi + pad)
    ax.set_ylim(lo, hi + headroom * (hi - lo))


def _finish_row(axes_row, ylabel: str, extent: list[float], ns, last: bool, headroom=0.0) -> None:
    """Label and scale one row of panels; sharey='row' spreads the limits across it."""
    for ax in axes_row:
        ax.axhline(0, color="#444444", lw=0.8, zorder=1)
        ax.set_xticks(list(range(len(ns))))
        ax.set_xticklabels(ns)
        ax.set_xlim(-0.6, len(ns) - 0.4)
        if last:
            ax.set_xlabel("constituents $N$")
    axes_row[0].set_ylabel(ylabel)
    _set_ylim(axes_row[0], extent, headroom)


def _blank_axis(ax, n: int) -> None:
    """Strip an axis to bare text: the FLOPs footer carries no bars, ticks or frame."""
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(-0.6, n - 0.4)
    ax.grid(False)
    ax.spines[:].set_visible(False)


def _flops_footer(axes_row, flops_rows: list[list[float | None]], ns) -> None:
    """The FLOPs footer: one number per (model, N), so architecture cost reads at a glance.

    :param flops_rows: one list of values per model, in the order of ``ns``.
    """
    for ax, flops in zip(axes_row, flops_rows):
        for k, value in enumerate(flops):
            text, where = _fmt_flops(value), ax.get_xaxis_transform()
            ax.text(k, 0.5, text, transform=where, ha="center", va="center", fontsize=10, bbox=BOX)
        _blank_axis(ax, len(ns))
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
    print(f"[warn] {name}: {len(missing)} combos absent from the results tables, left blank:")
    for key in missing:
        print(f"        {key}")


def _row_specs(inv_metric: str) -> list[tuple[str, str, str, str]]:
    """(entry key, deviation key, row label, y-axis label) for each bar row, top to bottom."""
    return [
        ("dAcc", "dAccSd", "Accuracy (clean)", "$\\Delta$ accuracy, clean (pp)"),
        (
            "dAccTrans",
            "dAccTransSd",
            "Accuracy (transformed)",
            "$\\Delta$ accuracy, transformed (pp)",
        ),
        ("dW", "dWsd", _metric_display(inv_metric), f"$\\Delta$ {_metric_display(inv_metric)}"),
        ("dStab", "dStabSd", "Self-consistency", "$\\Delta$ stable fraction $1-\\mathrm{FR}$"),
    ]


def _series(D, regime: str, model: str, ns, group: str, keys: tuple[str, str], missing):
    """Bar positions, heights, deviations and recover rates for one group of one panel.

    Every (regime, model, N, group) with no row, or no value under the key, is appended to
    ``missing`` and left out of the panel.
    """
    key, sdkey = keys
    found = []
    for k, n in enumerate(ns):
        entry = D.get((regime, model, n, group)) or {}
        if entry.get(key) is None:
            missing.append((regime, model, n, group, key))
            continue
        found.append((k + OFFSET[group], entry[key], entry.get(sdkey) or 0.0, entry.get("dRec")))
    return (
        [f[0] for f in found],
        [f[1] for f in found],
        [f[2] for f in found],
        [f[3] for f in found],
    )


def _panel(ax, D, regime: str, model: str, ns, keys: tuple[str, str], missing) -> list[float]:
    """One (row, model) panel: a bar cluster per group, recover bands on the last row."""
    extent: list[float] = []
    for group in GROUPS:
        xs, ys, es, recs = _series(D, regime, model, ns, group, keys, missing)
        bands = [(x, r) for x, r in zip(xs, recs) if r is not None] if keys[0] == "dStab" else []
        extent += _draw_group(ax, xs, ys, es, COL[group], bands)
    return extent


def _draw_rows(axes, specs, D, regime: str, models, ns) -> list[tuple]:
    """The bar rows of one regime figure; returns the cells that had no row to draw."""
    missing: list[tuple] = []
    for i, (key, sdkey, _, ylabel) in enumerate(specs):
        extent: list[float] = []
        for j, model in enumerate(models):
            extent += _panel(axes[i, j], D, regime, model, ns, (key, sdkey), missing)
        _finish_row(axes[i], ylabel, extent, ns, last=i == len(specs) - 1)
    return missing


def _flops_of(D, regime: str, model: str, n: int) -> float | None:
    """FLOPs are a property of the architecture, so they are read off the unaugmented entry."""
    entry = D.get((regime, model, n, "None"))
    return entry.get("flops") if entry else None


def fig_regime(D, models, ns, regime: str, out_dir: Path, inv_metric: str) -> None:
    """Draw one regime's figure and write it as ``fig_<stem>.svg`` under ``out_dir``."""
    specs = _row_specs(inv_metric)
    fig, axes = scaffold([label for _, _, label, _ in specs], models)
    missing = _draw_rows(axes, specs, D, regime, models, ns)
    flops = [[_flops_of(D, regime, model, n) for n in ns] for model in models]
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
        sys.exit("No usable rows found (empty model or nconstituents set). Check the tables.")


def main() -> None:
    """Draw one figure per regime the results directory holds a table for."""
    args = _parse_args()
    D, models, ns, regimes, inv_metric = build_dataset(args.results_dir, args.features)
    _require_rows(regimes, models, ns, args)
    out_dir = args.out_dir or args.results_dir / args.features
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[dataset] features={args.features} regimes={regimes} models={models} "
        f"n_constituents={ns} inv_metric={inv_metric}"
    )
    for regime in regimes:
        fig_regime(D, models, ns, regime, out_dir, inv_metric)
    names = ", ".join(f"fig_{REGIME_FILE_STEM[regime]}" for regime in regimes)
    print(f"\ndone: {names} (svg) -> {out_dir}")


if __name__ == "__main__":
    main()
