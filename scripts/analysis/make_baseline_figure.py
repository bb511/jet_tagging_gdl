#!/usr/bin/env python3
"""Render the baseline-only counterpart of make_figures.py, in absolute terms.

make_figures.py draws every regime as a delta against the unaugmented run of the same model, so a
model trained in the baseline regime alone (LorentzNet, the embedded class) never appears in it.
This figure keeps that grid, one column per model and one bar cluster per constituent count with
SO(2), SO(3) and Lorentz bars, and measures the model against its own clean prediction::

    accuracy (transformed)  pp, transformed minus clean, paired per fold, read off the table's
                            own accuracy_transformed columns; the clean accuracy is printed on
                            the panel
    <invariance metric>     the results table's own value (total variation), absolute
    self-consistency        the unconditional flip rate FR (%), the recover rate overlaid as a
                            band
    FLOPs                   a text footer, one number per (model, N)

Lower is better in the last two rows, unlike the delta figures where positive is always better.

Reads results_table_*.csv files, keeps the augmentation_scheme=none rows and writes
<out-dir>/<name>.svg, by default fig_baseline.svg beside the first table::

    python3 scripts/analysis/make_baseline_figure.py results/embedded/results_table_baseline.csv
"""

import argparse
import sys
from pathlib import Path

import common

# make_figures.py applies the thesis rcParams on import and owns the drawing helpers and the
# palette the two figures share; everything taken from it is spelled make_figures.<name> below.
import make_figures
import numpy as np
from matplotlib.axes import Axes

from src.symmetry import ACCURACY_TRANSFORMED, FLIP, FLIP_RECOVER


def _group_entry(
    row: dict[str, str], acc: list[float], scheme: str, metric: str
) -> dict | None:
    """One group's three panels for one row, each as (mean, deviation) over the folds.

    The accuracy after the transform is the probe's own measurement, not a
    reconstruction from the transition rates, which is what it used to be before the
    table carried the column.
    """
    acc_t = common.parse_folds(row.get(f"{scheme}_{ACCURACY_TRANSFORMED}_folds"))
    flip = common.parse_folds(row.get(f"{scheme}_{FLIP}_folds"))
    recover = common.parse_folds(row.get(f"{scheme}_{FLIP_RECOVER}_folds"))
    distance = common.parse_folds(row.get(f"{scheme}_{metric}_folds"))
    if acc_t is None or flip is None or recover is None or distance is None:
        return None
    if len(acc_t) != len(acc):
        return None
    return {
        "dacc": common.stats([100.0 * (t - a) for t, a in zip(acc_t, acc)]),
        "w": common.stats(distance),
        "flip": common.stats([100.0 * f for f in flip]),
        "recover": common.stats(recover)[0] * 100.0,
    }


def _row_entry(row: dict[str, str], metric: str) -> dict | None:
    """One results row as the entry a panel reads, or None when it carries no per-fold
    accuracy."""
    acc = common.parse_folds(row.get("test_acc_folds"))
    if acc is None:
        return None
    entry = {
        "acc": common.stats([100.0 * a for a in acc]),
        "flops": common.opt_float(row, "flops_mean"),
    }
    for group, scheme in make_figures.GROUPS.items():
        entry[group] = _group_entry(row, acc, scheme, metric)
    return entry


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    """The augmentation_scheme=none rows of every table named on the command line."""
    rows = [row for path in paths for row in common.read_csv(path)]
    return [row for row in rows if common.scheme_of(row) == "none"]


def _model_rank(model: str) -> tuple[int, str]:
    """MODEL_ORDER first, then whatever else the tables held, alphabetically."""
    order = make_figures.MODEL_ORDER
    return (order.index(model) if model in order else len(order), model)


def build_dataset(rows: list[dict[str, str]], metric: str):
    """Index the rows as (model, N) -> entry.

    :returns: the index, the models it covers and the constituent counts, both in figure
        order.
    """
    data = {}
    for row in rows:
        entry = _row_entry(row, metric)
        if entry is not None:
            data[
                (common.model_display(row["model"]), int(row["nconstituents"]))
            ] = entry
    models = sorted({m for m, _ in data}, key=_model_rank)
    return data, models, sorted({n for _, n in data})


def _row_specs(metric: str) -> list[tuple[str, str, str]]:
    """(entry key, row label, y-axis label) for each bar row, top to bottom."""
    name = common.metric_display(metric)
    return [
        ("dacc", "Accuracy (transformed)", "$\\Delta$ accuracy (pp)"),
        ("w", name, name),
        ("flip", "Self-consistency", "flip rate $\\mathrm{FR}$ (%)"),
    ]


def _annotate_clean(ax: Axes, entries_by_n: list[dict | None]) -> None:
    """Print the clean accuracy per N at the top of the delta-accuracy panel."""
    for k, entry in enumerate(entries_by_n):
        if not entry:
            continue
        mean, sd = entry["acc"]
        text = f"{mean:.1f} $\\pm$ {sd:.1f} %" if sd else f"{mean:.1f} %"
        ax.text(
            k,
            0.95,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            bbox=make_figures.BOX,
            zorder=5,
        )


def _panel(ax: Axes, entries_by_n: list[dict | None], key: str) -> list[float]:
    """One (row, model) panel: a bar cluster per group, recover bands on the flip-rate
    row."""
    extent: list[float] = []
    for group in make_figures.GROUPS:
        # (constituent-count index, mean, deviation), for the entries that hold this group.
        vals = [
            (k, *e[group][key])
            for k, e in enumerate(entries_by_n)
            if e and e.get(group)
        ]
        xs = [k + make_figures.OFFSET[group] for k, _, _ in vals]
        ys, es = [m for _, m, _ in vals], [s for _, _, s in vals]
        bands = [
            (x, entries_by_n[k][group]["recover"]) for x, (k, _, _) in zip(xs, vals)
        ]
        extent += make_figures.draw_group(
            ax, xs, ys, es, make_figures.COL[group], bands if key == "flip" else []
        )
    return extent


def _draw_rows(
    axes: np.ndarray,
    specs: list[tuple[str, str, str]],
    data: dict[tuple[str, int], dict],
    models: list[str],
    ns: list[int],
) -> None:
    """The bar rows, each scaled to its own extent; the clean accuracy annotates the
    first."""
    for i, (key, _, ylabel) in enumerate(specs):
        extent: list[float] = []
        for j, model in enumerate(models):
            entries = [data.get((model, n)) for n in ns]
            extent += _panel(axes[i, j], entries, key)
            if key == "dacc":
                _annotate_clean(axes[i, j], entries)
        # The clean-accuracy annotation sits at the top of its panel, so that row is given room.
        headroom = 0.3 if key == "dacc" else 0.0
        make_figures.finish_row(
            axes[i], ylabel, extent, ns, i == len(specs) - 1, headroom
        )


def draw(
    data: dict[tuple[str, int], dict],
    models: list[str],
    ns: list[int],
    metric: str,
    out_dir: Path,
    name: str,
) -> None:
    """Draw the figure and write it as ``<name>.svg`` under ``out_dir``."""
    specs = _row_specs(metric)
    fig, axes = make_figures.scaffold([label for _, label, _ in specs], models)
    _draw_rows(axes, specs, data, models, ns)
    flops = [
        [(data.get((model, n)) or {}).get("flops") for n in ns] for model in models
    ]
    make_figures.flops_footer(axes[len(specs)], flops, ns)
    # Four legend entries do not fit on one row of a single-column figure.
    legend_rows = 1 if len(models) > 1 else 2
    make_figures.legend(fig, ncol=4 // legend_rows)
    fig.tight_layout(rect=[0.0, 0.035 * legend_rows, 1, 0.975], h_pad=0.4)
    make_figures.save(fig, out_dir, name, [])


def _parse_args() -> argparse.Namespace:
    """Command line: the tables to read, where the figure goes and what it is called."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "tables", nargs="+", type=Path, help="raw results_table_*.csv files"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: the first table's directory",
    )
    parser.add_argument(
        "--name", default="fig_baseline", help="output stem, written as <name>.svg"
    )
    return parser.parse_args()


def main() -> None:
    """Draw the baseline figure from the tables named on the command line."""
    args = _parse_args()
    rows = _read_rows(args.tables)
    metric = common.detect_inv_metric_suffix(rows)
    if not rows or metric is None:
        sys.exit(
            "no augmentation_scheme=none rows with per-fold invariance columns found"
        )
    data, models, ns = build_dataset(rows, metric)
    out_dir = args.out_dir or args.tables[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(data, models, ns, metric, out_dir, args.name)
    print(
        f"done: {out_dir / args.name}.svg models={models} n_constituents={ns} inv_metric={metric}"
    )


if __name__ == "__main__":
    main()
