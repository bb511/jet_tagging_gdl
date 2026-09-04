#!/usr/bin/env python3
"""Render the baseline-only counterpart of make_figures.py, in absolute terms.

make_figures.py draws every regime as a delta against the unaugmented run of the same model, so a
model trained in the baseline regime alone (LorentzNet, the embedded class) never appears in it.
This figure keeps that grid, one column per model and one bar cluster per constituent count with
SO(2), SO(3) and Lorentz bars, and measures the model against its own clean prediction::

    accuracy (transformed)  pp, transformed minus clean, paired per fold and reconstructed from
                            the flip breakdown; the clean accuracy is printed on the panel
    <invariance metric>     the results table's own value (Wasserstein distance), absolute
    self-consistency        flip rate 1 - stable fraction (%), the recover rate overlaid as a band
    FLOPs                   a text footer, one number per (model, N)

Lower is better in the last two rows, unlike the delta figures where positive is always better.

Reads results_table_*.csv files, keeps the augmentation_scheme=none rows and writes
<out-dir>/<name>.svg::

    python3 scripts/analysis/make_baseline_figure.py \
        results/lorentz_net_clamp/nconst*/final_5fold/results_table_baseline.csv \
        --out-dir results/lorentz_net_clamp --name fig_lorentznet_5fold
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    """Import a sibling script by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# make_figures.py applies the thesis rcParams on import, loads make_latex_table.py itself, and owns
# the drawing helpers and the palette both figures share.
_figs = _load("make_figures")
_tex = _figs._make_latex_table
COL, GROUPS, OFFSET = _figs.COL, _figs.GROUPS, _figs.OFFSET


def _folds(row: dict[str, str], key: str) -> list[float] | None:
    return _figs._parse_folds(row.get(key, ""))


def _acc_after_transform_folds(
    acc: list[float] | None, stable: list[float] | None, recover: list[float] | None
) -> list[float] | None:
    """Per-fold accuracy on the transformed test set, by total probability over correct-before::

        acc_after = acc_before * stable + (1 - acc_before) * recover

    All three are proportions, the units the results table's ``*_folds`` columns are written in.
    ``stable`` and ``recover`` are the model's own P(correct after | correct before) and
    P(correct after | wrong before) for one group.
    """
    if acc is None or stable is None or recover is None:
        return None
    if not len(acc) == len(stable) == len(recover):
        return None
    return [a * s + (1.0 - a) * r for a, s, r in zip(acc, stable, recover)]


def _group_entry(row: dict[str, str], acc: list[float], scheme: str, metric: str) -> dict | None:
    """One group's three panels for one row, each as (mean, deviation) over the folds."""
    stable = _folds(row, f"{scheme}_flip_stable_folds")
    recover = _folds(row, f"{scheme}_flip_recover_folds")
    distance = _folds(row, f"{scheme}_{metric}_folds")
    acc_t = _acc_after_transform_folds(acc, stable, recover)
    if acc_t is None or distance is None:
        return None
    return {
        "dacc": _figs._stats([100.0 * (t - a) for t, a in zip(acc_t, acc)]),
        "w": _figs._stats(distance),
        "flip": _figs._stats([100.0 * (1.0 - s) for s in stable]),
        "recover": _figs._stats(recover)[0] * 100.0,
    }


def _row_entry(row: dict[str, str], metric: str) -> dict | None:
    """One results row as the entry a panel reads, or None when it carries no per-fold accuracy."""
    acc = _folds(row, "test_acc_folds")
    if acc is None:
        return None
    entry = {
        "acc": _figs._stats([100.0 * a for a in acc]),
        "flops": _figs._opt_float(row, "flops_mean"),
    }
    for group, scheme in GROUPS.items():
        entry[group] = _group_entry(row, acc, scheme, metric)
    return entry


def _read_rows(paths: list[Path]) -> list[dict[str, str]]:
    """The augmentation_scheme=none rows of every table named on the command line."""
    rows = []
    for path in paths:
        found = _tex.read_csv(path)
        rows += [r for r in found if r.get("augmentation_scheme", "").lower() == "none"]
    return rows


def _model_rank(model: str) -> tuple[int, str]:
    """MODEL_ORDER first, then whatever else the tables held, alphabetically."""
    order = _figs.MODEL_ORDER
    return (order.index(model) if model in order else len(order), model)


def build_dataset(rows: list[dict[str, str]], metric: str):
    """Index the rows as (model, N) -> entry.

    :returns: the index, the models it covers and the constituent counts, both in figure order.
    """
    data = {}
    for row in rows:
        entry = _row_entry(row, metric)
        if entry is not None:
            data[(_tex._model_display(row["model"]), int(row["nconstituents"]))] = entry
    models = sorted({m for m, _ in data}, key=_model_rank)
    return data, models, sorted({n for _, n in data})


def _row_specs(metric: str) -> list[tuple[str, str, str]]:
    """(entry key, row label, y-axis label) for each bar row, top to bottom."""
    name = _tex._metric_display(metric)
    return [
        ("dacc", "Accuracy (transformed)", "$\\Delta$ accuracy (pp)"),
        ("w", name, name),
        ("flip", "Self-consistency", "flip rate $\\mathrm{FR}$ (%)"),
    ]


def _annotate_clean(ax, entries_by_n) -> None:
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
            bbox=_figs.BOX,
            zorder=5,
        )


def _group_values(entries_by_n, group: str, key: str) -> list[tuple[int, float, float]]:
    """(constituent-count index, mean, deviation) for the entries that hold this group."""
    return [(k, *e[group][key]) for k, e in enumerate(entries_by_n) if e and e.get(group)]


def _panel(ax, entries_by_n, key: str) -> list[float]:
    """One (row, model) panel: a bar cluster per group, recover bands on the flip-rate row."""
    extent: list[float] = []
    for group in GROUPS:
        vals = _group_values(entries_by_n, group, key)
        xs = [k + OFFSET[group] for k, _, _ in vals]
        ys, es = [m for _, m, _ in vals], [s for _, _, s in vals]
        bands = [(x, entries_by_n[k][group]["recover"]) for x, (k, _, _) in zip(xs, vals)]
        extent += _figs._draw_group(ax, xs, ys, es, COL[group], bands if key == "flip" else [])
    return extent


def _draw_rows(axes, specs, data, models, ns) -> None:
    """The bar rows, each scaled to its own extent; the clean accuracy annotates the first."""
    for i, (key, _, ylabel) in enumerate(specs):
        extent: list[float] = []
        for j, model in enumerate(models):
            entries = [data.get((model, n)) for n in ns]
            extent += _panel(axes[i, j], entries, key)
            if key == "dacc":
                _annotate_clean(axes[i, j], entries)
        # The clean-accuracy annotation sits at the top of its panel, so that row is given room.
        headroom = 0.3 if key == "dacc" else 0.0
        _figs._finish_row(axes[i], ylabel, extent, ns, i == len(specs) - 1, headroom)


def draw(data, models, ns, metric: str, out_dir: Path, name: str) -> None:
    """Draw the figure and write it as ``<name>.svg`` under ``out_dir``."""
    specs = _row_specs(metric)
    fig, axes = _figs.scaffold([label for _, label, _ in specs], models)
    _draw_rows(axes, specs, data, models, ns)
    flops = [[(data.get((model, n)) or {}).get("flops") for n in ns] for model in models]
    _figs._flops_footer(axes[len(specs)], flops, ns)
    # Four legend entries do not fit on one row of a single-column figure.
    legend_rows = 1 if len(models) > 1 else 2
    _figs._legend(fig, ncol=4 // legend_rows)
    fig.tight_layout(rect=[0.0, 0.035 * legend_rows, 1, 0.975], h_pad=0.4)
    _figs._save(fig, out_dir, name, [])


def _parse_args() -> argparse.Namespace:
    """Command line: the tables to read, where the figure goes and what it is called."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tables", nargs="+", type=Path, help="raw results_table_*.csv files")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="default: the first table's directory"
    )
    parser.add_argument(
        "--name", default="fig_baseline", help="output stem, written as <name>.svg"
    )
    return parser.parse_args()


def main() -> None:
    """Draw the baseline figure from the tables named on the command line."""
    args = _parse_args()
    rows = _read_rows(args.tables)
    metric = _tex._detect_inv_metric_suffix(rows)
    if not rows or metric is None:
        sys.exit("no augmentation_scheme=none rows with per-fold invariance columns found")
    data, models, ns = build_dataset(rows, metric)
    out_dir = args.out_dir or args.tables[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)
    draw(data, models, ns, metric, out_dir, args.name)
    print(
        f"done: {out_dir / args.name}.svg models={models} n_constituents={ns} inv_metric={metric}"
    )


if __name__ == "__main__":
    main()
