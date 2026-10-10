"""What every analysis script needs: reading the tables, and naming things for a reader.

The scripts in this directory are run as scripts, so Python puts this directory first on
the import path and a plain ``import common`` finds this file; the tests, which load a
script by path, put it there themselves. The bootstrap below adds the repository root as
well, for ``src.symmetry``.

Nothing heavy is imported here or by anything it pulls in, so a bare interpreter with
matplotlib alone still runs the figure scripts.
"""

import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.symmetry import (
    ACCURACY_TRANSFORMED,
    FLIP,
    TABULATED_FAMILIES,
    TABULATED_GROUPS,
    TOTAL_VARIATION,
)

# augmentation_scheme -> the label the figures and tables print for it. The labels are
# presentation; which groups are drawn, and in what order, is TABULATED_GROUPS, so a group the
# study adds raises here rather than going missing.
GROUP_LABEL = {"so2": "SO(2)", "so3": "SO(3)", "lorentz": "Lorentz"}
AUG_DISPLAY = {
    "none": "None",
    **{group: GROUP_LABEL[group] for group in TABULATED_GROUPS},
}
AUG_ORDER = {
    "none": 0,
    **{group: i + 1 for i, group in enumerate(("lorentz", "so2", "so3"))},
}

# vn_hgq precedes vn, and jedi_linear_hgq and jedi_linear_moments precede jedi_linear: a
# name that is no key, such as vn_hgq_beta<tag>, takes the label of the first key it
# contains.
MODEL_DISPLAY = {
    "deepsets": "DeepSets",
    "mlp": "MLP",
    "jetformer": "JetFormer",
    "lorentznet": "LorentzNet",
    "pelican": "PELICAN",
    "pelican_lorentz": "PELICAN (Lorentz)",
    "lgatr_slim": "L-GATr-slim",
    "lgatr_slim_lorentz": "L-GATr-slim (Lorentz)",
    "lloca_lorentz": "LLoCa (Lorentz)",
    "lloca_z": "LLoCa (SO(2)xSO(1,1))",
    "lloca_so2": "LLoCa (SO(2))",
    "lloca_so3": "LLoCa (SO(3))",
    "vn_hgq": "Vector neurons (HGQ2)",
    "vn": "Vector neurons",
    "jedi_linear_hgq": "JEDI-linear (HGQ2)",
    "jedi_linear_moments": "JEDI-linear + moments",
    "jedi_linear": "JEDI-linear",
    "linformer": "Linformer",
    "deepsets_moments": "DeepSets + moments",
    "moments_only": "Moments only",
    "jet_level_all53": "Jet observables (53)",
    "jet_level_noeta52": "Jet observables w/o eta (52)",
    "jet_level_bdt_all53": "Jet observables BDT (53)",
    "jet_level_bdt_noeta52": "Jet observables BDT w/o eta (52)",
}

METRIC_DISPLAY = {
    TOTAL_VARIATION: "Total variation",
    FLIP: "Flip rate",
    ACCURACY_TRANSFORMED: "Accuracy (transformed)",
}

# What a cell carries in place of a number: the tables write "nan" where no fold
# reported the quantity.
NO_VALUE = {"", "nan"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a results table into one dict per row, keyed by column name.

    Short rows are padded rather than truncated, so a table whose header grew between
    runs still yields every column the header names.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = [row for row in reader if any(cell.strip() for cell in row)]
    return [dict(zip(header, row + [""] * (len(header) - len(row)))) for row in rows]


def parse_folds(raw: str | None, keep_nan: bool = False) -> list[float] | None:
    """Parse a semicolon-separated per-fold column; None if it is empty or unparsable.

    :param keep_nan: keep the folds that reported nothing, as NaN. A figure needs a
        complete series to draw a bar from and so takes the default; a paired comparison
        keeps them, so that it can drop the pairs it cannot form and still use the folds
        that did report.
    """
    if not raw:
        return None
    try:
        values = [float(v) for v in str(raw).strip().split(";") if v != ""]
    except ValueError:
        return None
    if not values:
        return None
    return (
        values if keep_nan else (None if any(math.isnan(v) for v in values) else values)
    )


def stats(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation; the deviation is zero for a single value."""
    return sum(values) / len(values), (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )


def opt_float(row: dict[str, str], key: str) -> float | None:
    """A numeric column, or None when it is absent, blank or NaN."""
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def safe_fmt(value: str | float | None, digits: int) -> str:
    """Fixed-point format, ``N/A`` for anything that is not a number."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if math.isnan(number) else f"{number:.{digits}f}"


def delta_key(row: dict, group: str, metric: str) -> tuple[str, str, str, str, str]:
    """The key one deltas row is filed under, read off a row of either table."""
    return (
        str(row.get("method", "")),
        str(row.get("model", "")),
        str(row.get("nconstituents", "")).strip(),
        group,
        metric,
    )


def index_deltas(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str, str], dict]:
    """Index the rows of the deltas tables ``compare_to_baseline.py`` writes.

    The key carries the method and the group, so the augmentation and the SCL tables,
    which cover the same models, counts and groups, can be named on one command line
    without colliding.
    """
    return {
        delta_key(row, row.get("group", ""), row.get("metric", "")): row for row in rows
    }


def read_tables(paths: list[Path], noun: str) -> list[dict[str, str]]:
    """Every row the given tables hold, reporting what was read, counted in *noun*, and
    what was missing.

    The caller decides what an empty result means: a figure of deltas cannot be drawn at all,
    whereas a LaTeX table prints dashes in those columns.
    """
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            print(f"[skip] {path} is missing")
            continue
        found = read_csv(path)
        print(f"[load] {len(found):3d} {noun} <- {shorten(path)}")
        rows += found
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], header: list[str]) -> None:
    """Rewrite *path* from the rows: unlike a results table, it is never appended to."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def shorten(path: Path) -> Path:
    """A path as the logs print it, relative to the repository where that is shorter."""
    return path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path


def detect_inv_metric_suffix(rows: list[dict[str, str]]) -> str | None:
    """Infer the invariance metric (``tv``, ``flip``) from a ``so2_<metric>_folds``
    column.

    None when every per-group column is a tabulated family, which is what a table
    written with ``invariance_metric=flip`` looks like: the family is then written once,
    not twice.
    """
    for row in rows:
        for column in row:
            if column.startswith("so2_") and column.endswith("_folds"):
                metric = column[len("so2_") : -len("_folds")]
                if metric not in TABULATED_FAMILIES:
                    return metric
    return None


def scheme_of(row: dict[str, str]) -> str:
    """The augmentation scheme of a results row, lowercased."""
    return str(row.get("augmentation_scheme", "")).lower()


def model_display(model: str) -> str:
    """The label the tables print for an algorithm name; a name that is no key takes
    that of the first key it contains."""
    if model in MODEL_DISPLAY:
        return MODEL_DISPLAY[model]
    return next(
        (label for name, label in MODEL_DISPLAY.items() if name in model), model
    )


def metric_display(metric: str | None) -> str:
    """The column heading for an invariance metric suffix."""
    return METRIC_DISPLAY.get(metric, metric.capitalize()) if metric else "Metric"
