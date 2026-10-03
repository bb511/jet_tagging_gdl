"""What every analysis script needs: reading the tables, and naming things for a reader.

The scripts in this directory are run as scripts, so Python puts this directory first on
the import path and a plain ``import common`` finds this file. The bootstrap below adds
the repository root as well, for ``src.symmetry``, and re-adds this directory for the
case where a script is loaded by path rather than run, as the tests do.

Nothing heavy is imported here or by anything it pulls in, so a bare interpreter with
matplotlib alone still runs the figure scripts.
"""

from __future__ import annotations

import csv
import math
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.symmetry import FLIP_FAMILIES, TABULATED_GROUPS

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
    "vn": "Vector neurons",
    "jedi_linear": "JEDI-linear",
    "linformer": "Linformer",
    "transformer": "JetFormer",
}

METRIC_DISPLAY = {
    "wasserstein": "Wasserstein",
    "flip": "Flip rate",
    "accuracy_transformed": "Accuracy (transformed)",
}

# Sentinels a cell carries in place of a number: the tables write "nan" where no fold reported the
# quantity, and the tables written before this one carried the other words.
NO_VALUE = {"", "nan", "n/a", "baseline", "no_baseline", "no_folds", "—"}


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


def parse_folds(raw, keep_nan: bool = False) -> list[float] | None:
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


def safe_fmt(value, digits: int) -> str:
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


def read_deltas(paths: list[Path]) -> list[dict[str, str]]:
    """Every deltas row the given tables hold, reporting what was read and what was
    missing.

    The caller decides what an empty result means: a figure of deltas cannot be drawn at all,
    whereas a LaTeX table prints dashes in those columns.
    """
    rows: list[dict[str, str]] = []
    for path in paths:
        if not path.exists():
            print(f"[skip] {path} is missing")
            continue
        found = read_csv(path)
        print(f"[load] {len(found):3d} deltas <- {shorten(path)}")
        rows += found
    return rows


def shorten(path: Path) -> Path:
    """A path as the logs print it, relative to the repository where that is shorter."""
    return path.relative_to(REPO_ROOT) if REPO_ROOT in path.parents else path


def detect_inv_metric_suffix(rows: list[dict[str, str]]) -> str | None:
    """Infer the invariance metric (``wasserstein``, ``flip``) from a
    ``so2_<metric>_folds`` column.

    None when every per-group column is a flip family, which is what a table written
    with ``invariance_metric=flip`` looks like: the family is then written once, not
    twice.
    """
    for row in rows:
        for column in row:
            if column.startswith("so2_") and column.endswith("_folds"):
                metric = column[len("so2_") : -len("_folds")]
                if metric not in FLIP_FAMILIES:
                    return metric
    return None


def model_display(model_raw: str) -> str:
    """Map a short algorithm name or a full class path to the label the tables print."""
    key = model_raw.lower().split(".")[-1]
    if key in MODEL_DISPLAY:
        return MODEL_DISPLAY[key]
    for name, label in MODEL_DISPLAY.items():
        if name in key:
            return label
    return model_raw.split(".")[-1]


def metric_display(metric: str | None) -> str:
    """The column heading for an invariance metric suffix."""
    return METRIC_DISPLAY.get(metric, metric.capitalize()) if metric else "Metric"
