#!/usr/bin/env python3
"""Collect the parameter and FLOP counts of every main training run into one table.

The results tables carry the FLOPs and the eBOPs per jet but no parameter count, which
only the training run's MLflow params hold, as ``algorithm/params/total``. A quantised
row (``vn_hgq``, ``jedi_linear_hgq``), whose weights are Keras variables and whose torch
count is zero, is given the count of the float network it was copied from: its
``algorithm/init_from`` names that network's run directory, or a ``.ckpt`` in its
``checkpoints/``, relative to the root, and the count is that of the run whose main
manifest entry stands there. Its own ``algorithm/keras_params`` is not used, since
Keras counts the quantisers' per-weight bit widths beside the weights. A quantised row
trained from scratch, as the JEDI-linear port is, names no ``init_from``, and its count
is ``nan``. The BDT rows have
no network, so their count is ``nan``.

Each root holds ``logs/train/hls4ml`` and ``logs/mlflow/mlruns``, as a clone of this
repository or ``/shared/deodagiu/jet_tagging_gdl`` does. Every
``checkpoint_manifest.json`` under ``logs/train/hls4ml`` outside a ``fold_<i>/``
directory whose runs include the main one (``fold_idx`` null) is one row, named from its
directory, ``[<features>/]<model>/nconst<N>/<variant>``: ``label`` is the features
directory, ``default`` where there is none, and ``regime`` and ``augmentation_scheme``
are read off the variant (``aug_so2`` is augmentation under SO(2), ``baseline_1block``
the baseline). The counts come from the run the main entry names, found in the store by
its id alone. A run the store lacks, or a manifest that names none, as those written
before the manifests recorded the run, gives a row of ``nan`` and a warning on stderr; a
quantised run whose float network cannot be traced gives one in ``params`` alone::

    python3 scripts/analysis/collect_params.py <root> [<root> ...] --out params.csv
"""

import argparse
import json
import operator
import re
import sys
from pathlib import Path

import common

COLUMNS = [
    "model",
    "nconstituents",
    "regime",
    "augmentation_scheme",
    "label",
    "variant",
    "params",
    "flops",
    "mlflow_run_id",
    "run_dir",
]
TRAIN = Path("logs/train/hls4ml")
STORE = Path("logs/mlflow/mlruns")
MANIFEST = "checkpoint_manifest.json"
NAN = "nan"

# Variant directory prefix -> the regime its runs are tabulated under.
REGIMES = {"aug_": "augmentation", "symm_": "symm_loss"}


def _warn(message: str) -> None:
    print(f"[warn] {message}", file=sys.stderr)


def _regime(variant: str) -> tuple[str, str]:
    """The regime and the augmentation scheme a variant directory names; ``nan`` for
    both when it names none."""
    for prefix, regime in REGIMES.items():
        if variant.startswith(prefix):
            return regime, variant.removeprefix(prefix)
    return ("baseline", "none") if variant.startswith("baseline") else (NAN, NAN)


def _layout(run_dir: Path, train: Path) -> dict[str, object] | None:
    """The naming columns of a run directory; None, with a warning, for one laid out
    otherwise than ``[<features>/]<model>/nconst<N>/<variant>``."""
    parts = run_dir.relative_to(train).parts
    count = re.fullmatch(r"nconst(\d+)", parts[-2]) if len(parts) in (3, 4) else None
    if count is None:
        _warn(f"{run_dir}: not [<features>/]<model>/nconst<N>/<variant>, skipped")
        return None
    regime, scheme = _regime(parts[-1])
    label = parts[0] if len(parts) == 4 else "default"
    head = {"model": parts[-3], "nconstituents": int(count[1]), "regime": regime}
    return head | {"augmentation_scheme": scheme, "label": label, "variant": parts[-1]}


def _main_entry(manifest: Path) -> dict | None:
    """The manifest's main run, the one entry whose ``fold_idx`` is null."""
    runs = json.loads(manifest.read_text()).get("runs", [])
    return next((run for run in runs if run.get("fold_idx") is None), None)


def _store_run(root: Path, run_id: str | None) -> Path | None:
    """The run's directory in the MLflow file store, under whichever experiment."""
    found = sorted(root.joinpath(STORE).glob(f"*/{run_id}")) if run_id else []
    return found[0] if found else None


def _param(run: Path | None, key: str) -> str:
    """One MLflow param as the store holds it, ``nan`` when absent."""
    path = run / "params" / key if run else None
    return path.read_text().strip() if path and path.is_file() else NAN


def _run_dir(init_from: str) -> Path:
    """The run directory an ``init_from`` names: itself, or that of a ``.ckpt`` in its
    ``checkpoints/``."""
    path = Path(init_from)
    return path.parents[1] if path.suffix == ".ckpt" else path


def _float_params(root: Path, run: Path) -> str:
    """The count of the float network a quantised run names as its ``init_from``, nan
    for one that names none."""
    source = _param(run, "algorithm/init_from")
    if source in (NAN, "None"):
        return NAN
    manifest = root / _run_dir(source) / MANIFEST
    entry = _main_entry(manifest) if source != NAN and manifest.is_file() else None
    float_run = _store_run(root, entry.get("mlflow_run_id") if entry else None)
    count = _param(float_run, "algorithm/params/total")
    if count == NAN:
        _warn(f"{run}: no float count behind init_from={source}, so its params are nan")
    return count


def _params(root: Path, run: Path | None, model: str) -> str:
    """The torch count; the float network's on a quantised row, none on a BDT row."""
    if "bdt" in model or run is None:
        return NAN
    if model.startswith(("vn_hgq", "jedi_linear_hgq")):
        return _float_params(root, run)
    return _param(run, "algorithm/params/total")


def _check(run: Path, names: dict[str, object]) -> None:
    """Warn where the store disagrees with the directory on the model or the count.

    The count is the data's: ``algorithm/nconstituents`` is logged unresolved, as
    ``${data.nconstituents}``, and is 1 on the jet-level rows.
    """
    logged = {
        "model": _param(run, "algorithm/name"),
        "nconstituents": _param(run, "data/nconstituents"),
    }
    for column, value in logged.items():
        if value not in (NAN, str(names[column])):
            _warn(f"{run} logs {column}={value}, its path says {names[column]}")


def _counts(root: Path, run_id: str | None, names: dict, where: Path) -> dict:
    """The row's two counts, with a warning where the store lacks the run."""
    run = _store_run(root, run_id)
    if run is None:
        _warn(f"{where}: no run {run_id} in {root / STORE}, so its counts are nan")
    else:
        _check(run, names)
    flops = _param(run, "algorithm/flops")
    return {"params": _params(root, run, names["model"]), "flops": flops}


def _row(root: Path, manifest: Path) -> dict[str, object] | None:
    """One manifest's row; None when it holds no main run or sits elsewhere."""
    entry = _main_entry(manifest)
    names = _layout(manifest.parent, root / TRAIN) if entry else None
    if names is None:
        return None
    run_id = entry.get("mlflow_run_id")
    counts = _counts(root, run_id, names, manifest.parent)
    return names | counts | {"mlflow_run_id": run_id or NAN, "run_dir": manifest.parent}


def _manifests(root: Path) -> list[Path]:
    """Every manifest under the root's training logs but those of the folds."""
    train = root / TRAIN
    if not train.is_dir():
        _warn(f"{root} holds no {TRAIN}")
    found = sorted(train.rglob(MANIFEST))
    return [path for path in found if not path.parent.name.startswith("fold_")]


def collect(roots: list[Path]) -> list[dict[str, object]]:
    """One row per main run under the roots, sorted by label, model, count, variant."""
    found = [(root, path) for root in roots for path in _manifests(root)]
    rows = [row for root, path in found if (row := _row(root, path))]
    key = operator.itemgetter("label", "model", "nconstituents", "variant", "run_dir")
    return sorted(rows, key=key)


def _parse_args() -> argparse.Namespace:
    """Command line: the roots to read and the table to write."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Directories holding logs/train/hls4ml and logs/mlflow/mlruns.",
    )
    parser.add_argument("--out", type=Path, required=True, help="The CSV to write.")
    return parser.parse_args()


def main() -> None:
    """Write one row per main run under every root the command line names."""
    args = _parse_args()
    rows = collect(args.roots)
    common.write_csv(args.out, rows, COLUMNS)
    print(f"[done] {len(rows)} main runs -> {args.out}")


if __name__ == "__main__":
    main()
