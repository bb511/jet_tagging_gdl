#!/usr/bin/env python3
"""Pick the checkpoint of a quantised training (vn_hgq, jedi_linear_hgq) that goes to
synthesis, by the selection of the published JEDI-linear (arXiv:2508.15468).

Their training keeps, as HGQ2's ``ParetoFront``, every epoch on the Pareto front of the
validation accuracy (maximised, above 0.5) against the EBOPs (minimised), both as the
training logs them; the design is the front's checkpoint whose EBOPs, traced at its
evaluation, lie nearest a target. A training therefore keeps a checkpoint every epoch
(``+callbacks.model_checkpoint.every_n_epochs=1``)::

    python scripts/experiments/quantised_picks.py front <run dir>
    python scripts/experiments/quantised_picks.py pick <run dir> <evaluations> \\
        <target EBOPs> <picks dir>

``front`` prints the front's epochs, one per line, read off ``<run dir>/train.log``;
each is evaluated by ``src/eval.py`` with ``ckpt_path=<run dir>/checkpoints/
epoch_<k>.ckpt`` and ``hydra.run.dir=<evaluations>/epoch_<k>``. ``pick`` reads their
traced EBOPs from ``<evaluations>/epoch_<k>/outputs/main_hgq/export.json`` and writes the
run's own manifest, its checkpoint replaced by the pick's, to ``<picks dir>`` for the main
run and to ``<picks dir>/fold_<i>`` for fold i: the layout in which ``src/eval.py``
merges a main run with its folds, so that ``checkpoint_manifest_path=<picks dir>``
evaluates the picks of every fold into one results row. The test split enters no step.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.checkpoints import (  # noqa: E402
    MANIFEST_FILENAME,
    read_manifest,
    write_manifest,
)

# The per-epoch lines of keras_base.py and of the base class's validation, behind their
# logger prefixes; Lightning's sanity check logs a validation line at epoch 0 first, so
# the last one of an epoch is kept.
STEP = re.compile(r"\[epoch\s+(\d+)\] EBOPs ([0-9.eE+-]+),")
VAL = re.compile(r"\[epoch\s+(\d+)\]\s+val \| loss \S+ acc ([0-9.]+)")
# ParetoFront's enable_if in their train.py: a collapsed network enters no front.
VAL_MIN = 0.5


def logged(log: Path) -> dict[int, tuple[float, float]]:
    """Return {epoch: (validation accuracy, EBOPs)} of the epochs that logged both."""
    ebops, acc = {}, {}
    for line in log.read_text(errors="replace").splitlines():
        if m := STEP.search(line):
            ebops[int(m[1])] = float(m[2])
        elif m := VAL.search(line):
            acc[int(m[1])] = float(m[2])
    return {epoch: (acc[epoch], ebops[epoch]) for epoch in ebops.keys() & acc.keys()}


def front(points: dict[int, tuple[float, float]]) -> list[int]:
    """The epochs on the front, by rising EBOPs; of equal EBOPs, the more accurate."""
    kept, best = [], -math.inf
    for epoch, (acc, _) in sorted(points.items(), key=lambda p: (p[1][1], -p[1][0])):
        if acc > VAL_MIN and acc > best:
            kept.append(epoch)
            best = acc
    return kept


def _stem(epoch: int) -> str:
    # The checkpoint name of configs/callbacks/util/model_checkpoint.yaml.
    return f"epoch_{epoch:03d}"


def traced_ebops(evaluations: Path, epoch: int) -> float:
    """The EBOPs that the evaluation of *epoch* traced, from its export."""
    export = evaluations / _stem(epoch) / "outputs" / "main_hgq" / "export.json"
    return float(json.loads(export.read_text())["ebops"])


def pick(run_dir: Path, evaluations: Path, target: float) -> int:
    """The front's epoch whose traced EBOPs lie nearest *target*, the earlier on a
    tie."""
    epochs = front(logged(run_dir / "train.log"))
    if not epochs:
        raise SystemExit(f"{run_dir}: no epoch on the front")
    return min(epochs, key=lambda e: (abs(traced_ebops(evaluations, e) - target), e))


def write_pick(run_dir: Path, epoch: int, picks_dir: Path) -> Path:
    """Write the manifest of *run_dir* with the checkpoint of *epoch* under *picks_dir*,
    in ``fold_<i>/`` for a fold."""
    manifest = read_manifest(run_dir / MANIFEST_FILENAME)
    checkpoint = (run_dir / "checkpoints" / f"{_stem(epoch)}.ckpt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{checkpoint}: the picked epoch kept no checkpoint")
    run = replace(manifest.runs[0], checkpoint_path=str(checkpoint))
    out = picks_dir if run.fold_idx is None else picks_dir / run.fold_label
    return write_manifest(replace(manifest, runs=[run]), out)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("front").add_argument("run_dir", type=Path)
    chosen = commands.add_parser("pick")
    for name, kind in (("run_dir", Path), ("evaluations", Path), ("target", float)):
        chosen.add_argument(name, type=kind)
    chosen.add_argument("picks_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.command == "front":
        print(*front(logged(args.run_dir / "train.log")), sep="\n")
        return
    epoch = pick(args.run_dir, args.evaluations, args.target)
    path = write_pick(args.run_dir, epoch, args.picks_dir)
    print(
        f"{_stem(epoch)}: {traced_ebops(args.evaluations, epoch):.0f} EBOPs -> {path}"
    )


if __name__ == "__main__":
    main()
