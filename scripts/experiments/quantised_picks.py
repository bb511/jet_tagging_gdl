#!/usr/bin/env python3
"""Pick the checkpoint of a quantised training (vn_hgq, jedi_linear_hgq) that goes to
synthesis, by the selection of the published JEDI-linear (arXiv:2508.15468) on a
calibrated validation accuracy.

Their training keeps, as HGQ2's ``ParetoFront``, every epoch on the Pareto front of the
validation accuracy (maximised, above 0.5) against the EBOPs (minimised); the design is
the front's checkpoint whose EBOPs lie nearest a target. Here both are calibrated: an
epoch's port is tested on the validation jets of its own training
(``data.test_split=validation``) once its integer bits are traced on the training split,
as at the test. The validation accuracy the training logs is not, since it runs on the
integer bits traced on the epoch's last batch, past which the vector neurons' inputs
wrap. A training therefore keeps a checkpoint every epoch
(``+callbacks.model_checkpoint.every_n_epochs=1``)::

    python scripts/experiments/quantised_picks.py candidates <run dir>
    python scripts/experiments/quantised_picks.py pick <run dir> <evaluations> \\
        <target EBOPs> <picks dir>

``candidates`` prints, one per line, the epochs with a checkpoint on the front of the
logged validation accuracy or on that of the logged training accuracy, each against the
logged EBOPs, read off ``<run dir>/train.log``: the training accuracy does not wrap, so
its front keeps the epochs a wrapped validation hides. An epoch whose logged training
accuracy lies below ``TRAIN_FRACTION`` of the run's best is collapsing and is left out,
a guard their selection lacks. Each is evaluated by
``src/eval.py`` with ``ckpt_path=<run dir>/checkpoints/epoch_<k>.ckpt
data.test_split=validation fold=<i, or null for the main run>
hydra.run.dir=<evaluations>/epoch_<k>``. ``pick`` reads each candidate's micro accuracy
off ``outputs/main.npz`` and its traced EBOPs off ``outputs/main_hgq/export.json``,
takes their front and writes the run's own manifest, its checkpoint replaced by the
pick's, to ``<picks dir>`` for the main run and to ``<picks dir>/fold_<i>`` for fold i:
the layout in which ``src/eval.py`` merges a main run with its folds, so that
``checkpoint_manifest_path=<picks dir>`` evaluates the picks of every fold into one
results row. The test split enters no step.
"""

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

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
STEP = re.compile(r"\[epoch\s+(\d+)\] EBOPs ([0-9.eE+-]+),.*train acc ([0-9.]+)")
VAL = re.compile(r"\[epoch\s+(\d+)\]\s+val \| loss \S+ acc ([0-9.]+)")
# ParetoFront's enable_if in their train.py: a collapsed network enters no front.
ACC_MIN = 0.5
# A collapsing network can still validate above ACC_MIN; its training accuracy, which does
# not wrap, has by then fallen below this fraction of the run's best, which no JEDI-linear
# pick at N = 16 does (2026-10-14).
TRAIN_FRACTION = 0.95


def logged(log: Path) -> dict[int, tuple[float, float, float]]:
    """Return {epoch: (validation accuracy, training accuracy, EBOPs)}, as logged, of
    the epochs that logged all three."""
    step, val = {}, {}
    for line in log.read_text(errors="replace").splitlines():
        if m := STEP.search(line):
            step[int(m[1])] = (float(m[3]), float(m[2]))
        elif m := VAL.search(line):
            val[int(m[1])] = float(m[2])
    return {epoch: (val[epoch], *step[epoch]) for epoch in step.keys() & val.keys()}


def front(points: dict[int, tuple[float, float]]) -> list[int]:
    """The epochs on the front of {epoch: (accuracy, EBOPs)}, by rising EBOPs; of equal
    EBOPs, the more accurate."""
    kept, best = [], -math.inf
    for epoch, (acc, _) in sorted(points.items(), key=lambda p: (p[1][1], -p[1][0])):
        if acc > ACC_MIN and acc > best:
            kept.append(epoch)
            best = acc
    return kept


def _stem(epoch: int) -> str:
    # The checkpoint name of configs/callbacks/util/model_checkpoint.yaml.
    return f"epoch_{epoch:03d}"


def candidates(run_dir: Path) -> list[int]:
    """The epochs to evaluate: on either logged front, not collapsing, with a checkpoint
    kept."""
    points = logged(run_dir / "train.log")
    on_fronts = set(front({e: (val, ebops) for e, (val, _, ebops) in points.items()}))
    on_fronts |= set(front({e: (acc, ebops) for e, (_, acc, ebops) in points.items()}))
    floor = TRAIN_FRACTION * max((acc for _, acc, _ in points.values()), default=0.0)
    kept = run_dir / "checkpoints"
    return sorted(
        e
        for e in on_fronts
        if points[e][1] >= floor and (kept / f"{_stem(e)}.ckpt").is_file()
    )


def calibrated(evaluations: Path, epoch: int) -> tuple[float, float]:
    """The micro accuracy on the validation jets and the traced EBOPs of the evaluation
    of *epoch*."""
    outputs = evaluations / _stem(epoch) / "outputs"
    with np.load(outputs / "main.npz") as arrays:
        predicted = arrays["logits_clean"].argmax(axis=1)
        accuracy = float((predicted == arrays["labels"]).mean())
    ebops = json.loads((outputs / "main_hgq" / "export.json").read_text())["ebops"]
    return accuracy, float(ebops)


def pick(run_dir: Path, evaluations: Path, target: float) -> tuple[int, float, float]:
    """The calibrated front's epoch whose EBOPs lie nearest *target*, the earlier on a
    tie, with its accuracy and EBOPs.

    :raises FileNotFoundError: when a candidate lacks its evaluation.
    """
    points = {e: calibrated(evaluations, e) for e in candidates(run_dir)}
    epochs = front(points)
    if not epochs:
        raise SystemExit(f"{run_dir}: no epoch on the front")
    epoch = min(epochs, key=lambda e: (abs(points[e][1] - target), e))
    return epoch, *points[epoch]


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
    """Parse the ``candidates`` and ``pick`` subcommands."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("candidates").add_argument("run_dir", type=Path)
    chosen = commands.add_parser("pick")
    for name, kind in (("run_dir", Path), ("evaluations", Path), ("target", float)):
        chosen.add_argument(name, type=kind)
    chosen.add_argument("picks_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Print a run's candidates, or write its pick's manifest."""
    args = _parse_args(argv)
    if args.command == "candidates":
        print(*candidates(args.run_dir), sep="\n")
        return
    epoch, accuracy, ebops = pick(args.run_dir, args.evaluations, args.target)
    path = write_pick(args.run_dir, epoch, args.picks_dir)
    print(f"{_stem(epoch)}: {accuracy:.4f} at {ebops:.0f} EBOPs -> {path}")


if __name__ == "__main__":
    main()
