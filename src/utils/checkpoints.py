# The checkpoint manifest: what a training run leaves behind for src/eval.py to read.
"""Manifest of the checkpoints one training run produced.

``src/train.py`` fits a main run and, when ``data.n_folds >= 2``, one run per fold, then writes a
single JSON manifest listing them. ``src/eval.py`` reads that manifest and tests every checkpoint
in it, so the two entry points share no state beyond this file.

Paths in the manifest are absolute: moving ``logs/`` after training breaks a later evaluation,
which is why ``scripts/symbolink.sh`` is meant to be run before the first training rather than
after.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lightning import Trainer

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "checkpoint_manifest.json"


@dataclass
class ManifestRun:
    """One fitted model: the main run, or one cross-validation fold."""

    fold_label: str
    fold_idx: int | None
    checkpoint_path: str
    val_acc_best: float | None = None


@dataclass
class Manifest:
    """Everything ``src/eval.py`` needs to find and label a training run's checkpoints."""

    n_folds: int
    hydra_output_dir: str
    experiment_name: str
    runs: list[ManifestRun] = field(default_factory=list)
    version: int = MANIFEST_VERSION


def write_manifest(manifest: Manifest, output_dir: str | Path) -> Path:
    """Write *manifest* to ``<output_dir>/checkpoint_manifest.json`` and return its path."""
    path = Path(output_dir) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2))
    log.info(f"Wrote checkpoint manifest with {len(manifest.runs)} runs to {path}")
    return path


def read_manifest(path: str | Path) -> Manifest:
    """Load a manifest written by :func:`write_manifest`.

    :raises FileNotFoundError: if *path* does not exist.
    :raises ValueError: if the file was written by an incompatible version, since an older manifest
        may name fields this reader would silently drop.
    """
    payload = json.loads(Path(path).read_text())
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"{path} has manifest version {version!r}, but this code reads version "
            f"{MANIFEST_VERSION}. Re-run the training that wrote it."
        )
    runs = [ManifestRun(**run) for run in payload.pop("runs", [])]
    return Manifest(runs=runs, **payload)


def best_or_last_checkpoint(trainer: Trainer) -> str:
    """Absolute path of the checkpoint a finished run should be evaluated from.

    The best checkpoint by ``val/acc``, or the last one when no ``ModelCheckpoint`` ran (a
    fast_dev_run, or the search callbacks).

    :raises RuntimeError: when the run wrote no checkpoint at all.
    """
    callback = getattr(trainer, "checkpoint_callback", None)
    best = getattr(callback, "best_model_path", "")
    last = getattr(callback, "last_model_path", "")
    for candidate in (best, last):
        if candidate:
            return str(Path(candidate).resolve())
    raise RuntimeError(
        "The run produced no checkpoint, so nothing can be evaluated later. Enable the "
        "model_checkpoint callback (callbacks=default)."
    )
