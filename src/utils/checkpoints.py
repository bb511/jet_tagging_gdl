"""Manifest of the checkpoints one training run produced."""

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
    """One fitted model: the main run, or one cross-validation fold.

    ``mlflow_run_id`` is the MLflow run it logged into, ``None`` under ``logger=none``;
    the folds of a training nest under the main run's.
    """

    fold_label: str
    fold_idx: int | None
    checkpoint_path: str
    val_acc_best: float | None = None
    mlflow_run_id: str | None = None


@dataclass
class Manifest:
    """Everything ``src/eval.py`` needs to find and label a run's checkpoints."""

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

    :raises ValueError: on a version mismatch, since an older manifest may name fields
        this reader would silently drop.
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


def _fold_number(path: Path) -> int:
    """Sort key of a ``fold_<i>/`` manifest: its index, so fold_10 follows fold_2."""
    return int(path.parent.name.removeprefix("fold_"))


def _main_manifest(base_dir: Path) -> Manifest:
    """The main run's manifest, at the top of *base_dir*.

    :raises FileNotFoundError: when it is missing: the merge takes the run's name and
        directory from it, and the folds are the uncertainty on it, not a run of their
        own.
    """
    path = base_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"No main run manifest at {path}. A fold=null process with data.n_folds=1 "
            "writes it there; the fold_<i>/ directories beside it hold the folds alone."
        )
    return read_manifest(path)


def _one_fold_manifest(path: Path) -> Manifest:
    """The manifest a ``fold=<i>`` process wrote, holding that one fold and nothing else.

    :raises ValueError: otherwise, as for a sequential k-fold manifest dropped into a
        ``fold_<i>/`` directory, whose folds would then be counted twice.
    """
    manifest = read_manifest(path)
    if len(manifest.runs) != 1 or manifest.runs[0].fold_idx is None:
        labels = [run.fold_label for run in manifest.runs]
        raise ValueError(
            f"{path} lists {labels}, but a fold manifest holds exactly one fold run, as a "
            "fold=<i> process writes it."
        )
    return manifest


def _fold_manifests(base_dir: Path) -> list[Manifest]:
    """The manifests of the folds trained beside the main run, in fold order.

    A fold still training has written none, so it is left out, with a warning, rather than waited
    for.
    """
    paths = sorted(base_dir.glob(f"fold_*/{MANIFEST_FILENAME}"), key=_fold_number)
    folds = [_one_fold_manifest(path) for path in paths]
    if any(fold.n_folds != len(folds) for fold in folds):
        log.warning(
            f"{len(folds)} fold manifest(s) under {base_dir}, but the folds were trained as "
            f"{folds[0].n_folds}: one is still running, or died."
        )
    log.info(
        f"Merging the main manifest in {base_dir} with {[str(path) for path in paths]}"
    )
    return folds


def merge_manifests(base_dir: str | Path) -> Manifest:
    """Join the manifest of a main run with those of the folds trained beside it as processes.

    The layout is the one ``scripts/cluster/submit_train.sh --folds`` launches: the main run
    (``fold=null``, ``data.n_folds=1``) in *base_dir*, fold i (``fold=i``) in ``fold_<i>/`` under
    it.

    :return: the main run first, then the folds by index. ``n_folds`` counts the fold manifests
        found.
    """
    base_dir = Path(base_dir)
    main = _main_manifest(base_dir)
    folds = _fold_manifests(base_dir)
    return Manifest(
        n_folds=len(folds),
        hydra_output_dir=main.hydra_output_dir,
        experiment_name=main.experiment_name,
        runs=[*main.runs, *(fold.runs[0] for fold in folds)],
    )


def best_or_last_checkpoint(trainer: Trainer) -> str:
    """Absolute path of the checkpoint a finished run is evaluated from.

    The best by ``val/acc``, or the last when no ``ModelCheckpoint`` ranked any.

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
