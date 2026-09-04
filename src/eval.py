"""Test the checkpoints of one training run on the record's held-out test split.

Every checkpoint of the manifest ``src/train.py`` wrote is tested, or a single one with
``ckpt_path``. A run directory in place of the manifest merges the main run's manifest with those
of the folds trained as separate processes under ``fold_<i>/``. The k-fold pooling never touches
the test split, so all folds are scored on the same jets. The runs entering the results table share
one ``ResultsTableCallback``, so a k-fold training yields one row carrying the spread over folds.

A second manifest, ``baseline_checkpoint_manifest_path``, names the baseline training of the same
architecture and constituent count. Its runs are matched to this one's fold by fold and handed to
the invariance probe, which scores both models on the same transformed batches and logs the
difference; without it the row's ``baseline_*`` and ``delta_*`` columns are NaN.
"""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${len:} and ${short_hash:}.
register_resolvers()

from src.callbacks.invariance import InvarianceMetricCallback
from src.callbacks.results_table import ResultsTableCallback
from src.utils import RankedLogger, extras, instantiate_loggers, task_wrapper
from src.utils.checkpoints import ManifestRun, merge_manifests, read_manifest
from src.utils.runs import (
    log_run_hyperparameters,
    prepare_callbacks,
    shared_results_callback,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _runs_from_path(path: str | Path) -> list[ManifestRun]:
    """The runs a manifest lists, *path* naming either the manifest file or the run directory.

    A directory holds the main run's manifest and, under ``fold_<i>/``, those of the folds trained
    as separate processes; the runs of all of them are covered.
    """
    manifest = merge_manifests(path) if Path(path).is_dir() else read_manifest(path)
    return manifest.runs


def _runs_to_evaluate(cfg: DictConfig) -> list[ManifestRun]:
    """The checkpoints this evaluation covers: a manifest's runs, or a single ``ckpt_path``.

    :raises ValueError: when neither is set, so a mistyped path fails rather than test nothing.
    """
    manifest_path = cfg.get("checkpoint_manifest_path")
    if manifest_path:
        runs = _runs_from_path(manifest_path)
        log.info(f"Evaluating {len(runs)} run(s) listed in {manifest_path}")
        return runs
    if cfg.get("ckpt_path"):
        return [ManifestRun(fold_label="main", fold_idx=None, checkpoint_path=str(cfg.ckpt_path))]
    raise ValueError(
        "Nothing to evaluate. Set checkpoint_manifest_path=<the checkpoint_manifest.json of a "
        "training run> or ckpt_path=<a single checkpoint>."
    )


def _baseline_checkpoints(cfg: DictConfig) -> dict[str, str]:
    """The baseline training's checkpoint for each of its fold labels, empty when none is set.

    The folds of two regimes are comparable because ``kfold_seed`` fixes the splits, so fold i of
    the baseline was fitted on the same jets as fold i of the run being evaluated. A checkpoint the
    manifest names but that is no longer on disk is left out, which costs its fold the deltas.

    :raises FileNotFoundError: when the path is set but names no manifest, so a mistyped baseline
        fails rather than write a row of silent NaNs.
    """
    path = cfg.get("baseline_checkpoint_manifest_path")
    if not path:
        return {}
    if not Path(path).exists():
        raise FileNotFoundError(f"No baseline checkpoint manifest at {path}.")
    runs = _runs_from_path(path)
    log.info(f"Comparing against the {len(runs)} baseline run(s) listed in {path}")
    return {
        run.fold_label: run.checkpoint_path for run in runs if Path(run.checkpoint_path).exists()
    }


def _baseline_for(baselines: dict[str, str], run: ManifestRun) -> str | None:
    """The baseline checkpoint matching one run's fold, warning when the fold has none."""
    if not baselines:
        return None
    checkpoint = baselines.get(run.fold_label)
    if checkpoint is None:
        log.warning(
            f"[{run.fold_label}] The baseline manifest lists no such fold, so this fold is "
            "scored without deltas."
        )
    return checkpoint


def _load_baseline(algorithm: LightningModule, checkpoint_path: str) -> LightningModule:
    """Rebuild the baseline from its own checkpoint, its layer sizes included.

    Each regime carries the best trial of its own study, so the baseline's widths need not match
    those of the run being evaluated: what shapes it is the hyperparameters saved in its own
    checkpoint. The optimiser and the scheduler are excluded from those, and the module is never
    fitted, so nothing beyond the network has to be reconstructed.
    """
    log.info(f"Loading the baseline module from {checkpoint_path}")
    # weights_only=False: the hyperparameters are plain containers, which torch.load 2.6 rejects.
    return type(algorithm).load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )


def _attach_baseline(
    callbacks: list[Callback], algorithm: LightningModule, checkpoint_path: str | None
) -> None:
    """Hand the baseline to the invariance probe, which feeds both models the same transforms.

    :param checkpoint_path:``None`` when no baseline is configured, which leaves the probe scoring
        this run alone and every delta unlogged.
    """
    if checkpoint_path is None:
        return
    baseline = _load_baseline(algorithm, checkpoint_path)
    for callback in callbacks:
        if isinstance(callback, InvarianceMetricCallback):
            callback.baseline_module = baseline


def _aggregated_labels(runs: list[ManifestRun]) -> set[str]:
    """The labels of the runs whose test metrics enter the results table.

    The folds when the manifest holds any, the main run otherwise: the folds re-split the pool the
    main run trained on, so mixing the two would count the same jets twice.
    """
    folds = {run.fold_label for run in runs if run.fold_idx is not None}
    return folds or {run.fold_label for run in runs}


def _eval_trainer(
    cfg: DictConfig,
    run: ManifestRun,
    results_callback: ResultsTableCallback | None,
    baseline_checkpoint: str | None,
) -> tuple[LightningModule, Trainer]:
    """Build the algorithm and the trainer that will test one checkpoint.

    :param baseline_checkpoint: the matching fold of the baseline training, handed to the
        invariance probe so this run is scored against it.
    """
    log.info(f"[{run.fold_label}] Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    callbacks = prepare_callbacks(cfg, results_callback, run.fold_label)
    _attach_baseline(callbacks, algorithm, baseline_checkpoint)
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    if loggers:
        stage = cfg.get("run_stage", "eval")
        log_run_hyperparameters(cfg, algorithm, trainer, run.fold_label, stage)
    return algorithm, trainer


def _eval_one_checkpoint(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    run: ManifestRun,
    results_callback: ResultsTableCallback | None,
    baseline_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Load one checkpoint into a fresh algorithm and test it.

    Re-seeded first: the invariance callback draws its transforms from the global torch RNG, so the
    spread over folds would otherwise mix model variation with transform variation.
    """
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    algorithm, trainer = _eval_trainer(cfg, run, results_callback, baseline_checkpoint)
    log.info(f"[{run.fold_label}] Testing {run.checkpoint_path}")
    # weights_only=False: the hyperparameters are plain containers, which torch.load 2.6 rejects.
    trainer.test(
        model=algorithm, datamodule=datamodule, ckpt_path=run.checkpoint_path, weights_only=False
    )
    return dict(trainer.callback_metrics)


def _testable(runs: list[ManifestRun]) -> list[ManifestRun]:
    """The runs whose checkpoint is still on disk; a vanished one is reported, not tested."""
    kept = []
    for run in runs:
        if run.checkpoint_path and Path(run.checkpoint_path).exists():
            kept.append(run)
        else:
            log.warning(f"[{run.fold_label}] No checkpoint at {run.checkpoint_path}, skipping")
    return kept


def _test_every_run(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    runs: list[ManifestRun],
    results_callback: ResultsTableCallback | None,
) -> dict[str, Any]:
    """Test each run in turn, the aggregated ones sharing one results table.

    :return: every run's metrics, keyed ``<fold_label>/<metric>``.
    """
    aggregated = _aggregated_labels(runs)
    baselines = _baseline_checkpoints(cfg)
    if results_callback is not None:
        # The manifest, not data.n_folds, says how many runs the row waits for: one checkpoint
        # tested under a five-fold experiment config would otherwise never yield a row.
        results_callback.n_folds = len(aggregated)
    all_metrics: dict[str, Any] = {}
    for run in _testable(runs):
        shared = results_callback if run.fold_label in aggregated else None
        metrics = _eval_one_checkpoint(cfg, datamodule, run, shared, _baseline_for(baselines, run))
        all_metrics.update({f"{run.fold_label}/{key}": value for key, value in metrics.items()})
    return all_metrics


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test every checkpoint this evaluation covers, one after the other.

    :return: every checkpoint's metrics, keyed ``<fold_label>/<metric>``, and the shared objects.
    """
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    runs = _runs_to_evaluate(cfg)
    results_callback = shared_results_callback(cfg)
    metrics = _test_every_run(cfg, datamodule, runs, results_callback)

    return metrics, {"cfg": cfg, "datamodule": datamodule}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoints ``cfg`` points at."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
