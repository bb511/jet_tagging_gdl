# Run the evaluation.

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${len:} and ${short_hash:}.
register_resolvers()

from src.callbacks.invariance import InvarianceMetricCallback
from src.callbacks.plotting import PlottingCallback
from src.callbacks.results_table import ResultsTableCallback
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    nest_under,
    task_wrapper,
)
from src.utils.checkpoints import ManifestRun, merge_manifests, read_manifest

log = RankedLogger(__name__, rank_zero_only=True)


def _manifest_runs(path: str) -> list[ManifestRun]:
    """Get the manifest to all trainings corresponding to a run."""
    return (merge_manifests(path) if Path(path).is_dir() else read_manifest(path)).runs


def _runs_to_evaluate(cfg: DictConfig) -> list[ManifestRun]:
    """Decide whether to compose the manifest for one run only or for a whole training.

    You either evaluate one set of checkpoints, or you aggregate over kfolds; this is
    configured in the main configuration.

    :raises ValueError: when neither is set, so a mistyped path fails rather than test nothing.
    """
    if cfg.get("checkpoint_manifest_path"):
        return _manifest_runs(cfg.checkpoint_manifest_path)
    if cfg.get("ckpt_path"):
        return [ManifestRun("main", None, str(cfg.ckpt_path))]
    raise ValueError("Nothing to evaluate: set checkpoint_manifest_path=... or ckpt_path=...")


def _aggregated_labels(runs: list[ManifestRun]) -> set[str]:
    """Select which of the runs are considered and aggregated for the results table.

    This is done so the main run is not mixed with the folds, since this would double-
    count jets.
    """
    folds = {run.fold_label for run in runs if run.fold_idx is not None}
    return folds or {run.fold_label for run in runs}


def _baseline_checkpoints(cfg: DictConfig) -> dict[str, str]:
    """The baseline checkpoints (trained without augmentations or SCL).

    Inference of the baseline models are used to compute deviations from this baseline.
    If this is not specified, then absolute values are reported only.
    The kfolding is seeded deterministically. Therefore, the baseline and the augmented
    or SCL model are trained on the same baseline.
    """
    path = cfg.get("baseline_checkpoint_manifest_path")
    return {run.fold_label: run.checkpoint_path for run in _manifest_runs(path)} if path else {}


def _callbacks(cfg: DictConfig, label: str, table: ResultsTableCallback | None) -> list[Callback]:
    """Build the callbacks that run per checkpoint evaluation."""
    composed = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = [table if isinstance(cb, ResultsTableCallback) else cb for cb in composed]
    for callback in callbacks:
        if isinstance(callback, PlottingCallback):
            callback.subdir = label
    return [cb for cb in callbacks if cb is not None]


def _attach_baseline(
    callbacks: list[Callback], algorithm: LightningModule, checkpoint_path: str | None
) -> None:
    """Get the baseline to perform the invariance probe."""
    if checkpoint_path is None:
        return
    # weights_only=False: the hyperparameters are plain containers, which torch.load 2.6 rejects.
    baseline = type(algorithm).load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    for callback in callbacks:
        if isinstance(callback, InvarianceMetricCallback):
            callback.baseline_module = baseline


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test every checkpoint of one training on the held-out test split.

    The k-folds share the same results table: the whole training yields one row.
    The k-fold runs are nested underneath the main run in MLFlow.

    :return: every checkpoint's metrics, keyed ``<fold_label>/<metric>``, and the shared objects.
    """
    runs = _runs_to_evaluate(cfg)
    aggregated = _aggregated_labels(runs)
    baselines = _baseline_checkpoints(cfg)
    table = None
    if cfg.get("callbacks") and cfg.callbacks.get("results_table"):
        # The manifest, not data.n_folds, says how many runs the row waits for.
        table = hydra.utils.instantiate(cfg.callbacks.results_table, n_folds=len(aggregated))

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    metric_dict: dict[str, Any] = {}
    for run in runs:
        # Re-seeded per checkpoint: the invariance probe draws its transforms from the global torch
        # RNG, so the spread over folds would otherwise mix model and transform variation.
        if cfg.get("seed") is not None:
            L.seed_everything(cfg.seed, workers=True)
        algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
        callbacks = _callbacks(
            cfg, run.fold_label, table if run.fold_label in aggregated else None
        )
        _attach_baseline(callbacks, algorithm, baselines.get(run.fold_label))
        logger = instantiate_loggers(cfg.get("logger"))
        nest_under(logger, run.mlflow_run_id)
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
        if logger:
            objects = {"cfg": cfg, "algorithm": algorithm, "trainer": trainer}
            log_hyperparameters({**objects, "fold_idx": run.fold_label})

        log.info(f"[{run.fold_label}] Testing {run.checkpoint_path}")
        # weights_only=False: the hyperparameters are plain containers, which torch.load 2.6 rejects.
        trainer.test(
            model=algorithm,
            datamodule=datamodule,
            ckpt_path=run.checkpoint_path,
            weights_only=False,
        )
        metrics = trainer.callback_metrics
        metric_dict.update({f"{run.fold_label}/{key}": value for key, value in metrics.items()})

    return metric_dict, {"cfg": cfg, "datamodule": datamodule}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoints ``cfg`` points at."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
