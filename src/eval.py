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

from src.callbacks.plotting import PlottingCallback
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
from src.utils.results_table import write_results_row

log = RankedLogger(__name__, rank_zero_only=True)


def _runs_to_evaluate(cfg: DictConfig) -> list[ManifestRun]:
    """Decide whether to compose the manifest for one run only or for a whole training.

    You either evaluate one set of checkpoints, or you aggregate over kfolds; this is configured in
    the main configuration.

    :raises ValueError: when neither is set, so a mistyped path fails rather than test nothing.
    """
    path = cfg.get("checkpoint_manifest_path")
    if path:
        return (
            merge_manifests(path) if Path(path).is_dir() else read_manifest(path)
        ).runs
    if cfg.get("ckpt_path"):
        return [ManifestRun("main", None, str(cfg.ckpt_path))]
    raise ValueError(
        "Nothing to evaluate: set checkpoint_manifest_path=... or ckpt_path=..."
    )


def _aggregated_labels(runs: list[ManifestRun]) -> set[str]:
    """Select which of the runs are considered and aggregated for the results table.

    This is done so the main run is not mixed with the folds, since this would double- count jets.
    """
    folds = {run.fold_label for run in runs if run.fold_idx is not None}
    return folds or {run.fold_label for run in runs}


def _callbacks(cfg: DictConfig, label: str) -> list[Callback]:
    """Build the callbacks that run per checkpoint evaluation, drawing into *label*'s own plots."""
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    for callback in callbacks:
        if isinstance(callback, PlottingCallback):
            callback.subdir = label
    return callbacks


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test every checkpoint of one training on the held-out test split.

    The folds are summarised into one results row once the last of them has been tested; the main
    run is left out of it whenever folds exist, so no jet is counted twice. The k-fold runs are
    nested underneath the main run in MLFlow.

    :return: every checkpoint's metrics, keyed ``<fold_label>/<metric>``, and the shared objects.
    """
    runs = _runs_to_evaluate(cfg)
    aggregated = _aggregated_labels(runs)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    metric_dict: dict[str, Any] = {}
    folds: list[dict[str, Any]] = []
    for run in runs:
        # Re-seeded per checkpoint: the invariance probe draws its transforms from the global torch
        # RNG, so the spread over folds would otherwise mix model and transform variation.
        if cfg.get("seed") is not None:
            L.seed_everything(cfg.seed, workers=True)
        algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
        callbacks = _callbacks(cfg, run.fold_label)
        logger = instantiate_loggers(cfg.get("logger"))
        nest_under(logger, run.mlflow_run_id)
        trainer: Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=logger
        )
        if logger:
            objects = {"cfg": cfg, "algorithm": algorithm, "trainer": trainer}
            log_hyperparameters({**objects, "fold_idx": run.fold_label})

        log.info(f"[{run.fold_label}] Testing {run.checkpoint_path}")
        # weights_only=False: the hyperparameters are plain containers, torch.load 2.6 rejects.
        trainer.test(
            model=algorithm,
            datamodule=datamodule,
            ckpt_path=run.checkpoint_path,
            weights_only=False,
        )
        metrics = trainer.callback_metrics
        metric_dict.update(
            {f"{run.fold_label}/{key}": value for key, value in metrics.items()}
        )
        if run.fold_label in aggregated:
            # flops_total is an attribute of the module rather than a logged metric, and the row
            # reads every number of a fold from one dict.
            folds.append(dict(metrics) | {"flops_total": algorithm.flops_total})

    write_results_row(cfg, folds)
    return metric_dict, {"cfg": cfg, "datamodule": datamodule}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoints ``cfg`` points at."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
