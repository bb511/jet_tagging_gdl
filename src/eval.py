# Evaluation entry point: test the checkpoints a training run recorded in its manifest.
"""Test the checkpoints of one training run on the record's held-out test split.

``src/train.py`` writes a checkpoint manifest; this script reads it and tests every checkpoint in
it. The test split is the record's own, which the k-fold pooling never touches, so all folds are
scored on the same jets. A single checkpoint can be tested directly with ``ckpt_path`` instead.

One ``ResultsTableCallback`` is shared by the runs that enter the results table, so a k-fold
training yields one row carrying the spread over folds. The main run and the folds are never mixed
in that row: the folds re-split the pool the main run trained on.
"""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Before any other import of the project: .env carries PROJECT_ROOT, which every path in
# configs/paths/default.yaml resolves against, and HF_HUB_CACHE, which huggingface_hub
# reads once at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes anything, since the configs interpolate ${len:} and ${short_hash:}.
register_resolvers()

from src.callbacks.plotting import PlottingCallback
from src.callbacks.results_table import ResultsTableCallback
from src.utils import (
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.checkpoints import ManifestRun, read_manifest

log = RankedLogger(__name__, rank_zero_only=True)


def _runs_to_evaluate(cfg: DictConfig) -> list[ManifestRun]:
    """The checkpoints this evaluation covers: a manifest's runs, or a single ``ckpt_path``.

    :raises ValueError: when neither ``checkpoint_manifest_path`` nor ``ckpt_path`` is set, so that
        a mistyped path fails here rather than testing nothing.
    """
    manifest_path = cfg.get("checkpoint_manifest_path")
    if manifest_path:
        manifest = read_manifest(manifest_path)
        log.info(f"Evaluating {len(manifest.runs)} run(s) listed in {manifest_path}")
        return manifest.runs
    if not cfg.get("ckpt_path"):
        raise ValueError(
            "Nothing to evaluate. Set checkpoint_manifest_path=<the checkpoint_manifest.json "
            "of a training run> or ckpt_path=<a single checkpoint>."
        )
    return [ManifestRun(fold_label="main", fold_idx=None, checkpoint_path=str(cfg.ckpt_path))]


def _aggregated_labels(runs: list[ManifestRun]) -> set[str]:
    """The labels of the runs whose test metrics enter the results table.

    The folds when the manifest holds any, the main run otherwise. Mixing the two would count the
    same jets twice, since the folds re-split the pool the main run trained on.
    """
    folds = {run.fold_label for run in runs if run.fold_idx is not None}
    return folds or {run.fold_label for run in runs}


def _shared_results_callback(cfg: DictConfig) -> ResultsTableCallback | None:
    """One results table for every run that enters it, so k folds yield one row, not k."""
    if not cfg.get("callbacks") or not cfg.callbacks.get("results_table"):
        return None
    return hydra.utils.instantiate(cfg.callbacks.results_table)


def _eval_callbacks(
    cfg: DictConfig, results_callback: ResultsTableCallback | None, fold_label: str
) -> list[Callback]:
    """Instantiate one run's callbacks.

    :param results_callback: the instance shared across the runs that enter the results table,
        replacing the one this run composed; ``None`` drops the results table from this run
        altogether, which is how a main run is kept out of a k-fold row.
    :param fold_label: names the plot subdirectory, so that the checkpoints of one manifest do not
        draw over each other's figures.
    """
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    if results_callback is None:
        callbacks = [cb for cb in callbacks if not isinstance(cb, ResultsTableCallback)]
    else:
        callbacks = [
            results_callback if isinstance(cb, ResultsTableCallback) else cb for cb in callbacks
        ]
    for callback in callbacks:
        if isinstance(callback, PlottingCallback):
            callback.subdir = fold_label
    return callbacks


def _eval_one_checkpoint(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    run: ManifestRun,
    results_callback: ResultsTableCallback | None,
) -> dict[str, Any]:
    """Load one checkpoint into a fresh algorithm and test it.

    Re-seeded first: the invariance callback draws its SO(2), SO(3) and Lorentz transforms
    from the global torch RNG, so without this the folds would see different draws and their
    spread would mix model variation with transform variation.

    :return: the trainer's callback metrics for this checkpoint.
    """
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"[{run.fold_label}] Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    callbacks = _eval_callbacks(cfg, results_callback, run.fold_label)
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log_hyperparameters(
            {
                "cfg": cfg,
                "algorithm": algorithm,
                "trainer": trainer,
                "fold_idx": run.fold_label,
                "run_stage": cfg.get("run_stage", "eval"),
            }
        )

    log.info(f"[{run.fold_label}] Testing {run.checkpoint_path}")
    # weights_only=False: the checkpoints carry the hyperparameters as plain containers,
    # which torch.load rejects under its 2.6 default.
    trainer.test(
        model=algorithm, datamodule=datamodule, ckpt_path=run.checkpoint_path, weights_only=False
    )
    return dict(trainer.callback_metrics)


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test every checkpoint this evaluation covers, one after the other.

    :param cfg: the config Hydra composed; ``checkpoint_manifest_path`` or ``ckpt_path`` says what
        to test.
    :return: the metrics of every checkpoint, keyed ``<fold_label>/<metric>``, and the objects the
        runs share.
    """
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    runs = _runs_to_evaluate(cfg)
    aggregated = _aggregated_labels(runs)
    results_callback = _shared_results_callback(cfg)
    if results_callback is not None:
        # The row is written once this many runs have been tested. The manifest, not
        # data.n_folds, says how many there are: a single checkpoint evaluated under an
        # experiment config that trains five folds would otherwise never yield a row.
        results_callback.n_folds = len(aggregated)

    all_metrics: dict[str, Any] = {}
    for run in runs:
        if not run.checkpoint_path or not Path(run.checkpoint_path).exists():
            log.warning(f"[{run.fold_label}] No checkpoint at {run.checkpoint_path}, skipping")
            continue
        shared = results_callback if run.fold_label in aggregated else None
        metrics = _eval_one_checkpoint(cfg, datamodule, run, shared)
        all_metrics.update({f"{run.fold_label}/{k}": v for k, v in metrics.items()})

    return all_metrics, {"cfg": cfg, "datamodule": datamodule}


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoints ``cfg`` points at.

    :param cfg: the config Hydra composed.
    """
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
