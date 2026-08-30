# Training entry point: one main run, then the cross-validation folds, then the manifest.
"""Fit one architecture on the hls4ml jet data set.

Every run fits a main model on the record's own train/validation split. When ``data.n_folds >= 2``
it then fits one model per fold over the pooled train and validation jets, which is where the
uncertainty on a reported accuracy comes from. The metrics are returned three times over: once
under ``main/``, once per fold under ``fold<i>/``, and once at the top level mirroring the main
run, which is what an Optuna trial reads back.

Nothing is tested here. The run writes a checkpoint manifest (``src/utils/checkpoints.py``) and
``src/eval.py`` tests every checkpoint listed in it.
"""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
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
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.checkpoints import (
    Manifest,
    ManifestRun,
    best_or_last_checkpoint,
    write_manifest,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _shared_results_callback(cfg: DictConfig) -> ResultsTableCallback | None:
    """One results table for every fold, so that k folds yield one row rather than k."""
    if not cfg.get("callbacks") or not cfg.callbacks.get("results_table"):
        return None
    return hydra.utils.instantiate(cfg.callbacks.results_table)


def _fold_callbacks(
    cfg: DictConfig, results_callback: ResultsTableCallback | None, checkpoint_subdir: str
) -> list[Callback]:
    """Instantiate one run's callbacks, with its checkpoints in a directory of its own.

    :param results_callback: the instance the folds share, which replaces the one this run
        composed; ``None`` leaves the composed callbacks untouched.
    :param checkpoint_subdir: directory under ``${paths.output_dir}/checkpoints``.
    """
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    if results_callback is not None:
        callbacks = [
            results_callback if isinstance(cb, ResultsTableCallback) else cb for cb in callbacks
        ]
    checkpoint_dir = Path(cfg.paths.output_dir) / "checkpoints" / checkpoint_subdir
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            callback.dirpath = str(checkpoint_dir)
        elif isinstance(callback, PlottingCallback):
            # Same subdirectory as the checkpoints: without it every fold would draw over the
            # previous fold's figures, since they all share ${paths.output_dir}/plots.
            callback.subdir = checkpoint_subdir
    return callbacks


def _drop_lr_monitor(callbacks: list[Callback]) -> list[Callback]:
    """Remove LearningRateMonitor, which raises at on_train_start when there is no logger.

    Searches, debug runs and the catalogue blocks all pass ``logger=none`` while composing
    ``callbacks=default`` or ``callbacks=search``, and the pair is a crash rather than a warning in
    Lightning, so the monitor is dropped here instead of from the configs.
    """
    return [cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)]


def _run_one_fold(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    results_callback: ResultsTableCallback | None,
    fold_label: str,
    checkpoint_subdir: str,
    ckpt_path: str | None = None,
) -> tuple[Trainer, LightningModule, list[Callback], list[Logger]]:
    """Fit one model on whichever split *datamodule* currently holds.

    The datamodule is not rebuilt between folds: the record is read and the k-fold pool
    assembled once, and ``set_fold`` alone moves the split.

    :param fold_label: names the run in the logs and in the manifest, "main" or "fold_<i>".
    :param ckpt_path: a checkpoint to resume the fit from.
    :return: the trainer, the fitted algorithm, its callbacks and its loggers.
    """
    log.info(f"[{fold_label}] Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    callbacks = _fold_callbacks(cfg, results_callback, checkpoint_subdir)
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    if not loggers:
        callbacks = _drop_lr_monitor(callbacks)
    log.info(f"[{fold_label}] Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    if loggers:
        log_hyperparameters(
            {
                "cfg": cfg,
                "algorithm": algorithm,
                "trainer": trainer,
                "fold_idx": fold_label,
                "run_stage": "train",
            }
        )

    log.info(f"[{fold_label}] Starting training")
    trainer.fit(model=algorithm, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, algorithm, callbacks, loggers


def _manifest_run(
    trainer: Trainer, fold_label: str, fold_idx: int | None, metrics: dict[str, Any]
) -> ManifestRun | None:
    """The manifest entry of a finished run, or ``None`` when it saved no checkpoint.

    A debug run, or any run composed without the model_checkpoint callback, trains normally but
    leaves nothing for ``src/eval.py`` to load. That is worth a warning, not a failure.
    """
    try:
        checkpoint_path = best_or_last_checkpoint(trainer)
    except RuntimeError as error:
        log.warning(f"[{fold_label}] {error}")
        return None
    val_acc_best = metrics.get("val/acc_best")
    best = None if val_acc_best is None else float(val_acc_best)
    return ManifestRun(fold_label, fold_idx, checkpoint_path, best)


def _run_folds(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int
) -> tuple[dict[str, Any], list[ManifestRun]]:
    """Fit one model per fold over the pooled train and validation jets.

    :return: the fold metrics, keyed ``fold<i>/<metric>`` with i counting from one, and the
        manifest entries of the folds that saved a checkpoint.
    """
    results_callback = _shared_results_callback(cfg)
    datamodule.n_folds = n_folds
    metrics: dict[str, Any] = {}
    runs: list[ManifestRun | None] = []
    for fold_idx in range(n_folds):
        label = f"fold_{fold_idx + 1}"
        log.info(f"Fold {fold_idx + 1} of {n_folds}")
        datamodule.set_fold(fold_idx)
        trainer = _run_one_fold(cfg, datamodule, results_callback, label, label)[0]
        fold_metrics = dict(trainer.callback_metrics)
        metrics.update({f"fold{fold_idx + 1}/{k}": v for k, v in fold_metrics.items()})
        runs.append(_manifest_run(trainer, label, fold_idx + 1, fold_metrics))
    return metrics, [run for run in runs if run is not None]


def _write_manifest(cfg: DictConfig, n_folds: int, runs: list[ManifestRun]) -> None:
    """Record what this training left behind, for ``src/eval.py`` to pick up.

    :param n_folds: the number of folds fitted, or 0 when only the main run was, which is what
        tells the evaluation whether it has folds to aggregate.
    """
    manifest = Manifest(
        n_folds=n_folds if n_folds >= 2 else 0,
        hydra_output_dir=str(cfg.paths.output_dir),
        experiment_name=str(cfg.get("experiment_name")),
        runs=runs,
    )
    write_manifest(manifest, cfg.paths.output_dir)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the main run, then the folds when there are any, then write the manifest.

    :param cfg: the config Hydra composed.
    :return: the metrics of all runs, and the objects of the main run.
    """
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    n_folds = int(cfg.data.get("n_folds") or 1)

    # The main run keeps the record's own train/validation split whatever n_folds asks for,
    # so the number a search reads back comes from one clean split, not a fold average.
    datamodule.n_folds = 1
    datamodule.set_fold(0)
    trainer, algorithm, callbacks, loggers = _run_one_fold(
        cfg, datamodule, None, "main", "main", cfg.get("ckpt_path")
    )
    main_metrics = dict(trainer.callback_metrics)
    metric_dict: dict[str, Any] = {f"main/{k}": v for k, v in main_metrics.items()}
    main_run = _manifest_run(trainer, "main", None, main_metrics)
    runs = [main_run] if main_run is not None else []

    if n_folds >= 2:
        fold_metrics, fold_runs = _run_folds(cfg, datamodule, n_folds)
        metric_dict.update(fold_metrics)
        runs += fold_runs
    else:
        log.info(f"No cross-validation folds, since data.n_folds={n_folds} is below 2")

    # Mirrored at the top level, where get_metric_value and the Optuna sweeper look.
    metric_dict.update(main_metrics)
    _write_manifest(cfg, n_folds, runs)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Train, and return ``cfg.optimized_metric`` for an Optuna sweep to maximise.

    :param cfg: the config Hydra composed.
    :return: the optimised metric of the main run, or ``None`` when none is configured.
    """
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
