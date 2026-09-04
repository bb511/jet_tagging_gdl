"""Fit one architecture on the hls4ml jet data set.

The main run keeps the record's own train/validation split; ``data.n_folds >= 2`` adds one run per
fold over the pooled train and validation jets, which is where the quoted uncertainty comes from.
``fold=<i>`` fits that one fold in place of the main run and the loop, so the k folds can run as k
processes, each in a run directory of its own, whose manifests ``src/eval.py`` merges. Metrics come
back under ``main/``, per fold under ``fold<i>/``, and once more at the top level, which is what an
Optuna trial reads. Nothing is tested here: the run writes a checkpoint manifest that
``src/eval.py`` picks up.
"""

from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${len:} and ${short_hash:}.
register_resolvers()

from src.callbacks.results_table import ResultsTableCallback
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_loggers,
    task_wrapper,
)
from src.utils.checkpoints import (
    Manifest,
    ManifestRun,
    best_or_last_checkpoint,
    write_manifest,
)
from src.utils.runs import (
    log_run_hyperparameters,
    prepare_callbacks,
    shared_results_callback,
)

log = RankedLogger(__name__, rank_zero_only=True)

# What one fit leaves behind: the trainer, the fitted algorithm, its callbacks and its loggers.
Fitted = tuple[Trainer, LightningModule, list[Callback], list[Logger]]


def _fit_objects(
    cfg: DictConfig, results_callback: ResultsTableCallback | None, checkpoint_subdir: str
) -> tuple[list[Callback], list[Logger]]:
    """The callbacks and the loggers of one fit, its checkpoints under *checkpoint_subdir*.

    LearningRateMonitor is dropped when nothing is logging, since it raises at ``on_train_start``
    rather than warning, and searches and debug runs compose the callbacks with ``logger=none``.
    """
    checkpoint_dir = Path(cfg.paths.output_dir) / "checkpoints" / checkpoint_subdir
    callbacks = prepare_callbacks(
        cfg, results_callback, checkpoint_subdir, checkpoint_dir, keep_results_table=True
    )
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))
    if not loggers:
        callbacks = [cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)]
    return callbacks, loggers


def _fit_run(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    results_callback: ResultsTableCallback | None,
    fold_label: str,
    checkpoint_subdir: str,
    ckpt_path: str | None = None,
) -> Fitted:
    """Fit one model on whichever split *datamodule* currently holds.

    The datamodule is not rebuilt between folds: the pool is assembled once and ``set_fold`` alone
    moves the split.

    :param fold_label: names the run in the logs and in the manifest, "main" or "fold_<i>".
    :return: the trainer, the fitted algorithm, its callbacks and its loggers.
    """
    log.info(f"[{fold_label}] Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)
    callbacks, loggers = _fit_objects(cfg, results_callback, checkpoint_subdir)
    log.info(f"[{fold_label}] Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)
    if loggers:
        log_run_hyperparameters(cfg, algorithm, trainer, fold_label, "train")
    log.info(f"[{fold_label}] Starting training")
    trainer.fit(model=algorithm, datamodule=datamodule, ckpt_path=ckpt_path)
    return trainer, algorithm, callbacks, loggers


def _manifest_run(
    trainer: Trainer, fold_label: str, fold_idx: int | None, metrics: dict[str, Any]
) -> ManifestRun | None:
    """The manifest entry of a finished run, or ``None`` when it saved no checkpoint.

    A run composed without model_checkpoint trains normally but leaves nothing for ``src/eval.py``
    to load, which is worth a warning rather than a failure.
    """
    try:
        checkpoint_path = best_or_last_checkpoint(trainer)
    except RuntimeError as error:
        log.warning(f"[{fold_label}] {error}")
        return None
    val_acc_best = metrics.get("val/acc_best")
    best = None if val_acc_best is None else float(val_acc_best)
    return ManifestRun(fold_label, fold_idx, checkpoint_path, best)


def _fit_fold(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    results_callback: ResultsTableCallback | None,
    fold_idx: int,
    n_folds: int,
) -> tuple[dict[str, Any], ManifestRun | None]:
    """Fit fold *fold_idx*, counting from zero, of the k-fold loop.

    :return: its metrics keyed ``fold<i>/<metric>`` with i counting from one, and its manifest
        entry when it saved a checkpoint.
    """
    label = f"fold_{fold_idx + 1}"
    log.info(f"Fold {fold_idx + 1} of {n_folds}")
    datamodule.set_fold(fold_idx)
    trainer = _fit_run(cfg, datamodule, results_callback, label, label)[0]
    fold_metrics = dict(trainer.callback_metrics)
    metrics = {f"fold{fold_idx + 1}/{key}": value for key, value in fold_metrics.items()}
    return metrics, _manifest_run(trainer, label, fold_idx + 1, fold_metrics)


def _run_folds(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int
) -> tuple[dict[str, Any], list[ManifestRun]]:
    """Fit one model per fold over the pooled train and validation jets.

    :return: metrics keyed ``fold<i>/<metric>`` with i counting from one, and the manifest entries
        of the folds that saved a checkpoint.
    """
    results_callback = shared_results_callback(cfg)
    datamodule.n_folds = n_folds
    metrics: dict[str, Any] = {}
    runs: list[ManifestRun] = []
    for fold_idx in range(n_folds):
        fold_metrics, run = _fit_fold(cfg, datamodule, results_callback, fold_idx, n_folds)
        metrics.update(fold_metrics)
        if run is not None:
            runs.append(run)
    return metrics, runs


def _fold_to_train(cfg: DictConfig, n_folds: int) -> int | None:
    """The one fold this process fits, or ``None`` for the main run and every fold in turn.

    :raises ValueError: when ``fold`` is set with no folds to pick from, or names one past them.
    """
    fold = cfg.get("fold")
    if fold is None:
        return None
    if n_folds < 2 or not 1 <= int(fold) <= n_folds:
        raise ValueError(
            f"fold={fold} needs data.n_folds >= 2 and 1 <= fold <= data.n_folds, "
            f"but data.n_folds={n_folds}."
        )
    return int(fold)


def _run_single_fold(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int, fold: int
) -> tuple[dict[str, Any], list[ManifestRun], Fitted]:
    """Fit fold *fold* of *n_folds* alone, in a process that owns no other run.

    The main run is skipped: a sibling process launched with ``fold=null`` and ``data.n_folds=1``
    supplies it, and :func:`src.utils.checkpoints.merge_manifests` joins the manifests at
    evaluation time. ``ckpt_path`` resumes this fold, there being no other run it could mean.

    :return: metrics keyed ``fold<i>/<metric>`` and mirrored at the top level, where
        ``get_metric_value`` looks; the manifest entry, when a checkpoint was saved; the objects.
    """
    label = f"fold_{fold}"
    log.info(f"Fold {fold} of {n_folds}, alone in this process")
    datamodule.n_folds = n_folds
    datamodule.set_fold(fold - 1)
    fitted = _fit_run(cfg, datamodule, None, label, label, cfg.get("ckpt_path"))
    fold_metrics = dict(fitted[0].callback_metrics)
    metrics = {f"fold{fold}/{key}": value for key, value in fold_metrics.items()}
    metrics.update(fold_metrics)
    run = _manifest_run(fitted[0], label, fold, fold_metrics)
    return metrics, [run] if run is not None else [], fitted


def _folds_beside_main(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int
) -> tuple[dict[str, Any], list[ManifestRun]]:
    """The folds fitted after the main run, none of them below two."""
    if n_folds < 2:
        log.info(f"No cross-validation folds, since data.n_folds={n_folds} is below 2")
        return {}, []
    return _run_folds(cfg, datamodule, n_folds)


def _run_main_and_folds(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int
) -> tuple[dict[str, Any], list[ManifestRun], Fitted]:
    """Fit the main run on the record's own split, then every fold in turn.

    :return: metrics keyed ``main/<metric>`` and ``fold<i>/<metric>``, the main run's mirrored at
        the top level, where ``get_metric_value`` and the Optuna sweeper look; the manifest
        entries; and the main run's objects.
    """
    # The main run keeps the record's own split: a search then reads one clean number, not a mean.
    datamodule.n_folds = 1
    datamodule.set_fold(0)
    fitted = _fit_run(cfg, datamodule, None, "main", "main", cfg.get("ckpt_path"))
    main_metrics = dict(fitted[0].callback_metrics)
    main_run = _manifest_run(fitted[0], "main", None, main_metrics)
    fold_metrics, fold_runs = _folds_beside_main(cfg, datamodule, n_folds)
    prefixed = {f"main/{key}": value for key, value in main_metrics.items()}
    runs = ([main_run] if main_run is not None else []) + fold_runs
    return prefixed | fold_metrics | main_metrics, runs, fitted


def _write_manifest(cfg: DictConfig, n_folds: int, runs: list[ManifestRun]) -> None:
    """Record what this training left behind, for ``src/eval.py`` to pick up.

    :param n_folds: 0 when only the main run was fitted, which is how the evaluation knows it has
        no folds to aggregate.
    """
    manifest = Manifest(
        n_folds=n_folds if n_folds >= 2 else 0,
        hydra_output_dir=str(cfg.paths.output_dir),
        experiment_name=str(cfg.get("experiment_name")),
        runs=runs,
    )
    write_manifest(manifest, cfg.paths.output_dir)


def _fit_runs(
    cfg: DictConfig, datamodule: LightningDataModule, n_folds: int, fold: int | None
) -> tuple[dict[str, Any], list[ManifestRun], Fitted]:
    """The main run with its folds, or the single fold *fold* names."""
    if fold is None:
        return _run_main_and_folds(cfg, datamodule, n_folds)
    return _run_single_fold(cfg, datamodule, n_folds, fold)


def _object_dict(
    cfg: DictConfig, datamodule: LightningDataModule, fitted: Fitted
) -> dict[str, Any]:
    """The objects of the run a caller is handed back, the main run's or the single fold's."""
    trainer, algorithm, callbacks, loggers = fitted
    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the main run and then the folds, or one fold alone under ``fold``, then write the
    manifest.

    :return: the metrics of all runs, and the objects of the main run, or of the one fold.
    """
    n_folds = int(cfg.data.get("n_folds") or 1)
    fold = _fold_to_train(cfg, n_folds)
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()
    metric_dict, runs, fitted = _fit_runs(cfg, datamodule, n_folds, fold)
    _write_manifest(cfg, n_folds, runs)
    return metric_dict, _object_dict(cfg, datamodule, fitted)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Train, and return the main run's ``cfg.optimized_metric`` for Optuna to maximise."""
    extras(cfg)
    metric_dict, _ = train(cfg)
    return get_metric_value(metric_dict, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
