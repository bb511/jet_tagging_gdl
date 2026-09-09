# Run the training.

import gc
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger, MLFlowLogger
from omegaconf import DictConfig

# Before any src import: .env carries PROJECT_ROOT and HF_HUB_CACHE, both read at import time.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from src.utils.omegaconf import register_resolvers

# Before Hydra composes: the configs interpolate ${len:}, ${short_hash:} and ${fold_label:}.
register_resolvers()

from src.callbacks.results_table import ResultsTableCallback
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    nest_under,
    task_wrapper,
)
from src.utils.checkpoints import (
    Manifest,
    ManifestRun,
    best_or_last_checkpoint,
    write_manifest,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _fold_kwargs(cfg: DictConfig) -> dict[str, int]:
    """Configure the data fold to train on (or the splits if k-folding not used).

    :raises ValueError: when ``fold`` is set with no folds to pick from, or is oob.
    """
    n_folds, fold = int(cfg.data.get("n_folds") or 1), cfg.get("fold")
    if fold is None:
        return {"n_folds": 1, "fold_idx": 0}
    if n_folds < 2 or not 1 <= int(fold) <= n_folds:
        raise ValueError(
            f"fold={fold} needs data.n_folds >= 2 and 1 <= fold <= data.n_folds, "
            f"but data.n_folds={n_folds}."
        )
    return {"n_folds": n_folds, "fold_idx": int(fold) - 1}


def _manifest_run(cfg: DictConfig, trainer: Trainer) -> ManifestRun | None:
    """This run's manifest entry, or ``None`` when it saved no checkpoint.

    ``src/eval.py`` depends on loading checkpoints, so if no checkpoints are produced
    then no evaluation can be made.
    """
    try:
        checkpoint_path = best_or_last_checkpoint(trainer)
    except RuntimeError as error:
        log.warning(str(error))
        return None
    fold, best = cfg.get("fold"), trainer.callback_metrics.get("val/acc_best")
    mlflow_run_id = next(
        (lg.run_id for lg in trainer.loggers if isinstance(lg, MLFlowLogger)), None
    )
    return ManifestRun(
        fold_label=cfg.run_label,
        fold_idx=None if fold is None else int(fold),
        checkpoint_path=checkpoint_path,
        val_acc_best=None if best is None else float(best),
        mlflow_run_id=mlflow_run_id,
    )


def _write_manifest(cfg: DictConfig, trainer: Trainer) -> None:
    """Write out the results of the training for ``src/eval.py`` to pick up."""
    run = _manifest_run(cfg, trainer)
    manifest = Manifest(
        n_folds=0 if cfg.get("fold") is None else int(cfg.data.n_folds),
        hydra_output_dir=str(cfg.paths.output_dir),
        experiment_name=str(cfg.get("experiment_name")),
        runs=[] if run is None else [run],
    )
    write_manifest(manifest, cfg.paths.output_dir)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit one run and write its manifest: the main run on the record's own split, or the fold
    ``fold`` names, nested under ``parent_run_id`` in MLflow.

    :return: the run's metrics, and the objects it was built from.
    """
    fold_kwargs = _fold_kwargs(cfg)
    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, **fold_kwargs)

    log.info(f"Instantiating algorithm <{cfg.algorithm._target_}>")
    algorithm: LightningModule = hydra.utils.instantiate(cfg.algorithm)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    # The results table writes its row at test end, in src/eval.py; here it would only log.
    callbacks = [cb for cb in callbacks if not isinstance(cb, ResultsTableCallback)]

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    nest_under(logger, cfg.get("parent_run_id"))
    if not logger:
        # LearningRateMonitor raises, rather than warns, with nothing to log to (the searches).
        callbacks = [cb for cb in callbacks if not isinstance(cb, LearningRateMonitor)]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "algorithm": algorithm,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    trainer.fit(model=algorithm, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    _write_manifest(cfg, trainer)
    return dict(trainer.callback_metrics), object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Train, and return ``cfg.optimized_metric`` for Optuna to maximise."""
    extras(cfg)
    metric_dict, object_dict = train(cfg)
    value = get_metric_value(metric_dict, cfg.get("optimized_metric"))
    # A sweep runs its trials in one process, and the trainer, the module and the datamodule
    # reference one another, so a finished trial's split tensors are cyclic garbage that only a
    # collection frees: CPython's own trigger counts allocations, not bytes, and a driver of the
    # 150-constituent studies grows by a dataset per trial without this.
    del metric_dict, object_dict
    gc.collect()
    return value


if __name__ == "__main__":
    main()
