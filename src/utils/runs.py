# The per-run objects the two entry points build on top of the composed callbacks: the shared
# results table, the callbacks of one run, and the hyperparameters recorded beside it. Kept out of
# ``src.utils``'s namespace: this module imports the callbacks, which import ``src.utils``.

from pathlib import Path

import hydra
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from src.callbacks.plotting import PlottingCallback
from src.callbacks.results_table import ResultsTableCallback
from src.utils.instantiators import instantiate_callbacks
from src.utils.logging_utils import log_hyperparameters


def shared_results_callback(cfg: DictConfig) -> ResultsTableCallback | None:
    """One results table for every run that enters it, so k folds yield one row rather than k.

    :return:``None`` when the composed callbacks hold no results table, as under
        ``callbacks=search``.
    """
    if not cfg.get("callbacks") or not cfg.callbacks.get("results_table"):
        return None
    return hydra.utils.instantiate(cfg.callbacks.results_table)


def _with_results_table(
    callbacks: list[Callback], shared: ResultsTableCallback | None, keep_composed: bool
) -> list[Callback]:
    """Put *shared* in place of the results table this run composed.

    :param keep_composed: what to do without a shared table: keep the composed one, which is how a
        training's main run writes its own row, or drop it, which is how an evaluation's main run
        is kept out of the row its folds fill.
    """
    if shared is not None:
        return [shared if isinstance(cb, ResultsTableCallback) else cb for cb in callbacks]
    if keep_composed:
        return callbacks
    return [cb for cb in callbacks if not isinstance(cb, ResultsTableCallback)]


def prepare_callbacks(
    cfg: DictConfig,
    results_callback: ResultsTableCallback | None,
    subdir: str,
    checkpoint_dir: Path | None = None,
    keep_results_table: bool = False,
) -> list[Callback]:
    """Instantiate one run's callbacks, with its figures in a subdirectory of its own.

    Without that subdirectory every fold would draw over the previous one's figures.

    :param results_callback: the shared results table, replacing the composed one.
    :param keep_results_table: what a ``None`` *results_callback* means, see
        :func:`_with_results_table`.
    :param checkpoint_dir: where ``ModelCheckpoint`` writes; ``None`` leaves it where the config
        put it, which is what an evaluation wants, having no checkpoint to write.
    """
    composed = instantiate_callbacks(cfg.get("callbacks"))
    callbacks = _with_results_table(composed, results_callback, keep_results_table)
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint) and checkpoint_dir is not None:
            callback.dirpath = str(checkpoint_dir)
        elif isinstance(callback, PlottingCallback):
            callback.subdir = subdir
    return callbacks


def log_run_hyperparameters(
    cfg: DictConfig, algorithm: LightningModule, trainer: Trainer, fold_label: str, run_stage: str
) -> None:
    """Record the config, the module and the fold under this run's entry in the loggers.

    :param fold_label: "main", or "fold_<i>", logged as ``fold_idx``.
    :param run_stage: "train" or "eval", which is what tells a training and its later evaluation
        apart in MLflow.
    """
    log_hyperparameters(
        {
            "cfg": cfg,
            "algorithm": algorithm,
            "trainer": trainer,
            "fold_idx": fold_label,
            "run_stage": run_stage,
        }
    )
