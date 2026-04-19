from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
import omegaconf
from omegaconf import DictConfig

# PyTorch 2.6+ defaults to weights_only=True in torch.load, which rejects
# OmegaConf types saved inside Lightning checkpoints (hyperparameters).
torch.serialization.safe_globals([omegaconf.base.ContainerMetadata])
torch.serialization.add_safe_globals([omegaconf.listconfig.ListConfig])

rootutils.setup_root(__file__, dotenv=True, pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.callbacks.results_table_callback import ResultsTableCallback

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model with k-fold cross validation. Evaluates on a test set
    using the best weights from each fold.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # ------------------------------------------------------------------
    # Datamodule — instantiated ONCE, reused across folds.
    # prepare_data() is called once here to download/process data.
    # set_fold() + setup() are called per fold inside the loop.
    # ------------------------------------------------------------------
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    n_folds = cfg.data.get("n_folds", 1)
    use_kfold = n_folds > 1

    # ------------------------------------------------------------------
    # ResultsTableCallback — instantiated ONCE so it accumulates metrics
    # across all folds and writes the aggregated row after the last fold.
    # ------------------------------------------------------------------
    results_callback: Optional[ResultsTableCallback] = None
    if cfg.get("callbacks") and cfg.callbacks.get("results_table"):
        results_callback = hydra.utils.instantiate(cfg.callbacks.results_table)
        log.info("ResultsTableCallback instantiated (shared across folds).")


    # --- Main training/validation (single split) ---
    log.info("=== MAIN TRAINING (single split) ===")
    datamodule.set_fold(0)
    datamodule.setup(stage="fit")
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    if results_callback is not None:
        callbacks = [results_callback if isinstance(c, ResultsTableCallback) else c for c in callbacks]
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    object_dict = {
        "cfg":        cfg,
        "datamodule": datamodule,
        "model":      model,
        "callbacks":  callbacks,
        "logger":     logger,
        "trainer":    trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)
    metric_dict = {}
    if cfg.get("train"):
        log.info("Starting main training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if cfg.get("test"):
        log.info("Starting main testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        datamodule.setup(stage="test")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
        log.info(f"Best ckpt path: {ckpt_path}")
    metric_dict.update(trainer.callback_metrics)

    # --- K-fold uncertainty estimation (always runs) ---
    kfold_n = cfg.data.get("n_folds", 5)
    if kfold_n < 2:
        kfold_n = 5
    log.info(f"=== K-FOLD UNCERTAINTY EVALUATION (n_folds={kfold_n}) ===")
    kfold_accs = []
    for fold_idx in range(kfold_n):
        datamodule.set_fold(fold_idx)
        datamodule.setup(stage="fit")
        # New model for each fold
        kfold_model: LightningModule = hydra.utils.instantiate(cfg.model)
        kfold_callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
        kfold_logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
        kfold_trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=kfold_callbacks, logger=kfold_logger)
        kfold_trainer.fit(model=kfold_model, datamodule=datamodule)
        datamodule.setup(stage="test")
        kfold_trainer.test(model=kfold_model, datamodule=datamodule)
        acc = kfold_trainer.callback_metrics.get("test/acc")
        if acc is not None:
            kfold_accs.append(float(acc))
    if kfold_accs:
        import numpy as np
        acc_mean = np.mean(kfold_accs)
        acc_std = np.std(kfold_accs)
        log.info(f"K-fold test/acc: {acc_mean:.4f} ± {acc_std:.4f} (n={len(kfold_accs)})")
        print(f"K-fold test/acc: {acc_mean:.4f} ± {acc_std:.4f} (n={len(kfold_accs)})")
        metric_dict["kfold/test_acc_mean"] = acc_mean
        metric_dict["kfold/test_acc_std"] = acc_std

    return metric_dict, {
        "cfg":        cfg,
        "datamodule": datamodule,
        "model":      model,
        "callbacks":  callbacks,
        "logger":     logger,
        "trainer":    trainer,
    }


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)

    metric_dict, _ = train(cfg)

    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()