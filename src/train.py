import json
from pathlib import Path
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
torch.serialization.add_safe_globals([
    omegaconf.base.ContainerMetadata,
    omegaconf.listconfig.ListConfig,
])

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


def _get_last_ckpt(trainer: Trainer) -> Optional[str]:
    """Safely retrieve the best checkpoint path, falling back to last."""
    # Fallback: use last saved checkpoint
    last = trainer.checkpoint_callback.last_model_path
    log.info(f"Saving last checkpoint: {last}")
    return last


def _run_one_fold(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    results_callback: Optional[ResultsTableCallback],
    fold_label: str,
    checkpoint_subdir: str,
    ckpt_path: Optional[str] = None,
    use_results_callback: bool = True,
    run_stage: Optional[str] = None,
) -> Tuple[Trainer, LightningModule, Dict]:
    """Instantiate a fresh model/trainer/logger and run fit + test for one fold."""

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Optionally disable results callback for this run (e.g. main run when
    # doing separate K-fold uncertainty evaluation).
    if not use_results_callback:
        callbacks = [c for c in callbacks if not isinstance(c, ResultsTableCallback)]
    # Replace the per-run ResultsTableCallback instance with the shared one
    # so it accumulates metrics across all folds.
    elif results_callback is not None:
        callbacks = [
            results_callback if isinstance(c, ResultsTableCallback) else c
            for c in callbacks
        ]

    # Route each run's checkpoints into a dedicated directory
    # (e.g. checkpoints/main, checkpoints/fold_1, ...).
    base_ckpt_dir = Path(cfg.paths.output_dir) / "checkpoints"
    run_ckpt_dir = str(base_ckpt_dir / checkpoint_subdir)
    for cb in callbacks:
        if hasattr(cb, "dirpath"):
            cb.dirpath = run_ckpt_dir

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg, "datamodule": datamodule, "model": model,
        "callbacks": callbacks, "logger": logger, "trainer": trainer,
        "fold_idx": fold_label, "run_stage": run_stage,
    }
    if logger:
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info(f"[{fold_label}] Starting training...")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        last_ckpt = _get_last_ckpt(trainer)
        log.info(f"[{fold_label}] Best checkpoint path: {last_ckpt}")
        

    # if cfg.get("test"):
    #     log.info(f"[{fold_label}] Starting testing...")
    #     last_ckpt = _get_last_ckpt(trainer)
    #     log.info(f"[{fold_label}] Loading checkpoint: {last_ckpt}")
    #     datamodule.setup(stage="test")
    #     trainer.test(
    #         model=model,
    #         datamodule=datamodule,
    #         ckpt_path=last_ckpt,
    #         weights_only=False,
    #     )

    return trainer, model, callbacks, logger


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model.

    Structure:
        1. Main training run  — uses the full processed train/val split from disk
                                (n_folds forced to 1, no KFold splitting)
        2. Optional K-fold runs — if ``data.n_folds >= 2``, run K folds over the
                                same pool for uncertainty estimation.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    separate_mode: bool = bool(cfg.get("separate_train_eval_mode", False))
    run_stage: Optional[str] = "train" if separate_mode else None

    # ------------------------------------------------------------------
    # Datamodule — instantiated ONCE, reused across all runs.
    # prepare_data() downloads/processes data once.
    # ------------------------------------------------------------------
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.prepare_data()

    # Resolve k-fold setting from data config.
    kfold_n_cfg = cfg.data.get("n_folds", 1)
    kfold_n = int(kfold_n_cfg) if kfold_n_cfg is not None else 1
    run_kfold = kfold_n >= 2

    # ------------------------------------------------------------------
    # ResultsTableCallback:
    # - if run_kfold=True: use ONE shared instance across all folds
    # - if run_kfold=False: keep the regular callback in the main run
    # ------------------------------------------------------------------
    results_callback: Optional[ResultsTableCallback] = None
    if run_kfold and cfg.get("callbacks") and cfg.callbacks.get("results_table"):
        results_callback = hydra.utils.instantiate(cfg.callbacks.results_table)
        log.info("ResultsTableCallback instantiated (shared across k-fold runs).")

    metric_dict = {}

    # ==================================================================
    # RUN 0 — Main training on the full disk split (no KFold)
    # Force n_folds=1 on the datamodule so setup() uses train/val as-is
    # ==================================================================
    log.info("=" * 60)
    log.info("  RUN 0 — Main training (full single split)")
    log.info("=" * 60)

    datamodule.n_folds = 1      # disable KFold splitting for this run
    datamodule.set_fold(0)      # resets data_train/val/test references
    datamodule.setup(stage="fit")
    log.info(
        f"  Train: {datamodule.n_train_samples}  "
        f"Val: {datamodule.n_val_samples}"
    )

    main_trainer, main_model, main_callbacks, main_logger = _run_one_fold(
        cfg=cfg,
        datamodule=datamodule,
        results_callback=None,          # use default per-run callback handling
        fold_label="main",
        checkpoint_subdir="main",
        ckpt_path=cfg.get("ckpt_path"),
        use_results_callback=not run_kfold,
        run_stage=run_stage,
    )
    metric_dict.update({
        f"main/{k}": v for k, v in main_trainer.callback_metrics.items()
    })

    # Collect best checkpoint path for manifest
    manifest_runs: List[Dict] = []
    main_ckpt = _get_last_ckpt(main_trainer)
    manifest_runs.append({"fold_label": "main", "fold_idx": 0, "checkpoint_path": main_ckpt})

    # ==================================================================
    # RUNS 1..K — Optional K-fold cross validation for uncertainty estimation
    # ==================================================================
    if not run_kfold:
        log.info(
            "Skipping K-fold uncertainty evaluation because "
            f"data.n_folds={kfold_n} (<2)."
        )
    else:
        log.info("=" * 60)
        log.info(f"  K-FOLD UNCERTAINTY EVALUATION (n_folds={kfold_n})")
        log.info("=" * 60)

        datamodule.n_folds = kfold_n    # re-enable KFold splitting

    last_trainer, last_model, last_callbacks, last_logger = None, None, None, None

    for fold_idx in range(kfold_n if run_kfold else 0):
        log.info("=" * 60)
        log.info(f"  FOLD {fold_idx + 1}/{kfold_n}")
        log.info("=" * 60)

        datamodule.set_fold(fold_idx)
        datamodule.setup(stage="fit")
        log.info(
            f"  Train: {datamodule.n_train_samples}  "
            f"Val: {datamodule.n_val_samples}"
        )

        fold_trainer, fold_model, fold_callbacks, fold_logger = _run_one_fold(
            cfg=cfg,
            datamodule=datamodule,
            results_callback=results_callback,  # shared — accumulates across folds
            fold_label=f"fold {fold_idx + 1}/{kfold_n}",
            checkpoint_subdir=f"fold_{fold_idx + 1}",
            ckpt_path=None,                     # each fold starts fresh
            use_results_callback=True,
            run_stage=run_stage,
        )

        # Namespace fold metrics to avoid collisions
        metric_dict.update({
            f"fold{fold_idx + 1}/{k}": v
            for k, v in fold_trainer.callback_metrics.items()
        })

        fold_ckpt = _get_last_ckpt(fold_trainer)
        manifest_runs.append({
            "fold_label": f"fold_{fold_idx + 1}",
            "fold_idx": fold_idx + 1,
            "checkpoint_path": fold_ckpt,
        })

        last_trainer   = fold_trainer
        last_model     = fold_model
        last_callbacks = fold_callbacks
        last_logger    = fold_logger
        
    # Expose the final fold's raw metrics at the top level so Hydra's
    # optimized_metric (e.g. val/acc_best) still works for multirun sweeps.
    # If K-fold is disabled, expose main-run metrics instead.
    if last_trainer is not None:
        metric_dict.update(last_trainer.callback_metrics)
    else:
        metric_dict.update(main_trainer.callback_metrics)

    # ------------------------------------------------------------------
    # Write checkpoint manifest so a separate eval pass can discover all
    # fold checkpoints without relying on callback state.
    # Written unconditionally (harmless in non-separated mode too).
    # ------------------------------------------------------------------
    manifest = {
        "n_folds": kfold_n if run_kfold else 0,
        "include_main": True,
        "separate_train_eval_mode": separate_mode,
        "hydra_output_dir": str(cfg.paths.output_dir),
        "runs": manifest_runs,
    }
    manifest_path = Path(cfg.paths.output_dir) / "checkpoint_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info(f"Checkpoint manifest written to {manifest_path}")

    return metric_dict, {
        "cfg":        cfg,
        "datamodule": datamodule,
        "model":      last_model if last_model is not None else main_model,
        "callbacks":  last_callbacks if last_callbacks is not None else main_callbacks,
        "logger":     last_logger if last_logger is not None else main_logger,
        "trainer":    last_trainer if last_trainer is not None else main_trainer,
    }




@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training."""
    extras(cfg)
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value
if __name__ == "__main__":
    main()
