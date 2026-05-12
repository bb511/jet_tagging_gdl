import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import rootutils
import torch
import omegaconf
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig  # noqa: F401 (open_dict removed; keep for type hints)

# PyTorch 2.6+ defaults weights_only=True in torch.load, which rejects
# OmegaConf types saved inside Lightning checkpoints (hyperparameters).
torch.serialization.add_safe_globals([
    omegaconf.base.ContainerMetadata,
    omegaconf.listconfig.ListConfig,
])

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _eval_one_checkpoint(
    cfg: DictConfig,
    ckpt_path: str,
    fold_label: str,
    run_stage: str = "eval",
) -> Dict[str, Any]:
    """Run trainer.test for a single checkpoint and return callback_metrics."""
    log.info(f"[{fold_label}] Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"[{fold_label}] Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"[{fold_label}] Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"[{fold_label}] Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "fold_idx": fold_label,
        "run_stage": run_stage,
    }

    if logger:
        log.info(f"[{fold_label}] Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info(f"[{fold_label}] Starting testing with checkpoint: {ckpt_path}")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    return dict(trainer.callback_metrics)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates checkpoint(s) on the test set.

    Two modes:
    - Manifest mode: reads ``checkpoint_manifest_path`` (JSON written by
      train.py) and evaluates main + all folds in sequence. Each fold gets its
      own logger run tagged with fold_idx and run_stage=eval.
    - Legacy mode: evaluates a single checkpoint at ``cfg.ckpt_path``.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    manifest_path: Optional[str] = cfg.get("checkpoint_manifest_path")

    # ------------------------------------------------------------------
    # Manifest mode — iterate all folds from training manifest
    # ------------------------------------------------------------------
    if manifest_path:
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"checkpoint_manifest_path does not exist: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        runs = manifest.get("runs", [])
        log.info(
            f"Manifest mode: evaluating {len(runs)} run(s) from {manifest_path}"
        )

        all_metrics: Dict[str, Any] = {}
        last_object_dict: Dict[str, Any] = {}

        for run_info in runs:
            fold_label: str = run_info["fold_label"]
            ckpt: str = run_info["checkpoint_path"]

            if not ckpt or not Path(ckpt).exists():
                log.warning(f"[{fold_label}] Checkpoint not found, skipping: {ckpt}")
                continue

            metrics = _eval_one_checkpoint(
                cfg=cfg,
                ckpt_path=ckpt,
                fold_label=fold_label,
                run_stage=cfg.get("run_stage", "eval"),
            )
            all_metrics.update({f"{fold_label}/{k}": v for k, v in metrics.items()})

        return all_metrics, last_object_dict

    # ------------------------------------------------------------------
    # Legacy mode — single checkpoint
    # ------------------------------------------------------------------
    assert cfg.ckpt_path, (
        "Provide either ckpt_path (legacy mode) or checkpoint_manifest_path (manifest mode)."
    )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
        "run_stage": cfg.get("run_stage", "eval"),
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
