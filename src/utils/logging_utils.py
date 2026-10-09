# Choose what of a run's config and of the fitted algorithm reaches the loggers.

from typing import Any

from lightning import LightningModule
from lightning.pytorch.loggers import Logger, MLFlowLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf


def nest_under(loggers: list[Logger], parent_run_id: str | None) -> None:
    """Make the MLflow run of *loggers* a child of *parent_run_id*; ``None`` leaves it
    where it is.

    A fold's training nests under the main run's, and a checkpoint's evaluation under
    the training run that wrote it. The tag has to be in place before the run is
    created, which the logger does on its first use.
    """
    for logger in loggers:
        if parent_run_id and isinstance(logger, MLFlowLogger):
            logger.tags = {**(logger.tags or {}), "mlflow.parentRunId": parent_run_id}


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Send the config, the parameter counts and the FLOP and eBOP counts to every
    attached logger.

    :param object_dict: holds ``cfg``, ``algorithm`` and ``trainer``; an evaluation adds
        the runtime-only ``fold_idx``, which a training reads off the config.
    """
    cfg, algorithm = object_dict["cfg"], object_dict["algorithm"]
    hparams = _config_hparams(cfg, object_dict.get("fold_idx"))
    hparams.update(_algorithm_hparams(cfg, algorithm))
    for logger in object_dict["trainer"].loggers:
        logger.log_hyperparams(hparams)


def _resolved(node: DictConfig | ListConfig | None) -> dict | list | None:
    """*node* as plain containers, its interpolations resolved.

    The config is resolved node by node, never as a whole: a composition with
    ``return_hydra_config`` leaves ``hydra.sweep.subdir`` at ``${hydra.job.num}``, which
    has no value.
    """
    return None if node is None else OmegaConf.to_container(node, resolve=True)


def _config_hparams(cfg: DictConfig, fold_idx: str | None) -> dict[str, Any]:
    """The parts of the composed config worth keeping beside a run."""
    return {
        "data": _resolved(cfg.data),
        "trainer": _resolved(cfg.trainer),
        "callbacks": _resolved(cfg.get("callbacks")),
        "extras": _resolved(cfg.get("extras")),
        "task_name": cfg.get("task_name"),
        "tags": _resolved(cfg.get("tags")),
        "ckpt_path": cfg.get("ckpt_path"),
        "seed": cfg.get("seed"),
        "experiment_name": cfg.get("experiment_name"),
        "run_name": cfg.get("run_name"),
        "fold_idx": cfg.get("run_label") if fold_idx is None else fold_idx,
        "run_stage": cfg.get("run_stage"),
    }


def _algorithm_hparams(cfg: DictConfig, algorithm: LightningModule) -> dict[str, Any]:
    """The algorithm's config, its parameter counts and its per-jet FLOP and eBOP
    counts.

    The counts are computed at construction, so they are read off the module, not the
    config.
    """
    parameters = list(algorithm.parameters())
    return {
        "algorithm": _resolved(cfg.algorithm),
        "algorithm/name": cfg.algorithm.get("name"),
        "algorithm/params/total": sum(p.numel() for p in parameters),
        "algorithm/params/trainable": sum(
            p.numel() for p in parameters if p.requires_grad
        ),
        "algorithm/params/non_trainable": sum(
            p.numel() for p in parameters if not p.requires_grad
        ),
        "algorithm/flops": algorithm.flops_total,
        "algorithm/ebops": algorithm.ebops_total,
        "algorithm/ebops_unpriced": algorithm.ebops_unpriced,
    }
