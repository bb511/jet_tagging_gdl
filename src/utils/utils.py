"""Startup work, failure handling and metric retrieval, shared by both entry points."""

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Apply the optional conveniences configured under ``cfg.extras``.

    Tag prompting, the config tree, the float32 matmul precision (``high`` trades a
    little accuracy for tensor cores on Volta and later; ``null`` leaves torch alone)
    and the per-process GPU memory cap (a fraction of the card; ``null`` leaves torch
    alone).
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg)

    if cfg.extras.get("matmul_precision"):
        precision = cfg.extras.get("matmul_precision")
        log.info(f"Setting matmul precision! <cfg.extras.matmul_precision={precision}>")
        torch.set_float32_matmul_precision(precision)

    if cfg.extras.get("gpu_memory_fraction") and torch.cuda.is_available():
        fraction = float(cfg.extras.get("gpu_memory_fraction"))
        log.info(f"Capping GPU memory! <cfg.extras.gpu_memory_fraction={fraction}>")
        torch.cuda.set_per_process_memory_fraction(fraction)


def task_wrapper(task_func: Callable) -> Callable:
    """Log the traceback and the output directory when the task fails, then re-raise.

    Re-raising is deliberate: an Optuna trial must fail loudly rather than report a silent
    objective.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            return task_func(cfg=cfg)
        except Exception:
            log.exception("")
            raise
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str) -> float:
    """Read one scalar metric out of a finished run's metrics.

    :raises KeyError: when the run never logged *metric_name*, which a study would
        otherwise optimise blindly; the message names both places the name has to match,
        the module and the search config.
    """
    if metric_name not in metric_dict:
        raise KeyError(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )
    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")
    return metric_value
