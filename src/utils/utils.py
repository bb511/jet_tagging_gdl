# Small helpers wrapping a Hydra task: optional startup work, failure handling, metrics.
"""Utilities shared by ``src/train.py`` and ``src/eval.py``."""

import warnings
from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from src.utils import pylogger, rich_utils

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def extras(cfg: DictConfig) -> None:
    """Apply the optional conveniences configured under ``cfg.extras``.

    Silencing warnings, prompting for tags, printing the config tree, and setting the float32
    matmul precision (``high`` trades a little accuracy for tensor cores on the V100s and above;
    ``null`` leaves torch's default alone).

    :param cfg: the composed config tree.
    """
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        rich_utils.enforce_tags(cfg, save_to_file=True)

    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        rich_utils.print_config_tree(cfg, resolve=True, save_to_file=True)

    if cfg.extras.get("matmul_precision"):
        precision = cfg.extras.get("matmul_precision")
        log.info(f"Setting matmul precision! <cfg.extras.matmul_precision={precision}>")
        torch.set_float32_matmul_precision(precision)


def task_wrapper(task_func: Callable) -> Callable:
    """Log the traceback and the output directory when the task fails, then re-raise.

    Re-raising is deliberate: an Optuna trial that dies of an invalid hyperparameter
    combination should fail loudly rather than report a silent objective.
    """

    def wrap(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as ex:
            log.exception("")
            raise ex
        finally:
            log.info(f"Output dir: {cfg.paths.output_dir}")

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict[str, Any], metric_name: str | None) -> float | None:
    """Read one metric out of a finished run's metric dict.

    :param metric_dict: the metrics a task returned, values being scalar tensors.
    :param metric_name: the metric to read; ``None`` returns ``None``, which is what a run outside
        a hyperparameter search wants.
    :raises Exception: if the metric was never logged, since an Optuna study would otherwise
        optimise nothing.
    """
    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
