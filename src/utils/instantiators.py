# The callbacks and loggers a composed config names, instantiated.


import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


def instantiate_group(cfg: DictConfig | None, kind: str) -> list[Callback | Logger]:
    """Instantiate every entry of one config node that carries a ``_target_``.

    :param kind: names the node in the log messages, "callback" or "logger".
    """
    if not cfg:
        log.warning(f"No {kind} configs found! Skipping...")
        return []
    objects = []
    for entry in cfg.values():
        if isinstance(entry, DictConfig) and "_target_" in entry:
            log.info(f"Instantiating {kind} <{entry._target_}>")
            objects.append(hydra.utils.instantiate(entry))
    return objects


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Callback]:
    """Instantiate every entry of the ``callbacks`` node that carries a ``_target_``."""
    return instantiate_group(callbacks_cfg, "callback")


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Logger]:
    """Instantiate every entry of the ``logger`` node that carries a ``_target_``.

    ``logger=none`` composes an empty node, and so yields no loggers at all.
    """
    return instantiate_group(logger_cfg, "logger")
