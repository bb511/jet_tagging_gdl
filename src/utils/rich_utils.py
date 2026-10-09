# Rich rendering of the composed config tree, and the interactive tag prompt.

from pathlib import Path

import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


# The top-level fields the tree renders first; the rest follow in the config's order.
_PRINT_ORDER = (
    "data",
    "algorithm",
    "callbacks",
    "logger",
    "trainer",
    "paths",
    "extras",
)


def _ordered_fields(cfg: DictConfig) -> list[str]:
    """The top-level fields of *cfg* to render, those of ``_PRINT_ORDER`` first.

    A field of ``_PRINT_ORDER`` that the config does not hold is reported: the order is
    written by hand, so a renamed config group would otherwise drop out of the tree
    unnoticed.
    """
    ordered = []
    for field in _PRINT_ORDER:
        if field in cfg:
            ordered.append(field)
        else:
            log.warning(
                f"Field '{field}' not found in config. Skipping '{field}' printing..."
            )
    return ordered + [field for field in cfg if field not in ordered]


@rank_zero_only
def print_config_tree(cfg: DictConfig) -> None:
    """Render the resolved config as a Rich tree, one branch per top-level field, and
    write the tree to ``config_tree.log`` in the run directory."""
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    for field in _ordered_fields(cfg):
        node = cfg[field]
        if isinstance(node, DictConfig):
            node = OmegaConf.to_yaml(node, resolve=True)
        branch = tree.add(field, style=style, guide_style=style)
        branch.add(rich.syntax.Syntax(str(node), "yaml"))
    rich.print(tree)
    with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
        rich.print(tree, file=file)


def _prompt_for_tags() -> list[str]:
    """Ask for the tags on the command line, splitting the answer on commas.

    :raises ValueError: under a multirun, whose jobs would each stop at the prompt.
    """
    if "id" in HydraConfig().cfg.hydra.job:
        raise ValueError("Specify tags before launching a multirun!")
    log.warning("No tags provided in config. Prompting user to input tags...")
    tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
    return [tag.strip() for tag in tags.split(",") if tag != ""]


@rank_zero_only
def enforce_tags(cfg: DictConfig) -> None:
    """Prompt for tags when the config carries none, write them back into it, and write
    them to ``tags.log`` in the run directory."""
    if not cfg.get("tags"):
        with open_dict(cfg):
            cfg.tags = _prompt_for_tags()
        log.info(f"Tags: {cfg.tags}")
    with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
        rich.print(cfg.tags, file=file)
