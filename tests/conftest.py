"""Fixtures composing the project's own Hydra roots, cut down to one cheap CPU epoch.

The record gate is read here rather than in each test module because importing ``src.train`` runs
``rootutils.setup_root(dotenv=True)``, which copies ``.env``'s ``HF_HUB_CACHE`` into the
environment: a gate evaluated after that import would never skip, and the data tests would start a
3 GB download.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

from src.utils.omegaconf import register_resolvers

# The configs interpolate ${len:} and ${short_hash:}, so nothing composes without these.
register_resolvers()

ROOT = rootutils.find_root(indicator=".project-root")

# configs/paths/default.yaml resolves against it, and a bare pytest has neither .env nor the
# variable.
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

# The record's per-nconstituents caches survive between runs: rebuilding one costs more than every
# test in this suite together.
CACHE_DIR = os.environ.get("HLS4ML_CACHE_DIR", str(ROOT / "data" / "hf"))

needs_record = pytest.mark.skipif(
    "HF_HUB_CACHE" not in os.environ,
    reason="Export HF_HUB_CACHE to the record's hub snapshot to run the data tests.",
)


def _composed(config_name: str) -> DictConfig:
    """Compose train.yaml or eval.yaml down to what a test can afford, Hydra node included."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name=config_name, return_hydra_config=True, overrides=[])

    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT)
        cfg.data.cache_dir = CACHE_DIR
        cfg.data.n_folds = 1
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.trainer.accelerator = "cpu"
        cfg.trainer.devices = 1
        cfg.trainer.max_epochs = 1
        cfg.trainer.limit_train_batches = 0.01
        cfg.trainer.limit_val_batches = 0.05
        cfg.trainer.limit_test_batches = 0.05
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = None

    return cfg


def _per_test(cfg: DictConfig, tmp_path: Path) -> Iterator[DictConfig]:
    """Point every output of *cfg* inside *tmp_path*, and leave Hydra clean afterwards."""
    cfg = cfg.copy()
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path / "results")
        # run_name interpolates ${hydra:runtime.output_dir}, which resolves only against a live
        # HydraConfig; a test that needs it calls HydraConfig().set_config(cfg).
        cfg.hydra.runtime.output_dir = str(tmp_path)

    yield cfg

    GlobalHydra.instance().clear()


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    """The training root, composed once for the whole package."""
    return _composed("train.yaml")


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    """The evaluation root, composed once for the whole package."""
    return _composed("eval.yaml")


@pytest.fixture
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> Iterator[DictConfig]:
    """A training config writing everything under this test's own directory."""
    yield from _per_test(cfg_train_global, tmp_path)


@pytest.fixture
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> Iterator[DictConfig]:
    """An evaluation config writing everything under this test's own directory."""
    yield from _per_test(cfg_eval_global, tmp_path)
