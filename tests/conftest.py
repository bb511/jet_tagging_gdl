import os
from pathlib import Path
from typing import Any

import pytest
import rootutils
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.omegaconf import register_resolvers

# The configs interpolate ${len:} and ${short_hash:}, so nothing composes without these.
register_resolvers()

ROOT = rootutils.find_root(indicator=".project-root")
# configs/paths/default.yaml reads it, and a bare pytest has neither .env nor the variable.
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
# The record's caches survive between runs; rebuilding one costs more than the whole suite.
CACHE_DIR = os.environ.get("HLS4ML_CACHE_DIR", str(ROOT / "data" / "hf"))

# Read here rather than in the data tests: importing src.train runs rootutils.setup_root with
# dotenv=True, which copies .env's HF_HUB_CACHE into the environment, and a gate read after that
# would never skip.
needs_record = pytest.mark.skipif(
    "HF_HUB_CACHE" not in os.environ,
    reason="Export HF_HUB_CACHE to the record's hub snapshot to run the data tests.",
)

MODELS = (
    "deepsets",
    "mlp",
    "jetformer",
    "lorentznet",
    "pelican",
    "pelican_lorentz",
    "lgatr_slim",
    "lgatr_slim_lorentz",
    "lloca_lorentz",
    "lloca_z",
    "lloca_so2",
    "lloca_so3",
)
# The symmetry-embedded class, trained in the baseline regime alone and never under the SCL.
EMBEDDED = (
    "lorentznet",
    "pelican",
    "pelican_lorentz",
    "lgatr_slim",
    "lgatr_slim_lorentz",
    "lloca_lorentz",
    "lloca_z",
    "lloca_so2",
    "lloca_so3",
)
NCONSTITUENTS = (16, 50, 150)
# The record's 16 constituent columns, in its own order.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
FOUR_MOMENTUM = ["px", "py", "pz", "e"]
# In GeV. A massless constituent puts m^2 = E^2 - |p|^2 at the float32 rounding of E^2, and the
# boost tests would then measure the arithmetic rather than the transform.
MOMENTUM_SCALE = 20.0
MASS = 20.0


class StubDataModule:
    feature_names = RECORD_COLUMNS
    norm_scales = None

    def __init__(
        self,
        features: list[str] = FOUR_MOMENTUM,
        keep_full_features_at_test: bool = False,
    ) -> None:
        self.features = list(features)
        self.keep_full_features_at_test = keep_full_features_at_test

    def feature_indices(self) -> None:
        return None


def jets(
    n_jets: int = 6,
    n_slots: int = 8,
    n_real: int = 5,
    n_columns: int = 4,
    mass: float = MASS,
    seed: int = 0,
) -> torch.Tensor:
    # Momenta in the first three columns, the energy in the fourth, padded slots all-zero.
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * MOMENTUM_SCALE
    batch = torch.zeros(n_jets, n_slots, n_columns)
    batch[..., :3] = momenta
    batch[..., 3] = torch.sqrt((momenta**2).sum(-1) + mass**2)
    batch[:, n_real:] = 0.0
    return batch


def labels(n_jets: int, seed: int = 1) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.eye(5)[torch.randint(0, 5, (n_jets,), generator=generator)]


def compose_root(config_name: str, *overrides: str) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name=config_name, return_hydra_config=True, overrides=list(overrides)
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT)
    return cfg


def point_into(cfg: DictConfig, tmp_path: Path) -> DictConfig:
    # HydraConfig().set_config makes the hydra node read-only, and a copy taken afterwards keeps
    # that, so a second run's copy (a fold beside its main run) needs the flag lifted first.
    OmegaConf.set_readonly(cfg.hydra, False)
    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path / "results")
        # run_name interpolates ${hydra:runtime.output_dir}, which resolves only against a live
        # HydraConfig; a test that needs it calls HydraConfig().set_config(cfg).
        cfg.hydra.runtime.output_dir = str(tmp_path)
    return cfg


def mlflow_logger(tmp_path: Path) -> dict[str, Any]:
    # A file store under the test's directory; MLflow 3 opens one only with
    # MLFLOW_ALLOW_FILE_STORE.
    return {
        "mlflow": {
            "_target_": "lightning.pytorch.loggers.mlflow.MLFlowLogger",
            "experiment_name": "test",
            "run_name": "${run_name}",
            "tracking_uri": f"file:{tmp_path}/mlruns",
            "tags": None,
        }
    }


def _cheap(config_name: str) -> DictConfig:
    cfg = compose_root(config_name)
    with open_dict(cfg):
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


@pytest.fixture(scope="package")
def cfg_train_global() -> DictConfig:
    return _cheap("train.yaml")


@pytest.fixture(scope="package")
def cfg_eval_global() -> DictConfig:
    return _cheap("eval.yaml")


@pytest.fixture
def cfg_train(cfg_train_global: DictConfig, tmp_path: Path) -> DictConfig:
    return point_into(cfg_train_global.copy(), tmp_path)


@pytest.fixture
def cfg_eval(cfg_eval_global: DictConfig, tmp_path: Path) -> DictConfig:
    return point_into(cfg_eval_global.copy(), tmp_path)
