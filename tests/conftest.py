import csv
import functools
import importlib.util
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.components.augmentation import (
    compute_eta,
    compute_phi,
    compute_pt,
    rebase_on_kept_axis,
)
from src.symmetry import CLASS_NAMES
from src.utils.checkpoints import MANIFEST_FILENAME, read_manifest
from src.utils.omegaconf import register_resolvers

# The configs interpolate ${len:} and ${short_hash:}, so nothing composes without these.
register_resolvers()

ROOT = rootutils.find_root(indicator=".project-root")
# configs/paths/default.yaml reads it, and a bare pytest has neither .env nor it.
os.environ.setdefault("PROJECT_ROOT", str(ROOT))
# MLflow 3 opens a file store only with this set.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# The record's caches survive between runs; rebuilding one outlasts the whole suite.
CACHE_DIR = os.environ.get("HLS4ML_CACHE_DIR", str(ROOT / "data" / "hf"))

# Read here rather than in the tests: importing src.train runs rootutils.setup_root with
# dotenv=True, which copies .env's HF_HUB_CACHE into the environment, and a gate read
# after that would never skip.
needs_record = pytest.mark.skipif(
    "HF_HUB_CACHE" not in os.environ,
    reason="Export HF_HUB_CACHE to the record's hub snapshot to run the data tests.",
)

MODELS = tuple(
    "deepsets mlp jetformer lorentznet pelican pelican_lorentz lgatr_slim "
    "lgatr_slim_lorentz lloca_lorentz lloca_z lloca_so2 lloca_so3 vn jedi_linear "
    "linformer deepsets_moments jedi_linear_moments moments_only".split()
)
# The rows that read the jet moments, pinned to [pt, etarel, phirel].
MOMENT_ROWS = ("deepsets_moments", "jedi_linear_moments", "moments_only")
# The symmetry-embedded class, trained in the baseline regime alone and never under the
# SCL; the moment rows are filed with it.
EMBEDDED = (
    *"lorentznet pelican pelican_lorentz lgatr_slim lgatr_slim_lorentz lloca_lorentz "
    "lloca_z lloca_so2 lloca_so3 vn".split(),
    *MOMENT_ROWS,
)
# The MLP and the gradient-boosted trees on the record's jet-level observables, at
# N = 150 alone: rows of no architecture, outside MODELS.
JET_LEVEL = tuple(
    "jet_level_all53 jet_level_noeta52 jet_level_bdt_all53 "
    "jet_level_bdt_noeta52".split()
)
# The gradient-boosted trees, fitted by scikit-learn on the CPU: their catalogue blocks
# select trainer=cpu and no device, and their costs are NaN.
CPU_ROWS = ("jet_level_bdt_all53", "jet_level_bdt_noeta52")
# Trained in the baseline regime alone: the embedded rows and the jet-level rows.
BASELINE_ONLY = (*EMBEDDED, *JET_LEVEL)
# Trained quantisation-aware from a float row's checkpoint, outside MODELS: no study,
# hand-written catalogues, and an algorithm that needs the optional hgq dependencies.
QUANTISED = ("vn_hgq",)
NCONSTITUENTS = (16, 50, 150)
# The record's 16 constituent columns, in its own order.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
FOUR_MOMENTUM = ["px", "py", "pz", "e"]
PT_ETA_PHI = ["pt", "eta", "phi"]
PT_ETAREL_PHIREL = ["pt", "etarel", "phirel"]
# The rows whose required_features are not the four-momentum.
READS = {"vn": PT_ETA_PHI, **dict.fromkeys(MOMENT_ROWS, PT_ETAREL_PHIREL)}
# The jet-level columns of the synthetic shards, 53 as in the record: the constant
# j_ptfrac, which has no spread to divide by, the jet rapidity and fillers.
JET_LEVEL_COLUMNS = ["j_ptfrac", "j_pt", "j_eta", *(f"j_obs{i}" for i in range(50))]
# Jets per split of the synthetic shards; the test split ends in a partial batch of 16.
JET_LEVEL_SIZES = {"train": 120, "validation": 40, "test": 50}
# In GeV. A massless constituent puts m^2 = E^2 - |p|^2 at the float32 rounding of E^2,
# and the boost tests would then measure the arithmetic rather than the transform.
MOMENTUM_SCALE = 20.0
MASS = 20.0


class StubDataModule:
    # What the base class and the probe read off HLS4MLDataModule after its setup().
    feature_names = RECORD_COLUMNS

    def __init__(
        self,
        features: list[str] = FOUR_MOMENTUM,
        norm_scales: dict[str, float] | None = None,
        keep_full_features_at_test: bool = True,
        class_names: tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        self.features = list(features)
        self.norm_scales = norm_scales
        self.keep_full_features_at_test = keep_full_features_at_test
        self.class_names = class_names


def jets(
    n_jets: int = 6,
    n_slots: int = 8,
    n_real: int = 5,
    n_columns: int = 4,
    mass: float = MASS,
    seed: int = 0,
) -> torch.Tensor:
    # Momenta in the first three columns, the energy in the fourth, padded slots zero.
    generator = torch.Generator().manual_seed(seed)
    momenta = torch.randn(n_jets, n_slots, 3, generator=generator) * MOMENTUM_SCALE
    batch = torch.zeros(n_jets, n_slots, n_columns)
    batch[..., :3] = momenta
    batch[..., 3] = torch.sqrt((momenta**2).sum(-1) + mass**2)
    batch[:, n_real:] = 0.0
    return batch


def labels(n_jets: int, seed: int = 1, n_classes: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    drawn = torch.randint(0, n_classes, (n_jets,), generator=generator)
    return torch.eye(n_classes)[drawn]


def wrap_phi(phi: torch.Tensor) -> torch.Tensor:
    return (phi + math.pi) % (2 * math.pi) - math.pi


def record_jets(
    n_jets: int = 8, n_slots: int = 16, n_real: int = 11, mass: float = 0.0
) -> torch.Tensor:
    # The record's columns in float64 from the momenta of jets() at ten times their
    # scale, the energy computed in float64 so that a massless constituent has E = |p|
    # to 1e-16 and a beam boost shifts its eta rigidly; the axis columns measured from
    # the kept constituents, as the datamodule rebases them. erel, etarot, phirot,
    # costheta and costhetarel stay zero, which no transform rewrites.
    momenta = jets(n_jets, n_slots, n_real).double()[..., :3] * 10.0
    px, py, pz = momenta.unbind(dim=-1)
    real = momenta.abs().sum(dim=-1) > 0
    columns = {
        "px": px,
        "py": py,
        "pz": pz,
        "e": torch.sqrt((momenta**2).sum(dim=-1) + mass**2) * real,
        "pt": compute_pt(px, py),
        "eta": compute_eta(px, py, pz),
        "phi": compute_phi(px, py),
    }
    batch = torch.zeros(n_jets, n_slots, len(RECORD_COLUMNS), dtype=torch.float64)
    for name, values in columns.items():
        batch[..., RECORD_COLUMNS.index(f"j1_{name}")] = values
    return rebase_on_kept_axis(batch, RECORD_COLUMNS)


def cylindrical_jets(
    n_jets: int = 5, n_slots: int = 8, n_real: int = 6
) -> torch.Tensor:
    # Rows [p_T GeV, eta, phi] of narrow jets about (0.4, 0.3) in float64, the padded
    # rows all zero and last.
    generator = torch.Generator().manual_seed(1)
    gauss = torch.randn(3, n_jets, n_real, generator=generator, dtype=torch.float64)
    x = torch.zeros(n_jets, n_slots, 3, dtype=torch.float64)
    x[:, :n_real, 0] = 40.0 * gauss[0].exp()
    x[:, :n_real, 1] = 0.4 + 0.3 * gauss[1]
    x[:, :n_real, 2] = wrap_phi(0.3 + 0.3 * gauss[2])
    return x


def algorithm(
    name: str,
    nconstituents: int = 8,
    method: str = "augmentation",
    group: str = "none",
    features: list[str] = FOUR_MOMENTUM,
    **overrides: object,
) -> torch.nn.Module:
    # The algorithm configs interpolate ${data.*} and ${trainer.*}, so nodes of our own
    # stand in.
    cfg = OmegaConf.create(
        {
            "data": {
                "nconstituents": nconstituents,
                "method": method,
                "augmentation_scheme": group,
                "transform_kwargs": {"lorentz_eta_max": 2.0},
                "features": features,
            },
            "trainer": {"accelerator": "cpu", "devices": 1},
        }
    )
    cfg.algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
    cfg.algorithm.merge_with(overrides)
    return instantiate(cfg.algorithm)


def attach(module: torch.nn.Module, datamodule: object, stage: str = "fit"):
    # setup() reads self._trainer.datamodule; the public trainer property would raise.
    module._trainer = SimpleNamespace(datamodule=datamodule)
    module.setup(stage)
    return module


def algorithm_yaml(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / "algorithm" / f"{name}.yaml"
    return OmegaConf.to_container(OmegaConf.load(path))


def with_random_gates(module: torch.nn.Module) -> torch.nn.Module:
    # Every vector-neuron gate starts at exactly 1/2, a constant under which a test
    # would pass without exercising it; N(0, 1) weights make it a function of the
    # scalars.
    from src.algorithms.components.vector_neurons import ChannelGate

    with torch.no_grad():
        for gate in module.modules():
            if isinstance(gate, ChannelGate):
                gate.linear.weight.normal_()
                gate.linear.bias.normal_()
    return module


def _with_large_coordinate_steps(module: torch.nn.Module) -> torch.nn.Module:
    # A fresh LorentzNet gate is near zero and so is every coordinate step, which is how
    # the reference's per-component clamp went unnoticed: it breaks the equivariance
    # only once a gate exceeds it, as every trained gate did. At c_weight 1e-3 a 100 GeV
    # step moves nothing either; at 1 it is a momentum.
    with torch.no_grad():
        for block in module.net.blocks:
            if block.phi_x is not None:
                block.phi_x[-1].weight.mul_(1e5)
            block.c_weight = 1.0
    return module


def _calibrated(net: torch.nn.Module, batch: torch.Tensor) -> torch.nn.Module:
    # One training pass at momentum 1 sets the moments' batch normalisation to the
    # batch's statistics; at their initial (0, 1) the raw moments, about 1e10 on these
    # jets, would reach the head unscaled and drown every other input.
    net.moments.norm.momentum = 1.0
    with torch.no_grad():
        net.train()(batch)
    return net.eval()


@functools.cache
def exact_network(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    # The row in float64 and eval mode. forward() rounds a batch to float32, but the
    # de-normalisation against the float64 buffer promotes everything after it; the
    # moment rows have no such buffer before their first Linear, so their network is
    # called on the three columns directly.
    torch.manual_seed(0)
    module = algorithm(name, nconstituents=16)
    if name in MOMENT_ROWS:
        attach(module, StubDataModule(PT_ETAREL_PHIREL))
        columns = module.model_feature_indices
        net = _calibrated(module.net.double(), record_jets()[..., columns])
        return lambda batch: net(batch[..., columns])
    if name == "lorentznet":
        return _with_large_coordinate_steps(module.eval()).double()
    if name == "vn":
        module = with_random_gates(module.eval().double())
        return attach(module, StubDataModule(PT_ETA_PHI))
    return module.eval().double()


@pytest.fixture(scope="session")
def jet_level_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Two shards per split, named as the record names them, so that their order matters.
    # Each split sits at its own loc, so that statistics fitted on another split differ;
    # j_ptfrac is 1 on every jet, as in the record.
    directory, rng = tmp_path_factory.mktemp("shards"), np.random.default_rng(0)
    scales = np.linspace(0.5, 50.0, len(JET_LEVEL_COLUMNS))
    for loc, (split, n_jets) in enumerate(JET_LEVEL_SIZES.items()):
        for shard, size in enumerate((n_jets // 2, n_jets - n_jets // 2)):
            x = rng.normal(loc, 1.0, (size, len(JET_LEVEL_COLUMNS))) * scales
            x[:, 0] = 1.0
            counts = [rng.random(rng.integers(1, 6)).tolist() for _ in range(size)]
            columns = {"j1_pt": pa.array(counts, type=pa.list_(pa.float32()))}
            columns |= {
                n: x[:, i].astype("f4") for i, n in enumerate(JET_LEVEL_COLUMNS)
            }
            columns["label"] = (np.arange(size) % 5).astype(np.int8)
            path = directory / f"{split}-{shard:05d}-of-00002.parquet"
            pq.write_table(pa.table(columns), path)
    return directory


def compose_root(config_name: str, *overrides: str) -> DictConfig:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name=config_name, return_hydra_config=True, overrides=list(overrides)
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT)
    return cfg


def point_into(cfg: DictConfig, run_dir: Path) -> DictConfig:
    # HydraConfig().set_config makes the hydra node read-only, and a copy taken
    # afterwards keeps that, so a second run's copy (a fold beside its main run)
    # needs it lifted.
    OmegaConf.set_readonly(cfg.hydra, False)
    with open_dict(cfg):
        cfg.paths.output_dir = str(run_dir)
        cfg.paths.log_dir = str(run_dir)
        cfg.paths.results_dir = str(run_dir / "results")
        # run_name interpolates ${hydra:runtime.output_dir}, which resolves only
        # against a live HydraConfig.
        cfg.hydra.runtime.output_dir = str(run_dir)
    return cfg


def mlflow_logger(directory: Path) -> dict[str, Any]:
    return {
        "mlflow": {
            "_target_": "lightning.pytorch.loggers.mlflow.MLFlowLogger",
            "experiment_name": "test",
            "run_name": "${run_name}",
            "tracking_uri": f"file:{directory}/mlruns",
            "tags": None,
        }
    }


def sweeper(tmp_path: Path, *overrides: str) -> dict[str, Any]:
    # Resolved, which needs a live HydraConfig; the experiments select the GPU trainer,
    # which the CPU one replaces.
    cfg = compose_root("train.yaml", *overrides, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))
    return OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)


def _cheap(config_name: str) -> DictConfig:
    cfg = compose_root(config_name)
    with open_dict(cfg):
        cfg.data.cache_dir = CACHE_DIR
        cfg.data.n_folds = 1
        cfg.data.batch_size = 256
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


def train_main_and_folds(cfg: DictConfig, n_folds: int = 2) -> list[tuple]:
    # The main run in its run directory, then fold 1 of n_folds in fold_1/ under it, its
    # MLflow run nested under the main one's, as submit_train.sh --folds lays them out.
    from src.train import train

    run_dir = Path(cfg.paths.output_dir)
    HydraConfig().set_config(cfg)
    trained = [train(cfg)]
    parent = read_manifest(run_dir / MANIFEST_FILENAME).runs[0].mlflow_run_id
    cfg_fold = point_into(cfg.copy(), run_dir / "fold_1")
    with open_dict(cfg_fold):
        cfg_fold.data.n_folds, cfg_fold.fold, cfg_fold.parent_run_id = (
            n_folds,
            1,
            parent,
        )
    HydraConfig().set_config(cfg_fold)
    return [*trained, train(cfg_fold)]


def read_results_row(directory: Path) -> tuple[list[str], dict[str, str]]:
    (table,) = directory.rglob("results_table_*.csv")
    with open(table, newline="") as handle:
        header = next(csv.reader(handle))
        handle.seek(0)
        (row,) = list(csv.DictReader(handle))
    return header, row


def load_script(relative: str) -> ModuleType:
    # scripts/ is not a package: a script is loaded by path with its own directory first
    # on the import path, as running it puts it there, and registered before it runs,
    # since dataclasses resolves its annotations through sys.modules.
    path = ROOT / relative
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def results_header() -> list[str]:
    # Spelled out rather than read off the table or off src.symmetry: the tables on disk
    # are appended to, so the column order is part of the interface, which every writer
    # shares.
    summary = ("mean", "std", "pm", "folds")
    groups = ("so2", "so3", "lorentz")
    performance = ("rejection_es30", "tpr_eb1", "tpr_eb10", "auc")
    tabulated = (
        "flip flip_recover flip_stable accuracy_transformed rejection_retention_es30 "
        "tpr_retention_eb1 tpr_retention_eb10 efficiency_transformed_es30 "
        "fpr_transformed_eb1 fpr_transformed_eb10"
    ).split()
    return [
        *"method model augmentation_scheme nconstituents folds_used".split(),
        *(f"test_acc_{c}" for c in ("mean", "std", "pm", "ci_lower", "ci_upper")),
        *(f"{p}_{c}" for p in performance for c in summary),
        "selected_inv_key",
        *(f"selected_inv_{c}" for c in ("mean", "std", "pm")),
        *(f"{g}_tv_{c}" for g in groups for c in summary),
        *(f"{s}_{c}" for s in tabulated for c in summary),
        *(f"{g}_{s}_{c}" for s in tabulated for g in groups for c in summary),
        "flops_mean",
        "ebops_mean",
        "test_acc_folds",
        "selected_inv_folds",
    ]


# The rows of the top-tagging replica that follow a published recipe rather than a
# search.
TOP_TAGGING_RECIPES = tuple(
    "lorentznet pelican pelican_lorentz lgatr_slim lgatr_slim_lorentz lloca_lorentz "
    "lloca_z lloca_so2 lloca_so3".split()
)
# Every row of the replica, at N = 100 in the baseline regime alone: the embedded rows,
# the five grid rows on [pt, etarel, phirel] and the moment rows.
TOP_TAGGING = (
    *TOP_TAGGING_RECIPES,
    "vn",
    *"deepsets mlp jetformer jedi_linear linformer".split(),
    *MOMENT_ROWS,
)
TOP_TAGGING_NCONSTITUENTS = (100,)
# The columns of the converted cache, in its order.
TOP_TAGGING_COLUMNS = ("px", "py", "pz", "e", "pt", "eta", "phi", "etarel", "phirel")
# Jets per raw file of the synthetic record, read in chunks of 32: the training file in
# three, the test split ending in a partial batch of 16. N = 8 truncates the jets of
# more constituents.
TOP_TAGGING_SIZES = {"train.h5": 96, "val.h5": 32, "test.h5": 40}
TOP_TAGGING_N = 8
# The jets of every synthetic file whose first two constituents are swapped.
TOP_TAGGING_PLANTED = (1, 3)


@functools.cache
def top_tagging_raw(name: str) -> tuple[np.ndarray, np.ndarray]:
    # The raw file *name* as the record stores it: [E, px, py, pz] in GeV (jets, 200, 4)
    # of 1 to 20 massless constituents at p_T U(1, 100) GeV in falling p_T but in the
    # planted jets, zero-padded, and is_signal_new alternating from 0.
    n_jets = TOP_TAGGING_SIZES[name]
    rng = np.random.default_rng(list(TOP_TAGGING_SIZES).index(name))
    pt = -np.sort(-rng.uniform(1.0, 100.0, (n_jets, 20)), axis=1)
    eta, phi = (rng.normal(centre, 0.3, (n_jets, 20)) for centre in (0.5, 0.2))
    p4 = np.stack(
        [pt * np.cosh(eta), pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], -1
    )
    multiplicity = rng.integers(1, 21, n_jets)
    multiplicity[list(TOP_TAGGING_PLANTED)] = 20
    p4[np.arange(20) >= multiplicity[:, None]] = 0.0
    p4[list(TOP_TAGGING_PLANTED), :2] = p4[list(TOP_TAGGING_PLANTED), 1::-1]
    padded = np.zeros((n_jets, 200, 4), dtype=np.float32)
    padded[:, :20] = p4
    return padded, np.arange(n_jets) % 2


def _read_top_tagging(path: Path) -> Any:
    p4, flags = top_tagging_raw(path.name)
    for start in range(0, len(p4), 32):
        yield p4[start : start + 32], flags[start : start + 32]


@pytest.fixture
def top_tagging_cache(tmp_path: Path) -> Path:
    # The cache root of the synthetic record at N = TOP_TAGGING_N, as the datamodule's
    # cache_dir: the cache exists, so nothing is downloaded.
    from src.data.components.top_tagging import convert

    def count(path: Path) -> int:
        return TOP_TAGGING_SIZES[path.name]

    convert(tmp_path / "raw", tmp_path, TOP_TAGGING_N, _read_top_tagging, count, 50)
    return tmp_path
