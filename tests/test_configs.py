import hashlib
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import IntervalSweep
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import LambdaLR

from src.utils.omegaconf import short_hash
from tests.conftest import (
    BASELINE_ONLY,
    CPU_ROWS,
    EMBEDDED,
    JET_LEVEL,
    MODELS,
    MOMENT_ROWS,
    NCONSTITUENTS,
    PT_ETA_PHI,
    QUANTISED,
    ROOT,
    TOP_TAGGING,
    TOP_TAGGING_NCONSTITUENTS,
    TOP_TAGGING_RECIPES,
    algorithm_yaml,
    compose_root,
    point_into,
    sweeper,
)

EXPERIMENT_DIR = ROOT / "configs" / "experiment" / "hls4ml"
SEARCH_DIR = ROOT / "configs" / "hparams_search" / "hls4ml"
EXPERIMENTS = sorted(path.stem for path in EXPERIMENT_DIR.glob("*.yaml"))
TOP_EXPERIMENT_DIR = ROOT / "configs" / "experiment" / "top_tagging"
TOP_EXPERIMENTS = sorted(path.stem for path in TOP_EXPERIMENT_DIR.glob("*.yaml"))
TOP_SEARCH_DIR = ROOT / "configs" / "hparams_search" / "top_tagging"
TOP_SEARCHES = sorted(path.stem for path in TOP_SEARCH_DIR.glob("*.yaml"))
SEARCHES = sorted(path.stem for path in SEARCH_DIR.glob("*.yaml"))
# The searches of the cheap vector neurons with and without the full jet's axis, which
# belong to no row.
CHEAP_SEARCHES = ("vn_cheap_optuna", "vn_cheap_axis_optuna")
# Digests of each experiment's state-dict layout at N = 16, 50 and 150 (the jet-level
# rows at 150 alone): the keys in order with their shapes and dtypes, then the Keras
# weights of vn_hgq, from the K1 golden of commit 50d78b93. The checkpoints of the
# finals hold these layouts, and eval.py loads them strictly.
LAYOUTS = {
    "deepsets": "9294bcdce881 181f7342cf6e 181f7342cf6e",
    "mlp": "3d445a0a0618 78bf9b67580c 8a92848f3053",
    "jetformer": "396ef5d47808 396ef5d47808 396ef5d47808",
    "lorentznet": "f7b122f582b9 f7b122f582b9 f7b122f582b9",
    "pelican": "ffadb7cc6a1b ffadb7cc6a1b ffadb7cc6a1b",
    "pelican_lorentz": "816ee4ee78da 816ee4ee78da 9e3f131dcde2",
    "lgatr_slim": "7b4746935bf2 7b4746935bf2 7b4746935bf2",
    "lgatr_slim_lorentz": "7b4746935bf2 7b4746935bf2 7b4746935bf2",
    "lloca_lorentz": "7d6d372f2e86 7d6d372f2e86 7d6d372f2e86",
    "lloca_z": "f7d323974944 f7d323974944 f7d323974944",
    "lloca_so2": "5842a90f20b0 5842a90f20b0 5842a90f20b0",
    "lloca_so3": "f7d323974944 f7d323974944 f7d323974944",
    "vn": "e3a06fa147b8 c195801f4616 6e4923c8de4e",
    "jedi_linear": "52bfba5cff0e 52bfba5cff0e 52bfba5cff0e",
    "linformer": "65f74d9bbd94 cb44afba1ef0 6a1caad50656",
    "deepsets_moments": "bc017939538b bc017939538b bc017939538b",
    "jedi_linear_moments": "89beb94b7da7 89beb94b7da7 89beb94b7da7",
    "moments_only": "a631eed1a09e a631eed1a09e a631eed1a09e",
    "jet_level_all53": "cba69c105b19",
    "jet_level_noeta52": "e123efd972cf",
    "jet_level_bdt_all53": "0454180ac26f",
    "jet_level_bdt_noeta52": "0454180ac26f",
    "vn_hgq": "8d315eab0580 a376435f525e 9845c7c24750",
}
LAYOUT = {
    f"{row}_{n}": digest
    for row, digests in LAYOUTS.items()
    for n, digest in zip((150,) if row in JET_LEVEL else NCONSTITUENTS, digests.split())
}
SUBDIRS = {
    **dict.fromkeys(EMBEDDED, "embedded"),
    **dict.fromkeys(JET_LEVEL, "jet_level"),
    **dict.fromkeys(QUANTISED, "quantised"),
}
# The rows that share a network keep standalone configs, since results and log
# directories are keyed by algorithm.name: (original, copy, the keys the copy may
# change), algorithm configs and search spaces alike.
COPIES = [
    ("algorithm/pelican", "algorithm/pelican_lorentz", "name add_beams nbar"),
    ("algorithm/lgatr_slim", "algorithm/lgatr_slim_lorentz", "name add_beams add_time"),
    *(
        ("algorithm/lloca_lorentz", f"algorithm/lloca_{frames}", "name frames")
        for frames in ("z", "so2", "so3")
    ),
    (
        "algorithm/deepsets",
        "algorithm/jedi_linear",
        "_target_ name phi_layers rho_layers dropout hidden head_layers",
    ),
    (
        "algorithm/deepsets",
        "algorithm/deepsets_moments",
        "_target_ name nbr_feats moments",
    ),
    (
        "algorithm/deepsets",
        "algorithm/moments_only",
        "_target_ name nbr_feats phi_layers aggregation moments moments_only",
    ),
    (
        "algorithm/jedi_linear",
        "algorithm/jedi_linear_moments",
        "_target_ name nbr_feats moments",
    ),
    *(
        ("algorithm/mlp", f"algorithm/{row}", "name nconstituents nbr_feats")
        for row in ("jet_level_all53", "jet_level_noeta52")
    ),
    (
        "algorithm/jet_level_bdt_all53",
        "algorithm/jet_level_bdt_noeta52",
        "name nbr_feats",
    ),
    (
        "algorithm/vn",
        "algorithm/vn_hgq",
        "_target_ name units optimizer scheduler quantisation beta beta_schedule "
        "init_from calibration_jets floatx jax_platform jax_devices warmup_fraction",
    ),
    (
        "search/deepsets",
        "search/jedi_linear",
        "storage study_name params.algorithm.phi_layers params.algorithm.rho_layers "
        "params.algorithm.hidden params.algorithm.head_layers",
    ),
    ("search/deepsets_scl", "search/jedi_linear_scl", ""),
    (
        "search/deepsets",
        "search/deepsets_moments",
        "storage study_name params.algorithm.moments",
    ),
    (
        "search/deepsets",
        "search/moments_only",
        "storage study_name params.algorithm.phi_layers params.algorithm.aggregation "
        "params.algorithm.moments",
    ),
    (
        "search/jedi_linear",
        "search/jedi_linear_moments",
        "storage study_name params.algorithm.moments",
    ),
    *(
        ("search/mlp", f"search/{row}", "storage study_name")
        for row in ("jet_level_all53", "jet_level_noeta52")
    ),
    (
        "search/jet_level_bdt_all53",
        "search/jet_level_bdt_noeta52",
        "storage study_name",
    ),
    ("search/vn_cheap", "search/vn_cheap_axis", "study_name"),
]
# (trainer.precision, extras.matmul_precision) of each row's experiments, None where the
# trainer's own stays. fp32 for the embedded rows, whose invariants overflow fp16, with
# TF32 except for vn, whose linear layers act on the vector components themselves; the
# moment rows keep 16-mixed, their moments leaving autocast in float32 themselves.
ARITHMETIC = {
    **dict.fromkeys(EMBEDDED, (32, "high")),
    **dict.fromkeys(MOMENT_ROWS, (None, None)),
    **dict.fromkeys(("vn", *CPU_ROWS, *QUANTISED), (32, None)),
}
FPGA_POOLS = "choice(sum, max, sum_over_n)"


def _layout(module: torch.nn.Module) -> str:
    lines = [
        f"{name} {list(v.shape)} {v.dtype}"
        if torch.is_tensor(v)
        else f"{name} {type(v).__name__}"
        for name, v in module.state_dict().items()
    ]
    keras = getattr(module.net, "keras_model", None)
    weights = [] if keras is None else keras.get_weights()
    lines += [f"{list(w.shape)} {w.dtype}" for w in weights]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]


def _flat(node: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat = {}
    for key, value in node.items():
        if isinstance(value, dict):
            flat |= _flat(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _kept(spec: str, changed: list[str]) -> dict[str, Any]:
    kind, name = spec.split("/")
    if kind == "algorithm":
        config = algorithm_yaml(name)
    else:
        path = SEARCH_DIR / f"{name}_optuna.yaml"
        config = OmegaConf.to_container(OmegaConf.load(path).hydra.sweeper)
    # A changed key covers every key nested under it.
    kept = _flat(config).items()
    return {
        k: v
        for k, v in kept
        if not any(k == c or k.startswith(f"{c}.") for c in changed)
    }


def test_the_configs_are_those_the_row_lists_name() -> None:
    experiments = {f"{row}_{n}" for row in (*MODELS, *QUANTISED) for n in NCONSTITUENTS}
    searches = {
        f"{row}{scl}_optuna"
        for row in (*MODELS, *JET_LEVEL)
        for scl in ("", "_scl")
        if not (row in BASELINE_ONLY and scl)
    }

    assert set(EXPERIMENTS) == experiments | {f"{row}_150" for row in JET_LEVEL}
    assert set(SEARCHES) == searches | set(CHEAP_SEARCHES)
    assert set(TOP_EXPERIMENTS) == {
        f"{row}_{n}" for row in TOP_TAGGING for n in TOP_TAGGING_NCONSTITUENTS
    }
    assert set(TOP_SEARCHES) == {
        f"{row}_optuna" for row in set(TOP_TAGGING) - set(TOP_TAGGING_RECIPES)
    }


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_builds_its_row(experiment: str, tmp_path: Path) -> None:
    model, n = experiment.rsplit("_", 1)
    if model in QUANTISED:
        pytest.importorskip("hgq")
    overrides = ("trainer=cpu", "logger=none")
    cfg = compose_root("train.yaml", f"experiment=hls4ml/{experiment}", *overrides)
    HydraConfig().set_config(point_into(cfg, tmp_path))
    module = hydra.utils.instantiate(cfg.algorithm)
    monitored = ("model_checkpoint", "early_stopping")
    monitors = [
        cfg.callbacks[name].monitor for name in monitored if name in cfg.callbacks
    ]

    assert hydra.utils.instantiate(cfg.data).nconstituents == int(n)
    assert hydra.utils.instantiate(cfg.trainer)
    assert module.hparams.name == model
    assert module.hparams.nconstituents == (1 if model in JET_LEVEL else int(n))
    assert _layout(module) == LAYOUT[experiment]
    # save_hyperparameters collects these through Lightning's frame walk; without them a
    # checkpoint cannot rebuild the network it holds.
    unsaved = {"_target_", "_convert_", "optimizer", "scheduler"}
    assert set(cfg.algorithm) - unsaved <= set(module.hparams)
    flops, ebops = module.flops_total, module.ebops_total
    if model in CPU_ROWS:
        assert math.isnan(flops) and math.isnan(ebops)
    elif model in QUANTISED:
        assert math.isnan(ebops) and flops > 0
    else:
        # Every MAC the FLOP counter prices at 2 the eBOP counter prices at 8 x 8.
        assert ebops >= 32 * flops > 0
    assert cfg.results_table.subdir == SUBDIRS.get(model)
    assert cfg.callbacks.invariance_metric.probe == (model not in JET_LEVEL)
    # The quantised row's last epoch is the model: unranked, and never stopped early.
    assert monitors == ([None] if model in QUANTISED else ["val/acc", "val/acc"])


@pytest.mark.parametrize("search", sorted(set(SEARCHES) - set(CHEAP_SEARCHES)))
def test_every_search_names_its_study_from_the_regime(
    search: str, tmp_path: Path
) -> None:
    model = search.removesuffix("_optuna").removesuffix("_scl")
    n = 150 if model in JET_LEVEL else 16
    method, scheme = (
        ("symm_loss", "so2") if "_scl" in search else ("augmentation", "none")
    )
    regime = (f"data.method={method}", f"data.augmentation_scheme={scheme}")
    experiment = (f"experiment=hls4ml/{model}_{n}", f"hparams_search=hls4ml/{search}")

    found = sweeper(tmp_path, *experiment, *regime)

    assert found["params"] and all(str(value) for value in found["params"].values())
    assert ("algorithm.symm_loss_weight" in found["params"]) == (method == "symm_loss")
    assert found["study_name"] == f"{model}_nconst{n}_{method}_{scheme}"
    assert found["storage"] == f"sqlite:///logs/optuna/hls4ml/{model}.db?timeout=600"


def _cheap_search(search: str, tmp_path: Path) -> tuple[DictConfig, Any]:
    regime = ("data.method=augmentation", "data.augmentation_scheme=none")
    overrides = ("experiment=hls4ml/vn_16", f"hparams_search=hls4ml/{search}", *regime)
    cfg = compose_root("train.yaml", *overrides, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))
    return cfg, hydra.utils.instantiate(cfg.algorithm)


def test_the_cheap_vn_searches_differ_by_the_full_axis_alone(tmp_path: Path) -> None:
    (control, plain), (axis, offset) = (
        _cheap_search(search, tmp_path) for search in CHEAP_SEARCHES
    )
    found = [
        OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)
        for cfg in (control, axis)
    ]
    fixed = [
        (cfg.algorithm.num_blocks, cfg.algorithm.gate, cfg.algorithm.stream_channels)
        for cfg in (control, axis)
    ]
    widths = [module.net.blocks[0].norm.num_features for module in (plain, offset)]

    assert [study["study_name"] for study in found] == [
        "vn_cheap_nconst16_augmentation_none",
        "vn_cheap_axis_nconst16_augmentation_none",
    ]
    assert {study["storage"] for study in found} == {
        "sqlite:///logs/optuna/hls4ml/vn_cheap.db?timeout=600"
    }
    assert fixed == [(1, False, 0)] * 2
    assert not hydra.utils.instantiate(control.data).full_axis
    assert hydra.utils.instantiate(axis.data).full_axis
    assert plain.required_features == list(control.data.features) == PT_ETA_PHI
    assert offset.required_features == list(axis.data.features)
    assert offset.required_features == [*PT_ETA_PHI, "etarel", "phirel"]
    # The leader's two angles from the full axis, and nothing else, join the first block.
    assert widths[1] == widths[0] + 2
    assert offset.flops_total > plain.flops_total > 0


@pytest.mark.parametrize("original, copy, changed", COPIES)
def test_each_standalone_copy_equals_its_original_but_for_its_own_keys(
    original: str, copy: str, changed: str
) -> None:
    assert _kept(copy, changed.split()) == _kept(original, changed.split())


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_each_experiment_sets_the_arithmetic_of_its_row(experiment: str) -> None:
    model = experiment.rsplit("_", 1)[0]
    cfg = OmegaConf.to_container(OmegaConf.load(EXPERIMENT_DIR / f"{experiment}.yaml"))
    trainer, extras, data = (
        cfg.get(node) or {} for node in ("trainer", "extras", "data")
    )
    manual = model in (*CPU_ROWS, *QUANTISED)

    expected = ARITHMETIC.get(model, (None, None))
    assert (trainer.get("precision"), extras.get("matmul_precision")) == expected
    # Lightning refuses gradient clipping under manual optimisation, and the threads JAX
    # starts rule out forked loader workers.
    assert (
        "gradient_clip_val" in trainer and trainer["gradient_clip_val"] is None
    ) == manual
    assert (data.get("num_workers") == 0) == (model in QUANTISED)


@pytest.mark.parametrize(
    "row", ["deepsets", "mlp", "linformer", "jedi_linear", *MOMENT_ROWS]
)
def test_the_fpga_rows_search_relu_and_the_three_pools_alone(row: str) -> None:
    # tanh costs an FPGA a lookup table and the mean a per-jet division; sum_over_n
    # divides by the constant N. With no activation searched, every trial takes ReLU.
    params = OmegaConf.load(SEARCH_DIR / f"{row}_optuna.yaml").hydra.sweeper.params

    assert not any("activation" in key for key in params)
    assert params.get("algorithm.aggregation", FPGA_POOLS) == FPGA_POOLS
    assert algorithm_yaml(row).get("activation", "relu") == "relu"


def _arithmetic(path: Path) -> tuple[Any, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path))
    trainer, extras = (cfg.get(node) or {} for node in ("trainer", "extras"))
    return trainer.get("precision"), extras.get("matmul_precision")


@pytest.mark.parametrize("experiment", TOP_EXPERIMENTS)
def test_every_top_tagging_experiment_builds_a_two_class_row(
    experiment: str, tmp_path: Path
) -> None:
    model, n = experiment.rsplit("_", 1)
    overrides = (f"experiment=top_tagging/{experiment}", "trainer=cpu", "logger=none")
    cfg = compose_root("train.yaml", *overrides)
    HydraConfig().set_config(point_into(cfg, tmp_path))
    module = hydra.utils.instantiate(cfg.algorithm)
    datamodule = hydra.utils.instantiate(cfg.data)

    assert model in TOP_TAGGING and int(n) in TOP_TAGGING_NCONSTITUENTS
    assert (module.hparams.output_dim, datamodule.class_names) == (2, ("qcd", "top"))
    assert module.hparams.nconstituents == datamodule.nconstituents == int(n)
    assert cfg.results_table.output_path.endswith("top_tagging/results_table.csv")
    assert cfg.results_table.subdir == ("embedded" if model in EMBEDDED else None)
    assert "top_tagging" in cfg.tags
    # The arithmetic of the row's hls4ml experiment at N = 150.
    assert _arithmetic(TOP_EXPERIMENT_DIR / f"{experiment}.yaml") == _arithmetic(
        EXPERIMENT_DIR / f"{model}_150.yaml"
    )
    # The optimiser and the scheduler are built once the trainer knows the run's length;
    # a recipe's schedule replaces the algorithm's whole, none of its keys surviving.
    module._trainer = SimpleNamespace(estimated_stepping_batches=100)
    scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, LambdaLR) == (model in TOP_TAGGING_RECIPES)


@pytest.mark.parametrize("search", TOP_SEARCHES)
def test_every_top_tagging_search_holds_its_experiments_point(
    search: str, tmp_path: Path
) -> None:
    model = search.removesuffix("_optuna")
    overrides = (
        f"experiment=top_tagging/{model}_100",
        f"hparams_search=top_tagging/{search}",
    )
    cfg = compose_root("train.yaml", *overrides, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))
    found = OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)
    point = _flat({"algorithm": OmegaConf.to_container(cfg.algorithm, resolve=True)})
    parser = OverridesParser.create()
    database = f"logs/optuna/top_tagging/{model}.db"

    assert found["study_name"] == f"top_tagging_{model}_nconst100_augmentation_none"
    assert found["storage"] == f"sqlite:///{database}?timeout=600"
    # The experiment's values, which a campaign enqueues as trial 0 under these names,
    # lie in the space, and the learning rate spans a factor 3 either side at most.
    for key, spec in found["params"].items():
        sweep = parser.parse_override(f"{key}={spec}").value()
        if isinstance(sweep, IntervalSweep):
            assert sweep.start < point[key] < sweep.end <= 9 * sweep.start
        else:
            assert point[key] in sweep.list


def test_the_resolvers_count_features_hash_runs_and_label_folds(tmp_path: Path) -> None:
    # nbr_feats is ${len:${data.features}}, so a feature-set ablation is one override.
    features = ("experiment=hls4ml/mlp_16", "data.features=[pt,eta,phi]")
    cfg = compose_root("train.yaml", *features, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))

    assert hydra.utils.instantiate(cfg.algorithm).hparams.nbr_feats == 3
    assert (cfg.run_name, cfg.run_label) == (short_hash(str(tmp_path), 12), "main")
    cfg.fold = 3
    assert (cfg.run_name, cfg.run_label) == ("fold_3", "fold_3")
