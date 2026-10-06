from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.algorithms.deepsets import (
    DeepSets,
    DeepSetsMoments,
    JEDILinear,
    JEDILinearMoments,
)
from src.algorithms.vector_neurons import VectorNeurons
from tests.conftest import (
    BASELINE_ONLY,
    EMBEDDED,
    MODELS,
    MOMENT_ROWS,
    NCONSTITUENTS,
    PT_ETAREL_PHIREL,
    ROOT,
    compose_root,
    point_into,
)

EXPECTED_EXPERIMENTS = {f"{model}_{n}" for model in MODELS for n in NCONSTITUENTS}
# The baseline-only models (the embedded class and the JEDI-linear comparator) are never
# trained under the SCL, so they have no SCL space.
EXPECTED_SEARCHES = {
    f"{model}{variant}_optuna"
    for model in MODELS
    for variant in ("", "_scl")
    if not (model in BASELINE_ONLY and variant == "_scl")
}


def _names(group: str) -> list[str]:
    return sorted(path.stem for path in (ROOT / "configs" / group).glob("*.yaml"))


EXPERIMENTS = _names("experiment/hls4ml")
SEARCHES = _names("hparams_search/hls4ml")
# Each moment row's parent, its class and the parent's keys it does not have.
MOMENT_PARENTS = {
    "deepsets_moments": ("deepsets", "DeepSetsMoments", ()),
    "jedi_linear_moments": ("jedi_linear", "JEDILinearMoments", ()),
    "moments_only": ("deepsets", "DeepSetsMoments", ("phi_layers", "aggregation")),
}


def _composed(tmp_path: Path, *overrides: str) -> DictConfig:
    # The experiments select the GPU trainer and MLflow; both are overridden so that the
    # composition can be checked on any machine.
    cfg = compose_root("train.yaml", *overrides, "trainer=cpu", "logger=none")
    HydraConfig().set_config(point_into(cfg, tmp_path))
    return cfg


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_config_builds_data_algorithm_and_trainer(
    experiment: str, tmp_path: Path
) -> None:
    assert set(EXPERIMENTS) == EXPECTED_EXPERIMENTS
    cfg = _composed(tmp_path, f"experiment=hls4ml/{experiment}")
    model, nconstituents = experiment.rsplit("_", 1)

    assert hydra.utils.instantiate(cfg.data).nconstituents == int(nconstituents)
    assert hydra.utils.instantiate(cfg.trainer)
    module = hydra.utils.instantiate(cfg.algorithm)
    assert module.hparams.name == model
    assert module.hparams.nconstituents == int(nconstituents)
    # Every MAC the FLOP counter prices at 2 the eBOP counter prices at 8 x 8.
    assert module.ebops_total >= 32 * module.flops_total > 0
    assert cfg.results_table.subdir == ("embedded" if model in EMBEDDED else None)


@pytest.mark.parametrize("search", SEARCHES)
def test_every_search_config_offers_a_non_empty_space(
    search: str, tmp_path: Path
) -> None:
    assert set(SEARCHES) == EXPECTED_SEARCHES
    model = search.removesuffix("_optuna").removesuffix("_scl")
    cfg = _composed(
        tmp_path, f"experiment=hls4ml/{model}_16", f"hparams_search=hls4ml/{search}"
    )
    params = cfg.hydra.sweeper.params

    assert params
    assert all(str(value) for value in params.values())
    assert ("algorithm.symm_loss_weight" in params) == search.endswith("_scl_optuna")


def test_a_feature_list_on_the_command_line_reaches_data_and_algorithm(
    tmp_path: Path,
) -> None:
    # The feature-set ablations are one override: nbr_feats is ${len:${data.features}}, so
    # the network's input width follows the list rather than a second knob of its own.
    cfg = _composed(tmp_path, "experiment=hls4ml/mlp_16", "data.features=[pt,eta,phi]")

    assert cfg.algorithm.nbr_feats == 3
    assert hydra.utils.instantiate(cfg.data).features == ["pt", "eta", "phi"]
    assert hydra.utils.instantiate(cfg.algorithm).hparams.nbr_feats == 3


def test_the_study_name_follows_the_regime(tmp_path: Path) -> None:
    for method, scheme in (("augmentation", "none"), ("symm_loss", "so2")):
        cfg = _composed(
            tmp_path,
            "experiment=hls4ml/deepsets_16",
            "hparams_search=hls4ml/deepsets_optuna",
            f"data.method={method}",
            f"data.augmentation_scheme={scheme}",
        )
        study_name = OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)[
            "study_name"
        ]

        assert study_name == f"deepsets_nconst16_{method}_{scheme}"


def test_the_beamless_pelican_differs_from_the_beams_on_one_in_the_switch_alone() -> None:
    # pelican_lorentz.yaml is a standalone copy of pelican.yaml, because the algorithm tests load
    # these files with OmegaConf alone and a defaults list would not resolve there; this is what
    # keeps the copy from drifting.
    beams, lorentz = (
        OmegaConf.to_container(OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml"))
        for name in ("pelican", "pelican_lorentz")
    )
    assert (beams["add_beams"], lorentz["add_beams"]) == (True, False)
    assert (beams["name"], lorentz["name"]) == ("pelican", "pelican_lorentz")
    for cfg in (beams, lorentz):
        for key in ("name", "add_beams", "nbar"):
            cfg.pop(key)
    assert beams == lorentz


def test_the_lorentz_lgatr_differs_from_the_spurion_one_in_the_switches_alone() -> None:
    # lgatr_slim_lorentz.yaml is a standalone copy of lgatr_slim.yaml, because the
    # algorithm tests load these files with OmegaConf alone and a defaults list would
    # not resolve there; this is what keeps the copy from drifting.
    spurion, lorentz = (
        OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
        )
        for name in ("lgatr_slim", "lgatr_slim_lorentz")
    )
    assert (spurion["add_beams"], spurion["add_time"]) == (True, True)
    assert (lorentz["add_beams"], lorentz["add_time"]) == (False, False)
    assert (spurion["name"], lorentz["name"]) == ("lgatr_slim", "lgatr_slim_lorentz")
    for cfg in (spurion, lorentz):
        for key in ("name", "add_beams", "add_time"):
            cfg.pop(key)
    assert spurion == lorentz


def test_the_lloca_rows_differ_in_name_and_frames_alone() -> None:
    # lloca_z.yaml, lloca_so2.yaml and lloca_so3.yaml are standalone copies of
    # lloca_lorentz.yaml, because the algorithm tests load these files with OmegaConf
    # alone and a defaults list would not resolve there; this is what keeps the copies
    # from drifting.
    rows = {
        row: OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / "algorithm" / f"lloca_{row}.yaml")
        )
        for row in ("lorentz", "z", "so2", "so3")
    }
    for row, cfg in rows.items():
        assert (cfg["name"], cfg["frames"]) == (f"lloca_{row}", row)
        for key in ("name", "frames"):
            cfg.pop(key)
    for cfg in rows.values():
        assert cfg == rows["lorentz"]


def test_jedi_linear_differs_from_deepsets_in_the_network_alone() -> None:
    # jedi_linear.yaml is a standalone copy of deepsets.yaml, because the algorithm
    # tests load these files with OmegaConf alone and a defaults list would not resolve
    # there. The comparator is read against DeepSets, so the training recipe is held
    # fixed and only the network moves; this is what keeps the copy from drifting.
    deepsets, jedi = (
        OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
        )
        for name in ("deepsets", "jedi_linear")
    )
    assert (deepsets["_target_"], jedi["_target_"]) == (
        "src.algorithms.deepsets.DeepSets",
        "src.algorithms.deepsets.JEDILinear",
    )
    assert (deepsets["name"], jedi["name"]) == ("deepsets", "jedi_linear")
    architectures = (
        (deepsets, "phi_layers rho_layers aggregation dropout"),
        (jedi, "hidden head_layers aggregation"),
    )
    for cfg, keys in architectures:
        for key in ("_target_", "name", *keys.split()):
            cfg.pop(key)
    assert deepsets == jedi


def test_jedi_linear_searches_the_pooling_and_training_choices_of_deepsets() -> None:
    # The two rows differ in the network alone, so every choice they share is tuned over
    # the same values, the SCL weight included; neither searches an activation (see the
    # test below). The SCL configs hold only the weight, over a defaults list.
    deepsets, jedi, deepsets_scl, jedi_scl = (
        OmegaConf.load(
            ROOT / "configs" / "hparams_search" / "hls4ml" / f"{name}.yaml"
        ).hydra.sweeper.params
        for name in (
            "deepsets_optuna",
            "jedi_linear_optuna",
            "deepsets_scl_optuna",
            "jedi_linear_scl_optuna",
        )
    )
    for key in ("aggregation", "optimizer.lr", "optimizer.weight_decay"):
        assert jedi[f"algorithm.{key}"] == deepsets[f"algorithm.{key}"], key
    assert dict(jedi_scl) == dict(deepsets_scl)


@pytest.mark.parametrize(
    "row", ["deepsets", "jedi_linear", "deepsets_moments", "jedi_linear_moments"]
)
def test_the_search_offers_relu_and_the_fpga_cheap_pools_alone(row: str) -> None:
    # tanh costs an FPGA a lookup table and the mean a per-jet division. With no
    # activation in the space, every trial takes the network's ReLU; sum_over_n divides
    # by the constant N.
    params = OmegaConf.load(
        ROOT / "configs" / "hparams_search" / "hls4ml" / f"{row}_optuna.yaml"
    ).hydra.sweeper.params
    algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{row}.yaml")

    assert "algorithm.activation" not in params
    assert params["algorithm.aggregation"] == "choice(sum, max, sum_over_n)"
    assert algorithm.aggregation == "sum"


@pytest.mark.parametrize("experiment", [f"vn_{n}" for n in NCONSTITUENTS])
def test_the_vn_experiments_keep_full_float32_matmuls_and_their_own_columns(
    experiment: str,
) -> None:
    # TF32 rounds the inputs of every float32 matmul to about 5e-4, and this network's
    # linear layers act on the vector components themselves, so a transformed jet would
    # be rounded differently and the embedded group would hold to that precision only.
    cfg = OmegaConf.load(
        ROOT / "configs" / "experiment" / "hls4ml" / f"{experiment}.yaml"
    )
    assert cfg.trainer.precision == 32
    assert "matmul_precision" not in (cfg.get("extras") or {})
    # The one embedded row that reads pt, eta and phi rather than the four-momentum of
    # configs/data/hls4ml.yaml, which its setup() would refuse.
    assert list(cfg.data.features) == VectorNeurons.required_features


def _algorithm_yaml(name: str) -> dict:
    return OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
    )


def _search_params(name: str) -> dict[str, str]:
    path = ROOT / "configs" / "hparams_search" / "hls4ml" / f"{name}_optuna.yaml"
    return dict(OmegaConf.load(path).hydra.sweeper.params)


@pytest.mark.parametrize("row", MOMENT_ROWS)
def test_each_moment_row_differs_from_its_parent_in_the_moments_alone(row: str) -> None:
    # Standalone copies, as jedi_linear.yaml is of deepsets.yaml, read against their
    # parents: the training recipe is held fixed and only the moments move. nbr_feats is
    # 3 rather than ${len:...}, since tests/test_algorithms.py builds every row on the
    # four-momentum.
    parent, target, dropped = MOMENT_PARENTS[row]
    moments, reference = _algorithm_yaml(row), _algorithm_yaml(parent)

    assert moments["_target_"] == f"src.algorithms.deepsets.{target}"
    assert (moments["name"], moments["nbr_feats"]) == (row, 3)
    assert moments["moments"] == "uniform"
    assert moments.pop("moments_only", False) == (row == "moments_only")
    for key in ("_target_", "name", "nbr_feats", "moments"):
        moments.pop(key)
        reference.pop(key, None)
    for key in dropped:
        reference.pop(key)
    assert moments == reference


@pytest.mark.parametrize("row", MOMENT_ROWS)
def test_each_moment_row_searches_its_parents_space_and_the_moment_set(
    row: str, tmp_path: Path
) -> None:
    parent, _, dropped = MOMENT_PARENTS[row]
    params, reference = _search_params(row), _search_params(parent)
    cfg = _composed(
        tmp_path, f"experiment=hls4ml/{row}_16", f"hparams_search=hls4ml/{row}_optuna"
    )
    sweeper = OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)

    assert params.pop("algorithm.moments") == "choice(uniform, rankcut)"
    for key in dropped:
        reference.pop(f"algorithm.{key}")
    assert params == reference
    assert sweeper["storage"] == f"sqlite:///logs/optuna/hls4ml/{row}.db?timeout=600"
    assert sweeper["study_name"] == f"{row}_nconst16_augmentation_none"


@pytest.mark.parametrize(
    "experiment", [f"{row}_{n}" for row in MOMENT_ROWS for n in NCONSTITUENTS]
)
def test_the_moment_experiments_pin_their_columns_and_keep_mixed_precision(
    experiment: str,
) -> None:
    # The moments leave autocast for float32 themselves, so the trainer's 16-mixed stays,
    # and TF32 would round the inputs of their products while pair, chain and triangle
    # cancel between large terms.
    cfg = OmegaConf.load(
        ROOT / "configs" / "experiment" / "hls4ml" / f"{experiment}.yaml"
    )
    assert "precision" not in cfg.trainer
    assert "matmul_precision" not in (cfg.get("extras") or {})
    assert list(cfg.data.features) == PT_ETAREL_PHIREL
    assert DeepSetsMoments.required_features == PT_ETAREL_PHIREL
    assert JEDILinearMoments.required_features == PT_ETAREL_PHIREL
    # The parents read whatever data.features names, as every grid row does.
    assert DeepSets.required_features is None and JEDILinear.required_features is None
