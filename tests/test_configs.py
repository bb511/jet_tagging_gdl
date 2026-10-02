from pathlib import Path

import hydra
import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from tests.conftest import (
    BASELINE_ONLY,
    EMBEDDED,
    MODELS,
    NCONSTITUENTS,
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
    assert module.flops_total > 0
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


def test_the_vn_rows_differ_in_name_and_symmetry_alone() -> None:
    # vn_so2.yaml is a standalone copy of vn_z.yaml, because the algorithm tests load
    # these files with OmegaConf alone and a defaults list would not resolve there; this
    # is what keeps the copy from drifting.
    rows = {
        row: OmegaConf.to_container(
            OmegaConf.load(ROOT / "configs" / "algorithm" / f"vn_{row}.yaml")
        )
        for row in ("z", "so2")
    }
    for row, cfg in rows.items():
        assert (cfg["name"], cfg["symmetry"]) == (f"vn_{row}", row)
        for key in ("name", "symmetry"):
            cfg.pop(key)
    assert rows["so2"] == rows["z"]


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
        (deepsets, "phi_layers rho_layers aggregation dropout mask"),
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


@pytest.mark.parametrize("row", ["deepsets", "jedi_linear"])
def test_the_search_offers_relu_and_the_fpga_cheap_pools_alone(row: str) -> None:
    # tanh costs an FPGA a lookup table and the mean a per-jet division. With no
    # activation in the space, every trial takes the config's ReLU; sum_over_n divides
    # by the constant N.
    params = OmegaConf.load(
        ROOT / "configs" / "hparams_search" / "hls4ml" / f"{row}_optuna.yaml"
    ).hydra.sweeper.params
    algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{row}.yaml")

    assert "algorithm.activation" not in params
    assert params["algorithm.aggregation"] == "choice(sum, max, sum_over_n)"
    assert (algorithm.activation, algorithm.aggregation) == ("relu", "sum")


def test_the_vn_rows_share_one_search_space() -> None:
    # The two rows are searched at equal effort, so that they differ in the embedded
    # group alone; the storage and the study name are the row's own.
    z, so2 = (
        OmegaConf.to_container(
            OmegaConf.load(
                ROOT / "configs" / "hparams_search" / "hls4ml" / f"vn_{row}_optuna.yaml"
            ).hydra.sweeper
        )
        for row in ("z", "so2")
    )
    for key in ("params", "n_trials", "direction", "sampler"):
        assert z[key] == so2[key], key


@pytest.mark.parametrize(
    "experiment", [f"vn_{row}_{n}" for row in ("z", "so2") for n in NCONSTITUENTS]
)
def test_the_vn_experiments_keep_full_float32_matmuls(experiment: str) -> None:
    # TF32 rounds the inputs of every float32 matmul to about 5e-4, and this network's
    # linear layers act on the vector components themselves, so a transformed jet would
    # be rounded differently and the embedded group would hold to that precision only.
    cfg = OmegaConf.load(
        ROOT / "configs" / "experiment" / "hls4ml" / f"{experiment}.yaml"
    )
    assert cfg.trainer.precision == 32
    assert "matmul_precision" not in (cfg.get("extras") or {})
