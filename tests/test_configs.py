"""Every config the repository ships must compose and build the objects it names.

A broken config surfaces only when a run has already been queued, hours after the edit, so the
whole grid is composed here instead: the two roots, the twelve experiments and the seven searches.
The experiments build every algorithm config three times over, so those need no test of their own.
"""

from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from tests.conftest import ROOT

MODELS = ("deepsets", "mlp", "jetformer", "lorentznet")
NCONSTITUENTS = (16, 50, 150)
# What the catalogues under scripts/hls4ml name, and nothing else.
EXPECTED_EXPERIMENTS = {f"{model}_{n}" for model in MODELS for n in NCONSTITUENTS}
# LorentzNet is the symmetry-embedded class, trained baseline only, so it has no SCL space.
EXPECTED_SEARCHES = {
    f"{model}{variant}_optuna"
    for model in MODELS
    for variant in ("", "_scl")
    if not (model == "lorentznet" and variant == "_scl")
}


def _names(group: str) -> list[str]:
    """The config names of one group, as the command line spells them."""
    return sorted(path.stem for path in (ROOT / "configs" / group).glob("*.yaml"))


EXPERIMENTS = _names("experiment/hls4ml")
SEARCHES = _names("hparams_search/hls4ml")


def _composed(tmp_path: Path, *overrides: str) -> DictConfig:
    """Compose ``train.yaml`` as the command line would, on the cheapest trainer.

    The experiment configs select the GPU trainer and MLflow; both are overridden here so that the
    composition can be checked on any machine, with *tmp_path* standing in for the run directory.
    """
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[*overrides, "trainer=cpu", "logger=none"],
        )
    with open_dict(cfg):
        cfg.paths.root_dir = str(ROOT)
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.paths.results_dir = str(tmp_path / "results")
        cfg.hydra.runtime.output_dir = str(tmp_path)
    HydraConfig().set_config(cfg)

    return cfg


@pytest.mark.parametrize("root", ["cfg_train", "cfg_eval"])
def test_each_root_builds_its_data_algorithm_and_trainer(root: str, request) -> None:
    """``train.yaml`` alone is what a debug run composes; ``eval.yaml`` adds the manifest path."""
    cfg = request.getfixturevalue(root)
    HydraConfig().set_config(cfg)

    assert hydra.utils.instantiate(cfg.data)
    assert hydra.utils.instantiate(cfg.algorithm)
    assert hydra.utils.instantiate(cfg.trainer)
    assert root == "cfg_train" or "checkpoint_manifest_path" in cfg


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_config_builds_data_algorithm_and_trainer(
    experiment: str, tmp_path: Path
) -> None:
    """What a training run does before it reaches the data, done without the data."""
    assert set(EXPERIMENTS) == EXPECTED_EXPERIMENTS
    cfg = _composed(tmp_path, f"experiment=hls4ml/{experiment}")
    model, nconstituents = experiment.rsplit("_", 1)

    assert hydra.utils.instantiate(cfg.data).nconstituents == int(nconstituents)
    assert hydra.utils.instantiate(cfg.trainer)
    module = hydra.utils.instantiate(cfg.algorithm)
    assert module.hparams.name == model
    assert module.hparams.nconstituents == int(nconstituents)
    assert module.flops_total > 0


@pytest.mark.parametrize("search", SEARCHES)
def test_every_search_config_offers_a_non_empty_space(search: str, tmp_path: Path) -> None:
    """An empty space would let a sweep run its whole budget on one point."""
    assert set(SEARCHES) == EXPECTED_SEARCHES
    model = search.split("_")[0]
    cfg = _composed(tmp_path, f"experiment=hls4ml/{model}_16", f"hparams_search=hls4ml/{search}")
    params = cfg.hydra.sweeper.params

    assert params
    assert all(str(value) for value in params.values())
    assert ("algorithm.symm_loss_weight" in params) == search.endswith("_scl_optuna")


def test_the_study_name_follows_the_regime(tmp_path: Path) -> None:
    """One study per regime, named from the config alone, so no driver has to pass it."""
    for method, scheme in (("augmentation", "none"), ("symm_loss", "so2")):
        cfg = _composed(
            tmp_path,
            "experiment=hls4ml/deepsets_16",
            "hparams_search=hls4ml/deepsets_optuna",
            f"data.method={method}",
            f"data.augmentation_scheme={scheme}",
        )
        study_name = OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)["study_name"]

        assert study_name == f"deepsets_nconst16_{method}_{scheme}"
