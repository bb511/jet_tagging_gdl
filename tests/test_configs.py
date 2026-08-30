"""Every config the repository ships must compose and build the objects it names.

A broken config surfaces only when a run has already been queued, hours after the edit, so the
whole grid is composed here instead: the two roots, the four algorithms, the twelve experiments and
the eight searches.
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


def _names(group: str) -> list[str]:
    """The config names of one group, as the command line spells them."""
    return sorted(path.stem for path in (ROOT / "configs" / group).glob("*.yaml"))


ALGORITHMS = _names("algorithm")
EXPERIMENTS = _names("experiment/hls4ml")
SEARCHES = _names("hparams_search/hls4ml")


def _composed(tmp_path: Path, *overrides: str) -> DictConfig:
    """Compose ``train.yaml`` as the command line would, on the cheapest trainer.

    The experiment configs select the GPU trainer and MLflow; both are overridden here so that the
    composition can be checked on any machine.

    :param tmp_path: stands in for the Hydra run directory the paths hang off.
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


def test_the_training_root_builds_its_data_algorithm_and_trainer(cfg_train: DictConfig) -> None:
    """``train.yaml`` alone, with no experiment, is what a debug run composes."""
    HydraConfig().set_config(cfg_train)

    assert hydra.utils.instantiate(cfg_train.data)
    assert hydra.utils.instantiate(cfg_train.algorithm)
    assert hydra.utils.instantiate(cfg_train.trainer)


def test_the_evaluation_root_builds_its_data_algorithm_and_trainer(cfg_eval: DictConfig) -> None:
    """``eval.yaml`` composes the same stack, and adds the manifest path train.py writes."""
    HydraConfig().set_config(cfg_eval)

    assert hydra.utils.instantiate(cfg_eval.data)
    assert hydra.utils.instantiate(cfg_eval.algorithm)
    assert hydra.utils.instantiate(cfg_eval.trainer)
    assert "checkpoint_manifest_path" in cfg_eval


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_every_algorithm_config_builds_its_network(algorithm: str, tmp_path: Path) -> None:
    """An algorithm config is only correct once it has built the network it describes."""
    cfg = _composed(tmp_path, f"algorithm={algorithm}")
    module = hydra.utils.instantiate(cfg.algorithm)

    assert module.hparams.name == algorithm
    assert module.hparams.nbr_feats == len(cfg.data.features)


def test_the_experiment_grid_covers_every_model_and_constituent_count() -> None:
    """The catalogues under scripts/hls4ml name these twelve configs and nothing else."""
    assert set(EXPERIMENTS) == {f"{m}_{n}" for m in MODELS for n in NCONSTITUENTS}


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_every_experiment_config_builds_data_algorithm_and_trainer(
    experiment: str, tmp_path: Path
) -> None:
    """What a training run does before it reaches the data, done without the data."""
    cfg = _composed(tmp_path, f"experiment=hls4ml/{experiment}")
    model, nconstituents = experiment.rsplit("_", 1)

    assert hydra.utils.instantiate(cfg.data).nconstituents == int(nconstituents)
    assert hydra.utils.instantiate(cfg.trainer)
    module = hydra.utils.instantiate(cfg.algorithm)
    assert module.hparams.name == model
    assert module.hparams.nconstituents == int(nconstituents)
    assert module.hparams["flops_total"] > 0


def test_the_search_grid_covers_every_model_in_both_regimes() -> None:
    """One space per architecture, plus the variant that adds the SCL weight to it."""
    assert set(SEARCHES) == {f"{m}{v}_optuna" for m in MODELS for v in ("", "_scl")}


@pytest.mark.parametrize("search", SEARCHES)
def test_every_search_config_offers_a_non_empty_space(search: str, tmp_path: Path) -> None:
    """An empty space would let a sweep run its whole budget on one point."""
    model = search.split("_")[0]
    cfg = _composed(tmp_path, f"experiment=hls4ml/{model}_16", f"hparams_search=hls4ml/{search}")
    params = cfg.hydra.sweeper.params

    assert params
    assert all(str(value) for value in params.values())
    assert ("algorithm.symm_loss_weight" in params) == search.endswith("_scl_optuna")


@pytest.mark.parametrize("method, scheme", [("augmentation", "none"), ("symm_loss", "so2")])
def test_the_study_name_follows_the_regime(method: str, scheme: str, tmp_path: Path) -> None:
    """One study per regime, named from the config alone, so no driver has to pass it."""
    cfg = _composed(
        tmp_path,
        "experiment=hls4ml/deepsets_16",
        "hparams_search=hls4ml/deepsets_optuna",
        f"data.method={method}",
        f"data.augmentation_scheme={scheme}",
    )
    study_name = OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)["study_name"]

    assert study_name == f"deepsets_nconst16_{method}_{scheme}"
