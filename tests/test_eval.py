"""The evaluation entry point, from a training run's manifest and from a single checkpoint.

Both paths fit a model first, so these are the slowest tests in the suite as well as the only ones
that exercise the handover from train.py to eval.py: the manifest, the shared results table and the
row it writes.
"""

import csv
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from src.utils.checkpoints import MANIFEST_FILENAME
from tests.conftest import needs_record

pytestmark = [needs_record, pytest.mark.slow]


def _results_row(results_dir: Path) -> dict[str, str]:
    """The single row the shared results table holds after an evaluation."""
    tables = list(results_dir.rglob("results_table_*.csv"))
    assert len(tables) == 1, f"expected one results table, found {tables}"
    with open(tables[0], newline="") as table:
        rows = list(csv.DictReader(table))
    assert len(rows) == 1

    return rows[0]


def test_the_folds_of_one_training_become_one_results_row(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """A k-fold training must yield one row carrying the spread, not k rows carrying none."""
    with open_dict(cfg_train):
        cfg_train.data.n_folds = 2
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    with open_dict(cfg_eval):
        cfg_eval.data.n_folds = 2
        cfg_eval.checkpoint_manifest_path = str(tmp_path / MANIFEST_FILENAME)
    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    assert {"fold_1/test/acc", "fold_2/test/acc"} <= set(metric_dict)
    row = _results_row(Path(cfg_eval.paths.results_dir))
    assert row["folds_used"] == "2"
    assert row["model"] == cfg_eval.algorithm.name
    assert 0.0 <= float(row["test_acc_mean"]) <= 1.0
    assert float(row["flops_mean"]) > 0.0


def test_a_fold_less_training_still_yields_a_row(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """With no folds the main run is what enters the table, so a search run is still reported."""
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = str(tmp_path / MANIFEST_FILENAME)
    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    assert metric_dict["main/test/acc"] > 0.0
    assert _results_row(Path(cfg_eval.paths.results_dir))["folds_used"] == "1"


def test_a_single_checkpoint_can_be_evaluated_on_its_own(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """Testing one checkpoint by hand must not need a manifest to be written first."""
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "main" / "last.ckpt")
    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    assert metric_dict["main/test/acc"] > 0.0


def test_an_evaluation_with_nothing_to_test_is_refused(cfg_eval: DictConfig) -> None:
    """A mistyped manifest path must fail here rather than report a run that tested nothing."""
    HydraConfig().set_config(cfg_eval)
    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = None

    with pytest.raises(ValueError, match="Nothing to evaluate"):
        evaluate(cfg_eval)
