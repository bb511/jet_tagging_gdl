"""The evaluation entry point, from a training run's manifest and from a single checkpoint.

The fixtures that fit a model are the slowest tests in the suite as well as the only ones that
exercise the handover from train.py to eval.py: the manifest, the shared results table and the row
it writes. The baseline comparison is matched to those runs fold by fold, and the matching itself
is checked on hand-written manifests, which need no training at all.
"""

import csv
import logging
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import _baseline_checkpoints, _baseline_for, evaluate
from src.train import train
from src.utils.checkpoints import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestRun,
    write_manifest,
)
from tests.conftest import needs_record


def _results_row(results_dir: Path) -> dict[str, str]:
    """The single row the shared results table holds after an evaluation."""
    tables = list(results_dir.rglob("results_table_*.csv"))
    assert len(tables) == 1, f"expected one results table, found {tables}"
    with open(tables[0], newline="") as table:
        rows = list(csv.DictReader(table))
    assert len(rows) == 1

    return rows[0]


def _metric(metrics: dict[str, object], key: str) -> float:
    """One metric of the evaluation's return value, whether it is a tensor or a number."""
    value = metrics[key]

    return float(value.item()) if hasattr(value, "item") else float(value)


def _manifest_of(tmp_path: Path, labels: list[str]) -> Path:
    """A manifest naming one empty checkpoint per label, which is all the matching reads."""
    runs = []
    for idx, label in enumerate(labels):
        checkpoint = tmp_path / f"{label}.ckpt"
        checkpoint.touch()
        fold_idx = None if label == "main" else idx
        runs.append(ManifestRun(label, fold_idx, str(checkpoint)))
    manifest = Manifest(len(labels), str(tmp_path), "test", runs)

    return write_manifest(manifest, tmp_path / "baseline")


def test_the_baseline_is_matched_to_the_run_fold_by_fold(
    cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """kfold_seed fixes the splits, so fold i of two regimes is a comparison of like with like."""
    path = _manifest_of(tmp_path, ["main", "fold_1", "fold_2"])
    with open_dict(cfg_eval):
        cfg_eval.baseline_checkpoint_manifest_path = str(path)

    baselines = _baseline_checkpoints(cfg_eval)

    assert set(baselines) == {"main", "fold_1", "fold_2"}
    assert baselines["fold_2"].endswith("fold_2.ckpt")
    run = ManifestRun("fold_2", 2, str(tmp_path / "own.ckpt"))
    assert _baseline_for(baselines, run) == baselines["fold_2"]


def test_a_fold_the_baseline_never_trained_is_scored_without_deltas(caplog) -> None:
    """Failing instead would throw away four good folds because the fifth has no partner."""
    run = ManifestRun("fold_5", 5, "own.ckpt")

    with caplog.at_level(logging.WARNING):
        assert _baseline_for({"fold_1": "base.ckpt"}, run) is None

    assert "fold_5" in caplog.text


def test_an_evaluation_with_no_baseline_configured_compares_against_nothing(
    cfg_eval: DictConfig,
) -> None:
    """The default is null, and a baseline run's own evaluation must leave it that way."""
    assert cfg_eval.baseline_checkpoint_manifest_path is None
    assert _baseline_checkpoints(cfg_eval) == {}
    assert _baseline_for({}, ManifestRun("main", None, "own.ckpt")) is None


def test_a_baseline_path_naming_no_manifest_is_refused(
    cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """A mistyped baseline must fail here rather than write a row of silent NaNs."""
    with open_dict(cfg_eval):
        cfg_eval.baseline_checkpoint_manifest_path = str(tmp_path / "nowhere.json")

    with pytest.raises(FileNotFoundError):
        _baseline_checkpoints(cfg_eval)


@pytest.mark.slow
@needs_record
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


@pytest.mark.slow
@needs_record
def test_a_run_scored_against_its_own_checkpoints_reports_deltas_of_zero(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    """The baseline is rebuilt from its own checkpoint, so the same one must score identically."""
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    manifest = str(tmp_path / MANIFEST_FILENAME)
    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = manifest
        cfg_eval.baseline_checkpoint_manifest_path = manifest
    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    assert _metric(metric_dict, "main/delta_test_acc") == pytest.approx(0.0, abs=1e-6)
    assert _metric(metric_dict, "main/invariance/so2_delta_flip") == pytest.approx(0.0, abs=1e-6)
    # baseline/test_acc counts jets, whereas test/acc is the macro average over the five classes,
    # so the two coincide on a balanced test set alone.
    assert 0.0 <= _metric(metric_dict, "main/baseline/test_acc") <= 1.0
    row = _results_row(Path(cfg_eval.paths.results_dir))
    assert row["delta_test_acc_mean"] == "0.0"
    # This run was trained on no group at all, so its per-group delta family stays blank.
    assert row["delta_flip_mean"] == "nan"


@pytest.mark.slow
@needs_record
def test_a_fold_less_training_is_evaluated_from_its_manifest_or_from_one_checkpoint(
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

    # Testing one checkpoint by hand must not need a manifest to be written first.
    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "main" / "last.ckpt")
    metric_dict, _ = evaluate(cfg_eval)

    assert metric_dict["main/test/acc"] > 0.0


@pytest.mark.slow
@needs_record
def test_an_evaluation_with_nothing_to_test_is_refused(cfg_eval: DictConfig) -> None:
    """A mistyped manifest path must fail here rather than report a run that tested nothing."""
    HydraConfig().set_config(cfg_eval)
    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = None

    with pytest.raises(ValueError, match="Nothing to evaluate"):
        evaluate(cfg_eval)
