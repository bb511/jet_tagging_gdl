import csv
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from mlflow import MlflowClient
from omegaconf import DictConfig, open_dict

from src.eval import _baseline_checkpoints, evaluate
from src.train import train
from src.utils.checkpoints import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestRun,
    merge_manifests,
    write_manifest,
)
from tests.conftest import mlflow_logger, needs_record, point_into


def _results_row(results_dir: Path) -> dict[str, str]:
    tables = list(results_dir.rglob("results_table_*.csv"))
    assert len(tables) == 1, f"expected one results table, found {tables}"
    with open(tables[0], newline="") as table:
        rows = list(csv.DictReader(table))
    assert len(rows) == 1
    return rows[0]


def _metric(metrics: dict[str, object], key: str) -> float:
    value = metrics[key]
    return float(value.item()) if hasattr(value, "item") else float(value)


def _manifest_of(tmp_path: Path, labels: list[str]) -> Path:
    # One empty checkpoint per label, which is all the matching reads.
    runs = []
    for idx, label in enumerate(labels):
        checkpoint = tmp_path / f"{label}.ckpt"
        checkpoint.touch()
        runs.append(ManifestRun(label, None if label == "main" else idx, str(checkpoint)))
    manifest = Manifest(len(labels), str(tmp_path), "test", runs)
    return write_manifest(manifest, tmp_path / "baseline")


def test_the_baseline_is_matched_to_the_run_fold_by_fold(
    cfg_eval: DictConfig, tmp_path: Path
) -> None:
    # The default is null, and a baseline run's own evaluation leaves it that way.
    assert cfg_eval.baseline_checkpoint_manifest_path is None
    assert _baseline_checkpoints(cfg_eval) == {}

    path = _manifest_of(tmp_path, ["main", "fold_1", "fold_2"])
    with open_dict(cfg_eval):
        cfg_eval.baseline_checkpoint_manifest_path = str(path)

    baselines = _baseline_checkpoints(cfg_eval)

    # kfold_seed fixes the splits, so fold i of two regimes is a comparison of like with like.
    assert set(baselines) == {"main", "fold_1", "fold_2"}
    assert baselines["fold_2"].endswith("fold_2.ckpt")


def test_a_baseline_path_naming_no_manifest_is_refused(
    cfg_eval: DictConfig, tmp_path: Path
) -> None:
    with open_dict(cfg_eval):
        cfg_eval.baseline_checkpoint_manifest_path = str(tmp_path / "nowhere.json")

    with pytest.raises(FileNotFoundError):
        _baseline_checkpoints(cfg_eval)


@pytest.mark.slow
@needs_record
def test_the_folds_of_one_training_become_one_results_row(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    # The main run and its two folds are three processes, laid out as submit_train.sh --folds does.
    HydraConfig().set_config(cfg_train)
    train(cfg_train)
    for fold in (1, 2):
        cfg_fold = point_into(cfg_train.copy(), tmp_path / f"fold_{fold}")
        with open_dict(cfg_fold):
            cfg_fold.data.n_folds = 2
            cfg_fold.fold = fold
        HydraConfig().set_config(cfg_fold)
        train(cfg_fold)

    with open_dict(cfg_eval):
        cfg_eval.data.n_folds = 2
        cfg_eval.checkpoint_manifest_path = str(tmp_path)  # a run dir: main merged with fold_*/
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
def test_each_evaluation_nests_under_the_training_run_of_its_checkpoint(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    with open_dict(cfg_train):
        cfg_train.logger = mlflow_logger(tmp_path)
    HydraConfig().set_config(cfg_train)
    train(cfg_train)
    cfg_fold = point_into(cfg_train.copy(), tmp_path / "fold_1")
    with open_dict(cfg_fold):
        cfg_fold.data.n_folds = 2
        cfg_fold.fold = 1
    HydraConfig().set_config(cfg_fold)
    train(cfg_fold)
    trained = {run.fold_label: run.mlflow_run_id for run in merge_manifests(tmp_path).runs}

    with open_dict(cfg_eval):
        cfg_eval.data.n_folds = 2
        cfg_eval.checkpoint_manifest_path = str(tmp_path)
        cfg_eval.logger = mlflow_logger(tmp_path)
    HydraConfig().set_config(cfg_eval)
    evaluate(cfg_eval)

    client = MlflowClient(f"file:{tmp_path}/mlruns")
    experiment = client.get_experiment_by_name("test")
    evals = client.search_runs([experiment.experiment_id], "params.run_stage = 'eval'")
    parents = {r.data.params["fold_idx"]: r.data.tags.get("mlflow.parentRunId") for r in evals}
    assert set(trained) == {"main", "fold_1"} and all(trained.values())
    assert parents == trained


@pytest.mark.slow
@needs_record
def test_a_run_scored_against_its_own_checkpoints_reports_deltas_of_zero(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
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
    # Both are the macro average over the classes present, so the same checkpoint reads the same.
    assert _metric(metric_dict, "main/baseline/test_acc") == pytest.approx(
        _metric(metric_dict, "main/test/acc"), abs=1e-6
    )
    row = _results_row(Path(cfg_eval.paths.results_dir))
    assert row["delta_test_acc_mean"] == "0.0"
    # Trained on no group, so the per-group delta family stays blank.
    assert row["delta_flip_mean"] == "nan"


@pytest.mark.slow
@needs_record
def test_a_fold_less_training_is_evaluated_from_its_manifest_or_from_one_checkpoint(
    cfg_train: DictConfig, cfg_eval: DictConfig, tmp_path: Path
) -> None:
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = str(tmp_path / MANIFEST_FILENAME)
    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    assert metric_dict["main/test/acc"] > 0.0
    assert _results_row(Path(cfg_eval.paths.results_dir))["folds_used"] == "1"

    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "last.ckpt")
    metric_dict, _ = evaluate(cfg_eval)

    assert metric_dict["main/test/acc"] > 0.0


@pytest.mark.slow
@needs_record
def test_an_evaluation_with_nothing_to_test_is_refused(cfg_eval: DictConfig) -> None:
    HydraConfig().set_config(cfg_eval)
    with open_dict(cfg_eval):
        cfg_eval.checkpoint_manifest_path = None
        cfg_eval.ckpt_path = None

    with pytest.raises(ValueError, match="Nothing to evaluate"):
        evaluate(cfg_eval)
