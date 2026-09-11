import csv
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from mlflow import MlflowClient
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train
from src.utils.checkpoints import MANIFEST_FILENAME, merge_manifests
from tests.conftest import mlflow_logger, needs_record, point_into


def _results_row(results_dir: Path) -> dict[str, str]:
    tables = list(results_dir.rglob("results_table_*.csv"))
    assert len(tables) == 1, f"expected one results table, found {tables}"
    with open(tables[0], newline="") as table:
        rows = list(csv.DictReader(table))
    assert len(rows) == 1
    return rows[0]


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
