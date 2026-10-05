import json
from pathlib import Path
from typing import Any

import pytest
from hydra.core.hydra_config import HydraConfig
from mlflow import MlflowClient
from omegaconf import DictConfig, open_dict

from src.train import train
from src.utils.checkpoints import MANIFEST_FILENAME
from tests.conftest import mlflow_logger, needs_record, point_into


def _manifest_run(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text())
    assert len(manifest["runs"]) == 1
    return manifest["runs"][0]


def _fold_config(
    cfg_train: DictConfig, tmp_path: Path, fold: int, n_folds: int
) -> DictConfig:
    # A fold is its own process in its own run directory, as submit_train.sh --folds lays it out.
    cfg = point_into(cfg_train.copy(), tmp_path / f"fold_{fold}")
    with open_dict(cfg):
        cfg.data.n_folds = n_folds
        cfg.fold = fold
    HydraConfig().set_config(cfg)
    return cfg


@needs_record
def test_the_main_run_keeps_the_record_split_and_leaves_a_manifest_of_itself(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.n_folds = (
            2  # as the experiments set it: the folds' business, not main's
        )

    metric_dict, object_dict = train(cfg_train)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())
    run = manifest["runs"][0]

    assert manifest["n_folds"] == 0
    assert [(r["fold_label"], r["fold_idx"]) for r in manifest["runs"]] == [
        ("main", None)
    ]
    assert (
        Path(run["checkpoint_path"]).is_absolute()
        and Path(run["checkpoint_path"]).exists()
    )
    assert Path(run["checkpoint_path"]).parent == tmp_path / "checkpoints"
    assert run["mlflow_run_id"] is None
    assert (object_dict["datamodule"].n_folds, object_dict["datamodule"].fold_idx) == (
        1,
        0,
    )
    # Optuna reads what train returns, the clean split's number.
    assert float(metric_dict["val/acc_best"]) == run["val_acc_best"]


@needs_record
def test_a_fold_nests_under_the_main_run_it_is_given(
    cfg_train: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.logger = mlflow_logger(tmp_path)
    train(cfg_train)
    main = _manifest_run(tmp_path)

    cfg = _fold_config(cfg_train, tmp_path, fold=2, n_folds=2)
    with open_dict(cfg):
        cfg.parent_run_id = main[
            "mlflow_run_id"
        ]  # what submit_train.sh reads off the manifest
    train(cfg)
    manifest = json.loads((tmp_path / "fold_2" / MANIFEST_FILENAME).read_text())
    fold = manifest["runs"][0]

    assert manifest["n_folds"] == 2
    assert [(r["fold_label"], r["fold_idx"]) for r in manifest["runs"]] == [
        ("fold_2", 2)
    ]
    client = MlflowClient(f"file:{tmp_path}/mlruns")
    assert main["mlflow_run_id"] and fold["mlflow_run_id"]
    assert "mlflow.parentRunId" not in client.get_run(main["mlflow_run_id"]).data.tags
    fold_run = client.get_run(fold["mlflow_run_id"])
    assert fold_run.data.tags["mlflow.parentRunId"] == main["mlflow_run_id"]
    assert fold_run.info.run_name == "fold_2"


@pytest.mark.parametrize("n_folds, fold", [(1, 1), (2, 0), (2, 3)])
def test_a_fold_without_folds_or_past_them_is_refused(
    cfg_train: DictConfig, n_folds: int, fold: int
) -> None:
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.n_folds = n_folds
        cfg_train.fold = fold

    with pytest.raises(ValueError, match="fold="):
        train(cfg_train)
