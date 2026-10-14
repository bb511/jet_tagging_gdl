from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf, open_dict

from src.algorithms.bdt import BoostedTrees
from src.utils.checkpoints import (
    MANIFEST_FILENAME,
    Manifest,
    ManifestRun,
    merge_manifests,
    read_manifest,
    write_manifest,
)
from tests.conftest import (
    CPU_ROWS,
    JET_LEVEL_SIZES,
    compose_root,
    mlflow_logger,
    needs_record,
    point_into,
    read_results_row,
    results_header,
    train_main_and_folds,
)

TREES = 10
# The columns a jet-level row fills: the test accuracy and the clean working points.
GROUPLESS_FOLDS = {
    "test_acc_folds",
    "rejection_es30_folds",
    "tpr_eb1_folds",
    "tpr_eb10_folds",
    "auc_folds",
}


@pytest.fixture(scope="module")
def trained(cfg_train_global: DictConfig, tmp_path_factory: pytest.TempPathFactory):
    # The main run and fold 1 of 2 on the record, logged to MLflow; data.n_folds is 2 as
    # the experiments set it, the folds' business and not the main run's.
    run_dir = tmp_path_factory.mktemp("train")
    cfg = point_into(cfg_train_global.copy(), run_dir)
    with open_dict(cfg):
        cfg.logger = mlflow_logger(run_dir)
        cfg.data.n_folds = 2
    return run_dir, train_main_and_folds(cfg)


@pytest.mark.slow
@needs_record
def test_training_leaves_a_manifest_per_run_and_nests_the_fold(trained) -> None:
    run_dir, ((metric_dict, objects), _) = trained
    main_manifest = read_manifest(run_dir / MANIFEST_FILENAME)
    fold_manifest = read_manifest(run_dir / "fold_1" / MANIFEST_FILENAME)
    (main,), (fold,) = main_manifest.runs, fold_manifest.runs
    checkpoint = Path(main.checkpoint_path)
    client = MlflowClient(f"file:{run_dir}/mlruns")
    fold_run = client.get_run(fold.mlflow_run_id)

    assert (main_manifest.n_folds, main.fold_label, main.fold_idx) == (0, "main", None)
    assert (fold_manifest.n_folds, fold.fold_label, fold.fold_idx) == (2, "fold_1", 1)
    assert checkpoint.is_absolute() and checkpoint.parent == run_dir / "checkpoints"
    assert checkpoint.exists()
    # The main run keeps the record's own split, whatever data.n_folds says.
    assert (objects["datamodule"].n_folds, objects["datamodule"].fold_idx) == (1, 0)
    # Optuna reads what train returns, the clean split's number.
    assert float(metric_dict["val/acc_best"]) == main.val_acc_best
    assert "mlflow.parentRunId" not in client.get_run(main.mlflow_run_id).data.tags
    assert fold_run.data.tags["mlflow.parentRunId"] == main.mlflow_run_id
    assert fold_run.info.run_name == "fold_1"


@pytest.mark.slow
@needs_record
@pytest.mark.parametrize("source", ["run_dir", "manifest", "checkpoint"])
def test_evaluation_writes_one_row_from_the_folds_alone(
    trained, cfg_eval: DictConfig, tmp_path: Path, source: str
) -> None:
    from src.eval import evaluate

    run_dir, _ = trained
    trained_ids = {
        run.fold_label: run.mlflow_run_id for run in merge_manifests(run_dir).runs
    }
    # A directory merges the main run with every fold_*/ beside it, a manifest file
    # tests its run alone, and a checkpoint is tested as an unnamed main run.
    name, path = {
        "run_dir": ("checkpoint_manifest_path", run_dir),
        "manifest": ("checkpoint_manifest_path", run_dir / MANIFEST_FILENAME),
        "checkpoint": ("ckpt_path", run_dir / "checkpoints" / "last.ckpt"),
    }[source]
    with open_dict(cfg_eval):
        cfg_eval[name] = str(path)
        cfg_eval.logger = mlflow_logger(tmp_path)
    HydraConfig().set_config(cfg_eval)

    metric_dict, _ = evaluate(cfg_eval)

    tested = ["main", "fold_1"] if source == "run_dir" else ["main"]
    _, row = read_results_row(Path(cfg_eval.paths.results_dir))
    stored = set()
    for label in tested:
        with np.load(
            Path(cfg_eval.paths.output_dir) / "outputs" / f"{label}.npz"
        ) as arrays:
            stored |= {array.shape[0] for array in arrays.values()}
    client = MlflowClient(f"file:{tmp_path}/mlruns")
    experiment = client.get_experiment_by_name("test").experiment_id
    evals = client.search_runs([experiment], "params.run_stage = 'eval'")
    parents = {
        r.data.params["fold_idx"]: r.data.tags.get("mlflow.parentRunId") for r in evals
    }

    assert sorted({name.split("/")[0] for name in metric_dict}) == sorted(tested)
    # The folds alone fill the row whenever they exist, so that no jet counts twice.
    accuracy = float(metric_dict[f"{tested[-1]}/test/acc"])
    assert (row["model"], row["folds_used"]) == (cfg_eval.algorithm.name, "1")
    assert row["test_acc_folds"] == f"{accuracy:.6f}"
    assert float(row["flops_mean"]) > 0.0 and float(row["ebops_mean"]) > 0.0
    # Every checkpoint leaves its outputs, the main run's included, over the same jets.
    assert len(stored) == 1
    expected = {
        label: None if source == "checkpoint" else trained_ids[label]
        for label in tested
    }
    assert parents == expected


@pytest.mark.parametrize(
    "entry, overrides, match",
    [
        ("train", {"data.n_folds": 1, "fold": 1}, "fold="),
        ("train", {"data.n_folds": 2, "fold": 0}, "fold="),
        ("train", {"data.n_folds": 2, "fold": 3}, "fold="),
        (
            "eval",
            {"checkpoint_manifest_path": None, "ckpt_path": None},
            "Nothing to evaluate",
        ),
    ],
)
def test_the_entry_points_refuse_what_they_cannot_run(
    cfg_train: DictConfig, cfg_eval: DictConfig, entry: str, overrides: dict, match: str
) -> None:
    from src.eval import evaluate
    from src.train import train

    cfg, run = (cfg_train, train) if entry == "train" else (cfg_eval, evaluate)
    HydraConfig().set_config(cfg)
    for name, value in overrides.items():
        OmegaConf.update(cfg, name, value)

    with pytest.raises(ValueError, match=match):
        run(cfg)


def test_the_evaluation_selects_on_validation_jets_one_checkpoint_at_a_time(
    cfg_eval: DictConfig, tmp_path: Path
) -> None:
    from src.eval import evaluate

    # A main run and its fold validated on different jets, so no one datamodule serves both.
    picks = tmp_path / "picks"
    for run, out in (
        (ManifestRun("main", None, "main.ckpt"), picks),
        (ManifestRun("fold_1", 1, "fold.ckpt"), picks / "fold_1"),
    ):
        write_manifest(
            Manifest(int(run.fold_idx is not None), str(out), "x", [run]), out
        )
    HydraConfig().set_config(cfg_eval)
    OmegaConf.update(cfg_eval, "checkpoint_manifest_path", str(picks))
    OmegaConf.update(cfg_eval, "data.test_split", "validation")

    with pytest.raises(ValueError, match="one checkpoint"):
        evaluate(cfg_eval)


def _row_config(config_name: str, row: str, shards: Path, run_dir: Path) -> DictConfig:
    # The experiment as the catalogues run it, on the synthetic shards and the CPU.
    overrides = (f"experiment=hls4ml/{row}_150", "trainer=cpu", "logger=none")
    cfg = compose_root(config_name, *overrides)
    with open_dict(cfg):
        cfg.data.shards_dir = str(shards)
        cfg.data.batch_size = 16
        cfg.trainer.max_epochs = 2
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
    return point_into(cfg, run_dir)


@pytest.mark.parametrize("row", ["jet_level_noeta52", "jet_level_bdt_noeta52"])
def test_a_jet_level_row_trains_and_evaluates_into_one_row(
    row: str, jet_level_shards: Path, tmp_path: Path
) -> None:
    from src.eval import evaluate

    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    train_main_and_folds(_row_config("train.yaml", row, jet_level_shards, train_dir))
    cfg = _row_config("eval.yaml", row, jet_level_shards, eval_dir)
    with open_dict(cfg):
        cfg.checkpoint_manifest_path = str(train_dir)
    HydraConfig().set_config(cfg)

    metric_dict, _ = evaluate(cfg)

    header, result = read_results_row(eval_dir / "results")
    costs = (result["flops_mean"], result["ebops_mean"])
    assert header == results_header()
    assert (result["method"], result["model"], result["nconstituents"]) == (
        "baseline",
        row,
        "150",
    )
    # The one fold fills the row; the observables are never transformed.
    assert result["folds_used"] == "1"
    for column in header:
        if column.endswith("_folds"):
            assert (result[column] == "nan") == (column not in GROUPLESS_FOLDS), column
    # The trees run in scikit-learn, which neither counter sees.
    assert costs == ("nan", "nan") if row in CPU_ROWS else min(map(float, costs)) > 0
    assert "fold_1/performance/auc" in metric_dict
    assert not any("invariance/" in name for name in metric_dict)
    for label in ("main", "fold_1"):
        with np.load(eval_dir / "outputs" / f"{label}.npz") as stored:
            assert set(stored) == {"logits_clean", "labels"}
            assert stored["logits_clean"].shape == (JET_LEVEL_SIZES["test"], 5)
    for run in merge_manifests(train_dir).runs if row in CPU_ROWS else ():
        # The checkpoint of a run's best epoch holds the trees fitted up to that epoch.
        checkpoint = torch.load(run.checkpoint_path, weights_only=False)
        trees = BoostedTrees(n_class=5, trees_per_epoch=TREES, random_state=0)
        state = checkpoint["state_dict"]["net._extra_state"]
        trees.load_state_dict({"_extra_state": state})
        assert trees.estimator.n_iter_ == (checkpoint["epoch"] + 1) * TREES
