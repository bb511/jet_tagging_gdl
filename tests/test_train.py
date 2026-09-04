"""The training entry point, run against the real record.

These are the only tests that fit a model on real jets, so they are gated on the record's hub
snapshot being present. What they check is what nothing else can: that the folds are fitted and
recorded, and that the number an Optuna trial reads back is the main run's.
"""

import json
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.train import train
from src.utils.checkpoints import MANIFEST_FILENAME
from tests.conftest import needs_record
from tests.helpers.run_if import RunIf

pytestmark = needs_record


def test_a_fast_dev_run_reaches_the_end_of_the_loop(cfg_train: DictConfig) -> None:
    """One batch through fit is the cheapest proof that the whole stack composes and runs."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True

    metric_dict, object_dict = train(cfg_train)

    assert "train/loss" in metric_dict
    assert object_dict["algorithm"].flops_total > 0


def test_two_folds_leave_a_manifest_of_three_runs(cfg_train: DictConfig, tmp_path: Path) -> None:
    """eval.py finds its checkpoints only here: the main run, and one per fold.

    The directories and the top-level metrics are checked in the same fit, since three fits of the
    real record are the expensive part of this file.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.n_folds = 2

    metric_dict, _ = train(cfg_train)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())

    assert manifest["n_folds"] == 2
    assert [run["fold_label"] for run in manifest["runs"]] == ["main", "fold_1", "fold_2"]
    assert manifest["runs"][0]["fold_idx"] is None
    assert all(Path(run["checkpoint_path"]).is_absolute() for run in manifest["runs"])
    assert all(Path(run["checkpoint_path"]).exists() for run in manifest["runs"])
    assert "fold1/val/acc" in metric_dict and "fold2/val/acc" in metric_dict
    # One directory per run: a shared one would have fold 2 overwrite fold 1's best.
    assert {path.name for path in (tmp_path / "checkpoints").iterdir()} == {
        "main",
        "fold_1",
        "fold_2",
    }
    # Optuna maximises the top level, and must get the clean split's number, not a fold's.
    assert metric_dict["val/acc_best"] == metric_dict["main/val/acc_best"]
    assert metric_dict["val/acc"] == metric_dict["main/val/acc"]


def test_one_fold_alone_leaves_a_manifest_of_that_fold(
    cfg_train: DictConfig, tmp_path: Path
) -> None:
    """One fold per process is what lets k folds run side by side: under fold=<i> the main run is
    skipped, the datamodule serves fold i of n, and the manifest names that fold alone for
    merge_manifests to join."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.n_folds = 2
        cfg_train.fold = 2

    metric_dict, object_dict = train(cfg_train)
    manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text())

    assert manifest["n_folds"] == 2
    assert [(run["fold_label"], run["fold_idx"]) for run in manifest["runs"]] == [("fold_2", 2)]
    assert {path.name for path in (tmp_path / "checkpoints").iterdir()} == {"fold_2"}
    assert (object_dict["datamodule"].n_folds, object_dict["datamodule"].fold_idx) == (2, 1)
    assert "main/val/acc" not in metric_dict
    # With no main run in the process, the fold's own number is what the top level carries.
    assert metric_dict["val/acc_best"] == metric_dict["fold2/val/acc_best"]


@pytest.mark.parametrize("n_folds, fold", [(1, 1), (2, 0), (2, 3)])
def test_a_fold_without_folds_or_past_them_is_refused(
    cfg_train: DictConfig, n_folds: int, fold: int
) -> None:
    """Refused before the data is touched: each would fit something and label it as a fold it is
    not."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.data.n_folds = n_folds
        cfg_train.fold = fold

    with pytest.raises(ValueError, match="fold="):
        train(cfg_train)


@RunIf(min_gpus=1)
def test_a_fast_dev_run_reaches_the_end_of_the_loop_on_a_gpu(cfg_train: DictConfig) -> None:
    """The trainings run on GPU, where a dtype or a device mismatch shows up and nowhere else."""
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"

    train(cfg_train)


@pytest.mark.slow
def test_a_run_resumes_from_its_own_last_checkpoint(cfg_train: DictConfig, tmp_path: Path) -> None:
    """A queued training that dies overnight is restarted from the checkpoint it left."""
    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    # Progress is read off the checkpoint, not the file names: save_top_k: 1 on val/acc writes a
    # new epoch_*.ckpt only on an improvement, which one epoch of a few batches need not give.
    last = tmp_path / "checkpoints" / "main" / "last.ckpt"
    before = torch.load(last, weights_only=False)

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(last)
        cfg_train.trainer.max_epochs = 2

    train(cfg_train)
    after = torch.load(last, weights_only=False)

    # The second epoch was trained on top of the first, not in place of it.
    assert after["epoch"] > before["epoch"]
    assert after["global_step"] > before["global_step"]
