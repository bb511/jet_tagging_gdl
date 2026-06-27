import json
import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import evaluate
from src.train import train


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Legacy: train 1 epoch then evaluate via single ckpt_path."""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, _ = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints" / "main")

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = str(tmp_path / "checkpoints" / "main" / "last.ckpt")

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0


@pytest.mark.slow
def test_eval_manifest_mode(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Manifest mode: train, then eval reads the checkpoint manifest."""
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1

    HydraConfig().set_config(cfg_train)
    train(cfg_train)

    manifest_path = tmp_path / "checkpoint_manifest.json"
    assert manifest_path.exists(), "manifest not written by train"

    with open_dict(cfg_eval):
        cfg_eval.ckpt_path = None
        cfg_eval.checkpoint_manifest_path = str(manifest_path)

    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    # manifest mode returns metrics namespaced by fold_label
    assert any(k.startswith("main/") for k in metric_dict), (
        f"Expected 'main/*' metrics in manifest eval output, got: {list(metric_dict.keys())}"
    )
