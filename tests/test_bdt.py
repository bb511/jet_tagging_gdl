import csv
import pickle
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from lightning import Callback, LightningModule, Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from src.algorithms.bdt import BoostedTreesClassifier
from src.algorithms.components.bdt import BoostedTrees
from src.data.hls4ml_jet_level_datamodule import JetLevelDataModule
from src.eval import evaluate
from src.train import train
from src.utils.checkpoints import merge_manifests
from tests.conftest import (
    CPU_ROWS,
    JET_LEVEL_SIZES,
    ROOT,
    compose_root,
    mlflow_logger,
    point_into,
    results_header,
)
from tests.test_jet_level import GROUPLESS_FOLDS

TREES = 10
SPACE = {
    "algorithm.learning_rate": "tag(log, interval(0.01, 0.3))",
    "algorithm.max_leaf_nodes": "choice(15, 31, 63, 127, 255)",
    "algorithm.max_depth": "choice(0, 6, 12)",
    "algorithm.min_samples_leaf": "choice(20, 100, 500)",
    "algorithm.l2_regularization": "choice(0.0, 0.1, 1.0)",
    "algorithm.max_features": "choice(0.5, 1.0)",
}


def _jets(n_jets: int = 200, n_features: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    # One-slot jets whose class shifts the first observable, so that trees separate them.
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(n_jets) % 5
    x = torch.randn(n_jets, 1, n_features, generator=generator)
    x[:, 0, 0] += 2.0 * labels
    return x, labels


def _trees(**kwargs) -> BoostedTrees:
    return BoostedTrees(n_class=5, trees_per_epoch=TREES, random_state=0, **kwargs)


def test_the_trees_predict_uniformly_before_their_first_fit() -> None:
    x, _ = _jets()

    log_p = _trees()(x)

    assert log_p.dtype == torch.float32
    assert torch.equal(log_p, torch.zeros(len(x), 5))


def test_each_fit_adds_trees_per_epoch_iterations_of_finite_log_probabilities() -> None:
    x, labels = _jets()
    trees = _trees()

    for epoch in range(1, 4):
        trees.fit_more(x, labels)
        assert trees.estimator.n_iter_ == epoch * TREES
    log_p = trees(x)

    assert log_p.shape == (len(x), 5) and log_p.dtype == torch.float32
    assert torch.isfinite(log_p).all() and (log_p <= 0.0).all()
    # Probabilities up to the floor of 1e-7 on each of the five.
    sums = log_p.exp().sum(dim=1)
    torch.testing.assert_close(sums, torch.ones(len(x)), atol=1e-5, rtol=0.0)
    assert (log_p.argmax(dim=1) == labels).float().mean() > 0.9


def test_three_fits_of_the_same_jets_grow_the_trees_of_one_fit_of_them_all() -> None:
    # Why the module fits the whole training split every epoch: refitted on the same
    # jets, the bin edges are the same and the earlier trees read as they were grown.
    # Half the features per split, so that the subsampling generator is carried over.
    x, labels = _jets()
    grown, whole = _trees(max_features=0.5), _trees(max_features=0.5)

    for _ in range(3):
        grown.fit_more(x, labels)
    whole.trees_per_epoch = 3 * TREES
    whole.fit_more(x, labels)

    torch.testing.assert_close(grown(x), whole(x))


def test_the_estimator_travels_in_the_state_dict() -> None:
    x, labels = _jets()
    trees, restored = _trees(), _trees()
    trees.fit_more(x, labels)

    state = trees.state_dict()
    restored.load_state_dict(state)

    assert set(state) == {"_extra_state"}
    assert restored.estimator.n_iter_ == TREES
    assert torch.equal(restored(x), trees(x))


@pytest.mark.parametrize("max_depth, expected", [(None, None), (0, None), (6, 6)])
def test_a_depth_of_zero_or_none_sets_no_limit(
    max_depth: int | None, expected: int | None
) -> None:
    # 0 is the search's spelling: the sweeper writes a None choice as the string 'None'.
    assert _trees(max_depth=max_depth).estimator.max_depth == expected


class _IterationsAtValidation(Callback):
    def __init__(self) -> None:
        self.seen: list[int] = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if not trainer.sanity_checking:
            self.seen.append(pl_module.net.estimator.n_iter_)


def test_each_epoch_is_validated_on_the_trees_fitted_in_it(
    jet_level_shards: Path, tmp_path: Path
) -> None:
    # Lightning validates an epoch before on_train_epoch_end, so the trees are added on
    # its last batch for epoch k to be scored on (k + 1) * trees_per_epoch iterations.
    module = BoostedTreesClassifier(name="bdt", nconstituents=1, nbr_feats=53)
    datamodule = JetLevelDataModule(shards_dir=str(jet_level_shards), batch_size=16)
    recorder = _IterationsAtValidation()
    trainer = Trainer(
        max_epochs=3,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        callbacks=[recorder],
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    trainer.fit(module, datamodule=datamodule)

    assert recorder.seen == [TREES, 2 * TREES, 3 * TREES]
    # One mock step per batch, as the networks take one optimiser step per batch.
    assert trainer.global_step == 3 * (JET_LEVEL_SIZES["train"] // 16)


def _row_config(config_name: str, row: str, shards: Path, run_dir: Path) -> DictConfig:
    # The experiment as the catalogues run it, with its own CPU trainer and the default
    # callbacks, on the synthetic shards; MLflow logs into the test's directory.
    cfg = compose_root(config_name, f"experiment=hls4ml/{row}_150")
    with open_dict(cfg):
        cfg.data.shards_dir = str(shards)
        cfg.data.batch_size = 16
        cfg.trainer.max_epochs = 3
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
        cfg.logger = mlflow_logger(run_dir.parent)
    return point_into(cfg, run_dir)


def _iterations(trained: tuple[dict, dict]) -> int:
    return trained[1]["algorithm"].net.estimator.n_iter_


def _train_main_and_fold(cfg: DictConfig, run_dir: Path) -> list[int]:
    # The main run and fold 1 of 3, laid out as submit_train.sh --folds lays them out;
    # the iterations each run's estimator ends with.
    HydraConfig().set_config(cfg)
    grown = [_iterations(train(cfg))]
    cfg_fold = point_into(cfg.copy(), run_dir / "fold_1")
    with open_dict(cfg_fold):
        cfg_fold.data.n_folds = 3
        cfg_fold.fold = 1
    HydraConfig().set_config(cfg_fold)
    return grown + [_iterations(train(cfg_fold))]


def _assert_best_checkpoints(train_dir: Path) -> None:
    # The checkpoint of a run's best epoch holds the trees fitted up to that epoch.
    for run in merge_manifests(train_dir).runs:
        checkpoint = torch.load(run.checkpoint_path, weights_only=False)
        estimator = pickle.loads(checkpoint["state_dict"]["net._extra_state"])
        assert run.val_acc_best is not None and run.val_acc_best > 0.0
        assert estimator.n_iter_ == (checkpoint["epoch"] + 1) * TREES


def _results_row(eval_dir: Path) -> tuple[list[str], dict[str, str]]:
    table = eval_dir / "results" / "jet_level" / "results_table_baseline.csv"
    with open(table, newline="") as handle:
        header = next(csv.reader(handle))
        handle.seek(0)
        (row,) = list(csv.DictReader(handle))
    return header, row


def test_a_bdt_row_trains_its_runs_and_evaluates_into_one_results_row(
    jet_level_shards: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    name, train_dir, eval_dir = CPU_ROWS[1], tmp_path / "train", tmp_path / "eval"
    grown = _train_main_and_fold(
        _row_config("train.yaml", name, jet_level_shards, train_dir), train_dir
    )
    cfg = _row_config("eval.yaml", name, jet_level_shards, eval_dir)
    with open_dict(cfg):
        cfg.checkpoint_manifest_path = str(train_dir)
    HydraConfig().set_config(cfg)

    evaluate(cfg)

    assert grown == [3 * TREES, 3 * TREES]
    _assert_best_checkpoints(train_dir)
    header, row = _results_row(eval_dir)
    assert header == results_header()
    assert (row["method"], row["model"]) == ("baseline", name)
    # The folds alone fill the row, here the one fold trained.
    assert (row["nconstituents"], row["folds_used"]) == ("150", "1")
    assert (row["flops_mean"], row["ebops_mean"]) == ("nan", "nan")
    for column in header:
        if column.endswith("_folds") and column not in GROUPLESS_FOLDS:
            assert row[column] == "nan", column
    assert all("nan" not in row[column] for column in GROUPLESS_FOLDS)
    for label in ("main", "fold_1"):
        with np.load(eval_dir / "outputs" / f"{label}.npz") as stored:
            assert set(stored) == {"logits_clean", "labels"}
            assert stored["logits_clean"].shape == (JET_LEVEL_SIZES["test"], 5)


def _algorithm_yaml(name: str) -> dict:
    path = ROOT / "configs" / "algorithm" / f"{name}.yaml"
    return OmegaConf.to_container(OmegaConf.load(path))


def test_the_two_bdt_rows_differ_in_name_and_width_alone() -> None:
    # Standalone copies, as the MLP jet-level rows are of mlp.yaml.
    every, noeta = (_algorithm_yaml(row) for row in CPU_ROWS)

    assert every["_target_"] == "src.algorithms.bdt.BoostedTreesClassifier"
    assert (every["name"], every["nbr_feats"]) == (CPU_ROWS[0], 53)
    assert (noeta["name"], noeta["nbr_feats"]) == (CPU_ROWS[1], 52)
    assert (every["optimizer"], every["scheduler"]) == (None, None)
    assert every["trees_per_epoch"] == TREES
    for cfg in (every, noeta):
        for key in ("name", "nbr_feats"):
            cfg.pop(key)
    assert every == noeta


def _sweeper(tmp_path: Path, row: str) -> dict:
    cfg = compose_root(
        "train.yaml",
        f"experiment=hls4ml/{row}_150",
        f"hparams_search=hls4ml/{row}_optuna",
        "logger=none",
    )
    HydraConfig().set_config(point_into(cfg, tmp_path))
    return OmegaConf.to_container(cfg.hydra.sweeper, resolve=True)


def test_the_two_bdt_searches_share_one_space_on_two_studies(tmp_path: Path) -> None:
    every, noeta = (_sweeper(tmp_path, row) for row in CPU_ROWS)

    for row, sweeper in zip(CPU_ROWS, (every, noeta)):
        storage = f"sqlite:///logs/optuna/hls4ml/{row}.db?timeout=600"
        assert sweeper.pop("storage") == storage
        assert sweeper.pop("study_name") == f"{row}_nconst150_augmentation_none"
    assert every == noeta
    assert (every["n_trials"], every["direction"]) == (200, "maximize")
    sampler = every["sampler"]
    assert (sampler["seed"], sampler["n_startup_trials"]) == (1234, 20)
    assert every["params"] == SPACE


@pytest.mark.parametrize("row", CPU_ROWS)
def test_each_bdt_experiment_runs_on_the_cpu_in_fp32_without_clipping(
    row: str, tmp_path: Path
) -> None:
    # Composed without a trainer override, so the experiment's own trainer is checked;
    # max_depth=0 is how a tuned block spells no limit.
    overrides = ("logger=none", "algorithm.max_depth=0")
    cfg = compose_root("train.yaml", f"experiment=hls4ml/{row}_150", *overrides)
    HydraConfig().set_config(point_into(cfg, tmp_path))

    trainer = hydra.utils.instantiate(cfg.trainer)
    module = hydra.utils.instantiate(cfg.algorithm)

    assert (cfg.trainer.accelerator, cfg.trainer.precision) == ("cpu", 32)
    assert trainer.precision == "32-true" and cfg.trainer.gradient_clip_val is None
    assert cfg.callbacks.invariance_metric.probe is False
    assert cfg.results_table.subdir == "jet_level"
    assert (cfg.data.n_folds, cfg.data.batch_size, cfg.trainer.max_epochs) == (
        5,
        2048,
        200,
    )
    assert list(cfg.data.exclude) == ([] if row == CPU_ROWS[0] else ["j_eta"])
    assert not module.automatic_optimization
    assert module.net.estimator.max_depth is None
