import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.data.hls4ml_datamodule import HLS4MLDataModule, fold_indices
from src.data.hls4ml_jet_level_datamodule import JetLevelDataModule, shard_paths
from src.eval import evaluate
from src.train import train
from tests.conftest import (
    CACHE_DIR,
    JET_LEVEL_COLUMNS,
    JET_LEVEL_SIZES,
    ROOT,
    compose_root,
    needs_record,
    point_into,
    results_header,
)

NO_ETA = [name for name in JET_LEVEL_COLUMNS if name != "j_eta"]
# The columns the jet-level rows fill: the test accuracy and the clean working points.
GROUPLESS_FOLDS = {
    "test_acc_folds",
    "rejection_es30_folds",
    "tpr_eb1_folds",
    "tpr_eb10_folds",
    "auc_folds",
}


def _raw(directory: Path, split: str, columns: list[str]) -> tuple[np.ndarray, ...]:
    # Read back independently of the datamodule, in float64 as it standardises.
    paths = sorted(directory.glob(f"{split}-*.parquet"))
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    x = np.stack([table[name].to_numpy() for name in columns], axis=1)
    return x.astype(np.float64), table["label"].to_numpy().astype(np.int64)


def _assert_served(dataset, split: tuple[np.ndarray, ...], mean, std) -> None:
    x, labels = split
    standardised = torch.from_numpy(((x - mean) / std).astype(np.float32))
    assert torch.equal(dataset.X, standardised.unsqueeze(1))
    assert torch.equal(dataset.y, torch.eye(5)[torch.from_numpy(labels)])


def _jet_level(shards: Path, **kwargs) -> JetLevelDataModule:
    return JetLevelDataModule(shards_dir=str(shards), **kwargs)


def test_the_row_reads_the_jet_columns_in_the_shards_order_less_those_excluded(
    jet_level_shards: Path,
) -> None:
    every = _jet_level(jet_level_shards)
    noeta = _jet_level(jet_level_shards, exclude=["j_eta"])

    assert every.feature_names == noeta.feature_names == JET_LEVEL_COLUMNS
    assert every.features == JET_LEVEL_COLUMNS and every.feature_indices() is None
    assert noeta.features == NO_ETA
    assert noeta.feature_indices() == [0, 1, *range(3, 53)]
    # What the base class and the callbacks read off a datamodule.
    assert (noeta.norm_scales, noeta.keep_full_features_at_test) == (None, False)


@pytest.mark.parametrize("exclude", [[], ["j_eta"]])
def test_every_split_and_fold_is_standardised_by_the_record_training_split(
    jet_level_shards: Path, exclude: list[str]
) -> None:
    columns = [name for name in JET_LEVEL_COLUMNS if name not in exclude]
    raw = {split: _raw(jet_level_shards, split, columns) for split in JET_LEVEL_SIZES}
    mean, std = raw["train"][0].mean(axis=0), raw["train"][0].std(axis=0)
    # j_ptfrac has no spread, so it is centred and left undivided.
    assert std[0] == 0.0
    std[0] = 1.0
    record = _jet_level(jet_level_shards, exclude=exclude)
    folded = _jet_level(jet_level_shards, exclude=exclude, n_folds=5, kfold_seed=7)
    folded.set_fold(3)
    record.setup()
    folded.setup()

    _assert_served(record.data_train, raw["train"], mean, std)
    _assert_served(record.data_val, raw["validation"], mean, std)
    pool = tuple(
        np.concatenate(arrays) for arrays in zip(raw["train"], raw["validation"])
    )
    train_idx, val_idx = fold_indices(len(pool[1]), 5, 7, 3)
    _assert_served(folded.data_train, tuple(a[train_idx] for a in pool), mean, std)
    _assert_served(folded.data_val, tuple(a[val_idx] for a in pool), mean, std)
    for datamodule in (record, folded):
        _assert_served(datamodule.data_test, raw["test"], mean, std)
    assert torch.all(record.data_train.X[..., 0] == 0.0)


def test_the_batches_are_one_slot_jets_and_the_test_split_keeps_its_tail(
    jet_level_shards: Path,
) -> None:
    datamodule = _jet_level(jet_level_shards, exclude=["j_eta"], batch_size=16)
    datamodule.setup()
    loaders = {
        "train": datamodule.train_dataloader(),
        "validation": datamodule.val_dataloader(),
        "test": datamodule.test_dataloader(),
    }

    shapes = {split: [x.shape for x, _ in loader] for split, loader in loaders.items()}

    assert shapes["train"] == [(16, 1, 52)] * (120 // 16)
    assert shapes["validation"] == [(16, 1, 52)] * (40 // 16)
    # Every test jet is scored: three full batches and the trailing two jets.
    assert shapes["test"] == [(16, 1, 52)] * 3 + [(2, 1, 52)]
    for x, y in loaders["test"]:
        assert x.dtype == y.dtype == torch.float32
        assert y.shape == (len(x), 5) and torch.all(y.sum(dim=1) == 1.0)


@pytest.mark.parametrize(
    "method, scheme",
    [("symm_loss", "none"), ("augmentation", "so2"), ("symm_loss", "lorentz")],
)
def test_every_regime_but_the_baseline_is_refused_before_anything_is_read(
    method: str, scheme: str, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="only the baseline"):
        _jet_level(tmp_path / "absent", method=method, augmentation_scheme=scheme)

    # Constructing reads nothing; setup() is the first to need the shards.
    datamodule = _jet_level(tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="No train shards"):
        datamodule.setup()


def test_an_exclusion_that_names_no_observable_is_refused(
    jet_level_shards: Path,
) -> None:
    datamodule = _jet_level(jet_level_shards, exclude=["j_eta", "j_rapidity"])

    with pytest.raises(ValueError, match="j_rapidity"):
        datamodule.setup()


def _row_config(config_name: str, shards: Path, run_dir: Path) -> DictConfig:
    # The experiment as the catalogues run it, on the synthetic shards and the CPU.
    cfg = compose_root(
        config_name,
        "experiment=hls4ml/jet_level_noeta52_150",
        "trainer=cpu",
        "logger=none",
    )
    with open_dict(cfg):
        cfg.data.shards_dir = str(shards)
        cfg.data.batch_size = 16
        cfg.trainer.max_epochs = 1
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
    return point_into(cfg, run_dir)


def _train_main_and_folds(cfg: DictConfig, run_dir: Path) -> None:
    # Three processes, laid out as submit_train.sh --folds lays them out.
    HydraConfig().set_config(cfg)
    train(cfg)
    for fold in (1, 2):
        cfg_fold = point_into(cfg.copy(), run_dir / f"fold_{fold}")
        with open_dict(cfg_fold):
            cfg_fold.data.n_folds = 2
            cfg_fold.fold = fold
        HydraConfig().set_config(cfg_fold)
        train(cfg_fold)


def test_a_jet_level_row_trains_its_folds_and_evaluates_into_one_results_row(
    jet_level_shards: Path, tmp_path: Path
) -> None:
    train_dir, eval_dir = tmp_path / "train", tmp_path / "eval"
    _train_main_and_folds(
        _row_config("train.yaml", jet_level_shards, train_dir), train_dir
    )
    cfg = _row_config("eval.yaml", jet_level_shards, eval_dir)
    with open_dict(cfg):
        cfg.data.n_folds = 2
        cfg.checkpoint_manifest_path = str(train_dir)
    HydraConfig().set_config(cfg)

    metric_dict, _ = evaluate(cfg)

    table = eval_dir / "results" / "jet_level" / "results_table_baseline.csv"
    with open(table, newline="") as handle:
        header = next(csv.reader(handle))
        handle.seek(0)
        (row,) = list(csv.DictReader(handle))
    assert header == results_header()
    assert (row["method"], row["model"]) == ("baseline", "jet_level_noeta52")
    assert (row["nconstituents"], row["folds_used"]) == ("150", "2")
    assert row["selected_inv_key"] == "invariance/mean_tv"
    for column in header:
        if column.endswith("_folds") and column not in GROUPLESS_FOLDS:
            assert row[column] == "nan;nan", column
    assert all("nan" not in row[column] for column in GROUPLESS_FOLDS)
    assert float(row["flops_mean"]) > 0.0 and float(row["ebops_mean"]) > 0.0
    assert "fold_1/performance/auc" in metric_dict
    assert not any("invariance/" in name for name in metric_dict)
    for label in ("main", "fold_1", "fold_2"):
        with np.load(eval_dir / "outputs" / f"{label}.npz") as stored:
            assert set(stored) == {"logits_clean", "labels"}
            assert stored["logits_clean"].shape == (JET_LEVEL_SIZES["test"], 5)


def _constituent_counts(directory: Path, split: str) -> np.ndarray:
    lengths = [
        pc.list_value_length(pq.read_table(path, columns=["j1_pt"])["j1_pt"])
        for path in shard_paths(directory, split)
    ]
    return np.concatenate([length.to_numpy() for length in lengths])


@pytest.mark.slow
@needs_record
def test_the_shards_hold_the_loaders_jets_in_its_order() -> None:
    # The folds select the networks' jets only if the shards hold them in the order of the
    # loader's cache, which the networks read. The record tests build this cache too.
    revision = OmegaConf.load(ROOT / "configs" / "data" / "jet_level.yaml").revision
    networks = HLS4MLDataModule(
        revision=revision, cache_dir=CACHE_DIR, nconstituents=16
    )
    networks.prepare_data()
    jet_level = JetLevelDataModule(revision=revision)
    jet_level.setup()
    served = {
        "train": jet_level.data_train,
        "validation": jet_level.data_val,
        "test": jet_level.data_test,
    }

    assert len(jet_level.feature_names) == 53 and "j_eta" in jet_level.feature_names
    assert [dataset.n for dataset in served.values()] == [496_000, 124_000, 260_000]
    for split, dataset in served.items():
        cached = networks.record.load(split)
        # A padded slot is an all-zero row, which the loader's normalisation, a division,
        # keeps at zero.
        counts = (cached.x != 0).any(dim=-1).sum(dim=1).numpy()
        stored = np.minimum(_constituent_counts(jet_level.shards_directory, split), 16)
        assert torch.equal(dataset.y.argmax(dim=1), cached.y.argmax(dim=1)), split
        assert np.array_equal(counts, stored), split
