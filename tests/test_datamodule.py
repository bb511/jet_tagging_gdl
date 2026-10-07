from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.data import hls4ml_datamodule as datamodule_module
from src.data.components.augmentation import compute_eta
from src.data.components.dataset import HLS4MLDataset
from src.data.hls4ml_datamodule import (
    HLS4MLDataModule,
    compose_record_config,
    fold_indices,
)
from tests.conftest import RECORD_COLUMNS, ROOT, labels

SPLIT_SIZES = {"train": 40, "validation": 20, "test": 16}
NCONSTITUENTS = 8
ETAREL, PHIREL = RECORD_COLUMNS.index("j1_etarel"), RECORD_COLUMNS.index("j1_phirel")
# px, py, pz, e, pt, eta and phi: what the rebase leaves as the record stored it.
ABSOLUTE_COLUMNS = [0, 1, 2, 3, 5, 7, 10]


class FakeRecord:
    # The surface HLS4MLDataModule asks of loader.datamodule.JetData, over jets from a fixed seed.

    def __init__(self) -> None:
        self.prepared = False
        self.cache_folder = Path("/nowhere/nconst_8")
        self.overrides: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(RECORD_COLUMNS)

    @property
    def pt_idx(self) -> int:
        return RECORD_COLUMNS.index("j1_pt")

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        # None until prepare() has run, as the record's own property is.
        if not self.prepared:
            return None
        return {
            name: ("centered", 1.0 + idx) for idx, name in enumerate(RECORD_COLUMNS)
        }

    def prepare(self) -> None:
        self.prepared = True

    def load(self, split: str) -> SimpleNamespace:
        generator = torch.Generator().manual_seed(len(split))
        n_jets = SPLIT_SIZES[split]
        x = torch.randn(n_jets, NCONSTITUENTS, len(RECORD_COLUMNS), generator=generator)
        return SimpleNamespace(x=x, y=labels(n_jets, seed=len(split)))


@pytest.fixture
def record(monkeypatch, tmp_path: Path) -> FakeRecord:
    fake = FakeRecord()

    def remember(overrides: list[str]) -> FakeRecord:
        fake.overrides = list(overrides)
        return fake

    monkeypatch.setattr(
        datamodule_module, "snapshot_download", lambda *a, **k: str(tmp_path)
    )
    monkeypatch.setattr(
        datamodule_module,
        "compose_record_config",
        lambda root, overrides: SimpleNamespace(data=overrides),
    )
    monkeypatch.setattr(datamodule_module, "instantiate", remember)
    return fake


def _datamodule(**overrides) -> HLS4MLDataModule:
    defaults = {
        "cache_dir": "/nowhere/hf",
        "nconstituents": NCONSTITUENTS,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
    }
    return HLS4MLDataModule(**(defaults | overrides))


def _blank_the_jet_relative_columns(record: FakeRecord) -> None:
    # Zero on disk, so whatever the splits carry afterwards is the rebase's own work.
    loaded = record.load

    def load(split: str) -> SimpleNamespace:
        data = loaded(split)
        data.x[..., [ETAREL, PHIREL]] = 0.0
        return data

    record.load = load


def _kept_axis_etarel(x: torch.Tensor, scales: dict[str, float]) -> torch.Tensor:
    # In the record's own normalisation, and over every slot: the fake record pads none.
    px, py, pz = (x[..., i] * scales[RECORD_COLUMNS[i]] for i in (0, 1, 2))
    jet = [p.sum(dim=1, keepdim=True) for p in (px, py, pz)]
    return (compute_eta(px, py, pz) - compute_eta(*jet)) / scales["j1_etarel"]


def test_every_split_comes_out_of_setup_on_the_axis_of_the_constituents_kept(
    record: FakeRecord,
) -> None:
    _blank_the_jet_relative_columns(record)
    datamodule = _datamodule(features=RECORD_COLUMNS)
    datamodule.prepare_data()
    datamodule.setup()
    splits = {
        "train": datamodule.data_train,
        "validation": datamodule.data_val,
        "test": datamodule.data_test,
    }

    for split, dataset in splits.items():
        stored = FakeRecord().load(split).x
        expected = _kept_axis_etarel(stored, datamodule.norm_scales)
        assert torch.allclose(dataset.X[..., ETAREL], expected, atol=1e-6), split
        assert (dataset.X[..., PHIREL] != 0.0).all(), split
        assert torch.equal(
            dataset.X[..., ABSOLUTE_COLUMNS], stored[..., ABSOLUTE_COLUMNS]
        ), split


@pytest.mark.parametrize(
    "method, scheme, keep_full, fit_width, test_width",
    [
        ("augmentation", "none", True, 4, 16),
        ("augmentation", "none", False, 4, 4),
        ("augmentation", "so3", True, 4, 16),
        # The SCL transforms inside model_step, so its batches are full width whatever the flag.
        ("symm_loss", "so2", False, 16, 16),
    ],
)
def test_the_batch_widths_follow_the_method_and_the_test_flag(
    record: FakeRecord,
    method: str,
    scheme: str,
    keep_full: bool,
    fit_width: int,
    test_width: int,
) -> None:
    datamodule = _datamodule(
        method=method, augmentation_scheme=scheme, keep_full_features_at_test=keep_full
    )
    datamodule.setup()
    x_train, y_train = next(iter(datamodule.train_dataloader()))
    x_val, _ = next(iter(datamodule.val_dataloader()))
    x_test, _ = next(iter(datamodule.test_dataloader()))

    assert x_train.shape[1:] == (NCONSTITUENTS, fit_width)
    assert x_val.shape[-1] == fit_width
    assert x_test.shape == (8, NCONSTITUENTS, test_width)
    assert x_train.dtype == torch.float32
    assert y_train.shape == (x_train.shape[0], 5)


def test_an_scl_run_keeps_its_group_and_fills_the_callback_surface(
    record: FakeRecord,
) -> None:
    datamodule = _datamodule(method="symm_loss", augmentation_scheme="lorentz")
    datamodule.setup(stage="fit")

    # The results table and ${data.augmentation_scheme} must agree on what a run did.
    assert datamodule.augmentation_scheme == "lorentz"
    assert datamodule.data_train.transform is None
    assert datamodule.feature_names == RECORD_COLUMNS
    assert datamodule.features == ["px", "py", "pz", "e"]
    assert datamodule.keep_full_features_at_test is True
    assert datamodule.n_train_samples == SPLIT_SIZES["train"]
    assert datamodule.n_val_samples == SPLIT_SIZES["validation"]


def test_a_method_or_a_scheme_outside_the_domain_is_refused() -> None:
    with pytest.raises(ValueError, match="method"):
        _datamodule(method="equivariance")
    with pytest.raises(ValueError, match="augmentation_scheme"):
        _datamodule(augmentation_scheme="permute")


def test_the_features_are_found_by_name_and_the_scales_arrive_with_the_record(
    record: FakeRecord,
) -> None:
    datamodule = _datamodule()

    assert datamodule.feature_indices() == [0, 1, 2, 3]
    assert _datamodule(features=["j1_pt", "etarel"]).feature_indices() == [5, 8]
    assert _datamodule(features=RECORD_COLUMNS).feature_indices() is None
    with pytest.raises(ValueError, match="nonesuch"):
        _datamodule(features=["nonesuch"]).feature_indices()

    assert datamodule.norm_scales is None
    datamodule.prepare_data()
    assert record.prepared
    assert datamodule.norm_scales["j1_px"] == 1.0
    assert datamodule.norm_scales["j1_costhetarel"] == 16.0
    assert datamodule.norm_params["j1_pt"] == ("centered", 6.0)


def test_the_folds_pool_the_two_splits_and_a_fold_is_selected_within_range(
    record: FakeRecord,
) -> None:
    def fitted(fold_idx: int) -> HLS4MLDataModule:
        datamodule = _datamodule(n_folds=2)
        datamodule.set_fold(fold_idx)
        datamodule.setup(stage="fit")
        return datamodule

    pool = SPLIT_SIZES["train"] + SPLIT_SIZES["validation"]
    first = fitted(0)

    assert first.n_train_samples == pool // 2 and first.n_val_samples == pool // 2
    assert torch.equal(fitted(0).data_val.X, first.data_val.X)
    assert not torch.equal(fitted(1).data_val.X, first.data_val.X)

    # The test split is never pooled.
    first.setup(stage="test")
    assert first.data_test.n == SPLIT_SIZES["test"]

    for out_of_range in (2, -1):
        with pytest.raises(ValueError, match="out of range"):
            first.set_fold(out_of_range)
    # The main run of every training sets n_folds to 1 and then selects fold 0.
    single = _datamodule(n_folds=1)
    single.set_fold(4)
    assert single.fold_idx == 0


def test_fold_indices_partition_the_pool() -> None:
    folds = [fold_indices(23, 5, 7, fold_idx) for fold_idx in range(5)]

    # Every jet is validated in exactly one fold and trained on in the other four.
    validated = np.concatenate([val_idx for _, val_idx in folds])
    assert sorted(validated) == list(range(23))
    for train_idx, val_idx in folds:
        assert sorted(np.concatenate([train_idx, val_idx])) == list(range(23))
    # The pool's length and the seed fix the split.
    assert np.array_equal(fold_indices(23, 5, 7, 2)[1], folds[2][1])
    assert not np.array_equal(fold_indices(23, 5, 8, 2)[1], folds[2][1])


def test_the_served_fold_is_fold_indices_of_the_pooled_jets(
    record: FakeRecord,
) -> None:
    datamodule = _datamodule(n_folds=3, kfold_seed=7, features=RECORD_COLUMNS)
    datamodule.set_fold(2)
    datamodule.setup(stage="fit")
    # Training then validation: the order a script must pool them in to select the same jets.
    splits = [datamodule._load(record, split) for split in ("train", "validation")]
    x_all, y_all = (torch.cat(tensors) for tensors in zip(*splits))

    train_idx, val_idx = fold_indices(len(x_all), 3, 7, 2)

    assert torch.equal(datamodule.data_train.X, x_all[train_idx])
    assert torch.equal(datamodule.data_train.y, y_all[train_idx])
    assert torch.equal(datamodule.data_val.X, x_all[val_idx])
    assert torch.equal(datamodule.data_val.y, y_all[val_idx])


def test_a_whole_split_is_the_record_s_own_at_full_width_and_rebased(
    record: FakeRecord,
) -> None:
    # The quantised row traces its quantisers on the record's training split at test
    # time, when setup("fit") has not run; at n_folds 1 that split is what it trained on.
    datamodule = _datamodule(features=["pt", "eta", "phi"], n_folds=1)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    x, y = datamodule.load_split("train")

    assert x.shape[-1] == len(RECORD_COLUMNS)
    assert torch.equal(x[..., datamodule.feature_indices()], datamodule.data_train.X)
    assert torch.equal(y, datamodule.data_train.y)


def test_the_record_is_told_the_cache_and_the_constituent_count(
    record: FakeRecord,
) -> None:
    datamodule = _datamodule(nconstituents=150, cache_dir="/scratch/hf")
    datamodule._ensure_record()

    assert record.overrides[0].startswith("paths.root_dir=")
    assert record.overrides[1:] == [
        "data.nconstituents=150",
        "paths.base_data_dir=/scratch/hf",
    ]
    assert datamodule.feature_names == RECORD_COLUMNS
    # Spelled 0: a Hydra override cannot carry None through as an int.
    assert "data.nconstituents=0" in _datamodule(nconstituents=0)._record_overrides(
        "/root"
    )


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path: Path) -> None:
    stand_in = tmp_path / "record"
    (stand_in / "configs").mkdir(parents=True)
    (stand_in / "configs" / "config.yaml").write_text("data:\n  nconstituents: 32\n")

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        outer = GlobalHydra.instance().hydra
        composed = compose_record_config(str(stand_in), ["data.nconstituents=150"])

        assert composed.data.nconstituents == 150
        assert GlobalHydra.instance().hydra is outer


def test_persistent_workers_reshuffle_every_epoch_and_cover_every_jet_once() -> None:
    # A worker forked afresh each epoch would draw from the shuffler's untouched state and replay
    # the first permutation; persistence lets its copy advance as the driver's own would.
    x = torch.arange(40.0).view(40, 1, 1)
    shuffler = torch.Generator().manual_seed(0)
    dataset = HLS4MLDataset(x, labels(40), batch_size=10, shuffler=shuffler)
    loader = _datamodule(num_workers=2)._make_dataloader(dataset, 2)

    epochs = [[int(batch_x[0, 0, 0]) for batch_x, _ in loader] for _ in range(3)]

    assert len({tuple(epoch) for epoch in epochs}) == 3
    assert sorted(int(v) for batch_x, _ in loader for v in batch_x[:, 0, 0]) == list(
        range(40)
    )


def test_only_the_test_split_is_scored_in_full(record: FakeRecord) -> None:
    # A batch of 12 divides none of the 40 training, 20 validation and 16 test jets.
    datamodule = _datamodule(batch_size=12)
    datamodule.setup()

    def scored(loader) -> int:
        return sum(batch_x.shape[0] for batch_x, _ in loader)

    assert scored(datamodule.train_dataloader()) == 36
    assert scored(datamodule.val_dataloader()) == 12
    assert scored(datamodule.test_dataloader()) == SPLIT_SIZES["test"]
