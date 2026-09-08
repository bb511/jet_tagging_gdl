from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from src.data import hls4ml_datamodule as datamodule_module
from src.data.components.dataset import HLS4MLDataset
from src.data.hls4ml_datamodule import HLS4MLDataModule, compose_record_config
from tests.conftest import RECORD_COLUMNS, ROOT, labels

SPLIT_SIZES = {"train": 40, "validation": 20, "test": 16}
NCONSTITUENTS = 8


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
        return {name: ("centered", 1.0 + idx) for idx, name in enumerate(RECORD_COLUMNS)}

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

    monkeypatch.setattr(datamodule_module, "snapshot_download", lambda *a, **k: str(tmp_path))
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


@pytest.mark.parametrize(
    "method, scheme, keep_full, fit_width, test_width",
    [
        ("augmentation", "none", True, 4, 16),
        ("augmentation", "none", False, 4, 4),
        ("augmentation", "so3", True, 4, 16),
        ("augmentation", "lorentz_iso", True, 4, 16),
        # The SCL transforms inside model_step, so its batches are full width whatever the flag.
        ("symm_loss", "so2", False, 16, 16),
    ],
)
def test_the_batch_widths_follow_the_method_and_the_test_flag(
    record: FakeRecord, method: str, scheme: str, keep_full: bool, fit_width: int, test_width: int
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


def test_an_scl_run_keeps_its_group_and_fills_the_callback_surface(record: FakeRecord) -> None:
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


def test_the_record_is_told_the_cache_and_the_constituent_count(record: FakeRecord) -> None:
    datamodule = _datamodule(nconstituents=150, cache_dir="/scratch/hf")
    datamodule._ensure_record()

    assert record.overrides[0].startswith("paths.root_dir=")
    assert record.overrides[1:] == ["data.nconstituents=150", "paths.base_data_dir=/scratch/hf"]
    assert datamodule.feature_names == RECORD_COLUMNS
    # Spelled 0: a Hydra override cannot carry None through as an int.
    assert "data.nconstituents=0" in _datamodule(nconstituents=0)._record_overrides("/root")


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
    loader = _datamodule(num_workers=2)._make_dataloader(dataset)

    epochs = [[int(batch_x[0, 0, 0]) for batch_x, _ in loader] for _ in range(3)]

    assert len({tuple(epoch) for epoch in epochs}) == 3
    assert sorted(int(v) for batch_x, _ in loader for v in batch_x[:, 0, 0]) == list(range(40))
