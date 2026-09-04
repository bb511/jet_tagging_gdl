"""The datamodule's contract, checked against a stand-in for the record's loader.

Every test is offline: ``FakeRecord`` offers exactly the surface ``HLS4MLDataModule`` asks of
``loader.datamodule.JetData``, which is enough for the record plumbing, the k-fold pooling and the
batch-width contract the width guard in ``JetTaggingModule.forward`` is written against.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data import hls4ml_datamodule as datamodule_module
from src.data.hls4ml_datamodule import HLS4MLDataModule, compose_record_config
from tests.conftest import ROOT

# The record's 16 constituent columns, in the order it carries them.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()
SPLIT_SIZES = {"train": 40, "validation": 20, "test": 16}
NCONSTITUENTS = 8


class FakeRecord:
    """The surface of ``loader.datamodule.JetData``, over jets drawn from a fixed seed."""

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
        """None until prepare() has run, as the record's own property is."""
        if not self.prepared:
            return None
        return {name: ("centered", 1.0 + idx) for idx, name in enumerate(RECORD_COLUMNS)}

    def prepare(self) -> None:
        self.prepared = True

    def load(self, split: str) -> SimpleNamespace:
        """Jets of shape (jets, NCONSTITUENTS, 16) and one-hot labels, as ``x`` and ``y``."""
        generator = torch.Generator().manual_seed(len(split))
        n_jets = SPLIT_SIZES[split]
        x = torch.randn(n_jets, NCONSTITUENTS, len(RECORD_COLUMNS), generator=generator)
        y = torch.zeros(n_jets, 5)
        y[torch.arange(n_jets), torch.randint(0, 5, (n_jets,), generator=generator)] = 1.0

        return SimpleNamespace(x=x, y=y)


@pytest.fixture
def record(monkeypatch, tmp_path: Path) -> FakeRecord:
    """Wire the fake in where ``_ensure_record`` would put the real one, hub unreachable."""
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
    """A datamodule whose defaults are cheap and whose repo_id names no real repository."""
    defaults = {
        "repo_id": "stand-in/record",
        "cache_dir": "/nowhere/hf",
        "nconstituents": NCONSTITUENTS,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
    }

    return HLS4MLDataModule(**(defaults | overrides))


def _first_batch(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    return next(iter(loader))


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
    """The width guard in forward() reduces on a mismatch, so these cases decide it."""
    datamodule = _datamodule(
        method=method, augmentation_scheme=scheme, keep_full_features_at_test=keep_full
    )
    datamodule.setup()
    x_train, y_train = _first_batch(datamodule.train_dataloader())
    x_val, _ = _first_batch(datamodule.val_dataloader())
    x_test, _ = _first_batch(datamodule.test_dataloader())

    assert x_train.shape[1:] == (NCONSTITUENTS, fit_width)
    assert x_val.shape[-1] == fit_width
    assert x_test.shape == (8, NCONSTITUENTS, test_width)
    assert x_train.dtype == torch.float32
    # The labels follow the batch, whether or not a transform dropped jets from it.
    assert y_train.shape == (x_train.shape[0], 5)


def test_an_scl_run_keeps_its_group_and_fills_the_callback_surface(record: FakeRecord) -> None:
    """The results table and ${data.augmentation_scheme} must agree on what a run did."""
    datamodule = _datamodule(method="symm_loss", augmentation_scheme="lorentz")
    datamodule.setup(stage="fit")

    assert datamodule.augmentation_scheme == "lorentz"
    # Under the SCL the model transforms, so the training dataset carries no transform.
    assert datamodule.data_train.transform is None
    assert datamodule.feature_names == RECORD_COLUMNS
    assert datamodule.features == ["px", "py", "pz", "e"]
    assert datamodule.keep_full_features_at_test is True
    assert datamodule.n_train_samples == SPLIT_SIZES["train"]
    assert datamodule.n_val_samples == SPLIT_SIZES["validation"]


def test_a_method_or_a_scheme_outside_the_domain_is_refused() -> None:
    """Both retired values of the old grid, rejected before anything is downloaded."""
    with pytest.raises(ValueError, match="method"):
        _datamodule(method="equivariance")
    with pytest.raises(ValueError, match="augmentation_scheme"):
        _datamodule(augmentation_scheme="permute")


def test_the_features_are_found_by_name_and_the_scales_arrive_with_the_record(
    record: FakeRecord,
) -> None:
    """Configs name the columns bare, the record with ``j1_``, and both must resolve; the probe
    reads norm_scales, and a None there disables it rather than lying."""
    datamodule = _datamodule()

    assert datamodule.feature_indices() == [0, 1, 2, 3]
    assert _datamodule(features=["j1_pt", "etarel"]).feature_indices() == [5, 8]
    # All 16 columns in the record's own order: nothing to slice.
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
    """kfold_seed is fixed, so that two runs' per-fold numbers can be compared at all, and a fold
    index out of range must not silently serve fold 0."""

    def fitted(fold_idx: int) -> HLS4MLDataModule:
        datamodule = _datamodule(n_folds=2)
        datamodule.set_fold(fold_idx)
        datamodule.setup(stage="fit")
        return datamodule

    # The record's own validation split is too small to carry a five-fold spread alone.
    pool = SPLIT_SIZES["train"] + SPLIT_SIZES["validation"]
    first = fitted(0)

    assert first.n_train_samples == pool // 2 and first.n_val_samples == pool // 2
    assert torch.equal(fitted(0).data_val.X, first.data_val.X)
    assert not torch.equal(fitted(1).data_val.X, first.data_val.X)

    # The test split is never pooled, so it does not depend on the fold.
    first.setup(stage="test")
    assert first.data_test.n == SPLIT_SIZES["test"]

    for out_of_range in (2, -1):
        with pytest.raises(ValueError, match="out of range"):
            first.set_fold(out_of_range)
    # The main run of every training sets n_folds to 1 and then selects fold 0.
    single = _datamodule(n_folds=1)
    single.set_fold(4)
    assert single.fold_idx == 0


def test_the_record_is_told_the_cache_and_the_constituent_count(
    record: FakeRecord, tmp_path: Path
) -> None:
    """These three overrides are the whole interface to the record's own config tree."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    datamodule = _datamodule(nconstituents=150, cache_dir="/scratch/hf")
    datamodule._ensure_record()

    assert record.overrides[0].startswith("paths.root_dir=")
    assert record.overrides[1:] == ["data.nconstituents=150", "paths.base_data_dir=/scratch/hf"]
    assert datamodule.feature_names == RECORD_COLUMNS
    # Every constituent is spelled 0: a Hydra override cannot carry None through as an int.
    assert "data.nconstituents=0" in _datamodule(nconstituents=0)._record_overrides("/root")
    # A mirror is checked from disk before it is uploaded, so no hub call is made.
    assert _datamodule(repo_id=str(mirror))._record_root() == str(mirror.resolve())


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path: Path) -> None:
    """initialize_config_dir restores only the Hydra it saw at its own entry, never the outer."""
    stand_in = tmp_path / "record"
    (stand_in / "configs").mkdir(parents=True)
    (stand_in / "configs" / "config.yaml").write_text("data:\n  nconstituents: 32\n")

    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        outer = GlobalHydra.instance().hydra
        composed = compose_record_config(str(stand_in), ["data.nconstituents=150"])

        assert composed.data.nconstituents == 150
        assert GlobalHydra.instance().hydra is outer


def _data_config(tmp_path: Path, **overrides) -> OmegaConf:
    """The shipped data config, with its ${paths.*} interpolations satisfied."""
    cfg = OmegaConf.create(
        {
            "paths": {"base_data_dir": str(tmp_path)},
            "data": OmegaConf.load(ROOT / "configs" / "data" / "hls4ml.yaml"),
        }
    )
    for key, value in overrides.items():
        OmegaConf.update(cfg, f"data.{key}", value)

    return cfg


def test_the_shipped_config_builds_without_reaching_the_hub(monkeypatch, tmp_path: Path) -> None:
    """Instantiation is offline, which is what every configuration test above relies on."""
    monkeypatch.setattr(
        datamodule_module, "snapshot_download", lambda *a, **k: pytest.fail("the hub was reached")
    )
    datamodule = instantiate(_data_config(tmp_path).data)

    assert isinstance(datamodule, HLS4MLDataModule)
    assert datamodule.record is None
    assert datamodule.cache_dir == str(tmp_path / "hf")
    assert datamodule.features == ["px", "py", "pz", "e"]
