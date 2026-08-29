"""The record's tensors must reach the datasets exactly as the raw route's do.

Everything here is offline: no hub, no h5 files, no parquet. The record is stood in
for by FakeRecord, which is all setup() ever asks for.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from src.data.hls4ml_datamodule import CENTERED_FEATURES
from src.data.hls4ml_hf_datamodule import HLS4MLHFDataModule, compose_record_config

NAMES = [
    "j1_px", "j1_py", "j1_pz", "j1_e", "j1_erel", "j1_pt", "j1_ptrel", "j1_eta",
    "j1_etarel", "j1_etarot", "j1_phi", "j1_phirel", "j1_phirot", "j1_deltaR",
    "j1_costheta", "j1_costhetarel",
]
NJETS = 8


@dataclass(frozen=True)
class _Split:
    x: torch.Tensor
    y: torch.Tensor


class FakeRecord:
    """Stands in for the record's JetData: eight jets, four constituents, 16 features."""

    @property
    def norm_params(self) -> dict[str, tuple[str, float]]:
        return {n: (_kind(n), 1.0) for n in NAMES}

    def load(self, split: str) -> _Split:
        x = torch.arange(NJETS * 4 * 16, dtype=torch.float32).reshape(NJETS, 4, 16)
        y = torch.zeros(NJETS, 5)
        y[:, 0] = 1.0

        return _Split(x, y)


def _kind(name: str) -> str:
    return "centered" if name in CENTERED_FEATURES else "positive"


def _write_norm_params(folder: Path) -> None:
    """The npz the inherited _get_feature_indices reads, in the legacy format."""
    np.savez(
        folder / "norm_params.npz",
        feature_names=np.array(NAMES, dtype="U64"),
        kinds=np.array([_kind(n) for n in NAMES], dtype="U16"),
        scales=np.ones(len(NAMES), dtype=np.float32),
    )


def _mini_record(tmp_path: Path) -> Path:
    """A stand-in for the record: a root holding nothing but a config tree."""
    configs = tmp_path / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "config.yaml").write_text("marker: record\nvalue: 1\n")

    return tmp_path


def _instantiated(tmp_path: Path, *overrides: str) -> HLS4MLHFDataModule:
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=hls4ml_hf", f"paths.root_dir={tmp_path}", *overrides],
        )

    return instantiate(cfg.data)


def _prepared(tmp_path: Path, monkeypatch, *overrides: str) -> HLS4MLHFDataModule:
    """A datamodule wired to FakeRecord, as _ensure_record would have wired it."""
    datamodule = _instantiated(tmp_path, *overrides)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    _write_norm_params(cache)

    datamodule.record = FakeRecord()
    datamodule.processed_data_dir = str(cache)
    datamodule.feature_names = list(NAMES)
    datamodule.pt_idx = NAMES.index("j1_pt")
    # The diagnostic reads the cached .npy arrays, which no test writes.
    monkeypatch.setattr(
        HLS4MLHFDataModule, "_plot_online_aug_diagnostic", lambda *a, **k: None
    )

    return datamodule


def test_record_config_composes(tmp_path):
    """The record is composed from its own directory, not from this project's."""
    composed = compose_record_config(str(_mini_record(tmp_path)), [])

    assert composed.marker == "record"


def test_record_overrides_reach_the_record(tmp_path):
    composed = compose_record_config(str(_mini_record(tmp_path)), ["value=2"])

    assert composed.value == 2


def test_composing_the_record_leaves_the_running_hydra_alone(tmp_path):
    """The project's own tree must still compose after the record's has been read.

    initialize_config_dir refuses to run while a global Hydra is live and restores only
    what it saw at its own entry, so a naive composition would leave the application
    without the Hydra it started with.
    """
    record = _mini_record(tmp_path)
    with initialize(version_base="1.3", config_path="../configs"):
        before = compose(config_name="train.yaml", overrides=["data=hls4ml_hf"])
        composed = compose_record_config(str(record), [])
        after = compose(config_name="train.yaml", overrides=["data=hls4ml_hf"])

    assert composed.marker == "record"
    assert after.data._target_ == before.data._target_


def test_hls4ml_hf_instantiates_without_reaching_the_hub(tmp_path):
    """Building the datamodule must not download anything; prepare_data does that."""
    datamodule = _instantiated(tmp_path)

    assert datamodule.record is None
    assert datamodule.repo_id == "fastmachinelearning/hls4ml_lhc_jets_150p"
    assert datamodule.cache_dir == f"{tmp_path}/data/hf"


def test_a_local_repo_id_is_read_in_place(tmp_path, monkeypatch):
    """The mirror is tested from disk before it is uploaded, so no hub call is made."""
    def refuse(*args, **kwargs):
        raise RuntimeError("the hub was reached")

    monkeypatch.setattr("src.data.hls4ml_hf_datamodule.snapshot_download", refuse)
    mirror = _mini_record(tmp_path)
    datamodule = _instantiated(tmp_path, f"data.repo_id={mirror}")

    assert datamodule._record_root() == str(mirror.resolve())
    with pytest.raises(RuntimeError):
        _instantiated(tmp_path, "data.repo_id=podagiu/nowhere")._record_root()


def test_record_overrides_carry_the_root_and_the_constituent_count(tmp_path):
    datamodule = _instantiated(tmp_path, "data.nconstituents=16")

    assert datamodule._record_overrides("/record") == [
        "paths.root_dir=/record",
        "data.nconstituents=16",
        f"paths.base_data_dir={tmp_path}/data/hf",
    ]


def test_keeping_all_constituents_is_spelled_zero(tmp_path):
    """The record takes 0, since a Hydra override cannot carry None through as an int."""
    datamodule = _instantiated(tmp_path, "data.nconstituents=null")

    assert "data.nconstituents=0" in datamodule._record_overrides("/record")


def test_every_split_is_reduced_to_the_model_features(tmp_path, monkeypatch):
    datamodule = _prepared(
        tmp_path, monkeypatch, "data.n_folds=1", "data.keep_full_features_at_test=false"
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    assert datamodule.data_train.X.shape == (NJETS, 4, 5)
    assert datamodule.data_val.X.shape == (NJETS, 4, 5)
    assert datamodule.data_test.X.shape == (NJETS, 4, 5)


def test_the_full_feature_set_is_kept_at_test_on_request(tmp_path, monkeypatch):
    """InvarianceMetricCallback needs px/py/pz to transform the test batches."""
    datamodule = _prepared(
        tmp_path, monkeypatch, "data.n_folds=1", "data.keep_full_features_at_test=true"
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    assert datamodule.data_val.X.shape[2] == 5
    assert datamodule.data_test.X.shape[2] == 16


def test_symm_loss_keeps_the_full_feature_set_everywhere(tmp_path, monkeypatch):
    """The symmetric partner is built inside the model, out of the full feature set."""
    datamodule = _prepared(
        tmp_path, monkeypatch, "data.n_folds=1", "data.method=symm_loss"
    )
    datamodule.setup("fit")
    datamodule.setup("test")

    assert datamodule.data_train.X.shape[2] == 16
    assert datamodule.data_val.X.shape[2] == 16
    assert datamodule.data_test.X.shape[2] == 16


def test_an_online_transform_leaves_the_train_tensor_wide(tmp_path, monkeypatch):
    """so2 needs px/py/pz at batch time and reduces to data.features itself."""
    datamodule = _prepared(
        tmp_path, monkeypatch, "data.n_folds=1", "data.augmentation_scheme=so2"
    )
    datamodule.setup("fit")

    assert datamodule.data_train.transform is not None
    assert datamodule.data_train.X.shape[2] == 16
    assert datamodule.data_val.X.shape[2] == 5


def test_kfold_pools_train_and_validation(tmp_path, monkeypatch):
    datamodule = _prepared(
        tmp_path, monkeypatch, "data.n_folds=5", "data.batch_size=1"
    )
    datamodule.set_fold(0)
    datamodule.setup("fit")

    assert datamodule.n_train_samples + datamodule.n_val_samples == 2 * NJETS
    assert datamodule.n_val_samples == 4


def test_the_callback_contract_holds_after_test_setup(tmp_path, monkeypatch):
    """What InvarianceMetricCallback.on_test_start reads off the datamodule."""
    datamodule = _prepared(tmp_path, monkeypatch, "data.n_folds=1")
    datamodule.setup("test")

    assert datamodule.feature_names == NAMES
    assert datamodule.norm_params["j1_pt"] == ("positive", 1.0)
    assert (Path(datamodule.processed_data_dir) / "norm_params.npz").exists()
    indices = datamodule._get_feature_indices(Path(datamodule.processed_data_dir))
    assert [NAMES[i] for i in indices] == [f"j1_{f}" for f in datamodule.features]
