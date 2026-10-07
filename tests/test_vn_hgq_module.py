import os

# Before anything imports keras: the olgpu ~/.keras/keras.json names tensorflow.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("KERAS_BACKEND", "jax")

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

pytest.importorskip("hgq")

import keras
from hgq.layers.core import QLayerBase

from src.algorithms import JetTaggingModule
from src.algorithms.vector_neurons_hgq import VectorNeuronsHGQ
from src.callbacks.invariance import InvarianceMetricCallback
from src.data.components.dataset import HLS4MLDataset
from src.utils.checkpoints import Manifest, ManifestRun, write_manifest
from tests.conftest import RECORD_COLUMNS, ROOT, labels

N_SLOTS = 8
ARCHITECTURE = {
    "num_blocks": 2,
    "vector_channels": 8,
    "scalar_channels": 16,
    "invariants": 8,
    "tensor_invariants": 2,
    "stream_channels": 8,
    "head_layers": [16],
    "dropout": 0.0,
}
PT_ETA_PHI = ["pt", "eta", "phi"]
COLUMNS = [RECORD_COLUMNS.index(f"j1_{name}") for name in PT_ETA_PHI]
# Jets per split, in batches of 16; the test split ends in a partial batch.
SIZES = {"train": 64, "validation": 32, "test": 40}
BATCH = 16
# Beta 1e-6 in the first epoch and 3e-6 from the second on.
SCHEDULE = [[0, 1e-6, "constant"], [1, 3e-6, "constant"]]


def _record(n_jets: int, seed: int) -> torch.Tensor:
    # The record's 16 columns, of which the module reads pt (GeV), eta and phi: 1 to
    # N_SLOTS real constituents of at least 1 GeV, so that every keep flag min(4 p_T, 1)
    # is one, and the padded rows all zero.
    generator = torch.Generator().manual_seed(seed)
    gauss = torch.randn(3, n_jets, N_SLOTS, generator=generator)
    columns = [1.0 + 40.0 * gauss[0].exp(), 0.4 + 0.25 * gauss[1], 0.3 * gauss[2]]
    x = torch.zeros(n_jets, N_SLOTS, len(RECORD_COLUMNS))
    x[..., COLUMNS] = torch.stack(columns, dim=-1)
    multiplicity = torch.randint(1, N_SLOTS + 1, (n_jets, 1), generator=generator)
    x[torch.arange(N_SLOTS) >= multiplicity] = 0.0
    return x


class _Record(LightningDataModule):
    # What the module, the probe and eval.py read of HLS4MLDataModule: training and
    # validation batches of [pt, eta, phi], test batches of all 16 columns, the fit's
    # training jets and the record's splits whole.
    feature_names = RECORD_COLUMNS
    features = PT_ETA_PHI
    keep_full_features_at_test = True
    norm_scales = None

    def __init__(self) -> None:
        super().__init__()
        self.splits = {
            split: (_record(n_jets, seed), labels(n_jets, seed))
            for seed, (split, n_jets) in enumerate(SIZES.items())
        }
        self.data_train = self._dataset("train", COLUMNS)
        self.requested: list[str] = []

    def _dataset(self, split: str, columns=None, **kwargs) -> HLS4MLDataset:
        x, y = self.splits[split]
        x = x if columns is None else x[..., columns]
        return HLS4MLDataset(x, y, batch_size=BATCH, **kwargs)

    def feature_indices(self) -> list[int]:
        return COLUMNS

    def load_split(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.requested.append(split)
        return self.splits[split]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset("validation", COLUMNS), batch_size=None)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset("test", drop_last=False), batch_size=None)


def _algorithm(name: str, **overrides: object) -> JetTaggingModule:
    # The algorithm configs interpolate ${data.*} and ${trainer.*}, so nodes of our own
    # stand in. Seeded, so that two ports built alike start from the same weights.
    cfg = OmegaConf.create(
        {
            "data": {
                "nconstituents": N_SLOTS,
                "method": "augmentation",
                "augmentation_scheme": "none",
                "transform_kwargs": {},
            },
            "trainer": {"accelerator": "cpu", "devices": 1},
        }
    )
    cfg.algorithm = OmegaConf.load(ROOT / "configs" / "algorithm" / f"{name}.yaml")
    cfg.algorithm.merge_with({**ARCHITECTURE, **overrides})
    torch.manual_seed(0)
    return instantiate(cfg.algorithm)


def _port(**overrides: object) -> VectorNeuronsHGQ:
    return _algorithm("vn_hgq", calibration_jets=SIZES["train"], **overrides)


def _trainer(root: Path, *callbacks: Callback, **kwargs: object) -> Trainer:
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=root,
        callbacks=list(callbacks),
        **kwargs,
    )


def _betas(module: VectorNeuronsHGQ) -> list[float]:
    # The beta of every layer that weighs its EBOPs into the loss.
    layers = module.net.keras_model._flatten_layers()
    priced = [x for x in layers if isinstance(x, QLayerBase) and x.enable_ebops]
    assert priced
    return [float(keras.ops.convert_to_numpy(layer._beta)) for layer in priced]


class _BetaRecorder(Callback):
    def __init__(self) -> None:
        self.seen: list[list[float]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: VectorNeuronsHGQ) -> None:
        self.seen.append(_betas(pl_module))


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    # Two epochs under the beta schedule, checkpointed as configs/callbacks/quantised.yaml
    # does: unranked, the newest epoch alone.
    root = tmp_path_factory.mktemp("fit")
    datamodule, betas = _Record(), _BetaRecorder()
    module = _port(beta_schedule=SCHEDULE)
    checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="epoch_{epoch:03d}",
        monitor=None,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )
    trainer = _trainer(root, checkpoint, betas, max_epochs=2)
    trainer.fit(module, datamodule=datamodule)
    return SimpleNamespace(
        root=root,
        datamodule=datamodule,
        module=module,
        trainer=trainer,
        checkpoint=checkpoint,
        betas=betas,
    )


def test_training_steps_keras_per_batch_and_checkpoints_the_last_epoch(
    fitted: SimpleNamespace,
) -> None:
    model = fitted.module.net.keras_model
    steps = 2 * len(fitted.datamodule.train_dataloader())
    saved = torch.load(fitted.checkpoint.best_model_path, weights_only=False)
    metrics = fitted.trainer.callback_metrics

    # The mock optimiser's step advances global_step, without which nothing is saved.
    assert fitted.trainer.global_step == steps
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == steps
    assert [path.name for path in (fitted.root / "checkpoints").iterdir()] == [
        "epoch_001.ckpt"
    ]
    assert Path(fitted.checkpoint.best_model_path).name == "epoch_001.ckpt"
    weights = zip(saved["keras_weights"], model.get_weights(), strict=True)
    assert all(np.array_equal(ours, theirs) for ours, theirs in weights)
    assert {"train/loss", "train/acc", "train/lr", "val/acc"} <= set(metrics)
    assert metrics["train/ebops"] > 0 and torch.isfinite(metrics["train/loss_total"])


def test_a_checkpoint_restores_the_keras_weights_in_the_port_s_float_type(
    fitted: SimpleNamespace,
) -> None:
    trained = fitted.module.net.keras_model
    checkpoint = torch.load(fitted.checkpoint.best_model_path, weights_only=False)
    rows = fitted.module._keras_inputs(fitted.datamodule.data_train.X)
    restored = _port()
    assert not np.array_equal(
        restored.net.keras_model.predict_on_batch(rows), trained.predict_on_batch(rows)
    )

    restored.on_load_checkpoint(checkpoint)

    assert np.array_equal(
        restored.net.keras_model.predict_on_batch(rows), trained.predict_on_batch(rows)
    )
    # As a float32 training saves them, loaded into the float64 port.
    checkpoint["keras_weights"] = [
        w.astype(np.float32) if w.dtype == np.float64 else w
        for w in checkpoint["keras_weights"]
    ]
    restored.on_load_checkpoint(checkpoint)
    for ours, theirs in zip(restored.net.keras_model.weights, trained.weights):
        assert ours.dtype == theirs.dtype
        assert np.allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6)


def test_beta_reaches_every_quantised_layer_constant_or_scheduled(
    fitted: SimpleNamespace,
) -> None:
    scheduled = [np.unique(betas).tolist() for betas in fitted.betas.seen]
    port = _port(beta=2e-7)
    port._trainer = SimpleNamespace(current_epoch=5)

    port.on_train_epoch_start()

    assert [len(betas) for betas in scheduled] == [1, 1]
    assert [betas[0] for betas in scheduled] == pytest.approx([1e-6, 3e-6])
    assert float(fitted.trainer.callback_metrics["train/beta"]) == pytest.approx(3e-6)
    assert np.unique(_betas(port)).tolist() == pytest.approx([2e-7])


def test_evaluation_traces_the_training_split_and_exports_the_reference(
    fitted: SimpleNamespace, tmp_path: Path
) -> None:
    # As eval.py runs it: a port built anew, the checkpoint loaded into it, and the probe
    # writing outputs/main.npz.
    probe = InvarianceMetricCallback(probe=False)
    probe.subdir = "main"
    datamodule, port = _Record(), _port(beta_schedule=SCHEDULE)
    trainer = _trainer(tmp_path, probe, enable_checkpointing=False)
    ckpt_path = fitted.checkpoint.best_model_path

    trainer.test(port, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)

    export = tmp_path / "outputs" / "main_hgq"
    meta = json.loads((export / "export.json").read_text())
    with np.load(tmp_path / "outputs" / "main.npz") as outputs:
        clean = outputs["logits_clean"]
    with np.load(export / "inputs.npz") as stored:
        words = stored["inputs_clean"]
    reloaded = keras.models.load_model(export / "model.keras", compile=False)
    logits = np.asarray(reloaded.predict_on_batch(words.astype(np.float64)))

    assert datamodule.requested == ["train"]
    assert words.shape == (SIZES["test"], N_SLOTS, 3)
    assert np.array_equal(logits.astype(np.float32), clean)
    assert math.isfinite(port.ebops_total) and port.ebops_total > 0
    assert meta["ebops"] == port.ebops_total
    assert float(trainer.callback_metrics["test/ebops"]) == port.ebops_total
    assert (meta["row"], meta["nconstituents"]) == ("vn_hgq", N_SLOTS)
    assert (meta["n_test"], meta["floatx"]) == (SIZES["test"], "float64")
    assert meta["ckpt_path"] == ckpt_path


def test_the_float_twin_prices_the_flops_and_the_ebops_wait_for_the_trace() -> None:
    port, float_row = _port(), _algorithm("vn")

    assert port.flops_total == float_row.flops_total > 0
    assert math.isnan(port.ebops_total) and math.isnan(port.ebops_unpriced)
    # Keras holds every weight.
    assert not list(port.parameters())


def test_init_from_copies_a_float_run_of_the_same_architecture_alone(
    tmp_path: Path,
) -> None:
    # A float vn of random weights, saved as train.py leaves a run: a checkpoint and the
    # manifest naming it, which init_from reaches through the run directory.
    float_row = _algorithm("vn")
    with torch.no_grad():
        for parameter in float_row.parameters():
            parameter.normal_(0.0, 0.5)
    path = tmp_path / "checkpoints" / "epoch_000.ckpt"
    path.parent.mkdir()
    state = {"hyper_parameters": dict(float_row.hparams)}
    torch.save(state | {"state_dict": float_row.state_dict()}, path)
    run = ManifestRun("main", None, str(path))
    write_manifest(Manifest(0, str(tmp_path), "test", [run]), tmp_path)
    datamodule = _Record()
    x = datamodule.splits["test"][0][..., COLUMNS]
    with torch.no_grad():
        expected = float_row.double().eval()(x)
    port = _port(init_from=str(tmp_path), quantisation={"float_mode": True})
    before = port(x)

    port._trainer = SimpleNamespace(datamodule=datamodule)
    port.setup("fit")

    # The float port differs from the float network by the rounding of the float32 rows
    # alone.
    tolerance = 1e-5 * max(1.0, float(expected.abs().max()))
    assert (port(x) - expected).abs().max() <= tolerance
    assert (before - expected).abs().max() > tolerance
    mismatched = _port(init_from=str(tmp_path), num_blocks=1)
    mismatched._trainer = SimpleNamespace(datamodule=datamodule)
    with pytest.raises(ValueError, match="num_blocks 2 vs 1"):
        mismatched.setup("fit")
