import os

# Before anything imports keras: the olgpu ~/.keras/keras.json names tensorflow.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("KERAS_BACKEND", "jax")

import csv
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

pytest.importorskip("hgq")

import jax
import keras
from hgq.layers.core import QLayerBase

from src.callbacks.invariance import InvarianceMetricCallback
from src.data.components.dataset import HLS4MLDataset
from src.utils.checkpoints import Manifest, ManifestRun, write_manifest
from tests.conftest import (
    PT_ETA_PHI,
    RECORD_COLUMNS,
    algorithm,
    attach,
    labels,
    load_script,
)

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
# (gate, mlp_layers, stream_channels, tensor_invariants, fold_norm): the network of the
# finals and the latency ablation's variants, each switching one more piece off, the
# batch normalisations folded into the next layer in every other one.
VARIANTS = [
    (True, 2, 8, 2, False),
    (False, 2, 8, 2, True),
    (False, 1, 8, 2, False),
    (False, 1, 0, 2, True),
    (False, 1, 0, 0, False),
]
COLUMNS = [RECORD_COLUMNS.index(f"j1_{name}") for name in PT_ETA_PHI]
# Jets per split, in batches of 16; the test split ends in a partial batch.
SIZES = {"train": 64, "validation": 32, "test": 40}
BATCH = 16
# Beta 1e-6 in the first epoch and 3e-6 from the second on.
SCHEDULE = [[0, 1e-6, "constant"], [1, 3e-6, "constant"]]
# synthesis.csv's columns, in order: the file is appended to across runs.
SYNTHESIS_COLUMNS = (
    "model nconstituents beta ebops part clock_ns latency_cutoff ternary method0 ii "
    "latency_cycles latency_ns lut ff dsp bram_tile ramb18 uram wns_ns fmax_mhz "
    "alkaid_cost reg_bits acc_macro_keras acc_macro_alkaid acc_micro_alkaid n_test "
    "n_mismatch max_abs_diff n_cosim cosim_exact vivado_done export_dir prj_dir "
    "alkaid_version hgq_version keras_version timestamp"
).split()


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


def _port(**overrides: object) -> torch.nn.Module:
    # Seeded, so that two ports built alike start from the same weights.
    torch.manual_seed(0)
    overrides = ARCHITECTURE | {"calibration_jets": SIZES["train"]} | overrides
    return algorithm("vn_hgq", N_SLOTS, **overrides)


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


class _BetaRecorder(Callback):
    # The beta of every layer that weighs its EBOPs into the loss, at each epoch's end.
    def __init__(self) -> None:
        self.seen: list[list[float]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: torch.nn.Module) -> None:
        layers = pl_module.net.keras_model._flatten_layers()
        priced = [x for x in layers if isinstance(x, QLayerBase) and x.enable_ebops]
        self.seen.append([float(keras.ops.convert_to_numpy(x._beta)) for x in priced])


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    # Two epochs under the beta schedule, checkpointed as
    # configs/callbacks/quantised.yaml does: unranked, the newest epoch alone.
    root = tmp_path_factory.mktemp("fit")
    datamodule, betas, module = (
        _Record(),
        _BetaRecorder(),
        _port(beta_schedule=SCHEDULE),
    )
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
        root=root, datamodule=datamodule, module=module, trainer=trainer, betas=betas
    )


@pytest.fixture(scope="module")
def evaluated(fitted: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory):
    # As eval.py runs it: a port built anew, the checkpoint loaded into it, and the
    # probe writing outputs/main.npz beside the port's export.
    root = tmp_path_factory.mktemp("eval")
    probe = InvarianceMetricCallback(probe=False)
    probe.subdir = "main"
    datamodule, port = _Record(), _port(beta_schedule=SCHEDULE)
    trainer = _trainer(root, probe, enable_checkpointing=False)
    ckpt_path = fitted.trainer.checkpoint_callback.best_model_path
    trainer.test(port, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    outputs = root / "outputs"
    return SimpleNamespace(
        export=outputs / "main_hgq",
        npz=outputs / "main.npz",
        datamodule=datamodule,
        port=port,
        trainer=trainer,
    )


@pytest.mark.parametrize(
    "gate, mlp_layers, stream_channels, tensor_invariants, fold_norm", VARIANTS
)
def test_the_port_copies_a_float_vn_so_keras_and_torch_agree(
    tmp_path: Path,
    gate: bool,
    mlp_layers: int,
    stream_channels: int,
    tensor_invariants: int,
    fold_norm: bool,
) -> None:
    # A float vn of random weights saved as train.py leaves a run, a checkpoint and the
    # manifest naming it, which init_from reaches through the run directory.
    variant = dict(
        gate=gate,
        mlp_layers=mlp_layers,
        stream_channels=stream_channels,
        tensor_invariants=tensor_invariants,
    )
    torch.manual_seed(0)
    float_row = algorithm("vn", N_SLOTS, **(ARCHITECTURE | variant))
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
    quantisation = {"float_mode": True, "fold_norm": fold_norm}
    port = _port(init_from=str(tmp_path), quantisation=quantisation, **variant)
    before = port(x)

    attach(port, datamodule)

    # The float port differs from the float network by the rounding of the float32
    # rows alone.
    tolerance = 1e-5 * max(1.0, float(expected.abs().max()))
    assert (port(x) - expected).abs().max() <= tolerance
    assert (before - expected).abs().max() > tolerance
    mismatched = _port(init_from=str(tmp_path), num_blocks=1, **variant)
    with pytest.raises(ValueError, match="num_blocks 2 vs 1"):
        attach(mismatched, datamodule)


def test_the_module_trains_on_keras_and_round_trips_a_checkpoint(
    fitted: SimpleNamespace,
) -> None:
    model = fitted.module.net.keras_model
    steps = 2 * len(fitted.datamodule.train_dataloader())
    best = fitted.trainer.checkpoint_callback.best_model_path
    saved = torch.load(best, weights_only=False)
    rows = fitted.module._keras_inputs(fitted.datamodule.data_train.X)
    restored = _port()
    untrained = restored.net.keras_model.predict_on_batch(rows)
    restored.on_load_checkpoint(saved)
    metrics = fitted.trainer.callback_metrics
    scheduled = [np.unique(betas).tolist() for betas in fitted.betas.seen]

    # The mock optimiser's step advances global_step, without which nothing is saved.
    assert fitted.trainer.global_step == steps
    assert int(keras.ops.convert_to_numpy(model.optimizer.iterations)) == steps
    assert [path.name for path in Path(best).parent.iterdir()] == ["epoch_001.ckpt"]
    weights = zip(saved["keras_weights"], model.get_weights(), strict=True)
    assert all(np.array_equal(ours, theirs) for ours, theirs in weights)
    assert not np.array_equal(untrained, model.predict_on_batch(rows))
    assert np.array_equal(
        restored.net.keras_model.predict_on_batch(rows), model.predict_on_batch(rows)
    )
    # Every quantised layer takes the beta of the epoch, as HGQ2's BetaScheduler gives.
    assert [len(betas) for betas in scheduled] == [1, 1]
    assert [betas[0] for betas in scheduled] == pytest.approx([1e-6, 3e-6])
    assert {"train/loss", "train/acc", "train/lr", "val/acc"} <= set(metrics)
    assert metrics["train/ebops"] > 0 and torch.isfinite(metrics["train/loss_total"])
    # As a float32 training saves them, cast into the float64 port.
    saved["keras_weights"] = [
        w.astype(np.float32) if w.dtype == np.float64 else w
        for w in saved["keras_weights"]
    ]
    restored.on_load_checkpoint(saved)
    for ours, theirs in zip(restored.net.keras_model.weights, model.weights):
        assert ours.dtype == theirs.dtype
        assert np.allclose(np.asarray(ours), np.asarray(theirs), rtol=1e-6)


def test_the_evaluation_exports_the_bit_exact_reference(
    evaluated: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta = json.loads((evaluated.export / "export.json").read_text())
    with np.load(evaluated.npz) as outputs:
        clean = outputs["logits_clean"]
    with np.load(evaluated.export / "inputs.npz") as stored:
        words = stored["inputs_clean"]
    reloaded = keras.models.load_model(evaluated.export / "model.keras", compile=False)
    logits = np.asarray(reloaded.predict_on_batch(words.astype(np.float64)))
    ebops = evaluated.port.ebops_total

    # The integer bits are traced afresh on the record's whole training split, the range
    # the hardware is built for.
    assert evaluated.datamodule.requested == ["train"]
    assert words.shape == (SIZES["test"], N_SLOTS, 3)
    assert np.array_equal(logits.astype(np.float32), clean)
    assert math.isfinite(ebops) and ebops > 0
    assert (
        meta["ebops"]
        == ebops
        == float(evaluated.trainer.callback_metrics["test/ebops"])
    )
    assert (meta["row"], meta["nconstituents"]) == ("vn_hgq", N_SLOTS)
    assert (meta["n_test"], meta["floatx"]) == (SIZES["test"], "float64")
    # On a GPU, XLA's float64 2**f is off by an ulp for some f, so the stored logits
    # would not be bit-exact against alkaid.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    with pytest.raises(ValueError, match="algorithm.jax_platform=cpu"):
        _port().on_test_start()


@pytest.fixture(scope="module")
def synth():
    pytest.importorskip("alkaid")
    return load_script("scripts/synthesis/synthesise_vn.py")


def _synthesise(synth, evaluated: SimpleNamespace, out: Path, *options: str) -> dict:
    files = ("--export-dir", evaluated.export, "--outputs-npz", evaluated.npz)
    argv = [*map(str, files), "--out", str(out), "--threads", "2", *options]
    assert synth.main(argv) == 0
    return synth.read_json(out / "check.json")


def test_synthesise_vn_emulates_the_export_bit_for_bit(
    synth, evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    out = tmp_path / "synth"
    check = _synthesise(synth, evaluated, out, "--no-cosim", "--no-vivado")
    rows = list(csv.DictReader((out / "synthesis.csv").open()))
    # Keras 3.15 floors in float32 on JAX, where 626 - 2^-15 is a tie rounding to 626.
    floor = keras.ops.floor(
        keras.ops.convert_to_tensor([626.0 - 2.0**-15], "float64")
    )

    assert (check["n_mismatch"], check["n_mismatch_keras"]) == (0, 0)
    assert check["n_test"] == SIZES["test"] and check["cosim_exact"] is None
    with np.load(evaluated.npz) as probe:
        for name in ("keras", "alkaid"):
            with np.load(out / f"{name}_outputs.npz") as predicted:
                assert predicted["logits_clean"].dtype == np.float32
                assert predicted["labels"].dtype == np.int8
                assert np.array_equal(predicted["logits_clean"], probe["logits_clean"])
                assert np.array_equal(predicted["labels"], probe["labels"])
    assert (out / "prj" / "build_vivado_prj.tcl").is_file()
    assert list(rows[0]) == SYNTHESIS_COLUMNS
    assert (rows[0]["n_mismatch"], rows[0]["vivado_done"], rows[0]["lut"]) == (
        "0",
        "False",
        "nan",
    )
    assert keras.backend.standardize_dtype(floor.dtype) == "float64"
    assert np.asarray(floor).tolist() == [625.0]
    # A table written before a column was added would misalign every row appended.
    old = tmp_path / "old.csv"
    old.write_text(",".join(SYNTHESIS_COLUMNS[:-1]) + "\n")
    with pytest.raises(SystemExit, match="another header"):
        synth.main(["--out", str(out), "--report-only", "--csv", str(old)])


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
def test_the_design_cosimulates_exactly(
    synth, evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    jets = str(SIZES["test"])
    check = _synthesise(
        synth, evaluated, tmp_path / "synth", "--no-vivado", "--cosim-jets", jets
    )

    assert (check["n_cosim"], check["cosim_exact"]) == (SIZES["test"], True)
