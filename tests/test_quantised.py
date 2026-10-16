import os

# Before anything imports keras: the olgpu ~/.keras/keras.json names tensorflow.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("KERAS_BACKEND", "jax")

import contextlib
import csv
import json
import logging
import math
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra.errors import InstantiationException
from lightning import Callback, LightningDataModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

pytest.importorskip("hgq")

import jax
import keras
from hgq.layers.core import QLayerBase
from hgq.quantizer.internal import FixedPointQuantizerKBI, FixedPointQuantizerKIF

from src.algorithms.components.hgq_port import WEIGHT_I_MIN
from src.algorithms.utils.keras_schedules import CosineRestarts
from src.callbacks.invariance import InvarianceMetricCallback
from src.data.components.dataset import HLS4MLDataset
from src.utils.checkpoints import Manifest, ManifestRun, write_manifest
from tests.conftest import (
    PT_ETA_PHI,
    PT_ETAREL_PHIREL,
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
# (gate, mlp_layers, stream_channels, tensor_invariants, fold_norm, axis_offset): the
# network of the finals and the latency ablation's variants, each switching one more
# piece off, the batch normalisations folded into the next layer in every other one,
# then the finals' network and the cheap one reading the leader's offset from the full
# jet's axis, unfolded and folded.
VARIANTS = [
    (True, 2, 8, 2, False, False),
    (False, 2, 8, 2, True, False),
    (False, 1, 8, 2, False, False),
    (False, 1, 0, 2, True, False),
    (False, 1, 0, 0, False, False),
    (True, 2, 8, 2, False, True),
    (False, 2, 0, 2, True, True),
]
AXIS_FEATURES = [*PT_ETA_PHI, "etarel", "phirel"]
COLUMNS = [RECORD_COLUMNS.index(f"j1_{name}") for name in PT_ETA_PHI]
AXIS_COLUMNS = [RECORD_COLUMNS.index(f"j1_{name}") for name in ("etarel", "phirel")]
# A port whose p_T, angle and offset words are trained, the offset reading the axis.
TRAINED_WORDS = {
    "axis_offset": True,
    "nbr_feats": len(AXIS_FEATURES),
    "quantisation": {"trained_words": True},
}
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
# The scales of the stored columns the JEDI port reads, about the record's at N = 16: p_T
# in GeV, the angles in radians.
JEDI_SCALES = {"pt": 1559.19, "etarel": 1.242, "phirel": 0.463, "eta": 2.4, "phi": 3.14}
JEDI_ARCHITECTURE = {"hidden": 8, "head_layers": [16, 8]}
# The trigger's words of the absolute columns, the angle word wide enough for phi.
JEDI_ABSOLUTE = {"inputs": "words", "quantisation": {"angle_kif": [1, 2, 8]}}
JEDI_DENSE = {"embed", "local", "context", "shared", "head0", "head1", "head_last"}
JEDI_LAYERS = JEDI_DENSE | {"pool0", "context_add", "pool1"}
# What each JEDI export records of its port: the columns, the input mode and whether the
# datalane widths are per slot.
JEDI_EXPORTS = {
    "jedi_evaluated": (PT_ETAREL_PHIREL, "standardised", True),
    "jedi_words_evaluated": (PT_ETA_PHI, "words", False),
}
# The per-epoch INFO message of a quantised row, which the campaign scripts parse.
EPOCH_LINE = (
    r"\[epoch (?P<epoch>[ \d]{3}\d)\] EBOPs \d+, beta \S+, lr \S+, "
    r"train acc (?P<acc>\d\.\d{4})$"
)


def _record(n_jets: int, seed: int) -> torch.Tensor:
    # The record's 16 columns, of which the module reads pt (GeV), eta and phi, and
    # etarel and phirel under the axis offset: 1 to N_SLOTS real constituents of at
    # least 1 GeV, so that every keep flag min(4 p_T, 1) is one, and the padded rows all
    # zero.
    generator = torch.Generator().manual_seed(seed)
    gauss = torch.randn(3, n_jets, N_SLOTS, generator=generator)
    columns = [1.0 + 40.0 * gauss[0].exp(), 0.4 + 0.25 * gauss[1], 0.3 * gauss[2]]
    x = torch.zeros(n_jets, N_SLOTS, len(RECORD_COLUMNS))
    x[..., COLUMNS] = torch.stack(columns, dim=-1)
    multiplicity = torch.randint(1, N_SLOTS + 1, (n_jets, 1), generator=generator)
    x[..., AXIS_COLUMNS] = 0.3 * torch.randn(n_jets, N_SLOTS, 2, generator=generator)
    x[torch.arange(N_SLOTS) >= multiplicity] = 0.0
    return x


def _stored(x: torch.Tensor, norm_scales: dict[str, float] | None) -> torch.Tensor:
    # The record's columns as stored, those named in norm_scales divided by their scale.
    for name, scale in (norm_scales or {}).items():
        x[..., RECORD_COLUMNS.index(f"j1_{name}")] /= scale
    return x


class _Record(LightningDataModule):
    # What the module, the probe and eval.py read of HLS4MLDataModule: training and
    # validation batches of *features*, test batches of all 16 columns, the fit's
    # training jets, the record's splits whole and the scales of the stored columns.
    feature_names = RECORD_COLUMNS
    keep_full_features_at_test = True

    def __init__(
        self,
        features: list[str] = PT_ETA_PHI,
        norm_scales: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.features, self.norm_scales = features, norm_scales
        self.columns = [RECORD_COLUMNS.index(f"j1_{name}") for name in features]
        self.splits = {
            split: (_stored(_record(n_jets, seed), norm_scales), labels(n_jets, seed))
            for seed, (split, n_jets) in enumerate(SIZES.items())
        }
        self.data_train = self._dataset("train", self.columns)
        self.requested: list[str] = []

    def _dataset(self, split: str, columns=None, **kwargs) -> HLS4MLDataset:
        x, y = self.splits[split]
        x = x if columns is None else x[..., columns]
        return HLS4MLDataset(x, y, batch_size=BATCH, **kwargs)

    def feature_indices(self) -> list[int]:
        return self.columns

    def load_split(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        self.requested.append(split)
        return self.splits[split]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.data_train, batch_size=None)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._dataset("validation", self.columns), batch_size=None)

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
    "gate, mlp_layers, stream_channels, tensor_invariants, fold_norm, axis_offset",
    VARIANTS,
)
def test_the_port_copies_a_float_vn_so_keras_and_torch_agree(
    tmp_path: Path,
    gate: bool,
    mlp_layers: int,
    stream_channels: int,
    tensor_invariants: int,
    fold_norm: bool,
    axis_offset: bool,
) -> None:
    # A float vn of random weights saved as train.py leaves a run, a checkpoint and the
    # manifest naming it, which init_from reaches through the run directory.
    features = AXIS_FEATURES if axis_offset else PT_ETA_PHI
    variant = dict(
        gate=gate,
        mlp_layers=mlp_layers,
        stream_channels=stream_channels,
        tensor_invariants=tensor_invariants,
        axis_offset=axis_offset,
        nbr_feats=len(features),
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
    datamodule = _Record(features)
    x = datamodule.splits["test"][0][..., datamodule.columns]
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
    assert meta["port"] == "vn_hgq"
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


def test_the_estimate_predicts_the_project_of_its_setting(
    synth, evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    flow = ("--latency-cutoff", "4.2", "--ternary", "aware")
    estimated = tmp_path / "estimate"
    argv = ["--export-dir", str(evaluated.export), "--out", str(estimated)]
    assert synth.main([*argv, "--estimate-only", *flow]) == 0
    estimate = synth.read_json(estimated / "estimate.json")
    setting = ("--method0", estimate["method0"], "--hard-dc", str(estimate["hard_dc"]))
    options = ("--no-cosim", "--no-vivado", *flow, *setting)
    _synthesise(synth, evaluated, tmp_path / "synth", *options)
    metadata = synth.read_json(tmp_path / "synth" / "prj" / "metadata.json")

    # The cycles and the cost of the project the scan's setting writes.
    assert estimate["cycles"] == metadata["latency"]
    assert estimate["cost"] == pytest.approx(metadata["cost"])
    assert estimate["latency_ns"] == 5.0 * estimate["cycles"]
    assert estimate["dsp"] >= 0 and 0 < estimate["lut"] < estimate["cost"]
    assert not (estimated / "prj").exists()


def test_the_trace_keeps_a_constant_product_of_a_matrix_row(synth) -> None:
    from alkaid.trace import FVArrayInput, trace

    synth._keep_constant_products()
    inputs = FVArrayInput((2, 3))
    # Row 1 keeps a variable that meets only zero weights beside two constants at zero
    # bits, so that its product is constant: alkaid 0.8.1 raised StopIteration on it.
    word = np.array([[1, 1, 1], [1, 0, 0]])
    k, i, f = word, 3 * word, 4 * word
    weights = np.array([[0.0, 0.0], [0.25, 3.0], [-1.0, 0.5]])
    comb = trace(inputs, inputs.quantize(k=k, i=i, f=f) @ weights)
    data = np.random.default_rng(0).uniform(-8, 8, size=(1000, 6))
    words = np.floor(data * 16) / 16 * word.ravel()
    expected = (words.reshape(-1, 2, 3) @ weights).reshape(-1, 4)

    assert np.array_equal(np.asarray(comb.predict(data), np.float64), expected)


@pytest.fixture(scope="module")
def axis_evaluated(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    # A port reading the leader's offset from the full jet's axis, its bits seeded as a
    # fit seeds them and left untrained, then tested as eval.py tests a checkpoint.
    root = tmp_path_factory.mktemp("axis")
    probe = InvarianceMetricCallback(probe=False)
    probe.subdir = "main"
    datamodule = _Record(AXIS_FEATURES)
    port = _port(axis_offset=True, nbr_feats=len(AXIS_FEATURES))
    attach(port, datamodule)
    trainer = _trainer(root, probe, enable_checkpointing=False)
    trainer.test(port, datamodule=datamodule)
    outputs = root / "outputs"
    return SimpleNamespace(export=outputs / "main_hgq", npz=outputs / "main.npz")


def test_synthesise_vn_emulates_the_axis_offset_port_bit_for_bit(
    synth, axis_evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    out = tmp_path / "synth"
    check = _synthesise(synth, axis_evaluated, out, "--no-cosim", "--no-vivado")
    with np.load(axis_evaluated.export / "inputs.npz") as stored:
        words = stored["inputs_clean"]

    # The leader's two offset words, and zeros that nothing reads below them.
    assert words.shape == (SIZES["test"], N_SLOTS, 5)
    assert np.any(words[:, 0, 3:]) and not np.any(words[:, 1:, 3:])
    # The reloaded model.keras on the stored words and alkaid's emulation both give the
    # probe's logits.
    assert (check["n_mismatch"], check["n_mismatch_keras"]) == (0, 0)
    assert check["n_test"] == SIZES["test"]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
def test_the_design_cosimulates_exactly(
    synth, evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    jets = str(SIZES["test"])
    check = _synthesise(
        synth, evaluated, tmp_path / "synth", "--no-vivado", "--cosim-jets", jets
    )

    assert (check["n_cosim"], check["cosim_exact"]) == (SIZES["test"], True)


def _word_quantiser(model: keras.Model, name: str) -> FixedPointQuantizerKIF:
    lanes = model.get_layer(name)._flatten_layers()
    return next(lane for lane in lanes if isinstance(lane, FixedPointQuantizerKIF))


def test_trained_words_start_at_the_trigger_widths_one_per_slot() -> None:
    names = ("pt_word", "angle_word", "offset_word")
    trained = attach(_port(**TRAINED_WORDS), _Record(AXIS_FEATURES)).net.keras_model
    fixed = _port(axis_offset=True, nbr_feats=len(AXIS_FEATURES)).net.keras_model
    pt, angles, offset = (
        keras.ops.convert_to_numpy(_word_quantiser(trained, name)._f) for name in names
    )

    # One width per slot, the two angles of a slot sharing theirs and the two offset
    # words one, each seeded at its trigger word's fraction bits.
    assert (pt.size, angles.size, offset.size) == (N_SLOTS, N_SLOTS, 1)
    assert np.all(pt == 2) and np.all(angles == 8) and np.all(offset == 8)
    assert all(_word_quantiser(trained, name).trainable for name in names)
    assert not any(_word_quantiser(fixed, name).trainable for name in names)


@pytest.fixture(scope="module")
def trained_words_evaluated(tmp_path_factory: pytest.TempPathFactory):
    # Two epochs with trained words, then the checkpoint tested as eval.py tests it.
    root = tmp_path_factory.mktemp("trained_words")
    checkpoint = ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="epoch_{epoch:03d}",
        monitor=None,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    fit, module = _trainer(root, checkpoint, max_epochs=2), _port(**TRAINED_WORDS)
    fit.fit(module, datamodule=_Record(AXIS_FEATURES))
    probe = InvarianceMetricCallback(probe=False)
    probe.subdir = "main"
    test = _trainer(root / "eval", probe, enable_checkpointing=False)
    port, datamodule = _port(**TRAINED_WORDS), _Record(AXIS_FEATURES)
    ckpt_path = checkpoint.best_model_path
    test.test(port, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    outputs = root / "eval" / "outputs"
    return SimpleNamespace(export=outputs / "main_hgq", npz=outputs / "main.npz")


def test_synthesise_vn_emulates_the_trained_word_port_bit_for_bit(
    synth, trained_words_evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    out = tmp_path / "synth"
    check = _synthesise(
        synth, trained_words_evaluated, out, "--no-cosim", "--no-vivado"
    )

    assert (check["n_mismatch"], check["n_mismatch_keras"]) == (0, 0)
    assert check["n_test"] == SIZES["test"]


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
def test_the_trained_word_design_cosimulates_exactly(
    synth, trained_words_evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    jets = str(SIZES["test"])
    check = _synthesise(
        synth,
        trained_words_evaluated,
        tmp_path / "synth",
        "--no-vivado",
        "--cosim-jets",
        jets,
    )

    assert (check["n_cosim"], check["cosim_exact"]) == (SIZES["test"], True)


def _jedi_port(features: list[str] = PT_ETAREL_PHIREL, **overrides: object):
    # Seeded, so that two ports built alike start from the same weights.
    torch.manual_seed(0)
    overrides = JEDI_ARCHITECTURE | overrides
    return algorithm("jedi_linear_hgq", N_SLOTS, features=features, **overrides)


def _kbi_quantisers(model: keras.Model) -> list[FixedPointQuantizerKBI]:
    return [x for x in model._flatten_layers() if isinstance(x, FixedPointQuantizerKBI)]


@pytest.mark.parametrize("per_slot", [True, False])
@pytest.mark.parametrize("inputs", ["standardised", "words"])
def test_the_jedi_port_builds_the_published_network(
    inputs: str, per_slot: bool
) -> None:
    port = _jedi_port(inputs=inputs, per_slot=per_slot)
    model, words = port.net.keras_model, port.net.inputs_model
    names = {layer.name for layer in model.layers}
    embed = model.get_layer("embed")
    weights = _kbi_quantisers(model)

    assert JEDI_LAYERS <= names
    # Under words the fixed trigger words come first; standardised rows meet the first
    # layer's trainable quantiser alone.
    assert ({"pt_word", "angle_word"} <= names) == (inputs == "words")
    assert words.output_shape == model.input_shape == (None, N_SLOTS, 3)
    assert model.output_shape == (None, 5)
    # One width per slot and feature, or one per feature shared by the slots.
    assert tuple(embed.iq.quantizer._f.shape) == (1, N_SLOTS if per_slot else 1, 3)
    assert model.get_layer("pool0").scale == 1 / N_SLOTS
    assert model.get_layer("pool1").scale == 1 / 16
    assert not model.get_layer("head_last").enable_oq
    # Every weight and bias of seven bits, saturating, with WEIGHT_I_MIN integer bits
    # at least.
    assert len(weights) == 2 * len(JEDI_DENSE)
    assert {q.overflow_mode for q in weights} == {"SAT_SYM"}
    assert all(np.all(np.asarray(q.b) == 7) for q in weights)
    assert {q.i_constraint.min_value for q in weights} == {WEIGHT_I_MIN}


def test_the_jedi_port_refuses_other_columns_and_a_narrow_angle_word() -> None:
    absolute = _Record(PT_ETA_PHI, JEDI_SCALES)

    with pytest.raises(ValueError, match="but data.features is"):
        attach(_jedi_port(), _Record([*PT_ETA_PHI, "etarel"], JEDI_SCALES))
    with pytest.raises(ValueError, match=r"angle_kif to \[1,2,8\]"):
        attach(_jedi_port(PT_ETA_PHI, inputs="words"), absolute)
    with pytest.raises(InstantiationException, match="inputs must be one of"):
        _jedi_port(inputs="raw")
    # Standardised rows have no angle word, and the wider word holds any phi.
    attach(_jedi_port(PT_ETA_PHI), absolute)
    attach(_jedi_port(PT_ETA_PHI, **JEDI_ABSOLUTE), absolute)


def _jedi_linear_rate(epoch: int) -> float:
    # cosine_decay_restarts of JEDI-linear's src/train.py at its N = 16 settings.
    cycle_t = min(epoch % 100 / (100 - 10), 1)
    return 1e-6 + 0.5 * (3e-3 - 1e-6) * (1 + math.cos(math.pi * cycle_t))


def test_the_restarting_cosine_follows_jedi_linear_at_every_epoch_start() -> None:
    steps = 4
    schedule = CosineRestarts(
        3e-3, 200 * steps, alpha=1e-6 / 3e-3, cycles=2, flat_fraction=0.1
    )
    # The absolute tolerance is float32's rounding of 1 + cos near the cycle's floor.
    for epoch in (0, 45, 89, 90, 95, 99, 100, 145, 199):
        rate = float(schedule(epoch * steps))
        assert rate == pytest.approx(_jedi_linear_rate(epoch), rel=1e-6, abs=1e-9)
    with pytest.raises(ValueError, match="warm-up"):
        CosineRestarts(3e-3, 200 * steps, warmup_steps=10)


def _last_epoch(root: Path) -> ModelCheckpoint:
    # As configs/callbacks/quantised.yaml checkpoints: unranked, the newest epoch alone.
    return ModelCheckpoint(
        dirpath=root / "checkpoints",
        filename="epoch_{epoch:03d}",
        monitor=None,
        save_top_k=1,
        save_last=False,
        auto_insert_metric_name=False,
    )


def _main_probe() -> InvarianceMetricCallback:
    # Writes outputs/main.npz, beside which the port writes its export.
    probe = InvarianceMetricCallback(probe=False)
    probe.subdir = "main"
    return probe


@contextlib.contextmanager
def _info_lines(name: str) -> Iterator[list[str]]:
    # The INFO messages of the logger *name*, which caplog cannot collect for a
    # module-scoped fixture.
    logger, lines, level = logging.getLogger(name), [], logging.getLogger(name).level
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield lines
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@pytest.fixture(scope="module")
def jedi_fitted(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    # Two epochs on standardised rows under the beta schedule, the per-epoch lines kept.
    root = tmp_path_factory.mktemp("jedi_fit")
    datamodule, betas = _Record(PT_ETAREL_PHIREL, JEDI_SCALES), _BetaRecorder()
    module = _jedi_port(beta_schedule=SCHEDULE)
    trainer = _trainer(root, _last_epoch(root), betas, max_epochs=2)
    with _info_lines("src.algorithms.keras_base") as lines:
        trainer.fit(module, datamodule=datamodule)
    return SimpleNamespace(
        datamodule=datamodule, module=module, trainer=trainer, betas=betas, lines=lines
    )


def test_the_jedi_standardisation_is_that_of_the_pooled_splits_padding_included(
    jedi_fitted: SimpleNamespace,
) -> None:
    datamodule, module = jedi_fitted.datamodule, jedi_fitted.module
    scales = torch.tensor([JEDI_SCALES[name] for name in PT_ETAREL_PHIREL])
    pooled = torch.cat([datamodule.splits[s][0] for s in ("train", "validation")])
    rows = (pooled[..., datamodule.columns] * scales).double().numpy()
    pt = rows[..., 0]

    assert datamodule.requested == ["train", "validation"]
    assert np.allclose(module.input_mean.numpy(), rows.mean(axis=(0, 1)), rtol=1e-10)
    assert np.allclose(module.input_std.numpy(), rows.std(axis=(0, 1)), rtol=1e-10)
    # The padded rows count: the real constituents alone have a larger mean p_T.
    assert pt[pt > 0].mean() > 1.2 * float(module.input_mean[0])


def test_the_jedi_module_trains_from_scratch_and_round_trips_a_checkpoint(
    jedi_fitted: SimpleNamespace,
) -> None:
    module, model = jedi_fitted.module, jedi_fitted.module.net.keras_model
    steps = 2 * len(jedi_fitted.datamodule.train_dataloader())
    best = jedi_fitted.trainer.checkpoint_callback.best_model_path
    saved = torch.load(best, weights_only=False)
    jets = jedi_fitted.datamodule.data_train.X
    rows = module._keras_inputs(jets)
    restored = _jedi_port()
    untrained = restored.net.keras_model.predict_on_batch(rows)
    restored.load_state_dict(saved["state_dict"])
    restored.on_load_checkpoint(saved)
    metrics = jedi_fitted.trainer.callback_metrics
    scheduled = [np.unique(betas).tolist() for betas in jedi_fitted.betas.seen]
    epochs = [re.search(EPOCH_LINE, line) for line in jedi_fitted.lines]
    epochs = [match for match in epochs if match]

    assert jedi_fitted.trainer.global_step == steps
    assert [path.name for path in Path(best).parent.iterdir()] == ["epoch_001.ckpt"]
    # The standardisation travels in the checkpoint beside the Keras weights.
    assert not torch.equal(module.input_std, torch.ones(3, dtype=torch.float64))
    assert np.array_equal(restored._keras_inputs(jets), rows)
    assert not np.array_equal(untrained, model.predict_on_batch(rows))
    assert np.array_equal(
        restored.net.keras_model.predict_on_batch(rows), model.predict_on_batch(rows)
    )
    assert [len(betas) for betas in scheduled] == [1, 1]
    assert [betas[0] for betas in scheduled] == pytest.approx([1e-6, 3e-6])
    assert metrics["train/ebops"] > 0 and torch.isfinite(metrics["train/loss_total"])
    # One line per epoch, which the campaign scripts parse; its accuracy is the mean of
    # Keras's over the epoch's jets, as Lightning's epoch mean of train/acc.
    assert [match["epoch"] for match in epochs] == ["   0", "   1"]
    assert float(epochs[-1]["acc"]) == pytest.approx(
        float(metrics["train/acc"]), abs=1e-4
    )


@pytest.fixture(scope="module")
def jedi_evaluated(
    jedi_fitted: SimpleNamespace, tmp_path_factory: pytest.TempPathFactory
) -> SimpleNamespace:
    # As eval.py runs it: a port built anew and the checkpoint loaded into it.
    root = tmp_path_factory.mktemp("jedi_eval")
    datamodule = _Record(PT_ETAREL_PHIREL, JEDI_SCALES)
    port = _jedi_port(beta_schedule=SCHEDULE)
    trainer = _trainer(root, _main_probe(), enable_checkpointing=False)
    ckpt_path = jedi_fitted.trainer.checkpoint_callback.best_model_path
    trainer.test(port, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False)
    outputs = root / "outputs"
    return SimpleNamespace(
        export=outputs / "main_hgq",
        npz=outputs / "main.npz",
        datamodule=datamodule,
        port=port,
    )


@pytest.fixture(scope="module")
def jedi_words_evaluated(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    # The trigger's words of the absolute columns, the widths shared by the slots,
    # untrained and tested as eval.py tests a checkpoint.
    root = tmp_path_factory.mktemp("jedi_words")
    datamodule = _Record(PT_ETA_PHI, JEDI_SCALES)
    port = _jedi_port(PT_ETA_PHI, per_slot=False, **JEDI_ABSOLUTE)
    attach(port, datamodule)
    _trainer(root, _main_probe(), enable_checkpointing=False).test(
        port, datamodule=datamodule
    )
    outputs = root / "outputs"
    return SimpleNamespace(
        export=outputs / "main_hgq",
        npz=outputs / "main.npz",
        datamodule=datamodule,
        port=port,
    )


@pytest.mark.parametrize("exported", list(JEDI_EXPORTS))
def test_the_jedi_evaluation_exports_the_bit_exact_reference(
    exported: str, request: pytest.FixtureRequest
) -> None:
    evaluated = request.getfixturevalue(exported)
    port = evaluated.port
    meta = json.loads((evaluated.export / "export.json").read_text())
    with np.load(evaluated.npz) as outputs:
        clean = outputs["logits_clean"]
    with np.load(evaluated.export / "inputs.npz") as stored:
        words = stored["inputs_clean"]
    reloaded = keras.models.load_model(evaluated.export / "model.keras", compile=False)
    logits = np.asarray(reloaded.predict_on_batch(words.astype(np.float64)))
    features, inputs, per_slot = JEDI_EXPORTS[exported]
    standardised = inputs == "standardised"
    moments = {"mean": port.input_mean.tolist(), "std": port.input_std.tolist()}

    assert evaluated.datamodule.requested == ["train"]
    assert words.shape == (SIZES["test"], N_SLOTS, 3)
    assert np.array_equal(logits.astype(np.float32), clean)
    # The stored inputs are those the port reads, on which its input quantisers are the
    # identity.
    assert np.array_equal(port.net.quantised_inputs(words.astype(np.float64)), words)
    assert (meta["row"], meta["port"]) == ("jedi_linear_hgq", "jedi_linear_hgq")
    assert meta["ebops"] == port.ebops_total > 0
    assert (meta["features"], meta["inputs"], meta["per_slot"]) == (
        features,
        inputs,
        per_slot,
    )
    assert meta["round_mode"] == "RND"
    assert meta["standardisation"] == (moments if standardised else None)
    assert (meta["quantisation"] is None) == standardised
    # The words in GeV and radians, sorted by decreasing p_T, the padded rows zero.
    if not standardised:
        assert np.all(np.diff(words[..., 0], axis=1) <= 0)
        assert np.abs(words[..., 0]).max() > 1.0
        assert not np.any(words[words[..., 0] == 0])


@pytest.mark.parametrize("exported", list(JEDI_EXPORTS))
def test_synthesise_vn_emulates_the_jedi_exports_bit_for_bit(
    synth, exported: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    out = tmp_path / "synth"
    evaluated = request.getfixturevalue(exported)
    check = _synthesise(synth, evaluated, out, "--no-cosim", "--no-vivado")
    tcl = (out / "prj" / "build_vivado_prj.tcl").read_text()

    assert (check["n_mismatch"], check["n_mismatch_keras"]) == (0, 0)
    assert check["n_test"] == SIZES["test"]
    # The top module, and the Vivado output directory, take the port's name.
    assert check["project"] == "jedi_linear_hgq"
    assert 'set project_name "jedi_linear_hgq"' in tcl


def test_synthesise_vn_names_the_project_from_the_option_or_the_export(
    synth, evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    # A vn export written before export.json named its port.
    export = tmp_path / "main_hgq"
    shutil.copytree(evaluated.export, export)
    meta = json.loads((export / "export.json").read_text())
    del meta["port"]
    (export / "export.json").write_text(json.dumps(meta))
    old = SimpleNamespace(export=export, npz=evaluated.npz)
    check = _synthesise(synth, old, tmp_path / "synth", "--no-cosim", "--no-vivado")
    tcl = (tmp_path / "synth" / "prj" / "build_vivado_prj.tcl").read_text()
    custom = SimpleNamespace(project="custom")

    assert check["project"] == "vn_hgq"
    assert 'set project_name "vn_hgq"' in tcl
    assert synth.project_name(custom, {"port": "jedi_linear_hgq"}) == "custom"


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
@pytest.mark.parametrize("exported", list(JEDI_EXPORTS))
def test_the_jedi_design_cosimulates_exactly(
    synth, exported: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    jets = str(SIZES["test"])
    evaluated = request.getfixturevalue(exported)
    options = ("--no-vivado", "--cosim-jets", jets)
    check = _synthesise(synth, evaluated, tmp_path / "synth", *options)

    assert (check["n_cosim"], check["cosim_exact"]) == (SIZES["test"], True)
