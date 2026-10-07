import csv
import importlib.util
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy

# Before the first keras import: olgpu's ~/.keras/keras.json names tensorflow.
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "synthesis" / "synthesise_vn.py"
)
N_SLOTS = 4
N_JETS = 64

CHECK = {
    "model": "vn_hgq",
    "nconstituents": 16,
    "beta": 1e-7,
    "ebops": 1.5e5,
    "part": "xcvu13p-flga2577-2-e",
    "clock_ns": 5.0,
    "latency_cutoff": 5.0,
    "export_dir": "/export",
    "prj_dir": "/prj",
    "alkaid_version": "0.8.1",
    "hgq_version": "0.2.0",
    "keras_version": "3.15.1",
    "n_test": 100,
    "n_mismatch": 0,
    "max_abs_diff": 0.0,
    "n_mismatch_keras": 0,
    "acc_macro_keras": 0.7,
    "acc_macro_alkaid": 0.7,
    "acc_micro_alkaid": 0.71,
    "ii": 1,
    "n_cosim": 0,
    "cosim_exact": None,
}
METADATA = {"cost": 1234.0, "reg_bits": 999, "latency": 6, "clock_period": 5.0}
VIVADO = {
    "LUT": 50_000,
    "FF": 20_000,
    "DSP": 0,
    "Block RAM Tile": 0,
    "RAMB18": 0,
    "URAM": 0,
    "WNS(ns)": 0.25,
}


@pytest.fixture(scope="module")
def synth():
    spec = importlib.util.spec_from_file_location("synthesise_vn", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_row_reads_check_and_vivado_report(synth, tmp_path):
    synth.write_json(tmp_path / "check.json", CHECK)
    report = {**METADATA, **VIVADO}
    row = synth.csv_row(synth.read_json(tmp_path / "check.json"), report)
    assert tuple(row) == synth.CSV_FIELDS
    expected = {
        "lut": 50_000,
        "ff": 20_000,
        "dsp": 0,
        "bram_tile": 0,
        "latency_cycles": 6,
        "alkaid_cost": 1234.0,
        "reg_bits": 999,
        "wns_ns": 0.25,
        "vivado_done": True,
        "model": "vn_hgq",
        "beta": 1e-7,
        "n_mismatch": 0,
    }
    assert {key: row[key] for key in expected} == expected
    assert row["fmax_mhz"] == pytest.approx(1000.0 / 4.75)
    assert row["latency_ns"] == pytest.approx(6 * 4.75)
    assert math.isnan(row["cosim_exact"])


def test_csv_row_without_vivado_holds_nan(synth):
    row = synth.csv_row(CHECK, METADATA)
    assert row["vivado_done"] is False
    vivado = ("lut", "ff", "dsp", "bram_tile", "ramb18", "uram", "wns_ns", "fmax_mhz")
    assert all(math.isnan(row[column]) for column in (*vivado, "latency_ns"))
    assert (row["latency_cycles"], row["alkaid_cost"]) == (6, 1234.0)


def test_append_row_writes_the_header_once(synth, tmp_path):
    path = tmp_path / "new" / "synthesis.csv"
    for report in (METADATA, {**METADATA, **VIVADO}):
        synth.append_row(path, synth.csv_row(CHECK, report))
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 2
    assert (rows[0]["lut"], rows[1]["lut"]) == ("nan", "50000")
    assert list(rows[0]) == list(synth.CSV_FIELDS)


@pytest.mark.parametrize(
    "unlabelled, unpredicted", [((), ()), ((3,), ()), ((2, 4), (4,))]
)
def test_macro_accuracy_matches_torchmetrics(synth, unlabelled, unpredicted):
    rng = np.random.default_rng(0)
    logits, labels = rng.normal(size=(500, 5)), rng.integers(0, 5, 500)
    labels[np.isin(labels, unlabelled)] = 0
    logits[:, list(unpredicted)] -= 100.0
    metric = MulticlassAccuracy(num_classes=5)
    expected = metric(torch.from_numpy(logits), torch.from_numpy(labels)).item()
    assert synth.macro_accuracy(logits, labels) == pytest.approx(expected, abs=1e-6)


@pytest.fixture
def keras64(synth):
    pytest.importorskip("hgq")
    pytest.importorskip("alkaid")
    import jax
    import keras

    floatx, x64 = keras.config.floatx(), jax.config.jax_enable_x64
    yield synth.import_keras("float64")
    keras.config.set_floatx(floatx)
    jax.config.update("jax_enable_x64", x64)


def _fixed(k: int, i: int, f: int, overflow: str):
    from hgq.config import QuantizerConfig

    kif = {"k0": k, "i0": i, "f0": f, "overflow_mode": overflow, "trainable": False}
    return QuantizerConfig("kif", "datalane", **kif)


def _tiny_model(keras):
    from hgq.layers import QDense, QSum
    from hgq.quantizer import Quantizer

    # alkaid traces only WRAP quantisers on a model input.
    pt, angles = keras.Input((N_SLOTS, 1)), keras.Input((N_SLOTS, 2))
    words = [Quantizer(_fixed(0, 12, 2, "WRAP"))(pt)]
    words.append(Quantizer(_fixed(1, 1, 8, "WRAP"))(angles))
    x = QDense(8, activation="relu")(keras.layers.Concatenate()(words))
    x = QDense(5, enable_oq=True, oq_conf=_fixed(1, 6, 8, "SAT"))(QSum(axes=1)(x))
    return keras.Model([pt, angles], x)


def _words(rng) -> np.ndarray:
    pt = rng.integers(1, 400, (N_JETS, N_SLOTS, 1)) * 0.25
    angles = rng.integers(-512, 512, (N_JETS, N_SLOTS, 2)) / 256
    words = np.concatenate([pt, angles], axis=-1)
    words[: N_JETS // 2, -1] = 0.0
    return words


def _export(synth, keras, directory: Path) -> list[str]:
    from hgq.utils import trace_minmax

    rng = np.random.default_rng(0)
    model, words = _tiny_model(keras), _words(rng)
    parts = synth.split_inputs(model, words)
    trace_minmax(model, parts)
    export_dir = directory / "outputs" / "main_hgq"
    export_dir.mkdir(parents=True)
    model.save(export_dir / "model.keras")
    np.savez(export_dir / "inputs.npz", inputs_clean=words.astype(np.float32))
    # float32 as the probe stores them: exact behind the 15-bit output quantiser.
    logits = model.predict(parts, verbose=0).astype(np.float32)
    labels = rng.integers(0, 5, N_JETS).astype(np.int8)
    np.savez(directory / "outputs" / "main.npz", logits_clean=logits, labels=labels)
    meta = {"row": "vn_hgq", "nconstituents": N_SLOTS, "beta": 1e-7, "ebops": 1.0}
    synth.write_json(export_dir / "export.json", {**meta, "floatx": "float64"})
    npz = directory / "outputs" / "main.npz"
    return ["--export-dir", str(export_dir), "--outputs-npz", str(npz)]


def test_tiny_export_is_bit_exact(synth, keras64, tmp_path):
    out = tmp_path / "synth"
    argv = [*_export(synth, keras64, tmp_path), "--out", str(out), "--threads", "2"]
    assert synth.main([*argv, "--no-cosim", "--no-vivado"]) == 0
    check = synth.read_json(out / "check.json")
    expected = {"n_mismatch": 0, "n_mismatch_keras": 0, "n_test": N_JETS, "ii": 1}
    assert {key: check[key] for key in expected} == expected
    assert check["acc_macro_alkaid"] == check["acc_macro_keras"]
    assert check["cosim_exact"] is None
    for name in ("metadata.json", "report.json", "build_vivado_prj.tcl"):
        assert (out / "prj" / name).is_file()
    assert synth.main(["--out", str(out), "--report-only"]) == 0
    rows = list(csv.DictReader((out / "synthesis.csv").open()))
    assert len(rows) == 2
    expected = {"n_mismatch": "0", "vivado_done": "False", "lut": "nan"}
    assert {key: rows[0][key] for key in expected} == expected


@pytest.mark.skipif(shutil.which("verilator") is None, reason="no Verilator")
def test_tiny_export_cosimulates_exactly(synth, keras64, tmp_path):
    out = tmp_path / "synth"
    argv = [*_export(synth, keras64, tmp_path), "--out", str(out), "--threads", "2"]
    assert synth.main([*argv, "--no-vivado", "--cosim-jets", str(N_JETS)]) == 0
    check = synth.read_json(out / "check.json")
    assert (check["n_cosim"], check["cosim_exact"]) == (N_JETS, True)
