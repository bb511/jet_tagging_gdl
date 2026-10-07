#!/usr/bin/env python3
"""Synthesise an exported ``vn_hgq`` network with alkaid and append one CSV row.

The input is what ``src/eval.py`` leaves for a ``vn_hgq`` checkpoint: the directory
``outputs/<label>_hgq/`` (``model.keras``, ``inputs.npz``, ``export.json``) and the
probe's ``outputs/<label>.npz``, whose clean logits are the bit-exact reference.

1. Keras runs on JAX's CPU backend at the ``floatx`` of ``export.json`` (float64
   reproduces the fixed-point arithmetic exactly, float32 does not); another backend is
   refused, since XLA's float64 2**f is inexact on a GPU;
2. ``import hgq`` registers its layers and alkaid's plugin for them; the model loads;
3. Keras predicts the exported words, and a drift from the probe's logits is logged;
4. alkaid traces the model into combinational logic, as ``alkaid convert`` does;
5. the Keras and the alkaid predictions are written to ``<out>/keras_outputs.npz`` and
   ``<out>/alkaid_outputs.npz`` in the layout of the probe's file, so that
   ``scripts/analysis/compare_outputs.py`` reads off either its working points and its
   ROC gap against the probe's logits; alkaid's emulation is then compared with the
   probe's logits: ``n_mismatch`` counts the logit entries that differ, and more than
   ``--max-mismatch`` stops the run, after the two files are written; the macro (mean
   per-class recall, torchmetrics' default) and micro accuracies follow;
6. the Verilog project ``<out>/prj/`` is written for ``--part`` at ``--clock`` ns, and
   ``<out>/check.json`` records steps 3 to 7;
7. unless ``--no-cosim``, Verilator co-simulates ``--cosim-jets`` jets, which must
   match alkaid's emulation exactly;
8. unless ``--no-vivado``, Vivado implements the project out of context through
   alkaid's ``build_vivado_prj.tcl`` (reports in ``prj/output_vn_hgq/reports/``);
9. alkaid's project report (``prj/report.json``) and ``check.json`` make one row of
   ``--csv``; the Vivado columns hold NaN when Vivado has not run.

Nothing is imported from ``src``, so the script runs wherever
``scripts/synthesis/requirements.txt`` is installed. A project written on one machine
and implemented on another takes two calls, the second (step 9 alone) once Vivado has
run in ``<dir>/prj``::

    python scripts/synthesis/synthesise_vn.py --export-dir <eval dir>/outputs/main_hgq \\
        --outputs-npz <eval dir>/outputs/main.npz --out <dir> --no-cosim --no-vivado
    python scripts/synthesis/synthesise_vn.py --out <dir> --report-only
"""

import argparse
import csv
import json
import logging
import math
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

LOG = logging.getLogger("synthesise_vn")

# The top module, which also names the Vivado output directory output_<PROJECT>/.
PROJECT = "vn_hgq"
VIVADO_ARGS = (
    "-mode",
    "batch",
    "-nojournal",
    "-log",
    "vivado.log",
    "-source",
    "build_vivado_prj.tcl",
)
PREDICT_BATCH = 16384
DISTRIBUTIONS = (("alkaid", "alkaid"), ("hgq", "hgq2"), ("keras", "keras"))

# Command-line option -> keyword of alkaid's RTLModel.
RTL_OPTIONS = {
    "part": "part_name",
    "clock": "clock_period",
    "clock_uncertainty": "clock_uncertainty",
    "latency_cutoff": "latency_cutoff",
}

# CSV column -> key of alkaid's project report, which merges metadata.json (cost,
# registers, latency in cycles, nominal clock) with the post-route Vivado reports.
REPORT_FIELDS = {
    "latency_cycles": "latency",
    "alkaid_cost": "cost",
    "reg_bits": "reg_bits",
    "lut": "LUT",
    "ff": "FF",
    "dsp": "DSP",
    "bram_tile": "Block RAM Tile",
    "ramb18": "RAMB18",
    "uram": "URAM",
}
VIVADO_DONE = {"LUT", "WNS(ns)"}

CSV_FIELDS = (
    "model",
    "nconstituents",
    "beta",
    "ebops",
    "part",
    "clock_ns",
    "latency_cutoff",
    "ii",
    "latency_cycles",
    "latency_ns",
    "lut",
    "ff",
    "dsp",
    "bram_tile",
    "ramb18",
    "uram",
    "wns_ns",
    "fmax_mhz",
    "alkaid_cost",
    "reg_bits",
    "acc_macro_keras",
    "acc_macro_alkaid",
    "acc_micro_alkaid",
    "n_test",
    "n_mismatch",
    "max_abs_diff",
    "n_cosim",
    "cosim_exact",
    "vivado_done",
    "export_dir",
    "prj_dir",
    "alkaid_version",
    "hgq_version",
    "keras_version",
    "timestamp",
)

OPTIONS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("--export-dir", {"type": Path, "help": "outputs/<label>_hgq/ of a vn_hgq eval"}),
    ("--outputs-npz", {"type": Path, "help": "the probe's outputs/<label>.npz"}),
    ("--out", {"type": Path, "required": True, "help": "for check.json and prj/"}),
    ("--csv", {"type": Path, "help": "CSV appended to; None: <out>/synthesis.csv"}),
    ("--part", {"default": "xcvu13p-flga2577-2-e", "help": "FPGA part"}),
    ("--clock", {"type": float, "default": 5.0, "help": "clock period (ns)"}),
    (
        "--clock-uncertainty",
        {"type": float, "default": 0.1, "help": "fraction of the clock period"},
    ),
    (
        "--latency-cutoff",
        {"type": float, "default": 5.0, "help": "combinational delay per stage (ns)"},
    ),
    ("--hard-dc", {"type": int, "default": 5, "help": "alkaid's delay constraint"}),
    ("--cosim-jets", {"type": int, "default": 1000, "help": "jets co-simulated"}),
    ("--no-cosim", {"action": "store_true", "help": "skip the co-simulation"}),
    ("--no-vivado", {"action": "store_true", "help": "skip the Vivado run"}),
    ("--vivado", {"default": "vivado", "help": "Vivado executable"}),
    ("--threads", {"type": int, "default": 16, "help": "emulation, Verilator"}),
    (
        "--max-mismatch",
        {"type": int, "default": 0, "help": "differing logit entries tolerated"},
    ),
    (
        "--report-only",
        {"action": "store_true", "help": "only <out>/prj and check.json to a row"},
    ),
)


class _Formatter(
    argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter
):
    """Keep the steps of the description as written and show every default."""


def _timed(step: str, fn: Callable[..., Any], *args: Any) -> Any:
    """Call ``fn(*args)`` and log the time it took."""
    LOG.info("%s ...", step)
    start = time.perf_counter()
    result = fn(*args)
    LOG.info("%s: %.1f s", step, time.perf_counter() - start)
    return result


def read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object."""
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write one JSON object, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def missing_tools(args: argparse.Namespace) -> list[str]:
    """The external tools that the requested steps need and PATH lacks."""
    skipped = {args.vivado: args.no_vivado, "verilator": args.no_cosim}
    needed = [tool for tool, skip in skipped.items() if not (skip or args.report_only)]
    return [tool for tool in needed if shutil.which(tool) is None]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line, refusing a run whose inputs or tools are missing."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=_Formatter)
    for flag, options in OPTIONS:
        parser.add_argument(flag, **options)
    args = parser.parse_args(argv)
    if not args.report_only and None in (args.export_dir, args.outputs_npz):
        parser.error("--export-dir and --outputs-npz are needed without --report-only")
    if missing := missing_tools(args):
        parser.error(f"not on PATH: {', '.join(missing)}; see --no-cosim, --no-vivado")
    args.csv = args.csv or args.out / "synthesis.csv"
    return args


def _float64_kept(name: str) -> Callable[[Any], Any]:
    """Keras's elementwise *name* on JAX, computed in float64 for a float64 tensor.

    A copy of ``_float64_kept`` in src/algorithms/components/vector_neurons_hgq.py,
    since this script imports nothing from src.
    """
    import jax
    import keras
    from keras.src import backend as keras_backend

    keras_fn, jax_fn = getattr(keras_backend.numpy, name), getattr(jax.numpy, name)

    def fn(x):
        x = keras.ops.convert_to_tensor(x)
        if keras.backend.standardize_dtype(x.dtype) == "float64":
            return jax_fn(x)
        return keras_fn(x)

    return fn


def _keep_float64_rounding() -> None:
    """Make Keras's floor and ceil keep a float64 tensor in float64.

    Keras 3.15 casts it to float32 there on JAX, so HGQ2's rounding floor(x 2^f + 1/2)
    misrounds a value within float32 resolution of a tie, and the Keras check would
    report logits that differ from alkaid's where the hardware agrees. The component
    src/algorithms/components/vector_neurons_hgq.py applies the same patch at import.
    """
    from keras.src import backend as keras_backend

    for name in ("floor", "ceil"):
        setattr(keras_backend.numpy, name, _float64_kept(name))


def _start_jax(floatx: str) -> None:
    """Start JAX on the CPU, unless ``JAX_PLATFORMS`` names another platform, and in x64
    under float64.

    :raises SystemExit: when JAX runs elsewhere than on the CPU: on a GPU, XLA's float64
        2**f is off by an ulp for some f, so that Keras would round some jets otherwise
        than alkaid.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", floatx == "float64")
    if (backend := jax.default_backend()) != "cpu":
        raise SystemExit(
            f"JAX runs on {backend!r}, where XLA's float64 2**f is inexact, so Keras "
            "would not be bit-exact against alkaid; set JAX_PLATFORMS=cpu."
        )


def import_keras(floatx: str) -> Any:
    """Import Keras on JAX at ``floatx``, with HGQ2's layers registered for loading.

    Under float64 the x64 mode is switched on before any array exists, since Keras would
    otherwise round the products of quantised values to float32, and floor and ceil are
    made to keep float64.
    """
    os.environ["KERAS_BACKEND"] = "jax"
    _start_jax(floatx)
    import hgq  # noqa: F401
    import keras

    keras.config.set_floatx(floatx)
    if keras.backend.backend() != "jax":
        raise ImportError(f"Keras runs on {keras.backend.backend()}, not on JAX")
    _keep_float64_rounding()
    return keras


def load_model(export_dir: Path, floatx: str) -> Any:
    """Load the exported ``model.keras`` without its training configuration."""
    keras = import_keras(floatx)
    return keras.models.load_model(export_dir / "model.keras", compile=False)


def load_reference(
    export_dir: Path, outputs_npz: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the quantised words (jets, N, 3) in the evaluation's jet order, the
    probe's clean logits (jets, classes) and the class labels (jets,)."""
    with np.load(export_dir / "inputs.npz") as inputs, np.load(outputs_npz) as outputs:
        words = inputs["inputs_clean"].astype(np.float64)
        logits = outputs["logits_clean"].astype(np.float64)
        labels = outputs["labels"].astype(np.int64)
    if not len(words) == len(logits) == len(labels):
        raise ValueError(f"{len(words)} exported jets against {len(logits)} probed")
    return words, logits, labels


def split_inputs(model: Any, words: np.ndarray) -> list[np.ndarray]:
    """Split the words (jets, N, 3) along the feature axis into the model's inputs.

    An input of shape (N,) takes one feature and one of shape (N, k) takes k, in the
    order of ``model.inputs``, which is also the order in which alkaid flattens them.
    """
    parts, start = [], 0
    for shape in (tuple(tensor.shape[1:]) for tensor in model.inputs):
        width = shape[-1] if len(shape) == 2 else 1
        parts.append(words[..., start : start + width].reshape(len(words), *shape))
        start += width
    if start != words.shape[-1]:
        raise ValueError(f"model reads {start} features, export {words.shape[-1]}")
    return parts


def flatten(parts: list[np.ndarray]) -> np.ndarray:
    """Concatenate the inputs per jet, the layout of the RTL's single input port."""
    return np.concatenate([part.reshape(len(part), -1) for part in parts], axis=1)


def keras_predict(model: Any, parts: list[np.ndarray]) -> np.ndarray:
    """Return the Keras logits (jets, classes) in float64."""
    feed = parts[0] if len(parts) == 1 else parts
    logits = model.predict(feed, batch_size=PREDICT_BATCH, verbose=0)
    return np.asarray(logits, dtype=np.float64)


def trace_comb(model: Any, hard_dc: int) -> Any:
    """Trace the model into alkaid's combinational logic, as ``alkaid convert`` does."""
    from alkaid.converter import trace_model
    from alkaid.trace import HWConfig, trace

    inputs, outputs = trace_model(model, HWConfig(1, 1, -1), {"hard_dc": hard_dc})
    return trace(inputs, outputs)


def macro_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Return the mean per-class recall of the argmax, torchmetrics' default
    ``MulticlassAccuracy``.

    As there, a class neither present nor predicted is left out of the mean, and a class
    predicted but absent enters it with recall 0.
    """
    classes = np.arange(logits.shape[1])[:, None]
    predicted, true = logits.argmax(axis=1) == classes, labels == classes
    hits, support = (predicted & true).sum(axis=1), true.sum(axis=1)
    recall = np.divide(hits, support, out=np.zeros(len(classes)), where=support > 0)
    return float(recall[(predicted | true).any(axis=1)].mean())


def emulation_stats(
    y_keras: np.ndarray, y_alkaid: np.ndarray, logits: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    """Compare both predictions (jets, classes) with the probe's logits and score
    them."""
    diff = np.abs(y_alkaid - logits)
    return {
        "n_test": len(labels),
        "n_mismatch": int(np.count_nonzero(diff)),
        "max_abs_diff": float(diff.max(initial=0.0)),
        "n_mismatch_keras": int(np.count_nonzero(y_keras != logits)),
        "acc_macro_keras": macro_accuracy(y_keras, labels),
        "acc_macro_alkaid": macro_accuracy(y_alkaid, labels),
        "acc_micro_alkaid": float(np.mean(y_alkaid.argmax(axis=1) == labels)),
    }


def run_record(args: argparse.Namespace, export: dict[str, Any]) -> dict[str, Any]:
    """The run's identity: the exported row, the target and the tool versions."""
    target = {"part": args.part, "clock_ns": args.clock}
    return {
        "model": export["row"],
        **{key: export[key] for key in ("nconstituents", "beta", "ebops")},
        **target,
        "latency_cutoff": args.latency_cutoff,
        "export_dir": str(args.export_dir.resolve()),
        "prj_dir": str((args.out / "prj").resolve()),
        **{f"{name}_version": version(dist) for name, dist in DISTRIBUTIONS},
    }


def record_emulation(
    args: argparse.Namespace, export: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any]:
    """Write and return ``check.json``; stop if alkaid strays from the probe's
    logits."""
    check = {**run_record(args, export), **stats}
    write_json(args.out / "check.json", check)
    level = logging.WARNING if stats["n_mismatch_keras"] else logging.INFO
    LOG.log(level, "Keras: %(n_mismatch_keras)d logits differ", stats)
    LOG.info("alkaid: %(n_mismatch)d logits differ, max %(max_abs_diff)g", stats)
    LOG.info("macro accuracy: Keras %(acc_macro_keras).6f", stats)
    LOG.info("macro accuracy: alkaid %(acc_macro_alkaid).6f", stats)
    if stats["n_mismatch"] > args.max_mismatch:
        raise SystemExit(f"{stats['n_mismatch']} mismatches exceed --max-mismatch")
    return check


def write_rtl(args: argparse.Namespace, comb: Any, meta: dict[str, Any]) -> Any:
    """Pipeline the logic and write the Verilog project ``<out>/prj/``."""
    from alkaid.codegen import RTLModel

    kwargs = {name: getattr(args, key) for key, name in RTL_OPTIONS.items()}
    rtl = RTLModel(comb, args.out / "prj", PROJECT, flavor="verilog", **kwargs)
    rtl.write(meta)
    return rtl


def initiation_interval(rtl: Any) -> int | None:
    """The period of the input schedule in clock cycles, None without a schedule."""
    schedule = rtl.fsm.inp_signals[0].schedule
    return None if schedule is None else int(schedule.period)


def cosimulate(
    rtl: Any, meta: dict[str, Any], flat: np.ndarray, y_alkaid: np.ndarray, threads: int
) -> dict[str, Any]:
    """Build the Verilator emulator of the project and compare it with alkaid's."""
    # compile() writes the project again, so the metadata has to come along.
    rtl.compile(nproc=threads, metadata=meta)
    y_rtl = np.asarray(rtl.predict(flat, n_threads=threads), dtype=np.float64)
    return {"n_cosim": len(flat), "cosim_exact": bool(np.array_equal(y_rtl, y_alkaid))}


def generate(
    args: argparse.Namespace,
    comb: Any,
    flat: np.ndarray,
    y_alkaid: np.ndarray,
    check: dict[str, Any],
) -> None:
    """Steps 6 and 7: write the RTL project and, unless ``--no-cosim``, co-simulate."""
    meta = {key: check[key] for key in ("model", "nconstituents", "beta", "ebops")}
    rtl = _timed("RTL generation", write_rtl, args, comb, meta)
    check.update(ii=initiation_interval(rtl), n_cosim=0, cosim_exact=None)
    if not args.no_cosim:
        jets = slice(args.cosim_jets)
        cosim = (rtl, meta, flat[jets], y_alkaid[jets], args.threads)
        check.update(_timed("Verilator co-simulation", cosimulate, *cosim))
    write_json(args.out / "check.json", check)
    if check["cosim_exact"] is False:
        raise SystemExit("the Verilator co-simulation differs from alkaid's emulation")


def write_predictions(out: Path, labels: np.ndarray, **logits: np.ndarray) -> None:
    """Write ``<out>/<name>_outputs.npz`` for each keyword, in the layout of the probe's
    file: ``logits_clean`` (jets, classes) in float32, exact behind the output
    quantiser, and ``labels`` (jets,) in int8."""
    out.mkdir(parents=True, exist_ok=True)
    for name, values in logits.items():
        arrays = {"logits_clean": values.astype(np.float32)}
        np.savez(out / f"{name}_outputs.npz", **arrays, labels=labels.astype(np.int8))


def predict(
    args: argparse.Namespace, model: Any, parts: list[np.ndarray], labels: np.ndarray
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Steps 3 to 5 up to the comparison: the Keras prediction, alkaid's trace and its
    emulation, both predictions written for ``compare_outputs.py``.

    :return: the traced logic and the Keras and alkaid logits (jets, classes).
    """
    y_keras = _timed("Keras prediction", keras_predict, model, parts)
    comb = _timed("alkaid trace", trace_comb, model, args.hard_dc)
    y_alkaid = _timed("alkaid emulation", comb.predict, parts, args.threads)
    write_predictions(args.out, labels, keras=y_keras, alkaid=y_alkaid)
    return comb, y_keras, y_alkaid


def build(args: argparse.Namespace) -> None:
    """Steps 1 to 7: load, predict, trace, emulate, write the project, co-simulate."""
    export = read_json(args.export_dir / "export.json")
    model = _timed("loading", load_model, args.export_dir, export["floatx"])
    words, logits, labels = load_reference(args.export_dir, args.outputs_npz)
    parts = split_inputs(model, words)
    comb, y_keras, y_alkaid = predict(args, model, parts, labels)
    stats = emulation_stats(y_keras, y_alkaid, logits, labels)
    check = record_emulation(args, export, stats)
    generate(args, comb, flatten(parts), y_alkaid, check)


def run_vivado(vivado: str, prj: Path) -> None:
    """Implement the project out of context; Vivado's own log is ``prj/vivado.log``."""
    subprocess.run(
        [vivado, *VIVADO_ARGS], cwd=prj, check=True, stdout=subprocess.DEVNULL
    )


def implement(args: argparse.Namespace) -> None:
    """Step 8, or the command that runs it elsewhere."""
    prj = args.out / "prj"
    if args.no_vivado:
        LOG.info(
            "no Vivado run; elsewhere: cd %s && vivado %s", prj, " ".join(VIVADO_ARGS)
        )
    else:
        _timed("Vivado implementation", run_vivado, args.vivado, prj)


def project_report(prj: Path) -> dict[str, Any]:
    """Read the project as ``alkaid report prj -o prj/report.json`` does, and write that
    file."""
    from alkaid._cli.report import load_project

    report = load_project(prj)
    if report is None:
        raise RuntimeError(f"alkaid cannot read the project {prj}")
    write_json(prj / "report.json", report)
    return report


def _number(report: dict[str, Any], key: str) -> float:
    """A numeric report entry, NaN when absent or unparsed."""
    value = report.get(key)
    return value if isinstance(value, int | float) else math.nan


def vivado_timing(report: dict[str, Any]) -> dict[str, float]:
    """The post-route WNS and the Fmax and latency it implies, derived as ``alkaid
    report`` derives them: the achieved period is the clock period minus the WNS."""
    wns_ns = _number(report, "WNS(ns)")
    period_ns = _number(report, "clock_period") - wns_ns
    latency_ns = _number(report, "latency") * period_ns
    return {"wns_ns": wns_ns, "fmax_mhz": 1000.0 / period_ns, "latency_ns": latency_ns}


def csv_row(check: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Merge ``check.json`` and alkaid's project report into one row of ``CSV_FIELDS``,
    NaN wherever a step has not run.

    The row holds no working point: ``compare_outputs.py`` reads those off
    ``keras_outputs.npz`` and ``alkaid_outputs.npz`` beside ``check.json``.
    """
    row = {column: _number(report, key) for column, key in REPORT_FIELDS.items()}
    row.update(vivado_timing(report), **check)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    row.update(vivado_done=report.keys() >= VIVADO_DONE, timestamp=now)
    return {key: math.nan if row.get(key) is None else row[key] for key in CSV_FIELDS}


def append_row(path: Path, row: dict[str, Any]) -> None:
    """Append the row, writing the header first into a new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_row(args: argparse.Namespace) -> None:
    """Step 9: the project report and ``check.json`` to one CSV row."""
    report = _timed("alkaid report", project_report, args.out / "prj")
    row = csv_row(read_json(args.out / "check.json"), report)
    append_row(args.csv, row)
    LOG.info("row appended to %s: %s", args.csv, json.dumps(row))


def main(argv: list[str] | None = None) -> int:
    """Run the steps the command line asks for; return the exit status."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not args.report_only:
        build(args)
        implement(args)
    write_row(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
