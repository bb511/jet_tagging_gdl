"""Components of porting a network to HGQ.

Every quantised model uses these: the quantisers, the Keras network behind a torch
interface, and the steps that seed, trace, price and export a port. Importing this
module selects Keras's JAX backend, so it is imported before Keras.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
# JAX reserves most of a GPU's memory at its first allocation otherwise, beside torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import keras
import numpy as np
import torch
from hgq.config import HardSigmoidConfig, QuantizerConfig
from hgq.constraints import Min
from hgq.layers import QBatchNormalization, QDense, QEinsumDense, Quantizer
from hgq.layers.core import QLayerBase
from hgq.quantizer.internal import (
    FixedPointQuantizerKBI,
    FixedPointQuantizerKIF,
    TrainableQuantizerBase,
)
from hgq.utils import trace_minmax
from jax._src.xla_bridge import backends_are_initialized
from keras import ops
from keras.src import backend as keras_backend
from torch import nn

if keras.backend.backend() != "jax":
    raise ImportError(
        f"Keras runs on the {keras.backend.backend()!r} backend, but the HGQ2 port "
        "needs 'jax'. Unset KERAS_BACKEND, or set it to jax, before anything imports "
        "keras."
    )


def _float64_kept(name: str):
    """Keras's elementwise *name* on JAX, computed in float64 for a float64 tensor."""
    keras_fn, jax_fn = getattr(keras_backend.numpy, name), getattr(jax.numpy, name)

    def fn(x):
        x = ops.convert_to_tensor(x)
        if keras.backend.standardize_dtype(x.dtype) == "float64":
            return jax_fn(x)
        return keras_fn(x)

    return fn


# Keras 3.15 casts a float64 tensor to float32 in floor and ceil on JAX.
for _name in ("floor", "ceil"):
    setattr(keras_backend.numpy, _name, _float64_kept(_name))

# The keep flag and the count: 0 or 1, one unsigned integer bit.
KEEP_KIF = (0, 1, 0)
# The least integer bits of a weight or bias.
WEIGHT_I_MIN = -12
# Fraction bits of the gate in float_mode, where its quantiser only implements the clamp.
FLOAT_GATE_F = 24
# A GPU run keeps the CPU backend beside CUDA, the default: HGQ2's trace_minmax stages
# its data under keras.device("cpu:0").
_PLATFORMS = {"gpu": "cuda,cpu", "cuda": "cuda,cpu", "cpu": "cpu"}


def word_range(kif: Sequence[int]) -> tuple[float, float]:
    """The least and the greatest value of a fixed-point object."""
    k, i, f = kif
    return -k * 2.0**i, 2.0**i - 2.0**-f


@dataclass
class Quantisation:
    """The quantisers of the port.

    The fields ending in ``_kif``, (keep_negative, integer bits, fraction bits), fix the
    trigger words and the logits; those ending in ``_bits`` are the widths that
    ``seed_datalane_bits`` and ``seed_weight_bits`` give to the trainable quantisers
    before training.

    :param float_mode: replaces every quantiser but the gate's by an identity, for the
        comparison with the float network.
    :param fold_norm: builds no batch normalisation, the copy of a float network folding
        each into the dense layer that reads it.
    :param round_mode: HGQ2's rounding of every trainable datalane: RND adds half an LSB
        before the floor, which alkaid emits as a constant adder, and TRN truncates.
    """

    pt_kif: tuple[int, int, int] = (0, 12, 2)
    angle_kif: tuple[int, int, int] = (1, 1, 8)
    output_kif: tuple[int, int, int] = (1, 6, 8)
    data_bits: int = 14
    weight_bits: int = 8
    gate_f: int = 8
    float_mode: bool = False
    fold_norm: bool = False
    round_mode: str = "RND"

    def datalane(self) -> QuantizerConfig:
        """WRAP per element, rounding by ``round_mode``, f trained and i traced."""
        if self.float_mode:
            return QuantizerConfig("dummy", "datalane")
        return QuantizerConfig("kif", "datalane", round_mode=self.round_mode)

    def weight(self) -> QuantizerConfig:
        """SAT_SYM per element, for kernels and batch-normalisation scales."""
        return self._kbi("weight")

    def bias(self) -> QuantizerConfig:
        """WRAP per element, for biases and batch-normalisation offsets."""
        return self._kbi("bias")

    def _kbi(self, place: str) -> QuantizerConfig:
        if self.float_mode:
            return QuantizerConfig("dummy", place)
        return QuantizerConfig("kbi", place, b0=self.weight_bits, ic=Min(WEIGHT_I_MIN))

    def fixed(self, kif: Sequence[int], overflow_mode: str = "SAT") -> QuantizerConfig:
        """An untrainable quantiser of the given (k, i, f), rounding to nearest."""
        if self.float_mode:
            return QuantizerConfig("dummy", "datalane")
        k, i, f = kif
        kwargs = {"k0": k, "i0": i, "f0": f, "overflow_mode": overflow_mode}
        return QuantizerConfig("kif", "datalane", trainable=False, **kwargs)

    def gate(self) -> QuantizerConfig:
        """HGQ2's hard sigmoid q(z) / 4 + 1 / 2, q saturating on [-2, 2)."""
        f = FLOAT_GATE_F if self.float_mode else self.gate_f
        return HardSigmoidConfig(f0=f, trainable=False)


def _take(x, axis: int, start: int, stop: int):
    """The slice [start, stop) of *x* along *axis*, a GetItem node."""
    index = [slice(None)] * len(x.shape)
    index[axis] = slice(start, stop)
    return x[tuple(index)]


def _metric(x, axis: int):
    """G x along *axis*, of length 4: (x_3, -2 x_1, -2 x_2, x_0).

    A permutation and a factor -2, exact in Keras and in alkaid.
    """
    parts = [_take(x, axis, 3, 4), _take(x, axis, 1, 3) * -2.0, _take(x, axis, 0, 1)]
    return ops.concatenate(parts, axis=axis)


class KerasNetwork(nn.Module):
    """The Keras port behind the interface of a torch network.

    It holds no torch parameter: Keras owns the weights, the optimiser and the step, and
    Lightning calls ``forward`` for validation, testing and the invariance probe as for
    any other row.
    """

    def __init__(self, keras_model: keras.Model, inputs_model: keras.Model) -> None:
        super().__init__()
        self.keras_model = keras_model
        self.inputs_model = inputs_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the (jets, n_class) logits of prepared (jets, N, F) rows, in Keras's
        float type, on the device of *x*."""
        rows = x.detach().cpu().numpy().astype(self.keras_model.inputs[0].dtype)
        logits = self.keras_model.predict_on_batch(rows)
        return torch.from_numpy(np.array(logits)).to(x.device)

    def quantised_inputs(self, x: np.ndarray) -> np.ndarray:
        """The (jets, N, F) words the network reads from rows *x*, as the port's builder
        describes them."""
        return np.array(self.inputs_model.predict_on_batch(x))

    def compile_keras(self, optimizer: keras.optimizers.Optimizer) -> None:
        """Compile with the cross entropy of the logits; the layers add beta * EBOPs.

        ``train_on_batch`` then reports ``loss`` (the total), ``ce`` and ``acc``.
        """
        loss = keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [
            keras.metrics.CategoricalCrossentropy(from_logits=True, name="ce"),
            keras.metrics.CategoricalAccuracy(name="acc"),
        ]
        self.keras_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_on_batch(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """One optimiser step on rows *x* and one-hot labels *y*."""
        logs = self.keras_model.train_on_batch(x, y, return_dict=True)
        return {key: float(value) for key, value in logs.items()}

    def learning_rate(self) -> float:
        """The learning rate of the next step."""
        return float(ops.convert_to_numpy(self.keras_model.optimizer.learning_rate))


def configure_jax(platform: str, devices: int | Sequence[int], floatx: str) -> None:
    """Set JAX's precision and Keras's float type, the platform, and the devices.

    :param platform: Lightning's accelerator, ``gpu`` (or ``cuda``) or ``cpu``.
    :param devices: Lightning's devices, a count or a list of CUDA indices.
    :param floatx: Keras's float type, float32 or float64; float64 also switches on
        JAX's x64.
    """
    jax.config.update("jax_enable_x64", floatx == "float64")
    # TF32 would round the quantised inputs of every product on a GPU.
    jax.config.update("jax_default_matmul_precision", "highest")
    keras.config.set_floatx(floatx)
    # The device list is read once, when the first backend starts.
    if backends_are_initialized():
        return
    ids = range(devices) if isinstance(devices, int) else devices
    jax.config.update("jax_platforms", _PLATFORMS.get(platform, platform))
    jax.config.update("jax_cuda_visible_devices", ",".join(str(i) for i in ids))


def require_cpu_jax() -> None:
    """Refuse to evaluate unless JAX runs on the CPU.

    :raises ValueError: on any other backend, where XLA's float64 2**f is off by an ulp
        for some f, so that the port would round some jets otherwise than alkaid.
    """
    backend = jax.default_backend()
    if backend != "cpu":
        raise ValueError(
            f"JAX runs on {backend!r}, where XLA's float64 2**f is inexact, so the "
            "exported port and its stored logits would not match alkaid bit for bit. "
            "Evaluate a quantised row with algorithm.jax_platform=cpu."
        )


def _weight_values(layer: keras.Layer) -> list[tuple[Quantizer, Any]]:
    """Pair each weight and bias quantiser of *layer* with the value it quantises."""
    if isinstance(layer, QBatchNormalization):
        return list(zip((layer.kq, layer.bq), layer._scaler_and_offset()))
    if not isinstance(layer, (QDense, QEinsumDense)):
        return []
    biases = [] if layer.bias is None else [(layer.bq, layer.bias)]
    return [(layer.kq, layer.kernel), *biases]


def seed_weight_bits(model: keras.Model, bits: int) -> None:
    """Give every trainable weight and bias quantiser *bits* bits.

    The integer bits receive its present value needs, at least ``WEIGHT_I_MIN``, such
    that no copied weight saturates or wraps.
    """
    for layer in model.layers:
        for quantiser, value in _weight_values(layer):
            inner = quantiser.quantizer
            if not (isinstance(inner, FixedPointQuantizerKBI) and inner.trainable):
                continue
            inner._b.assign(ops.full(inner._b.shape, bits, inner._b.dtype))
            inner._i.assign(ops.maximum(inner.get_minimal_i(value), WEIGHT_I_MIN))


def _wrap_lanes(model: keras.Model) -> list[FixedPointQuantizerKIF]:
    """The trainable datalane quantisers, whose integer bits are traced."""
    return [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, FixedPointQuantizerKIF)
        and layer.overflow_mode == "WRAP"
        and layer.trainable
    ]


def calibrate(model: keras.Model, inputs: np.ndarray, batch_size: int = 16384) -> int:
    """Trace the integer bits of every WRAP quantiser on rows *inputs* afresh.

    Return the EBOPs of the traced model.
    """
    return int(trace_minmax(model, inputs, batch_size=batch_size))


def seed_datalane_bits(
    model: keras.Model, inputs: np.ndarray, bits: int, batch_size: int = 16384
) -> int:
    """Give every trainable datalane i + f = *bits*, i traced on rows *inputs*.

    :return: the EBOPs of the last trace, within a bit per lane of the seeded model.
    """
    # Twice: the first trace reads each lane through upstream lanes still at HGQ2's
    # default f = 6, and a lane whose range the second trace widens would wrap.
    for _ in range(2):
        ebops = calibrate(model, inputs, batch_size)
        for lane in _wrap_lanes(model):
            lane._f.assign(bits - lane.i)
    return ebops


def set_beta(model: keras.Model, beta: float) -> None:
    """Set the EBOPs weight of every quantised layer, as HGQ2's BetaScheduler does."""
    for layer in model._flatten_layers():
        if getattr(layer, "_beta", None) is not None:
            layer._beta.assign(ops.convert_to_tensor(beta, dtype=layer._beta.dtype))


def total_ebops(model: keras.Model) -> int:
    """The EBOPs of the last training step or trace, summed over the layers."""
    layers = [
        layer
        for layer in model._flatten_layers()
        if isinstance(layer, QLayerBase) and layer.enable_ebops
    ]
    return sum(int(ops.convert_to_numpy(layer._ebops)) for layer in layers)


def quantiser_variables(model: keras.Model) -> list[keras.Variable]:
    """The variables of every quantiser, which weight decay must leave alone."""
    return [
        variable
        for layer in model._flatten_layers()
        if isinstance(layer, TrainableQuantizerBase)
        for variable in layer.variables
    ]


def export_port(
    directory: str | Path,
    model: keras.Model,
    inputs: np.ndarray,
    meta: dict[str, Any],
) -> Path:
    """Write what synthesis reads: ``model.keras``, ``inputs.npz`` (``inputs_clean``,
    the (jets, N, F) words, exact in float32) and ``export.json`` (*meta*).

    :return: *directory*, created if missing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.keras")
    words = np.asarray(inputs, dtype=np.float32)
    np.savez(directory / "inputs.npz", inputs_clean=words)
    (directory / "export.json").write_text(json.dumps(meta, indent=2))
    return directory
