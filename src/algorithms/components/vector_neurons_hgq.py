"""Quantisation-aware port of the vector neurons to HGQ2 on Keras and JAX.

The port reproduces ``VectorNeuronsClassifier`` (components/vector_neurons.py) behind
HGQ2 quantisers, so that alkaid can replay it bit for bit. It reads the rows [p_T GeV,
eta, phi] sorted by decreasing p_T, the angles measured from the leading constituent and
zero on a padded row; the sort, the leader subtraction and the wrap of phi stay outside,
as integer pre-processing of the trigger words. The first two layers quantise p_T and
the angles to those words; every later product, sum and affine map reads its inputs
through a quantiser of its own.

Vectors are stored (jets, 4, channels), components first, so that every channel mixing
is a dense layer on the last axis. p_T is counted in units of 16 GeV, a power of two, so
that p is an exact shift of the word, and the multiplicity enters as the count n rather
than n / N; ``vector_neurons_port.copy_float_weights`` folds both differences into the
batch normalisations of a trained float network.

This module selects Keras's backend: it sets JAX before the import and raises if Keras
came up on another one, so the port and the subclass import it before anything else.
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
# JAX reserves most of a GPU's memory at its first allocation otherwise, beside torch.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# isort: split

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import log2
from pathlib import Path
from typing import Any

import jax
import keras
import numpy as np
import torch
from hgq.config import HardSigmoidConfig, LayerConfigScope, QuantizerConfig
from hgq.layers import (
    QAdd,
    QBatchNormalization,
    QDense,
    QEinsum,
    QEinsumDense,
    QMultiply,
    QSum,
    Quantizer,
)
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


# Keras 3.15 casts a float64 tensor to float32 in floor and ceil on JAX (its
# result_type(float64, float) is float32), so HGQ2's rounding floor(x 2^f + 1/2)
# misrounds a scaled value within float32 resolution of a tie, and the float64 port
# disagrees with alkaid's integer arithmetic (39 of 12288 test jets of the dry run).
# Both keep float64 here, before any quantiser is built or loaded.
for _name in ("floor", "ceil"):
    setattr(keras_backend.numpy, _name, _float64_kept(_name))

# The keep flag and the count: 0 or 1, one unsigned integer bit.
KEEP_KIF = (0, 1, 0)
# torch's default, so that the copied running statistics normalise as they did there.
BN_EPSILON = 1e-5
# Fraction bits of the gate in float_mode, where its quantiser only implements the clamp.
FLOAT_GATE_F = 24
# A GPU run keeps the CPU backend beside CUDA, the default: HGQ2's trace_minmax stages
# its data under keras.device("cpu:0").
_PLATFORMS = {"gpu": "cuda,cpu", "cuda": "cuda,cpu", "cpu": "cpu"}


@dataclass
class Quantisation:
    """The quantisers of the port.

    The kif triples (keep_negative, integer bits, fraction bits) fix the trigger words
    and the logits. ``data_bits`` and ``weight_bits`` are the widths that
    ``seed_datalane_bits`` and ``seed_weight_bits`` give the trainable quantisers before
    training. ``float_mode`` replaces every quantiser but the gate's by an identity, for
    the comparison with the float network.
    """

    pt_kif: tuple[int, int, int] = (0, 12, 2)
    angle_kif: tuple[int, int, int] = (1, 1, 8)
    output_kif: tuple[int, int, int] = (1, 6, 8)
    data_bits: int = 14
    weight_bits: int = 8
    gate_f: int = 8
    float_mode: bool = False

    def datalane(self) -> QuantizerConfig:
        """WRAP and RND per element, f trained and i traced."""
        if self.float_mode:
            return QuantizerConfig("dummy", "datalane")
        return QuantizerConfig("kif", "datalane")

    def weight(self) -> QuantizerConfig:
        """SAT_SYM per element, for kernels and batch-normalisation scales."""
        return self._kbi("weight")

    def bias(self) -> QuantizerConfig:
        """WRAP per element, for biases and batch-normalisation offsets."""
        return self._kbi("bias")

    def _kbi(self, place: str) -> QuantizerConfig:
        if self.float_mode:
            return QuantizerConfig("dummy", place)
        return QuantizerConfig("kbi", place, b0=self.weight_bits)

    def fixed(self, kif: Sequence[int], overflow_mode: str = "SAT") -> QuantizerConfig:
        """An untrainable quantiser of the given (k, i, f), rounding to nearest."""
        if self.float_mode:
            return QuantizerConfig("dummy", "datalane")
        k, i, f = kif
        kwargs = {"k0": k, "i0": i, "f0": f, "overflow_mode": overflow_mode}
        return QuantizerConfig("kif", "datalane", trainable=False, **kwargs)

    def gate(self) -> QuantizerConfig:
        """HGQ2's hard sigmoid q(z) / 4 + 1 / 2, q saturating on [-2, 2).

        On z = (2/3) z_torch it equals torch's relu6(z_torch + 3) / 6 up to the
        rounding of z, 2^-(f + 2) at most: a clamp, a shift and an add in hardware,
        where Keras's hard_sigmoid would become a lookup table.
        """
        f = FLOAT_GATE_F if self.float_mode else self.gate_f
        return HardSigmoidConfig(f0=f, trainable=False)


@dataclass(frozen=True)
class _Port:
    """The quantisers and widths every stage of the builder reads."""

    q: Quantisation
    nconstituents: int
    vector: int
    scalar: int
    invariants: int
    tensors: int
    stream: int
    dropout: float
    units: float


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


def _dense(q: Quantisation, units: int, name: str, activation=None, **kwargs):
    """A dense layer on the last axis, its input, kernel and bias each quantised."""
    return QDense(
        units,
        activation=activation,
        iq_conf=q.datalane(),
        kq_conf=q.weight(),
        bq_conf=q.bias(),
        name=name,
        **kwargs,
    )


def _einsum_dense(port: _Port, equation: str, shape: tuple[int, ...], name: str):
    """A bias-free linear map of *equation*, its input and kernel quantised."""
    q = port.q
    return QEinsumDense(
        equation, shape, iq_conf=q.datalane(), kq_conf=q.weight(), name=name
    )


def _einsum(port: _Port, equation: str, name: str, *inputs):
    """The einsum of two tensors, each read through a datalane quantiser."""
    return QEinsum(equation, iq_confs=port.q.datalane(), name=name)(list(inputs))


def _sum(port: _Port, name: str, x, keepdims: bool = False, iq_conf=None):
    """The sum over the slots, axis 1, of *x* read through *iq_conf* (a datalane)."""
    iq_conf = iq_conf or port.q.datalane()
    return QSum(iq_conf=iq_conf, axes=1, keepdims=keepdims, name=name)(x)


def _norm(q: Quantisation, name: str) -> QBatchNormalization:
    """A batch normalisation with torch's epsilon and momentum, 0.9 in Keras being 0.1
    in torch; alkaid replays it as one quantised scale and offset per channel."""
    return QBatchNormalization(
        momentum=0.9,
        epsilon=BN_EPSILON,
        iq_conf=q.datalane(),
        kq_conf=q.weight(),
        bq_conf=q.bias(),
        name=name,
    )


def _keep_multiply(port: _Port, name: str, x, keep):
    """Zero the padded slots of *x*, the 0/1 flag *keep* read through one fixed bit."""
    confs = [port.q.datalane(), port.q.fixed(KEEP_KIF)]
    return QMultiply(iq_confs=confs, name=name)([x, keep])


def _dropout(port: _Port, x, name: str):
    if port.dropout <= 0.0:
        return x
    return keras.layers.Dropout(port.dropout, name=name)(x)


def _words(port: _Port, x):
    """Quantise rows (jets, N, 3) to p_T words (jets, N) and angle words (jets, N, 2).

    The words wrap: alkaid traces nothing but WRAP on a model input, so the caller
    clamps the rows to the words' ranges.
    """
    q = port.q
    pt = Quantizer(q.fixed(q.pt_kif, "WRAP"), name="pt_word")(x[:, :, 0])
    angles = Quantizer(q.fixed(q.angle_kif, "WRAP"), name="angle_word")(x[:, :, 1:])
    return pt, angles


def _lift(port: _Port, pt, angles):
    """Return p (jets, N), keep (jets, N, 1), the count n (jets, 1), the unit lift's
    tail (eta, phi, eta^2 + phi^2) (jets, N, 3) and the lifts u (jets, N, 4)."""
    # A division rather than the product by 1 / units: Keras marks the product of a
    # tensor and a Python scalar sparse, which the float32 build fails to trace. alkaid
    # replays both as the same shift.
    p = pt / port.units
    # Exact: a real p_T is a multiple of the word's LSB, and a cast of a bool has no
    # alkaid handler.
    keep = ops.expand_dims(ops.minimum(pt * 2.0 ** port.q.pt_kif[2], 1.0), -1)
    n = _sum(port, "count", keep, iq_conf=port.q.fixed(KEEP_KIF))
    r2 = _einsum(port, "bnk,bnk->bn", "lift_r2", angles, angles)
    tail = ops.concatenate([angles, ops.expand_dims(r2, -1)], axis=-1)
    u_tail = _einsum(port, "bn,bnk->bnk", "lift_u", p, tail)
    return p, keep, n, tail, ops.concatenate([ops.expand_dims(p, -1), u_tail], axis=-1)


def _stream(port: _Port, r, keep):
    """JEDI-linear's per-rank network of ``Stream`` on the (jets, N, 2) scalars *r*."""
    q, width = port.q, port.stream
    e = _dense(q, width, "stream_embed", activation="relu")(_norm(q, "stream_norm")(r))
    e = _keep_multiply(port, "stream_keep0", e, keep)
    pooled = _sum(port, "stream_pool", e, keepdims=True)
    local = _dense(q, width, "stream_local")(e)
    context = _dense(q, width, "stream_context")(pooled)
    h = QAdd(iq_confs=q.datalane(), name="stream_add")([local, context])
    h = keras.layers.ReLU(name="stream_relu")(h)
    return _sum(port, "stream_sum", _keep_multiply(port, "stream_keep1", h, keep))


def _chain(port: _Port, u, gm0):
    """The two-edge path of ``TensorInvariants``, (jets, K)."""
    shape = (port.tensors, 4)
    left = _einsum_dense(port, "bnm,nk->bkm", shape, "tensors_left")(u)
    right = _einsum_dense(port, "bnm,nk->bkm", shape, "tensors_right")(u)
    left_gm = _einsum(port, "bki,bkij->bkj", "tensors_chain_left", left, gm0)
    return _einsum(port, "bkj,bkj->bk", "tensors_chain", left_gm, _metric(right, 2))


def _tensor_invariants(port: _Port, u, tail) -> list:
    """Chain, pair4 and triangle of ``TensorInvariants``, (jets, K) each."""
    k = port.tensors
    outer = _einsum(port, "bni,bnj->bnij", "tensors_t", u, tail)
    t = ops.concatenate([ops.expand_dims(u, -1), outer], axis=-1)
    mixed = _einsum_dense(port, "bnij,nw->bwij", (3 * k, 4, 4), "tensors_mix")(t)
    gm = _metric(mixed, 2)
    gm0, gm1, gm2 = gm[:, :k], gm[:, k : 2 * k], gm[:, 2 * k :]
    pair4 = _einsum(port, "bkij,bkji->bk", "tensors_pair4", gm0, gm1)
    gm01 = _einsum(port, "bkij,bkjl->bkil", "tensors_gm01", gm0, gm1)
    triangle = _einsum(port, "bkil,bkli->bk", "tensors_triangle", gm01, gm2)
    return [_chain(port, u, gm0), pair4, triangle]


def _first_scalars(port: _Port, p, keep, tail, u):
    """The first block's input scalars [p, f, sum p, B(U, U), stream, chain, pair4,
    triangle], where torch interleaves p and f."""
    jet = _sum(port, "jet_sum", u)
    g_jet = _metric(jet, 1)
    f = _einsum(port, "bnm,bm->bn", "rank_form", u, g_jet)
    r = ops.concatenate([ops.expand_dims(p, -1), ops.expand_dims(f, -1)], axis=-1)
    p_sum = _sum(port, "p_sum", p, keepdims=True)
    jet_form = ops.expand_dims(_einsum(port, "bm,bm->b", "jet_form", jet, g_jet), -1)
    extras = [_stream(port, r, keep), *_tensor_invariants(port, u, tail)]
    return ops.concatenate([p, f, p_sum, jet_form, *extras], axis=-1)


def _read(port: _Port, v, n, s, prefix: str):
    """Concatenate the scalars *s*, the forms and first components of the (jets, 4, C)
    vectors *v* and the count *n*, as ``FormInvariants`` and the blocks do."""
    left = _dense(port.q, port.invariants, f"{prefix}_left", use_bias=False)(v)
    right = _dense(port.q, port.invariants, f"{prefix}_right", use_bias=False)(v)
    forms = _einsum(port, "bmi,bmi->bi", f"{prefix}_forms", left, _metric(right, 1))
    return ops.concatenate([s, forms, v[:, 0, :], n], axis=-1)


def _scalar_update(port: _Port, v, s, n, b: int):
    """The batch normalisation and the scalar MLP of block *b*."""
    q, width = port.q, port.scalar
    x = _norm(q, f"block{b}_norm")(_read(port, v, n, s, f"block{b}"))
    x = _dropout(port, x, f"block{b}_dropout0")
    x = _dense(q, width, f"block{b}_mlp0", activation="relu")(x)
    return _dense(q, width, f"block{b}_mlp1")(_dropout(port, x, f"block{b}_dropout1"))


def _gate(port: _Port, s, v, b: int):
    """Scale the (jets, 4, C) vectors *v* per channel by the hard sigmoid of *s*."""
    z = _dense(port.q, port.vector, f"block{b}_gate")(s)
    g = Quantizer(port.q.gate(), name=f"block{b}_hard_sigmoid")(z)
    return _einsum(port, "bc,bmc->bmc", f"block{b}_gated", g, v)


def _first_block(port: _Port, u, s, n):
    """``EquivariantBlock`` 0: it mixes the N ranks of the (jets, N, 4) lifts *u* into
    (jets, 4, C) vectors and has no residual."""
    shape = (4, port.vector)
    mixed = _einsum_dense(port, "bnm,nc->bmc", shape, "block0_mix")(u)
    s_new = _scalar_update(port, mixed, s, n, 0)
    return _gate(port, s_new, mixed, 0), s_new


def _later_block(port: _Port, v, s, n, b: int):
    """``EquivariantBlock`` *b* >= 1, with a residual on both states."""
    q = port.q
    mixed = _dense(q, port.vector, f"block{b}_mix", use_bias=False)(v)
    update = _scalar_update(port, mixed, s, n, b)
    s_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_s")([update, s])
    gated = _gate(port, s_new, mixed, b)
    v_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_v")([gated, v])
    return v_new, s_new


def _head(port: _Port, v, s, n, head_layers: Sequence[int], n_class: int):
    """The readout forms, the head's batch normalisation and its MLP; no softmax."""
    q = port.q
    x = _norm(q, "head_norm")(_read(port, v, n, s, "readout"))
    for j, width in enumerate(head_layers):
        x = _dropout(port, x, f"head_dropout{j}")
        x = _dense(q, width, f"head{j}", activation="relu")(x)
    oq = q.fixed(q.output_kif)
    last = _dense(q, n_class, "head_last", enable_oq=True, oq_conf=oq)
    return last(_dropout(port, x, f"head_dropout{len(head_layers)}"))


def build_port(
    nconstituents: int,
    head_layers: Sequence[int],
    num_blocks: int = 1,
    vector_channels: int = 16,
    scalar_channels: int = 64,
    invariants: int = 16,
    tensor_invariants: int = 2,
    stream_channels: int = 16,
    dropout: float = 0.0,
    units: float = 16.0,
    n_class: int = 5,
    quantisation: Quantisation | None = None,
) -> tuple[keras.Model, keras.Model]:
    """Build the port and the model of its input words.

    Call ``configure_jax`` first: the layers take Keras's floatx at construction.

    :param units: GeV per unit of p_T, a power of two so that p is a shift of the word.
    :return: the network, (jets, N, 3) rows to (jets, n_class) logits, and the model of
        its inputs, the same rows to the quantised words [p_T, eta, phi] it reads.
    :raises ValueError: when *units* is not a power of two.
    """
    if not log2(units).is_integer():
        raise ValueError(f"units must be a power of two, got {units}.")
    q = quantisation or Quantisation()
    widths = (vector_channels, scalar_channels, invariants, tensor_invariants)
    port = _Port(q, nconstituents, *widths, stream_channels, dropout, units)
    return _build(port, num_blocks, head_layers, n_class)


def _build(port: _Port, num_blocks: int, head_layers: Sequence[int], n_class: int):
    """Wire the stages in the order of ``VectorNeuronsClassifier.forward``."""
    with LayerConfigScope(beta0=0.0, enable_ebops=True):
        x = keras.Input((port.nconstituents, 3), name="constituents")
        pt, angles = _words(port, x)
        p, keep, n, tail, u = _lift(port, pt, angles)
        v, s = _first_block(port, u, _first_scalars(port, p, keep, tail, u), n)
        for b in range(1, num_blocks):
            v, s = _later_block(port, v, s, n, b)
        logits = _head(port, v, s, n, head_layers, n_class)
    words = ops.concatenate([ops.expand_dims(pt, -1), angles], axis=-1)
    return keras.Model(x, logits, name="vn_hgq"), keras.Model(x, words, name="words")


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
        """Return the (jets, n_class) logits of prepared (jets, N, 3) rows, in Keras's
        float type, on the device of *x*."""
        rows = x.detach().cpu().numpy().astype(self.keras_model.inputs[0].dtype)
        logits = self.keras_model.predict_on_batch(rows)
        return torch.from_numpy(np.array(logits)).to(x.device)

    def quantised_inputs(self, x: np.ndarray) -> np.ndarray:
        """The (jets, N, 3) words [p_T, eta, phi] the network reads from rows *x*."""
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
    """Set JAX's precision and Keras's float type, and, if no backend has started yet,
    the platform and the devices.

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


def _weight_values(layer: keras.Layer) -> list[tuple[Quantizer, Any]]:
    """Pair each weight and bias quantiser of *layer* with the value it quantises."""
    if isinstance(layer, QBatchNormalization):
        return list(zip((layer.kq, layer.bq), layer._scaler_and_offset()))
    if not isinstance(layer, (QDense, QEinsumDense)):
        return []
    biases = [] if layer.bias is None else [(layer.bq, layer.bias)]
    return [(layer.kq, layer.kernel), *biases]


def seed_weight_bits(model: keras.Model, bits: int) -> None:
    """Give every trainable weight and bias quantiser *bits* bits and the integer bits
    its present value needs, so that no copied weight saturates or wraps."""
    for layer in model.layers:
        for quantiser, value in _weight_values(layer):
            inner = quantiser.quantizer
            if not (isinstance(inner, FixedPointQuantizerKBI) and inner.trainable):
                continue
            inner._b.assign(ops.full(inner._b.shape, bits, inner._b.dtype))
            inner._i.assign(inner.get_minimal_i(value))


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
    """Trace the integer bits of every WRAP quantiser on rows *inputs* afresh and return
    the EBOPs of the traced model."""
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
    the (jets, N, 3) words, exact in float32) and ``export.json`` (*meta*).

    :return: *directory*, created if missing.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / "model.keras")
    words = np.asarray(inputs, dtype=np.float32)
    np.savez(directory / "inputs.npz", inputs_clean=words)
    (directory / "export.json").write_text(json.dumps(meta, indent=2))
    return directory
