"""Quantisation-aware port of the vector neurons to HGQ2 on Keras and JAX.

The port reproduces ``VectorNeuronsClassifier`` (components/vector_neurons.py). The pT
sort, the leader constituent subtraction and the wrap of phi stay outside the model and
hence are not counted in latency or resources. The first two layers quantise p_T and the
angles, and under the axis offset a third the leader's eta and phi from the full jet's
axis. Every later product, sum and affine map reads its inputs through a quantiser of
its own. The pieces shared with the other quantised rows are in hgq_port.py.

Vectors are stored (jets, 4, channels) such that every channel mixing is a dense layer
on the last axis (more efficient). p_T is counted in units of 16 GeV so it can be
implemented as a bit shift. The multiplicity enters as the count n.
"""

# Before Keras: the shared module selects its JAX backend.
from src.algorithms.components.hgq_port import (
    KEEP_KIF,
    PortBase,
    Quantisation,
    _dense,
    _einsum,
    _einsum_dense,
    _keep_multiply,
    _lifts,
    _metric,
    _p_keep,
    _sum,
    _word_rows,
    _words,
)

# isort: split

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2

import keras
from hgq.config import LayerConfigScope
from hgq.layers import QAdd, QBatchNormalization, Quantizer
from keras import ops

# torch's default, keep batch normalisation the same.
BN_EPSILON = 1e-5


@dataclass(frozen=True)
class _Port(PortBase):
    """The widths and switches of the vector neurons beside the shared fields."""

    vector: int
    scalar: int
    invariants: int
    tensors: int
    stream: int
    mlp_layers: int
    gate: bool
    dropout: float


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


def _normalise(q: Quantisation, name: str, x):
    """*x* through the batch normalisation *name*, or *x* itself under ``fold_norm``,
    where the copy folds the normalisation into the dense layer that reads *x*."""
    return x if q.fold_norm else _norm(q, name)(x)


def _dropout(port: _Port, x, name: str):
    if port.dropout <= 0.0:
        return x
    return keras.layers.Dropout(port.dropout, name=name)(x)


def _offset_words(port: _Port, x):
    """Quantise the leader's eta and phi from the full jet's axis, columns 3:5 of row 0
    of rows (jets, N, 5), to angle words (jets, 2), which wrap as ``_words`` do."""
    q = port.q
    # In hardware each is the leader's word minus the jet axis's, phi wrapped, computed
    # outside the block as the leader subtraction is.
    return Quantizer(q.fixed(q.angle_kif, "WRAP"), name="offset_word")(x[:, 0, 3:])


def _stream(port: _Port, r, keep):
    """JEDI-linear's per-rank network of ``Stream`` on the (jets, N, 2) scalars *r*."""
    q, width = port.q, port.stream
    r = _normalise(q, "stream_norm", r)
    e = _dense(q, width, "stream_embed", activation="relu")(r)
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


def _first_scalars(port: _Port, p, keep, tail, u, offset):
    """The first block's input scalars [p, f, sum p, B(U, U), stream, chain, pair4,
    triangle, offset], where torch interleaves p and f; no stream and no tensor
    invariants at width 0, and no offset words when *offset* is None."""
    jet = _sum(port, "jet_sum", u)
    g_jet = _metric(jet, 1)
    f = _einsum(port, "bnm,bm->bn", "rank_form", u, g_jet)
    r = ops.concatenate([ops.expand_dims(p, -1), ops.expand_dims(f, -1)], axis=-1)
    p_sum = _sum(port, "p_sum", p, keepdims=True)
    jet_form = ops.expand_dims(_einsum(port, "bm,bm->b", "jet_form", jet, g_jet), -1)
    stream = [_stream(port, r, keep)] if port.stream > 0 else []
    tensors = _tensor_invariants(port, u, tail) if port.tensors > 0 else []
    axis = [] if offset is None else [offset]
    return ops.concatenate([p, f, p_sum, jet_form, *stream, *tensors, *axis], axis=-1)


def _read(port: _Port, v, n, s, prefix: str):
    """Concatenate the scalars *s*, the forms and first components of the (jets, 4, C)
    vectors *v* and the count *n*, as ``FormInvariants`` and the blocks do."""
    left = _dense(port.q, port.invariants, f"{prefix}_left", use_bias=False)(v)
    right = _dense(port.q, port.invariants, f"{prefix}_right", use_bias=False)(v)
    forms = _einsum(port, "bmi,bmi->bi", f"{prefix}_forms", left, _metric(right, 1))
    return ops.concatenate([s, forms, v[:, 0, :], n], axis=-1)


def _scalar_update(port: _Port, v, s, n, b: int):
    """The batch normalisation and the scalar MLP of block *b*, ReLU between its dense
    layers and none after the last."""
    q, width = port.q, port.scalar
    x = _normalise(q, f"block{b}_norm", _read(port, v, n, s, f"block{b}"))
    for j in range(port.mlp_layers):
        activation = "relu" if j < port.mlp_layers - 1 else None
        x = _dropout(port, x, f"block{b}_dropout{j}")
        x = _dense(q, width, f"block{b}_mlp{j}", activation=activation)(x)
    return x


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
    if not port.gate:
        return mixed, s_new
    return _gate(port, s_new, mixed, 0), s_new


def _later_block(port: _Port, v, s, n, b: int):
    """``EquivariantBlock`` *b* >= 1, with a residual on both states."""
    q = port.q
    mixed = _dense(q, port.vector, f"block{b}_mix", use_bias=False)(v)
    update = _scalar_update(port, mixed, s, n, b)
    s_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_s")([update, s])
    gated = _gate(port, s_new, mixed, b) if port.gate else mixed
    v_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_v")([gated, v])
    return v_new, s_new


def _head(port: _Port, v, s, n, head_layers: Sequence[int], n_class: int):
    """The readout forms, the head's batch normalisation and its MLP; no softmax."""
    q = port.q
    x = _normalise(q, "head_norm", _read(port, v, n, s, "readout"))
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
    mlp_layers: int = 2,
    gate: bool = True,
    dropout: float = 0.0,
    units: float = 16.0,
    axis_offset: bool = False,
    n_class: int = 5,
    quantisation: Quantisation | None = None,
) -> tuple[keras.Model, keras.Model]:
    """Build the port and the model of its input words, wiring the stages in the order
    of ``VectorNeuronsClassifier.forward``.

    Call ``configure_jax`` first: the layers take Keras's floatx at construction.

    :param units: GeV per unit of p_T, a power of two so that p is a shift of the word.
    :param axis_offset: read rows 5 wide and give the first block the leader's eta and
        phi from the full jet's axis, columns 3:5 of row 0, as its last two scalars.
    :return: the network, (jets, N, 3) rows (5 wide under the axis offset) to (jets,
        n_class) logits, and the model of its inputs, the same rows to the quantised
        words it reads, of their shape.
    :raises ValueError: when *units* is not a power of two.
    """
    if not log2(units).is_integer():
        raise ValueError(f"units must be a power of two, got {units}.")
    port = _Port(
        q=quantisation or Quantisation(),
        nconstituents=nconstituents,
        vector=vector_channels,
        scalar=scalar_channels,
        invariants=invariants,
        tensors=tensor_invariants,
        stream=stream_channels,
        mlp_layers=mlp_layers,
        gate=gate,
        dropout=dropout,
        units=units,
    )
    with LayerConfigScope(beta0=0.0, enable_ebops=True):
        x = keras.Input((nconstituents, 5 if axis_offset else 3), name="constituents")
        pt, angles = _words(port, x)
        offset = _offset_words(port, x) if axis_offset else None
        p, keep = _p_keep(port, pt)
        n = _sum(port, "count", keep, iq_conf=port.q.fixed(KEEP_KIF))
        tail, u = _lifts(port, p, angles)
        s = _first_scalars(port, p, keep, tail, u, offset)
        v, s = _first_block(port, u, s, n)
        for b in range(1, num_blocks):
            v, s = _later_block(port, v, s, n, b)
        logits = _head(port, v, s, n, head_layers, n_class)
    words = _word_rows(port, pt, angles, offset)
    return keras.Model(x, logits, name="vn_hgq"), keras.Model(x, words, name="words")
