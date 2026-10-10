"""Quantisation-aware port of JEDI-linear, with or without the jet moments, to HGQ2 on
Keras and JAX.

The port reproduces ``JEDILinearNet`` (components/deepsets.py) on the rows [p_T GeV,
etarel, phirel] sorted by decreasing p_T; the sort and the subtraction of the jet axis
stay outside the block and hence are not counted in latency or resources. The first two
layers quantise p_T and the angles, and every later product, sum and affine map reads
its inputs through a quantiser of its own.

A padded slot is zeroed by the keep flag before each pool. The per-constituent layers
end in a ReLU, so the maximum over the slots equals the float network's maximum over the
real constituents of any jet that has one. ``sum_over_n`` pools by the plain sum, the
copy dividing the layers that read the pools by N. The moments need no mask, a padded
row having p = 0 and hence u = 0. Only the moments a rank cut leaves nonzero are built,
and the port has no batch normalisation: the copy folds the moments' into the head's
first layer.
"""

# Before Keras: the shared module selects its JAX backend.
from src.algorithms.components.hgq_port import (
    PortBase,
    Quantisation,
    _dense,
    _einsum,
    _keep_multiply,
    _lifts,
    _metric,
    _p_keep,
    _sum,
    _take,
    _word_rows,
    _words,
)

# isort: split

from collections.abc import Sequence
from dataclasses import dataclass
from math import log2

import keras
from hgq.config import LayerConfigScope
from hgq.layers import QAdd, QGlobalMaxPooling1D
from keras import ops

from src.algorithms.components.deepsets import (
    AGGREGATIONS,
    MOMENT_SETS,
    MOMENT_SUPPORT,
    RANK_CUTS,
    _check_choice,
)


@dataclass(frozen=True)
class _Port(PortBase):
    """The width, the pooling and the rank cuts of JEDI-linear beside the shared fields.

    :param cuts: the rank cuts of the moment set, ``RANK_CUTS`` of it, empty without
        moments.
    """

    hidden: int
    aggregation: str
    cuts: tuple[int | None, ...]


def _make_port(
    nconstituents: int,
    hidden: int,
    aggregation: str,
    moments: str,
    units: float,
    quantisation: Quantisation | None,
) -> _Port:
    """Check the options and gather what the stages read."""
    _check_choice("aggregation", aggregation, AGGREGATIONS)
    _check_choice("moments", moments, MOMENT_SETS)
    if not log2(units).is_integer():
        raise ValueError(f"units must be a power of two, got {units}.")
    cuts = () if moments == "none" else RANK_CUTS[moments]
    q = quantisation or Quantisation()
    return _Port(q, nconstituents, units, hidden, aggregation, cuts)


def _pool(port: _Port, name: str, x, keepdims: bool = False):
    """Pool the (jets, N, hidden) *x*, its padded slots zero, over the slots: the
    maximum, or the sum under sum and sum_over_n."""
    if port.aggregation != "max":
        return _sum(port, name, x, keepdims=keepdims)
    pool = QGlobalMaxPooling1D(keepdims=keepdims, iq_conf=port.q.datalane(), name=name)
    return pool(x)


def _constituents(port: _Port, p, angles, keep):
    """The per-constituent network from the embedding to the second pool, (jets,
    hidden)."""
    q, width = port.q, port.hidden
    rows = ops.concatenate([ops.expand_dims(p, -1), angles], axis=-1)
    h = _dense(q, width, "embed", activation="relu")(rows)
    h = _keep_multiply(port, "keep0", h, keep)
    pooled = _pool(port, "pool0", h, keepdims=True)
    context = _dense(q, width, "context", activation="relu")(pooled)
    local = _dense(q, width, "local", activation="relu")(h)
    g = QAdd(iq_confs=q.datalane(), name="context_add")([local, context])
    g = _dense(q, width, "shared", activation="relu")(g)
    return _pool(port, "pool1", _keep_multiply(port, "keep1", g, keep))


def _support(cut: int | None) -> int:
    """How many of the five moments the cut leaves nonzero; they are the leading ones,
    since ``MOMENT_SUPPORT`` does not decrease."""
    return sum(cut is None or need <= cut for need in MOMENT_SUPPORT)


def _matrices(port: _Port, c: int, u, tail):
    """U = sum_i u_i, (jets, 4), and M = sum_i u_i (1, tail_i)^T, (jets, 4, 4), of the
    lifts u and tails of the slots of cut *c*."""
    big_u = _sum(port, f"moments{c}_u", u)
    outer = _einsum(port, "bni,bnj->bij", f"moments{c}_m", u, tail)
    return big_u, ops.concatenate([ops.expand_dims(big_u, -1), outer], axis=-1)


def _triangle(port: _Port, c: int, gm):
    """The triangle tr(GM GM GM), (jets,), of the (jets, 4, 4) GM of cut *c*."""
    gm2 = _einsum(port, "bij,bjl->bil", f"moments{c}_gm2", gm, gm)
    return _einsum(port, "bil,bli->b", f"moments{c}_triangle", gm2, gm)


def _forms(port: _Port, c: int, big_u, m, kept: int) -> list:
    """The moments pair, chain, pair4 and, when the cut keeps all five, triangle,
    (jets,) each, as ``_weighted_moments`` computes them."""
    gu, gm = _metric(big_u, 1), _metric(m, 1)
    pair = _einsum(port, "bm,bm->b", f"moments{c}_pair", big_u, gu)
    mgu = _einsum(port, "bij,bj->bi", f"moments{c}_mgu", m, gu)
    chain = _einsum(port, "bi,bi->b", f"moments{c}_chain", gu, mgu)
    pair4 = _einsum(port, "bij,bji->b", f"moments{c}_pair4", gm, gm)
    forms = [pair, chain, pair4]
    if kept == len(MOMENT_SUPPORT):
        forms.append(_triangle(port, c, gm))
    return forms


def _cut_moments(port: _Port, c: int, cut: int | None, p, tail, u) -> list:
    """The moments that cut *c* leaves nonzero, (jets, 1) each: those of the leading
    *cut* slots, or of every slot under None."""
    kept = _support(cut)
    if kept == 1:
        # sum_p of the leading constituent alone: its p, with no layer.
        return [p[:, :1]]
    if cut is not None:
        u, tail = _take(u, 1, 0, cut), _take(tail, 1, 0, cut)
    big_u, m = _matrices(port, c, u, tail)
    forms = _forms(port, c, big_u, m, kept)
    return [big_u[:, :1], *(ops.expand_dims(form, -1) for form in forms)]


def _moments(port: _Port, p, angles) -> list:
    """The moments the rank cuts leave nonzero, (jets, 1) each, in the order of the
    flattened ``JetMoments.support``; none without moments."""
    if not port.cuts:
        return []
    tail, u = _lifts(port, p, angles)
    cuts = enumerate(port.cuts)
    return [m for c, cut in cuts for m in _cut_moments(port, c, cut, p, tail, u)]


def _head(port: _Port, columns: list, head_layers: Sequence[int], n_class: int):
    """The head's MLP on the pooled vector and the moments; no softmax."""
    q = port.q
    x = ops.concatenate(columns, axis=-1) if len(columns) > 1 else columns[0]
    for j, width in enumerate(head_layers):
        x = _dense(q, width, f"head{j}", activation="relu")(x)
    oq = q.fixed(q.output_kif)
    return _dense(q, n_class, "head_last", enable_oq=True, oq_conf=oq)(x)


def build_port(
    nconstituents: int,
    hidden: int,
    head_layers: Sequence[int],
    aggregation: str = "sum",
    moments: str = "none",
    units: float = 16.0,
    n_class: int = 5,
    quantisation: Quantisation | None = None,
) -> tuple[keras.Model, keras.Model]:
    """Build the port and the model of its input words, wiring the stages in the order
    of ``JEDILinearNet.forward``.

    Call ``configure_jax`` first: the layers take Keras's floatx at construction.

    :param aggregation: one of ``AGGREGATIONS``, for both pools.
    :param moments: one of ``MOMENT_SETS``; under none the head reads the pooled vector
        alone.
    :param units: GeV per unit of p_T, a power of two so that p is a shift of the word.
    :return: the network, (jets, N, 3) rows to (jets, n_class) logits, and the model of
        its inputs, the same rows to the quantised words it reads, of their shape.
    :raises ValueError: on an unknown pooling or moment set, or when *units* is not a
        power of two.
    """
    port = _make_port(nconstituents, hidden, aggregation, moments, units, quantisation)
    with LayerConfigScope(beta0=0.0, enable_ebops=True):
        x = keras.Input((nconstituents, 3), name="constituents")
        pt, angles = _words(port, x)
        p, keep = _p_keep(port, pt)
        columns = [_constituents(port, p, angles, keep), *_moments(port, p, angles)]
        logits = _head(port, columns, head_layers, n_class)
    words = _word_rows(port, pt, angles, None)
    model = keras.Model(x, logits, name="jedi_linear_hgq")
    return model, keras.Model(x, words, name="words")
