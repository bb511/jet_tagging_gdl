"""Quantisation-aware port of the vector neurons to HGQ2 on Keras and JAX.

The port reproduces ``VectorNeuronsClassifier`` (vector_neurons.py). The pT sort, the
leader constituent subtraction and the wrap of phi stay outside the model and hence are
not counted in latency or resources. The first two layers quantise p_T and the angles,
and under the axis offset a third the leader's eta and phi from the full jet's axis.
Every later product, sum and affine map reads its inputs through a quantiser of its own.
The pieces shared with the other quantised rows are in components/hgq_port.py and
keras_base.py.

Vectors are stored (jets, 4, channels) such that every channel mixing is a dense layer
on the last axis (more efficient). p_T is counted in units of 16 GeV so it can be
implemented as a bit shift. The multiplicity enters as the count n.

The port reads the rows [p_T GeV, eta, phi] that the trigger provides, sorted by p_T and
measured from the leading constituent, which ``_prepare_inputs`` computes in torch. The
training, the trace and the export are those of every quantised row (keras_base.py).
"""

from math import log2
from typing import Any

import torch

from src.algorithms.components.geometry import _relative_angles, _sort_by_pt
from src.algorithms.components.hgq_port import KEEP_KIF, KerasNetwork, _metric
from src.algorithms.keras_base import KerasQuantisedModule
from src.algorithms.utils.vector_neurons_port import (
    architecture,
    copy_float_weights,
    load_float_network,
)
from src.algorithms.vector_neurons import VectorNeurons, VectorNeuronsClassifier

# isort: split

# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.config import LayerConfigScope
from hgq.layers import QAdd, QBatchNormalization, Quantizer
from keras import ops

# torch's default, keep batch normalisation the same.
BN_EPSILON = 1e-5


class VectorNeuronsHGQ(KerasQuantisedModule, VectorNeurons):
    """The vector neurons trained quantisation-aware with HGQ2, for synthesis by alkaid.

    Their ``required_features`` are those of ``VectorNeurons``.
    """

    def build_net(
        self,
        head_layers: list[int],
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
        quantisation: dict[str, Any] | None = None,
        beta: float = 1e-7,
        beta_schedule: list[list] | None = None,
        init_from: str | None = None,
        calibration_jets: int = 100_000,
        floatx: str = "float64",
        jax_platform: str = "cpu",
        jax_devices: int | list[int] = 1,
    ) -> KerasNetwork:
        """Build the port; the arguments after ``units`` are described in
        configs/algorithm/vn_hgq.yaml, and the training ones are read from the
        hyperparameters when they are needed."""
        if axis_offset:
            # As in VectorNeurons.build_net: meaningful under full_axis=True alone.
            self.required_features = [*self.required_features, "etarel", "phirel"]
        return self._keras_net(quantisation, floatx, jax_platform, jax_devices)

    def _norm(self, name: str) -> QBatchNormalization:
        """A batch normalisation with torch's epsilon and momentum, 0.9 in Keras being
        0.1 in torch; alkaid replays it as one quantised scale and offset per
        channel."""
        return QBatchNormalization(
            momentum=0.9,
            epsilon=BN_EPSILON,
            iq_conf=self.q.datalane(),
            kq_conf=self.q.weight(),
            bq_conf=self.q.bias(),
            name=name,
        )

    def _normalise(self, name: str, x):
        """*x* through the batch normalisation *name*, or *x* itself under
        ``fold_norm``, where the copy folds the normalisation into the dense layer that
        reads *x*."""
        return x if self.q.fold_norm else self._norm(name)(x)

    def _dropout(self, x, name: str):
        if self.hparams.dropout <= 0.0:
            return x
        return keras.layers.Dropout(self.hparams.dropout, name=name)(x)

    def _offset_words(self, x):
        """Quantise the leader's eta and phi from the full jet's axis, columns 3:5 of
        row 0 of rows (jets, N, 5), to angle words (jets, 2), which wrap as ``_words``
        do and, trained, share one width."""
        q = self.q
        # In hardware each is the leader's word minus the jet axis's, phi wrapped,
        # computed outside the block as the leader subtraction is.
        return Quantizer(q.word(q.angle_kif, (0, 1)), name="offset_word")(x[:, 0, 3:])

    def _stream(self, r, keep):
        """JEDI-linear's per-rank network of ``Stream`` on the (jets, N, 2) scalars
        *r*."""
        q, width = self.q, self.hparams.stream_channels
        r = self._normalise("stream_norm", r)
        e = self._dense(width, "stream_embed", activation="relu")(r)
        e = self._keep_multiply("stream_keep0", e, keep)
        pooled = self._sum("stream_pool", e, keepdims=True)
        local = self._dense(width, "stream_local")(e)
        context = self._dense(width, "stream_context")(pooled)
        h = QAdd(iq_confs=q.datalane(), name="stream_add")([local, context])
        h = keras.layers.ReLU(name="stream_relu")(h)
        return self._sum("stream_sum", self._keep_multiply("stream_keep1", h, keep))

    def _chain(self, u, gm0):
        """The two-edge path of ``TensorInvariants``, (jets, K)."""
        shape = (self.hparams.tensor_invariants, 4)
        left = self._einsum_dense("bnm,nk->bkm", shape, "tensors_left")(u)
        right = self._einsum_dense("bnm,nk->bkm", shape, "tensors_right")(u)
        left_gm = self._einsum("bki,bkij->bkj", "tensors_chain_left", left, gm0)
        return self._einsum("bkj,bkj->bk", "tensors_chain", left_gm, _metric(right, 2))

    def _tensor_invariants(self, u, tail) -> list:
        """Chain, pair4 and triangle of ``TensorInvariants``, (jets, K) each."""
        k = self.hparams.tensor_invariants
        outer = self._einsum("bni,bnj->bnij", "tensors_t", u, tail)
        t = ops.concatenate([ops.expand_dims(u, -1), outer], axis=-1)
        mixed = self._einsum_dense("bnij,nw->bwij", (3 * k, 4, 4), "tensors_mix")(t)
        gm = _metric(mixed, 2)
        gm0, gm1, gm2 = gm[:, :k], gm[:, k : 2 * k], gm[:, 2 * k :]
        pair4 = self._einsum("bkij,bkji->bk", "tensors_pair4", gm0, gm1)
        gm01 = self._einsum("bkij,bkjl->bkil", "tensors_gm01", gm0, gm1)
        triangle = self._einsum("bkil,bkli->bk", "tensors_triangle", gm01, gm2)
        return [self._chain(u, gm0), pair4, triangle]

    def _optional_scalars(self, r, keep, tail, u, offset) -> list:
        """The stream, the tensor invariants and the offset words, each when on."""
        k = self.hparams.tensor_invariants
        stream = [self._stream(r, keep)] if self.hparams.stream_channels > 0 else []
        tensors = self._tensor_invariants(u, tail) if k > 0 else []
        axis = [] if offset is None else [offset]
        return [*stream, *tensors, *axis]

    def _first_scalars(self, p, keep, tail, u, offset):
        """The first block's input scalars [p, f, sum p, B(U, U), stream, chain, pair4,
        triangle, offset], where torch interleaves p and f; no stream and no tensor
        invariants at width 0, and no offset words when *offset* is None."""
        jet = self._sum("jet_sum", u)
        g_jet = _metric(jet, 1)
        f = self._einsum("bnm,bm->bn", "rank_form", u, g_jet)
        r = ops.concatenate([ops.expand_dims(p, -1), ops.expand_dims(f, -1)], axis=-1)
        p_sum = self._sum("p_sum", p, keepdims=True)
        jet_form = ops.expand_dims(self._einsum("bm,bm->b", "jet_form", jet, g_jet), -1)
        optional = self._optional_scalars(r, keep, tail, u, offset)
        return ops.concatenate([p, f, p_sum, jet_form, *optional], axis=-1)

    def _read(self, v, n, s, prefix: str):
        """Concatenate the scalars *s*, the forms and first components of the (jets, 4,
        C) vectors *v* and the count *n*, as ``FormInvariants`` and the blocks do."""
        invariants = self.hparams.invariants
        left = self._dense(invariants, f"{prefix}_left", use_bias=False)(v)
        right = self._dense(invariants, f"{prefix}_right", use_bias=False)(v)
        forms = self._einsum("bmi,bmi->bi", f"{prefix}_forms", left, _metric(right, 1))
        return ops.concatenate([s, forms, v[:, 0, :], n], axis=-1)

    def _scalar_update(self, v, s, n, b: int):
        """The batch normalisation and the scalar MLP of block *b*, ReLU between its
        dense layers and none after the last."""
        width = self.hparams.scalar_channels
        x = self._normalise(f"block{b}_norm", self._read(v, n, s, f"block{b}"))
        for j in range(self.hparams.mlp_layers):
            activation = "relu" if j < self.hparams.mlp_layers - 1 else None
            x = self._dropout(x, f"block{b}_dropout{j}")
            x = self._dense(width, f"block{b}_mlp{j}", activation=activation)(x)
        return x

    def _gate(self, s, v, b: int):
        """Scale the (jets, 4, C) vectors *v* per channel by the hard sigmoid of *s*."""
        z = self._dense(self.hparams.vector_channels, f"block{b}_gate")(s)
        g = Quantizer(self.q.gate(), name=f"block{b}_hard_sigmoid")(z)
        return self._einsum("bc,bmc->bmc", f"block{b}_gated", g, v)

    def _first_block(self, u, s, n):
        """``EquivariantBlock`` 0: it mixes the N ranks of the (jets, N, 4) lifts *u*
        into (jets, 4, C) vectors and has no residual."""
        shape = (4, self.hparams.vector_channels)
        mixed = self._einsum_dense("bnm,nc->bmc", shape, "block0_mix")(u)
        s_new = self._scalar_update(mixed, s, n, 0)
        if not self.hparams.gate:
            return mixed, s_new
        return self._gate(s_new, mixed, 0), s_new

    def _later_block(self, v, s, n, b: int):
        """``EquivariantBlock`` *b* >= 1, with a residual on both states."""
        q, width = self.q, self.hparams.vector_channels
        mixed = self._dense(width, f"block{b}_mix", use_bias=False)(v)
        update = self._scalar_update(mixed, s, n, b)
        s_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_s")([update, s])
        gated = self._gate(s_new, mixed, b) if self.hparams.gate else mixed
        v_new = QAdd(iq_confs=q.datalane(), name=f"block{b}_residual_v")([gated, v])
        return v_new, s_new

    def _equivariant_blocks(self, u, s, n) -> tuple:
        """The ``num_blocks`` blocks, from the lifts *u* and the first scalars *s*."""
        v, s = self._first_block(u, s, n)
        for b in range(1, self.hparams.num_blocks):
            v, s = self._later_block(v, s, n, b)
        return v, s

    def _head(self, v, s, n):
        """The readout forms, the head's batch normalisation and its MLP; no softmax."""
        q, n_class = self.q, self.hparams.output_dim
        head_layers = self.hparams.head_layers
        x = self._normalise("head_norm", self._read(v, n, s, "readout"))
        for j, width in enumerate(head_layers):
            x = self._dropout(x, f"head_dropout{j}")
            x = self._dense(width, f"head{j}", activation="relu")(x)
        oq = q.fixed(q.output_kif)
        last = self._dense(n_class, "head_last", enable_oq=True, oq_conf=oq)
        return last(self._dropout(x, f"head_dropout{len(head_layers)}"))

    def _check_options(self) -> None:
        if not log2(self.units).is_integer():
            raise ValueError(f"units must be a power of two, got {self.units}.")

    def _input_stage(self) -> tuple:
        """The input rows, their words, the offset words, p, keep, n, tail and u."""
        axis_offset = self.hparams.axis_offset
        x = keras.Input(
            (self.nconstituents, 5 if axis_offset else 3), name="constituents"
        )
        pt, angles = self._words(x)
        offset = self._offset_words(x) if axis_offset else None
        p, keep = self._p_keep(pt)
        n = self._sum("count", keep, iq_conf=self.q.fixed(KEEP_KIF))
        tail, u = self._lifts(p, angles)
        return x, pt, angles, offset, p, keep, n, tail, u

    def _build_port(self) -> tuple[keras.Model, keras.Model]:
        """Build the port and the model of its input words, wiring the stages in the
        order of ``VectorNeuronsClassifier.forward``.

        Call ``configure_jax`` first: the layers take Keras's floatx at construction.

        :return: the network, (jets, N, 3) rows (5 wide under the axis offset) to (jets,
            n_class) logits, and the model of its inputs, the same rows to the quantised
            words it reads, of their shape.
        :raises ValueError: when *units* is not a power of two.
        """
        self._check_options()
        with LayerConfigScope(beta0=0.0, enable_ebops=True):
            x, pt, angles, offset, p, keep, n, tail, u = self._input_stage()
            s = self._first_scalars(p, keep, tail, u, offset)
            v, s = self._equivariant_blocks(u, s, n)
            logits = self._head(v, s, n)
        words = self._word_rows(pt, angles, offset)
        model = keras.Model(x, logits, name="vn_hgq")
        return model, keras.Model(x, words, name="words")

    def _float_twin(self) -> VectorNeuronsClassifier:
        """The float vn of this architecture, as the ``vn`` row is priced."""
        n, n_class = self.hparams.nconstituents, self.hparams.output_dim
        return VectorNeuronsClassifier(n, n_class=n_class, **architecture(self.hparams))

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the float vn of ``init_from``, its units folded into the port's."""
        net = load_float_network(self.hparams.init_from, like=self.hparams)
        fold_norm = self.quantisation.fold_norm
        copy_float_weights(model, net, self.hparams.units, fold_norm=fold_norm)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3) [p_T GeV, eta, phi], and under
        the axis offset (jets, N, 5), with etarel and phirel behind them.

        They are sorted by decreasing p_T, eta and phi measured from the leading
        constituent and zero on a padded row, and each column is clamped to the range of
        its trigger word.
        """
        x = _sort_by_pt(super()._prepare_inputs(x))
        pt = x[..., :1]
        relative = torch.stack(_relative_angles(x), dim=-1) * (pt > 0)
        # x[..., 3:] is empty without the axis offset.
        angles = torch.cat([relative, x[..., 3:]], dim=-1)
        return self._clamp_words(pt, angles)
