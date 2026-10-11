"""Quantisation-aware port of JEDI-linear, with or without the jet moments, to HGQ2.

The port reproduces ``JEDILinearNet`` (jedi_linear.py) on the rows [p_T GeV, etarel,
phirel] sorted by decreasing p_T; the sort and the subtraction of the jet axis stay
outside the block and hence are not counted in latency or resources.

The port reads the record's rows [p_T GeV, etarel, phirel] sorted by p_T, the angles
measured outside the block from the axis of the kept constituents, as the published
network reads them; ``_prepare_inputs`` de-normalises, sorts and clamps them in torch.
"""

from math import log2
from typing import Any

import torch

from src.algorithms.components.geometry import _sort_by_pt
from src.algorithms.components.hgq_port import KerasNetwork, _metric, _take
from src.algorithms.components.moments import (
    MOMENT_FEATURES,
    MOMENT_SETS,
    MOMENT_SUPPORT,
    RANK_CUTS,
)
from src.algorithms.components.pooling import AGGREGATIONS, _check_choice
from src.algorithms.jedi_linear import JEDILinearNet
from src.algorithms.keras_base import CALIBRATION_BATCH, KerasQuantisedModule
from src.algorithms.utils.jedi_linear_port import (
    architecture,
    copy_float_weights,
    float_network,
    load_float_network,
)

# isort: split

# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.config import LayerConfigScope
from hgq.layers import QAdd, QGlobalMaxPooling1D
from keras import ops


def _support(cut: int | None) -> int:
    """How many of the five moments the cut leaves nonzero; they are the leading ones,
    since ``MOMENT_SUPPORT`` does not decrease."""
    return sum(cut is None or need <= cut for need in MOMENT_SUPPORT)


class JEDILinearHGQ(KerasQuantisedModule):
    """JEDI-linear trained quantisation-aware with HGQ2, for synthesis by alkaid.

    Its head reads the jet moments under ``moments`` uniform or rankcut; under none it
    is the control without them.
    """

    required_features = list(MOMENT_FEATURES)

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        aggregation: str = "sum",
        moments: str = "uniform",
        units: float = 16.0,
        quantisation: dict[str, Any] | None = None,
        beta: float = 1e-7,
        beta_schedule: list[list] | None = None,
        init_from: str | None = None,
        calibration_jets: int = 100_000,
        floatx: str = "float64",
        jax_platform: str = "cpu",
        jax_devices: int | list[int] = 1,
        warmup_fraction: float = 0.0,
    ) -> KerasNetwork:
        """Build the port; the arguments after ``units`` are described in
        configs/algorithm/jedi_linear_hgq.yaml, and the training ones are read from the
        hyperparameters when they are needed."""
        return self._keras_net(quantisation, floatx, jax_platform, jax_devices)

    def _pool(self, name: str, x, keepdims: bool = False):
        """Pool the (jets, N, hidden) *x*, its padded slots zero, over the slots: the
        maximum, or the sum under sum and sum_over_n."""
        if self.hparams.aggregation != "max":
            return self._sum(name, x, keepdims=keepdims)
        pool = QGlobalMaxPooling1D(
            keepdims=keepdims, iq_conf=self.q.datalane(), name=name
        )
        return pool(x)

    def _constituents(self, p, angles, keep):
        """The per-constituent network from the embedding to the second pool, (jets,
        hidden)."""
        q, width = self.q, self.hparams.hidden
        rows = ops.concatenate([ops.expand_dims(p, -1), angles], axis=-1)
        h = self._dense(width, "embed", activation="relu")(rows)
        h = self._keep_multiply("keep0", h, keep)
        pooled = self._pool("pool0", h, keepdims=True)
        context = self._dense(width, "context", activation="relu")(pooled)
        local = self._dense(width, "local", activation="relu")(h)
        g = QAdd(iq_confs=q.datalane(), name="context_add")([local, context])
        g = self._dense(width, "shared", activation="relu")(g)
        return self._pool("pool1", self._keep_multiply("keep1", g, keep))

    def _matrices(self, c: int, u, tail):
        """U = sum_i u_i, (jets, 4), and M = sum_i u_i (1, tail_i)^T, (jets, 4, 4), of the
        lifts u and tails of the slots of cut *c*."""
        big_u = self._sum(f"moments{c}_u", u)
        outer = self._einsum("bni,bnj->bij", f"moments{c}_m", u, tail)
        return big_u, ops.concatenate([ops.expand_dims(big_u, -1), outer], axis=-1)

    def _triangle(self, c: int, gm):
        """The triangle tr(GM GM GM), (jets,), of the (jets, 4, 4) GM of cut *c*."""
        gm2 = self._einsum("bij,bjl->bil", f"moments{c}_gm2", gm, gm)
        return self._einsum("bil,bli->b", f"moments{c}_triangle", gm2, gm)

    def _forms(self, c: int, big_u, m, kept: int) -> list:
        """The moments pair, chain, pair4 and, when the cut keeps all five, triangle,
        (jets,) each, as ``_weighted_moments`` computes them."""
        gu, gm = _metric(big_u, 1), _metric(m, 1)
        pair = self._einsum("bm,bm->b", f"moments{c}_pair", big_u, gu)
        mgu = self._einsum("bij,bj->bi", f"moments{c}_mgu", m, gu)
        chain = self._einsum("bi,bi->b", f"moments{c}_chain", gu, mgu)
        pair4 = self._einsum("bij,bji->b", f"moments{c}_pair4", gm, gm)
        forms = [pair, chain, pair4]
        if kept == len(MOMENT_SUPPORT):
            forms.append(self._triangle(c, gm))
        return forms

    def _cut_moments(self, c: int, cut: int | None, p, tail, u) -> list:
        """The moments that cut *c* leaves nonzero, (jets, 1) each: those of the leading
        *cut* slots, or of every slot under None."""
        kept = _support(cut)
        if kept == 1:
            # sum_p of the leading constituent alone: its p, with no layer.
            return [p[:, :1]]
        if cut is not None:
            u, tail = _take(u, 1, 0, cut), _take(tail, 1, 0, cut)
        big_u, m = self._matrices(c, u, tail)
        forms = self._forms(c, big_u, m, kept)
        return [big_u[:, :1], *(ops.expand_dims(form, -1) for form in forms)]

    def _moments(self, p, angles, cuts: tuple[int | None, ...]) -> list:
        """The moments the rank cuts leave nonzero, (jets, 1) each, in the order of the
        flattened ``JetMoments.support``; none without moments."""
        if not cuts:
            return []
        tail, u = self._lifts(p, angles)
        return [
            m
            for c, cut in enumerate(cuts)
            for m in self._cut_moments(c, cut, p, tail, u)
        ]

    def _head(self, columns: list):
        """The head's MLP on the pooled vector and the moments; no softmax."""
        q, n_class = self.q, self.hparams.output_dim
        x = ops.concatenate(columns, axis=-1) if len(columns) > 1 else columns[0]
        for j, width in enumerate(self.hparams.head_layers):
            x = self._dense(width, f"head{j}", activation="relu")(x)
        oq = q.fixed(q.output_kif)
        return self._dense(n_class, "head_last", enable_oq=True, oq_conf=oq)(x)

    def _check_options(self) -> None:
        if not self.q.fold_norm:
            raise ValueError(
                "The JEDI-linear port has no batch normalisation: the copy folds the "
                "moments' into the head's first layer. Set quantisation.fold_norm true."
            )
        _check_choice("aggregation", self.hparams.aggregation, AGGREGATIONS)
        _check_choice("moments", self.hparams.moments, MOMENT_SETS)
        if not log2(self.units).is_integer():
            raise ValueError(f"units must be a power of two, got {self.units}.")

    def _input_stage(self) -> tuple:
        """The input rows, their words, p and keep."""
        x = keras.Input((self.nconstituents, 3), name="constituents")
        pt, angles = self._words(x)
        p, keep = self._p_keep(pt)
        return x, pt, angles, p, keep

    def _build_port(self) -> tuple[keras.Model, keras.Model]:
        """Build the port and the model of its input words, wiring the stages in the
        order of ``JEDILinearNet.forward``.

        Call ``configure_jax`` first: the layers take Keras's floatx at construction.

        :return: the network, (jets, N, 3) rows to (jets, n_class) logits, and the model
            of its inputs, the same rows to the quantised words it reads, of their
            shape.
        :raises ValueError: unless ``fold_norm`` is true, on an unknown pooling or
            moment set, or when *units* is not a power of two.
        """
        self._check_options()
        moments = self.hparams.moments
        cuts = () if moments == "none" else RANK_CUTS[moments]
        with LayerConfigScope(beta0=0.0, enable_ebops=True):
            x, pt, angles, p, keep = self._input_stage()
            pooled = self._constituents(p, angles, keep)
            logits = self._head([pooled, *self._moments(p, angles, cuts)])
        words = self._word_rows(pt, angles, None)
        model = keras.Model(x, logits, name="jedi_linear_hgq")
        return model, keras.Model(x, words, name="words")

    def _float_twin(self) -> JEDILinearNet:
        """The float JEDI-linear of this architecture, as its float row is priced."""
        return float_network(architecture(self.hparams), self.hparams.output_dim)

    def _copy_float_network(self, model: keras.Model) -> None:
        """Copy the float JEDI-linear of ``init_from``, read on the record's scales."""
        net = load_float_network(self.hparams.init_from, like=self.hparams)
        copy_float_weights(model, net, self.hparams.units, self.denorm_scales)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3) [p_T GeV, etarel, phirel],
        sorted by decreasing p_T, each column clamped to the range of its word."""
        x = _sort_by_pt(super()._prepare_inputs(x))
        return self._clamp_words(x[..., :1], x[..., 1:])

    def _calibration_batch(self) -> int:
        """Fewer jets per trace batch at larger N."""
        # The trace holds (jets, N, hidden) float64 intermediates.
        return max(1024, CALIBRATION_BATCH * 16 // self.hparams.nconstituents)
