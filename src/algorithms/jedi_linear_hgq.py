"""Quantisation-aware port of the published JEDI-linear to HGQ2, trained from scratch.

The network, its quantisers and its training are those of Table II of arXiv:2508.15468
(``get_gnn`` and ``get_model`` of github.com/calad0i/JEDI-linear at 48032802, with
``configs/sweep-n16-f3.yaml``): every layer a dense layer with batch normalisation
folded in, no mask, no output quantiser, the two pools scaled by powers of two, and
every datalane quantised per slot and feature. The one deviation is the floor
``WEIGHT_I_MIN`` on the integer bits of every weight and bias, which keeps alkaid's
emulation exact (the error found on 2026-10-10).

Under ``inputs: standardised`` the port reads, as theirs does, the rows standardised by
the mean and standard deviation of every jet and slot of the pooled training and
validation splits, padded rows included; the first layer's trainable input quantiser is
then the only input quantisation. Under ``inputs: words`` it reads the trigger's fixed
p_T and angle words, p_T shifted to units of ``units`` GeV, as the vector-neuron port
does. ``_prepare_inputs`` de-normalises the rows, sorts them by p_T and standardises or
clamps them in torch, outside the block.
"""

import math
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from typing import Any

import torch

from src.algorithms.components.geometry import _sort_by_pt
from src.algorithms.components.hgq_port import WEIGHT_I_MIN, KerasNetwork, word_range
from src.algorithms.components.pooling import _check_choice
from src.algorithms.jedi_linear import JEDILinearNet
from src.algorithms.keras_base import CALIBRATION_BATCH, CHUNK, KerasQuantisedModule
from src.utils.pylogger import RankedLogger

# isort: split

# After the component, which selects Keras's JAX backend before the first import.
import keras
from hgq.config import LayerConfigScope, QuantizerConfigScope
from hgq.constraints import Min, MinMax
from hgq.layers import QAdd, QEinsumDenseBatchnorm, QSum
from hgq.quantizer.internal import FixedPointQuantizerKBI, FixedPointQuantizerKIF
from hgq.regularizers import MonoL1
from keras import ops

log = RankedLogger(__name__)

# The columns the port reads: the angles from the jet's axis, or absolute.
COLUMNS = (["pt", "etarel", "phirel"], ["pt", "eta", "phi"])
INPUTS = ("standardised", "words")
# get_model at Table II: the initial bits of every weight (b0) and the initial fraction
# bits of every datalane (f0), and the L1 weight of every bit count.
INITIAL_BITS = 7
BITS_L1 = 1e-8
# The readout pool's scale, a constant of the published code at every N.
READOUT_SCALE = 1 / 16


def _quantiser_scopes(per_slot: bool) -> list:
    """HGQ2's defaults as ``get_model`` and then ``get_gnn`` set them, followed by the
    floor on the integer bits of every weight and bias.

    :param per_slot: one width per slot and feature of every datalane; False shares the
        widths across the slots, their ``gnn_uq1``.
    """
    l1 = MonoL1(BITS_L1)
    kbi = dict(default_q_type="kbi", b0=INITIAL_BITS, i0=0, fr=l1, ir=l1)
    kif = dict(default_q_type="kif", place="datalane", f0=INITIAL_BITS, fr=l1)
    return [
        QuantizerConfigScope(overflow_mode="wrap", **kbi),
        QuantizerConfigScope(overflow_mode="wrap", ic=MinMax(0, 12), **kif),
        LayerConfigScope(beta0=0),
        QuantizerConfigScope(place=("weight", "bias"), overflow_mode="SAT_SYM"),
        QuantizerConfigScope(
            place="datalane", heterogeneous_axis=None if per_slot else (-1,)
        ),
        QuantizerConfigScope("kbi", ("weight", "bias"), ic=Min(WEIGHT_I_MIN)),
    ]


def _dense(equation: str, shape, name: str, activation: str | None = "relu"):
    """A dense layer with its batch normalisation folded in, as every layer of
    ``get_gnn``; its quantisers take the scopes' defaults."""
    return QEinsumDenseBatchnorm(
        equation, shape, bias_axes="C", activation=activation, name=name
    )


def _column_moments(
    batches: Iterable[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean and standard deviation of each column over every jet and slot of the
    (jets, N, F) *batches*, in float64, as numpy's mean and std over axes (0, 1)."""
    total = squares = count = 0
    for rows in batches:
        rows = rows.double().flatten(0, 1)
        total, squares = total + rows.sum(0), squares + rows.square().sum(0)
        count += len(rows)
    mean = total / count
    return mean, (squares / count - mean.square()).sqrt()


def _check_columns(features: list[str], inputs: str, angle_kif) -> None:
    """Refuse columns other than those of ``COLUMNS``, and absolute angles that the
    angle words cannot hold.

    :raises ValueError: on other columns, or under ``words`` on [pt, eta, phi] with an
        angle word that does not span [-pi, pi].
    """
    if features not in COLUMNS:
        raise ValueError(
            f"jedi_linear_hgq reads {COLUMNS[0]} or {COLUMNS[1]}, but data.features is "
            f"{features}."
        )
    low, high = word_range(angle_kif)
    narrow = low > -math.pi or high < math.pi
    if inputs == "words" and features == COLUMNS[1] and narrow:
        raise ValueError(
            f"The angle word {list(angle_kif)} spans [{low}, {high}], which cannot hold "
            "an absolute phi: set algorithm.quantisation.angle_kif to [1,2,8]."
        )


def _round_mode(model: keras.Model) -> str:
    """The rounding of every fixed-point quantiser of *model*, '/'-joined if several."""
    quantisers = (FixedPointQuantizerKBI, FixedPointQuantizerKIF)
    layers = model._flatten_layers()
    return "/".join(sorted({q.round_mode for q in layers if isinstance(q, quantisers)}))


class JEDILinearHGQ(KerasQuantisedModule):
    """JEDI-linear as published, trained quantisation-aware with HGQ2 for synthesis by
    alkaid.

    It reads whichever of [pt, etarel, phirel] and [pt, eta, phi] ``data.features``
    names; the first is the default before ``setup``.
    """

    required_features = COLUMNS[0]

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        inputs: str = "standardised",
        per_slot: bool = True,
        units: float = 16.0,
        quantisation: dict[str, Any] | None = None,
        beta: float | None = None,
        beta_schedule: list[list] | None = None,
        floatx: str = "float64",
        jax_platform: str = "cpu",
        jax_devices: int | list[int] = 1,
        warmup_fraction: float = 0.0,
    ) -> KerasNetwork:
        """Build the port; the arguments after ``head_layers`` are described in
        configs/algorithm/jedi_linear_hgq.yaml, and the training ones are read from the
        hyperparameters when they are needed."""
        # In the checkpoint; set from the pooled training and validation splits at
        # setup("fit").
        width = self.hparams.nbr_feats
        self.register_buffer("input_mean", torch.zeros(width, dtype=torch.float64))
        self.register_buffer("input_std", torch.ones(width, dtype=torch.float64))
        return self._keras_net(quantisation, floatx, jax_platform, jax_devices)

    def setup(self, stage: str) -> None:
        """Take the columns ``data.features`` names, then set up as every quantised row.

        :raises ValueError: as ``_check_columns`` does.
        """
        datamodule = getattr(self._trainer, "datamodule", None)
        if datamodule is not None:
            features = list(datamodule.features)
            _check_columns(features, self.hparams.inputs, self.quantisation.angle_kif)
            self.required_features = features
        super().setup(stage)

    def _initialise(self, datamodule) -> None:
        """Set the standardisation under ``standardised``.

        Nothing is copied and no bit seeded: the weights start at ``b0`` bits, the
        datalanes at ``f0`` fraction bits, and HGQ2 traces their integer bits in
        training.
        """
        if self.hparams.inputs != "standardised":
            return
        mean, std = _column_moments(self._pooled_rows(datamodule))
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)
        names = self.required_features
        log.info(f"Standardising {names}: mean {mean.tolist()}, std {std.tolist()}")

    def _pooled_rows(self, datamodule) -> Iterator[torch.Tensor]:
        """The rows of the training and validation splits, padded rows included."""
        for split in ("train", "validation"):
            jets, _ = datamodule.load_split(split)
            for chunk in jets.split(CHUNK):
                yield self._rows(chunk[..., self.model_feature_indices])

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        """Return (jets, N, 3) rows [p_T GeV, angle, angle], the angles in radians,
        sorted by decreasing p_T."""
        return _sort_by_pt(super()._prepare_inputs(x))

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Return the rows the port reads, (jets, N, 3): standardised, or each column
        clamped to the range of its word."""
        x = self._rows(x)
        if self.hparams.inputs == "words":
            return self._clamp_words(x[..., :1], x[..., 1:])
        return (x - self.input_mean) / self.input_std

    def _check_options(self) -> None:
        _check_choice("inputs", self.hparams.inputs, INPUTS)
        if not math.log2(self.units).is_integer():
            raise ValueError(f"units must be a power of two, got {self.units}.")

    def _input_stage(self, x) -> tuple:
        """The first layer's input and the words it is read from, None under
        ``standardised``."""
        if self.hparams.inputs == "standardised":
            return x, None
        pt, angles = self._words(x)
        # A division rather than the product by 1 / units, as in _p_keep.
        p = ops.expand_dims(pt / self.units, -1)
        return ops.concatenate([p, angles], axis=-1), self._word_rows(pt, angles, None)

    def _constituents(self, h):
        """``get_gnn`` from the embedding *h*, (jets, N, hidden), to the readout pool,
        (jets, hidden)."""
        n, width = self.nconstituents, self.hparams.hidden
        local = _dense("bnc,cC->bnC", (n, width), "local")(h)
        scale = 2.0 ** -round(math.log2(n))
        pooled = QSum(axes=1, scale=scale, keepdims=True, name="pool0")(h)
        context = _dense("bnc,cC->bnC", (1, width), "context")(pooled)
        h = QAdd(name="context_add")([local, context])
        h = _dense("bnc,cC->bnC", (n, width), "shared")(h)
        return QSum(axes=1, scale=READOUT_SCALE, name="pool1")(h)

    def _head(self, x):
        """The head's MLP; the logits have no softmax and no output quantiser."""
        for j, width in enumerate(self.hparams.head_layers):
            x = _dense("bc,cC->bC", width, f"head{j}")(x)
        n_class = self.hparams.output_dim
        return _dense("bc,cC->bC", n_class, "head_last", activation=None)(x)

    def _build_port(self) -> tuple[keras.Model, keras.Model]:
        """Build the port, wired as ``get_gnn`` wires it, and the model of its inputs.

        Call ``configure_jax`` first: the layers take Keras's floatx at construction.

        :return: the network, (jets, N, 3) rows to (jets, n_class) logits, and the model
            of its inputs, the same rows to the words under ``words`` or, under
            ``standardised``, to the output of the first layer's input quantiser.
        :raises ValueError: on an unknown ``inputs``, or when *units* is not a power of
            two.
        """
        self._check_options()
        n = self.nconstituents
        x = keras.Input((n, self.hparams.nbr_feats), name="constituents")
        rows, words = self._input_stage(x)
        with ExitStack() as scopes:
            for scope in _quantiser_scopes(self.hparams.per_slot):
                scopes.enter_context(scope)
            embed = _dense("bnc,cC->bnC", (n, self.hparams.hidden), "embed")
            logits = self._head(self._constituents(embed(rows)))
        # Under standardised the stored inputs are the first quantiser's fixed-point
        # output, on which alkaid's trace of the same quantiser is the identity.
        words = embed.iq(x) if words is None else words
        model = keras.Model(x, logits, name="jedi_linear_hgq")
        return model, keras.Model(x, words, name="words")

    def _float_twin(self) -> JEDILinearNet:
        """The float JEDI-linear of this architecture, sum pooling and no moments."""
        hp = self.hparams
        return JEDILinearNet(
            hp.nbr_feats,
            hp.hidden,
            hp.head_layers,
            hp.output_dim,
            aggregation="sum",
            moments="none",
        )

    def _calibration_batch(self) -> int:
        """Fewer jets per trace batch at larger N."""
        # The trace holds (jets, N, hidden) float64 intermediates.
        return max(1024, CALIBRATION_BATCH * 16 // self.hparams.nconstituents)

    def _export_meta(self, n_test: int) -> dict[str, Any]:
        """The base's record, with the columns, the input mode, whether the datalane
        widths are per slot, the standardisation and the rounding of the port's
        quantisers."""
        meta, q = super()._export_meta(n_test), self.quantisation
        standardised = self.hparams.inputs == "standardised"
        words = {"pt_kif": list(q.pt_kif), "angle_kif": list(q.angle_kif)}
        moments = {"mean": self.input_mean.tolist(), "std": self.input_std.tolist()}
        return meta | {
            "features": list(self.required_features),
            "inputs": self.hparams.inputs,
            "per_slot": self.hparams.per_slot,
            "quantisation": None if standardised else words,
            "standardisation": moments if standardised else None,
            "round_mode": _round_mode(self.net.keras_model),
        }
