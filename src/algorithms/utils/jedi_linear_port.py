"""Copy a trained float JEDI-linear, with or without the jet moments, into its HGQ2 port.

The port (jedi_linear_hgq.py) differs from ``JEDILinearNet`` in four places,
which the copy absorbs so that in eval mode both give the same logits:

- it reads (p_T / units, etarel, phirel) in GeV and radians, where the float network
  reads the record's columns divided by their scales s, so the embedding's kernel rows
  are scaled by (units / s_pt, 1 / s_eta, 1 / s_phi);
- under ``sum_over_n`` it pools by the plain sum, so the context's kernel and the
  pooled rows of the head's first kernel are divided by N;
- p is counted in ``units`` of GeV where the moments count 20 GeV, so a moment of
  degree d in p is scaled by lambda^d, lambda = 20 / units;
- it has no batch normalisation and builds only the moments that a rank cut leaves
  nonzero. The moments' normalisation, y = a x + t in eval mode, is folded into the
  head's first layer: the kernel rows of the built moments are scaled by a / lambda^d,
  and the bias gains the offsets t through the rows of every moment, the masked
  moments' being their constant output.

Kernels are stored (in, out) in Keras, so every weight is transposed.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn

from src.algorithms.components.moments import MOMENT_FEATURES, PT_UNIT_GEV, JetMoments
from src.algorithms.jedi_linear import JEDILinearNet
from src.algorithms.utils.vector_neurons_port import (
    _check_architecture,
    _checkpoint_path,
    _copy_linear,
    _linears,
    _net_state,
    _numpy,
    fold_statistics,
)

# For annotations alone, so that importing this module never selects Keras's backend:
# the component does, before its own first import of Keras.
if TYPE_CHECKING:
    import keras

# The hyperparameters of a float JEDI-linear that its port must share: N because the
# record's angle scales depend on it, and nbr_feats so that a four-momentum network is
# refused.
ARCHITECTURE_KEYS = (
    "nconstituents",
    "nbr_feats",
    "hidden",
    "head_layers",
    "aggregation",
    "moments",
)
# The plain jedi_linear row has no moments key.
LEGACY_DEFAULTS = {"moments": "none"}
# The degree in p of sum_p, pair, chain, pair4 and triangle.
MOMENT_DEGREES = (1, 2, 3, 2, 3)
# The composed config of a run, beside its checkpoints/ directory.
RUN_CONFIG = Path(".hydra") / "config.yaml"


def architecture(hparams: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``ARCHITECTURE_KEYS`` that *hparams* hold, with ``LEGACY_DEFAULTS``
    for those it lacks."""
    kwargs = {key: hparams[key] for key in ARCHITECTURE_KEYS if key in hparams}
    if "head_layers" in kwargs:
        kwargs["head_layers"] = list(kwargs["head_layers"])
    return {**LEGACY_DEFAULTS, **kwargs}


def float_network(arch: Mapping[str, Any], output_dim: int) -> JEDILinearNet:
    """The float JEDI-linear of the architecture *arch*, as ``architecture`` returns
    it."""
    widths = (arch["nbr_feats"], arch["hidden"], arch["head_layers"])
    return JEDILinearNet(*widths, output_dim, arch["aggregation"], arch["moments"])


def _check_columns(path: Path) -> None:
    """Refuse the checkpoint *path* unless its run read [pt, etarel, phirel], when the
    run's composed config lies beside its checkpoints/ directory.

    nbr_feats 3 alone cannot tell those columns from [pt, eta, phi].

    :raises ValueError: naming the columns the run read.
    """
    config, expected = path.parent.parent / RUN_CONFIG, list(MOMENT_FEATURES)
    if not config.is_file():
        return
    features = list(OmegaConf.load(config).data.features)
    if features != expected:
        raise ValueError(
            f"{path} was trained on {features}, the port reads {expected}."
        )


def load_float_network(init_from: str | Path, like: Mapping[str, Any]) -> JEDILinearNet:
    """Rebuild in eval mode the trained float JEDI-linear that *init_from* names.

    :param init_from: a run directory (its main run), a manifest or a ``.ckpt``.
    :param like: the hyperparameters of the port it is copied into.
    :raises ValueError: when its architecture is not that of *like*, or when its run
        read other columns.
    """
    path = _checkpoint_path(Path(init_from))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    theirs = architecture(hparams)
    _check_architecture(path, theirs, architecture(like), ARCHITECTURE_KEYS)
    _check_columns(path)
    net = float_network(theirs, hparams["output_dim"])
    net.load_state_dict(_net_state(checkpoint["state_dict"]))
    return net.eval()


def _copy_scaled(layer: keras.Layer, linear: nn.Linear, rows: np.ndarray) -> None:
    """Copy *linear*, each row of its (in, out) kernel times the entry of *rows*, one
    per input or one for all, the bias unchanged."""
    layer.kernel.assign(np.reshape(rows, (-1, 1)) * _numpy(linear.weight).T)
    layer.bias.assign(_numpy(linear.bias))


def _moment_fold(moments: JetMoments, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """The scale and the offset of the moments' normalisation on the port's moments,
    each of degree d in p *scale*^d times the float network's."""
    degrees = np.tile(MOMENT_DEGREES, len(moments.cuts))
    return fold_statistics(moments.norm, scale**degrees)


def _first_head_layer(
    linear: nn.Linear, net: JEDILinearNet, scale: float, n_pool: int
) -> tuple[np.ndarray, np.ndarray]:
    """The port's (in, out) kernel and bias of the head's first layer, whose input is
    the pooled vector, then the built moments."""
    kernel, bias = _numpy(linear.weight).T, _numpy(linear.bias)
    hidden = net.embed[0].out_features
    pooled = kernel[:hidden] / n_pool
    if net.moments is None:
        return pooled, bias
    a, t = _moment_fold(net.moments, scale)
    rows, kept = kernel[hidden:], np.flatnonzero(_numpy(net.moments.support))
    return np.concatenate([pooled, a[kept, None] * rows[kept]]), bias + t @ rows


def _copy_head(
    model: keras.Model, net: JEDILinearNet, scale: float, n_pool: int
) -> None:
    """Copy the head's MLP, the pools' 1 / *n_pool* and the moments' normalisation
    folded into its first layer."""
    linears = _linears(net.head)
    names = [f"head{j}" for j in range(len(linears) - 1)] + ["head_last"]
    for name, linear in zip(names[1:], linears[1:], strict=True):
        _copy_linear(model.get_layer(name), linear)
    kernel, bias = _first_head_layer(linears[0], net, scale, n_pool)
    model.get_layer(names[0]).kernel.assign(kernel)
    model.get_layer(names[0]).bias.assign(bias)


def copy_float_weights(
    model: keras.Model, net: JEDILinearNet, units: float, record_scales: torch.Tensor
) -> None:
    """Copy *net* into the port *model* built with the same architecture.

    Exact in eval mode: the port, in float mode, then reproduces the logits of *net*.

    :param units: GeV per unit of p_T of the port.
    :param record_scales: the factors taking the record's stored pt, etarel and phirel
        to GeV and radians.
    """
    n_pool = model.inputs[0].shape[1] if net.aggregation == "sum_over_n" else 1
    embed_rows = np.array([units, 1.0, 1.0]) / _numpy(record_scales)
    _copy_scaled(model.get_layer("embed"), net.embed[0], embed_rows)
    _copy_scaled(model.get_layer("context"), net.context[0], np.array(1.0 / n_pool))
    for name in ("local", "shared"):
        _copy_linear(model.get_layer(name), getattr(net, name)[0])
    _copy_head(model, net, PT_UNIT_GEV / units, n_pool)
