"""Copy a trained float vector-neuron network into its HGQ2 port.

The port (components/vector_neurons_hgq.py) differs from ``VectorNeuronsClassifier`` in
four places, which the copy absorbs so that in eval mode both give the same logits:

- p_T is counted in 16 GeV rather than in the float network's units, so every
  batch-normalised input of degree d in p is scaled by lambda^d, lambda the ratio of
  the units;
- the multiplicity enters as the count n rather than n / N, scaled by N;
- the first block reads [p_1..p_N, f_1..f_N] where torch's ``r.flatten(1)`` interleaves
  p_1, f_1, p_2, f_2, ...;
- HGQ2's hard sigmoid has the slope 1/4 where torch's has 1/6, so the gate's linear map
  is scaled by 2/3.

A batch normalisation that reads a channel scaled by s keeps its output if its running
mean becomes s * mean and its running variance s^2 (var + eps_torch) - eps_keras.
A port built under ``fold_norm`` has no batch normalisation: each, in eval mode the map
y = a x + t per channel with a = gamma / sqrt(var + eps) and t = beta - a mean on those
statistics, is folded into the dense layer z = W^T y + b that reads it, whose kernel
becomes diag(a) W and whose bias b + W^T t.
Vectors are stored components first in the port, so every channel mixing becomes the
transposed weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from src.algorithms.components.vector_neurons import (
    FormInvariants,
    Stream,
    TensorInvariants,
    VectorNeuronsClassifier,
)
from src.utils.checkpoints import MANIFEST_FILENAME, read_manifest

# For annotations alone, so that importing this module never selects Keras's backend:
# the component does, before its own first import of Keras.
if TYPE_CHECKING:
    import keras

# The constructor arguments of VectorNeuronsClassifier that a checkpoint's
# hyperparameters carry beside nconstituents and output_dim.
ARCHITECTURE_KEYS = (
    "head_layers",
    "num_blocks",
    "vector_channels",
    "scalar_channels",
    "invariants",
    "tensor_invariants",
    "stream_channels",
    "mlp_layers",
    "gate",
    "dropout",
    "units",
    "axis_offset",
)
# The keys that a checkpoint written before they existed lacks, at the values its
# network was built with.
LEGACY_DEFAULTS = {"mlp_layers": 2, "gate": True, "axis_offset": False}
# A segment of degree COUNT holds the multiplicity, scaled by N rather than by lambda^d.
COUNT = None
# A batch-normalised segment: its width and its degree in p.
Segment = tuple[int, int | None]


def architecture(hparams: Mapping[str, Any]) -> dict[str, Any]:
    """Return the arguments of ``VectorNeuronsClassifier`` that *hparams* hold, by
    keyword, with ``LEGACY_DEFAULTS`` for those it lacks."""
    kwargs = {key: hparams[key] for key in ARCHITECTURE_KEYS if key in hparams}
    if "head_layers" in kwargs:
        kwargs["head_layers"] = list(kwargs["head_layers"])
    return {**LEGACY_DEFAULTS, **kwargs}


def _checkpoint_path(init_from: Path) -> Path:
    """The main run's checkpoint of a run directory or a manifest, or *init_from*."""
    if init_from.suffix == ".ckpt":
        return init_from
    manifest = init_from / MANIFEST_FILENAME if init_from.is_dir() else init_from
    runs = [run for run in read_manifest(manifest).runs if run.fold_idx is None]
    if not runs:
        raise ValueError(f"{manifest} lists no main run (fold_idx null) to copy.")
    return Path(runs[0].checkpoint_path)


def _check_architecture(
    path: Path, theirs: dict[str, Any], ours: dict[str, Any]
) -> None:
    """Refuse the checkpoint *path* unless its architecture *theirs* is *ours*.

    The units may differ: the copy folds them into the batch normalisations.

    :raises ValueError: naming every argument that differs.
    """
    differ = [
        key
        for key in ARCHITECTURE_KEYS
        if key != "units" and theirs.get(key) != ours.get(key)
    ]
    if differ:
        pairs = ", ".join(f"{k} {theirs.get(k)} vs {ours.get(k)}" for k in differ)
        raise ValueError(f"{path} holds another architecture: {pairs}.")


def _net_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The network's entries of a Lightning state dict, without the ``net.`` prefix."""
    return {
        key.removeprefix("net."): value
        for key, value in state_dict.items()
        if key.startswith("net.")
    }


def load_float_network(
    init_from: str | Path, like: Mapping[str, Any]
) -> VectorNeuronsClassifier:
    """Rebuild in eval mode the trained float network that *init_from* names.

    :param init_from: a run directory (its main run), a manifest or a ``.ckpt``.
    :param like: the hyperparameters of the port it is copied into.
    :raises ValueError: when its architecture is not that of *like*, units aside.
    """
    path = _checkpoint_path(Path(init_from))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    theirs = architecture(hparams)
    _check_architecture(path, theirs, architecture(like))
    net = VectorNeuronsClassifier(
        hparams["nconstituents"], n_class=hparams["output_dim"], **theirs
    )
    net.load_state_dict(_net_state(checkpoint["state_dict"]))
    return net.eval()


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().double().numpy()


def _scales(segments: list[Segment], scale: float, nconstituents: int) -> np.ndarray:
    """Per channel, the factor by which the port's input exceeds torch's."""
    return np.concatenate(
        [
            np.full(width, nconstituents if degree is COUNT else scale**degree)
            for width, degree in segments
        ]
    )


def _read_segments(forms: FormInvariants) -> list[Segment]:
    """The forms, quadratic in the vectors, the first components, linear, and n."""
    invariants, channels = forms.left.weight.shape
    return [(invariants, 2), (channels, 1), (1, COUNT)]


def _first_segments(net: VectorNeuronsClassifier) -> list[Segment]:
    """The first block's input scalars, in the port's order."""
    n = net.nconstituents
    k = 0 if net.tensors is None else net.tensors.weight.shape[1]
    width = 0 if net.stream is None else net.stream.embed.out_features
    offset = 2 if net.axis_offset else 0
    # p and f of each rank, sum p, B(U, U), the stream, chain, pair4, triangle and the
    # leader's two angles from the full jet's axis.
    segments = [(n, 1), (n, 2), (1, 1), (1, 2), (width, 0), (k, 3), (k, 2), (k, 3)]
    return [*segments, (offset, 0)]


def _deinterleave(nconstituents: int, width: int) -> np.ndarray:
    """The torch channel of each port channel of the first block's normalisation."""
    pairs = 2 * nconstituents
    order = [np.arange(0, pairs, 2), np.arange(1, pairs, 2), np.arange(pairs, width)]
    return np.concatenate(order)


def _port_statistics(
    norm: nn.BatchNorm1d, scales: np.ndarray, rows: np.ndarray | None
) -> tuple[np.ndarray, ...]:
    """The weight, the bias, the running mean and the running variance plus epsilon of
    *norm* for the port's channels: permuted by *rows* and scaled by *scales*."""
    index = slice(None) if rows is None else rows
    stats = (norm.weight, norm.bias, norm.running_mean, norm.running_var)
    gamma, beta, mean, var = (_numpy(stat)[index] for stat in stats)
    return gamma, beta, scales * mean, scales**2 * (var + norm.eps)


def copy_batch_norm(
    layer: keras.Layer,
    norm: nn.BatchNorm1d,
    scales: np.ndarray,
    rows: np.ndarray | None = None,
) -> None:
    """Copy *norm* into the port's batch normalisation *layer*, *scales* folded into its
    running statistics and its channels permuted by *rows*."""
    gamma, beta, mean, var_eps = _port_statistics(norm, scales, rows)
    layer.bn_gamma.assign(gamma)
    layer.bn_beta.assign(beta)
    layer.moving_mean.assign(mean)
    layer.moving_variance.assign(var_eps - layer.epsilon)


def _fold_norm(
    dense: keras.Layer,
    norm: nn.BatchNorm1d,
    scales: np.ndarray,
    rows: np.ndarray | None = None,
) -> None:
    """Fold *norm*, on the statistics ``copy_batch_norm`` would give the port, into the
    copied *dense* that reads its output."""
    gamma, beta, mean, var_eps = _port_statistics(norm, scales, rows)
    a = gamma / np.sqrt(var_eps)
    # (in, out), the input channels in the port's order.
    kernel = np.asarray(dense.kernel)
    bias = np.asarray(dense.bias)
    dense.bias.assign(bias + (beta - a * mean) @ kernel)
    dense.kernel.assign(a[:, None] * kernel)


def _place_norm(
    model: keras.Model,
    names: tuple[str, str],
    norm: nn.BatchNorm1d,
    scales: np.ndarray,
    rows: np.ndarray | None = None,
    fold: bool = False,
) -> None:
    """Copy *norm* into the port's normalisation ``names[0]`` or, under *fold*, fold it
    into the dense ``names[1]`` that reads its output, which must hold its copy."""
    norm_name, dense_name = names
    if fold:
        _fold_norm(model.get_layer(dense_name), norm, scales, rows)
    else:
        copy_batch_norm(model.get_layer(norm_name), norm, scales, rows)


def _copy_linear(
    layer: keras.Layer,
    linear: nn.Linear,
    factor: float = 1.0,
    rows: np.ndarray | None = None,
) -> None:
    """Copy *linear* times *factor*, its kernel rows taken in the order *rows*."""
    index = slice(None) if rows is None else rows
    layer.kernel.assign(factor * _numpy(linear.weight).T[index])
    layer.bias.assign(factor * _numpy(linear.bias))


def _linears(stack: nn.Sequential) -> list[nn.Linear]:
    return [module for module in stack if isinstance(module, nn.Linear)]


def _copy_linears(
    model: keras.Model,
    linears: list[nn.Linear],
    names: list[str],
    rows: np.ndarray | None = None,
) -> None:
    """Copy *linears* in order into the dense layers *names*, the first kernel's rows in
    the order *rows*."""
    for j, (name, linear) in enumerate(zip(names, linears, strict=True)):
        _copy_linear(model.get_layer(name), linear, rows=rows if j == 0 else None)


def _copy_channels(layer: keras.Layer, weight: torch.Tensor) -> None:
    """A ``ChannelLinear`` (out, in) becomes the (in, out) kernel of the last axis."""
    layer.kernel.assign(_numpy(weight).T)


def _copy_forms(
    model: keras.Model, forms: FormInvariants | TensorInvariants, prefix: str
) -> None:
    """Copy the left and right channel mixings of forms or of tensor invariants."""
    _copy_channels(model.get_layer(f"{prefix}_left"), forms.left.weight)
    _copy_channels(model.get_layer(f"{prefix}_right"), forms.right.weight)


def _copy_stream(
    model: keras.Model, stream: Stream, scale: float, n: int, fold: bool
) -> None:
    for name in ("embed", "local", "context"):
        _copy_linear(model.get_layer(f"stream_{name}"), getattr(stream, name))
    # The stream normalises p, of degree 1, and B(u_n, U), of degree 2.
    scales = _scales([(1, 1), (1, 2)], scale, n)
    names = ("stream_norm", "stream_embed")
    _place_norm(model, names, stream.norm, scales, fold=fold)


def _copy_tensors(model: keras.Model, tensors: TensorInvariants) -> None:
    """Copy the left, right and stacked rank weights of the tensor invariants."""
    _copy_forms(model, tensors, "tensors")
    # (3, K, N) to (N, 3K): the port's channel w K + k holds C, D or E for w = 0, 1, 2.
    weight = _numpy(tensors.weight)
    kernel = weight.reshape(-1, weight.shape[-1]).T
    model.get_layer("tensors_mix").kernel.assign(kernel)


def _state_segments(norm: nn.BatchNorm1d, forms: FormInvariants) -> list[Segment]:
    """The input of a later block's or the head's normalisation: scalars of degree 0,
    then what *forms* reads."""
    return [(norm.num_features - forms.width - 1, 0), *_read_segments(forms)]


def _block_layout(
    net: VectorNeuronsClassifier, b: int
) -> tuple[list[Segment], np.ndarray | None]:
    """The segments of block *b*'s normalisation and, for the first block, the torch
    channel of each port channel."""
    block = net.blocks[b]
    if b > 0:
        return _state_segments(block.norm, block.invariants), None
    segments = [*_first_segments(net), *_read_segments(block.invariants)]
    return segments, _deinterleave(net.nconstituents, block.norm.num_features)


def _copy_block(
    model: keras.Model, net: VectorNeuronsClassifier, b: int, scale: float, fold: bool
) -> None:
    """Copy block *b*: its scalar MLP, normalisation, gate if any, mixing and forms."""
    block, (segments, rows) = net.blocks[b], _block_layout(net, b)
    mlp = _linears(block.mlp)
    _copy_linears(model, mlp, [f"block{b}_mlp{j}" for j in range(len(mlp))], rows)
    scales = _scales(segments, scale, net.nconstituents)
    names = (f"block{b}_norm", f"block{b}_mlp0")
    _place_norm(model, names, block.norm, scales, rows, fold)
    if block.gate is not None:
        _copy_linear(model.get_layer(f"block{b}_gate"), block.gate.linear, factor=2 / 3)
    _copy_channels(model.get_layer(f"block{b}_mix"), block.linear.weight)
    _copy_forms(model, block.invariants, f"block{b}")


def _copy_head(
    model: keras.Model, net: VectorNeuronsClassifier, scale: float, fold: bool
) -> None:
    """Copy the readout forms, the head's MLP and its normalisation."""
    _copy_forms(model, net.readout, "readout")
    head = _linears(net.head)
    names = [f"head{j}" for j in range(len(head) - 1)] + ["head_last"]
    _copy_linears(model, head, names)
    segments = _state_segments(net.head_norm, net.readout)
    scales = _scales(segments, scale, net.nconstituents)
    _place_norm(model, ("head_norm", names[0]), net.head_norm, scales, fold=fold)


def copy_float_weights(
    model: keras.Model,
    net: VectorNeuronsClassifier,
    units: float,
    fold_norm: bool = False,
) -> None:
    """Copy *net* into the port *model* built with the same architecture.

    Exact in eval mode: the port, in float mode, then reproduces the logits of *net*.

    :param units: GeV per unit of p_T of the port.
    :param fold_norm: the ``Quantisation.fold_norm`` *model* was built under; true folds
        every batch normalisation into the dense layer that reads it.
    """
    scale = net.units / units
    if net.stream is not None:
        _copy_stream(model, net.stream, scale, net.nconstituents, fold_norm)
    if net.tensors is not None:
        _copy_tensors(model, net.tensors)
    for b in range(len(net.blocks)):
        _copy_block(model, net, b, scale, fold_norm)
    _copy_head(model, net, scale, fold_norm)
