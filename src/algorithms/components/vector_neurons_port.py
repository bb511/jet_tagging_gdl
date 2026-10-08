"""Copy a trained float vector-neuron network into its HGQ2 port.

The port (vector_neurons_hgq.py) differs from ``VectorNeuronsClassifier`` in four places,
which the copy absorbs so that in eval mode both give the same logits:

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

from pathlib import Path

import keras
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
)
# The keys that a checkpoint written before they existed lacks, at the values its
# network was built with.
LEGACY_DEFAULTS = {"mlp_layers": 2, "gate": True}
# A segment of degree COUNT holds the multiplicity, scaled by N rather than by lambda^d.
COUNT = None
# A batch-normalised segment: its width and its degree in p.
Segment = tuple[int, int | None]


def _checkpoint_path(init_from: Path) -> Path:
    """The main run's checkpoint of a run directory or a manifest, or *init_from*."""
    if init_from.suffix == ".ckpt":
        return init_from
    manifest = init_from / MANIFEST_FILENAME if init_from.is_dir() else init_from
    runs = [run for run in read_manifest(manifest).runs if run.fold_idx is None]
    if not runs:
        raise ValueError(f"{manifest} lists no main run (fold_idx null) to copy.")
    return Path(runs[0].checkpoint_path)


def _net_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The network's entries of a Lightning state dict, without the ``net.`` prefix."""
    return {
        key.removeprefix("net."): value
        for key, value in state_dict.items()
        if key.startswith("net.")
    }


def load_float_network(init_from: str | Path) -> VectorNeuronsClassifier:
    """Rebuild a trained float network in eval mode.

    :param init_from: a run directory (its main run), a manifest or a ``.ckpt``.
    """
    path = _checkpoint_path(Path(init_from))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    kwargs = {key: hparams[key] for key in ARCHITECTURE_KEYS if key in hparams}
    kwargs["head_layers"] = list(kwargs["head_layers"])
    net = VectorNeuronsClassifier(
        hparams["nconstituents"], n_class=hparams["output_dim"], **kwargs
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
    # p and f of each rank, sum p, B(U, U), the stream, chain, pair4, triangle.
    return [(n, 1), (n, 2), (1, 1), (1, 2), (width, 0), (k, 3), (k, 2), (k, 3)]


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


def _copy_norm(
    layer: keras.Layer,
    norm: nn.BatchNorm1d,
    scales: np.ndarray,
    rows: np.ndarray | None = None,
) -> None:
    """Fold *scales* into the running statistics of *norm*, permuted by *rows*."""
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
    """Fold *norm*, on the statistics ``_copy_norm`` would give the port, into the
    copied *dense* that reads its output."""
    gamma, beta, mean, var_eps = _port_statistics(norm, scales, rows)
    a = gamma / np.sqrt(var_eps)
    # (in, out), the input channels in the port's order.
    kernel = keras.ops.convert_to_numpy(dense.kernel)
    bias = keras.ops.convert_to_numpy(dense.bias)
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
        _copy_norm(model.get_layer(norm_name), norm, scales, rows)


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


def _copy_channels(layer: keras.Layer, weight: torch.Tensor) -> None:
    """A ``ChannelLinear`` (out, in) becomes the (in, out) kernel of the last axis."""
    layer.kernel.assign(_numpy(weight).T)


def _copy_forms(model: keras.Model, forms: FormInvariants, prefix: str) -> None:
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
    _copy_channels(model.get_layer("tensors_left"), tensors.left.weight)
    _copy_channels(model.get_layer("tensors_right"), tensors.right.weight)
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


def _copy_mlp(
    model: keras.Model, mlp: nn.Sequential, b: int, rows: np.ndarray | None
) -> None:
    """Copy the scalar MLP of block *b*, the first kernel's rows in the order *rows*."""
    linears = [m for m in mlp if isinstance(m, nn.Linear)]
    for j, linear in enumerate(linears):
        first_rows = rows if j == 0 else None
        _copy_linear(model.get_layer(f"block{b}_mlp{j}"), linear, rows=first_rows)


def _copy_block(
    model: keras.Model, net: VectorNeuronsClassifier, b: int, scale: float, fold: bool
) -> None:
    """Copy block *b*: its scalar MLP, normalisation, gate if any, mixing and forms."""
    block, (segments, rows) = net.blocks[b], _block_layout(net, b)
    _copy_mlp(model, block.mlp, b, rows)
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
    linears = [m for m in net.head if isinstance(m, nn.Linear)]
    names = [f"head{j}" for j in range(len(linears) - 1)] + ["head_last"]
    for name, linear in zip(names, linears):
        _copy_linear(model.get_layer(name), linear)
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
