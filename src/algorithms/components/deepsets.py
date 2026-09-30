"""Permutation-invariant DeepSets classifiers for jets (arXiv:1703.06114).

``DeepSetsNet`` is the configurable DeepSets row. ``JEDILinearNet`` is JEDI-linear
(arXiv:2508.15468) in float: DeepSets with one permutation-equivariant context layer in
its per-constituent network.
"""

import torch
from torch import nn

AGGREGATIONS = ("sum", "mean", "max")

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "sigmoid": nn.Sigmoid,
}


class DeepSetsNet(nn.Module):
    """Apply phi to every constituent, pool over the jet, then read the pool with rho.

    :param phi_layers: widths of the per-constituent network, the last entry being the
        width of the pooled constituent embedding.
    :param rho_layers: hidden widths of the network applied to the pooled embedding.
    :param aggregation: pooling over the constituent axis, one of `AGGREGATIONS`.
    :param mask: zero the embeddings of padded constituents before pooling.
    """

    def __init__(
        self,
        in_dim: int,
        phi_layers: list[int],
        rho_layers: list[int],
        out_dim: int,
        activation: str = "relu",
        aggregation: str = "mean",
        dropout: float = 0.0,
        mask: bool = True,
    ) -> None:
        super().__init__()
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
            )
        self.aggregation = aggregation
        self.mask = mask
        self.activation = activation
        self.phi = self._stack([in_dim, *phi_layers], final_activation=True)
        self.rho = self._stack([phi_layers[-1], *rho_layers, out_dim], False, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents,
        features)."""
        h = self.phi(x)
        # A padded constituent is an all-zero row.
        real = (x.abs().sum(dim=-1, keepdim=True) > 0).to(h.dtype)
        if self.mask:
            h = h * real
        return self.rho(self._pool(h, real))

    def _pool(self, h: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        """Pool the embeddings of shape (jets, constituents, width) over the jet.

        :param real: array with 1 for a real constituent, 0 for a padded slot.
        """
        if self.aggregation == "sum":
            return h.sum(dim=1)
        if self.aggregation == "max":
            return h.max(dim=1).values
        return h.sum(dim=1) / real.sum(dim=1).clamp(min=1)

    def _activation(self) -> nn.Module:
        """A fresh instance of the configured activation."""
        factory = _ACTIVATIONS.get(self.activation.lower())
        if factory is None:
            raise ValueError(
                f"Unsupported activation {self.activation!r}, choose from {list(_ACTIVATIONS)}."
            )
        return factory()

    def _stack(
        self, dims: list[int], final_activation: bool, dropout: float = 0.0
    ) -> nn.Sequential:
        """Stack ``len(dims) - 1`` linear layers, with dropout in front of each."""
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if final_activation or i < len(dims) - 2:
                layers.append(self._activation())
        return nn.Sequential(*layers)


def _check_choices(aggregation: str, activation: str) -> None:
    """Refuse a pooling or an activation this module does not provide, by name."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
        )
    if activation not in _ACTIVATIONS:
        raise ValueError(
            f"activation must be one of {list(_ACTIVATIONS)}, got {activation!r}."
        )


def _masked_pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`.
    """
    if aggregation == "max":
        # A tanh embedding can be negative, so a padded slot must lose every comparison
        # rather than enter the maximum as zero.
        return h.masked_fill(~real, float("-inf")).amax(dim=1)
    total = (h * real).sum(dim=1)
    return total if aggregation == "sum" else total / real.sum(dim=1).clamp(min=1)


def _dense(dims: list[int], activation: str, final_activation: bool) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers with the activation between them and, if
    asked, after the last."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if final_activation or i < len(dims) - 2:
            layers.append(_ACTIVATIONS[activation]())
    return nn.Sequential(*layers)


class JEDILinearNet(nn.Module):
    """DeepSets whose per-constituent network carries one permutation-equivariant layer
    (Zaheer et al., arXiv:1703.06114): JEDI-linear (arXiv:2508.15468) in float.

    With s the activation, the context layer maps the embedding h_i of constituent i to
    s(W1 h_i + b1) + s(W2 pool_j h_j + b2); a shared layer follows, and the head reads
    the pool of the result over the jet. Both pools use one aggregation over the real
    constituents, chosen among those the DeepSets row searches, so that the two rows
    differ in the context layer alone; padding changes nothing, and only the sum carries
    the multiplicity to the head. The published point is the sum with ReLU. The released
    code scales each sum by a constant, which the next dense layer absorbs, and sums the
    padded rows too. Its BatchNorm is left out: it folds into the dense layers at
    inference and served the quantiser in training.

    :param hidden: width of the projection, the two context branches and the shared
        layer.
    :param head_layers: hidden widths of the MLP that reads the pooled vector.
    :param activation: a key of ``_ACTIVATIONS``, after every layer but the last.
    :param aggregation: one of `AGGREGATIONS`, for the context and the readout alike.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        head_layers: list[int],
        out_dim: int,
        activation: str = "relu",
        aggregation: str = "sum",
    ) -> None:
        super().__init__()
        _check_choices(aggregation, activation)
        self.aggregation = aggregation
        self.embed = _dense([in_dim, hidden], activation, True)
        self.local = _dense([hidden, hidden], activation, True)
        self.context = _dense([hidden, hidden], activation, True)
        self.shared = _dense([hidden, hidden], activation, True)
        self.head = _dense([hidden, *head_layers, out_dim], activation, False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pooled(x))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        """Return the jet vector the head reads, (jets, hidden), from a batch of shape
        (jets, constituents, features)."""
        h = self.embed(x)
        # A padded slot is an all-zero row but embeds to the activation of the first
        # bias, so the mask, not the input, keeps it out of both pools.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        context = self.context(_masked_pool(h, real, self.aggregation))
        h = self.local(h) + context.unsqueeze(1)
        return _masked_pool(self.shared(h), real, self.aggregation)
