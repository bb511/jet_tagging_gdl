"""Permutation-invariant DeepSets classifier for jets (arXiv:1703.06114).

A jet is a set of constituents, so the architecture is phi applied to every constituent, a pooling
that discards the ordering, and rho on the pooled summary. Nothing here is invariant under a
rotation or a boost, which is what makes it a fair testbed for the augmentation and the SCL.
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


def _activation(name: str) -> nn.Module:
    factory = _ACTIVATIONS.get(name.lower())
    if factory is None:
        raise ValueError(f"Unsupported activation {name!r}, choose from {list(_ACTIVATIONS)}.")
    return factory()


def _stack(
    dims: list[int], activation: str, final_activation: bool, dropout: float = 0.0
) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers, with dropout in front of each (0 omits it)."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if final_activation or i < len(dims) - 2:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


class DeepSetsNet(nn.Module):
    """Apply phi to every constituent, pool over the jet, then read the pool with rho.

    :param phi_layers: widths of the per-constituent network, the last entry being the width of the
        pooled constituent embedding.
    :param rho_layers: hidden widths of the network applied to the pooled embedding.
    :param aggregation: pooling over the constituent axis, one of :data:`AGGREGATIONS`.
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
        self.phi = _stack([in_dim, *phi_layers], activation, final_activation=True)
        self.rho = _stack([phi_layers[-1], *rho_layers, out_dim], activation, False, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents, features)."""
        h = self.phi(x)
        if self.mask:
            # A padded constituent is an all-zero row and phi has biases, so phi(0) is not zero:
            # without this the empty slots would contribute to the pool.
            h = h * (x.abs().sum(dim=-1, keepdim=True) > 0).to(h.dtype)
        return self.rho(self._pool(h))

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        """Pool the embeddings of shape (jets, constituents, width) over the jet."""
        if self.aggregation == "sum":
            return h.sum(dim=1)
        if self.aggregation == "max":
            return h.max(dim=1).values
        # The mean divides by the slot count, not the jet's own multiplicity, as published.
        return h.mean(dim=1)
