"""Permutation-invariant DeepSets classifier for jets (arXiv:1703.06114)."""

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
