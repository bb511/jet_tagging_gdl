"""Multi-layer preceptron implementation."""

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param nodes: widths of the hidden layers, in order.
    :param batchnorm: normalise each hidden layer's output before its activation.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        out_dim: int,
        batchnorm: bool | None = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = nodes
        self.out_dim = out_dim
        self.activation = self._get_activation(activation)
        self.batchnorm = batchnorm
        self.net = self._construct_net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _get_activation(self, activation: str) -> nn.Module:
        """Return the single instance every layer shares; none of these activations holds state."""
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()

        raise ValueError(f"Unsupported activation: {activation}")

    def _construct_net(self) -> nn.Sequential:
        """Stack the hidden layers, then the output layer that reaches the caller unactivated."""
        layers: list[nn.Module] = []

        current_dim = self.in_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(
                nn.BatchNorm1d(hidden_dim) if self.batchnorm else nn.Identity()
            )
            layers.append(self.activation)
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.out_dim))

        return nn.Sequential(*layers)
