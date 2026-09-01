from collections.abc import Callable

import torch
from torch import nn


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param nodes: widths of the hidden layers, in order.
    :param final_activation: whether an activation follows the output layer as well.
    :param init_weight: initialiser applied to every linear weight; ``None`` keeps the default.
    :param init_bias: initialiser applied to every linear bias.
    """

    def __init__(
        self,
        in_dim: int,
        nodes: list[int],
        out_dim: int,
        batchnorm: bool | None = False,
        affine: bool = True,
        activation: str = "relu",
        final_activation: bool = False,
        init_weight: Callable | None = None,
        init_bias: Callable | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = nodes
        self.out_dim = out_dim
        self.activation = self._get_activation(activation)
        self.final_activation = final_activation

        self.batchnorm = batchnorm
        self.affine = affine
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.net = self._construct_net()
        self._apply_weight_init()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _get_activation(self, activation: str) -> nn.Module:
        activation = activation.lower()
        if activation == "relu":
            return nn.ReLU()
        if activation == "gelu":
            return nn.GELU()
        if activation == "silu":
            return nn.SiLU()

        raise ValueError(f"Unsupported activation: {activation}")

    def _construct_net(self):
        """Build the neural network."""
        layers: list[nn.Module] = []

        current_dim = self.in_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(
                nn.BatchNorm1d(hidden_dim, affine=self.affine) if self.batchnorm else nn.Identity()
            )
            layers.append(self.activation)
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, self.out_dim))
        if self.final_activation:
            layers.append(self.activation)

        return nn.Sequential(*layers)

    def _init_weight_wrapper(self, layer: nn.Module):
        if isinstance(layer, nn.Linear):
            return self.init_weight(layer.weight)
        return None

    def _apply_weight_init(self):
        """Initialise the weights with the given weight initialisation method."""
        if self.init_weight is not None:
            self.net.apply(self._init_weight_wrapper)
        if self.init_bias is not None:
            self.net.apply(self._init_bias_wrapper)

    def _init_bias_wrapper(self, layer: nn.Module):
        """Initialise the biases according to the given bias initialisation method."""
        if isinstance(layer, nn.Linear) and layer.bias is not None:
            return self.init_bias(layer.bias)
        return None
