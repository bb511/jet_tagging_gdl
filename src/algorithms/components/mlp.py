"""Dense stacks: the MLP, the Linformer head and the DeepSets networks."""

import torch
from torch import nn


def dense_stack(
    in_dim: int, widths: list[int], out_dim: int, batchnorm: bool | None
) -> nn.Sequential:
    """Stack Linear, BatchNorm1d if *batchnorm* and ReLU for each hidden width, then a
    Linear to *out_dim* whose output is not activated."""
    dims = [in_dim, *widths]
    layers: list[nn.Module] = []
    for width_in, width_out in zip(dims, widths):
        norm = nn.BatchNorm1d(width_out) if batchnorm else nn.Identity()
        layers += [nn.Linear(width_in, width_out), norm, nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-1], out_dim))


def _dense(
    dims: list[int], final_activation: bool, dropout: float = 0.0
) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers, with dropout in front of each and a ReLU
    between them and, if asked, after the last."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if final_activation or i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param nodes: widths of the hidden layers, in order.
    :param batchnorm: normalise each hidden layer's output before its activation.
    :param activation: must be ``relu``; the argument stays because the checkpoints save
        it as a hyperparameter.
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
        if activation != "relu":
            raise ValueError(f"Unsupported activation: {activation}")
        self.net = dense_stack(in_dim, nodes, out_dim, batchnorm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
