"""Permutation-invariant DeepSets classifiers for jets (arXiv:1703.06114).

``DeepSetsNet`` is the configurable DeepSets row. ``JEDILinearNet`` is JEDI-linear
(arXiv:2508.15468) in float: DeepSets with one permutation-equivariant context layer in
its per-constituent network.
"""

import torch
from torch import nn

AGGREGATIONS = ("sum", "max", "sum_over_n")


def _masked_pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`. ``sum_over_n`` divides the masked sum by
        the slot count N, the same for every jet, so unlike the mean it needs no per-jet
        division on an FPGA.
    """
    if aggregation == "max":
        # The Linformer's tokens can be negative, so a padded slot must lose every
        # comparison rather than enter the maximum as zero.
        return h.masked_fill(~real, float("-inf")).amax(dim=1)
    total = (h * real).sum(dim=1)
    if aggregation == "sum":
        return total
    return total / h.shape[1]


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


def _check_choices(aggregation: str) -> None:
    """Refuse a pooling this module does not provide, by name."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"aggregation must be one of {list(AGGREGATIONS)}, got {aggregation!r}."
        )


class DeepSetsNet(nn.Module):
    """Apply phi to every constituent, pool over the jet, then read the pool with rho.

    :param phi_layers: widths of the per-constituent network, the last entry being the
        width of the pooled constituent embedding.
    :param rho_layers: hidden widths of the network applied to the pooled embedding.
    :param aggregation: pooling over the real constituents, one of `AGGREGATIONS`.
    """

    def __init__(
        self,
        in_dim: int,
        phi_layers: list[int],
        rho_layers: list[int],
        out_dim: int,
        aggregation: str = "sum",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _check_choices(aggregation)
        self.aggregation = aggregation
        self.phi = _dense([in_dim, *phi_layers], final_activation=True)
        self.rho = _dense([phi_layers[-1], *rho_layers, out_dim], False, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the class logits for a batch of shape (jets, constituents,
        features)."""
        # A padded constituent is an all-zero row.
        real = x.abs().sum(dim=-1, keepdim=True) > 0
        return self.rho(_masked_pool(self.phi(x), real, self.aggregation))


class JEDILinearNet(nn.Module):
    """DeepSets whose per-constituent network carries one permutation-equivariant layer
    (Zaheer et al., arXiv:1703.06114): JEDI-linear (arXiv:2508.15468) in float.

    With s the ReLU, the context layer maps the embedding h_i of each constituent i to
    s(W1 h_i + b1) + s(W2 pool_j h_j + b2); a shared layer follows, and the head reads
    the pool of the result over the jet. Both pools use one aggregation over the real
    constituents, chosen among those the DeepSets row searches, so that the two rows
    differ in the context layer alone; padding changes nothing, and only the sum carries
    the multiplicity to the head. The published point is the sum. The released code
    scales each sum by a constant, which the next dense layer absorbs, and sums the
    padded rows too. Its BatchNorm is left out: it folds into the dense layers at
    inference and served the quantiser in training.

    :param hidden: width of the projection, the two context branches and the shared
        layer.
    :param head_layers: hidden widths of the MLP that reads the pooled vector.
    :param aggregation: one of `AGGREGATIONS`, for the context and the readout alike.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        head_layers: list[int],
        out_dim: int,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()
        _check_choices(aggregation)
        self.aggregation = aggregation
        self.embed = _dense([in_dim, hidden], True)
        self.local = _dense([hidden, hidden], True)
        self.context = _dense([hidden, hidden], True)
        self.shared = _dense([hidden, hidden], True)
        self.head = _dense([hidden, *head_layers, out_dim], False)

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
