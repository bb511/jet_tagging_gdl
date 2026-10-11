import torch
from torch import nn

from src.algorithms import JetTaggingModule
from src.algorithms.components.mlp import _dense
from src.algorithms.components.moments import MOMENT_SETS, JetMoments
from src.algorithms.components.pooling import AGGREGATIONS, _check_choice, _masked_pool
from src.algorithms.deepsets import _moment_set, _MomentsRow


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
    :param moments: one of `MOMENT_SETS`; unless ``none``, the head also reads the
        `JetMoments` of that set, which need the inputs [p_T, etarel, phirel].
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        head_layers: list[int],
        out_dim: int,
        aggregation: str = "sum",
        moments: str = "none",
    ) -> None:
        super().__init__()
        _check_choice("aggregation", aggregation, AGGREGATIONS)
        _check_choice("moments", moments, MOMENT_SETS)
        self.aggregation = aggregation
        # A plain attribute under "none", as in DeepSetsNet.
        self.moments = None if moments == "none" else JetMoments(moments)
        self.embed = _dense([in_dim, hidden], True)
        self.local = _dense([hidden, hidden], True)
        self.context = _dense([hidden, hidden], True)
        self.shared = _dense([hidden, hidden], True)
        width = hidden if self.moments is None else hidden + self.moments.width
        self.head = _dense([width, *head_layers, out_dim], False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pooled(x)
        if self.moments is not None:
            pooled = torch.cat([pooled, self.moments(x)], dim=-1)
        return self.head(pooled)

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


class JEDILinear(JetTaggingModule):
    """JEDI-linear in float: DeepSets with one permutation-equivariant context layer,

    For more details, see (arXiv:2508.15468).
    """

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        aggregation: str = "sum",
    ) -> nn.Module:
        """Build the network; the arguments are documented in :class:`JEDILinearNet`."""
        return JEDILinearNet(
            in_dim=self.hparams.nbr_feats,
            hidden=hidden,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
        )


class JEDILinearMoments(_MomentsRow):
    """JEDI-linear whose head also reads the jet moments."""

    def build_net(
        self,
        hidden: int,
        head_layers: list[int],
        moments: str,
        aggregation: str = "sum",
    ) -> nn.Module:
        """Build the network; the arguments are documented in :class:`JEDILinearNet`."""
        return JEDILinearNet(
            in_dim=self.hparams.nbr_feats,
            hidden=hidden,
            head_layers=head_layers,
            out_dim=self.hparams.output_dim,
            aggregation=aggregation,
            moments=_moment_set(moments),
        )
