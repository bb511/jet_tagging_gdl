"""L-GATr-slim (arXiv:2512.17011), the scalars-and-vectors variant of L-GATr.

L-GATr is arXiv:2405.14806. The network is supplied by the ``lgatr`` package, and this
module adds only the tagger wrapper around it.

The metric is (+, -, -, -) and the momenta are in GeV. Batches must be de-normalised
before they reach this module.

Symmetry. With no spurions the logits are invariant under the full Lorentz group. The
time vector (1, 0, 0, 0) make the architecture SO(3) invariant. The two lightlike beams
(1, 0, 0, +-1), with or without the time vector, make it SO(2) invariant about the beam
axis.
"""

import lgatr
import torch
from torch import nn


def spurion_vectors(add_beams: bool, add_time: bool) -> torch.Tensor:
    """Reference four-vectors [E, px, py, pz], ``(k, 4)`` with k in {0, 1, 2, 3}."""
    rows = [[1.0, 0.0, 0.0, 0.0]] if add_time else []
    if add_beams:
        rows += [[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, -1.0]]
    return torch.tensor(rows).reshape(-1, 4)


class LGATrSlimClassifier(nn.Module):
    """L-GATr-slim over a batch of jets that are a fixed-width four-momentum tensor.

    The logits are invariant under any permutation of the constituents and under the
    group the spurions leave alone: the full Lorentz group without them, SO(3) with the
    time vector, SO(2) about the beam axis whenever the beams are on.

    :param units: GeV per unit of the constituent four-momenta, 20 as in the references;
        the spurions stay at unit scale.
    :param add_beams: add the lightlike beams (1, 0, 0, +-1) as two spurion tokens.
    :param add_time: add the time vector (1, 0, 0, 0) as a spurion token.
    :param n_class: classes, read off the scalar outputs of the global token.
    """

    def __init__(
        self,
        num_blocks: int = 12,
        hidden_v_channels: int = 32,
        hidden_s_channels: int = 96,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        units: float = 20.0,
        add_beams: bool = True,
        add_time: bool = True,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        self.units = units
        spurions = spurion_vectors(add_beams, add_time)
        self.register_buffer("spurions", spurions, persistent=False)
        # The logits are Lorentz scalars.
        self.net = lgatr.LGATrSlim(
            num_blocks=num_blocks,
            in_v_channels=1,
            out_v_channels=0,
            hidden_v_channels=hidden_v_channels,
            in_s_channels=1,
            out_s_channels=n_class,
            hidden_s_channels=hidden_s_channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout_prob=dropout or None,
        )

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits.
        """
        real = pmu[..., 0] != 0.0
        vectors, scalars, attn_mask = self._tokens(pmu / self.units, real)
        _, out_s = self.net(vectors, scalars, attn_mask=attn_mask)
        # Token 0 is the global token, whose scalar outputs are the logits.
        return out_s[:, 0]

    def _tokens(
        self, p: torch.Tensor, real: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Put the global and the spurion tokens ahead of the constituents.

        :param p: (jets, constituents, 4) four-momenta [E, px, py, pz] in ``units``.
        :param real: (jets, constituents), True on the real constituents.
        :return: the vectors (jets, tokens, 1, 4), the scalars (jets, tokens, 1) and the
            boolean key mask (jets, 1, 1, tokens), True on the keys attended to.
        """
        fixed = torch.cat([p.new_zeros(1, 4), self.spurions.to(p.dtype)])
        njets, nfixed = p.shape[0], fixed.shape[0]
        vectors = torch.cat([fixed.expand(njets, nfixed, 4), p], dim=1)
        scalars = p.new_zeros(njets, vectors.shape[1], 1)
        scalars[:, 0, 0] = 1.0
        attn_mask = torch.cat([real.new_ones(njets, nfixed), real], dim=1)
        return vectors.unsqueeze(2), scalars, attn_mask[:, None, None, :]
