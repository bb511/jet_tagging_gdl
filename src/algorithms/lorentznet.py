"""LorentzNet, the architecture of arXiv:2201.08187 (Gong et al., 2022).

A jet is a fully connected graph over its constituents, self-loops excluded.
Each node has its four-momentum :math:`x_i = (E, p_x, p_y, p_z)` and one Lorentz scalar,
the signed log of its squared mass.

The metric is :math:`(+, -, -, -)` and the momenta are in GeV throughout: the edge
invariants are dimensionful and the signed log compresses them assuming that scale.
Batches must be de-normalised before they reach this module!!!
"""

import torch
from torch import nn

from src.algorithms import FourMomentumModule


def normsq4(p: torch.Tensor) -> torch.Tensor:
    """Minkowski square :math:`p \\cdot p` over the last axis, metric (+, -, -, -)."""
    psq = p.pow(2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Minkowski product :math:`p \\cdot q` over the last axis, metric (+, -, -, -)."""
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    """Signed log :math:`\\mathrm{sign}(p)\\,\\log(|p| + 1)`.

    This is equation (5) of arXiv:2201.08187. The edge invariants span several orders of
    magnitude in GeV squared; the signed log brings them into the linear layers' range
    while keeping the sign that separates timelike from spacelike.
    """
    return torch.sign(p) * torch.log(p.abs() + 1)


def _message_mlp(n_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * n_hidden + 2, n_hidden, bias=False),
        nn.BatchNorm1d(n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
    )


def _scalar_mlp(n_hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2 * n_hidden + 1, n_hidden),
        nn.BatchNorm1d(n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
    )


def _coordinate_mlp(n_hidden: int) -> nn.Sequential:
    last = nn.Linear(n_hidden, 1, bias=False)
    nn.init.xavier_uniform_(last.weight, gain=0.001)
    return nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), last)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block: one round of message passing over a flat node
    set.

    Two edge attributes enter the messages, the norm of :math:`x_i - x_j` and the product
    :math:`x_i \\cdot x_j`.

    :param n_hidden: width of the node scalars and of the messages.
    :param c_weight: step size of the coordinate update.
    :param last_layer: drop the coordinate update, whose result no later block reads.
    """

    def __init__(
        self, n_hidden: int, c_weight: float = 1.0, last_layer: bool = False
    ) -> None:
        super().__init__()
        self.c_weight = c_weight
        self.phi_e = _message_mlp(n_hidden)
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.phi_h = _scalar_mlp(n_hidden)
        self.phi_x = None if last_layer else _coordinate_mlp(n_hidden)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated ``(h, x)``.

        :param h: hidden nodes with shape (nodes, n_hidden)
        :param x: data 4-vector with shape (nodes, 4)
        :param src: the source edges (edges,)
        :param dst: the destination edges (edges,)
        :param node_attr: the attributes associated with each node, here just (nodes, 1)
        """
        x_diff = x[src] - x[dst]
        m = self._messages(h, x, x_diff, src, dst)
        if self.phi_x is not None:
            x = self._update_coordinates(x, x_diff, m, src)
        return self._update_scalars(h, m, node_attr, src), x

    def _messages(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        x_diff: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        """The (edges, n_hidden) messages, each gated by its own learned weight."""
        norms = psi(normsq4(x_diff)).unsqueeze(1)
        dots = psi(dotsq4(x[src], x[dst])).unsqueeze(1)
        m = self.phi_e(torch.cat([h[src], h[dst], norms, dots], dim=1))
        return m * self.phi_m(m)

    def _update_scalars(
        self,
        h: torch.Tensor,
        m: torch.Tensor,
        node_attr: torch.Tensor,
        src: torch.Tensor,
    ) -> torch.Tensor:
        """Add to each node the response to its summed incoming messages."""
        agg = h.new_zeros(h.shape[0], m.shape[1]).index_add_(0, src, m)
        return h + self.phi_h(torch.cat([h, agg, node_attr], dim=1))

    def _update_coordinates(
        self, x: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Move each node along the mean of its weighted edge differences.

        The step is a combination of four-momentum differences, so it transforms like
        one and the block stays equivariant.
        """
        # tanh bounds the gate at 1, so no step exceeds |x_diff| and the next block's signed log
        # cannot overflow. Not the reference implementation's per-component clamp at +-100 GeV: a
        # rotation mixes the components, so the clamp is not equivariant, and a trained gate
        # exceeds it on most edges (flip rates of 8 to 20 % under the probe, 2026-09-03).
        trans = x_diff * torch.tanh(self.phi_x(m))
        step = x.new_zeros(x.shape).index_add_(0, src, trans)
        count = x.new_zeros(x.shape).index_add_(0, src, torch.ones_like(trans))
        return x + self.c_weight * step / count.clamp(min=1)


def _blocks(n_hidden: int, n_layers: int, c_weight: float) -> nn.ModuleList:
    return nn.ModuleList(
        [
            LGEB(n_hidden, c_weight, last_layer=(i == n_layers - 1))
            for i in range(n_layers)
        ]
    )


def _decoder(n_hidden: int, n_class: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(n_hidden, n_class),
    )


class LorentzNetGraph(nn.Module):
    """LorentzNet over a batch of jets that are a fixed-width four-momentum tensor.

    :param n_hidden: width of the node scalars inside the blocks.
    :param c_weight: step size of the coordinate updates.
    :param nconstituents: constituent slots per jet, the second axis of a batch.
    """

    def __init__(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        n_class: int = 5,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
        nconstituents: int = 16,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        # The squared mass is a node's only scalar; the four-momentum carries the rest.
        self.embedding = nn.Linear(1, n_hidden)
        self.blocks = _blocks(n_hidden, n_layers, c_weight)
        self.graph_dec = _decoder(n_hidden, n_class, dropout)
        # The edges of one fully connected jet, self-loops excluded.
        idx = torch.arange(nconstituents)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        self.register_buffer("edge_src", src[src != dst], persistent=False)
        self.register_buffer("edge_dst", dst[src != dst], persistent=False)

    def _batch_edges(
        self, atom_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Index the edges that join two real constituents, over the flattened node
        axis.

        Padded constituents are dropped to reduce the cost of the graphs.

        :param atom_mask: (jets, constituents) boolean, true on real constituents.
        :return: two (edges,) index tensors into the flattened jets * constituents axis.
        """
        valid = atom_mask[:, self.edge_src] & atom_mask[:, self.edge_dst]
        jet_idx, edge_idx = valid.nonzero(as_tuple=True)
        offset = jet_idx * atom_mask.shape[1]
        return self.edge_src[edge_idx] + offset, self.edge_dst[edge_idx] + offset

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits, invariant under a Lorentz transformation of
            *pmu* and unchanged by how many slots the padding fills.
        """
        n_jets, n_constituents, _ = pmu.shape
        atom_mask = pmu[..., 0] != 0.0
        src, dst = self._batch_edges(atom_mask)
        # A nominally massless constituent has m^2 at the float32 rounding of
        # E^2 - |p|^2, whose root is O(10 MeV) at 20 GeV, enough to move the logits by
        # a per cent under a Lorentz transformation that should not shift them.
        scalars = psi(normsq4(pmu)).reshape(n_jets * n_constituents, 1)
        x = pmu.reshape(n_jets * n_constituents, 4)

        h = self.embedding(scalars)
        for block in self.blocks:
            h, x = block(h, x, src, dst, scalars)
        return self.graph_dec(self._pool(h, atom_mask))

    def _pool(self, h: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Average the node scalars of each jet over its real constituents.

        Over the real constituents rather than the slots, unlike the reference
        implementation, so that widening the padding leaves the logits alone.
        """
        counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        masked = h * atom_mask.reshape(-1, 1).to(h.dtype)
        return masked.view(atom_mask.shape[0], -1, self.n_hidden).sum(dim=1) / counts


class LorentzNet(FourMomentumModule):
    """Jet tagger built on Lorentz Group Equivariant Blocks (arXiv:2201.08187)."""

    def build_net(
        self,
        n_hidden: int = 72,
        n_layers: int = 6,
        c_weight: float = 1e-3,
        dropout: float = 0.0,
    ) -> nn.Module:
        """Build the graph network; the arguments are in :class:`LorentzNetGraph`."""
        self._register_denorm_scales()
        return LorentzNetGraph(
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_class=self.hparams.output_dim,
            c_weight=c_weight,
            dropout=dropout,
            nconstituents=self.hparams.nconstituents,
        )
