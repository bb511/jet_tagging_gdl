# LorentzNet: Lorentz-equivariant message passing between the constituents of a jet.
"""LorentzNet, the architecture of arXiv:2201.08187 (Gong et al., 2022).

The constituents of a jet are the nodes of a fully connected graph without self-loops.
Every node carries its four-momentum :math:`x_i = (E, p_x, p_y, p_z)` and one Lorentz
scalar, the signed log of its squared mass. A block (LGEB) builds a message on each edge
out of the two Minkowski invariants of its endpoints, updates the node scalars with the
summed messages, and moves the four-momenta along the differences :math:`x_i - x_j`. The
scalars are therefore invariant and the coordinates equivariant under any Lorentz
transformation of the input, and the logits read off the pooled scalars are invariant.

The metric is :math:`(+, -, -, -)` and the momenta are in GeV throughout: the two edge
invariants are dimensionful and the signed log compresses them assuming that scale, so
batches must be de-normalised before they reach this module.
"""

import torch
from torch import nn

# The squared mass is the only scalar attached to a node; the four-momentum carries the rest.
N_SCALAR = 1


def normsq4(p: torch.Tensor) -> torch.Tensor:
    """Minkowski square :math:`p \\cdot p` over the last axis, metric (+, -, -, -)."""
    psq = p.pow(2)
    return 2 * psq[..., 0] - psq.sum(dim=-1)


def dotsq4(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Minkowski product :math:`p \\cdot q` over the last axis, metric (+, -, -, -)."""
    pq = p * q
    return 2 * pq[..., 0] - pq.sum(dim=-1)


def psi(p: torch.Tensor) -> torch.Tensor:
    """Signed log :math:`\\mathrm{sign}(p)\\,\\log(|p| + 1)`, equation (5) of arXiv:2201.08187.

    The edge invariants of a jet span several orders of magnitude in GeV squared; the signed log
    brings them into a range the linear layers can work in while keeping the sign that
    distinguishes timelike from spacelike separations.
    """
    return torch.sign(p) * torch.log(p.abs() + 1)


def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Sum the rows of *data* into *num_segments* buckets addressed by *segment_ids*.

    :param data: (rows, width) values to accumulate.
    :param segment_ids: (rows,) destination bucket of each row.
    :param num_segments: number of buckets; empty ones stay zero.
    """
    # index_add_ has no deterministic CUDA implementation, which is why the trainer runs
    # with deterministic=False.
    result = data.new_zeros((num_segments, data.shape[1]))
    result.index_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Mean of the rows of *data* per bucket; buckets that receive no row stay zero."""
    result = data.new_zeros((num_segments, data.shape[1]))
    count = data.new_zeros((num_segments, data.shape[1]))
    result.index_add_(0, segment_ids, data)
    count.index_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class LGEB(nn.Module):
    """Lorentz Group Equivariant Block: one round of message passing.

    :param n_input: width of the incoming node scalars.
    :param n_output: width of the outgoing node scalars.
    :param n_hidden: width of the messages and of the hidden layers.
    :param n_node_attr: width of the invariant node attributes appended to every update.
    :param c_weight: step size of the coordinate update.
    :param last_layer: drop the coordinate update, whose result no later block reads.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_hidden: int,
        n_node_attr: int = 1,
        c_weight: float = 1.0,
        last_layer: bool = False,
    ) -> None:
        super().__init__()
        self.c_weight = c_weight
        # Two edge attributes: the norm of x_i - x_j and the product x_i . x_j.
        self.phi_e = nn.Sequential(
            nn.Linear(2 * n_input + 2, n_hidden, bias=False),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
        )
        self.phi_m = nn.Sequential(nn.Linear(n_hidden, 1), nn.Sigmoid())
        self.phi_h = nn.Sequential(
            nn.Linear(n_hidden + n_input + n_node_attr, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
        self.phi_x = None if last_layer else self._build_phi_x(n_hidden)

    @staticmethod
    def _build_phi_x(n_hidden: int) -> nn.Sequential:
        """Return the network weighting each coordinate step, initialised near zero.

        The gain of 0.001 on the last layer is the reference implementation's: the coordinate
        update feeds the next block's invariants, and full-gain initial steps move the four-momenta
        far enough to stall training.
        """
        last = nn.Linear(n_hidden, 1, bias=False)
        nn.init.xavier_uniform_(last.weight, gain=0.001)
        return nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(), last)

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_attr: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one block over a flat node set.

        :param h: (nodes, n_input) node scalars.
        :param x: (nodes, 4) node four-momenta.
        :param src: (edges,) index of the node each edge updates.
        :param dst: (edges,) index of the node each edge reads from.
        :param node_attr: (nodes, n_node_attr) invariant node attributes.
        :return: the updated ``(h, x)``.
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
        """Return the (edges, n_hidden) messages, each gated by its own learned weight."""
        norms = psi(normsq4(x_diff)).unsqueeze(1)
        dots = psi(dotsq4(x[src], x[dst])).unsqueeze(1)
        m = self.phi_e(torch.cat([h[src], h[dst], norms, dots], dim=1))
        return m * self.phi_m(m)

    def _update_scalars(
        self, h: torch.Tensor, m: torch.Tensor, node_attr: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Add to each node the response to its summed incoming messages."""
        agg = unsorted_segment_sum(m, src, h.shape[0])
        return h + self.phi_h(torch.cat([h, agg, node_attr], dim=1))

    def _update_coordinates(
        self, x: torch.Tensor, x_diff: torch.Tensor, m: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        """Move each node along the weighted mean of its edge differences.

        The step is a combination of differences of four-momenta, so it transforms like one and the
        block stays equivariant.
        """
        trans = x_diff * self.phi_x(m)
        # Clamped at +-100 GeV per component as in the reference implementation: one
        # outlying pair would otherwise move a node far enough for the invariants of the
        # next block to overflow the signed log.
        trans = trans.clamp(min=-100, max=100)
        return x + self.c_weight * unsorted_segment_mean(trans, src, x.shape[0])


class LorentzNetGraph(nn.Module):
    """LorentzNet over a batch of jets held as a fixed-width four-momentum tensor.

    :param n_hidden: width of the node scalars inside the blocks.
    :param n_layers: number of LGEB blocks.
    :param n_class: number of jet classes.
    :param c_weight: step size of the coordinate updates.
    :param dropout: dropout rate in the classification head.
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
        self.embedding = nn.Linear(N_SCALAR, n_hidden)
        self.blocks = nn.ModuleList(
            [
                LGEB(
                    n_hidden,
                    n_hidden,
                    n_hidden,
                    n_node_attr=N_SCALAR,
                    c_weight=c_weight,
                    last_layer=(layer == n_layers - 1),
                )
                for layer in range(n_layers)
            ]
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_class),
        )
        self._register_edge_template(nconstituents)

    def _register_edge_template(self, nconstituents: int) -> None:
        """Store the edges of one fully connected jet, self-loops excluded.

        Held as buffers so that they follow the module onto the device and no edge list is rebuilt
        from Python at every step. Not persistent: they are a function of *nconstituents* alone, so
        a checkpoint need not carry them.
        """
        idx = torch.arange(nconstituents)
        src, dst = torch.meshgrid(idx, idx, indexing="ij")
        off_diagonal = src != dst
        self.register_buffer("edge_src", src[off_diagonal], persistent=False)
        self.register_buffer("edge_dst", dst[off_diagonal], persistent=False)

    def _batch_edges(self, atom_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Index the edges that join two real constituents, over the flattened node axis.

        Padded constituents are dropped here rather than masked downstream, so the cost follows the
        jet multiplicity and not the slot count: the dense graph on 150 slots carries 22 350 edges
        per jet against the 2 900 that the record's mean multiplicity of 49 constituents needs.

        :param atom_mask: (jets, constituents) boolean, true on real constituents.
        :return: two (edges,) index tensors into the flattened jets * constituents axis.
        """
        valid = atom_mask[:, self.edge_src] & atom_mask[:, self.edge_dst]
        jet_idx, edge_idx = valid.nonzero(as_tuple=True)
        offset = jet_idx * atom_mask.shape[1]
        return self.edge_src[edge_idx] + offset, self.edge_dst[edge_idx] + offset

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with padded
            constituents as all-zero rows.
        :return: (jets, n_class) logits, invariant under a Lorentz transformation of *pmu* and
            unchanged by how many slots the padding fills.
        """
        n_jets, n_constituents, _ = pmu.shape
        atom_mask = pmu[..., 0] != 0.0
        src, dst = self._batch_edges(atom_mask)
        # The signed squared mass, as in the reference implementation without beam nodes.
        # Its root, which the beam variant takes instead, is far worse conditioned: a
        # nominally massless constituent has m^2 at the float32 rounding of E^2 - |p|^2,
        # and the root of that noise is O(10 MeV) at 20 GeV, enough to move the logits by
        # a per cent under a Lorentz transformation that should leave them alone.
        scalars = psi(normsq4(pmu)).reshape(n_jets * n_constituents, 1)
        x = pmu.reshape(n_jets * n_constituents, 4)

        h = self.embedding(scalars)
        for block in self.blocks:
            h, x = block(h, x, src, dst, scalars)
        return self.graph_dec(self._pool(h, atom_mask))

    def _pool(self, h: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Average the node scalars of each jet over its real constituents.

        Averaged over the real constituents rather than over the slots, unlike the reference
        implementation, so that widening the padding leaves the logits alone.
        """
        counts = atom_mask.sum(dim=1, keepdim=True).clamp(min=1).to(h.dtype)
        masked = h * atom_mask.reshape(-1, 1).to(h.dtype)
        return masked.view(atom_mask.shape[0], -1, self.n_hidden).sum(dim=1) / counts
