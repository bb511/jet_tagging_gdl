"""Vector neurons (arXiv:2104.12229) on p_T-ordered constituents, with the weights
shared within blocks of the four-momentum.

The four-momentum is partitioned into splits, blocks of components that share their
weights: ``SPLITS`` holds the partitions. The components are ordered [E, px, py, pz],
the metric is (+, -, -, -), which every split inherits, and the momenta are in GeV.
Batches must be de-normalised before they reach this module.

Symmetry. Every layer mixes the vector channels with one weight matrix per split, and
the only other operations on a vector are the Minkowski product of two of them and a
scalar gate on each split. The rank invariants the first block reads are products within
blocks that are unions of splits. A transformation that acts within each split and
preserves the metric on it therefore leaves the logits alone. The ``z`` partition is
(px, py) and (E, pz), so its logits are invariant under O(2) x O(1,1): rotations about
the beam, boosts along it and the reflections phi -> -phi and pz -> -pz. The ``so2``
partition is E, pz and (px, py), so its logits are invariant under O(2) about the beam
and pz -> -pz. The constituents enter by rank of p_T, which both groups leave alone; the
sort also undoes any permutation of them.
"""

import torch
from torch import nn

from src.algorithms.components.lorentznet import psi

# The splits of each partition: indices into (E, px, py, pz).
SPLITS = {"z": [(1, 2), (0, 3)], "so2": [(0,), (3,), (1, 2)]}
# The blocks the rank invariants are taken on: the splits, and every union of splits
# that a coarser row uses, so that the inputs of so2 contain those of z.
RANK_BLOCKS = {"z": [(1, 2), (0, 3)], "so2": [(0,), (3,), (1, 2), (0, 3)]}
# Minkowski, on (E, px, py, pz); a split and a rank block inherit it. The sign it puts
# on a whole split is absorbed by that split's weights, so it leaves the groups alone.
METRIC = (1.0, -1.0, -1.0, -1.0)
_ALL_COMPONENTS = (0, 1, 2, 3)
# In units. Only a zero norm meets it, and 0 / floor = 0 keeps a padded rank at zero.
_NORM_FLOOR = 1e-20


def _check_symmetry(symmetry: str) -> None:
    if symmetry not in SPLITS:
        raise ValueError(f"symmetry must be one of {list(SPLITS)}, got {symmetry!r}.")


def _split_of(symmetry: str) -> torch.Tensor:
    """Return the split of each component, (4,)."""
    _check_symmetry(symmetry)
    split_of = torch.zeros(4, dtype=torch.long)
    for index, components in enumerate(SPLITS[symmetry]):
        split_of[list(components)] = index
    return split_of


def _mlp(dims: list[int], dropout: float) -> nn.Sequential:
    """Stack ``len(dims) - 1`` linear layers, ReLU between them and dropout in front of
    each."""
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _sort_by_pt(p: torch.Tensor) -> torch.Tensor:
    """Order the constituents of each jet by decreasing p_T.

    :param p: (jets, constituents, 4) four-momenta [E, px, py, pz]; a padded row has
        p_T = 0 and lands at the end.
    :return: the same rows, reordered along the constituent axis.
    """
    order = (p[..., 1] ** 2 + p[..., 2] ** 2).argsort(dim=1, descending=True)
    return p.gather(1, order.unsqueeze(-1).expand_as(p))


def _product(a: torch.Tensor, b: torch.Tensor, block: tuple[int, ...]) -> torch.Tensor:
    """Minkowski product of *a* and *b* over the components of *block*, elementwise over
    the leading axes."""
    return sum(METRIC[k] * a[..., k] * b[..., k] for k in block)


def _norm(a: torch.Tensor, block: tuple[int, ...]) -> torch.Tensor:
    """Square root of the absolute Minkowski square of *a* over *block*."""
    return _product(a, a, block).abs().sqrt()


def _deficit(
    p: torch.Tensor, jet: torch.Tensor, block: tuple[int, ...]
) -> torch.Tensor:
    """Return the Cauchy-Schwarz deficit of each constituent against the jet on *block*.

    :param p: (jets, constituents, 4) four-momenta in units.
    :param jet: (jets, 1, 4), their sum.
    :return: (jets, constituents), zero on a padded row.
    """
    p_norm, jet_norm = _norm(p, block), _norm(jet, block)
    # Half the squared norm of the difference of the unit vectors, times the norms:
    # |p||J| - p.J on a definite block and <p, J> - |p||J| on (E, pz), without
    # subtracting two nearly equal numbers as either of those forms would.
    unit_p = p / p_norm.clamp(min=_NORM_FLOOR).unsqueeze(-1)
    d = unit_p - jet / jet_norm.clamp(min=_NORM_FLOOR).unsqueeze(-1)
    return 0.5 * p_norm * jet_norm * _product(d, d, block).abs()


def _rank_entries(
    p: torch.Tensor, jet: torch.Tensor, blocks: list[tuple[int, ...]]
) -> torch.Tensor:
    """Return the (jets, constituents, 2 * blocks + 1) invariants of each rank: the
    square on every block, the deficit on every block, and the product with the jet."""
    squares = [_product(p, p, block) for block in blocks]
    deficits = [_deficit(p, jet, block) for block in blocks]
    return torch.stack([*squares, *deficits, _product(p, jet, _ALL_COMPONENTS)], -1)


def _jet_entries(jet: torch.Tensor, blocks: list[tuple[int, ...]]) -> torch.Tensor:
    """Return the (jets, blocks + 1) squares of the jet on every block and its mass
    squared."""
    squares = [_product(jet, jet, block) for block in (*blocks, _ALL_COMPONENTS)]
    return torch.stack(squares, dim=-1)


def _rank_invariants(p: torch.Tensor, blocks: list[tuple[int, ...]]) -> torch.Tensor:
    """Return the signed logs of the invariants of every rank and of the jet.

    A constituent is nearly parallel to its jet, so a product is dominated by the
    product of the norms, and the information sits in the Cauchy-Schwarz deficit, of
    relative size Delta R^2 ~ 1e-2. The signed log of a raw product buries it, and the
    learned products reach it only through a fine cancellation between two splits.

    :param p: (jets, N, 4) p_T-ordered four-momenta [E, px, py, pz] in units.
    :param blocks: the blocks of components, ``RANK_BLOCKS[symmetry]``.
    :return: (jets, N * (2 * n_blocks + 1) + n_blocks + 1). First, rank-major, for
        each rank i: <p_i, p_i>_b on every block b, the deficit of p_i against the jet
        J on every block, and <p_i, J> over all four components. Then <J, J>_b on every
        block and <J, J>. Every entry of a padded rank is exactly zero; on ``z`` the
        per-rank entries are -p_T^2, m_T^2, p_T p_TJ (1 - cos dphi),
        m_T m_TJ (cosh dy - 1) and <p_i, J>.
    """
    jet = p.sum(dim=1, keepdim=True)
    per_rank = _rank_entries(p, jet, blocks).flatten(1)
    return psi(torch.cat([per_rank, _jet_entries(jet.squeeze(1), blocks)], dim=-1))


def _rank_width(nconstituents: int, symmetry: str) -> int:
    """Width of the vector ``_rank_invariants`` returns."""
    n_blocks = len(RANK_BLOCKS[symmetry])
    return nconstituents * (2 * n_blocks + 1) + n_blocks + 1


class SplitLinear(nn.Module):
    """Mix the vector channels with one weight matrix per split, shared by the
    components of that split.

    No bias, since a constant vector would not transform.

    :param symmetry: a key of ``SPLITS``.
    """

    def __init__(self, c_in: int, c_out: int, symmetry: str) -> None:
        super().__init__()
        split_of = _split_of(symmetry)
        n_split = int(split_of.max()) + 1
        # Drawn directly: kaiming_uniform_ would read the fan-in of this 3-d tensor as
        # c_out * c_in.
        bound = c_in**-0.5
        weight = torch.empty(n_split, c_out, c_in).uniform_(-bound, bound)
        self.weight = nn.Parameter(weight)
        self.register_buffer("split_of", split_of, persistent=False)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Map (jets, c_in, 4) vectors to (jets, c_out, 4)."""
        # Each component meets the weights of its own split alone; a sum over the splits
        # would multiply it by the zeros of the others at full cost.
        return torch.einsum("moc,bcm->bom", self.weight[self.split_of], v)


class SplitInvariants(nn.Module):
    """Invariant products of pairs of learned combinations of the vector channels.

    The product is the Minkowski one, and summing it over all four components loses
    nothing: the weights of ``left`` and ``right`` are per split, so the relative weight
    and sign of each split's product are free.

    :param symmetry: a key of ``SPLITS``.
    """

    def __init__(self, channels: int, invariants: int, symmetry: str) -> None:
        super().__init__()
        self.left = SplitLinear(channels, invariants, symmetry)
        self.right = SplitLinear(channels, invariants, symmetry)
        self.register_buffer("metric", torch.tensor(METRIC), persistent=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Map (jets, channels, 4) vectors to (jets, invariants) signed-log scalars."""
        # Elementwise, not a matmul: invariant k pairs left and right channel k alone.
        return psi((self.left(u) * self.right(u) * self.metric).sum(dim=-1))


class SplitGate(nn.Module):
    """Scale each split of each vector channel by a sigmoid of the scalar state.

    :param symmetry: a key of ``SPLITS``.
    """

    def __init__(self, scalars: int, channels: int, symmetry: str) -> None:
        super().__init__()
        split_of = _split_of(symmetry)
        self.n_split = int(split_of.max()) + 1
        self.linear = nn.Linear(scalars, channels * self.n_split)
        # Every gate starts at exactly 1/2. One started near zero would stay there: the
        # first block has no residual and the invariants after it are quadratic in the
        # vectors, so their gradient vanishes with the vectors.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.register_buffer("split_of", split_of, persistent=False)

    def forward(self, s: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Gate the (jets, channels, 4) vectors *u* by the (jets, scalars) state *s*."""
        # (jets, channels * splits) -> (jets, channels, splits), channel-major.
        gates = torch.sigmoid(self.linear(s)).unflatten(-1, (-1, self.n_split))
        return u * gates[..., self.split_of]


class EquivariantBlock(nn.Module):
    """One update of the vector and scalar states of a jet.

    The block mixes the vectors, reads invariants off them, updates the scalars from
    those invariants and the context, and gates the mixed vectors with the new scalars.

    :param c_in: input vector channels, the constituent slots in the first block.
    :param s_in: input scalar channels, the width of the rank invariants in the first
        block.
    :param invariants: invariant products read off the mixed vectors.
    :param dropout: dropout rate in front of each linear layer of the scalar MLP.
    :param symmetry: a key of ``SPLITS``.
    :param residual: add the input to both states. Every block but the first does; the
        input of the first is the constituents by rank and their invariants, not a state
        of the same space.
    """

    def __init__(
        self,
        c_in: int,
        s_in: int,
        vector_channels: int,
        scalar_channels: int,
        invariants: int,
        dropout: float,
        symmetry: str,
        residual: bool,
    ) -> None:
        super().__init__()
        self.linear = SplitLinear(c_in, vector_channels, symmetry)
        self.invariants = SplitInvariants(vector_channels, invariants, symmetry)
        widths = [s_in + invariants + 1, scalar_channels, scalar_channels]
        self.mlp = _mlp(widths, dropout)
        self.gate = SplitGate(scalar_channels, vector_channels, symmetry)
        self.residual = residual

    def forward(
        self, v: torch.Tensor, s: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the updated ``(v, s)``.

        :param v: (jets, c_in, 4) vector state.
        :param s: (jets, s_in) scalar state.
        :param c: (jets, 1) context, the signed log of the multiplicity.
        :return: (jets, vector_channels, 4) vectors and (jets, scalar_channels) scalars.
        """
        u = self.linear(v)
        s_new = self.mlp(torch.cat([s, self.invariants(u), c], dim=-1))
        if self.residual:
            s_new = s_new + s
        v_new = self.gate(s_new, u)
        return (v_new + v if self.residual else v_new), s_new


def _blocks(
    nconstituents: int, rank_width: int, num_blocks: int, **block_kwargs
) -> nn.ModuleList:
    """Stack *num_blocks* blocks, the first reading the constituents by rank and their
    invariants."""
    first = (nconstituents, rank_width, False)
    later = (block_kwargs["vector_channels"], block_kwargs["scalar_channels"], True)
    return nn.ModuleList(
        EquivariantBlock(c_in, s_in, residual=residual, **block_kwargs)
        for c_in, s_in, residual in [first] + [later] * (num_blocks - 1)
    )


class VectorNeuronsClassifier(nn.Module):
    """Vector neurons over a fixed-width batch of jet four-momenta.

    The p_T-ordered constituents are the input vector channels of the first block and
    their rank invariants its input scalars; from there on the state is per jet. The
    logits are invariant under any permutation of the constituents and under the group
    that ``symmetry`` specifies.

    :param nconstituents: constituent slots per jet, the input vector channels of the
        first block.
    :param head_layers: hidden widths of the readout MLP.
    :param symmetry: a key of ``SPLITS``: ``z`` or ``so2``.
    :param vector_channels: four-vector channels of every block.
    :param scalar_channels: scalar channels of every block.
    :param invariants: invariant products read in every block and in the readout.
    :param dropout: dropout rate in front of each linear layer of the scalar MLPs and of
        the head; it never touches a vector.
    :param units: GeV per unit of the momenta the layers see, 20 as in the L-GATr-slim
        and LLoCa rows.
    :param n_class: classes, read off the scalars and the readout invariants.
    :raises ValueError: when ``symmetry`` is not a key of ``SPLITS``.
    """

    def __init__(
        self,
        nconstituents: int,
        head_layers: list[int],
        symmetry: str = "z",
        num_blocks: int = 3,
        vector_channels: int = 16,
        scalar_channels: int = 64,
        invariants: int = 16,
        dropout: float = 0.0,
        units: float = 20.0,
        n_class: int = 5,
    ) -> None:
        super().__init__()
        _check_symmetry(symmetry)
        self.units = units
        self.rank_blocks = RANK_BLOCKS[symmetry]
        self.blocks = _blocks(
            nconstituents,
            _rank_width(nconstituents, symmetry),
            num_blocks,
            vector_channels=vector_channels,
            scalar_channels=scalar_channels,
            invariants=invariants,
            dropout=dropout,
            symmetry=symmetry,
        )
        self.readout = SplitInvariants(vector_channels, invariants, symmetry)
        head = [scalar_channels + invariants + 1, *head_layers, n_class]
        self.head = _mlp(head, dropout)

    def forward(self, pmu: torch.Tensor) -> torch.Tensor:
        """Return the class logits of a batch of jets.

        :param pmu: (jets, constituents, 4) four-momenta [E, px, py, pz] in GeV, with
            padded constituents as all-zero rows.
        :return: (jets, n_class) logits.
        """
        real = pmu[..., 0] != 0.0
        p = _sort_by_pt(pmu / self.units)
        c = psi(real.sum(dim=1, keepdim=True).to(p.dtype))
        v, s = p, _rank_invariants(p, self.rank_blocks)
        for block in self.blocks:
            v, s = block(v, s, c)
        return self.head(torch.cat([s, self.readout(v), c], dim=-1))
