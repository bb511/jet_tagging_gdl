from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from src.algorithms.utils.bitops import BitOps, count_ebops, record_ebops
from src.algorithms.utils.flops import count_flops

NOTHING = frozenset()


class _Lambda(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


class _Scaled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.rand(4) + 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


class _Attention(nn.Module):
    def __init__(self, mask: torch.Tensor | None = None) -> None:
        super().__init__()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(x, x, x, attn_mask=self.mask)


class _Noisy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen_training = self.training
        return x + torch.rand_like(x)


def _counted(fn: Callable[[torch.Tensor], torch.Tensor]) -> BitOps:
    # (jets, constituents, features) = (2, 5, 4): 40 elements, positive for sqrt.
    return count_ebops(_Lambda(fn), torch.rand(2, 5, 4))


def test_a_linear_layer_costs_its_products_and_its_bias_adds_per_jet() -> None:
    layer, n = nn.Linear(4, 3), 16

    # Per jet, N x 3 outputs of 4 products each at 8 x 8 bits, and a bias add at 8 bits.
    record_ebops(layer, n, 4)
    assert layer.ebops_total == 64 * n * 12 + 8 * n * 3
    assert layer.ebops_unpriced == 0
    assert count_ebops(layer, torch.rand(2, n, 4)).unclassified == NOTHING


def test_a_sum_of_squares_costs_its_products_and_its_adds() -> None:
    # The sum over the constituents folds 40 squares into 8 outputs with 32 adds.
    assert _counted(lambda x: (x * x).sum(1)) == BitOps(64 * 40 + 8 * 32, 0, NOTHING)


def test_softmax_is_left_unpriced_and_counted_by_element() -> None:
    assert _counted(lambda x: x.softmax(-1)) == BitOps(0, 40, NOTHING)


def test_a_division_is_priced_when_the_divisor_is_constant() -> None:
    assert _counted(lambda x: x / 4.0) == BitOps(64 * 40, 0, NOTHING)
    assert count_ebops(_Scaled(), torch.rand(2, 5, 4)) == BitOps(64 * 40, 0, NOTHING)
    # The sum folds 40 elements into 10 with 30 adds; the division by it is unpriced.
    by_sum = _counted(lambda x: x / x.sum(-1, keepdim=True))
    assert by_sum == BitOps(8 * 30, 40, NOTHING)


def test_a_square_is_priced_and_a_square_root_is_not() -> None:
    assert _counted(lambda x: x**2) == BitOps(64 * 40, 0, NOTHING)
    assert _counted(lambda x: x.sqrt()) == BitOps(0, 40, NOTHING)


def test_the_cpu_attention_kernel_is_priced_as_the_math_path() -> None:
    x = torch.rand(2, 2, 6, 4)
    # (jets, heads, L = S, E = E_v) = (2, 2, 6, 4): 144 scores, each with 4 products in
    # Q·Kᵀ and 4 in A·V, and the scaling of 4 x 12 query and key rows of 4 elements.
    plain = BitOps(64 * (144 * 8 + 4 * 12 * 4), 144, NOTHING)
    masked = plain._replace(ebops=plain.ebops + 8 * 144)

    assert count_ebops(_Attention(), x) == plain
    assert count_ebops(_Attention(torch.zeros(6, 6)), x) == masked


def test_index_bookkeeping_and_data_movement_cost_nothing() -> None:
    def shuffled(x: torch.Tensor) -> torch.Tensor:
        # Integer arithmetic, comparisons and remainders on index tensors alone.
        order = (torch.arange(5) * 3 + 1) % 5
        kept = (order > 1).nonzero().squeeze(1)
        picked = x[:, order].index_select(1, kept)
        moved = x.transpose(1, 2).flip(-1).permute(0, 2, 1)
        return torch.cat([picked, moved], 1).reshape(2, -1).clone()

    assert _counted(shuffled) == BitOps(0, 0, NOTHING)


def test_counting_runs_in_eval_mode_and_restores_the_module_and_the_rng() -> None:
    module = _Noisy()
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)

    # Per jet, 5 x 4 adds of the noise.
    record_ebops(module, 5, 4)
    assert module.ebops_total == 8 * 20
    assert torch.equal(torch.rand(3), expected)
    assert not module.seen_training
    assert module.training


def test_the_flop_counter_prices_the_cpu_attention_kernel() -> None:
    # Q·Kᵀ and A·V of 2 x 2 heads, 6 x 6 x 4 MACs each: 2 x 576 MACs at 2 FLOPs.
    assert count_flops(_Attention(), torch.rand(2, 2, 6, 4)) == 2304
