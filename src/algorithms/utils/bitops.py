"""Effective bit operations (eBOPs) of one forward pass, every operand at b bits.

The price is HGQ's (Sun et al., arXiv:2405.00645v3, eq. 11) at uniform precision: b² per
scalar multiplication and b per explicit add, subtract, compare or mux, with the
accumulation inside a matrix product free. Ops with no form in those primitives are left
out of the eBOPs and counted by element in ``BitOps.unpriced``: softmax, exp, log, sqrt,
the reciprocal, division by a computed tensor, the smooth activations, sorts, linalg,
and the normalisations (batch norm folds into its neighbouring layer at inference; layer
norm needs a square root and a division by a computed tensor). Ops on integer or boolean
tensors alone are index bookkeeping, and views, copies, gathers and creation ops move
values without arithmetic: neither costs anything.

The ops priced are those that reach a ``TorchDispatchMode``, where CompositeImplicit ops
arrive in their parts: a Linear as ``t`` and ``addmm``, an einsum as ``bmm``. The tables
are keyed by the aten packet name, so an op absent from a torch version breaks nothing.
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from src.algorithms.utils.flops import restore_state, snapshot_buffers
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

BITS = 8

# (multiplications, adds, unpriced elements) of one op.
Price = tuple[float, float, float]
Pricer = Callable[[tuple, dict, Any, frozenset[int]], Price]
_FREE: Price = (0, 0, 0)

# Per output element, (multiplications, adds); a compare, a max or a mux costs an add.
_ELEMENTWISE: dict[str, tuple[int, int]] = {
    **dict.fromkeys("mul mul_".split(), (1, 0)),
    **dict.fromkeys("addcmul addcmul_".split(), (1, 1)),
    **dict.fromkeys(
        (
            "add add_ sub sub_ rsub neg abs sign relu relu_ leaky_relu leaky_relu_ "
            "hardsigmoid hardtanh hardtanh_ clamp clamp_ clamp_min clamp_min_ "
            "clamp_max clamp_max_ maximum minimum where masked_fill masked_fill_ "
            "gt ge lt le eq ne"
        ).split(),
        (0, 1),
    ),
}
# Counted by element, never priced.
_UNPRICED = frozenset(
    (
        "_softmax _safe_softmax _log_softmax logsumexp exp exp_ expm1 log log_ log1p "
        "log2 log10 sqrt sqrt_ rsqrt reciprocal native_batch_norm "
        "_native_batch_norm_legit _native_batch_norm_legit_no_training "
        "native_layer_norm tanh tanh_ gelu silu sigmoid sigmoid_ softplus erf sin cos "
        "tan asin acos atan atan2 sinh cosh asinh acosh atanh remainder sort argsort "
        "topk norm var var_mean std std_mean isinf isnan any all"
    ).split()
)
_UNPRICED_PREFIXES = ("linalg_", "_linalg_")
# Movement, creation, scalar extraction, index bookkeeping and random fills.
_IGNORED = frozenset(
    (
        "view _unsafe_view alias t transpose permute expand unsqueeze squeeze select "
        "slice unbind split split_with_sizes unfold cat stack clone _to_copy copy_ "
        "detach lift_fresh lift_fresh_copy index _unsafe_index index_select gather "
        "index_put index_put_ masked_select nonzero repeat repeat_interleave diagonal "
        "diag_embed flip roll constant_pad_nd scatter scatter_ slice_scatter "
        "select_scatter new_zeros new_ones new_empty new_full zeros zeros_like ones "
        "ones_like empty empty_like empty_strided full full_like fill_ zero_ eye "
        "arange linspace scalar_tensor _local_scalar_dense _fused_sdp_choice "
        "promote_types sym_size sym_stride sym_numel sym_storage_offset bitwise_and "
        "bitwise_or bitwise_xor bitwise_not logical_and logical_or logical_not rand "
        "rand_like randn randn_like uniform_ normal_ bernoulli_ random_"
    ).split()
)
# The fused kernel behind F.scaled_dot_product_attention on the CPU, where the count
# runs, priced as its math path.
_ATTENTION_KERNELS = ("_scaled_dot_product_flash_attention_for_cpu",)


class BitOps(NamedTuple):
    """The eBOPs of one pass, its unpriced elements and the float ops no table names."""

    ebops: float
    unpriced: float
    unclassified: frozenset[str]


def _first(out: Any) -> Any:
    """The result of an op; max.dim, sort and the norms return it first in a tuple."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _numel(value: Any) -> int:
    return value.numel() if isinstance(value, torch.Tensor) else 0


def _elements(args: tuple, out: Any) -> int:
    """Elements an unpriced op touches: its input for a reduction, else its output."""
    return max(_numel(args[0]) if args else 0, _numel(_first(out)))


def _touches_float(args: tuple, kwargs: dict) -> bool:
    leaves = tree_leaves((args, kwargs))
    return any(isinstance(x, torch.Tensor) and x.is_floating_point() for x in leaves)


def _is_constant(divisor: Any, constants: frozenset[int]) -> bool:
    """A Python number, a single element or a parameter or buffer of the model."""
    if not isinstance(divisor, torch.Tensor):
        return True
    return divisor.numel() == 1 or divisor.untyped_storage().data_ptr() in constants


def _product(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """The products mm, bmm, mv and dot: one multiplication per output element and
    contracted index."""
    return _numel(out) * args[0].shape[-1], 0, 0


def _biased_product(
    args: tuple, kwargs: dict, out: Any, constants: frozenset[int]
) -> Price:
    """The biased products addmm, baddbmm and addmv: the product of the last two
    operands, a bias add each."""
    return _numel(out) * args[1].shape[-1], _numel(out), 0


def _reduction(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """A reduction (sum, amax, max.dim, segment_reduce, ...): an add or a compare per
    element folded away."""
    return 0, _numel(args[0]) - _numel(_first(out)), 0


def _mean(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """The sum's adds and one multiplication per output element by the constant 1/n."""
    return _numel(out), _numel(args[0]) - _numel(out), 0


def _scatter_add(
    args: tuple, kwargs: dict, out: Any, constants: frozenset[int]
) -> Price:
    """index_add, scatter_add and their reduce siblings: one add per source element."""
    return 0, _numel(args[3]), 0


def _divide(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """A multiplication by a constant's reciprocal; unpriced by a computed tensor."""
    n = _numel(out)
    return (n, 0, 0) if _is_constant(args[1], constants) else (0, 0, n)


def _power(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """A multiplication per element for a square, unpriced for any other exponent."""
    n = _numel(out)
    square = isinstance(args[1], (int, float)) and args[1] == 2
    return (n, 0, 0) if square else (0, 0, n)


def _attention(args: tuple, kwargs: dict, out: Any, constants: frozenset[int]) -> Price:
    """The math path of scaled dot-product attention.

    Queries (..., L, E), keys (..., S, E), values (..., S, E_v): the scaling of queries
    and keys and the products Q·Kᵀ and A·V are multiplications, the mask adds, and the
    softmax is unpriced.
    """
    query, key, value = args[:3]
    heads, length, embed = math.prod(query.shape[:-2]), *query.shape[-2:]
    source, embed_v = key.shape[-2], value.shape[-1]
    scores = heads * length * source
    multiplications = scores * (embed + embed_v) + heads * (length + source) * embed
    masked = isinstance(kwargs.get("attn_mask"), torch.Tensor)
    return multiplications, scores * masked, scores


_SPECIAL: dict[str, Pricer] = {
    **dict.fromkeys("mm bmm mv dot".split(), _product),
    **dict.fromkeys("addmm baddbmm addmv".split(), _biased_product),
    **dict.fromkeys(
        "sum amax amin max min argmax argmin segment_reduce".split(), _reduction
    ),
    "mean": _mean,
    **dict.fromkeys(
        (
            "index_add index_add_ scatter_add scatter_add_ index_reduce index_reduce_ "
            "scatter_reduce scatter_reduce_"
        ).split(),
        _scatter_add,
    ),
    **dict.fromkeys("div div_".split(), _divide),
    **dict.fromkeys("pow pow_".split(), _power),
    **dict.fromkeys(_ATTENTION_KERNELS, _attention),
}


def _price(
    name: str, args: tuple, kwargs: dict, out: Any, constants: frozenset[int]
) -> Price | None:
    """The price of one op, None when no table names it."""
    if not _touches_float(args, kwargs) or name in _IGNORED:
        return _FREE
    if name in _ELEMENTWISE:
        multiplications, adds = _ELEMENTWISE[name]
        return multiplications * _numel(_first(out)), adds * _numel(_first(out)), 0
    if name in _SPECIAL:
        return _SPECIAL[name](args, kwargs, out, constants)
    if name in _UNPRICED or name.startswith(_UNPRICED_PREFIXES):
        return 0, 0, _elements(args, out)
    return None


def _constant_storages(model: torch.nn.Module) -> frozenset[int]:
    """Storage addresses of the parameters and buffers, which a view of one shares."""
    tensors = (*model.parameters(), *model.buffers())
    return frozenset(t.untyped_storage().data_ptr() for t in tensors)


class _BitOpsMode(TorchDispatchMode):
    """Price every op that reaches the dispatcher while the mode is active."""

    def __init__(self, constants: frozenset[int]) -> None:
        super().__init__()
        self.constants = constants
        self.ebops = 0.0
        self.unpriced = 0.0
        self.unclassified: set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Price an op by its parts when it decomposes, as FlopCounterMode does."""
        kwargs = kwargs or {}
        with self:
            out = func.decompose(*args, **kwargs)
        if out is not NotImplemented:
            return out
        out = func(*args, **kwargs)
        self._record(func.overloadpacket.__name__, args, kwargs, out)
        return out

    def _record(self, name: str, args: tuple, kwargs: dict, out: Any) -> None:
        price = _price(name, args, kwargs, out, self.constants)
        if price is None:
            self.unclassified.add(name)
            return
        multiplications, adds, unpriced = price
        self.ebops += BITS**2 * multiplications + BITS * adds
        self.unpriced += unpriced


def count_ebops(model: torch.nn.Module, example: torch.Tensor) -> BitOps:
    """The eBOPs of one eval-mode forward pass of *model* over the whole of *example*.

    The train/eval mode and the buffers are restored afterwards; a failed pass gives NaN
    and a warning.
    """
    was_training = model.training
    buffers = snapshot_buffers(model)
    model.eval()
    try:
        return _counted_forward(model, example)
    except Exception as exc:  # pragma: no cover - diagnostic path
        log.warning(f"eBOP counting failed for {type(model).__name__}: {exc}")
        return BitOps(math.nan, math.nan, frozenset())
    finally:
        restore_state(model, was_training, buffers)


def _counted_forward(model: torch.nn.Module, example: torch.Tensor) -> BitOps:
    """Price one pass; any draw inside it leaves the global RNG stream where it was."""
    mode = _BitOpsMode(_constant_storages(model))
    with torch.random.fork_rng(devices=[]), torch.no_grad(), mode:
        model(example)
    if mode.unclassified:
        log.warning(
            f"The eBOPs of {type(model).__name__} leave out the ops "
            f"{sorted(mode.unclassified)}, which no table of bitops.py names."
        )
    return BitOps(mode.ebops, mode.unpriced, frozenset(mode.unclassified))


def record_ebops(model: torch.nn.Module, nconstituents: int, nbr_feats: int) -> None:
    """Store the eBOPs of one fully occupied jet as ``model.ebops_total``.

    Its unpriced elements go to ``model.ebops_unpriced``. The example comes from a
    private generator, so the global RNG stream of the run that constructs the model is
    untouched.
    """
    generator = torch.Generator().manual_seed(0)
    example = torch.randn(2, nconstituents, nbr_feats, generator=generator)
    counts = count_ebops(model, example)
    model.ebops_total = counts.ebops / 2.0
    model.ebops_unpriced = counts.unpriced / 2.0
