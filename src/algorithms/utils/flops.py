"""Forward-FLOP counting for jet-tagging models."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


def _cpu_attention_flop(
    query_shape: torch.Size,
    key_shape: torch.Size,
    value_shape: torch.Size,
    *args,
    out_shape: tuple | None = None,
    **kwargs,
) -> int:
    """FLOPs of the CPU flash-attention kernel, by the formula of its CUDA siblings."""
    return sdpa_flop_count(query_shape, key_shape, value_shape)


_CPU_ATTENTION = {
    torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _cpu_attention_flop
}


@contextmanager
def counted_pass(model: torch.nn.Module, training: bool) -> Iterator[None]:
    """Hold *model* in train or eval mode, then restore its mode and its buffers."""
    was_training = model.training
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.train(training)
    try:
        yield
    finally:
        model.train(was_training)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])


def count_flops(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total forward FLOPs for one train-mode pass of *model* over *example_input*."""
    with counted_pass(model, training=True):
        try:
            return _counted_forward(model, example_input)
        except Exception as exc:  # pragma: no cover - diagnostic path
            log.warning(f"FLOP counting failed for {type(model).__name__}: {exc}")
            return float("nan")


def count_flops_per_jet(
    model: torch.nn.Module, nconstituents: int, nbr_feats: int
) -> float:
    """Forward FLOPs for a single jet of shape ``(nconstituents, nbr_feats)``."""
    example = torch.randn(2, nconstituents, nbr_feats)
    return count_flops(model, example) / 2.0


def _counted_forward(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total FLOPs of one traced forward pass, with a warning when it is zero."""
    counter = FlopCounterMode(display=False, custom_mapping=_CPU_ATTENTION)
    with counter, torch.no_grad():
        model(example_input)
    total = float(counter.get_total_flops())
    if total <= 0.0:
        log.warning(
            f"FLOP count for {type(model).__name__} is {total} - the forward pass "
            "dispatched no ops FlopCounterMode recognises (a fused kernel, or a "
            "model with no matmuls). Treat the reported FLOPs as unreliable."
        )
    return total
