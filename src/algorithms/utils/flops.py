"""Forward-FLOP counting for jet-tagging models."""

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def count_flops(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total forward FLOPs for one pass of *model* over *example_input*."""
    was_training = model.training
    buffers = _snapshot_buffers(model)
    model.train(True)
    try:
        return _counted_forward(model, example_input)
    except Exception as exc:  # pragma: no cover - diagnostic path
        log.warning(f"FLOP counting failed for {type(model).__name__}: {exc}")
        return float("nan")
    finally:
        _restore_state(model, was_training, buffers)


def count_flops_per_jet(
    model: torch.nn.Module, nconstituents: int, nbr_feats: int
) -> float:
    """Forward FLOPs for a single jet of shape ``(nconstituents, nbr_feats)``."""
    example = torch.randn(2, nconstituents, nbr_feats)
    return count_flops(model, example) / 2.0


def record_flops(model: torch.nn.Module, nconstituents: int, nbr_feats: int) -> float:
    """Count per-jet FLOPs and store them as ``model.flops_total``."""
    model.flops_total = count_flops_per_jet(model, nconstituents, nbr_feats)
    return model.flops_total


def _counted_forward(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total FLOPs of one traced forward pass."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(example_input)
    total = float(counter.get_total_flops())
    _warn_if_unrecognised(model, total)
    return total


def _warn_if_unrecognised(model: torch.nn.Module, total: float) -> None:
    """Warn on a count of zero, which the counter returns when not recognising ops."""
    if total > 0.0:
        return
    log.warning(
        f"FLOP count for {type(model).__name__} is {total} - the forward pass "
        "dispatched no ops FlopCounterMode recognises (a fused kernel, or a "
        "model with no matmuls). Treat the reported FLOPs as unreliable."
    )


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy every buffer, so the counted pass can be undone."""
    return {name: value.clone() for name, value in model.named_buffers()}


def _restore_state(
    model: torch.nn.Module, was_training: bool, buffers: dict[str, torch.Tensor]
) -> None:
    """Put the train/eval mode and the buffer values back to what they were."""
    model.train(was_training)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
