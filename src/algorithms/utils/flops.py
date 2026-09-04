"""Forward-FLOP counting for jet-tagging models.

``torch.utils.flop_counter.FlopCounterMode`` counts the FLOPs a real forward pass dispatches
(matmul/addmm/bmm/conv/SDPA). The ``torchinfo`` ``total_mult_adds`` used before counted *weight*
multiply-adds instead, independent of the constituent count N, so DeepSets reported the same cost
at N=8 as at N=32 and an attention model the same cost at every N despite being O(N^2).

The convention is per jet (a batch of 2, halved: the counted ops are exactly linear in the batch,
and BatchNorm refuses a batch of 1 in train mode), FLOPs rather than MACs, padded constituents
included, and element-wise work (activations, norms, masking) left out.
"""

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def count_flops(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total forward FLOPs for one pass of *model* over *example_input*.

    Counted in train mode, because in eval mode ``nn.TransformerEncoder`` dispatches to a fused
    kernel ``FlopCounterMode`` has no formula for and silently reports 0 for the whole attention
    stack; train mode runs the unfused path and leaves matmul-only models unchanged. The original
    train/eval mode and buffers are restored, and an untraceable model gives NaN with a warning
    rather than breaking training.
    """
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


def count_flops_per_jet(model: torch.nn.Module, nconstituents: int, nbr_feats: int) -> float:
    """Forward FLOPs for a single jet of shape ``(nconstituents, nbr_feats)``."""
    example = torch.randn(2, nconstituents, nbr_feats)
    return count_flops(model, example) / 2.0


def record_flops(model: torch.nn.Module, nconstituents: int, nbr_feats: int) -> float:
    """Count per-jet FLOPs and store them as ``model.flops_total``.

    An attribute, not an hparam: ``save_hyperparameters`` has already run, and a key added to
    ``hparams`` afterwards would reach ``__init__`` on ``load_from_checkpoint`` as an argument the
    network does not take.
    """
    model.flops_total = count_flops_per_jet(model, nconstituents, nbr_feats)
    return model.flops_total


def _counted_forward(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total FLOPs of one traced forward pass, warning when the count is not to be trusted."""
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(example_input)
    total = float(counter.get_total_flops())
    _warn_if_unrecognised(model, total)
    return total


def _warn_if_unrecognised(model: torch.nn.Module, total: float) -> None:
    """Warn on a count of zero, which the counter also returns when it recognised no op at all."""
    if total > 0.0:
        return
    log.warning(
        f"FLOP count for {type(model).__name__} is {total} - the forward pass "
        "dispatched no ops FlopCounterMode recognises (a fused kernel, or a "
        "model with no matmuls). Treat the reported FLOPs as unreliable."
    )


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy every buffer, so the counted pass can be undone.

    That pass runs in train mode, which would otherwise seed the BatchNorm statistics with a batch
    of noise.
    """
    return {name: value.clone() for name, value in model.named_buffers()}


def _restore_state(
    model: torch.nn.Module, was_training: bool, buffers: dict[str, torch.Tensor]
) -> None:
    """Put the train/eval mode and the buffer values back as the counted pass found them."""
    model.train(was_training)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
