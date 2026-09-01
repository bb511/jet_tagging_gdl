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
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    # Restored below: train mode would otherwise seed BatchNorm statistics with a batch of noise.
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.train(True)
    try:
        counter = FlopCounterMode(display=False)
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
    except Exception as exc:  # pragma: no cover - diagnostic path
        log.warning(f"FLOP counting failed for {type(model).__name__}: {exc}")
        return float("nan")
    finally:
        model.train(was_training)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])


def count_flops_per_jet(
    model: torch.nn.Module,
    nconstituents: int,
    nbr_feats: int,
    device: torch.device | None = None,
) -> float:
    """Forward FLOPs for a single jet of shape ``(nconstituents, nbr_feats)``."""
    example = torch.randn(2, nconstituents, nbr_feats, device=device)
    return count_flops(model, example) / 2.0


def record_flops(model: torch.nn.Module, nconstituents: int, nbr_feats: int) -> float:
    """Count per-jet FLOPs and store them in ``model.hparams``.

    Writes both ``flops_total`` and ``flops``; ``ResultsTableCallback`` reads ``flops_total`` first
    and falls back to ``flops``.
    """
    flops = count_flops_per_jet(model, nconstituents, nbr_feats)
    model.hparams["flops_total"] = flops
    model.hparams["flops"] = flops
    return flops
