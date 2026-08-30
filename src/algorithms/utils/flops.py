# Per-jet forward FLOP counting, reported beside accuracy in the results table.
"""Forward-FLOP counting for jet-tagging models.

Why this exists
---------------
The previous approach used ``torchinfo``'s ``total_mult_adds``, which for
``nn.Linear`` counts *weight* multiply-adds and is therefore independent of the
constituent dimension N. That made the reported "FLOPs" a parameter-count proxy:
it scaled with the (hard-coded) batch dimension of the example input rather than
with the number of constituents, so a DeepSets model at N=32 reported the same
cost as at N=8, and an attention model reported the same cost at every N despite
being O(N^2).

``torch.utils.flop_counter.FlopCounterMode`` instead counts the FLOPs actually
dispatched during a real forward pass (matmul/addmm/bmm/conv/SDPA), so the count
scales correctly with N for both set-based and attention-based architectures.

Convention
----------
- **Per jet**: counted over a batch of 2 and halved. FLOPs are exactly linear in
  the batch dimension for every counted op, so this is exact. Batch 2 rather
  than 1, because BatchNorm (JetFormer) refuses a batch of 1 in train mode.
- **Counted in train mode.** In eval mode ``nn.TransformerEncoder`` dispatches to
  a fused kernel that ``FlopCounterMode`` has no formula for and silently reports
  **zero**; train mode keeps the explicit matmuls. Verified to leave matmul-only
  models (DeepSets, MLP) bit-identical.
- **FLOPs, not MACs**: a multiply-add counts as 2 (this is what
  ``FlopCounterMode`` reports; no extra factor is applied).
- Padded constituents are included, since the models process a fixed-width tensor,
  so this is the real cost, not the cost of the average jet occupancy.
- Element-wise work (activations, norms, masking) is not counted, per the usual
  convention that FLOPs are dominated by the matmuls.
"""

import torch

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def count_flops(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Total forward FLOPs for one pass of *model* over *example_input*.

    Counted in **train mode**: ``nn.TransformerEncoder`` and friends dispatch to
    a fused inference kernel in eval mode that ``FlopCounterMode`` has no formula
    for, silently reporting 0 for the entire attention stack. Train mode runs the
    unfused path, whose matmuls are counted correctly. This is purely a tracing
    concern: dropout and norm layers contribute no matmul FLOPs either way, and
    counts are identical for models without a fused path.

    Restores the model's original train/eval mode and its buffers; returns NaN
    (with a warning) if the model cannot be traced, so FLOP accounting never
    breaks training.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    # Train mode updates the running statistics of any BatchNorm layer, so the buffers are
    # put back below: counting FLOPs must not seed a network with a batch of noise.
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
