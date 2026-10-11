import torch

AGGREGATIONS = ("sum", "max", "sum_over_n")


def _masked_pool(h: torch.Tensor, real: torch.Tensor, aggregation: str) -> torch.Tensor:
    """Pool (jets, constituents, width) over the real constituents, to (jets, width).

    The Linformer's components can be negative. This method accounts for that.

    :param real: (jets, constituents, 1), True for a real constituent.
    :param aggregation: one of `AGGREGATIONS`. ``sum_over_n`` divides the masked sum by
        the slot count N, the same for every jet, so unlike the mean it does not need
        per-jet division on FPGA.
    """
    if aggregation == "max":
        return h.masked_fill(~real, float("-inf")).amax(dim=1)
    total = (h * real).sum(dim=1)
    if aggregation == "sum":
        return total
    return total / h.shape[1]


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    """Refuse a *value* of the option *name* that is not one of *choices*."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {list(choices)}, got {value!r}.")
