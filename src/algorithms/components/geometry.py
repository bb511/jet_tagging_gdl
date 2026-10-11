import math

import torch


def _form_matrix() -> torch.Tensor:
    """Return the (4, 4) matrix G of the invariant form B(u, v) = u^T G v."""
    metric = torch.zeros(4, 4)
    metric[0, 3] = metric[3, 0] = 1.0
    metric[1, 1] = metric[2, 2] = -2.0
    return metric


def _form(u: torch.Tensor, v: torch.Tensor, metric: torch.Tensor) -> torch.Tensor:
    """B(u, v) over the last axis, of length 4, elementwise over the leading ones."""
    return (u * (v @ metric)).sum(dim=-1)


def _unit_lift(eta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Lift (jets, N) angles to (jets, N, 4) rows (1, eta, phi, eta^2 + phi^2)."""
    return torch.stack([torch.ones_like(eta), eta, phi, eta**2 + phi**2], dim=-1)


def _sort_by_pt(x: torch.Tensor) -> torch.Tensor:
    """Order (jets, N, F) rows, p_T first, by decreasing p_T, padded rows last."""
    order = x[..., 0].argsort(dim=1, descending=True)
    return x.gather(1, order.unsqueeze(-1).expand_as(x))


def _relative_angles(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return eta and phi of sorted (jets, N, 3) rows measured from the leading row."""
    eta = x[..., 1] - x[:, :1, 1]
    phi = (x[..., 2] - x[:, :1, 2] + math.pi) % (2 * math.pi) - math.pi
    return eta, phi
