"""Helpers for symmetry-regularized training.

This module provides a minimal API for symmetry loss workflows:

1. Build transformed partner batches (SO(2), SO(3), Lorentz).
2. Measure output discrepancy between original and transformed batches.
3. Combine classification and symmetry losses.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def random_so3_matrices(
    num_mats: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random SO(3) matrices via QR decomposition."""
    if num_mats <= 0:
        raise ValueError("num_mats must be positive.")

    m = torch.randn(num_mats, 3, 3, device=device, dtype=dtype, generator=generator)
    q, r = torch.linalg.qr(m)

    # Fix sign ambiguity in QR decomposition.
    d = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    d = torch.where(d == 0, torch.ones_like(d), d)
    q = q @ torch.diag_embed(d)

    # Ensure proper rotation (determinant +1).
    det_q = torch.linalg.det(q)
    neg = det_q < 0
    if neg.any():
        q[neg, :, 0] = -q[neg, :, 0]

    return q


def random_lorentz_boost_matrices(
    num_mats: int,
    eta_max: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Generate random pure Lorentz boosts acting on (E, px, py, pz)."""
    if num_mats <= 0:
        raise ValueError("num_mats must be positive.")
    if eta_max <= 0:
        raise ValueError("eta_max must be positive.")

    direction = torch.randn(num_mats, 3, device=device, dtype=dtype, generator=generator)
    direction_norm = direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    direction = direction / direction_norm

    eta = torch.empty(num_mats, device=device, dtype=dtype).uniform_(
        -eta_max, eta_max, generator=generator
    )
    beta = torch.tanh(eta).unsqueeze(1) * direction  # (N, 3)
    beta2 = (beta * beta).sum(dim=1)  # (N,)

    gamma = 1.0 / torch.sqrt((1.0 - beta2).clamp_min(1e-12))
    gb = torch.where(
        beta2 > 1e-12,
        (gamma - 1.0) / beta2,
        torch.zeros_like(beta2),
    )

    l = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(num_mats, 1, 1)
    l[:, 0, 0] = gamma
    l[:, 0, 1:] = -gamma.unsqueeze(1) * beta
    l[:, 1:, 0] = -gamma.unsqueeze(1) * beta
    l[:, 1:, 1:] = l[:, 1:, 1:] + gb[:, None, None] * (
        beta[:, :, None] * beta[:, None, :]
    )

    return l


def apply_random_so2(
    x: torch.Tensor,
    px_index: int = 0,
    py_index: int = 1,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    per_sample: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(2) rotations around the z-axis to a batch.

    Args:
        x: Input tensor of shape ``(batch, nconst, nfeats)``.
        px_index: Feature index for ``px``.
        py_index: Feature index for ``py``.
        min_angle: Lower angle bound in radians.
        max_angle: Upper angle bound in radians.
        per_sample: If ``True``, each sample gets a different angle.
            If ``False``, one angle is used for the whole batch.
        generator: Optional torch RNG for reproducibility.

    Returns:
        Tuple ``(x_rot, theta)`` where:
            - ``x_rot`` has the same shape as ``x``.
            - ``theta`` has shape ``(batch,)`` when ``per_sample=True``,
              otherwise shape ``(1,)``.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    nfeats = x.shape[-1]
    if not (0 <= px_index < nfeats and 0 <= py_index < nfeats):
        raise ValueError(
            f"Invalid px/py indices ({px_index}, {py_index}) for nfeats={nfeats}."
        )

    batch_size = x.shape[0]
    num_angles = batch_size if per_sample else 1

    theta = torch.empty(
        num_angles,
        device=x.device,
        dtype=x.dtype,
    ).uniform_(min_angle, max_angle, generator=generator)

    # Shape to broadcast over constituents.
    theta_b = theta[:, None] if per_sample else theta.view(1, 1)
    cos_t = torch.cos(theta_b)
    sin_t = torch.sin(theta_b)

    px = x[:, :, px_index]
    py = x[:, :, py_index]

    px_rot = cos_t * px - sin_t * py
    py_rot = sin_t * px + cos_t * py

    x_rot = x.clone()
    x_rot[:, :, px_index] = px_rot
    x_rot[:, :, py_index] = py_rot
    return x_rot, theta


def apply_random_so3(
    x: torch.Tensor,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    per_sample: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random SO(3) rotations to (px, py, pz)."""
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    nfeats = x.shape[-1]
    for idx_name, idx_value in (
        ("px_index", px_index),
        ("py_index", py_index),
        ("pz_index", pz_index),
    ):
        if not (0 <= idx_value < nfeats):
            raise ValueError(f"Invalid {idx_name}={idx_value} for nfeats={nfeats}.")

    batch_size = x.shape[0]
    num_mats = batch_size if per_sample else 1
    rot_mats = random_so3_matrices(
        num_mats=num_mats,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    vec = torch.stack(
        [x[:, :, px_index], x[:, :, py_index], x[:, :, pz_index]],
        dim=-1,
    )  # (B, N, 3)

    if per_sample:
        vec_rot = torch.einsum("bij,bnj->bni", rot_mats, vec)
    else:
        vec_rot = torch.einsum("ij,bnj->bni", rot_mats[0], vec)

    x_rot = x.clone()
    x_rot[:, :, px_index] = vec_rot[:, :, 0]
    x_rot[:, :, py_index] = vec_rot[:, :, 1]
    x_rot[:, :, pz_index] = vec_rot[:, :, 2]
    return x_rot, rot_mats


def apply_random_lorentz(
    x: torch.Tensor,
    feature_names: list[str],
    eta_max: float = 2.0,
    per_sample: bool = True,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply random Lorentz boosts to (E, px, py, pz), inferring indices from feature_names."""
    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got shape={tuple(x.shape)}")

    def idx(name: str) -> int:
        try:
            return feature_names.index(name)
        except ValueError:
            raise ValueError(f"Feature '{name}' not found in feature_names.")

    e_index = idx("j1_e") if "j1_e" in feature_names else idx("e")
    px_index = idx("j1_px") if "j1_px" in feature_names else idx("px")
    py_index = idx("j1_py") if "j1_py" in feature_names else idx("py")
    pz_index = idx("j1_pz") if "j1_pz" in feature_names else idx("pz")

    batch_size = x.shape[0]
    num_mats = batch_size if per_sample else 1
    boosts = random_lorentz_boost_matrices(
        num_mats=num_mats,
        eta_max=eta_max,
        device=x.device,
        dtype=x.dtype,
        generator=generator,
    )

    e = x[:, :, e_index]
    px = x[:, :, px_index]
    py = x[:, :, py_index]
    pz = x[:, :, pz_index]

    # Padded constituents are usually all zeros.
    active = (e.abs() + px.abs() + py.abs() + pz.abs()) > 0

    p4 = torch.stack([e, px, py, pz], dim=-1)
    if per_sample:
        p4_b = torch.einsum("bij,bnj->bni", boosts, p4)
    else:
        p4_b = torch.einsum("ij,bnj->bni", boosts[0], p4)

    e_b = p4_b[:, :, 0]
    px_b = p4_b[:, :, 1]
    py_b = p4_b[:, :, 2]
    pz_b = p4_b[:, :, 3]

    if preserve_jet_phi:
        active_f = active.to(dtype=x.dtype)
        jet_px_orig = (px * active_f).sum(dim=1, keepdim=True)
        jet_py_orig = (py * active_f).sum(dim=1, keepdim=True)
        jet_px_b = (px_b * active_f).sum(dim=1, keepdim=True)
        jet_py_b = (py_b * active_f).sum(dim=1, keepdim=True)

        phi_orig = torch.atan2(jet_py_orig, jet_px_orig)
        phi_b = torch.atan2(jet_py_b, jet_px_b)
        theta = phi_orig - phi_b  # (B, 1)

        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        px_b, py_b = cos_t * px_b - sin_t * py_b, sin_t * px_b + cos_t * py_b

    valid = active
    if require_positive_energy:
        valid = valid & (e_b > 0)

    x_b = x.clone()
    x_b[:, :, e_index] = torch.where(valid, e_b, e)
    x_b[:, :, px_index] = torch.where(valid, px_b, px)
    x_b[:, :, py_index] = torch.where(valid, py_b, py)
    x_b[:, :, pz_index] = torch.where(valid, pz_b, pz)
    return x_b, boosts


def build_symmetric_batch(
    x: torch.Tensor,
    group: str = "so2",
    feature_names: Optional[list[str]] = None,
    px_index: int = 0,
    py_index: int = 1,
    pz_index: int = 2,
    min_angle: float = 0.0,
    max_angle: float = 2.0 * math.pi,
    eta_max: float = 2.0,
    per_sample: bool = True,
    preserve_jet_phi: bool = False,
    require_positive_energy: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Create transformed partners ``T(x)`` for symmetry regularization.

    Returns:
        ``(x_transformed, transform_meta)``, where ``transform_meta`` is:
        - SO(2): sampled angle tensor
        - SO(3): rotation matrices
        - Lorentz: boost matrices
    """
    g = group.lower()
    if g == "so2":
        return apply_random_so2(
            x=x,
            px_index=px_index,
            py_index=py_index,
            min_angle=min_angle,
            max_angle=max_angle,
            per_sample=per_sample,
            generator=generator,
        )

    if g == "so3":
        return apply_random_so3(
            x=x,
            px_index=px_index,
            py_index=py_index,
            pz_index=pz_index,
            per_sample=per_sample,
            generator=generator,
        )

    if g == "lorentz":
        if feature_names is None:
            raise ValueError("feature_names must be provided for Lorentz transforms.")
        return apply_random_lorentz(
            x=x,
            feature_names=feature_names,
            eta_max=eta_max,
            per_sample=per_sample,
            preserve_jet_phi=preserve_jet_phi,
            require_positive_energy=require_positive_energy,
            generator=generator,
        )

    raise NotImplementedError(
        f"Group '{group}' is not implemented. Choose one of ['so2', 'so3', 'lorentz']."
    )


def symmetry_distance(
    logits: torch.Tensor,
    logits_sym: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute discrepancy between model outputs on ``x`` and ``T(x)``.

    Supported metrics:
        - ``l2``: mean normalised L2 on logits
        - ``mse``: mean squared error on logits
        - ``cosine``: mean cosine distance
        - ``js``: mean Jensen-Shannon divergence
        - ``kl``: mean symmetric KL divergence
    """
    if logits.shape != logits_sym.shape:
        raise ValueError(
            "logits and logits_sym must have identical shape, got "
            f"{tuple(logits.shape)} vs {tuple(logits_sym.shape)}."
        )

    # Calculate the l1 betwen the trasnformed and un transformed batch and provide the mean of that value
    l1 = F.l1_loss(logits, logits_sym, reduction="none").mean(dim=1)
    # Normalise by the mean absolute logit value to make the loss scale-invariant
    norm = logits.abs().mean(dim=1).clamp_min(eps)
    return (l1 / norm).mean()


def combine_with_symmetry_loss(
    cls_loss: torch.Tensor,
    logits: torch.Tensor,
    logits_sym: torch.Tensor,
    weight: float,
    metric: str = "l2",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(total_loss, sym_loss)`` for ``L = L_cls + weight * L_sym``."""
    sym_loss = symmetry_distance(logits, logits_sym, eps=eps)
    total_loss = cls_loss + float(weight) * sym_loss
    return total_loss
