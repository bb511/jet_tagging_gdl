"""Invariance metrics for symmetry testing of jet classifiers.

Three complementary metrics:

1. **MSLD** (Mean Squared Lie Derivative) – infinitesimal sensitivity via
   autograd on Lie-algebra generators.  Normalised by the variance of the
   original logits so the value is scale-independent.
2. **FTD** (Finite Transformation Deviation) – symmetric KL divergence
   between model outputs before / after a finite group transformation,
   averaged per sample.
3. **I-score** (Kvinge invariance score) – normalised [0, 1] measure:
   ``1 − E[‖h(Tx) − h(x)‖²] / E[‖h(xi) − h(xj)‖²]`` where the
   denominator is estimated from random sample pairs.

All metrics treat the model as a black box ``model(feats) → logits`` and
work identically for DeepSets, MLP, and Transformer architectures.
"""

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def compute_msld(
    model: torch.nn.Module,
    feats: torch.Tensor,
    generators: List[torch.Tensor],
) -> float:
    """Compute the normalised Mean Squared Lie Derivative (MSLD).

    The raw squared difference is divided by the variance of the original
    logits, making the metric independent of logit scale.  Returns 0 when
    the model is perfectly invariant.
    """
    model.eval()
    with torch.no_grad():
        original_logits = model(feats)

    # Normaliser: variance of original logits (scalar)
    logit_var = original_logits.var().item()
    if logit_var < 1e-12:
        logit_var = 1.0  # avoid division by zero for constant outputs

    msld_values = []
    for generator in generators:
        transformed_feats = feats + torch.matmul(feats, generator)
        with torch.no_grad():
            transformed_logits = model(transformed_feats)

        # Per-sample mean squared difference, normalised by logit variance
        msld_value = ((original_logits - transformed_logits) ** 2).mean().item() / logit_var
        msld_values.append(msld_value)

    return sum(msld_values) / len(msld_values)


def compute_ftd(
    model: torch.nn.Module,
    feats: torch.Tensor,
    transformations: List[torch.Tensor],
) -> float:
    """Compute the Finite Transformation Deviation (FTD).

    Uses symmetric KL divergence between softmax outputs, averaged per
    sample (``reduction='batchmean'``).
    """
    model.eval()
    with torch.no_grad():
        original_logits = model(feats)
        log_p = F.log_softmax(original_logits, dim=1)
        p = log_p.exp()

    ftd_values = []
    for transformation in transformations:
        transformed_feats = torch.matmul(feats, transformation)
        with torch.no_grad():
            transformed_logits = model(transformed_feats)
            log_q = F.log_softmax(transformed_logits, dim=1)
            q = log_q.exp()

        # Symmetric KL, averaged per sample
        ftd_value = 0.5 * (
            F.kl_div(log_q, p, reduction='batchmean')
            + F.kl_div(log_p, q, reduction='batchmean')
        )
        ftd_values.append(ftd_value.item())

    return sum(ftd_values) / len(ftd_values)


def compute_i_score(
    model: torch.nn.Module,
    feats: torch.Tensor,
    transformations: List[torch.Tensor],
    num_pairs: int = 5000,
) -> float:
    """Compute the Kvinge I-score (0 = not invariant, 1 = perfectly invariant).

    Uses the proper normalisation:
        I = 1 − E[‖h(Tx) − h(x)‖²] / E[‖h(xi) − h(xj)‖²]

    The denominator is estimated from ``num_pairs`` random sample pairs,
    representing the baseline output variability across different inputs.
    """
    model.eval()
    batch_size = feats.shape[0]

    with torch.no_grad():
        original_logits = model(feats)

    # ── Denominator: expected squared distance between random pairs ──
    idx_i = torch.randint(0, batch_size, (num_pairs,))
    idx_j = torch.randint(0, batch_size, (num_pairs,))
    pair_dists_sq = (original_logits[idx_i] - original_logits[idx_j]).pow(2).sum(dim=1)
    denom = pair_dists_sq.mean().item()

    if denom < 1e-12:
        # Model outputs are constant → trivially invariant
        return 1.0

    # ── Numerator: expected squared distance under transforms ──
    numer_values = []
    for transformation in transformations:
        transformed_feats = torch.matmul(feats, transformation)
        with torch.no_grad():
            transformed_logits = model(transformed_feats)

        dist_sq = (original_logits - transformed_logits).pow(2).sum(dim=1).mean().item()
        numer_values.append(dist_sq)

    numer = sum(numer_values) / len(numer_values)
    return max(0.0, 1.0 - numer / denom)

