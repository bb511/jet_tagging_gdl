"""Invariance metric callback.

Computes invariance metrics d(f(x), f(T(x))) for each symmetry group
(SO2, SO3, Lorentz) during the test pass.

Augmentation is applied directly to each test batch (already normalized,
on-device) using the same central pipeline as online training augmentation.
Metrics are accumulated per batch and averaged at epoch end.

Metrics per group:
- Flip rate
- Wasserstein-1 distance between the *distributions* of f(x) and f(T(x))
  (per class-logit channel, averaged over channels) — unlike flip rate,
  this is not a paired/pointwise comparison: it asks whether the augmented
  test set's logit distribution matches the original's overall, regardless
  of which jet produced which value.

Normalised L2, cosine distance, Jensen-Shannon divergence, and orbit
variance (E_x [ Var_g ( f(gx) ) ]) are also implemented below but disabled
at the call sites — only wasserstein + flip are reported. Search for
"disabled" to re-enable.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from src.utils.augmentation import (
    apply_random_lorentz, apply_random_so2, apply_random_so3, permute_constituents,
)

log = logging.getLogger(__name__)


class InvarianceMetricCallback(L.Callback):
    def __init__(
        self,
        lorentz_eta_max: float = 2.0,
        seed: int = 42,
        orbit_variance_k: int = 5,
        compute_wasserstein: bool = True,
        compute_flip_breakdown: bool = True,
        class_names: Sequence[str] = ("g", "q", "W", "Z", "t"),
    ):
        super().__init__()
        self.lorentz_eta_max = lorentz_eta_max
        self.rng = np.random.default_rng(seed)   # used only for permutation
        self.orbit_variance_k = orbit_variance_k
        self.compute_wasserstein = compute_wasserstein
        self.compute_flip_breakdown = compute_flip_breakdown
        self.class_names = list(class_names)

        self._model_feature_names: Optional[List[str]] = None
        self._full_feature_names: Optional[List[str]] = None
        self._norm_scales: Optional[dict] = None
        self._dm_method: Optional[str] = None
        self._batch_metrics: Dict[str, List[float]] = {}
        self._orig_logits: List[torch.Tensor] = []
        self._aug_logits: Dict[str, List[torch.Tensor]] = {}
        self._flip_counts: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        dm = trainer.datamodule
        if dm.feature_names is None or dm.norm_params is None:
            log.warning("InvarianceMetricCallback: feature_names or norm_params not available.")
            return

        feature_indices = dm._get_feature_indices(Path(dm.processed_data_dir))
        if feature_indices is not None:
            self._model_feature_names = [dm.feature_names[i] for i in feature_indices]
        else:
            self._model_feature_names = list(dm.feature_names)
        # Test batches may carry either the model's reduced feature set or
        # the full on-disk set (e.g. datamodule.keep_full_features_at_test,
        # or method="symm_loss" which always keeps the full set) — pick the
        # right name list per-batch in on_test_batch_end by matching width.
        self._full_feature_names = list(dm.feature_names)

        # Augmentation functions need {feature_name: scale} to undo/redo normalisation
        self._norm_scales = {name: scale for name, (_kind, scale) in dm.norm_params.items()}
        self._dm_method = getattr(dm, "method", None)
        self._batch_metrics = {}
        self._orig_logits = []
        self._aug_logits = {}
        self._flip_counts = {}

        log.info(f"InvarianceMetricCallback: ready. Features: {self._model_feature_names}")

    # ------------------------------------------------------------------
    # Per-batch metrics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _point_metrics(
        self, y_orig: torch.Tensor, y_aug: torch.Tensor
    ) -> Dict[str, float]:
        """Flip rate between two batches of logits (L2/cosine/JSD disabled — only wasserstein + flip are reported)."""
        # disabled — only wasserstein + flip are reported
        # diff_norm = (y_orig - y_aug).norm(dim=1)
        # base_norm = y_orig.norm(dim=1)
        # mean_norm = base_norm.mean().item()
        # l2_normed = 0.0 if mean_norm < 1e-12 else diff_norm.mean().item() / mean_norm
        #
        # cos_dist = (1 - F.cosine_similarity(y_orig, y_aug, dim=1)).mean().item()
        #
        # p = F.softmax(y_orig, dim=-1)
        # q = F.softmax(y_aug, dim=-1)
        # m_mix = 0.5 * (p + q)
        # eps = 1e-8
        # js = 0.5 * (
        #     (p * (p.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
        #     + (q * (q.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
        # ).clamp(min=0).mean().item()

        flip_rate = (y_orig.argmax(dim=1) != y_aug.argmax(dim=1)).float().mean().item()

        return {"flip": flip_rate}
        # return {"l2": l2_normed, "cosine": cos_dist, "js": js, "flip": flip_rate}

    @torch.no_grad()
    def _orbit_variance(
        self, pl_module: L.LightningModule, x: torch.Tensor, transform_fn
    ) -> float:
        """E_x [ Var_g ( f(gx) ) ] estimated over orbit_variance_k draws.

        ``transform_fn`` returns ``(x_t, keep_mask)``; for so3, a *different*
        random subset of jets can be rejected on each of the K draws. Rather
        than reverting rejected jets to x (which would make g(x) == x and
        deflate the variance), predictions are scattered into a (K, B, C)
        NaN-filled canvas by original jet index, and the variance is computed
        only over jets that survived on all K draws.
        """
        preds = None
        for k in range(self.orbit_variance_k):
            x_t, keep_mask = transform_fn(x)
            y = F.normalize(pl_module(x_t), dim=1)
            if preds is None:
                preds = torch.full(
                    (self.orbit_variance_k, x.shape[0], y.shape[-1]),
                    float("nan"), device=y.device, dtype=y.dtype,
                )
            if keep_mask is None:
                preds[k] = y
            else:
                preds[k, keep_mask] = y

        valid = ~torch.isnan(preds).any(dim=(0, 2))  # (B,) — kept on every draw
        if not valid.any():
            return float("nan")
        preds_valid = preds[:, valid, :]
        mean_g = preds_valid.mean(dim=0, keepdim=True)
        return ((preds_valid - mean_g) ** 2).mean().item()

    @staticmethod
    def _so3_transform(t: torch.Tensor, names: List[str], ns: Optional[dict]):
        """so3 transform_fn: drops (x, keep_mask), discarding the rotation matrices."""
        x_t, _mats, keep_mask = apply_random_so3(t, feature_names=names, norm_scales=ns, apply_filter=True)
        return x_t, keep_mask

    # ------------------------------------------------------------------
    # Test batch hook
    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_test_batch_end(  # noqa: ARG002
        self,        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self._model_feature_names is None:
            return

        x = batch[0]                          # (B, C, F) — normalised, on device
        # x may carry the model's reduced feature set or the full on-disk set
        # (see on_test_start) — match names to x's actual width so so2/so3/
        # lorentz can find px/py/pz by name when they're present.
        names = (
            self._full_feature_names
            if self._full_feature_names is not None and x.shape[-1] == len(self._full_feature_names)
            else self._model_feature_names
        )
        ns = self._norm_scales

        y_orig = pl_module(x)

        if self.compute_wasserstein:
            self._orig_logits.append(y_orig.detach().cpu())

        if self.compute_flip_breakdown:
            y_true = batch[1]
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                y_true = torch.argmax(y_true, dim=1)
            correct_before = (y_orig.argmax(dim=1) == y_true)

        transforms: Dict[str, object] = {
            "so2":     lambda t: (apply_random_so2(t, feature_names=names, norm_scales=ns)[0], None),
            "so3":     lambda t: self._so3_transform(t, names, ns),
            "lorentz": lambda t: (apply_random_lorentz(t, feature_names=names, eta_max=self.lorentz_eta_max, norm_scales=ns)[0], None),
        }
        if self._dm_method == "equivariance":
            transforms["permutation"] = lambda t: (permute_constituents(t, self.rng), None)

        for group, transform_fn in transforms.items():
            x_aug, keep_mask = transform_fn(x)
            y_aug = pl_module(x_aug)

            # so3 can reject jets and drop them from x_aug — subset y_orig/
            # y_true/correct_before to the same surviving jets before pairing
            # them against y_aug, instead of comparing against the full,
            # unfiltered batch (which would silently compare mismatched jets).
            y_orig_g = y_orig if keep_mask is None else y_orig[keep_mask]

            for k, v in self._point_metrics(y_orig_g, y_aug).items():
                self._batch_metrics.setdefault(f"invariance/{group}_{k}", []).append(v)

            if keep_mask is not None:
                reject_frac = 1.0 - keep_mask.float().mean().item()
                self._batch_metrics.setdefault(f"invariance/{group}_reject_frac", []).append(reject_frac)

            # disabled — only wasserstein + flip are reported (also saves the
            # extra orbit_variance_k forward passes per group per batch)
            # ov = self._orbit_variance(pl_module, x, transform_fn)
            # self._batch_metrics.setdefault(f"invariance/{group}_orbit_var", []).append(ov)

            if self.compute_flip_breakdown:
                y_true_g = y_true if keep_mask is None else y_true[keep_mask]
                correct_before_g = correct_before if keep_mask is None else correct_before[keep_mask]
                correct_after = (y_aug.argmax(dim=1) == y_true_g)
                wrong_before = ~correct_before_g
                c = self._flip_counts.setdefault(
                    group, {"wrong_before": 0, "recovered": 0, "correct_before": 0, "stable": 0}
                )
                c["wrong_before"] += int(wrong_before.sum().item())
                c["recovered"] += int((wrong_before & correct_after).sum().item())
                c["correct_before"] += int(correct_before_g.sum().item())
                c["stable"] += int((correct_before_g & correct_after).sum().item())

            if self.compute_wasserstein:
                self._aug_logits.setdefault(group, []).append(y_aug.detach().cpu())

    # ------------------------------------------------------------------
    # Wasserstein: distribution-level (unpaired) comparison of f(x) vs f(T(x))
    # ------------------------------------------------------------------
    def _wasserstein_metrics(self) -> Dict[str, float]:
        """Per-group Wasserstein-1 distance between the original and augmented
        logit distributions, computed per class-logit channel and averaged.
        """
        if not self._orig_logits:
            return {}

        y_orig_all = torch.cat(self._orig_logits, dim=0).numpy()  # (N, C)
        num_classes = y_orig_all.shape[1]
        names = self.class_names[:num_classes]

        results: Dict[str, float] = {}
        for group, chunks in self._aug_logits.items():
            y_aug_all = torch.cat(chunks, dim=0).numpy()  # (N, C)
            per_class = []
            for c, name in enumerate(names):
                w = wasserstein_distance(y_orig_all[:, c], y_aug_all[:, c])
                results[f"invariance/{group}_wasserstein_{name}"] = w
                per_class.append(w)
            results[f"invariance/{group}_wasserstein"] = float(np.mean(per_class))

        return results

    # ------------------------------------------------------------------
    # Flip breakdown: label-aware transition probabilities, conditioned on
    # correctness before augmentation (unlike the label-agnostic "flip" rate
    # in _point_metrics, which only checks whether the predicted class changed).
    # ------------------------------------------------------------------
    def _flip_breakdown_metrics(self) -> Dict[str, float]:
        """Per-group P(correct after | wrong before), P(correct after | correct
        before), and P(wrong after | correct before), from accumulated counts.
        """
        results: Dict[str, float] = {}
        for group, c in self._flip_counts.items():
            wb, rec, cb, st = c["wrong_before"], c["recovered"], c["correct_before"], c["stable"]
            results[f"invariance/{group}_flip_recover"] = (rec / wb) if wb > 0 else float("nan")
            results[f"invariance/{group}_flip_stable"] = (st / cb) if cb > 0 else float("nan")
            results[f"invariance/{group}_flip_degrade"] = ((cb - st) / cb) if cb > 0 else float("nan")
        return results

    # ------------------------------------------------------------------
    # Epoch end: average across batches and log
    # ------------------------------------------------------------------
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._batch_metrics:
            return

        results = {k: float(np.mean(v)) for k, v in self._batch_metrics.items()}

        if self.compute_wasserstein:
            results.update(self._wasserstein_metrics())

        if self.compute_flip_breakdown:
            results.update(self._flip_breakdown_metrics())

        for name, value in results.items():
            pl_module.log(name, value, prog_bar=False, on_epoch=True, sync_dist=True)

        log.info(" | ".join(f"{k}: {v:.4f}" for k, v in results.items()))
