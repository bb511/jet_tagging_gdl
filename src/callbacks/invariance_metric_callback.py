"""Invariance metric callback.

Computes invariance metrics d(f(x), f(T(x))) for each symmetry group
(SO2, SO3, Lorentz) during the test pass.

Augmentation is applied directly to each test batch (already normalized,
on-device) using the same central pipeline as online training augmentation.
Metrics are accumulated per batch and averaged at epoch end.

Metrics per group:
- Normalised L2
- Cosine distance
- Jensen-Shannon divergence
- Flip rate
- Orbit variance: E_x [ Var_g ( f(gx) ) ]
- Wasserstein-1 distance between the *distributions* of f(x) and f(T(x))
  (per class-logit channel, averaged over channels) — unlike the metrics
  above, this is not a paired/pointwise comparison: it asks whether the
  augmented test set's logit distribution matches the original's overall,
  regardless of which jet produced which value.
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
        class_names: Sequence[str] = ("g", "q", "W", "Z", "t"),
    ):
        super().__init__()
        self.lorentz_eta_max = lorentz_eta_max
        self.rng = np.random.default_rng(seed)   # used only for permutation
        self.orbit_variance_k = orbit_variance_k
        self.compute_wasserstein = compute_wasserstein
        self.class_names = list(class_names)

        self._model_feature_names: Optional[List[str]] = None
        self._full_feature_names: Optional[List[str]] = None
        self._norm_scales: Optional[dict] = None
        self._dm_method: Optional[str] = None
        self._batch_metrics: Dict[str, List[float]] = {}
        self._orig_logits: List[torch.Tensor] = []
        self._aug_logits: Dict[str, List[torch.Tensor]] = {}

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

        log.info(f"InvarianceMetricCallback: ready. Features: {self._model_feature_names}")

    # ------------------------------------------------------------------
    # Per-batch metrics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _point_metrics(
        self, y_orig: torch.Tensor, y_aug: torch.Tensor
    ) -> Dict[str, float]:
        """L2, cosine, JSD, flip between two batches of logits."""
        diff_norm = (y_orig - y_aug).norm(dim=1)
        base_norm = y_orig.norm(dim=1)
        mean_norm = base_norm.mean().item()
        l2_normed = 0.0 if mean_norm < 1e-12 else diff_norm.mean().item() / mean_norm

        cos_dist = (1 - F.cosine_similarity(y_orig, y_aug, dim=1)).mean().item()

        p = F.softmax(y_orig, dim=-1)
        q = F.softmax(y_aug, dim=-1)
        m_mix = 0.5 * (p + q)
        eps = 1e-8
        js = 0.5 * (
            (p * (p.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
            + (q * (q.add(eps).log() - m_mix.add(eps).log())).sum(dim=1)
        ).clamp(min=0).mean().item()

        flip_rate = (y_orig.argmax(dim=1) != y_aug.argmax(dim=1)).float().mean().item()

        return {"l2": l2_normed, "cosine": cos_dist, "js": js, "flip": flip_rate}

    @torch.no_grad()
    def _orbit_variance(
        self, pl_module: L.LightningModule, x: torch.Tensor, transform_fn
    ) -> float:
        """E_x [ Var_g ( f(gx) ) ] estimated over orbit_variance_k draws."""
        preds = []
        for _ in range(self.orbit_variance_k):
            y = F.normalize(pl_module(transform_fn(x)), dim=1)
            preds.append(y)
        preds = torch.stack(preds, dim=0)           # (K, B, C)
        mean_g = preds.mean(dim=0, keepdim=True)
        return ((preds - mean_g) ** 2).mean().item()

    # ------------------------------------------------------------------
    # Test batch hook
    # ------------------------------------------------------------------
    @torch.no_grad()
    def on_test_batch_end(  # noqa: ARG002
        self,
        trainer: L.Trainer,
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

        transforms: Dict[str, object] = {
            "so2":     lambda t: apply_random_so2(t, feature_names=names, norm_scales=ns)[0],
            "so3":     lambda t: apply_random_so3(t, feature_names=names, norm_scales=ns, apply_filter=True)[0],
            "lorentz": lambda t: apply_random_lorentz(t, feature_names=names, eta_max=self.lorentz_eta_max, norm_scales=ns)[0],
        }
        if self._dm_method == "equivariance":
            transforms["permutation"] = lambda t: permute_constituents(t, self.rng)

        for group, transform_fn in transforms.items():
            x_aug = transform_fn(x)
            y_aug = pl_module(x_aug)

            for k, v in self._point_metrics(y_orig, y_aug).items():
                self._batch_metrics.setdefault(f"invariance/{group}_{k}", []).append(v)

            ov = self._orbit_variance(pl_module, x, transform_fn)
            self._batch_metrics.setdefault(f"invariance/{group}_orbit_var", []).append(ov)

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
    # Epoch end: average across batches and log
    # ------------------------------------------------------------------
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self._batch_metrics:
            return

        results = {k: float(np.mean(v)) for k, v in self._batch_metrics.items()}

        if self.compute_wasserstein:
            results.update(self._wasserstein_metrics())

        for name, value in results.items():
            pl_module.log(name, value, prog_bar=False, on_epoch=True, sync_dist=True)

        log.info(" | ".join(f"{k}: {v:.4f}" for k, v in results.items()))
