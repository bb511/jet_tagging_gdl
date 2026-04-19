"""Invariance metric callback.

Computes multiple invariance metrics d(f(x), f(T(x))) for each symmetry group
(SO2, SO3, Lorentz) at the end of every validation epoch.
"""

import logging
from pathlib import Path
from typing import Optional, Dict

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.augmentation import (
    apply_so2_single_rotation,
    apply_so3_single_rotation,
    apply_boost_preserve_phi,
    create_lorentz_boosts,
)

log = logging.getLogger(__name__)


class InvarianceMetricCallback(L.Callback):
    def __init__(
        self,
        num_samples: int = 10000,
        lorentz_eta_max: float = 2.0,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.lorentz_eta_max = lorentz_eta_max
        self.rng = np.random.default_rng(seed)

        self._x_val_raw: Optional[np.ndarray] = None
        self._feature_names: Optional[list] = None
        self._norm_params: Optional[dict] = None
        self._feature_indices: Optional[np.ndarray] = None

    # ------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        dm = trainer.datamodule

        if dm.x_val_raw is None:
            log.warning("InvarianceMetricCallback: x_val_raw not available.")
            return

        x_raw = dm.x_val_raw
        n = min(self.num_samples, x_raw.shape[0])
        idx = self.rng.choice(x_raw.shape[0], size=n, replace=False)
        self._x_val_raw = x_raw[idx]

        self._feature_names = dm.feature_names
        self._norm_params = dm.norm_params
        self._feature_indices = dm._get_feature_indices(
            Path(dm.processed_data_dir)
        )

        if self._norm_params is None:
            raise RuntimeError("Normalization params not found.")

        log.info(f"Cached {n} validation jets for invariance testing.")

    # ------------------------------------------------------------
    # Convert raw → model input
    # ------------------------------------------------------------
    def _to_model_input(self, x_raw: np.ndarray, dm) -> torch.Tensor:
        x_norm = dm.apply_physics_norm(
            x_raw, self._feature_names, self._norm_params
        )

        if self._feature_indices is not None:
            x_norm = x_norm[:, :, self._feature_indices]

        return torch.from_numpy(x_norm).float()

    # ------------------------------------------------------------
    # Core metric computation
    # ------------------------------------------------------------
    @torch.no_grad()
    def _measure_invariance(
        self,
        pl_module: L.LightningModule,
        dm,
        x_raw: np.ndarray,
        x_aug: np.ndarray,
    ) -> Dict[str, float]:

        device = pl_module.device

        # ✅ SORT AFTER TRANSFORM
        x_raw_sorted = dm._restrict_and_sort_by_pt(
            x_raw, dm.pt_idx, dm.nconstituents
        )
        x_aug_sorted = dm._restrict_and_sort_by_pt(
            x_aug, dm.pt_idx, dm.nconstituents
        )

        x_orig_t = self._to_model_input(x_raw_sorted, dm).to(device)
        x_aug_t = self._to_model_input(x_aug_sorted, dm).to(device)

        batch_size = 256
        all_orig, all_aug = [], []

        for start in range(0, x_orig_t.shape[0], batch_size):
            end = start + batch_size
            all_orig.append(pl_module(x_orig_t[start:end]))
            all_aug.append(pl_module(x_aug_t[start:end]))

        y_orig = torch.cat(all_orig, dim=0)
        y_aug = torch.cat(all_aug, dim=0)

        diff_norm = (y_orig - y_aug).norm(dim=1)
        base_norm = y_orig.norm(dim=1)
        l2_normed = (
            0.0 if base_norm.mean() < 1e-12 else diff_norm.mean() / base_norm.mean()
        )

        cos_dist = (1 - F.cosine_similarity(y_orig, y_aug, dim=1)).mean().item()

        p = F.softmax(y_orig, dim=-1)
        q = F.softmax(y_aug, dim=-1)
        m = 0.5 * (p + q)

        eps = 1e-8
        js = 0.5 * (
            (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
            + (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
        ).clamp(min=0).mean().item()

        flip_rate = (y_orig.argmax(dim=1) != y_aug.argmax(dim=1)).float().mean().item()

        return {
            "l2": l2_normed,
            "cosine": cos_dist,
            "js": js,
            "flip": flip_rate,
        }

    # ------------------------------------------------------------
    # Orbit variance
    # ------------------------------------------------------------
    @torch.no_grad()
    def _measure_orbit_variance(
        self,
        pl_module: L.LightningModule,
        dm,
        x_raw: np.ndarray,
        transform_fn,
        K: int = 5,
    ) -> float:

        device = pl_module.device
        preds = []

        for _ in range(K):
            x_aug = transform_fn(x_raw)

            # ✅ SORT AFTER TRANSFORM
            x_aug_sorted = dm._restrict_and_sort_by_pt(
                x_aug, dm.pt_idx, dm.nconstituents
            )

            x_t = self._to_model_input(x_aug_sorted, dm).to(device)

            outputs = []
            for start in range(0, x_t.shape[0], 256):
                outputs.append(pl_module(x_t[start:start+256]))

            y = torch.cat(outputs, dim=0)
            preds.append(F.normalize(y, dim=1))

        preds = torch.stack(preds, dim=0)
        var_g = ((preds - preds.mean(dim=0, keepdim=True)) ** 2).mean(dim=0)

        return var_g.mean().item()

    # ------------------------------------------------------------
    # Validation hook
    # ------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        # ✅ Run ONLY on global rank 0
        if trainer.global_rank != 0:
            return

        if self._x_val_raw is None:
            return

        pl_module.eval()
        dm = trainer.datamodule
        x_raw = self._x_val_raw

        results = {}

        # ---------------- SO(2)
        theta = self.rng.uniform(0.3, 2 * np.pi - 0.3)
        c, s = np.cos(theta), np.sin(theta)

        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        x_aug = apply_so2_single_rotation(x_raw, R, self._feature_names)
        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug)

        for k, v in metrics.items():
            results[f"invariance/so2_{k}"] = v

        # ---------------- SO(3)
        angles = self.rng.uniform(0.3, 2 * np.pi - 0.3, size=3)

        cz, sz = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cx, sx = np.cos(angles[2]), np.sin(angles[2])

        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        R = (rz @ ry @ rx).astype(np.float32)

        x_aug = apply_so3_single_rotation(x_raw, R, self._feature_names)
        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug)

        for k, v in metrics.items():
            results[f"invariance/so3_{k}"] = v

        # ---------------- Lorentz
        boost = create_lorentz_boosts(1, self.lorentz_eta_max, self.rng)
        x_aug = apply_boost_preserve_phi(x_raw, boost[0], self._feature_names)

        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug)

        for k, v in metrics.items():
            results[f"invariance/lorentz_{k}"] = v

        # ---------------- Logging
        for k, v in results.items():
            pl_module.log(
                k,
                v,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,  # ✅ IMPORTANT: disable sync
            )

        log.info(" | ".join([f"{k}: {v:.4f}" for k, v in results.items()]))

    # Compute mean of each invariance metric after we are done
    def on_test_epoch_end(self, trainer, pl_module):
        # Gather all metrics logged during test
        metrics = trainer.callback_metrics  # This is a dict of all logged metrics

        # Filter for invariance metrics
        invariance_metrics = {k: v.item() if hasattr(v, 'item') else v
                            for k, v in metrics.items() if k.startswith("invariance/")}

        if invariance_metrics:
            log.info("Mean invariance metrics (test set):")
            for k, v in invariance_metrics.items():
                log.info(f"  {k}: {v:.4f}")
            print("Mean invariance metrics (test set):")
            for k, v in invariance_metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            log.info("No invariance metrics found in test metrics.")
            print("No invariance metrics found in test metrics.")

