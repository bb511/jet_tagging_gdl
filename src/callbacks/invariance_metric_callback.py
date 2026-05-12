"""Invariance metric callback.

Computes multiple invariance metrics d(f(x), f(T(x))) for each symmetry group
(SO2, SO3, Lorentz) at the end of every validation epoch.

Metrics:
- Normalised L2
- Cosine distance
- Jensen-Shannon divergence
- Flip rate
- Orbit variance
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
    permute_constituents,
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

        self._x_val_sorted: Optional[np.ndarray] = None
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

        x_sorted = dm._restrict_and_sort_by_pt(
            dm.x_val_raw, dm.pt_idx, dm.nconstituents
        )

        n = min(self.num_samples, x_sorted.shape[0])
        idx = self.rng.choice(x_sorted.shape[0], size=n, replace=False)
        self._x_val_sorted = x_sorted[idx]

        self._feature_names = dm.feature_names
        self._norm_params = dm.norm_params
        self._feature_indices = dm._get_feature_indices(
            Path(dm.processed_data_dir)
        )

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

        x_orig_t = self._to_model_input(x_raw, dm).to(device)
        x_aug_t = self._to_model_input(x_aug, dm).to(device)

        batch_size = 256
        all_orig, all_aug = [], []

        for start in range(0, x_orig_t.shape[0], batch_size):
            end = start + batch_size
            all_orig.append(pl_module(x_orig_t[start:end]))
            all_aug.append(pl_module(x_aug_t[start:end]))

        y_orig = torch.cat(all_orig, dim=0)
        y_aug = torch.cat(all_aug, dim=0)

        # -----------------------
        # 1. Normalised L2
        # -----------------------
        diff_norm = (y_orig - y_aug).norm(dim=1)
        base_norm = y_orig.norm(dim=1)

        mean_l2 = diff_norm.mean().item()
        mean_norm = base_norm.mean().item()

        l2_normed = 0.0 if mean_norm < 1e-12 else mean_l2 / mean_norm

        # -----------------------
        # 2. Cosine distance
        # -----------------------
        cos_sim = F.cosine_similarity(y_orig, y_aug, dim=1)
        cos_dist = (1 - cos_sim).mean().item()

        # -----------------------
        # 3. Jensen-Shannon
        # -----------------------
        p = F.softmax(y_orig, dim=-1)
        q = F.softmax(y_aug, dim=-1)
        m = 0.5 * (p + q)

        eps = 1e-8
        js = 0.5 * (
            (p * (p.add(eps).log() - m.add(eps).log())).sum(dim=1)
            + (q * (q.add(eps).log() - m.add(eps).log())).sum(dim=1)
        ).clamp(min=0).mean().item()

        # -----------------------
        # 4. Flip rate
        # -----------------------
        pred_orig = y_orig.argmax(dim=1)
        pred_aug = y_aug.argmax(dim=1)
        flip_rate = (pred_orig != pred_aug).float().mean().item()

        return {
            "l2": l2_normed,
            "cosine": cos_dist,
            "js": js,
            "flip": flip_rate,
        }

    # ------------------------------------------------------------
    # Orbit variance: E_x [ Var_g ( f(gx) ) ]
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
            x_t = self._to_model_input(x_aug, dm).to(device)

            outputs = []
            for start in range(0, x_t.shape[0], 256):
                outputs.append(pl_module(x_t[start:start+256]))

            y = torch.cat(outputs, dim=0)

            y = F.normalize(y, dim=1)

            preds.append(y)

        preds = torch.stack(preds, dim=0)  # (K, N, C)

        mean_g = preds.mean(dim=0, keepdim=True)
        var_g = ((preds - mean_g) ** 2).mean(dim=0)

        return var_g.mean().item()

    # ------------------------------------------------------------
    # Validation hook
    # ------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if self._x_val_sorted is None:
            return

        pl_module.eval()
        dm = trainer.datamodule
        x_raw = self._x_val_sorted  # numpy (N, C, F_raw)

        # augmentation.py is now torch-only; this helper converts numpy→torch,
        # applies fn, and converts the result back to numpy for _to_model_input.
        def aug(fn, x_np, *args):
            x_t = torch.from_numpy(x_np).float()
            return fn(x_t, *args).numpy()

        results = {}

        # ---------------- SO(2)
        theta = self.rng.uniform(0.3, 2 * np.pi - 0.3)
        cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

        rot_so2 = torch.tensor(
            [[cos_t, -sin_t, 0.0],
             [sin_t,  cos_t, 0.0],
             [0.0,    0.0,   1.0]],
            dtype=torch.float32,
        )

        x_aug_so2 = aug(apply_so2_single_rotation, x_raw, rot_so2, self._feature_names)

        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug_so2)
        for k, v in metrics.items():
            results[f"invariance/so2_{k}"] = v

        def _so2_transform(x):
            t = self.rng.uniform(0.3, 2 * np.pi - 0.3)
            c, s = float(np.cos(t)), float(np.sin(t))
            R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
            return aug(apply_so2_single_rotation, x, R, self._feature_names)

        results["invariance/so2_orbit_var"] = self._measure_orbit_variance(
            pl_module, dm, x_raw, _so2_transform
        )

        # ---------------- SO(3)
        angles = self.rng.uniform(0.3, 2 * np.pi - 0.3, size=3)

        cz, sz = np.cos(angles[0]), np.sin(angles[0])
        cy, sy = np.cos(angles[1]), np.sin(angles[1])
        cx, sx = np.cos(angles[2]), np.sin(angles[2])

        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

        rot_so3 = torch.from_numpy((rz @ ry @ rx).astype(np.float32))

        x_aug_so3 = aug(apply_so3_single_rotation, x_raw, rot_so3, self._feature_names)

        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug_so3)
        for k, v in metrics.items():
            results[f"invariance/so3_{k}"] = v

        def _so3_transform(x):
            a = self.rng.uniform(0.3, 2 * np.pi - 0.3, size=3)
            _cz, _sz = np.cos(a[0]), np.sin(a[0])
            _cy, _sy = np.cos(a[1]), np.sin(a[1])
            _cx, _sx = np.cos(a[2]), np.sin(a[2])
            _rz = np.array([[_cz, -_sz, 0], [_sz, _cz, 0], [0, 0, 1]])
            _ry = np.array([[_cy, 0, _sy], [0, 1, 0], [-_sy, 0, _cy]])
            _rx = np.array([[1, 0, 0], [0, _cx, -_sx], [0, _sx, _cx]])
            R = torch.from_numpy((_rz @ _ry @ _rx).astype(np.float32))
            return aug(apply_so3_single_rotation, x, R, self._feature_names)

        results["invariance/so3_orbit_var"] = self._measure_orbit_variance(
            pl_module, dm, x_raw, _so3_transform
        )

        # ---------------- Lorentz
        boost = create_lorentz_boosts(1, self.lorentz_eta_max, self.rng)

        x_aug_lor = aug(apply_boost_preserve_phi, x_raw, boost[0], self._feature_names)

        metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug_lor)
        for k, v in metrics.items():
            results[f"invariance/lorentz_{k}"] = v

        def _lorentz_transform(x):
            b = create_lorentz_boosts(1, self.lorentz_eta_max, self.rng)
            return aug(apply_boost_preserve_phi, x, b[0], self._feature_names)

        results["invariance/lorentz_orbit_var"] = self._measure_orbit_variance(
            pl_module, dm, x_raw, _lorentz_transform
        )

        # ---------------- Permutation (equivariance method only)
        if getattr(dm, "method", None) == "equivariance":
            x_aug_perm = aug(permute_constituents, x_raw, self.rng)
            metrics = self._measure_invariance(pl_module, dm, x_raw, x_aug_perm)
            for k, v in metrics.items():
                results[f"invariance/permutation_{k}"] = v

            results["invariance/permutation_orbit_var"] = self._measure_orbit_variance(
                pl_module, dm, x_raw,
                lambda x: aug(permute_constituents, x, self.rng),
            )

        # ---------------- Logging
        for name, value in results.items():
            pl_module.log(
                name,
                value,
                prog_bar=False,
                on_epoch=True,
                sync_dist=True,
            )

        log.info(
            " | ".join([f"{k}: {v:.4f}" for k, v in results.items()])
        )

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

