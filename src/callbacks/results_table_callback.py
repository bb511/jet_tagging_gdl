import csv
import logging
import math
from pathlib import Path
from typing import Dict

import lightning as L
import numpy as np

log = logging.getLogger(__name__)


class ResultsTableCallback(L.Callback):
    """Aggregates fold-level test accuracy, invariance, and FLOPs into one table row."""

    # Label-aware flip-breakdown metrics (see InvarianceMetricCallback._flip_breakdown_metrics).
    # Always extracted/aggregated regardless of `invariance_metric`, unlike the
    # so2/so3/lorentz_{m} columns below which are parametrized by it.
    FLIP_BREAKDOWN_SUFFIXES = ("flip_recover", "flip_stable", "flip_degrade")

    def __init__(
        self,
        output_path: str = "results/results_table.csv",
        n_folds: int = 1,
        method: str = "baseline",
        model_name: str = "model",
        augmentation_scheme: str = "none",
        nconstituents: int = 0,
        selected_metric_mode: str = "scheme",
        invariance_metric: str = "l2",
        features: list = None,
    ):
        super().__init__()
        # Results are split into a subdirectory named after the literal input
        # feature list (e.g. "pt_etarel_phirel" vs "px_py_pz_etarel_phirel"),
        # since that's the axis that varies across baseline/augmentation/symm_loss runs.
        self.features_label = "_".join(str(f) for f in features) if features else None

        self.n_folds = int(n_folds)
        self.run_method_raw = method
        self.model_name = model_name
        self.augmentation_scheme = str(augmentation_scheme).lower()
        
        # Determine the display method based on the raw method and augmentation scheme
        if self.run_method_raw == "augmentation" and self.augmentation_scheme == "none":
            self.display_method = "baseline"
        elif self.run_method_raw == "equivariance":
            self.display_method = "permutation"
        elif self.run_method_raw == "augmentation":
            self.display_method = "augmentation"
        else:
            self.display_method = self.run_method_raw
            
        self.nconstituents = int(nconstituents)

        # Override output_path to include the feature-set subdir and the method,
        # e.g., results/pt_etarel_phirel/results_table_baseline.csv
        base_path = Path(output_path)
        out_dir = base_path.parent / self.features_label if self.features_label else base_path.parent
        self.output_path = out_dir / f"{base_path.stem}_{self.display_method}{base_path.suffix}"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.selected_metric_mode = selected_metric_mode
        self.invariance_metric = invariance_metric
        self.INVARIANCE_KEYS = (
            f"invariance/so2_{invariance_metric}",
            f"invariance/so3_{invariance_metric}",
            f"invariance/lorentz_{invariance_metric}",
        )
        self._FLIP_BREAKDOWN_KEYS = tuple(
            f"invariance/{g}_{s}"
            for s in self.FLIP_BREAKDOWN_SUFFIXES
            for g in ("so2", "so3", "lorentz")
        )

        self._fold_invariance: list[Dict[str, float]] = []
        self._fold_flops: list[float] = []
        self._fold_test_acc: list[float] = []
        self._summary_written = False

    @staticmethod
    def _to_float(value) -> float:
        if value is None:
            return float("nan")
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    @staticmethod
    def _safe_mean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return float("nan")
        return float(arr.mean())

    @staticmethod
    def _safe_std(values: list[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size <= 1:
            return 0.0
        return float(arr.std(ddof=1))

    @staticmethod
    def _format_pm(mean_val: float, std_val: float) -> str:
        if math.isnan(mean_val):
            return "nan"
        return f"{mean_val:.4f} +/- {std_val:.4f}"

    def _extract_invariance(self, metrics: Dict[str, object]) -> Dict[str, float]:
        keys = self.INVARIANCE_KEYS + self._FLIP_BREAKDOWN_KEYS
        return {k: self._to_float(metrics.get(k, None)) for k in keys}

    def _selected_metric_for_fold(self, inv: Dict[str, float]) -> tuple[str, float]:
        m = self.invariance_metric
        if self.selected_metric_mode == "mean_all":
            vals = [inv.get(k, float("nan")) for k in self.INVARIANCE_KEYS]
            return f"invariance/mean_{m}", self._safe_mean(vals)

        if self.augmentation_scheme == "permute":
            return "n/a", float("nan")

        if self.augmentation_scheme in {"so2", "so3", "lorentz"}:
            key = f"invariance/{self.augmentation_scheme}_{m}"
            return key, inv.get(key, float("nan"))

        vals = [inv.get(k, float("nan")) for k in self.INVARIANCE_KEYS]
        return f"invariance/mean_{m}", self._safe_mean(vals)

    def _selected_group_value(self, inv: Dict[str, float], suffix: str) -> float:
        """Like _selected_metric_for_fold's group-selection, but for a fixed
        suffix (e.g. 'flip_recover') instead of the configurable invariance_metric."""
        keys = (f"invariance/so2_{suffix}", f"invariance/so3_{suffix}", f"invariance/lorentz_{suffix}")
        if self.selected_metric_mode == "mean_all":
            return self._safe_mean([inv.get(k, float("nan")) for k in keys])

        if self.augmentation_scheme == "permute":
            return float("nan")

        if self.augmentation_scheme in {"so2", "so3", "lorentz"}:
            return inv.get(f"invariance/{self.augmentation_scheme}_{suffix}", float("nan"))

        return self._safe_mean([inv.get(k, float("nan")) for k in keys])

    def _maybe_write_summary(self) -> None:
        if self._summary_written:
            return

        n_test = len(self._fold_test_acc)
        # In the eval-only pass, invariance and flops
        # are never collected (on_fit_end never fires). Allow writing the
        # summary with NaN invariance/flops when only test acc is available.
        has_inv = len(self._fold_invariance) > 0
        has_flops = len(self._fold_flops) > 0
        if has_inv and has_flops:
            n_ready = min(n_test, len(self._fold_invariance), len(self._fold_flops))
        else:
            n_ready = n_test
        if n_ready < max(self.n_folds, 1):
            return

        inv_folds = self._fold_invariance[:n_ready]
        test_folds = self._fold_test_acc[:n_ready]
        flops_folds = self._fold_flops[:n_ready]

        selected_key, _ = self._selected_metric_for_fold(inv_folds[0])
        selected_values = [self._selected_metric_for_fold(inv)[1] for inv in inv_folds]

        m = self.invariance_metric
        so2_values = [inv.get(f"invariance/so2_{m}", float("nan")) for inv in inv_folds]
        so3_values = [inv.get(f"invariance/so3_{m}", float("nan")) for inv in inv_folds]
        lor_values = [inv.get(f"invariance/lorentz_{m}", float("nan")) for inv in inv_folds]

        test_mean = self._safe_mean(test_folds)
        test_std = self._safe_std(test_folds)
        selected_mean = self._safe_mean(selected_values)
        selected_std = self._safe_std(selected_values)
        flops_mean = self._safe_mean(flops_folds)
        
        so2_mean = self._safe_mean(so2_values)
        so2_std = self._safe_std(so2_values)
        so3_mean = self._safe_mean(so3_values)
        so3_std = self._safe_std(so3_values)
        lor_mean = self._safe_mean(lor_values)
        lor_std = self._safe_std(lor_values)

        flip_breakdown_cols: Dict[str, object] = {}
        for suffix in self.FLIP_BREAKDOWN_SUFFIXES:
            vals = [self._selected_group_value(inv, suffix) for inv in inv_folds]
            v_mean, v_std = self._safe_mean(vals), self._safe_std(vals)
            flip_breakdown_cols[f"{suffix}_mean"] = round(v_mean, 6)
            flip_breakdown_cols[f"{suffix}_std"] = round(v_std, 6)
            flip_breakdown_cols[f"{suffix}_pm"] = self._format_pm(v_mean, v_std)
            flip_breakdown_cols[f"{suffix}_folds"] = ";".join(
                "nan" if math.isnan(v) else f"{v:.6f}" for v in vals
            )

        row = {
            "method": self.display_method,
            "model": self.model_name,
            "augmentation_scheme": self.augmentation_scheme,
            "nconstituents": self.nconstituents,
            "folds_used": n_ready,
            "test_acc_mean": round(test_mean, 6),
            "test_acc_std": round(test_std, 6),
            "test_acc_pm": self._format_pm(test_mean, test_std),
            "selected_inv_key": selected_key,
            "selected_inv_mean": round(selected_mean, 6),
            "selected_inv_std": round(selected_std, 6),
            "selected_inv_pm": self._format_pm(selected_mean, selected_std),
            f"so2_{m}_mean": round(so2_mean, 6),
            f"so2_{m}_std": round(so2_std, 6),
            f"so2_{m}_pm": self._format_pm(so2_mean, so2_std),
            f"so3_{m}_mean": round(so3_mean, 6),
            f"so3_{m}_std": round(so3_std, 6),
            f"so3_{m}_pm": self._format_pm(so3_mean, so3_std),
            f"lorentz_{m}_mean": round(lor_mean, 6),
            f"lorentz_{m}_std": round(lor_std, 6),
            f"lorentz_{m}_pm": self._format_pm(lor_mean, lor_std),
            f"so2_{m}_folds": ";".join("nan" if math.isnan(v) else f"{v:.6f}" for v in so2_values),
            f"so3_{m}_folds": ";".join("nan" if math.isnan(v) else f"{v:.6f}" for v in so3_values),
            f"lorentz_{m}_folds": ";".join("nan" if math.isnan(v) else f"{v:.6f}" for v in lor_values),
            **flip_breakdown_cols,
            "flops_mean": round(flops_mean, 3) if not math.isnan(flops_mean) else float("nan"),
            "test_acc_folds": ";".join(f"{v:.6f}" for v in test_folds),
            "selected_inv_folds": ";".join(
                "nan" if math.isnan(v) else f"{v:.6f}" for v in selected_values
            ),
        }

        fieldnames = [
            "method",
            "model",
            "augmentation_scheme",
            "nconstituents",
            "folds_used",
            "test_acc_mean",
            "test_acc_std",
            "test_acc_pm",
            "selected_inv_key",
            "selected_inv_mean",
            "selected_inv_std",
            "selected_inv_pm",
            f"so2_{m}_mean",
            f"so2_{m}_std",
            f"so2_{m}_pm",
            f"so3_{m}_mean",
            f"so3_{m}_std",
            f"so3_{m}_pm",
            f"lorentz_{m}_mean",
            f"lorentz_{m}_std",
            f"lorentz_{m}_pm",
            f"so2_{m}_folds",
            f"so3_{m}_folds",
            f"lorentz_{m}_folds",
            *flip_breakdown_cols.keys(),
            "flops_mean",
            "test_acc_folds",
            "selected_inv_folds",
        ]

        file_exists = self.output_path.exists()
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self._summary_written = True

        inv_summary = "" if selected_key == "n/a" else f", {selected_key}={row['selected_inv_pm']}"
        log.info(
            f"[ResultsTableCallback] Written summary row -> {self.output_path}\n"
            f"  method={self.display_method}, model={self.model_name}, aug={self.augmentation_scheme}, nconst={self.nconstituents}\n"
            f"  test_acc={row['test_acc_pm']}{inv_summary}, flops={row['flops_mean']}"
        )

        inv_line = "" if selected_key == "n/a" else f"{selected_key}:    {row['selected_inv_pm']}\n"
        print(
            f"\n{'='*60}\n"
            f"Results Table Updated: {self.output_path}\n"
            f"Method:             {self.display_method}\n"
            f"Model:              {self.model_name}\n"
            f"Augmentation:       {self.augmentation_scheme}\n"
            f"nconst:             {self.nconstituents}\n"
            f"Test acc:           {row['test_acc_pm']}\n"
            f"{inv_line}"
            f"FLOPs (mean):       {row['flops_mean']}\n"
            f"{'='*60}\n"
        )

    # ------------------------------------------------------------------
    # Called once per fold after fit (validation metrics + FLOPs are available here)
    # ------------------------------------------------------------------
    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        inv = self._extract_invariance(metrics)
        flops = self._to_float(
            pl_module.hparams.get(
                "flops_total",
                pl_module.hparams.get("flops", float("nan")),
            )
        )

        self._fold_invariance.append(inv)
        self._fold_flops.append(flops)

        fold_idx = len(self._fold_invariance)
        selected_key, selected_val = self._selected_metric_for_fold(inv)
        log.info(
            f"[ResultsTableCallback] Fold {fold_idx}/{self.n_folds} | "
            f"{selected_key}={selected_val:.4f} | FLOPs={flops:.0f}"
        )

    # ------------------------------------------------------------------
    # Called once per fold after test
    # ------------------------------------------------------------------
    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        metrics = trainer.callback_metrics
        test_acc = self._to_float(metrics.get("test/acc", None))
        self._fold_test_acc.append(test_acc)

        fold_idx = len(self._fold_test_acc)
        log.info(
            f"[ResultsTableCallback] Fold {fold_idx}/{self.n_folds} | test/acc={test_acc:.4f}"
        )

        # In the eval-only pass, on_fit_end never fires
        # so invariance and flops are not yet collected. Pick them up here from
        # callback_metrics (InvarianceMetricCallback logs them in on_test_epoch_end)
        # and from the model's saved hparams.
        if len(self._fold_invariance) < fold_idx:
            inv = self._extract_invariance(metrics)
            self._fold_invariance.append(inv)
            selected_key, selected_val = self._selected_metric_for_fold(inv)
            log.info(
                f"[ResultsTableCallback] Fold {fold_idx}/{self.n_folds} | "
                f"{selected_key}={selected_val:.4f}"
            )

        if len(self._fold_flops) < fold_idx:
            flops = self._to_float(
                pl_module.hparams.get(
                    "flops_total",
                    pl_module.hparams.get("flops", float("nan")),
                )
            )
            self._fold_flops.append(flops)

        self._maybe_write_summary()
