"""Callbacks for plotting training curves, ROC curves, and confusion matrix.

Logger-agnostic: metrics are collected via Lightning hooks, so plots work
regardless of which logger backend (MLflow, CSV, W&B, …) is active.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer


class PlottingCallback(Callback):
    """Collects metrics during training/testing and produces:

    1. **Training curves** – train & val loss, accuracy, LR vs epoch (``on_fit_end``).
    2. **Per-class ROC curves** – one-vs-rest on the **test** set (``on_test_epoch_end``).
    3. **Confusion matrix** – on the **test** set (``on_test_epoch_end``).
    """

    def __init__(
        self,
        class_names: Optional[list] = None,
        show: bool = False,
    ) -> None:
        super().__init__()
        self.class_names = class_names or ["g", "q", "W", "Z", "t"]
        self.show = show

        # Epoch-level metric accumulators (filled during training)
        self._train_loss: List[float] = []
        self._train_acc: List[float] = []
        self._val_loss: List[float] = []
        self._val_acc: List[float] = []
        self._lr: List[float] = []
        self._epochs: List[int] = []

        # Validation-time accumulators (for ROC / confusion at on_fit_end)
        self._val_logits: List[torch.Tensor] = []
        self._val_labels: List[torch.Tensor] = []

        # Test-time accumulators (filled during testing)
        self._test_logits: List[torch.Tensor] = []
        self._test_labels: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Training metric collection
    # ------------------------------------------------------------------
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Only collect val predictions from the LAST validation epoch
        # (reset at start of each val epoch, keep final).
        if trainer.sanity_checking:
            return
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x.to(pl_module.device))
        self._val_logits.append(logits.cpu())
        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        self._val_labels.append(y.cpu())

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        # Reset so we only keep the last epoch's predictions
        self._val_logits.clear()
        self._val_labels.clear()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Only record once per epoch (skip sanity check which has epoch=0 before training)
        if trainer.sanity_checking:
            return

        self._epochs.append(epoch)
        self._train_loss.append(metrics.get("train/loss", torch.tensor(float("nan"))).item())
        self._train_acc.append(metrics.get("train/acc", torch.tensor(float("nan"))).item())
        self._val_loss.append(metrics.get("val/loss", torch.tensor(float("nan"))).item())
        self._val_acc.append(metrics.get("val/acc", torch.tensor(float("nan"))).item())

        # Grab LR from the optimizer
        if trainer.optimizers:
            self._lr.append(trainer.optimizers[0].param_groups[0]["lr"])
        else:
            self._lr.append(float("nan"))

    # ------------------------------------------------------------------
    # Training curves plot
    # ------------------------------------------------------------------
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        save_dir = Path(trainer.default_root_dir) / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._plot_training_curves(save_dir)

        # Plot ROC + confusion on the validation set (properly processed data)
        if self._val_logits:
            logits = torch.cat(self._val_logits, dim=0)
            labels = torch.cat(self._val_labels, dim=0).numpy()
            probs = torch.softmax(logits, dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            if np.isnan(probs).any():
                print("PlottingCallback: Skipping ROC/confusion plots — predictions contain NaN (training diverged).")
            else:
                self._plot_roc_curves(labels, probs, save_dir, suffix="val", title_extra="Val Set")
                self._plot_confusion_matrix(labels, preds, save_dir, suffix="val", title_extra="Val Set")
                self._print_per_class_accuracy(labels, preds, split_name="val")

    def _plot_training_curves(self, save_dir: Path) -> None:
        if not self._epochs:
            print("PlottingCallback: No epoch metrics collected. Skipping training curves.")
            return

        epochs = np.array(self._epochs)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # --- Loss ---
        ax = axes[0]
        ax.plot(epochs, self._train_loss, "o-", label="Train", linewidth=2, markersize=4)
        ax.plot(epochs, self._val_loss, "o-", label="Val", linewidth=2, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Accuracy ---
        ax = axes[1]
        ax.plot(epochs, self._train_acc, "o-", label="Train", linewidth=2, markersize=4)
        ax.plot(epochs, self._val_acc, "o-", label="Val", linewidth=2, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Learning Rate ---
        ax = axes[2]
        lr_arr = np.array(self._lr)
        if not np.all(np.isnan(lr_arr)):
            ax.plot(epochs, lr_arr, "o-", linewidth=2, color="green", markersize=4)
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)

        # Fix x-axis for small epoch counts
        for ax in axes:
            ax.set_xlim(-0.5, max(epochs.max(), 1) + 0.5)
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        fig.tight_layout()
        path = save_dir / "training_curves.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"PlottingCallback: Saved training curves to {path}")
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    # ------------------------------------------------------------------
    # Test prediction collection
    # ------------------------------------------------------------------
    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._test_logits.clear()
        self._test_labels.clear()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        x, y = batch
        with torch.no_grad():
            logits = pl_module(x.to(pl_module.device))
        self._test_logits.append(logits.cpu())

        if y.ndim > 1 and y.shape[1] > 1:
            y = torch.argmax(y, dim=1)
        self._test_labels.append(y.cpu())

    # ------------------------------------------------------------------
    # Test-end plots: ROC + confusion matrix
    # ------------------------------------------------------------------
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self._test_logits:
            print("PlottingCallback: No test predictions collected. Skipping plots.")
            return

        save_dir = Path(trainer.default_root_dir) / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        logits = torch.cat(self._test_logits, dim=0)
        labels = torch.cat(self._test_labels, dim=0).numpy()
        probs = torch.softmax(logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)

        self._plot_roc_curves(labels, probs, save_dir, suffix="test", title_extra="Test Set")
        self._plot_confusion_matrix(labels, preds, save_dir, suffix="test", title_extra="Test Set")
        self._print_per_class_accuracy(labels, preds, split_name="test")

    def _plot_roc_curves(self, y_true: np.ndarray, y_score: np.ndarray, save_dir: Path,
                         suffix: str = "test", title_extra: str = "Test Set") -> None:
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        num_classes = y_score.shape[1]
        names = self.class_names[:num_classes]
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))

        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4), sharex=True, sharey=True)
        if num_classes == 1:
            axes = [axes]

        for k in range(num_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, k], y_score[:, k])
            roc_auc = auc(fpr, tpr)
            ax = axes[k]
            ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.4f}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
            ax.set_title(names[k])
            ax.set_xlabel("False Positive Rate")
            if k == 0:
                ax.set_ylabel("True Positive Rate")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Per-Class ROC Curves (One-vs-Rest) — {title_extra}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = save_dir / f"roc_curves_{suffix}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"PlottingCallback: Saved ROC curves to {path}")
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, save_dir: Path,
                               suffix: str = "test", title_extra: str = "Test Set") -> None:
        from sklearn.metrics import confusion_matrix

        num_classes = len(self.class_names)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        # Normalise each row (true class) to percentages
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Raw counts
        im1 = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
        ax1.set_title("Confusion Matrix (counts)")
        fig.colorbar(im1, ax=ax1, fraction=0.046)

        # Normalised
        im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
        ax2.set_title("Confusion Matrix (row-normalised)")
        fig.colorbar(im2, ax=ax2, fraction=0.046)

        for ax, data, fmt in [(ax1, cm, "d"), (ax2, cm_norm, ".2f")]:
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels(self.class_names, rotation=45)
            ax.set_yticklabels(self.class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            # Annotate cells
            thresh = data.max() / 2.0
            for i in range(num_classes):
                for j in range(num_classes):
                    ax.text(
                        j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black",
                        fontsize=9,
                    )

        fig.suptitle(f"{title_extra} Confusion Matrix", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        path = save_dir / f"confusion_matrix_{suffix}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"PlottingCallback: Saved confusion matrix to {path}")
        if self.show:
            plt.show()
        else:
            plt.close(fig)

    def _print_per_class_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, split_name: str = "test") -> None:
        num_classes = len(self.class_names)
        print(f"PlottingCallback: Per-class accuracy ({split_name} set):")
        for k in range(num_classes):
            mask = y_true == k
            if mask.sum() == 0:
                continue
            acc = (y_pred[mask] == k).sum() / mask.sum()
            print(f"  {self.class_names[k]:>4s}: {acc:.4f}  ({mask.sum()} samples)")
