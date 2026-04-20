"""Callbacks for plotting training curves and ROC curves at end of training."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer

from src.utils.utils import plot_roc_curves, plot_training_curves


class PlottingCallback(Callback):
    """Plots training curves and per-class ROC curves at the end of training.

    Training curves are read from the CSV logger's metrics file.
    ROC curves are computed by running inference on the validation set.
    """

    def __init__(
        self,
        class_names: Optional[list] = None,
        show: bool = False,
    ) -> None:
        super().__init__()
        self.class_names = class_names or ["g", "q", "W", "Z", "t"]
        self.show = show
        self._epoch_history: dict[int, dict[str, float]] = {}

    @staticmethod
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)
        except (TypeError, ValueError):
            return None

    def _capture_epoch_metrics(self, trainer: Trainer) -> None:
        if getattr(trainer, "sanity_checking", False):
            return

        epoch = int(trainer.current_epoch)
        row = self._epoch_history.setdefault(epoch, {"epoch": float(epoch)})

        metric_keys = (
            "train/loss",
            "train/acc",
            "val/loss",
            "val/acc",
            "val/acc_best",
        )
        for key in metric_keys:
            value = self._to_float(trainer.callback_metrics.get(key))
            if value is not None:
                row[key] = value

        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr", None)
            lr_value = self._to_float(lr)
            if lr_value is not None:
                row["lr"] = lr_value

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._capture_epoch_metrics(trainer)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._capture_epoch_metrics(trainer)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        save_dir = Path(trainer.default_root_dir) / "plots"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._capture_epoch_metrics(trainer)
        self._plot_training_curves(trainer, save_dir)
        self._plot_roc_curves(trainer, pl_module, save_dir)
        self._log_plots_to_mlflow(trainer, save_dir)

    def _plot_training_curves(self, trainer: Trainer, save_dir: Path) -> None:
        csv_path = None
        for logger in trainer.loggers:
            if hasattr(logger, "log_dir") and logger.log_dir is not None:
                candidate = Path(logger.log_dir) / "metrics.csv"
                if candidate.exists():
                    csv_path = str(candidate)
                    break

        if csv_path is None:
            output_dir = Path(trainer.default_root_dir)
            candidates = list(output_dir.rglob("metrics.csv"))
            if candidates:
                csv_path = str(candidates[0])

        if csv_path is not None:
            plot_training_curves(csv_path=csv_path, save_dir=str(save_dir), show=self.show)
        elif self._epoch_history:
            self._plot_training_curves_from_history(save_dir)
        else:
            print(
                "PlottingCallback: No metrics.csv or in-memory metrics found. "
                "Skipping training curves plot."
            )

    def _plot_training_curves_from_history(self, save_dir: Path) -> None:
        epochs = sorted(self._epoch_history.keys())
        if not epochs:
            return

        def values_for(key: str) -> list[float]:
            return [self._epoch_history[e].get(key, np.nan) for e in epochs]

        train_loss = values_for("train/loss")
        val_loss = values_for("val/loss")
        train_acc = values_for("train/acc")
        val_acc = values_for("val/acc")
        lr_vals = values_for("lr")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax = axes[0]
        if np.isfinite(train_loss).any():
            ax.plot(epochs, train_loss, label="Train Loss", linewidth=2)
        if np.isfinite(val_loss).any():
            ax.plot(epochs, val_loss, label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        ax = axes[1]
        if np.isfinite(train_acc).any():
            ax.plot(epochs, train_acc, label="Train Acc", linewidth=2)
        if np.isfinite(val_acc).any():
            ax.plot(epochs, val_acc, label="Val Acc", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        ax = axes[2]
        if np.isfinite(lr_vals).any():
            ax.plot(epochs, lr_vals, label="lr", linewidth=2, color="green")
            ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        if ax.has_data():
            ax.legend()

        fig.tight_layout()
        output_path = save_dir / "training_curves.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        if self.show:
            plt.show()
        else:
            plt.close(fig)

        print(f"PlottingCallback: Saved training curves to {output_path}.")

    def _plot_roc_curves(
        self, trainer: Trainer, pl_module: LightningModule, save_dir: Path
    ) -> None:
        datamodule = trainer.datamodule
        if datamodule is None:
            print("PlottingCallback: No datamodule found. Skipping ROC curves.")
            return

        val_dataloader = datamodule.val_dataloader()

        all_logits = []
        all_labels = []

        pl_module.eval()
        device = pl_module.device

        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                x = x.to(device)

                logits = pl_module(x)
                all_logits.append(logits.cpu())

                if y.ndim > 1 and y.shape[1] > 1:
                    y = torch.argmax(y, dim=1)
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        probs = torch.softmax(all_logits, dim=1).numpy()
        labels = all_labels.numpy()

        plot_roc_curves(
            y_true=labels,
            y_score=probs,
            class_names=self.class_names,
            save_dir=str(save_dir),
            show=self.show,
        )

    def _log_plots_to_mlflow(self, trainer: Trainer, save_dir: Path) -> None:
        artifact_files = [
            save_dir / "training_curves.png",
            save_dir / "roc_curves.png",
        ]
        artifact_files = [p for p in artifact_files if p.exists()]
        if not artifact_files:
            return

        for logger in trainer.loggers:
            run_id = getattr(logger, "run_id", None)
            experiment = getattr(logger, "experiment", None)
            if not run_id or experiment is None or not hasattr(experiment, "log_artifact"):
                continue

            for artifact_path in artifact_files:
                try:
                    experiment.log_artifact(run_id, str(artifact_path), artifact_path="plots")
                except TypeError:
                    experiment.log_artifact(run_id, str(artifact_path))
                except Exception as exc:
                    print(
                        "PlottingCallback: Failed to log artifact to MLflow "
                        f"({artifact_path.name}): {exc}"
                    )
