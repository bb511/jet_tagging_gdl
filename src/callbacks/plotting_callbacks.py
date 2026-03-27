"""Callbacks for plotting training curves and ROC curves at end of training."""

from pathlib import Path
from typing import Optional

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

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        save_dir = Path(trainer.default_root_dir) / "plots"
        self._plot_training_curves(trainer, save_dir)
        self._plot_roc_curves(trainer, pl_module, save_dir)

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
        else:
            print("PlottingCallback: No metrics.csv found. Skipping training curves plot.")

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
