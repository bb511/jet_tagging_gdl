# Diagnostic plots of one run: training curves, ROC curves and the confusion matrix.
"""Draw the per-run plots and attach them to MLflow when MLflow is logging.

Every number comes from Lightning's hooks rather than from a logger backend, so the plots are drawn
under ``logger=none`` too. They go to ``<default_root_dir>/plots``, which the trainer config
anchors to the run's Hydra output directory.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger

from src.plot import confusion, curves, roc
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def _step_logits(pl_module: LightningModule, outputs: Any, x: torch.Tensor) -> torch.Tensor:
    """Return the logits for a batch, reusing the step's own when it hands them over.

    The steps of :class:`~src.algorithms.JetTaggingModule` return their ``model_step`` dict, so the
    logits come free; a module whose step returns nothing pays one forward pass here.
    """
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    return pl_module(x)


def _scalar(value: torch.Tensor | float | None) -> float:
    """Return a logged metric as a float, NaN when the metric was not logged."""
    return float("nan") if value is None else float(value)


def _current_lr(trainer: Trainer) -> float:
    """Return the learning rate of the first parameter group, NaN before it exists."""
    if not trainer.optimizers:
        return float("nan")
    return float(trainer.optimizers[0].param_groups[0]["lr"])


class _Predictions:
    """Logits and true classes of one pass, accumulated on the CPU."""

    def __init__(self) -> None:
        self.logits: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def clear(self) -> None:
        """Drop everything collected so far."""
        self.logits.clear()
        self.labels.clear()

    def add(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Accumulate one batch; labels arrive one-hot and are stored as class indices."""
        self.logits.append(logits.detach().cpu())
        if labels.ndim > 1 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)
        self.labels.append(labels.detach().cpu())

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the true classes, the class probabilities and the predicted classes."""
        probs = torch.softmax(torch.cat(self.logits), dim=1).numpy()
        return torch.cat(self.labels).numpy(), probs, probs.argmax(axis=1)


class PlottingCallback(Callback):
    """Write the diagnostic plots of a run under ``<default_root_dir>/plots``.

    Three families: the training curves once the fit ends, and one-vs-rest ROC curves
    with a confusion matrix, both for the last validation epoch and for the test pass.
    The entry points set :attr:`subdir` per fold, so the folds of one training keep their
    figures apart.

    :param class_names: display names of the output classes, in column order.
    """

    def __init__(self, class_names: Sequence[str] = ("g", "q", "W", "Z", "t")) -> None:
        super().__init__()
        self.class_names = list(class_names)
        self._history: dict[str, list[float]] = {
            key: [] for key in ("epoch", "train/loss", "train/acc", "val/loss", "val/acc", "lr")
        }
        self._val = _Predictions()
        self._test = _Predictions()
        self._epoch_lr: float = float("nan")
        # The entry points set this to the run label ("main", "fold_1"), so that the folds of
        # one training do not draw over each other's figures. None keeps the plots directly
        # under the run directory.
        self.subdir: str | None = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Note the rate the epoch is about to run at.

        Lightning steps the epoch-interval schedulers before ``on_train_epoch_end``, so reading the
        rate there would report the next epoch's.
        """
        self._epoch_lr = _current_lr(trainer)

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the epoch empty: only the last epoch's predictions are ever plotted."""
        if not trainer.sanity_checking:
            self._val.clear()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate the validation predictions of one batch."""
        if trainer.sanity_checking:
            return
        x, y = batch
        self._val.add(_step_logits(pl_module, outputs, x), y)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Append this epoch's point to every curve.

        Not at the end of the validation epoch: the validation loop runs inside the training
        epoch, so ``callback_metrics`` still carries the previous epoch's ``train/*`` there, and
        none at all in the first epoch. Here both halves of the epoch are in.
        """
        metrics = trainer.callback_metrics
        self._history["epoch"].append(trainer.current_epoch)
        for key in ("train/loss", "train/acc", "val/loss", "val/acc"):
            self._history[key].append(_scalar(metrics.get(key)))
        self._history["lr"].append(self._epoch_lr)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Draw the curves of the whole fit and the plots of the last validation epoch."""
        if not self._history["epoch"]:
            log.warning("No epoch metrics were collected, so no training curves are drawn.")
        else:
            path = curves.plot(self._history, self._save_dir(trainer))
            self._log_artifacts(trainer, [path])
        self._draw_classification(trainer, self._val, "val")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the test pass empty, so a second pass does not stack on the first."""
        self._test.clear()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate the test predictions of one batch."""
        x, y = batch
        self._test.add(_step_logits(pl_module, outputs, x), y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Draw the ROC curves and the confusion matrix of the test pass."""
        self._draw_classification(trainer, self._test, "test")

    def _save_dir(self, trainer: Trainer) -> Path:
        """Plot directory of this run, beside its checkpoints under the Hydra output."""
        plots = Path(trainer.default_root_dir) / "plots"
        return plots if self.subdir is None else plots / self.subdir

    def _draw_classification(
        self, trainer: Trainer, predictions: _Predictions, split: str
    ) -> None:
        """Draw the ROC curves and the confusion matrix of one split, if it has scores."""
        if not predictions.logits:
            log.warning(f"No {split} predictions were collected, so those plots are skipped.")
            return
        labels, probs, preds = predictions.as_arrays()
        if np.isnan(probs).any():
            log.warning(f"The {split} scores hold NaN, so the run diverged; plots skipped.")
            return
        save_dir = self._save_dir(trainer)
        paths = [
            roc.plot(labels, probs, self.class_names, save_dir, split),
            confusion.plot(labels, preds, self.class_names, save_dir, split),
        ]
        self._log_artifacts(trainer, paths)

    @staticmethod
    def _log_artifacts(trainer: Trainer, paths: list[Path]) -> None:
        """Copy the figures into the MLflow run, when one of the loggers is MLflow."""
        for logger in trainer.loggers:
            if not isinstance(logger, MLFlowLogger):
                continue
            for path in paths:
                logger.experiment.log_artifact(logger.run_id, str(path), artifact_path="plots")
