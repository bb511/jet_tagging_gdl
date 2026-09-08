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

from src.plot import confusion, roc
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def _step_logits(pl_module: LightningModule, outputs: Any, x: torch.Tensor) -> torch.Tensor:
    """Return the logits for a batch, reusing the step's own when it hands them over.

    :class:`~src.algorithms.JetTaggingModule` returns its ``model_step`` dict, so the logits come
    free; a module whose step returns nothing pays one forward pass here.
    """
    if isinstance(outputs, dict) and "logits" in outputs:
        return outputs["logits"]
    return pl_module(x)


class _Predictions:
    """Logits and true classes of one pass, accumulated on the CPU."""

    def __init__(self) -> None:
        self.logits: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

    def clear(self) -> None:
        """Drop the predictions of the previous pass."""
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


def _scores(
    predictions: _Predictions, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The true classes, the probabilities and the predictions, None when nothing is plottable.

    A pass that collected nothing, and one whose scores hold NaN because the run diverged, both
    leave the figures undrawn rather than writing an empty or a blank one.
    """
    if not predictions.logits:
        log.warning(f"No {split} predictions were collected, so those plots are skipped.")
        return None
    labels, probs, preds = predictions.as_arrays()
    if np.isnan(probs).any():
        log.warning(f"The {split} scores hold NaN, so the run diverged; plots skipped.")
        return None
    return labels, probs, preds


def _log_artifacts(trainer: Trainer, paths: list[Path]) -> None:
    """Copy the figures into the MLflow run, when one of the loggers is MLflow."""
    for logger in trainer.loggers:
        if not isinstance(logger, MLFlowLogger):
            continue
        for path in paths:
            logger.experiment.log_artifact(logger.run_id, str(path), artifact_path="plots")


class PlottingCallback(Callback):
    """Write the diagnostic plots of a run under ``<default_root_dir>/plots``.

    Two families, one-vs-rest ROC curves and a confusion matrix, each for the last validation epoch
    and for the test pass. No training curves: the per-epoch metrics are the logger's.

    :param class_names: display names of the output classes, in column order.
    """

    def __init__(self, class_names: Sequence[str] = ("g", "q", "W", "Z", "t")) -> None:
        super().__init__()
        self.class_names = list(class_names)
        # eval.py sets this to the checkpoint's label ("main", "fold_1"), so that the checkpoints
        # it tests in one process do not draw over each other's figures.
        self.subdir: str | None = None
        self._val = _Predictions()
        self._test = _Predictions()

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

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Draw the plots of the last validation epoch."""
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
        scores = _scores(predictions, split)
        if scores is None:
            return
        labels, probs, preds = scores
        save_dir = self._save_dir(trainer)
        paths = [
            roc.plot(labels, probs, self.class_names, save_dir, split),
            confusion.plot(labels, preds, self.class_names, save_dir, split),
        ]
        _log_artifacts(trainer, paths)
