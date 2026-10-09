"""Draw the per-run plots and attach them to MLflow when MLflow is logging."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import MLFlowLogger

from src.plot import confusion, roc
from src.symmetry import CLASS_NAMES
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)


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
        """Accumulate one batch; labels arrive one-hot and are stored as class idxs."""
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.argmax(dim=1).detach().cpu())

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the true classes, the class probabilities and the predictions."""
        probs = torch.softmax(torch.cat(self.logits), dim=1).numpy()
        return torch.cat(self.labels).numpy(), probs, probs.argmax(axis=1)


def _scores(
    predictions: _Predictions,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The true classes, the probabilities and the predictions."""
    if not predictions.logits:
        log.warning("No test predictions were collected; plots are skipped...")
        return None

    labels, probs, preds = predictions.as_arrays()
    if np.isnan(probs).any():
        log.warning("The test scores are NaN; plots skipped...")
        return None

    return labels, probs, preds


def _log_artifacts(trainer: Trainer, paths: list[Path]) -> None:
    """Copy the figures into the MLflow run, when one of the loggers is MLflow."""
    for logger in trainer.loggers:
        if not isinstance(logger, MLFlowLogger):
            continue
        for path in paths:
            logger.experiment.log_artifact(
                logger.run_id, str(path), artifact_path="plots"
            )


class PlottingCallback(Callback):
    """Write the one-vs-rest ROC curves and the confusion matrix of the test pass.

    The figures go under ``<default_root_dir>/plots``, the classes named by
    `src.symmetry.CLASS_NAMES`.
    """

    def __init__(self) -> None:
        super().__init__()
        # eval.py sets this to the checkpoint's label ("main", "fold_1").
        self.subdir: str | None = None
        self._test = _Predictions()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Start the test pass empty."""
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
        _, y = batch
        self._test.add(outputs["logits"], y)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Draw the ROC curves and the confusion matrix of the test pass."""
        scores = _scores(self._test)
        if scores is None:
            return
        labels, probs, preds = scores
        save_dir = self._save_dir(trainer)
        paths = [
            roc.plot(labels, probs, CLASS_NAMES, save_dir),
            confusion.plot(labels, preds, CLASS_NAMES, save_dir),
        ]
        _log_artifacts(trainer, paths)

    def _save_dir(self, trainer: Trainer) -> Path:
        """Plot directory of this run, beside its checkpoints under the Hydra output."""
        plots = Path(trainer.default_root_dir) / "plots"
        return plots if self.subdir is None else plots / self.subdir
