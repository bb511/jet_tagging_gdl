"""Measure the invariance of a trained classifier under various symmetry groups."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from scipy.stats import wasserstein_distance

from src.data.components.augmentation import make_transform
from src.symmetry import (
    ACCURACY_TRANSFORMED,
    FLIP,
    FLIP_RECOVER,
    FLIP_STABLE,
    GROUPS,
    WASSERSTEIN,
    key,
)
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@dataclass
class FlipCounts:
    """The four categories that the labels are counted into, for one group."""

    # wrong_before: wrongly assigned jets before the symmetry transformation
    # recovered: jets that were wrongly assigned before but are correct after transformation
    # correct_before: jets that were correct before the transformation
    # stable: jets that did not change how they're assigned after the transformation
    wrong_before: int = 0
    recovered: int = 0
    correct_before: int = 0
    stable: int = 0


class FlipProbe:
    """Compute how often a group's transform moves the predicted class."""

    def __init__(self) -> None:
        # Per group, how many jets changed their predicted class under the transform.
        self.flips = dict.fromkeys(GROUPS, 0)
        self.counts = {group: FlipCounts() for group in GROUPS}

    @property
    def n_jets(self) -> int:
        # Total number of jets.
        counts = self.counts[GROUPS[0]]
        return counts.correct_before + counts.wrong_before

    def update(
        self,
        group: str,
        logits: torch.Tensor,
        logits_sym: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Add one batch's transitions under *group* to that group's counters."""
        predicted, predicted_sym = logits.argmax(dim=1), logits_sym.argmax(dim=1)
        correct, correct_sym = predicted == targets, predicted_sym == targets
        counts = self.counts[group]
        self.flips[group] += int((predicted != predicted_sym).sum().item())
        counts.wrong_before += int((~correct).sum().item())
        counts.recovered += int((~correct & correct_sym).sum().item())
        counts.correct_before += int(correct.sum().item())
        counts.stable += int((correct & correct_sym).sum().item())

    def metrics(self) -> dict[str, float]:
        """Group flip rates and the probabilities conditioned on being right before."""
        results: dict[str, float] = {}
        for group, counts in self.counts.items():
            seen = counts.correct_before + counts.wrong_before
            rate = self.flips[group] / seen if seen else float("nan")
            results[key(group, FLIP)] = rate
            results |= {key(group, family): v for family, v in _rates(counts).items()}
        return results


class WassersteinProbe:
    """Compute the Wasserstein distance between two sets of logits.

    This is used to see how different the predictions are when passing the basic data through a
    model and then the augmented data.
    """

    def __init__(self) -> None:
        self.clean: list[torch.Tensor] = []
        self.transformed: dict[str, list[torch.Tensor]] = {}

    def add_clean(self, logits: torch.Tensor) -> None:
        self.clean.append(logits.detach().cpu())

    def add_transformed(self, group: str, logits: torch.Tensor) -> None:
        self.transformed.setdefault(group, []).append(logits.detach().cpu())

    def metrics(self) -> dict[str, float]:
        """Distance between the clean and transformed logit distributions, per group."""
        if not self.clean:
            return {}
        clean = torch.cat(self.clean, dim=0).numpy()  # (jets, classes)
        return {
            key(group, WASSERSTEIN): _mean_channel_distance(
                clean, torch.cat(chunks, dim=0).numpy()
            )
            for group, chunks in self.transformed.items()
        }


class InvarianceMetricCallback(Callback):
    """Log ``invariance/*`` metrics for every symmetry group at the end of the test pass.

    The metrics are the flip rate with its label-aware breakdown,
    the accuracy after the transform, and the Wasserstein-1 distance between the clean
    and transformed logit distributions averaged over the class-logit channels.

    :param transform_kwargs: they hyperparameters that a ``make_transform`` in the data
        module would take.
    """

    def __init__(self, transform_kwargs: dict[str, float | bool] | None = None) -> None:
        super().__init__()
        self.transform_kwargs = dict(transform_kwargs or {})
        self._reset()

    def _reset(self) -> None:
        """Drop the state of the previous test pass; a fold reuses the same callback."""
        self._model_feature_names: list[str] | None = None
        self._full_feature_names: list[str] | None = None
        self._norm_scales: dict[str, float] | None = None
        self._flip = FlipProbe()
        self._wasserstein = WassersteinProbe()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Read the feature layout from the data; the probe stays off otherwise."""
        self._reset()
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or datamodule.feature_names is None:
            log.warning("The datamodule has no features; skipped.")
            return
        self._full_feature_names = list(datamodule.feature_names)
        self._model_feature_names = _model_names(datamodule)
        self._norm_scales = datamodule.norm_scales
        log.info(f"Invariance probe ready on features {self._model_feature_names}.")

    def _names_for(self, width: int) -> list[str]:
        """Collect the first *width* features of the data.

        A test batch has either all 16 columns (under ``keep_full_features_at_test``,
        or always under the SCL) or the reduced set that the network consumes.
        """
        if self._full_feature_names is not None and width == len(
            self._full_feature_names
        ):
            return self._full_feature_names
        return self._model_feature_names

    @torch.no_grad()
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Transform one test batch under each group and record how the logits move."""
        if self._model_feature_names is None:
            return
        x, y = batch
        targets = y.argmax(dim=1) if y.ndim > 1 and y.shape[1] > 1 else y
        # The clean (unaugmented data) logits are recomputed here due to precision:
        # the trainer can have 16-mixed precision, but here we enforce fp32.
        clean = pl_module(x)
        self._wasserstein.add_clean(clean)
        for group in GROUPS:
            self._probe_group(pl_module, group, x, clean, targets)

    def _probe_group(
        self,
        pl_module: LightningModule,
        group: str,
        x: torch.Tensor,
        clean: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """One draw per jet under *group*, compared with the same untranformed jets."""
        transform = make_transform(
            group,
            self._names_for(x.shape[-1]),
            self._norm_scales,
            **self.transform_kwargs,
        )
        logits_sym = pl_module(transform(x))
        self._flip.update(group, clean, logits_sym, targets)
        self._wasserstein.add_transformed(group, logits_sym)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reduce the counts and the kept logits over the whole test set, log all."""
        if self._flip.n_jets == 0:
            return
        results = self._flip.metrics() | self._wasserstein.metrics()
        for name, value in results.items():
            pl_module.log(name, value, prog_bar=False, on_epoch=True, sync_dist=True)
        log.info(" | ".join(f"{name}: {value:.4f}" for name, value in results.items()))


def _rates(counts: FlipCounts) -> dict[str, float]:
    """A group's transition probabilities, NaN when nothing is counted."""
    wrong, recovered = counts.wrong_before, counts.recovered
    correct, stable = counts.correct_before, counts.stable
    seen, nan = correct + wrong, float("nan")
    return {
        FLIP_RECOVER: recovered / wrong if wrong else nan,
        FLIP_STABLE: stable / correct if correct else nan,
        ACCURACY_TRANSFORMED: (stable + recovered) / seen if seen else nan,
    }


def _mean_channel_distance(clean: np.ndarray, transformed: np.ndarray) -> float:
    """Wasserstein-1 per class-logit over the whole test set, averaged over channels.

    The comparison is unpaired.
    """
    channels = range(clean.shape[1])
    return float(
        np.mean(
            [wasserstein_distance(clean[:, c], transformed[:, c]) for c in channels]
        )
    )


def _model_names(datamodule: Any) -> list[str]:
    """The record features the model consumes, in the order that it consumes them."""
    indices = datamodule.feature_indices()
    if indices is None:
        return list(datamodule.features)
    return [datamodule.feature_names[index] for index in indices]
