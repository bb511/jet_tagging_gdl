# Invariance probe: how far the logits move when a symmetry acts on the test jets.
"""Measure the invariance of a trained classifier under SO(2), SO(3) and Lorentz boosts.

The probe runs on the test set only, and reports two families per group. The flip family
is paired: it compares the prediction on a jet with the prediction on that same jet
transformed, and its label-aware breakdown says whether a flip repaired a wrong
prediction or broke a right one. The Wasserstein family is unpaired: it compares the two
logit distributions over the whole test set, so a model that scrambles individual jets
while preserving the ensemble scores well there and badly on the flips.

Transforms come from :func:`~src.data.components.augmentation.make_transform`, the same
factory the online augmentation and the symmetry-consistency loss draw from, so what is
measured cannot drift from what was trained on.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from scipy.stats import wasserstein_distance

from src.data.components.augmentation import Transform, make_transform
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

GROUPS = ("so2", "so3", "lorentz")


class InvarianceMetricCallback(Callback):
    """Log ``invariance/*`` metrics for every symmetry group at the end of the test pass.

    The metrics are the flip rate, the fraction of jets the SO(3) acceptance filter rejects, the
    Wasserstein-1 distance between the original and transformed logit distributions (per class and
    averaged over classes), and the label-aware flip breakdown.

    :param lorentz_eta_max: half-width of the rapidity range the boosts are drawn from; it defines
        the probe, so it is never tuned.
    :param compute_wasserstein: keep every test logit in host memory and compare the two
        distributions at the end of the pass.
    :param compute_flip_breakdown: count the label-aware transitions as well as the flips.
    :param class_names: display names of the output classes, in column order.
    """

    def __init__(
        self,
        lorentz_eta_max: float = 2.0,
        compute_wasserstein: bool = True,
        compute_flip_breakdown: bool = True,
        class_names: Sequence[str] = ("g", "q", "W", "Z", "t"),
    ) -> None:
        super().__init__()
        self.lorentz_eta_max = lorentz_eta_max
        self.compute_wasserstein = compute_wasserstein
        self.compute_flip_breakdown = compute_flip_breakdown
        self.class_names = list(class_names)
        self._reset()

    def _reset(self) -> None:
        """Drop the state of the previous test pass; a fold reuses the same callback."""
        self._model_feature_names: list[str] | None = None
        self._full_feature_names: list[str] | None = None
        self._norm_scales: dict[str, float] | None = None
        self._batch_metrics: dict[str, list[float]] = {}
        self._orig_logits: list[torch.Tensor] = []
        self._aug_logits: dict[str, list[torch.Tensor]] = {}
        self._flip_counts: dict[str, dict[str, int]] = {}

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Read the feature layout from the datamodule; without it the probe stays off."""
        self._reset()
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or datamodule.feature_names is None:
            log.warning(
                "The datamodule does not name its columns, so px/py/pz cannot be found "
                "and the invariance probe is skipped."
            )
            return
        self._full_feature_names = list(datamodule.feature_names)
        indices = datamodule.feature_indices()
        self._model_feature_names = (
            [self._full_feature_names[i] for i in indices]
            if indices is not None
            else list(datamodule.features)
        )
        self._norm_scales = datamodule.norm_scales
        log.info(f"Invariance probe ready on features {self._model_feature_names}.")

    def _names_for(self, width: int) -> list[str]:
        """Name the columns of a batch of this width.

        A test batch carries either the record's 16 columns (under ``keep_full_features_at_test``,
        or always under the SCL) or the reduced set the network consumes, and the transforms
        address px, py and pz by name.
        """
        if self._full_feature_names is not None and width == len(self._full_feature_names):
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
        x, y = batch  # x: (jets, constituents, features), normalised, on the module's device
        targets = y.argmax(dim=1) if y.ndim > 1 and y.shape[1] > 1 else y
        # JetTaggingModule.test_step hands its model_step dict over, so the untransformed
        # logits come free; a module whose step returns nothing pays one forward pass here.
        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if logits is None:
            logits = pl_module(x)
        if self.compute_wasserstein:
            self._orig_logits.append(logits.detach().cpu())
        names = self._names_for(x.shape[-1])
        for group in GROUPS:
            transform = make_transform(group, names, self._norm_scales, self.lorentz_eta_max)
            self._probe_group(pl_module, group, x, logits, targets, transform)

    def _probe_group(
        self,
        pl_module: LightningModule,
        group: str,
        x: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        transform: Transform,
    ) -> None:
        """Compare the logits on a batch with the logits on its transform, for one group."""
        x_sym, keep_mask = transform(x)
        if keep_mask is not None:
            self._record(f"{group}_reject_frac", 1.0 - keep_mask.float().mean().item())
            if not bool(keep_mask.any()):
                # The SO(3) filter rejected the whole batch, leaving nothing to pair; a
                # mean over the empty tensor would enter the epoch average as NaN.
                return
        # SO(3) drops the jets it rotates out of acceptance, so the survivors' logits are
        # what the transformed batch must be compared against.
        kept = logits if keep_mask is None else logits[keep_mask]
        kept_targets = targets if keep_mask is None else targets[keep_mask]
        logits_sym = pl_module(x_sym)
        flipped = kept.argmax(dim=1) != logits_sym.argmax(dim=1)
        self._record(f"{group}_flip", flipped.float().mean().item())
        if self.compute_flip_breakdown:
            self._count_flips(group, kept, logits_sym, kept_targets)
        if self.compute_wasserstein:
            self._aug_logits.setdefault(group, []).append(logits_sym.detach().cpu())

    def _record(self, name: str, value: float) -> None:
        """Add one batch's value to the series that is averaged at the end of the epoch."""
        self._batch_metrics.setdefault(f"invariance/{name}", []).append(value)

    def _count_flips(
        self, group: str, logits: torch.Tensor, logits_sym: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Accumulate the label-aware transition counts of one batch."""
        correct_before = logits.argmax(dim=1) == targets
        correct_after = logits_sym.argmax(dim=1) == targets
        counts = self._flip_counts.setdefault(
            group, {"wrong_before": 0, "recovered": 0, "correct_before": 0, "stable": 0}
        )
        counts["wrong_before"] += int((~correct_before).sum().item())
        counts["recovered"] += int((~correct_before & correct_after).sum().item())
        counts["correct_before"] += int(correct_before.sum().item())
        counts["stable"] += int((correct_before & correct_after).sum().item())

    def _wasserstein_metrics(self) -> dict[str, float]:
        """Distance between the original and transformed logit distributions per group.

        Computed per class-logit channel over the whole test set and then averaged over the
        channels. The comparison is unpaired, so it says nothing about which jet produced which
        value.
        """
        if not self._orig_logits:
            return {}
        original = torch.cat(self._orig_logits, dim=0).numpy()  # (jets, classes)
        names = self.class_names[: original.shape[1]]
        results: dict[str, float] = {}
        for group, chunks in self._aug_logits.items():
            transformed = torch.cat(chunks, dim=0).numpy()
            per_class = [
                wasserstein_distance(original[:, c], transformed[:, c]) for c in range(len(names))
            ]
            for name, value in zip(names, per_class):
                results[f"invariance/{group}_wasserstein_{name}"] = value
            results[f"invariance/{group}_wasserstein"] = float(np.mean(per_class))
        return results

    def _flip_breakdown_metrics(self) -> dict[str, float]:
        """Turn the transition counts into probabilities conditioned on being right.

        ``recover`` is P(correct after | wrong before), ``stable`` is P(correct after | correct
        before) and ``degrade`` its complement. Unlike the plain flip rate, these distinguish a
        transformation that repairs a prediction from one that breaks it.
        """
        results: dict[str, float] = {}
        for group, counts in self._flip_counts.items():
            wrong, recovered = counts["wrong_before"], counts["recovered"]
            correct, stable = counts["correct_before"], counts["stable"]
            nan = float("nan")
            results[f"invariance/{group}_flip_recover"] = recovered / wrong if wrong else nan
            results[f"invariance/{group}_flip_stable"] = stable / correct if correct else nan
            results[f"invariance/{group}_flip_degrade"] = (
                (correct - stable) / correct if correct else nan
            )
        return results

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Average the per-batch series, add the pass-level metrics, and log them all."""
        if not self._batch_metrics:
            return
        results = {name: float(np.mean(values)) for name, values in self._batch_metrics.items()}
        if self.compute_wasserstein:
            results.update(self._wasserstein_metrics())
        if self.compute_flip_breakdown:
            results.update(self._flip_breakdown_metrics())
        for name, value in results.items():
            pl_module.log(name, value, prog_bar=False, on_epoch=True, sync_dist=True)
        log.info(" | ".join(f"{name}: {value:.4f}" for name, value in results.items()))
