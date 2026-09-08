"""Measure the invariance of a trained classifier under SO(2), SO(3) and Lorentz boosts.

Four groups are probed: SO(2), SO(3), the boost along the beam (``lorentz``) and the boost of
isotropic direction (``lorentz_iso``), the last being the full-group check LorentzNet embeds.
The results table tabulates the first three, so ``lorentz_iso`` reaches MLflow alone.

The probe runs on the test set alone, reporting two families per group. The flip family is
paired, each jet against its own transform, and its label-aware breakdown says whether a flip
repaired a wrong prediction or broke a right one. The Wasserstein family is unpaired, over the
whole test set, so a model that scrambles individual jets while preserving the ensemble scores
well there and badly on the flips. Transforms come from
:func:`~src.data.components.augmentation.make_transform`, the factory the online augmentation and
the symmetry-consistency loss also draw from, so what is measured cannot drift from what was
trained on.

A second model may be attached as :attr:`InvarianceMetricCallback.baseline_module`, the same
architecture and constituent count trained without any symmetry. Both models then see the same
transformed batches, the baseline's own metrics are reported under ``baseline/``, and the
difference between the two is reported as ``delta_*``, signed so that a positive value means the
trained model does better.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from scipy.stats import wasserstein_distance

from src.data.components.augmentation import make_transform
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

GROUPS = ("so2", "so3", "lorentz", "lorentz_iso")

# The four populations a group's label transitions are counted into.
FLIP_KEYS = ("wrong_before", "recovered", "correct_before", "stable")


def _zero_counts() -> dict[str, int]:
    """A group's transition counters before any batch has been probed."""
    return dict.fromkeys(FLIP_KEYS, 0)


def _flip_rates(counts: dict[str, int]) -> dict[str, float]:
    """The three transition probabilities of one group, conditioned on being right before.

    ``flip_recover`` is P(correct after | wrong before), ``flip_stable`` is P(correct after |
    correct before) and ``flip_degrade`` its complement. NaN where the conditioning population is
    empty: a classifier that was never wrong has nothing to recover, which is not a zero rate.
    """
    wrong, recovered = counts["wrong_before"], counts["recovered"]
    correct, stable = counts["correct_before"], counts["stable"]
    nan = float("nan")
    return {
        "flip_recover": recovered / wrong if wrong else nan,
        "flip_stable": stable / correct if correct else nan,
        "flip_degrade": (correct - stable) / correct if correct else nan,
    }


def _accuracy_transformed(counts: dict[str, int]) -> float:
    """Fraction of the transformed jets a model gets right, from its own transition counts.

    Both models see the same transformed batches, so two of them compared through it are compared
    on one population.
    """
    survivors = counts["correct_before"] + counts["wrong_before"]
    correct_after = counts["stable"] + counts["recovered"]
    return correct_after / survivors if survivors else float("nan")


def _gap(results: dict[str, float], group: str, suffix: str) -> float:
    """Baseline minus tested module for a metric both reported, lower being the better score."""
    nan = float("nan")
    baseline = results.get(f"baseline/{group}_{suffix}", nan)
    return baseline - results.get(f"invariance/{group}_{suffix}", nan)


def _count_deltas(group: str, own: dict[str, int], base: dict[str, int]) -> dict[str, float]:
    """One group's deltas that follow from the transition counts alone.

    Accuracy-like rates are the tested module's own minus the baseline's, damage-like ones the
    other way round, so a positive value means the better model whichever the metric.
    """
    own_rates, base_rates = _flip_rates(own), _flip_rates(base)
    stem = f"invariance/{group}_delta_"
    return {
        f"{stem}accuracy_transformed": _accuracy_transformed(own) - _accuracy_transformed(base),
        f"{stem}flip_recover": own_rates["flip_recover"] - base_rates["flip_recover"],
        f"{stem}flip_stable": own_rates["flip_stable"] - base_rates["flip_stable"],
        f"{stem}flip_degrade": base_rates["flip_degrade"] - own_rates["flip_degrade"],
    }


def _model_names(datamodule: Any) -> list[str]:
    """The record columns the model consumes, in the order it consumes them."""
    indices = datamodule.feature_indices()
    if indices is None:
        return list(datamodule.features)
    return [datamodule.feature_names[index] for index in indices]


@dataclass
class _Accumulator:
    """One model's probe state over a test pass.

    :param prefix: the namespace its metrics are logged under, ``invariance`` for the module being
        tested and ``baseline`` for the model it is compared against.
    """

    prefix: str
    clean_logits: list[torch.Tensor] = field(default_factory=list)
    transformed_logits: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    flip_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    # Per-class counts, shape (classes,), sized from the first batch's logit width.
    clean_correct: torch.Tensor | None = None
    clean_total: torch.Tensor | None = None

    def count_clean(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add one batch's predictions on the untransformed jets to the per-class counts."""
        preds, targets = logits.argmax(dim=1).cpu(), targets.cpu()
        total = torch.bincount(targets, minlength=logits.shape[1])
        correct = torch.bincount(targets[preds == targets], minlength=logits.shape[1])
        if self.clean_total is None:
            self.clean_total = torch.zeros_like(total)
            self.clean_correct = torch.zeros_like(total)
        self.clean_total += total
        self.clean_correct += correct

    def clean_accuracy(self) -> float:
        """Accuracy on the untransformed test set, NaN before any batch has been seen.

        The mean recall over the classes present, which is the macro average torchmetrics computes
        for ``test/acc``, so that the baseline's accuracy and the delta read on the same footing as
        the module's own.
        """
        if self.clean_total is None:
            return float("nan")
        seen = self.clean_total > 0
        return float((self.clean_correct[seen] / self.clean_total[seen]).mean())


class InvarianceMetricCallback(Callback):
    """Log ``invariance/*`` metrics for every symmetry group at the end of the test pass.

    The metrics are the flip rate and its label-aware breakdown, and the Wasserstein-1 distance
    between the original and transformed logit distributions, per class and averaged over classes.
    With a baseline attached the same metrics are reported for it under ``baseline/``, together
    with the ``delta_*`` differences.

    :param transform_kwargs: the knobs of :func:`~src.data.components.augmentation.make_transform`
        the data pipeline used, so the probe measures the trained regime; they define the probe
        and are never tuned.
    :param compute_wasserstein: keeps every test logit in host memory until the end of the pass.
    :param class_names: display names of the output classes, in column order.
    """

    def __init__(
        self,
        transform_kwargs: dict[str, float | bool] | None = None,
        compute_wasserstein: bool = True,
        compute_flip_breakdown: bool = True,
        class_names: Sequence[str] = ("g", "q", "W", "Z", "t"),
    ) -> None:
        super().__init__()
        self.transform_kwargs = dict(transform_kwargs or {})
        self.compute_wasserstein = compute_wasserstein
        self.compute_flip_breakdown = compute_flip_breakdown
        self.class_names = list(class_names)
        # Set after construction, by src.eval when a baseline manifest names this fold.
        self.baseline_module: LightningModule | None = None
        self._reset()

    def _reset(self) -> None:
        """Drop the state of the previous test pass; a fold reuses the same callback."""
        self._model_feature_names: list[str] | None = None
        self._full_feature_names: list[str] | None = None
        self._norm_scales: dict[str, float] | None = None
        self._batch_metrics: dict[str, list[float]] = {}
        self._own = _Accumulator("invariance")
        self._baseline = _Accumulator("baseline")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Read the feature layout from the datamodule; without it the probe stays off."""
        self._reset()
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is None or datamodule.feature_names is None:
            log.warning("The datamodule names no columns, so px/py/pz cannot be found; skipped.")
            return
        self._resolve_features(datamodule)
        if self.baseline_module is not None:
            self._attach_baseline(trainer, pl_module)
        log.info(f"Invariance probe ready on features {self._model_feature_names}.")

    def _resolve_features(self, datamodule: Any) -> None:
        """Name the columns of both widths of batch the probe may be handed."""
        self._full_feature_names = list(datamodule.feature_names)
        self._model_feature_names = _model_names(datamodule)
        self._norm_scales = datamodule.norm_scales

    def _attach_baseline(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Put the baseline on the tested module's device with its feature layout resolved.

        The baseline never passes through ``trainer.test``, so Lightning calls none of its hooks:
        without ``setup("test")`` its ``model_feature_indices`` stays None and ``forward`` would
        hand a full 16-column batch to a network sized for the reduced feature set.
        """
        if not self.compute_flip_breakdown:
            log.warning(
                "A baseline is attached and every delta is built from the transition counts, "
                "so compute_flip_breakdown is forced on."
            )
            self.compute_flip_breakdown = True
        self.baseline_module.trainer = trainer
        self.baseline_module.setup("test")
        self.baseline_module.to(pl_module.device)
        self.baseline_module.eval()

    def _names_for(self, width: int) -> list[str]:
        """Name the columns of a batch of this width, for the transforms to address by name.

        A test batch carries either the record's 16 columns (under ``keep_full_features_at_test``,
        or always under the SCL) or the reduced set the network consumes.
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
        """Transform one test batch under each group and record how the logits move.

        Every probed model is handed the same transformed batch, so the comparison between them is
        paired jet by jet as well as group by group.
        """
        if self._model_feature_names is None:
            return
        x, y = batch  # x: (jets, constituents, features), normalised, on the module's device
        targets = y.argmax(dim=1) if y.ndim > 1 and y.shape[1] > 1 else y
        probes = self._probes_for_batch(pl_module, outputs, x)
        for accumulator, _, clean in probes:
            self._record_clean(accumulator, clean, targets)
        for group in GROUPS:
            self._probe_group(group, x, targets, probes)

    def _probes_for_batch(
        self, pl_module: LightningModule, outputs: Any, x: torch.Tensor
    ) -> list[tuple[_Accumulator, LightningModule, torch.Tensor]]:
        """Pair every probed model with its accumulator and its logits on the plain batch.

        JetTaggingModule.test_step hands its model_step dict over, so the tested module's own
        logits come free; a module whose step returns nothing pays one forward pass here.
        """
        logits = outputs.get("logits") if isinstance(outputs, dict) else None
        if logits is None:
            logits = pl_module(x)
        probes = [(self._own, pl_module, logits)]
        if self.baseline_module is not None:
            probes.append((self._baseline, self.baseline_module, self.baseline_module(x)))
        return probes

    def _record_clean(
        self, accumulator: _Accumulator, logits: torch.Tensor, targets: torch.Tensor
    ) -> None:
        """Keep one batch's untransformed logits and add it to the clean accuracy counts."""
        if self.compute_wasserstein:
            accumulator.clean_logits.append(logits.detach().cpu())
        accumulator.count_clean(logits, targets)

    def _probe_group(
        self,
        group: str,
        x: torch.Tensor,
        targets: torch.Tensor,
        probes: list[tuple[_Accumulator, LightningModule, torch.Tensor]],
    ) -> None:
        """Compare every model's logits on a batch with its logits on the batch's transform."""
        names = self._names_for(x.shape[-1])
        x_sym = make_transform(group, names, self._norm_scales, **self.transform_kwargs)(x)
        for accumulator, module, clean in probes:
            self._compare(accumulator, group, clean, module(x_sym), targets)

    def _compare(
        self,
        accumulator: _Accumulator,
        group: str,
        logits: torch.Tensor,
        logits_sym: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Record one model's flip rate, transition counts and transformed logits for one group."""
        flipped = logits.argmax(dim=1) != logits_sym.argmax(dim=1)
        self._record(f"{group}_flip", flipped.float().mean().item(), accumulator.prefix)
        if self.compute_flip_breakdown:
            self._count_flips(accumulator, group, logits, logits_sym, targets)
        if self.compute_wasserstein:
            accumulator.transformed_logits.setdefault(group, []).append(logits_sym.detach().cpu())

    def _record(self, name: str, value: float, prefix: str = "invariance") -> None:
        """Add one batch's value to the series that is averaged at the end of the epoch."""
        self._batch_metrics.setdefault(f"{prefix}/{name}", []).append(value)

    def _count_flips(
        self,
        accumulator: _Accumulator,
        group: str,
        logits: torch.Tensor,
        logits_sym: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate one model's label-aware transition counts for one batch."""
        correct_before = logits.argmax(dim=1) == targets
        correct_after = logits_sym.argmax(dim=1) == targets
        counts = accumulator.flip_counts.setdefault(group, _zero_counts())
        counts["wrong_before"] += int((~correct_before).sum().item())
        counts["recovered"] += int((~correct_before & correct_after).sum().item())
        counts["correct_before"] += int(correct_before.sum().item())
        counts["stable"] += int((correct_before & correct_after).sum().item())

    def _wasserstein_metrics(self, accumulator: _Accumulator) -> dict[str, float]:
        """Distance between one model's original and transformed logit distributions per group.

        Computed per class-logit channel over the whole test set and averaged over the channels.
        The comparison is unpaired, so it says nothing about which jet produced which value.
        """
        if not accumulator.clean_logits:
            return {}
        original = torch.cat(accumulator.clean_logits, dim=0).numpy()  # (jets, classes)
        results: dict[str, float] = {}
        for group, chunks in accumulator.transformed_logits.items():
            moved = torch.cat(chunks, dim=0).numpy()  # (jets, classes)
            results |= self._wasserstein_for_group(accumulator.prefix, group, original, moved)
        return results

    def _wasserstein_for_group(
        self, prefix: str, group: str, original: np.ndarray, transformed: np.ndarray
    ) -> dict[str, float]:
        """One group's per-class distances and their mean over the class-logit channels."""
        names = self.class_names[: original.shape[1]]
        per_class = [
            wasserstein_distance(original[:, c], transformed[:, c]) for c in range(len(names))
        ]
        results = {f"{prefix}/{group}_wasserstein_{n}": v for n, v in zip(names, per_class)}
        results[f"{prefix}/{group}_wasserstein"] = float(np.mean(per_class))
        return results

    def _flip_breakdown_metrics(self, accumulator: _Accumulator) -> dict[str, float]:
        """Turn one model's transition counts into probabilities conditioned on being right."""
        results: dict[str, float] = {}
        for group, counts in accumulator.flip_counts.items():
            rates = _flip_rates(counts)
            results |= {f"{accumulator.prefix}/{group}_{k}": v for k, v in rates.items()}
        return results

    def _accumulators(self) -> list[_Accumulator]:
        """The models probed: the one being tested, and the baseline when one is attached."""
        return [self._own] if self.baseline_module is None else [self._own, self._baseline]

    def _pass_metrics(self) -> dict[str, float]:
        """Everything each probed model reports once the whole test set has gone through."""
        results: dict[str, float] = {}
        for accumulator in self._accumulators():
            if self.compute_wasserstein:
                results |= self._wasserstein_metrics(accumulator)
            if self.compute_flip_breakdown:
                results |= self._flip_breakdown_metrics(accumulator)
        return results

    def _delta_metrics(self, results: dict[str, float]) -> dict[str, float]:
        """The tested module minus its baseline, signed so that positive is the better model.

        Accuracy-like quantities are the module's own minus the baseline's, distance-like and
        damage-like ones the baseline's minus the module's own, so every delta is read the same
        way.

        :param results: the metrics of both models, already averaged over the test set.
        """
        deltas = {
            "baseline/test_acc": self._baseline.clean_accuracy(),
            "delta_test_acc": self._own.clean_accuracy() - self._baseline.clean_accuracy(),
        }
        for group in GROUPS:
            deltas |= self._group_deltas(group, results)
        return deltas

    def _group_deltas(self, group: str, results: dict[str, float]) -> dict[str, float]:
        """One group's deltas, from the two models' counts and their averaged metrics."""
        own = self._own.flip_counts.get(group, _zero_counts())
        base = self._baseline.flip_counts.get(group, _zero_counts())
        deltas = _count_deltas(group, own, base)
        deltas[f"invariance/{group}_delta_flip"] = _gap(results, group, "flip")
        if self.compute_wasserstein:
            deltas[f"invariance/{group}_delta_wasserstein"] = _gap(results, group, "wasserstein")
        return deltas

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Average the per-batch series, add the pass-level metrics, and log them all."""
        if not self._batch_metrics:
            return
        results = {name: float(np.mean(values)) for name, values in self._batch_metrics.items()}
        results |= self._pass_metrics()
        if self.baseline_module is not None:
            results |= self._delta_metrics(results)
        for name, value in results.items():
            pl_module.log(name, value, prog_bar=False, on_epoch=True, sync_dist=True)
        log.info(" | ".join(f"{name}: {value:.4f}" for name, value in results.items()))
