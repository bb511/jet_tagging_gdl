"""Measure the invariance of a trained classifier under various symmetry groups."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer
from sklearn.metrics import roc_auc_score

from src.data.components.augmentation import make_drawing_transform
from src.symmetry import (
    ACCURACY_TRANSFORMED,
    AUC,
    CLASS_NAMES,
    FLIP,
    FLIP_RECOVER,
    FLIP_STABLE,
    GROUPS,
    TOTAL_VARIATION,
    WORKING_POINTS,
    WorkingPoint,
    class_names_of,
    clean_family,
    diagnostic_family,
    key,
    retention_family,
)
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

# Per class, the background and the signal jets of one score set: (classes,) each.
Counts = tuple[np.ndarray, np.ndarray]
# A batch to the transformed batch and the draws behind it, one per jet.
DrawingTransform = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


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

    def rates(self) -> dict[str, float]:
        """The group's transition probabilities, NaN when nothing is counted."""
        wrong, recovered = self.wrong_before, self.recovered
        correct, stable = self.correct_before, self.stable
        seen, nan = correct + wrong, float("nan")
        return {
            FLIP_RECOVER: recovered / wrong if wrong else nan,
            FLIP_STABLE: stable / correct if correct else nan,
            ACCURACY_TRANSFORMED: (stable + recovered) / seen if seen else nan,
        }


class FlipProbe:
    """Compute how often a group's transform moves the predicted class."""

    def __init__(self) -> None:
        # Per group, how many jets changed their predicted class under the transform.
        self.flips = dict.fromkeys(GROUPS, 0)
        self.counts = {group: FlipCounts() for group in GROUPS}

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
            results |= {key(group, family): v for family, v in counts.rates().items()}
        return results


class OutputStore:
    """Keep the per-jet outputs of one test pass on the CPU, in loader order.

    The logits and the draws are kept in float32 and the labels in int8, the layout of
    the ``.npz`` the callback writes.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, list[np.ndarray]] = {}

    def add(
        self,
        clean: torch.Tensor,
        targets: torch.Tensor,
        transformed: dict[str, torch.Tensor],
        draws: dict[str, torch.Tensor],
    ) -> None:
        """Keep one batch: its clean logits, its labels, and per group the transformed
        logits and the draws behind them."""
        batch = {"logits_clean": clean, "labels": targets}
        batch |= {f"logits_{group}": logits for group, logits in transformed.items()}
        batch |= {f"draw_{group}": draw for group, draw in draws.items()}
        for name, values in batch.items():
            dtype = np.int8 if name == "labels" else np.float32
            array = values.detach().cpu().numpy().astype(dtype)
            self.chunks.setdefault(name, []).append(array)

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: np.concatenate(chunks) for name, chunks in self.chunks.items()}


def _softmax(logits: np.ndarray) -> np.ndarray:
    """The class probabilities in float64, shape (jets, classes)."""
    # Shifted by the row maximum, which keeps exp finite and the softmax unchanged.
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return shifted / shifted.sum(axis=1, keepdims=True)


def one_vs_rest(logits: np.ndarray) -> np.ndarray:
    """Per class c, the log-odds z_c - logsumexp_{k != c} z_k in float64, shape (jets,
    classes): the scores `clean_metrics` reads its working points and AUC off.

    They order the jets as the softmax column of c does, without the ties the softmax
    saturates into at float precision.
    """
    logits = logits.astype(np.float64)
    classes = range(logits.shape[1])
    rest = [np.logaddexp.reduce(np.delete(logits, c, axis=1), axis=1) for c in classes]
    return logits - np.stack(rest, axis=1)


def _quantile(values: np.ndarray, level: float) -> float:
    """The quantile *level* of *values*, NaN for an empty set."""
    # "higher" picks a stored score, so the thresholds hold no interpolated value.
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, level, method="higher"))


def _thresholds(
    scores: np.ndarray, signal: np.ndarray, point: WorkingPoint
) -> np.ndarray:
    """Per class, the clean score above which the efficiency *point* fixes is at most
    its level: the quantile 1 - level of the signal or of the background scores.

    NaN for a class with no jet in that set.
    """
    fixed = signal if point.fixed == "signal" else ~signal
    level = 1.0 - point.level
    columns = range(scores.shape[1])
    return np.array([_quantile(scores[fixed[:, c], c], level) for c in columns])


def _passing(scores: np.ndarray, signal: np.ndarray, thresholds: np.ndarray) -> Counts:
    """Per class, the background and the signal jets strictly above the threshold."""
    above = scores > thresholds
    return (above & ~signal).sum(axis=0), (above & signal).sum(axis=0)


def _clean_value(point: WorkingPoint, clean: Counts, totals: Counts) -> np.ndarray:
    """Per class, the rejection 1/eps_B or the TPR at *point* on the clean scores."""
    (background, signal), (n_background, n_signal) = clean, totals
    if point.fixed == "signal":
        # Floored at one passing background jet, so that a point nothing passes reads
        # n_B, a finite lower bound, rather than infinity.
        return n_background / np.maximum(background, 1)
    return signal / n_signal


def _retention(point: WorkingPoint, clean: Counts, transformed: Counts) -> np.ndarray:
    """Per class, the transformed rejection over the clean one, or the transformed TPR
    over the clean one, at the clean threshold; both are floored as the rejection is."""
    (background, signal), (background_g, signal_g) = clean, transformed
    if point.fixed == "signal":
        return np.maximum(background, 1) / np.maximum(background_g, 1)
    return signal_g / np.maximum(signal, 1)


def _diagnostic(point: WorkingPoint, transformed: Counts, totals: Counts) -> np.ndarray:
    """Per class, the fixed efficiency, read on the transformed scores."""
    (background_g, signal_g), (n_background, n_signal) = transformed, totals
    if point.fixed == "signal":
        return signal_g / n_signal
    return background_g / n_background


def _class_auc(signal_c: np.ndarray, score_c: np.ndarray) -> float:
    """The one-vs-rest AUC of one class, NaN unless it and the rest both occur."""
    if signal_c.all() or not signal_c.any():
        return float("nan")
    return float(roc_auc_score(signal_c, score_c))


def _macro(
    group: str | None, family: str, values: np.ndarray, class_names: Sequence[str]
) -> dict[str, float]:
    """The macro mean of a per-class family, then one key per class."""
    per_class = zip(class_names, values, strict=True)
    results = {key(group, family): float(np.mean(values))}
    return results | {key(group, family, name): float(v) for name, v in per_class}


@dataclass(frozen=True)
class FrozenPoint:
    """One working point read on the clean scores: its per-class thresholds and the
    background and signal jets above them."""

    point: WorkingPoint
    thresholds: np.ndarray
    counts: Counts


def _frozen_points(scores: np.ndarray, signal: np.ndarray) -> list[FrozenPoint]:
    frozen = []
    for point in WORKING_POINTS:
        thresholds = _thresholds(scores, signal, point)
        counts = _passing(scores, signal, thresholds)
        frozen.append(FrozenPoint(point, thresholds, counts))
    return frozen


def _totals(signal: np.ndarray) -> Counts:
    return (~signal).sum(axis=0), signal.sum(axis=0)


def _clean_metrics(
    scores: np.ndarray,
    signal: np.ndarray,
    points: list[FrozenPoint],
    totals: Counts,
    class_names: Sequence[str],
) -> dict[str, float]:
    """The groupless families: each point's clean rejection or TPR, and the AUC."""
    results: dict[str, float] = {}
    for frozen in points:
        values = _clean_value(frozen.point, frozen.counts, totals)
        results |= _macro(None, clean_family(frozen.point), values, class_names)
    columns = range(scores.shape[1])
    auc = [_class_auc(signal[:, c], scores[:, c]) for c in columns]
    return results | _macro(None, AUC, np.array(auc), class_names)


def _finite(logits: np.ndarray, name: str) -> np.ndarray | None:
    """Return *logits* in float64, or None with a warning if one is not finite."""
    logits = logits.astype(np.float64)
    if np.isfinite(logits).all():
        return logits
    log.warning(f"Non-finite {name} logits: the metrics read off them are skipped.")
    return None


def _clean_reading(
    clean: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[FrozenPoint], Counts]:
    """What the groupless and the per-group metrics share, read off the float64 clean
    logits: the one-vs-rest scores, the signal mask (jets, classes), the working points
    frozen on the scores, and the per-class totals."""
    signal = labels[:, None] == np.arange(clean.shape[1])
    scores = one_vs_rest(clean)
    return scores, signal, _frozen_points(scores, signal), _totals(signal)


def clean_metrics(
    logits: np.ndarray, labels: np.ndarray, class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, float]:
    """The groupless ``performance/`` keys of one set of logits, as `output_metrics`
    reads them off the clean ones; empty if a logit is not finite.

    :param logits: shape (jets, classes), the columns in the order of *class_names*.
    :param labels: the class index of each jet, shape (jets,).
    :param class_names: the suffixes of the per-class keys.
    """
    clean = _finite(logits, "clean")
    if clean is None:
        return {}
    return _clean_metrics(*_clean_reading(clean, labels), class_names)


def _point_metrics(
    group: str,
    frozen: FrozenPoint,
    transformed: Counts,
    totals: Counts,
    class_names: Sequence[str],
) -> dict[str, float]:
    """The retention and the diagnostic of one working point under *group*."""
    point = frozen.point
    retention = _retention(point, frozen.counts, transformed)
    diagnostic = _diagnostic(point, transformed, totals)
    results = _macro(group, retention_family(point), retention, class_names)
    return results | _macro(group, diagnostic_family(point), diagnostic, class_names)


def _group_metrics(
    group: str,
    p_clean: np.ndarray,
    logits: np.ndarray,
    signal: np.ndarray,
    points: list[FrozenPoint],
    totals: Counts,
    class_names: Sequence[str],
) -> dict[str, float]:
    """The TV under *group* and, per working point, its retention and diagnostic."""
    logits = _finite(logits, group)
    if logits is None:
        return {}
    scores = one_vs_rest(logits)
    tv = 0.5 * np.abs(p_clean - _softmax(logits)).sum(axis=1).mean()
    results = {key(group, TOTAL_VARIATION): float(tv)}
    for frozen in points:
        transformed = _passing(scores, signal, frozen.thresholds)
        results |= _point_metrics(group, frozen, transformed, totals, class_names)
    return results


def output_metrics(
    arrays: dict[str, np.ndarray], class_names: Sequence[str] = CLASS_NAMES
) -> dict[str, float]:
    """Every metric read off the stored outputs of one test pass, in float64.

    :param arrays: the ``.npz`` layout of `OutputStore.arrays`.
    :param class_names: the classes of the logit columns, in their order: the suffixes
        of the per-class keys.
    :return: the groupless ``performance/`` keys and, per group, the TV, the retentions
        and the diagnostics. Non-finite clean logits leave every one of them out; those
        of one group leave that group's out.
    """
    clean = _finite(arrays["logits_clean"], "clean")
    if clean is None:
        return {}
    scores, signal, points, totals = _clean_reading(clean, arrays["labels"])
    results = _clean_metrics(scores, signal, points, totals, class_names)
    p_clean = _softmax(clean)
    for group in GROUPS:
        logits = arrays[f"logits_{group}"]
        results |= _group_metrics(
            group, p_clean, logits, signal, points, totals, class_names
        )
    return results


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write *arrays* to *path* by way of a partial file, so that an interrupted run
    leaves no truncated ``.npz`` under the final name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    # A file handle, since np.savez appends .npz to a name lacking it.
    with open(partial, "wb") as handle:
        np.savez(handle, **arrays)
    partial.replace(path)
    log.info(f"Per-jet test outputs written to {path}")


def _drawing_transforms(
    datamodule: Any, transform_kwargs: dict[str, float | bool]
) -> dict[str, DrawingTransform]:
    """One drawing transform per group, in the order of `src.symmetry.GROUPS`, on the
    record's columns.

    :raises ValueError: unless the datamodule keeps every record column in its test
        batches, where the transforms find the momenta.
    """
    names = getattr(datamodule, "feature_names", None)
    if not getattr(datamodule, "keep_full_features_at_test", False) or names is None:
        raise ValueError(
            f"{type(datamodule).__name__} does not keep every record column in its "
            "test batches, so the probe has no momenta to transform. Inputs that no "
            "group acts on take probe: false."
        )
    scales = datamodule.norm_scales
    return {
        group: make_drawing_transform(group, names, scales, **transform_kwargs)
        for group in GROUPS
    }


class InvarianceMetricCallback(Callback):
    """Log the ``invariance/*`` and ``performance/*`` metrics at the end of the test
    pass.

    Every test jet is transformed once under each of `src.symmetry.GROUPS`, with one
    draw per jet from the global torch RNG. Per group, the metrics are the flip rate
    with its label-aware breakdown and the accuracy after the transform; the paired
    total variation between the clean and the transformed softmax outputs; and, at each
    of `src.symmetry.WORKING_POINTS`, the retention of the clean rejection or TPR at the
    threshold frozen on the clean one-vs-rest scores, with the efficiency that threshold
    fixed as read on the transformed scores. Groupless, they are the clean rejection or
    TPR of each working point and the one-vs-rest AUC. The working-point and AUC
    families are logged per class, named by the datamodule's ``class_names``, and as the
    macro mean over the classes.

    The logits are recomputed here in fp32, so the AUC can differ from the ROC plot's,
    which reads the step's autocast logits under 16-mixed.

    :param transform_kwargs: they hyperparameters that a ``make_transform`` in the data
        module would take.
    :param probe: False transforms nothing and logs the groupless families alone, for
        inputs no group acts on (the jet-level observables), where a probe would report
        an invariance that is never tested. The outputs then hold the clean logits and
        the labels alone.
    """

    def __init__(
        self,
        transform_kwargs: dict[str, float | bool] | None = None,
        probe: bool = True,
    ) -> None:
        super().__init__()
        self.transform_kwargs = dict(transform_kwargs or {})
        self.probe = probe
        # Set by eval.py to the checkpoint's label ("main", "fold_1"): the per-jet
        # logits, labels and draws then go to <default_root_dir>/outputs/<subdir>.npz.
        # None writes no outputs.
        self.subdir: str | None = None
        self._reset()

    def _reset(self, class_names: tuple[str, ...] = CLASS_NAMES) -> None:
        """Drop the state of the previous test pass; a fold reuses the same callback."""
        self._class_names = class_names
        self._transforms: dict[str, DrawingTransform] = {}
        self._flip = FlipProbe()
        self._store = OutputStore()

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Read the datamodule's classes and build the pass's transforms on its columns,
        unless the probe is off."""
        self._reset(class_names_of(trainer.datamodule))
        if not self.probe:
            log.info("Invariance probe off: the clean metrics alone are logged.")
            return
        datamodule = trainer.datamodule
        self._transforms = _drawing_transforms(datamodule, self.transform_kwargs)
        log.info(f"Invariance probe ready on the columns {datamodule.feature_names}.")

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
        """Keep one test batch's clean logits and, under the probe, how each group's
        transform moves them."""
        x, y = batch
        targets = y.argmax(dim=1)
        # The clean (unaugmented data) logits are recomputed here due to precision:
        # the trainer can have 16-mixed precision, but here we enforce fp32.
        clean = pl_module(x)
        transformed, draws = {}, {}
        for group in GROUPS if self.probe else ():
            probed = self._probe_group(pl_module, group, x, clean, targets)
            transformed[group], draws[group] = probed
        self._store.add(clean, targets, transformed, draws)

    def _probe_group(
        self,
        pl_module: LightningModule,
        group: str,
        x: torch.Tensor,
        clean: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One draw per jet under *group*, compared with the same untranformed jets.

        :return: the transformed logits and the draws behind them.
        """
        x_sym, draw = self._transforms[group](x)
        logits_sym = pl_module(x_sym)
        self._flip.update(group, clean, logits_sym, targets)
        return logits_sym, draw

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reduce the counts and the kept outputs over the whole test set, log all, and
        write the outputs when the pass is named."""
        if not self._store.chunks:
            return
        arrays = self._store.arrays()
        results = self._results(arrays)
        pl_module.log_dict(results, prog_bar=False, on_epoch=True, sync_dist=True)
        macro = [n for n in results if n.rsplit("_", 1)[-1] not in self._class_names]
        log.info(" | ".join(f"{n}: {results[n]:.4f}" for n in macro))
        if self.subdir is not None:
            path = Path(trainer.default_root_dir) / "outputs" / f"{self.subdir}.npz"
            _save(path, arrays)

    def _results(self, arrays: dict[str, np.ndarray]) -> dict[str, float]:
        """Every metric of the pass: the groupless ones alone when nothing was
        probed."""
        names = self._class_names
        if not self.probe:
            return clean_metrics(arrays["logits_clean"], arrays["labels"], names)
        return self._flip.metrics() | output_metrics(arrays, names)
