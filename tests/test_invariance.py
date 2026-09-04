"""The invariance families the probe reports, computed from state fed in by hand.

The flip breakdown counts label transitions and the Wasserstein family compares two logit
distributions. Both feed the results table, and neither can be checked against a training run, so
the counters and the distances are exercised directly. The baseline comparison is checked the same
way, against a stub model whose logits are known in advance.
"""

import math

import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy

from src.callbacks.invariance import GROUPS, InvarianceMetricCallback, _Accumulator

CLASS_NAMES = ("g", "q", "W", "Z", "t")

# The record's own column order, which a test batch carries under keep_full_features_at_test.
RECORD_COLUMNS = (
    "j1_px j1_py j1_pz j1_e j1_erel j1_pt j1_ptrel j1_eta j1_etarel j1_etarot j1_phi "
    "j1_phirel j1_phirot j1_deltaR j1_costheta j1_costhetarel"
).split()

# In GeV. A massless constituent would put the invariant mass at the level of the float32
# rounding of E^2, and the boosts would then measure the arithmetic rather than the transform.
MOMENTUM_SCALE = 20.0
MASS = 20.0


class _StubModule(torch.nn.Module):
    """Stands in for a trained module: deterministic logits, and the hooks the probe calls.

    :param weight: scales the summed constituents, so two stubs of different weight disagree on the
        argmax of a jet whose columns do not all share a sign.
    """

    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def setup(self, stage: str) -> None:
        """The probe calls it on the baseline, which no trainer ever sets up."""

    def log(self, name: str, value: float, **kwargs) -> None:
        """Collect what the probe logs, in place of Lightning's own logger."""
        self.logged[name] = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Class logits for a batch of shape (jets, constituents, features)."""
        return self.weight * x[:, :, :5].sum(dim=1)


class _StubDataModule:
    """The surface :meth:`InvarianceMetricCallback.on_test_start` reads off a datamodule."""

    feature_names = RECORD_COLUMNS
    features = RECORD_COLUMNS
    norm_scales = None

    def feature_indices(self) -> None:
        """None, so the probe names the columns from ``features`` itself."""


class _StubTrainer:
    """A trainer that owns nothing but the datamodule the probe asks it for."""

    def __init__(self) -> None:
        self.datamodule = _StubDataModule()


def _counts(wrong: int, recovered: int, correct: int, stable: int) -> dict[str, int]:
    """One group's transition counters, in the order the breakdown reads them."""
    return {
        "wrong_before": wrong,
        "recovered": recovered,
        "correct_before": correct,
        "stable": stable,
    }


def _batch(n_jets: int = 32, n_slots: int = 6, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A test batch at the record's full width, with one-hot labels over the five classes.

    The first four columns are a massive four-momentum in GeV, so the boosts stay real; the rest
    are noise, which no transform rewrites.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n_jets, n_slots, len(RECORD_COLUMNS), generator=generator)
    x[:, :, :3] *= MOMENTUM_SCALE
    x[:, :, 3] = torch.sqrt((x[:, :, :3] ** 2).sum(-1) + MASS**2)
    labels = torch.randint(0, 5, (n_jets,), generator=generator)

    return x, torch.nn.functional.one_hot(labels, num_classes=5).float()


def _probed(callback: InvarianceMetricCallback, module: _StubModule) -> dict[str, float]:
    """Run one full test pass of the probe over a single batch, and return what it logged."""
    trainer = _StubTrainer()
    callback.on_test_start(trainer, module)
    x, y = _batch()
    callback.on_test_batch_end(trainer, module, None, (x, y), 0)
    callback.on_test_epoch_end(trainer, module)

    return module.logged


def test_the_flip_breakdown_splits_the_transitions_by_group_and_by_what_they_did() -> None:
    """Only this breakdown tells a repaired wrong prediction from a broken right one."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._own.flip_counts = {
        "so2": _counts(wrong=10, recovered=4, correct=90, stable=81),
        "lorentz": _counts(wrong=0, recovered=0, correct=50, stable=50),
        "lorentz_iso": _counts(wrong=20, recovered=5, correct=80, stable=60),
    }

    metrics = callback._flip_breakdown_metrics(callback._own)

    # The isotropic boost is the fourth group probed, logged but never tabulated.
    assert GROUPS == ("so2", "so3", "lorentz", "lorentz_iso")
    assert metrics["invariance/lorentz_iso_flip_degrade"] == pytest.approx(0.25)
    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(0.9)
    assert metrics["invariance/so2_flip_degrade"] == pytest.approx(0.1)
    # A perfect classifier gets no wrong predictions to recover, which is not a zero rate.
    assert math.isnan(metrics["invariance/lorentz_flip_recover"])
    assert metrics["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_a_probe_that_never_ran_reports_nothing() -> None:
    """A datamodule that does not name its columns disables the probe, and no key is logged."""
    callback = InvarianceMetricCallback()

    assert callback._flip_breakdown_metrics(callback._own) == {}
    assert callback._wasserstein_metrics(callback._own) == {}


def test_a_constant_shift_is_measured_at_its_own_size_and_a_reshuffling_is_not_seen() -> None:
    """Wasserstein-1 between two distributions offset by a constant is that constant."""
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
    # Three batches, since the distributions are compared over the whole test set at once.
    batches = [torch.randn(2000, 5) for _ in range(3)]
    callback._own.clean_logits = batches
    callback._own.transformed_logits = {
        "lorentz": [batch + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]) for batch in batches],
        "so2": [batch[torch.randperm(batch.shape[0])] for batch in batches],
    }

    metrics = callback._wasserstein_metrics(callback._own)

    assert metrics["invariance/lorentz_wasserstein_g"] == pytest.approx(3.0, abs=0.05)
    for name in CLASS_NAMES[1:]:
        assert metrics[f"invariance/lorentz_wasserstein_{name}"] < 0.05
    # The reported number is the mean over the five class channels.
    assert metrics["invariance/lorentz_wasserstein"] == pytest.approx(0.6, abs=0.05)
    # The family is unpaired on purpose, so a permutation of the same logits reads as zero.
    assert metrics["invariance/so2_wasserstein"] < 1e-6


def test_a_run_without_a_baseline_reports_no_baseline_and_no_delta() -> None:
    """Every regime but one is evaluated against a baseline, and the rest must be unaffected."""
    logged = _probed(InvarianceMetricCallback(), _StubModule(1.0))

    assert any(name.startswith("invariance/") for name in logged)
    assert not [name for name in logged if name.startswith("baseline/") or "delta" in name]


def test_the_baseline_is_probed_on_the_same_transformed_jets_as_the_model() -> None:
    """A different model must report its own flip rate and distance under every group."""
    callback = InvarianceMetricCallback(class_names=CLASS_NAMES)
    callback.baseline_module = _StubModule(-1.0)

    logged = _probed(callback, _StubModule(1.0))

    for group in GROUPS:
        for suffix in ("flip", "wasserstein", "flip_recover", "flip_stable", "flip_degrade"):
            assert f"baseline/{group}_{suffix}" in logged
            assert f"invariance/{group}_{suffix}" in logged
    assert "baseline/test_acc" in logged
    assert "delta_test_acc" in logged


def test_a_baseline_forces_the_flip_breakdown_back_on() -> None:
    """Every delta is built from the transition counts, so switching them off cannot stand."""
    callback = InvarianceMetricCallback(compute_flip_breakdown=False)
    callback.baseline_module = _StubModule(-1.0)

    logged = _probed(callback, _StubModule(1.0))

    assert callback.compute_flip_breakdown
    assert "invariance/so2_delta_flip_recover" in logged


def test_the_deltas_are_signed_so_that_a_positive_value_means_the_better_model() -> None:
    """A table read across regimes needs one sign convention, whatever the metric measures."""
    callback = InvarianceMetricCallback()
    callback.baseline_module = _StubModule(-1.0)
    callback._own.flip_counts = {"so2": _counts(wrong=10, recovered=5, correct=90, stable=81)}
    callback._baseline.flip_counts = {"so2": _counts(wrong=10, recovered=2, correct=90, stable=72)}
    callback._own.clean_correct = torch.tensor([45, 45])
    callback._own.clean_total = torch.tensor([50, 50])
    callback._baseline.clean_correct = torch.tensor([40, 40])
    callback._baseline.clean_total = torch.tensor([50, 50])
    averaged = {
        "invariance/so2_flip": 0.10,
        "baseline/so2_flip": 0.25,
        "invariance/so2_wasserstein": 0.4,
        "baseline/so2_wasserstein": 1.0,
    }

    deltas = callback._delta_metrics(averaged)

    assert deltas["baseline/test_acc"] == pytest.approx(0.8)
    assert deltas["delta_test_acc"] == pytest.approx(0.1)
    # Own minus baseline: recovering more and staying stable more often is the better model.
    assert deltas["invariance/so2_delta_flip_recover"] == pytest.approx(0.5 - 0.2)
    assert deltas["invariance/so2_delta_flip_stable"] == pytest.approx(0.9 - 0.8)
    # Baseline minus own: flipping less, degrading less and moving the logits less is better.
    assert deltas["invariance/so2_delta_flip_degrade"] == pytest.approx(0.2 - 0.1)
    assert deltas["invariance/so2_delta_flip"] == pytest.approx(0.15)
    assert deltas["invariance/so2_delta_wasserstein"] == pytest.approx(0.6)


def test_the_accuracy_after_the_transform_is_read_off_the_transition_counts() -> None:
    """It must be counted on the jets that survived the filter, not on the whole test set."""
    callback = InvarianceMetricCallback()
    callback.baseline_module = _StubModule(-1.0)
    callback._own.flip_counts = {"so3": _counts(wrong=10, recovered=5, correct=90, stable=81)}
    callback._baseline.flip_counts = {
        "so3": _counts(wrong=40, recovered=10, correct=60, stable=30)
    }

    deltas = callback._delta_metrics({})

    # (81 + 5) / 100 against (30 + 10) / 100, the two models' own surviving populations.
    assert deltas["invariance/so3_delta_accuracy_transformed"] == pytest.approx(0.86 - 0.40)
    # A group neither model was probed on leaves an empty population, which is not a zero rate.
    assert math.isnan(deltas["invariance/so2_delta_accuracy_transformed"])


def test_an_accumulator_reports_no_accuracy_before_it_has_seen_a_jet() -> None:
    """A fold that tested nothing must not enter the table as an accuracy of zero."""
    assert math.isnan(_Accumulator("invariance").clean_accuracy())


def test_the_clean_accuracy_is_the_macro_average_the_module_logs() -> None:
    """The baseline's accuracy is read against ``test/acc``, so the two must share a convention:

    the mean recall over the classes present rather than the fraction of jets, which differs from
    it once the batch is imbalanced.
    """
    accumulator = _Accumulator("baseline")
    # Class 0: two jets, both right. Class 1: three jets, one predicted as class 0. Class 2 absent.
    logits = torch.tensor([[1.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 0.0]] * 2)
    targets = torch.tensor([0, 0, 1, 1, 1])

    accumulator.count_clean(logits[:3], targets[:3])
    accumulator.count_clean(logits[3:], targets[3:])

    expected = MulticlassAccuracy(num_classes=3)(logits.argmax(dim=1), targets)
    assert accumulator.clean_accuracy() == pytest.approx(float(expected))
    assert accumulator.clean_accuracy() == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert accumulator.clean_accuracy() != pytest.approx(4.0 / 5.0)
