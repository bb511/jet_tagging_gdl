import math
from types import SimpleNamespace

import pytest
import torch
from torchmetrics.classification import MulticlassAccuracy

from src.callbacks.invariance import GROUPS, InvarianceMetricCallback, _Accumulator
from tests.conftest import RECORD_COLUMNS, StubDataModule, jets, labels

CLASS_NAMES = ("g", "q", "W", "Z", "t")
FAMILIES = ("flip", "wasserstein", "flip_recover", "flip_stable", "flip_degrade")


class _StubModule(torch.nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def setup(self, stage: str) -> None:
        pass

    def log(self, name: str, value: float, **kwargs) -> None:
        self.logged[name] = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Two stubs of opposite sign disagree on the argmax, so the baseline reads differently.
        return self.weight * x[:, :, :5].sum(dim=1)


def _counts(wrong: int, recovered: int, correct: int, stable: int) -> dict[str, int]:
    return {
        "wrong_before": wrong,
        "recovered": recovered,
        "correct_before": correct,
        "stable": stable,
    }


def _probed(callback: InvarianceMetricCallback, module: _StubModule) -> dict[str, float]:
    trainer = SimpleNamespace(datamodule=StubDataModule(features=RECORD_COLUMNS))
    callback.on_test_start(trainer, module)
    batch = (jets(32, 6, 6, n_columns=len(RECORD_COLUMNS)), labels(32))
    callback.on_test_batch_end(trainer, module, None, batch, 0)
    callback.on_test_epoch_end(trainer, module)
    return module.logged


def test_the_flip_breakdown_splits_the_transitions_by_group_and_by_what_they_did() -> None:
    callback = InvarianceMetricCallback(compute_flip_breakdown=True)
    callback._own.flip_counts = {
        "so2": _counts(wrong=10, recovered=4, correct=90, stable=81),
        "lorentz": _counts(wrong=0, recovered=0, correct=50, stable=50),
        "lorentz_iso": _counts(wrong=20, recovered=5, correct=80, stable=60),
    }

    metrics = callback._flip_breakdown_metrics(callback._own)

    assert GROUPS == ("so2", "so3", "lorentz", "lorentz_iso")
    assert metrics["invariance/lorentz_iso_flip_degrade"] == pytest.approx(0.25)
    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(0.9)
    assert metrics["invariance/so2_flip_degrade"] == pytest.approx(0.1)
    # Nothing wrong to recover is not a recovery rate of zero.
    assert math.isnan(metrics["invariance/lorentz_flip_recover"])
    assert metrics["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_a_constant_shift_is_measured_at_its_own_size_and_a_reshuffling_is_not_seen() -> None:
    torch.manual_seed(0)
    callback = InvarianceMetricCallback(compute_wasserstein=True, class_names=CLASS_NAMES)
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
    # The mean over the five class channels.
    assert metrics["invariance/lorentz_wasserstein"] == pytest.approx(0.6, abs=0.05)
    # Unpaired on purpose, so a permutation of the same logits reads as zero.
    assert metrics["invariance/so2_wasserstein"] < 1e-6


def test_a_run_without_a_baseline_reports_no_baseline_and_no_delta() -> None:
    logged = _probed(InvarianceMetricCallback(), _StubModule(1.0))

    assert any(name.startswith("invariance/") for name in logged)
    assert not [name for name in logged if name.startswith("baseline/") or "delta" in name]


def test_the_baseline_is_probed_on_the_same_transformed_jets_as_the_model() -> None:
    callback = InvarianceMetricCallback(compute_flip_breakdown=False, class_names=CLASS_NAMES)
    callback.baseline_module = _StubModule(-1.0)

    logged = _probed(callback, _StubModule(1.0))

    # Every delta is built from the transition counts, so a baseline switches them back on.
    assert callback.compute_flip_breakdown
    for group in GROUPS:
        for family in FAMILIES:
            assert f"invariance/{group}_{family}" in logged
            assert f"baseline/{group}_{family}" in logged
            assert f"invariance/{group}_delta_{family}" in logged
    assert "baseline/test_acc" in logged
    assert "delta_test_acc" in logged


def test_the_deltas_are_signed_so_that_a_positive_value_means_the_better_model() -> None:
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
    # Own minus baseline where more is better, baseline minus own where less is.
    assert deltas["invariance/so2_delta_flip_recover"] == pytest.approx(0.5 - 0.2)
    assert deltas["invariance/so2_delta_flip_stable"] == pytest.approx(0.9 - 0.8)
    assert deltas["invariance/so2_delta_flip_degrade"] == pytest.approx(0.2 - 0.1)
    assert deltas["invariance/so2_delta_flip"] == pytest.approx(0.15)
    assert deltas["invariance/so2_delta_wasserstein"] == pytest.approx(0.6)


def test_the_accuracy_after_the_transform_is_read_off_the_transition_counts() -> None:
    callback = InvarianceMetricCallback()
    callback.baseline_module = _StubModule(-1.0)
    callback._own.flip_counts = {"so3": _counts(wrong=10, recovered=5, correct=90, stable=81)}
    callback._baseline.flip_counts = {
        "so3": _counts(wrong=40, recovered=10, correct=60, stable=30)
    }

    deltas = callback._delta_metrics({})

    # (81 + 5) / 100 against (30 + 10) / 100, each model on its own surviving jets.
    assert deltas["invariance/so3_delta_accuracy_transformed"] == pytest.approx(0.86 - 0.40)
    assert math.isnan(deltas["invariance/so2_delta_accuracy_transformed"])


def test_the_clean_accuracy_is_the_macro_average_the_module_logs() -> None:
    accumulator = _Accumulator("baseline")
    # Class 0: two jets, both right. Class 1: three jets, one predicted as class 0. Class 2 absent.
    logits = torch.tensor([[1.0, 0.0, 0.0]] * 3 + [[0.0, 1.0, 0.0]] * 2)
    targets = torch.tensor([0, 0, 1, 1, 1])

    assert math.isnan(accumulator.clean_accuracy())
    accumulator.count_clean(logits[:3], targets[:3])
    accumulator.count_clean(logits[3:], targets[3:])

    expected = MulticlassAccuracy(num_classes=3)(logits.argmax(dim=1), targets)
    assert accumulator.clean_accuracy() == pytest.approx(float(expected))
    assert accumulator.clean_accuracy() == pytest.approx((1.0 + 2.0 / 3.0) / 2.0)
    assert accumulator.clean_accuracy() != pytest.approx(4.0 / 5.0)
