import math
from types import SimpleNamespace

import pytest
import torch

from src.callbacks.invariance import (
    FlipCounts,
    FlipProbe,
    InvarianceMetricCallback,
    WassersteinProbe,
)
from src.symmetry import FAMILIES, GROUPS, key
from tests.conftest import RECORD_COLUMNS, StubDataModule, jets, labels


class _StubModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float, **kwargs) -> None:
        self.logged[name] = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :5].sum(dim=1)


def _logits(predicted: list[int]) -> torch.Tensor:
    return torch.eye(3)[torch.tensor(predicted)]


def _probed(
    callback: InvarianceMetricCallback, module: _StubModule
) -> dict[str, float]:
    trainer = SimpleNamespace(datamodule=StubDataModule(features=RECORD_COLUMNS))
    callback.on_test_start(trainer, module)
    batch = (jets(32, 6, 6, n_columns=len(RECORD_COLUMNS)), labels(32))
    callback.on_test_batch_end(trainer, module, None, batch, 0)
    callback.on_test_epoch_end(trainer, module)
    return module.logged


def test_the_flip_breakdown_splits_the_transitions_by_group_and_by_what_they_did() -> (
    None
):
    probe = FlipProbe()
    probe.counts |= {
        "so2": FlipCounts(wrong_before=10, recovered=4, correct_before=90, stable=81),
        "lorentz": FlipCounts(
            wrong_before=0, recovered=0, correct_before=50, stable=50
        ),
        "lorentz_iso": FlipCounts(
            wrong_before=20, recovered=5, correct_before=80, stable=60
        ),
    }

    metrics = probe.metrics()

    assert GROUPS == ("so2", "so3", "lorentz", "lorentz_iso")
    assert metrics["invariance/so2_flip_recover"] == pytest.approx(0.4)
    assert metrics["invariance/so2_flip_stable"] == pytest.approx(0.9)
    # Nothing wrong to recover is not a recovery rate of zero.
    assert math.isnan(metrics["invariance/lorentz_flip_recover"])
    assert metrics["invariance/lorentz_flip_stable"] == pytest.approx(1.0)


def test_the_accuracy_after_the_transform_is_read_off_the_transition_counts() -> None:
    probe = FlipProbe()
    probe.counts["so3"] = FlipCounts(
        wrong_before=10, recovered=5, correct_before=90, stable=81
    )

    metrics = probe.metrics()

    # (81 + 5) of the 100 jets the transform was drawn for.
    assert metrics["invariance/so3_accuracy_transformed"] == pytest.approx(0.86)
    assert math.isnan(metrics["invariance/so2_accuracy_transformed"])


def test_the_flip_rate_is_a_ratio_of_jets_not_a_mean_of_per_batch_means() -> None:
    probe = FlipProbe()
    targets = torch.zeros(8, dtype=torch.long)
    probe.update("so2", _logits([0, 0]), _logits([1, 1]), targets[:2])
    probe.update("so2", _logits([0] * 8), _logits([0] * 8), targets)

    flip = probe.metrics()["invariance/so2_flip"]

    assert probe.n_jets == 10
    # Two flipped jets out of ten, not the mean of the batches' 1.0 and 0.0.
    assert flip == pytest.approx(0.2)


def test_a_constant_shift_is_measured_at_its_own_size_and_a_reshuffling_is_not_seen() -> (
    None
):
    torch.manual_seed(0)
    probe = WassersteinProbe()
    batches = [torch.randn(2000, 5) for _ in range(3)]
    probe.clean = list(batches)
    probe.transformed = {
        "lorentz": [batch + 3.0 for batch in batches],
        "so3": [batch + torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0]) for batch in batches],
        "so2": [batch[torch.randperm(batch.shape[0])] for batch in batches],
    }

    metrics = probe.metrics()

    assert metrics["invariance/lorentz_wasserstein"] == pytest.approx(3.0, abs=0.05)
    # Averaged over the five class channels, of which the shift moved one.
    assert metrics["invariance/so3_wasserstein"] == pytest.approx(0.6, abs=0.05)
    # Unpaired on purpose, so a permutation of the same logits reads as zero.
    assert metrics["invariance/so2_wasserstein"] < 1e-6


def test_the_probe_logs_one_key_per_group_and_family_and_nothing_besides() -> None:
    logged = _probed(InvarianceMetricCallback(), _StubModule())

    assert FAMILIES == (
        "flip",
        "flip_recover",
        "flip_stable",
        "accuracy_transformed",
        "wasserstein",
    )
    assert set(logged) == {
        key(group, family) for group in GROUPS for family in FAMILIES
    }
    assert len(logged) == 20
