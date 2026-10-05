import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.callbacks.invariance import (
    FlipCounts,
    FlipProbe,
    InvarianceMetricCallback,
    output_metrics,
)
from src.symmetry import (
    CLASS_NAMES,
    DIAGNOSTIC_FAMILIES,
    FLIP_FAMILIES,
    GROUPS,
    PERFORMANCE_FAMILIES,
    RETENTION_FAMILIES,
    key,
)
from tests.conftest import RECORD_COLUMNS, StubDataModule, jets, labels

OUTPUT_KEYS = {
    "logits_clean",
    "labels",
    *(f"logits_{group}" for group in GROUPS),
    *(f"draw_{group}" for group in GROUPS),
}
# Five logits from three inputs, no two of them equal.
MIXING = torch.tensor(
    [[1.0, 0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, -1.0, 0.5], [0.0, 0.0, 1.0, 0.0, -1.0]]
)


class _StubModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float], **kwargs) -> None:
        self.logged |= metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :5].sum(dim=1)


class _CylindricalModule(_StubModule):
    # The network of a [pt, eta, phi] run: it reads those three of the record's columns off
    # a full-width test batch, which is the layout the probe has to transform.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, [5, 7, 10]].sum(dim=1) @ MIXING


def _logits(predicted: list[int]) -> torch.Tensor:
    return torch.eye(3)[torch.tensor(predicted)]


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    x = jets(32, 6, 6, n_columns=len(RECORD_COLUMNS), seed=seed)
    return x, labels(32, seed=seed + 1)


def _probed(
    callback: InvarianceMetricCallback,
    module: _StubModule,
    datamodule: StubDataModule | None = None,
    batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    **trainer_attrs,
) -> dict[str, float]:
    trainer = SimpleNamespace(
        datamodule=datamodule or StubDataModule(features=RECORD_COLUMNS),
        **trainer_attrs,
    )
    callback.on_test_start(trainer, module)
    for index, batch in enumerate(batches or [_batch()]):
        callback.on_test_batch_end(trainer, module, None, batch, index)
    callback.on_test_epoch_end(trainer, module)
    return module.logged


def _expected_keys() -> set[str]:
    classes = (None, *CLASS_NAMES)
    per_group = {key(g, f) for g in GROUPS for f in (*FLIP_FAMILIES, "tv")}
    points = RETENTION_FAMILIES + DIAGNOSTIC_FAMILIES
    per_class = {key(g, f, c) for g in GROUPS for f in points for c in classes}
    performance = {key(None, f, c) for f in PERFORMANCE_FAMILIES for c in classes}
    return per_group | per_class | performance


def _arrays(
    clean: np.ndarray, truth: np.ndarray, **transformed: np.ndarray
) -> dict[str, np.ndarray]:
    # A group not named sees the clean logits again.
    arrays = {"logits_clean": clean, "labels": truth}
    return arrays | {f"logits_{g}": transformed.get(g, clean) for g in GROUPS}


def _class_g(
    signal_scores: list[float], background_scores: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    # Logits (u, 0, 0, 0, 0): the one-vs-rest score of g is u - log 4, so its thresholds
    # and counts are those of u. The background cycles through the other four classes.
    u = np.array(signal_scores + background_scores)
    logits = np.zeros((u.size, 5))
    logits[:, 0] = u
    background = [1 + i % 4 for i in range(len(background_scores))]
    return logits, np.array([0] * len(signal_scores) + background)


def _shifted_g(logits: np.ndarray, shift: float) -> np.ndarray:
    moved = logits.copy()
    moved[:, 0] += shift
    return moved


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
    # (81 + 4) of the 100 jets the transform was drawn for.
    assert metrics["invariance/so2_accuracy_transformed"] == pytest.approx(0.85)
    assert math.isnan(metrics["invariance/so3_accuracy_transformed"])


def test_the_flip_rate_is_a_ratio_of_jets_not_a_mean_of_per_batch_means() -> None:
    probe = FlipProbe()
    targets = torch.zeros(8, dtype=torch.long)
    probe.update("so2", _logits([0, 0]), _logits([1, 1]), targets[:2])
    probe.update("so2", _logits([0] * 8), _logits([0] * 8), targets)

    flip = probe.metrics()["invariance/so2_flip"]

    assert probe.n_jets == 10
    # Two flipped jets out of ten, not the mean of the batches' 1.0 and 0.0.
    assert flip == pytest.approx(0.2)


@pytest.mark.parametrize(
    "module, datamodule",
    [
        (_StubModule(), StubDataModule(features=RECORD_COLUMNS)),
        # keep_full_features_at_test is load-bearing under a reduced feature list: the
        # probe needs the momenta to transform, and the model reads its own columns in
        # forward().
        (
            _CylindricalModule(),
            StubDataModule(["pt", "eta", "phi"], keep_full_features_at_test=True),
        ),
    ],
)
def test_the_probe_logs_every_family_once_and_nothing_besides(
    module: _StubModule, datamodule: StubDataModule
) -> None:
    logged = _probed(InvarianceMetricCallback(), module, datamodule)

    assert set(logged) == _expected_keys()
    assert {"performance/auc_t", "invariance/lorentz_iso_tpr_retention_eb1_W"} <= set(
        logged
    )


def test_the_total_variation_is_paired_and_blind_to_a_shift_of_the_logits() -> None:
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 5, size=2000)
    clean = rng.normal(size=(2000, 5))
    shuffled = clean[rng.permutation(2000)]

    metrics = output_metrics(_arrays(clean, truth, so2=clean + 3.0, so3=shuffled))

    # A common shift leaves every softmax output where it was.
    assert metrics[key("so2", "tv")] < 1e-12
    # Paired, so the same logits dealt to other jets are seen.
    assert metrics[key("so3", "tv")] > 0.05
    assert metrics[key("lorentz_iso", "tv")] == 0.0

    one_hot = 20.0 * np.eye(5)[truth]
    opposite = np.roll(one_hot, 1, axis=1)
    tv = output_metrics(_arrays(one_hot, truth, lorentz=opposite))[key("lorentz", "tv")]

    assert tv == pytest.approx(1.0, abs=1e-6)


def test_the_working_points_are_read_at_thresholds_frozen_on_the_clean_scores() -> None:
    # Ten signal jets at 1, ..., 10 and twenty background jets at 0, 0.5, ..., 9.5.
    # es30: the 0.7 quantile of the signal ("higher", index 6.3 -> 7) is 8, passed by
    # signal 9, 10 and background 8.5, 9, 9.5. eb1: the 0.99 quantile of the background
    # (index 18.81 -> 19) is 9.5, passed by signal 10 alone. eb10: index 17.1 -> 18,
    # so 9.0, passed by signal 10 and background 9.5.
    logits, truth = _class_g(list(range(1, 11)), [0.5 * i for i in range(20)])
    # Under SO(2) every score of g moves up by one, at the same thresholds.
    metrics = output_metrics(_arrays(logits, truth, so2=_shifted_g(logits, 1.0)))
    g = {f: metrics[key(None, f, "g")] for f in PERFORMANCE_FAMILIES}
    points = RETENTION_FAMILIES + DIAGNOSTIC_FAMILIES
    so2 = {f: metrics[key("so2", f, "g")] for f in points}

    assert g["rejection_es30"] == pytest.approx(20 / 3)
    assert g["tpr_eb1"] == pytest.approx(0.1)
    assert g["tpr_eb10"] == pytest.approx(0.1)
    # Signal above 7 (3 of 10) and background above 7 (5): R falls from 20/3 to 20/5.
    assert so2["efficiency_transformed_es30"] == pytest.approx(0.3)
    assert so2["rejection_retention_es30"] == pytest.approx(3 / 5)
    # Above 8.5: signal 9, 10 and background 9, 9.5.
    assert so2["tpr_retention_eb1"] == pytest.approx(2.0)
    assert so2["fpr_transformed_eb1"] == pytest.approx(2 / 20)
    # Above 8: signal 9, 10 and background 8.5, 9, 9.5.
    assert so2["tpr_retention_eb10"] == pytest.approx(2.0)
    assert so2["fpr_transformed_eb10"] == pytest.approx(3 / 20)


def test_a_point_no_background_passes_is_floored_at_one_jet() -> None:
    # The background now stops at 4.75, below the es30 threshold of 8 and the eb1
    # threshold of 4.75 itself, so no background jet passes either clean.
    logits, truth = _class_g(list(range(1, 11)), [0.25 * i for i in range(20)])
    moved_up, moved_off = _shifted_g(logits, 4.0), _shifted_g(logits, -20.0)
    arrays = _arrays(logits, truth, so3=moved_up, lorentz=moved_off)

    metrics = output_metrics(arrays)

    # R = n_B / max(0, 1): the 20 background jets, finite.
    assert metrics["performance/rejection_es30_g"] == pytest.approx(20.0)
    assert math.isfinite(metrics["performance/rejection_es30"])
    # Moved up by 4, the background above 4 passes: 4.25, 4.5, 4.75; R falls to 20/3.
    assert metrics[key("so3", "rejection_retention_es30", "g")] == pytest.approx(1 / 3)
    # Nothing passes on either side: both counts floored, so the rejection is kept,
    # while no signal jet is left of the six (5, ..., 10) above 4.75.
    assert metrics[key("lorentz", "rejection_retention_es30", "g")] == 1.0
    assert metrics["performance/tpr_eb1_g"] == pytest.approx(0.6)
    assert metrics[key("lorentz", "tpr_retention_eb1", "g")] == 0.0
    assert metrics[key("lorentz", "efficiency_transformed_es30", "g")] == 0.0


def test_the_auc_of_each_class_is_sklearn_s_and_the_macro_is_their_mean() -> None:
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 5, size=500)
    clean = rng.normal(size=(500, 5)) + 1.5 * np.eye(5)[truth]
    # The softmax column orders the jets as the one-vs-rest log-odds do.
    probabilities = np.exp(clean) / np.exp(clean).sum(axis=1, keepdims=True)
    expected = [roc_auc_score(truth == c, probabilities[:, c]) for c in range(5)]

    metrics = output_metrics(_arrays(clean, truth))

    for name, auc in zip(CLASS_NAMES, expected):
        assert metrics[key(None, "auc", name)] == pytest.approx(auc)
    assert metrics[key(None, "auc")] == pytest.approx(np.mean(expected))


def test_only_a_named_pass_writes_its_outputs_in_loader_order(tmp_path: Path) -> None:
    batches = [_batch(seed=0), _batch(seed=1)]
    module, callback = _StubModule(), InvarianceMetricCallback()
    callback.subdir = "fold_1"

    _probed(callback, module, batches=batches, default_root_dir=str(tmp_path))
    _probed(InvarianceMetricCallback(), _StubModule(), default_root_dir=str(tmp_path))
    with np.load(tmp_path / "outputs" / "fold_1.npz") as stored:
        arrays = dict(stored)

    # Renamed into place: no partial file is left beside it, and the unnamed pass adds
    # nothing.
    assert [path.name for path in (tmp_path / "outputs").iterdir()] == ["fold_1.npz"]
    assert set(arrays) == OUTPUT_KEYS
    clean = torch.cat([module(x) for x, _ in batches]).numpy()
    truth = torch.cat([y.argmax(dim=1) for _, y in batches]).numpy()
    np.testing.assert_array_equal(arrays["logits_clean"], clean)
    np.testing.assert_array_equal(arrays["labels"], truth)
    assert arrays["labels"].dtype == np.int8
    assert all(arrays[k].dtype == np.float32 for k in OUTPUT_KEYS - {"labels"})
    assert all(arrays[f"logits_{group}"].shape == (64, 5) for group in GROUPS)
    assert arrays["draw_so2"].shape == arrays["draw_lorentz"].shape == (64,)
    assert arrays["draw_so3"].shape == (64, 3, 3)
    assert arrays["draw_lorentz_iso"].shape == (64, 4)
