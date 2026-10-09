import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from src.callbacks.invariance import (
    InvarianceMetricCallback,
    clean_metrics,
    output_metrics,
)
from src.symmetry import (
    ACCURACY_TRANSFORMED,
    CLASS_NAMES,
    FLIP,
    FLIP_RECOVER,
    FLIP_STABLE,
    GROUPS,
    TABULATED_FAMILIES,
    TABULATED_GROUPS,
    key,
)
from src.utils.checkpoints import (
    Manifest,
    ManifestRun,
    merge_manifests,
    read_manifest,
    write_manifest,
)
from src.utils.results_table import ResultsTable
from tests.conftest import (
    FOUR_MOMENTUM,
    PT_ETA_PHI,
    StubDataModule,
    labels,
    read_results_row,
    record_jets,
    results_header,
)

# Five logits from pt, eta and phi, no two of them equal.
MIXING = torch.tensor(
    [[1.0, 0.0, 0.0, 1.0, 0.5], [0.0, 1.0, 0.0, -1.0, 0.5], [0.0, 0.0, 1.0, 0.0, -1.0]]
)
# What the probe stores of 48 jets: the shapes of the logits and of each group's draws.
STORED = {
    "logits_clean": (48, 5),
    "labels": (48,),
    **{f"logits_{group}": (48, 5) for group in GROUPS},
    "draw_so2": (48,),
    "draw_so3": (48, 3, 3),
    "draw_lorentz": (48,),
    "draw_lorentz_iso": (48, 4),
}
# The groupless families and the value every fold reports for each.
CLEAN = {"rejection_es30": 20.0, "tpr_eb1": 0.3, "tpr_eb10": 0.6, "auc": 0.9}
# Per fold, the TV under so2, so3 and lorentz, which fills every tabulated family too.
FOLD_VALUES = [
    (0.1, 0.2, 0.3),
    (0.15, 0.25, 0.35),
    (0.2, 0.3, 0.4),
    (math.nan, 0.25, 0.35),
]


class _StubModule(torch.nn.Module):
    # The network of a [pt, eta, phi] run: it reads those three of the record's columns
    # off a full-width test batch, the layout the probe transforms.
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.logged: dict[str, float] = {}

    def log_dict(self, metrics: dict[str, float], **kwargs) -> None:
        self.logged |= metrics

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, [5, 7, 10]].sum(dim=1) @ MIXING


def _flips(arrays: dict[str, np.ndarray]) -> dict[str, float]:
    truth, clean = arrays["labels"], arrays["logits_clean"].argmax(axis=1)
    flips = {}
    for group in GROUPS:
        moved = arrays[f"logits_{group}"].argmax(axis=1)
        right, right_moved = clean == truth, moved == truth
        flips |= {
            key(group, FLIP): np.mean(clean != moved),
            key(group, FLIP_RECOVER): (~right & right_moved).sum() / (~right).sum(),
            key(group, FLIP_STABLE): (right & right_moved).sum() / right.sum(),
            key(group, ACCURACY_TRANSFORMED): np.mean(right_moved),
        }
    return flips


@pytest.mark.parametrize("probe", [True, False])
def test_the_probe_logs_what_its_stored_outputs_give(
    probe: bool, tmp_path: Path
) -> None:
    # Two batches of unequal size, so that a rate over the jets differs from a mean of
    # the per-batch rates. Without the probe (the jet-level rows) nothing is transformed
    # and the datamodule, here without a single attribute, is never read.
    full = record_jets(48, 6, 6).float()
    batches = [(full[:32], labels(32, seed=1)), (full[32:], labels(16, seed=2))]
    module, callback = _StubModule(), InvarianceMetricCallback(probe=probe)
    callback.subdir = "fold_1"
    datamodule = StubDataModule(PT_ETA_PHI) if probe else SimpleNamespace()
    trainer = SimpleNamespace(datamodule=datamodule, default_root_dir=str(tmp_path))

    callback.on_test_start(trainer, module)
    for index, batch in enumerate(batches):
        callback.on_test_batch_end(trainer, module, None, batch, index)
    callback.on_test_epoch_end(trainer, module)

    with np.load(tmp_path / "outputs" / "fold_1.npz") as stored:
        arrays = dict(stored)
    clean = torch.cat([module(x) for x, _ in batches]).numpy()
    truth = torch.cat([y.argmax(dim=1) for _, y in batches]).numpy()
    expected = (
        STORED if probe else {name: STORED[name] for name in ("logits_clean", "labels")}
    )
    assert {name: array.shape for name, array in arrays.items()} == expected
    assert arrays["labels"].dtype == np.int8
    assert all(arrays[name].dtype == np.float32 for name in set(expected) - {"labels"})
    np.testing.assert_array_equal(arrays["logits_clean"], clean)
    np.testing.assert_array_equal(arrays["labels"], truth)
    # Renamed into place: no partial file is left beside it.
    assert [path.name for path in (tmp_path / "outputs").iterdir()] == ["fold_1.npz"]
    assert len(module.logged) == (188 if probe else 24)
    if probe:
        assert module.logged == output_metrics(arrays) | _flips(arrays)
    else:
        assert module.logged == clean_metrics(clean, truth)
    reduced = SimpleNamespace(
        datamodule=StubDataModule(PT_ETA_PHI, keep_full_features_at_test=False)
    )
    with pytest.raises(ValueError, match="every record column"):
        InvarianceMetricCallback().on_test_start(reduced, _StubModule())


def _class_g(
    signal: list[float], background: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    # Logits (u, 0, 0, 0, 0): the one-vs-rest score of g is u - log 4, so its thresholds
    # and counts are those of u. The background cycles through the other four classes.
    u = np.array(signal + background)
    logits = np.zeros((u.size, 5))
    logits[:, 0] = u
    return logits, np.array(
        [0] * len(signal) + [1 + i % 4 for i in range(len(background))]
    )


def _arrays(
    clean: np.ndarray, truth: np.ndarray, **moved: np.ndarray
) -> dict[str, np.ndarray]:
    # A group not named sees the clean logits again.
    arrays = {"logits_clean": clean, "labels": truth}
    return arrays | {f"logits_{g}": moved.get(g, clean) for g in GROUPS}


def _moved_g(logits: np.ndarray, shift: float) -> np.ndarray:
    moved = logits.copy()
    moved[:, 0] += shift
    return moved


def test_the_metrics_equal_their_hand_counts() -> None:
    # Signal at 1, ..., 10 and background at 0, 0.5, ..., 9.5. es30: the 0.7 quantile of
    # the signal ("higher", index 6.3 -> 7) is 8, passed by signal 9, 10 and background
    # 8.5, 9, 9.5. eb1: the 0.99 quantile of the background (index 18.81 -> 19) is 9.5,
    # passed by signal 10 alone; eb10 (index 17.1 -> 18) is 9.0, passed by signal 10 and
    # background 9.5. Under SO(2) every score of g moves up by one, at the same
    # thresholds.
    logits, truth = _class_g(list(range(1, 11)), [0.5 * i for i in range(20)])
    metrics = output_metrics(_arrays(logits, truth, so2=_moved_g(logits, 1.0)))
    so2 = lambda family: metrics[key("so2", family, "g")]  # noqa: E731
    # The background now stops at 4.75, below both clean thresholds of es30 (8) and eb1
    # (4.75), so no background jet passes either clean.
    low, low_truth = _class_g(list(range(1, 11)), [0.25 * i for i in range(20)])
    moved = {"so3": _moved_g(low, 4.0), "lorentz": _moved_g(low, -20.0)}
    floored = output_metrics(_arrays(low, low_truth, **moved))
    rng = np.random.default_rng(0)
    truth_tv = rng.integers(0, 5, size=2000)
    clean = rng.normal(size=(2000, 5)) + 1.5 * np.eye(5)[truth_tv]
    shuffled = clean[rng.permutation(2000)]
    opposite = np.roll(20.0 * np.eye(5)[truth_tv], 1, axis=1)
    tv = output_metrics(_arrays(clean, truth_tv, so2=clean + 3.0, so3=shuffled))
    tv_opposite = output_metrics(
        _arrays(20.0 * np.eye(5)[truth_tv], truth_tv, lorentz=opposite)
    )
    softmax = np.exp(clean) / np.exp(clean).sum(axis=1, keepdims=True)
    auc = [roc_auc_score(truth_tv == c, softmax[:, c]) for c in range(5)]

    assert metrics[key(None, "rejection_es30", "g")] == pytest.approx(20 / 3)
    assert metrics[key(None, "tpr_eb1", "g")] == pytest.approx(0.1)
    assert metrics[key(None, "tpr_eb10", "g")] == pytest.approx(0.1)
    # Above 7: signal 8, 9, 10 and background 7.5, ..., 9.5, so R falls from 20/3 to 4.
    assert so2("efficiency_transformed_es30") == pytest.approx(0.3)
    assert so2("rejection_retention_es30") == pytest.approx(3 / 5)
    assert so2("tpr_retention_eb1") == so2("tpr_retention_eb10") == pytest.approx(2.0)
    assert so2("fpr_transformed_eb1") == pytest.approx(2 / 20)
    assert so2("fpr_transformed_eb10") == pytest.approx(3 / 20)
    # R = n_B / max(0, 1), a finite 20; moved up by 4, 4.25, 4.5 and 4.75 pass, so R
    # falls to 20/3; moved off, nothing passes on either side and the rejection is kept,
    # while none of the six signal jets above 4.75 is left.
    assert floored[key(None, "rejection_es30", "g")] == pytest.approx(20.0)
    assert floored[key("so3", "rejection_retention_es30", "g")] == pytest.approx(1 / 3)
    assert floored[key("lorentz", "rejection_retention_es30", "g")] == 1.0
    assert floored[key(None, "tpr_eb1", "g")] == pytest.approx(0.6)
    assert floored[key("lorentz", "tpr_retention_eb1", "g")] == 0.0
    # Paired, and blind to a common shift of the logits, which moves no softmax output.
    assert tv[key("so2", "tv")] < 1e-12 and tv[key("so3", "tv")] > 0.05
    assert tv[key("lorentz_iso", "tv")] == 0.0
    assert tv_opposite[key("lorentz", "tv")] == pytest.approx(1.0, abs=1e-6)
    # The softmax column orders the jets as the one-vs-rest log-odds do.
    assert [tv[key(None, "auc", c)] for c in CLASS_NAMES] == pytest.approx(auc)
    assert tv[key(None, "auc")] == pytest.approx(np.mean(auc))


def _table(directory: Path, **overrides) -> ResultsTable:
    defaults = {
        "output_path": str(directory / "results_table.csv"),
        "method": "augmentation",
        "model_name": "mlp",
        "augmentation_scheme": "so2",
        "nconstituents": 8,
        "invariance_metric": "tv",
    }
    return ResultsTable(**(defaults | overrides))


def _fold(values: tuple[float, float, float], acc: float, **rest: object) -> dict:
    fold = {key(None, family): value for family, value in CLEAN.items()}
    for family in ("tv", *TABULATED_FAMILIES):
        fold |= {key(g, family): v for g, v in zip(TABULATED_GROUPS, values)}
    costs = {"flops_total": 1000.0, "ebops_total": 64000.0}
    return fold | costs | {"test/acc": acc} | rest


@pytest.mark.parametrize(
    "scheme, selected_key, selected",
    [
        ("so2", "invariance/so2_tv", "0.100000;0.150000;0.200000;nan"),
        # A run trained on no group belongs to none: the three are averaged per fold.
        ("none", "invariance/mean_tv", "0.200000;0.250000;0.300000;0.300000"),
    ],
)
def test_the_results_row_summarises_the_folds_it_is_given(
    tmp_path: Path, scheme: str, selected_key: str, selected: str
) -> None:
    folds = [_fold(values, 0.90 + 0.01 * i) for i, values in enumerate(FOLD_VALUES)]
    # A fold whose module could not be counted reports no eBOPs at all.
    folds[1]["ebops_total"] = 32000.0
    del folds[3]["ebops_total"]

    _table(tmp_path, augmentation_scheme=scheme).write(folds)

    header, row = read_results_row(tmp_path)
    mean = np.nanmean([float(v) for v in selected.split(";")])
    assert header == results_header() and len(header) == 206
    assert (row["model"], row["folds_used"]) == ("mlp", "4")
    assert row["so2_tv_folds"] == "0.100000;0.150000;0.200000;nan"
    # The NaN fold is left out of the mean, as is the fold without eBOPs.
    assert float(row["so2_tv_mean"]) == pytest.approx(0.15)
    assert (row["flops_mean"], row["ebops_mean"]) == ("1000.0", "53333.333")
    assert (
        0.90
        <= float(row["test_acc_ci_lower"])
        < 0.915
        < float(row["test_acc_ci_upper"])
    )
    assert row["test_acc_folds"] == "0.900000;0.910000;0.920000;0.930000"
    for family, value in CLEAN.items():
        assert row[f"{family}_folds"] == ";".join([f"{value:.6f}"] * 4)
    assert (row["selected_inv_key"], row["selected_inv_folds"]) == (
        selected_key,
        selected,
    )
    assert float(row["selected_inv_mean"]) == pytest.approx(mean)
    assert float(row["flip_recover_mean"]) == pytest.approx(mean)
    assert row["lorentz_flip_stable_pm"] == "0.3500 +/- 0.0408"


def test_the_results_tables_are_placed_and_appended_by_regime(tmp_path: Path) -> None:
    baseline = {"augmentation_scheme": "none", "features": FOUR_MOMENTUM}
    placed = {
        "px_py_pz_e/results_table_baseline.csv": baseline,
        "px_py_pz_e/results_table_symm_loss.csv": baseline | {"method": "symm_loss"},
        "embedded/results_table_baseline.csv": baseline | {"subdir": "embedded"},
        "pt_eta_phi/results_table_baseline.csv": baseline | {"features": PT_ETA_PHI},
    }
    table, row = _table(tmp_path, **baseline), [_fold(FOLD_VALUES[0], 0.9)]
    table.write([])

    for path, overrides in placed.items():
        assert _table(tmp_path, **overrides).output_path == tmp_path / path
    # Nothing is created before a row is written, so a search leaves results/ untouched.
    assert not table.output_path.parent.exists()
    table.write(row)
    _table(tmp_path, **baseline).write(row)
    assert len(table.output_path.read_text().splitlines()) == 3
    # The flip metric fills its one family, and its header is not the tv table's.
    with pytest.raises(ValueError, match="misalign"):
        _table(tmp_path, **baseline, invariance_metric="flip").write(row)
    _table(tmp_path / "flip", invariance_metric="flip").write(row)
    header, flipped = read_results_row(tmp_path / "flip")
    assert header.count("so2_flip_folds") == 1 and "flip_mean" not in header
    assert flipped["selected_inv_key"] == "invariance/so2_flip"


def _manifest(directory: Path, n_folds: int, *runs: ManifestRun) -> Path:
    # As a process writes it: n_folds 0 for a main run alone, k for one of its folds.
    manifest = Manifest(n_folds, str(directory), "exp", list(runs))
    return write_manifest(manifest, directory)


def _fold_run(index: int) -> ManifestRun:
    return ManifestRun(f"fold_{index}", index, f"/ckpt/fold_{index}.ckpt", 0.5)


def test_the_manifests_round_trip_and_merge_main_first_then_folds(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    main = ManifestRun("main", None, "/ckpt/main.ckpt", 0.5, "run0")
    path = _manifest(tmp_path, 0, main)
    # A fold still training has written no manifest yet: three of four are found.
    for index in (10, 2, 1):
        _manifest(tmp_path / f"fold_{index}", 4, _fold_run(index))
    stale = tmp_path / "stale"
    _manifest(stale / "fold_1", 2, _fold_run(1))
    payload = json.loads(path.read_text()) | {"version": 0}
    (stale / "old.json").write_text(json.dumps(payload))

    with caplog.at_level(logging.WARNING):
        merged = merge_manifests(tmp_path)

    labels_merged = [run.fold_label for run in merged.runs]
    assert read_manifest(path) == Manifest(0, str(tmp_path), "exp", [main])
    assert labels_merged == ["main", "fold_1", "fold_2", "fold_10"]
    assert (merged.n_folds, merged.hydra_output_dir) == (3, str(tmp_path))
    assert "still running" in caplog.text
    with pytest.raises(FileNotFoundError, match="main run"):
        merge_manifests(stale)
    with pytest.raises(ValueError, match="version"):
        read_manifest(stale / "old.json")
    # A sequential k-fold manifest dropped into fold_1/ would count its folds twice.
    _manifest(stale, 0, main)
    _manifest(stale / "fold_1", 2, main, _fold_run(1), _fold_run(2))
    with pytest.raises(ValueError, match="one fold"):
        merge_manifests(stale)
