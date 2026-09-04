"""One row per run: how the folds are aggregated, and where the row is written.

The results tables are the paper's numbers, and a run that has already cost GPU-days writes its row
once, so every quantity below is pinned against hand-computed values. The column order is pinned
too: a table on disk is appended to, and a header written in another order would misfile every
value of the new row.
"""

import csv
from pathlib import Path

import pytest

from src.callbacks.results_table import ResultsTableCallback

GROUPS = ("so2", "so3", "lorentz")
FLIP_SUFFIXES = ("flip_recover", "flip_stable", "flip_degrade")
SUMMARY = ("mean", "std", "pm", "folds")
DELTA_COLUMNS = (*SUMMARY, "ci_lower", "ci_upper", "bootstrap_pvalue", "significant_boot_005")
DELTA_FAMILIES = (
    "test_acc",
    "accuracy_transformed",
    "wasserstein",
    "flip",
    "flip_recover",
    "flip_stable",
    "flip_degrade",
)
DELTA_SUFFIXES = DELTA_FAMILIES[1:]
MEAN_OF_THREE = (0.3 + 0.6 + 0.9) / 3


def _invariance_columns() -> list[str]:
    """The per-group and flip-breakdown columns, which the baseline comparison leaves alone."""
    return [
        *(f"{g}_wasserstein_{c}" for g in GROUPS for c in ("mean", "std", "pm")),
        *(f"{g}_wasserstein_folds" for g in GROUPS),
        *(f"{s}_{c}" for s in FLIP_SUFFIXES for c in SUMMARY),
        *(f"{g}_{s}_{c}" for s in FLIP_SUFFIXES for g in GROUPS for c in SUMMARY),
    ]


def _expected_header() -> list[str]:
    """The column order of the tables, spelled out here rather than read off the callback."""
    return [
        "method",
        "model",
        "augmentation_scheme",
        "nconstituents",
        "folds_used",
        *(f"test_acc_{c}" for c in ("mean", "std", "pm", "ci_lower", "ci_upper")),
        "selected_inv_key",
        *(f"selected_inv_{c}" for c in ("mean", "std", "pm")),
        *_invariance_columns(),
        *(f"baseline_test_acc_{c}" for c in SUMMARY),
        *(f"delta_{f}_{c}" for f in DELTA_FAMILIES for c in DELTA_COLUMNS),
        "flops_mean",
        "test_acc_folds",
        "selected_inv_folds",
    ]


def _callback(tmp_path: Path, **overrides) -> ResultsTableCallback:
    """A callback writing under *tmp_path*, configured as ``main/results_table.yaml`` does."""
    defaults = {
        "output_path": str(tmp_path / "results_table.csv"),
        "n_folds": 3,
        "method": "augmentation",
        "model_name": "mlp",
        "augmentation_scheme": "so2",
        "nconstituents": 8,
        "invariance_metric": "wasserstein",
    }

    return ResultsTableCallback(**(defaults | overrides))


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as table:
        return list(csv.DictReader(table))


def _wasserstein_fold(so2: float, so3: float, lorentz: float) -> dict[str, float]:
    """One fold reporting the three groups' Wasserstein distances."""
    return {f"invariance/{g}_wasserstein": v for g, v in zip(GROUPS, (so2, so3, lorentz))}


def _fold(so2: float, so3: float, lorentz: float) -> dict[str, float]:
    """One fold's invariance metrics, the same value for every family of a given group."""
    fold = _wasserstein_fold(0.0, 0.0, 0.0)
    for suffix in FLIP_SUFFIXES:
        fold |= {f"invariance/{g}_{suffix}": v for g, v in zip(GROUPS, (so2, so3, lorentz))}

    return fold


def _delta_fold(**per_group: float) -> dict[str, float]:
    """One fold's deltas against its baseline, the same value for every family of a group."""
    fold: dict[str, float] = {}
    for group, value in per_group.items():
        fold |= {f"invariance/{group}_delta_{s}": value for s in DELTA_SUFFIXES}

    return fold


def _fill(callback: ResultsTableCallback, *invariance: dict[str, float], **overrides) -> None:
    """Load the state the fit and test hooks would have collected over the folds."""
    absent = [float("nan")] * len(invariance)
    callback._fold_invariance = list(invariance)
    callback._fold_test_acc = overrides.get("test_acc", [0.9] * len(invariance))
    callback._fold_flops = overrides.get("flops", [1000.0] * len(invariance))
    callback._fold_baseline_test_acc = overrides.get("baseline_test_acc", absent)
    callback._fold_delta_test_acc = overrides.get("delta_test_acc", absent)


def test_every_group_keeps_its_per_fold_values_beside_the_mean(tmp_path: Path) -> None:
    """The paired bootstrap of the delta columns reads the fold values, not the summary."""
    callback = _callback(tmp_path, n_folds=4)
    _fill(
        callback,
        _wasserstein_fold(0.1, 0.2, 0.3),
        _wasserstein_fold(0.15, 0.25, 0.35),
        _wasserstein_fold(0.2, 0.3, 0.4),
        _wasserstein_fold(float("nan"), 0.25, 0.35),
        test_acc=[0.9, 0.91, 0.92, 0.93],
    )
    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["so2_wasserstein_folds"] == "0.100000;0.150000;0.200000;nan"
    assert row["so3_wasserstein_folds"] == "0.200000;0.250000;0.300000;0.250000"
    assert row["lorentz_wasserstein_folds"] == "0.300000;0.350000;0.400000;0.350000"
    # A NaN fold averaged in would silently drag the whole row towards zero.
    assert float(row["so2_wasserstein_mean"]) == pytest.approx((0.1 + 0.15 + 0.2) / 3)
    assert row["folds_used"] == "4"
    assert row["model"] == "mlp"
    # The published tables are appended to, so the column order is part of the interface.
    with open(callback.output_path, newline="") as table:
        assert next(csv.reader(table)) == _expected_header()


def test_the_accuracy_carries_its_own_interval_over_the_folds(tmp_path: Path) -> None:
    """The interval is on the accuracy itself, so a run with no baseline still reports one."""
    callback = _callback(tmp_path, n_folds=4)
    _fill(callback, *([_fold(0.3, 0.6, 0.9)] * 4), test_acc=[0.90, 0.91, 0.92, 0.93])
    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert float(row["test_acc_ci_lower"]) < 0.915 < float(row["test_acc_ci_upper"])
    assert float(row["test_acc_ci_lower"]) >= 0.90


@pytest.mark.parametrize(
    "scheme, mode, expected, expected_key",
    [
        # A run is judged on the group it was trained on: an SO(2)-augmented run must not be
        # scored on the Lorentz probe it never saw.
        ("so2", "scheme", 0.3, "invariance/so2_wasserstein"),
        # A run trained on no symmetry belongs to none of them, so all three are averaged, as
        # mean_all does whatever the run was trained on.
        ("none", "scheme", MEAN_OF_THREE, "invariance/mean_wasserstein"),
        ("so2", "mean_all", MEAN_OF_THREE, "invariance/mean_wasserstein"),
    ],
)
def test_the_selected_columns_follow_the_regime_or_average_over_it(
    tmp_path: Path, scheme: str, mode: str, expected: float, expected_key: str
) -> None:
    """The selected_inv_* columns are the one number a table is read across regimes by."""
    callback = _callback(
        tmp_path, augmentation_scheme=scheme, selected_metric_mode=mode, n_folds=1
    )
    _fill(callback, _fold(0.3, 0.6, 0.9))
    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert float(row["flip_recover_mean"]) == pytest.approx(expected)
    assert row["selected_inv_key"] == expected_key


def test_the_deltas_are_read_off_the_group_the_run_was_trained_on(tmp_path: Path) -> None:
    """An SO(2)-augmented run is compared with its baseline on SO(2), never on the other two."""
    callback = _callback(tmp_path, augmentation_scheme="so2", n_folds=3)
    folds = [
        _fold(0.3, 0.6, 0.9) | _delta_fold(so2=value, so3=-1.0, lorentz=-1.0)
        for value in (0.04, 0.05, 0.06)
    ]
    _fill(callback, *folds, baseline_test_acc=[0.8] * 3, delta_test_acc=[0.1] * 3)
    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["delta_wasserstein_folds"] == "0.040000;0.050000;0.060000"
    assert float(row["delta_flip_recover_mean"]) == pytest.approx(0.05)
    assert float(row["delta_flip_ci_lower"]) <= 0.05 <= float(row["delta_flip_ci_upper"])
    assert float(row["baseline_test_acc_mean"]) == pytest.approx(0.8)
    assert float(row["delta_test_acc_mean"]) == pytest.approx(0.1)


def test_a_delta_is_called_significant_only_when_the_folds_agree_on_it(tmp_path: Path) -> None:
    """The bootstrap is paired over folds, so a consistent gain is what earns the label."""
    agreed = _callback(tmp_path, n_folds=3)
    _fill(agreed, *[_delta_fold(so2=v) for v in (0.04, 0.05, 0.06)])
    agreed._maybe_write_summary()
    scattered = _callback(tmp_path, n_folds=3)
    _fill(scattered, *[_delta_fold(so2=v) for v in (-0.45, 0.55, 0.05)])
    scattered._maybe_write_summary()
    rows = _rows(agreed.output_path)

    assert rows[0]["delta_flip_significant_boot_005"] == "yes"
    assert float(rows[0]["delta_flip_bootstrap_pvalue"]) < 0.05
    # The same mean, ten times the spread: three folds cannot call that a difference.
    assert rows[1]["delta_flip_significant_boot_005"] == "no"


def test_a_baseline_row_carries_no_comparison_of_its_own(tmp_path: Path) -> None:
    """A baseline has no baseline, so its delta columns must read nan rather than zero."""
    callback = _callback(tmp_path, augmentation_scheme="none", n_folds=2)
    _fill(callback, _fold(0.3, 0.6, 0.9), _fold(0.3, 0.6, 0.9))
    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["baseline_test_acc_mean"] == "nan"
    assert row["baseline_test_acc_folds"] == "nan;nan"
    assert row["baseline_test_acc_std"] == "nan"
    assert row["delta_test_acc_std"] == "nan"
    assert row["delta_test_acc_pm"] == "nan"
    assert row["delta_flip_folds"] == "nan;nan"
    assert row["delta_flip_ci_lower"] == "nan"
    assert row["delta_flip_bootstrap_pvalue"] == "nan"
    # A run that reported nothing is not a run that reported no difference.
    assert row["delta_flip_significant_boot_005"] == "no"


def test_the_regime_and_the_feature_list_name_the_table(tmp_path: Path) -> None:
    """Three tables per feature list, so a baseline and an SCL run never share a file."""
    features = ["px", "py", "pz", "e"]
    baseline = _callback(tmp_path, augmentation_scheme="none", features=features)
    scl = _callback(tmp_path, method="symm_loss", augmentation_scheme="so3", features=features)

    assert baseline.display_method == "baseline"
    assert baseline.output_path == tmp_path / "px_py_pz_e" / "results_table_baseline.csv"
    assert scl.output_path == tmp_path / "px_py_pz_e" / "results_table_symm_loss.csv"
    # Nothing is created before a row is written, so a search leaves results/ untouched.
    assert not (tmp_path / "px_py_pz_e").exists()


def test_the_row_is_written_once_the_folds_are_in_and_a_second_run_appends(tmp_path: Path) -> None:
    """A row written after the first fold would carry no spread and no warning that it does not."""
    callback = _callback(tmp_path, n_folds=2)
    _fill(callback, _fold(0.4, 0.5, 0.6), test_acc=[0.9], flops=[1000.0])
    callback._maybe_write_summary()

    assert not callback.output_path.exists()

    _fill(callback, _fold(0.4, 0.5, 0.6), _fold(0.4, 0.5, 0.6), test_acc=[0.9, 0.92])
    callback._maybe_write_summary()
    second = _callback(tmp_path, n_folds=1)
    _fill(second, _fold(0.4, 0.5, 0.6), test_acc=[0.95])
    second._maybe_write_summary()
    rows = _rows(callback.output_path)

    assert len(rows) == 2
    assert rows[0]["folds_used"] == "2"
    assert rows[0]["test_acc_folds"] == "0.900000;0.920000"
    assert rows[1]["test_acc_mean"] == "0.95"


def test_appending_under_a_different_header_is_refused(tmp_path: Path) -> None:
    """csv.DictWriter would otherwise file every value under the wrong old column."""
    callback = _callback(tmp_path, n_folds=1)
    callback.output_path.parent.mkdir(parents=True, exist_ok=True)
    callback.output_path.write_text("method,model\nbaseline,mlp\n")
    _fill(callback, _fold(0.4, 0.5, 0.6))

    with pytest.raises(ValueError, match="misalign"):
        callback._maybe_write_summary()
