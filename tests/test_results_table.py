"""One row per run: how the folds are aggregated, and where the row is written.

The results tables are the paper's numbers, and a run that has already cost GPU-days writes its row
once, so every quantity below is pinned against hand-computed values.
"""

import csv
from pathlib import Path

import pytest

from src.callbacks.results_table import ResultsTableCallback

GROUPS = ("so2", "so3", "lorentz")
FLIP_SUFFIXES = ("flip_recover", "flip_stable", "flip_degrade")
MEAN_OF_THREE = (0.3 + 0.6 + 0.9) / 3

# The header of the tables written before the refactor, which new runs still append to.
LEGACY_HEADER = (
    "method,model,augmentation_scheme,nconstituents,folds_used,test_acc_mean,test_acc_std,"
    "test_acc_pm,selected_inv_key,selected_inv_mean,selected_inv_std,selected_inv_pm,"
    "so2_wasserstein_mean,so2_wasserstein_std,so2_wasserstein_pm,so3_wasserstein_mean,"
    "so3_wasserstein_std,so3_wasserstein_pm,lorentz_wasserstein_mean,lorentz_wasserstein_std,"
    "lorentz_wasserstein_pm,so2_wasserstein_folds,so3_wasserstein_folds,"
    "lorentz_wasserstein_folds,flip_recover_mean,flip_recover_std,flip_recover_pm,"
    "flip_recover_folds,flip_stable_mean,flip_stable_std,flip_stable_pm,flip_stable_folds,"
    "flip_degrade_mean,flip_degrade_std,flip_degrade_pm,flip_degrade_folds,"
    "so2_flip_recover_mean,so2_flip_recover_std,so2_flip_recover_pm,so2_flip_recover_folds,"
    "so3_flip_recover_mean,so3_flip_recover_std,so3_flip_recover_pm,so3_flip_recover_folds,"
    "lorentz_flip_recover_mean,lorentz_flip_recover_std,lorentz_flip_recover_pm,"
    "lorentz_flip_recover_folds,so2_flip_stable_mean,so2_flip_stable_std,so2_flip_stable_pm,"
    "so2_flip_stable_folds,so3_flip_stable_mean,so3_flip_stable_std,so3_flip_stable_pm,"
    "so3_flip_stable_folds,lorentz_flip_stable_mean,lorentz_flip_stable_std,"
    "lorentz_flip_stable_pm,lorentz_flip_stable_folds,so2_flip_degrade_mean,"
    "so2_flip_degrade_std,so2_flip_degrade_pm,so2_flip_degrade_folds,so3_flip_degrade_mean,"
    "so3_flip_degrade_std,so3_flip_degrade_pm,so3_flip_degrade_folds,"
    "lorentz_flip_degrade_mean,lorentz_flip_degrade_std,lorentz_flip_degrade_pm,"
    "lorentz_flip_degrade_folds,flops_mean,test_acc_folds,selected_inv_folds"
).split(",")


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


def _fill(callback: ResultsTableCallback, *invariance: dict[str, float], **overrides) -> None:
    """Load the state the fit and test hooks would have collected over the folds."""
    callback._fold_invariance = list(invariance)
    callback._fold_test_acc = overrides.get("test_acc", [0.9] * len(invariance))
    callback._fold_flops = overrides.get("flops", [1000.0] * len(invariance))


def test_every_group_keeps_its_per_fold_values_beside_the_mean(tmp_path: Path) -> None:
    """The paired bootstrap in add_pvalues.py reads the fold values, not the summary."""
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
        assert next(csv.reader(table)) == LEGACY_HEADER


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
