"""One row per run: how the folds are aggregated, and where the row is written.

The results tables are the paper's numbers, and a run that has already cost GPU-days writes its row
once. Every quantity below is therefore pinned against hand-computed values rather than against
whatever the callback happens to produce.
"""

import csv
from pathlib import Path

import pytest

from src.callbacks.results_table import ResultsTableCallback

GROUPS = ("so2", "so3", "lorentz")
FLIP_SUFFIXES = ("flip_recover", "flip_stable", "flip_degrade")

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


def _fold(so2: float, so3: float, lorentz: float) -> dict[str, float]:
    """One fold's invariance metrics, the same value for every family of a given group."""
    fold = {f"invariance/{group}_wasserstein": 0.0 for group in GROUPS}
    for suffix in FLIP_SUFFIXES:
        fold.update(
            {
                f"invariance/so2_{suffix}": so2,
                f"invariance/so3_{suffix}": so3,
                f"invariance/lorentz_{suffix}": lorentz,
            }
        )

    return fold


def _fill(callback: ResultsTableCallback, *invariance: dict[str, float], **overrides) -> None:
    """Load the state the fit and test hooks would have collected over the folds."""
    callback._fold_invariance = list(invariance)
    callback._fold_test_acc = overrides.get("test_acc", [0.9] * len(invariance))
    callback._fold_flops = overrides.get("flops", [1000.0] * len(invariance))


def test_every_group_keeps_its_per_fold_values_beside_the_mean(tmp_path: Path) -> None:
    """The paired bootstrap in add_pvalues.py reads the fold values, not the summary."""
    callback = _callback(tmp_path)
    callback._fold_invariance = [
        {f"invariance/{group}_wasserstein": value for group, value in zip(GROUPS, values)}
        for values in ([0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4])
    ]
    callback._fold_test_acc = [0.9, 0.91, 0.92]
    callback._fold_flops = [1000.0, 1000.0, 1000.0]

    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["so2_wasserstein_folds"] == "0.100000;0.150000;0.200000"
    assert row["so3_wasserstein_folds"] == "0.200000;0.250000;0.300000"
    assert row["lorentz_wasserstein_folds"] == "0.300000;0.350000;0.400000"
    assert float(row["so2_wasserstein_mean"]) == (0.1 + 0.15 + 0.2) / 3
    assert row["folds_used"] == "3"
    assert row["model"] == "mlp"


def test_a_fold_that_reported_no_value_is_left_out_of_the_mean(tmp_path: Path) -> None:
    """A NaN averaged in would silently drag a whole row towards zero."""
    callback = _callback(tmp_path)
    folds = [
        {f"invariance/{group}_wasserstein": value for group, value in zip(GROUPS, values)}
        for values in ([0.1, 0.2, 0.3], [float("nan"), 0.25, 0.35], [0.2, 0.3, 0.4])
    ]
    callback._fold_invariance = folds
    callback._fold_test_acc = [0.9, 0.91, 0.92]
    callback._fold_flops = [1000.0, 1000.0, 1000.0]

    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["so2_wasserstein_folds"] == "0.100000;nan;0.200000"
    assert float(row["so2_wasserstein_mean"]) == pytest.approx((0.1 + 0.2) / 2)


def test_a_run_is_judged_on_the_group_it_was_trained_on(tmp_path: Path) -> None:
    """An SO(2)-augmented run must not be scored on the Lorentz probe it never saw."""
    callback = _callback(tmp_path, augmentation_scheme="so2", n_folds=1)
    _fill(callback, _fold(0.4, 0.5, 0.6))

    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert float(row["flip_recover_mean"]) == pytest.approx(0.4)
    assert float(row["flip_stable_mean"]) == pytest.approx(0.4)
    assert float(row["flip_degrade_mean"]) == pytest.approx(0.4)


def test_a_baseline_is_judged_on_the_mean_over_the_three_groups(tmp_path: Path) -> None:
    """A run trained on no symmetry belongs to none of them, so all three are averaged."""
    callback = _callback(tmp_path, augmentation_scheme="none", n_folds=1)
    _fill(callback, _fold(0.3, 0.6, 0.9))

    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert float(row["flip_recover_mean"]) == pytest.approx((0.3 + 0.6 + 0.9) / 3)


def test_mean_all_averages_the_three_groups_whatever_the_run_was_trained_on(
    tmp_path: Path,
) -> None:
    """The other way to fill the selected columns, for a table read across regimes."""
    callback = _callback(
        tmp_path, augmentation_scheme="so2", selected_metric_mode="mean_all", n_folds=1
    )
    _fill(callback, _fold(0.3, 0.6, 0.9))

    callback._maybe_write_summary()
    row = _rows(callback.output_path)[0]

    assert row["selected_inv_key"] == "invariance/mean_wasserstein"
    assert float(row["flip_recover_mean"]) == pytest.approx((0.3 + 0.6 + 0.9) / 3)


def test_the_regime_and_the_feature_list_name_the_table(tmp_path: Path) -> None:
    """Three tables per feature list, so a baseline and an SCL run never share a file."""
    baseline = _callback(tmp_path, augmentation_scheme="none", features=["px", "py", "pz", "e"])
    scl = _callback(
        tmp_path,
        method="symm_loss",
        augmentation_scheme="so3",
        features=["px", "py", "pz", "e"],
    )

    assert baseline.display_method == "baseline"
    assert baseline.output_path == tmp_path / "px_py_pz_e" / "results_table_baseline.csv"
    assert scl.output_path == tmp_path / "px_py_pz_e" / "results_table_symm_loss.csv"
    # Nothing is created before a row is written, so a search leaves results/ untouched.
    assert not (tmp_path / "px_py_pz_e").exists()


def test_the_row_is_written_only_once_every_fold_is_in(tmp_path: Path) -> None:
    """A row written after the first fold would carry no spread and no warning that it does not."""
    callback = _callback(tmp_path, n_folds=2)
    _fill(callback, _fold(0.4, 0.5, 0.6), test_acc=[0.9], flops=[1000.0])

    callback._maybe_write_summary()
    assert not callback.output_path.exists()

    _fill(callback, _fold(0.4, 0.5, 0.6), _fold(0.4, 0.5, 0.6), test_acc=[0.9, 0.92])
    callback._maybe_write_summary()
    rows = _rows(callback.output_path)

    assert len(rows) == 1
    assert rows[0]["folds_used"] == "2"
    assert rows[0]["test_acc_folds"] == "0.900000;0.920000"


def test_a_second_run_appends_under_the_same_header(tmp_path: Path) -> None:
    """A regime's table collects every architecture and constituent count of that regime."""
    for accuracy in (0.9, 0.95):
        callback = _callback(tmp_path, n_folds=1)
        _fill(callback, _fold(0.4, 0.5, 0.6), test_acc=[accuracy])
        callback._maybe_write_summary()

    rows = _rows(callback.output_path)

    assert len(rows) == 2
    assert [row["test_acc_mean"] for row in rows] == ["0.9", "0.95"]


def test_appending_under_a_different_header_is_refused(tmp_path: Path) -> None:
    """csv.DictWriter would otherwise file every value under the wrong old column."""
    callback = _callback(tmp_path, n_folds=1)
    callback.output_path.parent.mkdir(parents=True, exist_ok=True)
    callback.output_path.write_text("method,model\nbaseline,mlp\n")
    _fill(callback, _fold(0.4, 0.5, 0.6))

    with pytest.raises(ValueError, match="misalign"):
        callback._maybe_write_summary()


def test_the_column_order_is_the_one_the_tables_on_disk_carry(tmp_path: Path) -> None:
    """The published tables are appended to, so the order is part of the interface."""
    callback = _callback(tmp_path, n_folds=1)
    _fill(callback, _fold(0.4, 0.5, 0.6))

    callback._maybe_write_summary()

    with open(callback.output_path, newline="") as table:
        assert next(csv.reader(table)) == LEGACY_HEADER
