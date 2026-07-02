import csv
import math
from pathlib import Path

import pytest

from src.callbacks.results_table_callback import ResultsTableCallback


def _make_callback(tmp_path: Path, **kwargs) -> ResultsTableCallback:
    defaults = dict(
        output_path=str(tmp_path / "results_table.csv"),
        n_folds=3,
        method="augmentation",
        model_name="src.models.mlp_module.MLPClassifier",
        augmentation_scheme="so2",
        nconstituents=8,
        invariance_metric="wasserstein",
    )
    defaults.update(kwargs)
    return ResultsTableCallback(**defaults)


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_per_group_folds_written_and_match_source_values(tmp_path: Path) -> None:
    cb = _make_callback(tmp_path)

    cb._fold_invariance = [
        {"invariance/so2_wasserstein": 0.1, "invariance/so3_wasserstein": 0.2, "invariance/lorentz_wasserstein": 0.3},
        {"invariance/so2_wasserstein": 0.15, "invariance/so3_wasserstein": 0.25, "invariance/lorentz_wasserstein": 0.35},
        {"invariance/so2_wasserstein": 0.2, "invariance/so3_wasserstein": 0.3, "invariance/lorentz_wasserstein": 0.4},
    ]
    cb._fold_test_acc = [0.9, 0.91, 0.92]
    cb._fold_flops = [1000.0, 1000.0, 1000.0]

    cb._maybe_write_summary()

    rows = _read_rows(cb.output_path)
    assert len(rows) == 1
    row = rows[0]

    assert row["so2_wasserstein_folds"] == "0.100000;0.150000;0.200000"
    assert row["so3_wasserstein_folds"] == "0.200000;0.250000;0.300000"
    assert row["lorentz_wasserstein_folds"] == "0.300000;0.350000;0.400000"
    assert float(row["so2_wasserstein_mean"]) == (0.1 + 0.15 + 0.2) / 3


def test_nan_entries_serialize_as_nan_and_excluded_from_mean(tmp_path: Path) -> None:
    cb = _make_callback(tmp_path)

    cb._fold_invariance = [
        {"invariance/so2_wasserstein": 0.1, "invariance/so3_wasserstein": 0.2, "invariance/lorentz_wasserstein": 0.3},
        {"invariance/so2_wasserstein": float("nan"), "invariance/so3_wasserstein": 0.25, "invariance/lorentz_wasserstein": 0.35},
        {"invariance/so2_wasserstein": 0.2, "invariance/so3_wasserstein": 0.3, "invariance/lorentz_wasserstein": 0.4},
    ]
    cb._fold_test_acc = [0.9, 0.91, 0.92]
    cb._fold_flops = [1000.0, 1000.0, 1000.0]

    cb._maybe_write_summary()

    rows = _read_rows(cb.output_path)
    row = rows[0]

    assert row["so2_wasserstein_folds"] == "0.100000;nan;0.200000"
    assert float(row["so2_wasserstein_mean"]) == pytest.approx((0.1 + 0.2) / 2)


def _flip_inv_fold(so2: float, so3: float, lorentz: float) -> dict:
    return {
        "invariance/so2_flip_recover": so2, "invariance/so3_flip_recover": so3, "invariance/lorentz_flip_recover": lorentz,
        "invariance/so2_flip_stable": so2, "invariance/so3_flip_stable": so3, "invariance/lorentz_flip_stable": lorentz,
        "invariance/so2_flip_degrade": so2, "invariance/so3_flip_degrade": so3, "invariance/lorentz_flip_degrade": lorentz,
        "invariance/so2_wasserstein": 0.0, "invariance/so3_wasserstein": 0.0, "invariance/lorentz_wasserstein": 0.0,
    }


def test_flip_breakdown_picks_matching_scheme(tmp_path: Path) -> None:
    cb = _make_callback(tmp_path, augmentation_scheme="so2")
    cb._fold_invariance = [_flip_inv_fold(0.4, 0.5, 0.6)]
    cb._fold_test_acc = [0.9]
    cb._fold_flops = [1000.0]
    cb.n_folds = 1

    cb._maybe_write_summary()

    row = _read_rows(cb.output_path)[0]
    assert float(row["flip_recover_mean"]) == pytest.approx(0.4)
    assert float(row["flip_stable_mean"]) == pytest.approx(0.4)
    assert float(row["flip_degrade_mean"]) == pytest.approx(0.4)


def test_flip_breakdown_baseline_uses_mean_across_groups(tmp_path: Path) -> None:
    cb = _make_callback(tmp_path, augmentation_scheme="none")
    cb._fold_invariance = [_flip_inv_fold(0.3, 0.6, 0.9)]
    cb._fold_test_acc = [0.9]
    cb._fold_flops = [1000.0]
    cb.n_folds = 1

    cb._maybe_write_summary()

    row = _read_rows(cb.output_path)[0]
    assert float(row["flip_recover_mean"]) == pytest.approx((0.3 + 0.6 + 0.9) / 3)


def test_flip_breakdown_permute_is_nan(tmp_path: Path) -> None:
    cb = _make_callback(tmp_path, augmentation_scheme="permute")
    cb._fold_invariance = [_flip_inv_fold(0.3, 0.6, 0.9)]
    cb._fold_test_acc = [0.9]
    cb._fold_flops = [1000.0]
    cb.n_folds = 1

    cb._maybe_write_summary()

    row = _read_rows(cb.output_path)[0]
    assert row["flip_recover_mean"] == "nan"
