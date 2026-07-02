import math

from src.utils.add_pvalues import (
    _detect_inv_metric_suffix,
    annotate_rows,
    build_baseline_index,
    build_metric_baseline_index,
    compute_delta_for_row,
)


def _row(**kwargs) -> dict:
    base = {
        "method": "augmentation",
        "model": "src.models.mlp_module.MLPClassifier",
        "augmentation_scheme": "none",
        "nconstituents": "8",
        "test_acc_folds": "0.9;0.9;0.9;0.9;0.9",
        "so2_wasserstein_folds": "1.0;1.0;1.0;1.0;1.0",
        "so3_wasserstein_folds": "1.0;1.0;1.0;1.0;1.0",
        "lorentz_wasserstein_folds": "1.0;1.0;1.0;1.0;1.0",
    }
    base.update(kwargs)
    return base


def test_detect_inv_metric_suffix_finds_wasserstein() -> None:
    rows = [_row()]
    assert _detect_inv_metric_suffix(rows) == "wasserstein"


def test_detect_inv_metric_suffix_returns_none_when_absent() -> None:
    rows = [{"method": "augmentation", "model": "m", "augmentation_scheme": "none"}]
    assert _detect_inv_metric_suffix(rows) is None
    assert _detect_inv_metric_suffix([]) is None


def test_build_metric_baseline_index_keys_and_excludes_non_baseline() -> None:
    rows = [
        _row(augmentation_scheme="none", nconstituents="8"),
        _row(augmentation_scheme="so2", nconstituents="8", so2_wasserstein_folds="0.5;0.5;0.5;0.5;0.5"),
    ]
    index = build_metric_baseline_index(rows, "wasserstein")

    model = "src.models.mlp_module.MLPClassifier"
    assert index[(model, "8", "so2")] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert index[(model, "8", "so3")] == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert index[(model, "8", "lorentz")] == [1.0, 1.0, 1.0, 1.0, 1.0]
    # the so2-scheme row itself must not be treated as a baseline entry
    assert len(index) == 3


def test_compute_delta_sign_convention() -> None:
    model = "m"
    index = {(model, "8", "so2"): [1.0] * 5}
    row = _row(model=model, nconstituents="8", augmentation_scheme="so2",
               so2_wasserstein_folds="0.5;0.5;0.5;0.5;0.5")

    out = compute_delta_for_row(row, index, "wasserstein", n_boot=200)

    assert float(out["delta_wasserstein_mean"]) == 0.5


def test_compute_delta_scope_blank_for_none_and_permute() -> None:
    index = {("m", "8", "so2"): [1.0] * 5}

    for scheme in ("none", "permute"):
        row = _row(model="m", nconstituents="8", augmentation_scheme=scheme)
        out = compute_delta_for_row(row, index, "wasserstein", n_boot=200)
        assert out["delta_wasserstein_mean"] == "nan"
        assert out["delta_wasserstein_pm"] == "n/a"


def test_compute_delta_missing_baseline() -> None:
    row = _row(model="m", nconstituents="8", augmentation_scheme="so3")
    out = compute_delta_for_row(row, {}, "wasserstein", n_boot=200)
    assert out["delta_wasserstein_bootstrap_pvalue"] == "no_baseline"


def test_compute_delta_pvalue_direction() -> None:
    model = "m"
    n = 5

    # Identical baseline/augmented folds -> delta ~ 0 -> high p-value, not significant.
    index_same = {(model, "8", "so2"): [1.0] * n}
    row_same = _row(model=model, nconstituents="8", augmentation_scheme="so2",
                     so2_wasserstein_folds=";".join(["1.0"] * n))
    out_same = compute_delta_for_row(row_same, index_same, "wasserstein", n_boot=2000)
    assert out_same["delta_wasserstein_significant_boot_005"] == "no"

    # Large, consistent offset across all folds -> low p-value, significant.
    index_diff = {(model, "8", "so2"): [3.0] * n}
    row_diff = _row(model=model, nconstituents="8", augmentation_scheme="so2",
                     so2_wasserstein_folds=";".join(["1.0"] * n))
    out_diff = compute_delta_for_row(row_diff, index_diff, "wasserstein", n_boot=2000)
    assert out_diff["delta_wasserstein_significant_boot_005"] == "yes"


def test_annotate_rows_end_to_end_adds_delta_columns_only_for_aug_row() -> None:
    baseline_row = _row(augmentation_scheme="none", nconstituents="8")
    so2_row = _row(augmentation_scheme="so2", nconstituents="8",
                    so2_wasserstein_folds="0.5;0.5;0.5;0.5;0.5")
    rows = [baseline_row, so2_row]

    baseline_index = build_baseline_index(rows)
    metric_baseline_index = build_metric_baseline_index(rows, "wasserstein")

    annotated = annotate_rows(
        rows, baseline_index, n_boot=200,
        metric_baseline_index=metric_baseline_index, inv_metric="wasserstein",
    )

    annotated_baseline = next(r for r in annotated if r["augmentation_scheme"] == "none")
    annotated_so2 = next(r for r in annotated if r["augmentation_scheme"] == "so2")

    assert annotated_so2["delta_wasserstein_mean"] == "0.500000"
    assert annotated_baseline["delta_wasserstein_pm"] == "n/a"
