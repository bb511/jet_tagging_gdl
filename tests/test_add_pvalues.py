"""The paired bootstrap that turns a results table into the paper's significance columns.

``scripts/analysis/add_pvalues.py`` is a script rather than a module of ``src``, so it is loaded by
path. Its arithmetic decides which augmentation-versus-baseline differences the paper calls
significant, which is worth pinning against hand-computed cases.
"""

import importlib.util
from types import ModuleType

from tests.conftest import ROOT

BASELINE_FOLDS = ";".join(["1.0"] * 5)


def _load_script() -> ModuleType:
    """Import ``scripts/analysis/add_pvalues.py`` under a name of its own."""
    path = ROOT / "scripts" / "analysis" / "add_pvalues.py"
    spec = importlib.util.spec_from_file_location("add_pvalues", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


add_pvalues = _load_script()


def _row(**overrides) -> dict[str, str]:
    """One results-table row, with every column the bootstrap reads."""
    row = {
        "method": "augmentation",
        "model": "mlp",
        "augmentation_scheme": "none",
        "nconstituents": "16",
        "test_acc_folds": ";".join(["0.9"] * 5),
        "so2_wasserstein_folds": BASELINE_FOLDS,
        "so3_wasserstein_folds": BASELINE_FOLDS,
        "lorentz_wasserstein_folds": BASELINE_FOLDS,
    }

    return row | overrides


def test_the_invariance_family_is_read_off_the_columns() -> None:
    """The table names its family in its column suffixes, and the script follows it."""
    assert add_pvalues._detect_inv_metric_suffix([_row()]) == "wasserstein"
    assert add_pvalues._detect_inv_metric_suffix([{"method": "augmentation"}]) is None
    assert add_pvalues._detect_inv_metric_suffix([]) is None


def test_only_the_unaugmented_rows_become_baselines() -> None:
    """An SO(2) row taken as its own baseline would compare a run against itself."""
    rows = [
        _row(augmentation_scheme="none"),
        _row(augmentation_scheme="so2", so2_wasserstein_folds=";".join(["0.5"] * 5)),
    ]
    index = add_pvalues.build_metric_baseline_index(rows, "wasserstein")

    assert index[("mlp", "16", "so2")] == [1.0] * 5
    assert index[("mlp", "16", "so3")] == [1.0] * 5
    assert index[("mlp", "16", "lorentz")] == [1.0] * 5
    assert len(index) == 3


def test_the_reported_difference_is_the_improvement_over_the_baseline() -> None:
    """A positive delta must mean the augmented run moved its logits less, not more."""
    row = _row(augmentation_scheme="so2", so2_wasserstein_folds=";".join(["0.5"] * 5))
    index = {("mlp", "16", "so2"): [1.0] * 5}

    out = add_pvalues.compute_delta_for_row(row, index, "wasserstein", n_boot=200)

    assert float(out["delta_wasserstein_mean"]) == 0.5


def test_a_row_with_no_group_to_compare_against_is_left_blank() -> None:
    """A baseline has no baseline, and 'permute' is a retired scheme kept for old tables."""
    index = {("mlp", "16", "so2"): [1.0] * 5}

    for scheme in ("none", "permute"):
        out = add_pvalues.compute_delta_for_row(
            _row(augmentation_scheme=scheme), index, "wasserstein", n_boot=200
        )
        assert out["delta_wasserstein_mean"] == "nan"
        assert out["delta_wasserstein_pm"] == "n/a"


def test_a_missing_baseline_is_said_to_be_missing() -> None:
    """Silently reporting no difference would read as a run that changed nothing."""
    out = add_pvalues.compute_delta_for_row(
        _row(augmentation_scheme="so3"), {}, "wasserstein", n_boot=200
    )

    assert out["delta_wasserstein_bootstrap_pvalue"] == "no_baseline"


def test_a_difference_is_called_significant_only_when_it_holds_across_the_folds() -> None:
    """The bootstrap is paired over folds, so a consistent offset is what earns the label."""
    identical = add_pvalues.compute_delta_for_row(
        _row(augmentation_scheme="so2", so2_wasserstein_folds=BASELINE_FOLDS),
        {("mlp", "16", "so2"): [1.0] * 5},
        "wasserstein",
        n_boot=2000,
    )
    offset = add_pvalues.compute_delta_for_row(
        _row(augmentation_scheme="so2", so2_wasserstein_folds=BASELINE_FOLDS),
        {("mlp", "16", "so2"): [3.0] * 5},
        "wasserstein",
        n_boot=2000,
    )

    assert identical["delta_wasserstein_significant_boot_005"] == "no"
    assert offset["delta_wasserstein_significant_boot_005"] == "yes"


def test_a_whole_table_is_annotated_in_one_pass() -> None:
    """What the script does to a real table: baselines indexed, then every row annotated."""
    rows = [
        _row(augmentation_scheme="none"),
        _row(augmentation_scheme="so2", so2_wasserstein_folds=";".join(["0.5"] * 5)),
    ]
    annotated = add_pvalues.annotate_rows(
        rows,
        add_pvalues.build_baseline_index(rows),
        n_boot=200,
        metric_baseline_index=add_pvalues.build_metric_baseline_index(rows, "wasserstein"),
        inv_metric="wasserstein",
    )

    by_scheme = {row["augmentation_scheme"]: row for row in annotated}

    assert by_scheme["so2"]["delta_wasserstein_mean"] == "0.500000"
    assert by_scheme["none"]["delta_wasserstein_pm"] == "n/a"
