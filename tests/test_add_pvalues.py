"""The paired bootstrap that turns a results table into the paper's significance columns.

``scripts/analysis/add_pvalues.py`` is a script rather than a module of ``src``, so it is loaded by
path. Its arithmetic decides which augmentation-versus-baseline differences the paper calls
significant.
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


def test_the_reported_difference_is_the_improvement_over_the_baseline() -> None:
    """A positive delta must mean the augmented run moved its logits less, not more."""
    index = {("mlp", "16", "so2"): [1.0] * 5}
    row = _row(augmentation_scheme="so2", so2_wasserstein_folds=";".join(["0.5"] * 5))

    out = add_pvalues.compute_delta_for_row(row, index, "wasserstein", n_boot=200)

    assert float(out["delta_wasserstein_mean"]) == 0.5
    # A baseline has no baseline, and 'permute' is a retired scheme kept for old tables.
    for scheme in ("none", "permute"):
        blank = add_pvalues.compute_delta_for_row(
            _row(augmentation_scheme=scheme), index, "wasserstein", n_boot=200
        )
        assert blank["delta_wasserstein_mean"] == "nan"
        assert blank["delta_wasserstein_pm"] == "n/a"

    # Silently reporting no difference would read as a run that changed nothing.
    absent = add_pvalues.compute_delta_for_row(
        _row(augmentation_scheme="so3"), {}, "wasserstein", n_boot=200
    )
    assert absent["delta_wasserstein_bootstrap_pvalue"] == "no_baseline"


def test_a_difference_is_called_significant_only_when_it_holds_across_the_folds() -> None:
    """The bootstrap is paired over folds, so a consistent offset is what earns the label."""
    row = _row(augmentation_scheme="so2", so2_wasserstein_folds=BASELINE_FOLDS)
    identical = add_pvalues.compute_delta_for_row(
        row, {("mlp", "16", "so2"): [1.0] * 5}, "wasserstein", n_boot=2000
    )
    offset = add_pvalues.compute_delta_for_row(
        row, {("mlp", "16", "so2"): [3.0] * 5}, "wasserstein", n_boot=2000
    )

    assert identical["delta_wasserstein_significant_boot_005"] == "no"
    assert offset["delta_wasserstein_significant_boot_005"] == "yes"


def test_a_whole_table_is_annotated_in_one_pass() -> None:
    """What the script does to a real table: the family detected off the columns, only the
    unaugmented rows indexed as baselines, then every row annotated."""
    rows = [
        _row(augmentation_scheme="none"),
        _row(augmentation_scheme="so2", so2_wasserstein_folds=";".join(["0.5"] * 5)),
    ]
    index = add_pvalues.build_metric_baseline_index(rows, "wasserstein")

    # An SO(2) row taken as its own baseline would compare a run against itself.
    assert add_pvalues._detect_inv_metric_suffix(rows) == "wasserstein"
    assert add_pvalues._detect_inv_metric_suffix([{"method": "augmentation"}]) is None
    assert len(index) == 3
    assert all(index[("mlp", "16", group)] == [1.0] * 5 for group in ("so2", "so3", "lorentz"))

    annotated = add_pvalues.annotate_rows(
        rows,
        add_pvalues.build_baseline_index(rows),
        n_boot=200,
        metric_baseline_index=index,
        inv_metric="wasserstein",
    )
    by_scheme = {row["augmentation_scheme"]: row for row in annotated}

    assert by_scheme["so2"]["delta_wasserstein_mean"] == "0.500000"
    assert by_scheme["none"]["delta_wasserstein_pm"] == "n/a"
