import csv
from pathlib import Path

import pytest

from src.symmetry import TABULATED_FAMILIES, TABULATED_GROUPS, key
from src.utils.results_table import ResultsTable
from tests.conftest import results_header

MEAN_OF_THREE = (0.3 + 0.6 + 0.9) / 3
# The groupless families and the value every fold below reports for each.
CLEAN = {"rejection_es30": 20.0, "tpr_eb1": 0.3, "tpr_eb10": 0.6, "auc": 0.9}


def _table(tmp_path: Path, **overrides) -> ResultsTable:
    defaults = {
        "output_path": str(tmp_path / "results_table.csv"),
        "method": "augmentation",
        "model_name": "mlp",
        "augmentation_scheme": "so2",
        "nconstituents": 8,
        "invariance_metric": "tv",
    }
    return ResultsTable(**(defaults | overrides))


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as table:
        return list(csv.DictReader(table))


def _header(path: Path) -> list[str]:
    with open(path, newline="") as table:
        return next(csv.reader(table))


def _tv_fold(so2: float, so3: float, lorentz: float, **rest) -> dict[str, object]:
    fold = {key(g, "tv"): v for g, v in zip(TABULATED_GROUPS, (so2, so3, lorentz))}
    fold |= {key(None, family): value for family, value in CLEAN.items()}
    costs = {"flops_total": 1000.0, "ebops_total": 64000.0}
    return fold | {"test/acc": 0.9} | costs | rest


def _fold(so2: float, so3: float, lorentz: float, **rest) -> dict[str, object]:
    fold = _tv_fold(0.0, 0.0, 0.0)
    for suffix in TABULATED_FAMILIES:
        fold |= {
            key(g, suffix): v for g, v in zip(TABULATED_GROUPS, (so2, so3, lorentz))
        }
    return fold | rest


def test_every_group_keeps_its_per_fold_values_beside_the_mean(tmp_path: Path) -> None:
    table = _table(tmp_path)
    folds = [
        _tv_fold(0.1, 0.2, 0.3, **{"test/acc": 0.90}),
        _tv_fold(0.15, 0.25, 0.35, **{"test/acc": 0.91}, ebops_total=32000.0),
        _tv_fold(0.2, 0.3, 0.4, **{"test/acc": 0.92}),
        _tv_fold(float("nan"), 0.25, 0.35, **{"test/acc": 0.93}),
    ]
    # A fold whose module could not be counted reports no eBOPs at all.
    del folds[3]["ebops_total"]
    table.write(folds)
    row = _rows(table.output_path)[0]

    assert row["so2_tv_folds"] == "0.100000;0.150000;0.200000;nan"
    assert row["so3_tv_folds"] == "0.200000;0.250000;0.300000;0.250000"
    assert row["lorentz_tv_folds"] == "0.300000;0.350000;0.400000;0.350000"
    # The NaN fold is left out of the mean, not averaged in.
    assert float(row["so2_tv_mean"]) == pytest.approx((0.1 + 0.15 + 0.2) / 3)
    assert float(row["test_acc_ci_lower"]) < 0.915 < float(row["test_acc_ci_upper"])
    assert float(row["test_acc_ci_lower"]) >= 0.90
    assert row["folds_used"] == "4"
    assert row["model"] == "mlp"
    assert row["flops_mean"] == "1000.0"
    # And so is the fold without eBOPs: (64000 + 32000 + 64000) / 3.
    assert row["ebops_mean"] == "53333.333"
    for family, value in CLEAN.items():
        assert float(row[f"{family}_mean"]) == pytest.approx(value)
        assert row[f"{family}_folds"] == ";".join([f"{value:.6f}"] * 4)
    assert _header(table.output_path) == results_header()


@pytest.mark.parametrize(
    "scheme, expected, expected_key",
    [
        ("so2", 0.3, "invariance/so2_tv"),
        # A run trained on no group belongs to none, so all three are averaged.
        ("none", MEAN_OF_THREE, "invariance/mean_tv"),
    ],
)
def test_the_selected_columns_follow_the_regime_or_average_over_it(
    tmp_path: Path, scheme: str, expected: float, expected_key: str
) -> None:
    table = _table(tmp_path, augmentation_scheme=scheme)
    table.write([_fold(0.3, 0.6, 0.9)])
    row = _rows(table.output_path)[0]

    assert float(row["flip_recover_mean"]) == pytest.approx(expected)
    assert row["selected_inv_key"] == expected_key


def test_every_tabulated_suffix_carries_a_family_per_group(tmp_path: Path) -> None:
    table = _table(tmp_path, augmentation_scheme="so2")
    table.write([_fold(0.3, 0.6, 0.9), _fold(0.5, 0.6, 0.9)])
    row = _rows(table.output_path)[0]

    for suffix in TABULATED_FAMILIES:
        assert float(row[f"so2_{suffix}_mean"]) == pytest.approx(0.4)
        assert row[f"so3_{suffix}_folds"] == "0.600000;0.600000"
        assert row[f"lorentz_{suffix}_pm"] == "0.9000 +/- 0.0000"
        # The run was trained on SO(2), so the group-less family repeats its SO(2) columns.
        assert float(row[f"{suffix}_mean"]) == pytest.approx(0.4)

    # A metric that is itself tabulated fills one family, not two. Its own directory: the header
    # differs from the one just written, and a table is only ever appended to under its own.
    flip_table = _table(tmp_path / "flip", invariance_metric="flip")
    flip_table.write([_fold(0.3, 0.6, 0.9)])
    header = _header(flip_table.output_path)

    assert header.count("so2_flip_folds") == 1
    # The selected columns already report the flip of the group the run is judged on.
    assert "flip_mean" not in header
    assert _rows(flip_table.output_path)[0]["selected_inv_key"] == "invariance/so2_flip"


def test_the_regime_and_the_feature_list_or_the_subdir_name_the_table(
    tmp_path: Path,
) -> None:
    features = ["px", "py", "pz", "e"]
    baseline = _table(tmp_path, augmentation_scheme="none", features=features)
    scl = _table(
        tmp_path, method="symm_loss", augmentation_scheme="so3", features=features
    )
    embedded = _table(
        tmp_path, augmentation_scheme="none", features=features, subdir="embedded"
    )
    ablation = _table(
        tmp_path, augmentation_scheme="none", features=["pt", "eta", "phi"]
    )

    assert baseline.display_method == "baseline"
    assert (
        baseline.output_path == tmp_path / "px_py_pz_e" / "results_table_baseline.csv"
    )
    assert scl.output_path == tmp_path / "px_py_pz_e" / "results_table_symm_loss.csv"
    assert embedded.output_path == tmp_path / "embedded" / "results_table_baseline.csv"
    # One directory per feature set, so an ablation never appends to the main tables.
    assert (
        ablation.output_path == tmp_path / "pt_eta_phi" / "results_table_baseline.csv"
    )
    # Nothing is created before a row is written, so a search leaves results/ untouched.
    assert not (tmp_path / "px_py_pz_e").exists()


def test_an_evaluation_of_no_fold_writes_nothing_and_a_second_run_appends(
    tmp_path: Path,
) -> None:
    table = _table(tmp_path)
    table.write([])

    assert not table.output_path.exists()

    folds = [_fold(0.4, 0.5, 0.6, **{"test/acc": acc}) for acc in (0.90, 0.92)]
    table.write(folds)
    _table(tmp_path).write([_fold(0.4, 0.5, 0.6, **{"test/acc": 0.95})])
    rows = _rows(table.output_path)

    assert len(rows) == 2
    assert rows[0]["folds_used"] == "2"
    assert rows[0]["test_acc_folds"] == "0.900000;0.920000"
    assert rows[1]["test_acc_mean"] == "0.95"

    # A table whose header is not this row's is never appended to.
    other = _table(tmp_path, method="symm_loss")
    other.output_path.parent.mkdir(parents=True, exist_ok=True)
    other.output_path.write_text("method,model\nbaseline,mlp\n")

    with pytest.raises(ValueError, match="misalign"):
        other.write([_fold(0.4, 0.5, 0.6)])
