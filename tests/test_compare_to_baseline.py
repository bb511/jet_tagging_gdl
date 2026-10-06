import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT

# scripts/ is not a package, so the module is loaded by path, with its own directory put on the
# import path first: running it as a script is what normally puts it there, and it imports common
# from beside itself.
ANALYSIS = ROOT / "scripts" / "analysis"
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

_SPEC = importlib.util.spec_from_file_location(
    "compare_to_baseline", ANALYSIS / "compare_to_baseline.py"
)
compare_to_baseline = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_to_baseline)
read_csv = compare_to_baseline.common.read_csv

# A retention column before the TV one, as in the tables: the invariance metric is still
# found among them.
COLUMNS = [
    "method",
    "model",
    "nconstituents",
    "augmentation_scheme",
    "test_acc_folds",
    "rejection_es30_folds",
    "tpr_eb1_folds",
    "tpr_eb10_folds",
    "auc_folds",
    "so2_tpr_retention_eb1_folds",
    "so2_fpr_transformed_eb1_folds",
    "so2_tv_folds",
    "so2_flip_stable_folds",
]
GROUPLESS = ["test_acc", "rejection_es30", "tpr_eb1", "tpr_eb10", "auc"]

FOLDS = 5


def _spread(value: float) -> list[float]:
    # A flat series would put every bootstrap resample on the same mean.
    return [value + 0.001 * (fold - 2) for fold in range(FOLDS)]


def _join(values: list[float]) -> str:
    return ";".join(f"{value:.6f}" for value in values)


def _row(
    model: str,
    *,
    acc: float,
    tv: float,
    stable: float,
    rejection: float = 40.0,
    tpr: float = 0.4,
    auc: float = 0.9,
    retention: float = 0.8,
    nconstituents: int = 16,
    scheme: str = "so2",
    method: str = "augmentation",
    folds: int = FOLDS,
) -> dict[str, str]:
    values = {
        "test_acc": acc,
        "rejection_es30": rejection,
        "tpr_eb1": tpr,
        "tpr_eb10": tpr,
        "auc": auc,
        "so2_tpr_retention_eb1": retention,
        "so2_fpr_transformed_eb1": 0.02,
        "so2_tv": tv,
        "so2_flip_stable": stable,
    }
    head = {
        "method": method,
        "model": model,
        "nconstituents": str(nconstituents),
        "augmentation_scheme": scheme,
    }
    folded = {f"{name}_folds": _join(_spread(v)[:folds]) for name, v in values.items()}
    return head | folded


def _table(path: Path, rows: list[dict[str, str]]) -> Path:
    with open(path, "w", newline="") as table:
        writer = csv.DictWriter(table, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _compare(
    tmp_path: Path, run_rows: list[dict], baseline_rows: list[dict]
) -> list[dict]:
    run = read_csv(_table(tmp_path / "run.csv", run_rows))
    baseline = read_csv(_table(tmp_path / "baseline.csv", baseline_rows))
    metrics = compare_to_baseline.metrics_covered(run + baseline)
    return compare_to_baseline.compare(run, baseline, metrics)


def _cell(
    rows: list[dict], metric: str, group: str = "so2", model: str = "mlp"
) -> dict:
    wanted = (model, group, metric)
    return next(r for r in rows if (r["model"], r["group"], r["metric"]) == wanted)


def test_rows_are_paired_on_the_model_and_the_constituent_count(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    run_rows = [
        _row("mlp", acc=0.75, tv=0.10, stable=0.90),
        _row("deepsets", acc=0.80, tv=0.05, stable=0.95),
    ]
    baseline_rows = [
        _row("deepsets", acc=0.78, tv=0.20, stable=0.80, scheme="none"),
        _row("mlp", acc=0.70, tv=0.20, stable=0.80, scheme="none"),
        _row("jetformer", acc=0.81, tv=0.20, stable=0.80, scheme="none"),
    ]

    rows = _compare(tmp_path, run_rows, baseline_rows)

    assert {row["model"] for row in rows} == {"mlp", "deepsets"}
    # Only the columns both tables carry: the run's groupless metrics, then SO(2) thrice
    # over. The diagnostic has no better direction and is never compared.
    assert {(row["group"], row["metric"]) for row in rows} == {
        *(("", metric) for metric in GROUPLESS),
        ("so2", "tpr_retention_eb1"),
        ("so2", "tv"),
        ("so2", "flip_stable"),
    }
    # The run's own naming columns, never the baseline's.
    assert _cell(rows, "test_acc", group="")["augmentation_scheme"] == "so2"

    # A run the baseline table never trained is reported rather than dropped in silence.
    rows = _compare(tmp_path, run_rows, baseline_rows[:1])
    reported = capsys.readouterr().out
    assert {row["model"] for row in rows} == {"deepsets"}
    assert "[skip]" in reported and "model=mlp" in reported

    # And a pair is only a pair at the same constituent count.
    elsewhere = [
        _row("mlp", acc=0.7, tv=0.2, stable=0.8, nconstituents=50, scheme="none")
    ]
    assert _compare(tmp_path, [run_rows[0]], elsewhere) == []


def test_positive_always_means_the_run_is_the_better_model(tmp_path: Path) -> None:
    # The run is better on every figure but the stable fraction, which is smaller.
    run_rows = [
        _row(
            "mlp",
            acc=0.75,
            tv=0.10,
            stable=0.70,
            rejection=50.0,
            tpr=0.5,
            auc=0.95,
            retention=0.9,
        )
    ]
    baseline_rows = [_row("mlp", acc=0.70, tv=0.20, stable=0.80, scheme="none")]

    rows = _compare(tmp_path, run_rows, baseline_rows)

    assert float(_cell(rows, "tv")["delta_mean"]) == pytest.approx(0.10)
    assert float(_cell(rows, "flip_stable")["delta_mean"]) == pytest.approx(-0.10)
    assert float(_cell(rows, "tpr_retention_eb1")["delta_mean"]) == pytest.approx(0.1)
    groupless = {metric: _cell(rows, metric, group="") for metric in GROUPLESS}
    assert float(groupless["test_acc"]["delta_mean"]) == pytest.approx(0.05)
    assert float(groupless["rejection_es30"]["delta_mean"]) == pytest.approx(10.0)
    assert float(groupless["tpr_eb1"]["delta_mean"]) == pytest.approx(0.1)
    assert float(groupless["tpr_eb10"]["delta_mean"]) == pytest.approx(0.1)
    assert float(groupless["auc"]["delta_mean"]) == pytest.approx(0.05)


def test_the_summary_columns_follow_the_per_fold_differences(tmp_path: Path) -> None:
    run_rows = [_row("mlp", acc=0.75, tv=0.10, stable=0.90)]
    baseline_rows = [_row("mlp", acc=0.70, tv=0.20, stable=0.80, scheme="none")]

    row = _cell(_compare(tmp_path, run_rows, baseline_rows), "tv")

    # The jitter is common to both tables, so every fold differs by the same amount.
    assert row["delta_folds"] == ";".join(["0.100000"] * FOLDS)
    assert row["delta_pm"] == "0.1000 +/- 0.0000"
    assert float(row["ci_lower"]) == pytest.approx(0.10)
    assert float(row["ci_upper"]) == pytest.approx(0.10)
    assert row["significant_boot_005"] == "yes"

    # One fold leaves the test undecided rather than negative.
    single = _compare(
        tmp_path,
        [_row("mlp", acc=0.75, tv=0.10, stable=0.90, folds=1)],
        [_row("mlp", acc=0.70, tv=0.20, stable=0.80, scheme="none", folds=1)],
    )
    assert math.isnan(float(_cell(single, "tv")["bootstrap_pvalue"]))
    assert _cell(single, "tv")["significant_boot_005"] == "nan"

    # Fold counts that disagree cannot be paired at all.
    with pytest.raises(ValueError, match="paired fold by fold"):
        _compare(
            tmp_path,
            [_row("mlp", acc=0.75, tv=0.10, stable=0.90, folds=5)],
            [_row("mlp", acc=0.70, tv=0.2, stable=0.80, scheme="none", folds=3)],
        )


def test_the_default_output_is_named_after_both_tables(
    tmp_path: Path, monkeypatch
) -> None:
    run = _table(
        tmp_path / "results_table_augmentation.csv",
        [_row("mlp", acc=0.75, tv=0.10, stable=0.90)],
    )
    baseline = _table(
        tmp_path / "results_table_baseline.csv",
        [_row("mlp", acc=0.70, tv=0.20, stable=0.80, scheme="none")],
    )
    monkeypatch.setattr(
        sys, "argv", ["compare_to_baseline.py", str(run), str(baseline)]
    )

    compare_to_baseline.main()

    written = tmp_path / "deltas_augmentation_vs_baseline.csv"
    with open(written, newline="") as table:
        rows = list(csv.DictReader(table))
    assert list(rows[0]) == compare_to_baseline.COLUMNS
    assert [row["metric"] for row in rows if row["group"] == ""] == GROUPLESS


def test_a_named_baseline_model_pairs_every_run_row_with_it(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch
) -> None:
    run_rows = [
        _row("deepsets_moments", acc=0.82, tv=0.0, stable=1.0, nconstituents=150),
        _row("vn", acc=0.825, tv=0.0, stable=1.0, nconstituents=150),
        _row("vn", acc=0.80, tv=0.0, stable=1.0, nconstituents=16),
    ]
    # The jet-level rows are never transformed: five NaNs in every per-group cell.
    ceiling = _row(
        "jet_level_noeta52", acc=0.831, tv=0.0, stable=1.0, nconstituents=150
    )
    nan = ";".join(["nan"] * FOLDS)
    ceiling |= {name: nan for name in COLUMNS if name.startswith("so2_")}
    # A row of the run's own model, which the named baseline overrides.
    own = _row("vn", acc=0.70, tv=0.0, stable=1.0, nconstituents=150)
    run = _table(tmp_path / "results_table_embedded.csv", run_rows)
    baseline = _table(tmp_path / "results_table_baseline.csv", [ceiling, own])

    rows = compare_to_baseline.compare(
        read_csv(run),
        read_csv(baseline),
        compare_to_baseline.metrics_covered(read_csv(run) + read_csv(baseline)),
        baseline_model="jet_level_noeta52",
    )

    assert {(row["model"], row["nconstituents"]) for row in rows} == {
        ("deepsets_moments", "150"),
        ("vn", "150"),
    }
    # The NaN cells pass the fold-count check and drop out, leaving the groupless rows.
    assert {(row["group"], row["metric"]) for row in rows} == {
        ("", metric) for metric in GROUPLESS
    }
    vn = _cell(rows, "test_acc", group="", model="vn")
    assert float(vn["delta_mean"]) == pytest.approx(0.825 - 0.831)
    reported = capsys.readouterr().out
    assert "[skip]" in reported and "nconstituents=16" in reported

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_to_baseline.py",
            str(run),
            str(baseline),
            "--baseline-model",
            "jet_level_noeta52",
        ],
    )
    compare_to_baseline.main()

    written = tmp_path / "deltas_embedded_vs_baseline_jet_level_noeta52.csv"
    assert len(read_csv(written)) == len(rows)
