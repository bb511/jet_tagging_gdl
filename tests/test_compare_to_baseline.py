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

COLUMNS = [
    "method",
    "model",
    "nconstituents",
    "augmentation_scheme",
    "test_acc_folds",
    "so2_wasserstein_folds",
    "so2_flip_stable_folds",
]

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
    wasserstein: float,
    stable: float,
    nconstituents: int = 16,
    scheme: str = "so2",
    method: str = "augmentation",
    folds: int = FOLDS,
) -> dict[str, str]:
    return {
        "method": method,
        "model": model,
        "nconstituents": str(nconstituents),
        "augmentation_scheme": scheme,
        "test_acc_folds": _join(_spread(acc)[:folds]),
        "so2_wasserstein_folds": _join(_spread(wasserstein)[:folds]),
        "so2_flip_stable_folds": _join(_spread(stable)[:folds]),
    }


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
        _row("mlp", acc=0.75, wasserstein=0.10, stable=0.90),
        _row("deepsets", acc=0.80, wasserstein=0.05, stable=0.95),
    ]
    baseline_rows = [
        _row("deepsets", acc=0.78, wasserstein=0.20, stable=0.80, scheme="none"),
        _row("mlp", acc=0.70, wasserstein=0.20, stable=0.80, scheme="none"),
        _row("jetformer", acc=0.81, wasserstein=0.20, stable=0.80, scheme="none"),
    ]

    rows = _compare(tmp_path, run_rows, baseline_rows)

    assert {row["model"] for row in rows} == {"mlp", "deepsets"}
    # Only the columns both tables carry: the run's own accuracy, and SO(2) twice over.
    assert {(row["group"], row["metric"]) for row in rows} == {
        ("", "test_acc"),
        ("so2", "wasserstein"),
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
        _row(
            "mlp", acc=0.7, wasserstein=0.2, stable=0.8, nconstituents=50, scheme="none"
        )
    ]
    assert _compare(tmp_path, [run_rows[0]], elsewhere) == []


def test_positive_always_means_the_run_is_the_better_model(tmp_path: Path) -> None:
    # The run is better on the distance (smaller) and worse on the stable fraction (smaller).
    run_rows = [_row("mlp", acc=0.75, wasserstein=0.10, stable=0.70)]
    baseline_rows = [
        _row("mlp", acc=0.70, wasserstein=0.20, stable=0.80, scheme="none")
    ]

    rows = _compare(tmp_path, run_rows, baseline_rows)

    assert float(_cell(rows, "wasserstein")["delta_mean"]) == pytest.approx(0.10)
    assert float(_cell(rows, "flip_stable")["delta_mean"]) == pytest.approx(-0.10)
    assert float(_cell(rows, "test_acc", group="")["delta_mean"]) == pytest.approx(0.05)


def test_the_summary_columns_follow_the_per_fold_differences(tmp_path: Path) -> None:
    run_rows = [_row("mlp", acc=0.75, wasserstein=0.10, stable=0.90)]
    baseline_rows = [
        _row("mlp", acc=0.70, wasserstein=0.20, stable=0.80, scheme="none")
    ]

    row = _cell(_compare(tmp_path, run_rows, baseline_rows), "wasserstein")

    # The jitter is common to both tables, so every fold differs by the same amount.
    assert row["delta_folds"] == ";".join(["0.100000"] * FOLDS)
    assert row["delta_pm"] == "0.1000 +/- 0.0000"
    assert float(row["ci_lower"]) == pytest.approx(0.10)
    assert float(row["ci_upper"]) == pytest.approx(0.10)
    assert row["significant_boot_005"] == "yes"

    # One fold leaves the test undecided rather than negative.
    single = _compare(
        tmp_path,
        [_row("mlp", acc=0.75, wasserstein=0.10, stable=0.90, folds=1)],
        [_row("mlp", acc=0.70, wasserstein=0.20, stable=0.80, scheme="none", folds=1)],
    )
    assert math.isnan(float(_cell(single, "wasserstein")["bootstrap_pvalue"]))
    assert _cell(single, "wasserstein")["significant_boot_005"] == "nan"

    # Fold counts that disagree cannot be paired at all.
    with pytest.raises(ValueError, match="paired fold by fold"):
        _compare(
            tmp_path,
            [_row("mlp", acc=0.75, wasserstein=0.10, stable=0.90, folds=5)],
            [
                _row(
                    "mlp",
                    acc=0.70,
                    wasserstein=0.2,
                    stable=0.80,
                    scheme="none",
                    folds=3,
                )
            ],
        )


def test_the_default_output_is_named_after_both_tables(
    tmp_path: Path, monkeypatch
) -> None:
    run = _table(
        tmp_path / "results_table_augmentation.csv",
        [_row("mlp", acc=0.75, wasserstein=0.10, stable=0.90)],
    )
    baseline = _table(
        tmp_path / "results_table_baseline.csv",
        [_row("mlp", acc=0.70, wasserstein=0.20, stable=0.80, scheme="none")],
    )
    monkeypatch.setattr(
        sys, "argv", ["compare_to_baseline.py", str(run), str(baseline)]
    )

    compare_to_baseline.main()

    written = tmp_path / "deltas_augmentation_vs_baseline.csv"
    with open(written, newline="") as table:
        rows = list(csv.DictReader(table))
    assert list(rows[0]) == compare_to_baseline.COLUMNS
    assert [row["metric"] for row in rows if row["group"] == ""] == ["test_acc"]
