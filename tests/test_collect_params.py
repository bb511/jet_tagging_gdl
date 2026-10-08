import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.utils.checkpoints import Manifest, ManifestRun, write_manifest
from tests.conftest import ROOT

# scripts/ is not a package, so the module is loaded by path.
_SPEC = importlib.util.spec_from_file_location(
    "collect_params", ROOT / "scripts" / "analysis" / "collect_params.py"
)
collect_params = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(collect_params)

TRAIN = Path("logs/train/hls4ml")


def _store(root: Path, run_id: str, params: dict[str, str]) -> None:
    for key, value in params.items():
        path = root / "logs/mlflow/mlruns/123" / run_id / "params" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)


def _manifest(
    root: Path, run_dir: str, run_id: str | None, fold_idx: int | None = None
) -> Path:
    directory = root / TRAIN / run_dir
    label = "main" if fold_idx is None else f"fold_{fold_idx}"
    run = ManifestRun(label, fold_idx, str(directory / "last.ckpt"), 0.7, run_id)
    write_manifest(Manifest(0, str(directory), "hls4ml", [run]), directory)
    return directory


def _network(root: Path, run_id: str, name: str, n: int, total: int) -> None:
    _store(
        root,
        run_id,
        {
            "algorithm/name": name,
            "algorithm/nconstituents": "${data.nconstituents}",
            "data/nconstituents": str(n),
            "algorithm/params/total": str(total),
            "algorithm/flops": f"{10.0 * total}",
        },
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    _manifest(tmp_path, "vn/nconst150/baseline", "vn150")
    _network(tmp_path, "vn150", "vn", 150, 300)
    _manifest(tmp_path, "vn/nconst16/baseline_1block", "vn16")
    _network(tmp_path, "vn16", "vn", 16, 100)
    _manifest(tmp_path, "vn/nconst16/baseline_1block/fold_1", "vn16f1", fold_idx=1)
    _network(tmp_path, "vn16f1", "vn", 16, 100)
    _manifest(tmp_path, "vn_hgq/nconst16/baseline_beta1e-08", "hgq")
    _network(tmp_path, "hgq", "vn_hgq", 16, 0)
    reference = {"algorithm/init_from": str(TRAIN / "vn/nconst16/baseline_1block")}
    _store(tmp_path, "hgq", reference)
    _manifest(tmp_path, "vn_hgq/nconst16/baseline_orphan", "orphan")
    _network(tmp_path, "orphan", "vn_hgq", 16, 0)
    _store(tmp_path, "orphan", {"algorithm/init_from": str(TRAIN / "vn/nconst16/gone")})
    _manifest(tmp_path, "jet_level_bdt_all53/nconst150/baseline", "bdt")
    _network(tmp_path, "bdt", "jet_level_bdt_all53", 150, 0)
    _store(tmp_path, "bdt", {"algorithm/flops": "nan"})
    _manifest(tmp_path, "pt_eta_phi/deepsets/nconst50/aug_so2", "ds")
    _network(tmp_path, "ds", "deepsets", 50, 40)
    _manifest(tmp_path, "deepsets/nconst16/symm_lorentz", "absent")
    return tmp_path


def _named(rows: list[dict]) -> list[tuple]:
    return [(row["label"], row["model"], row["nconstituents"]) for row in rows]


def test_one_row_per_main_run_in_order(
    tree: Path, capsys: pytest.CaptureFixture
) -> None:
    rows = collect_params.collect([tree])

    # The count sorts as a number: 16 before 150, which a string sort reverses.
    assert _named(rows) == [
        ("default", "deepsets", 16),
        ("default", "jet_level_bdt_all53", 150),
        ("default", "vn", 16),
        ("default", "vn", 150),
        ("default", "vn_hgq", 16),
        ("default", "vn_hgq", 16),
        ("pt_eta_phi", "deepsets", 50),
    ]
    absent, bdt, vn16, vn150, hgq, orphan, ds = rows
    assert (vn16["variant"], vn16["regime"], vn16["augmentation_scheme"]) == (
        "baseline_1block",
        "baseline",
        "none",
    )
    assert (vn16["params"], vn16["flops"], vn16["mlflow_run_id"]) == (
        "100",
        "1000.0",
        "vn16",
    )
    assert vn16["run_dir"] == tree / TRAIN / "vn/nconst16/baseline_1block"
    assert (ds["regime"], ds["augmentation_scheme"], ds["params"]) == (
        "augmentation",
        "so2",
        "40",
    )
    assert (absent["regime"], absent["augmentation_scheme"]) == ("symm_loss", "lorentz")
    # A quantised row carries the count of the float network it was copied from, and
    # nan, with a warning, when that network has no manifest; the trees have no count.
    assert (hgq["params"], orphan["params"]) == ("100", "nan")
    assert (bdt["params"], bdt["flops"]) == ("nan", "nan")
    # A run the store lacks is a row of nan and a warning, never an exception.
    assert (absent["params"], absent["flops"], absent["mlflow_run_id"]) == (
        "nan",
        "nan",
        "absent",
    )
    warnings = capsys.readouterr().err
    assert "no run absent" in warnings
    assert "behind init_from=logs/train/hls4ml/vn/nconst16/gone," in warnings
    # The unresolved algorithm/nconstituents is not what the count is checked against.
    assert warnings.count("[warn]") == 2


def test_a_manifest_without_a_run_id_and_a_store_that_disagrees_are_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # The manifests written before they recorded the MLflow run lack the key.
    directory = _manifest(tmp_path, "lorentznet/nconst50/baseline_r1", None)
    path = directory / "checkpoint_manifest.json"
    payload = json.loads(path.read_text())
    del payload["runs"][0]["mlflow_run_id"]
    path.write_text(json.dumps(payload))
    _manifest(tmp_path, "mlp/nconst16/baseline", "mlp")
    _network(tmp_path, "mlp", "deepsets", 50, 10)

    rows = collect_params.collect([tmp_path])

    assert [(row["model"], row["mlflow_run_id"]) for row in rows] == [
        ("lorentznet", "nan"),
        ("mlp", "mlp"),
    ]
    assert rows[0]["params"] == "nan"
    warnings = capsys.readouterr().err
    assert "no run None" in warnings
    assert "model=deepsets, its path says mlp" in warnings
    assert "nconstituents=50, its path says 16" in warnings


def test_the_table_carries_the_columns_in_order(tree: Path, monkeypatch) -> None:
    out = tree / "out" / "params.csv"
    monkeypatch.setattr(
        sys, "argv", ["collect_params.py", str(tree), "--out", str(out)]
    )

    collect_params.main()

    with open(out, newline="") as table:
        rows = list(csv.DictReader(table))
    assert list(rows[0]) == collect_params.COLUMNS
    assert [row["nconstituents"] for row in rows] == "16 150 16 150 16 16 50".split()
