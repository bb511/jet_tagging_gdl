import fcntl
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.algorithms.utils.schedules import warmup_restarts_decay
from src.data import top_tagging_datamodule as top_tagging
from src.data.components.augmentation import compute_eta, compute_phi, compute_pt
from src.data.hls4ml_datamodule import fold_indices
from src.data.top_tagging_datamodule import TopTaggingDataModule
from tests.conftest import (
    TOP_TAGGING_COLUMNS,
    TOP_TAGGING_N,
    TOP_TAGGING_PLANTED,
    TOP_TAGGING_SIZES,
    compose_root,
    point_into,
    read_results_row,
    results_header,
    top_tagging_raw,
    train_main_and_folds,
)

SPLITS = {"train": "train.h5", "validation": "val.h5", "test": "test.h5"}
AXIS = [TOP_TAGGING_COLUMNS.index(name) for name in ("etarel", "phirel")]


def _cache(root: Path, prefix: str) -> dict[str, np.ndarray]:
    folder = root / f"nconst_{TOP_TAGGING_N}"
    return {split: np.load(folder / f"{prefix}_{split}.npy") for split in SPLITS}


def _meta(root: Path) -> dict:
    return json.loads((root / f"nconst_{TOP_TAGGING_N}" / "meta.json").read_text())


def _leading(name: str) -> np.ndarray:
    # The leading constituents of every jet of raw file *name*, ranked by p_T one jet at
    # a time: [E, px, py, pz] in GeV, (jets, N, 4), zero-padded.
    p4, _ = top_tagging_raw(name)
    kept = np.zeros((len(p4), TOP_TAGGING_N, 4))
    for jet, constituents in enumerate(p4.astype(np.float64)):
        real = [c for c in constituents if c[0] > 0]
        ranked = sorted(real, key=lambda c: -math.hypot(c[1], c[2]))[:TOP_TAGGING_N]
        kept[jet, : len(ranked)] = ranked
    return kept


def _angles(p4: np.ndarray) -> np.ndarray:
    px, py, pz = (torch.from_numpy(p4[..., i]) for i in (1, 2, 3))
    columns = (compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py))
    return torch.stack(columns, dim=-1).numpy()


def _counts(name: str) -> dict[str, object]:
    p4, flags = top_tagging_raw(name)
    multiplicity = (p4[..., 0] > 0).sum(axis=1)
    return {
        "n_jets": len(p4),
        "class_counts": {"qcd": int((flags == 0).sum()), "top": int(flags.sum())},
        "jets_resorted": len(TOP_TAGGING_PLANTED),
        "jets_truncated": int((multiplicity > TOP_TAGGING_N).sum()),
        "mean_multiplicity": pytest.approx(multiplicity.mean()),
        "max_multiplicity": 20,
    }


def test_the_conversion_keeps_the_leading_constituents_at_the_records_scales(
    top_tagging_cache: Path,
) -> None:
    meta = _meta(top_tagging_cache)
    x, y = (_cache(top_tagging_cache, prefix) for prefix in ("x", "y"))
    params = meta["norm_params"]
    scales = np.array([params[name][1] for name in TOP_TAGGING_COLUMNS])

    assert (meta["format_version"], meta["nconstituents"]) == (1, TOP_TAGGING_N)
    assert meta["columns"] == list(TOP_TAGGING_COLUMNS)
    assert meta["class_names"] == ["qcd", "top"]
    for split, name in SPLITS.items():
        expected, (_, flags) = _leading(name), top_tagging_raw(name)
        real = expected[..., 0] > 0
        gev = x[split].astype(np.float64) * scales
        assert meta["splits"][split] == _counts(name)
        assert x[split].shape == (len(flags), TOP_TAGGING_N, len(TOP_TAGGING_COLUMNS))
        assert x[split].dtype == np.float32 and not gev[~real].any()
        # Sorted, truncated, and back in GeV to the float32 rounding of the division.
        assert np.allclose(gev[..., [3, 0, 1, 2]], expected, rtol=1e-6, atol=1e-6)
        assert np.allclose(gev[..., 4:7], _angles(expected), rtol=1e-6, atol=1e-6)
        assert np.array_equal(y[split], np.eye(2, dtype=np.float32)[flags])
    # Fitted on the training split, padded zeros included, as the hls4ml record fits
    # them: a non-negative column reaches 1, any other spans 2.
    for i, name in enumerate(TOP_TAGGING_COLUMNS):
        stored = x["train"][..., i]
        positive = name in ("pt", "e")
        span = stored.max() - (0.0 if positive else stored.min())
        assert params[name][0] == ("positive" if positive else "centered")
        assert span == pytest.approx(1.0 if positive else 2.0, rel=1e-6)


def _datamodule(cache: Path, **overrides) -> TopTaggingDataModule:
    defaults = {
        "cache_dir": str(cache),
        "nconstituents": TOP_TAGGING_N,
        "batch_size": 8,
        "num_workers": 0,
        "pin_memory": False,
    }
    return TopTaggingDataModule(**(defaults | overrides))


def test_the_datamodule_serves_the_cache_as_it_serves_the_hls4ml_record(
    top_tagging_cache: Path,
) -> None:
    datamodule = _datamodule(
        top_tagging_cache, n_folds=3, fold_idx=1, augmentation_scheme="so2"
    )
    datamodule.prepare_data()
    datamodule.setup()
    x, y = (_cache(top_tagging_cache, prefix) for prefix in ("x", "y"))
    pooled = [np.concatenate([a["train"], a["validation"]]) for a in (x, y)]
    train_idx, val_idx = fold_indices(len(pooled[0]), 3, 42, 1)
    params = _meta(top_tagging_cache)["norm_params"]
    widths = {
        split: next(iter(loader))[0].shape[-1]
        for split, loader in (
            ("train", datamodule.train_dataloader()),
            ("validation", datamodule.val_dataloader()),
            ("test", datamodule.test_dataloader()),
        )
    }
    served = datamodule.data_test.X
    absolute = torch.from_numpy(pooled[0][train_idx][..., :7])

    assert widths == {"train": 4, "validation": 4, "test": len(TOP_TAGGING_COLUMNS)}
    assert datamodule.class_names == ("qcd", "top")
    assert datamodule.norm_scales == {n: scale for n, (_, scale) in params.items()}
    assert datamodule.data_train.transform is not None
    # The inherited rebase recomputes the axis columns the conversion stored.
    stored = torch.from_numpy(x["test"])
    assert torch.allclose(served[..., AXIS], stored[..., AXIS], atol=1e-5)
    assert torch.equal(served[..., :7], stored[..., :7])
    # Fold 2 of 3 of the pooled training and validation jets, in that order.
    assert torch.equal(datamodule.data_train.X[..., :7], absolute)
    assert torch.equal(datamodule.data_val.y, torch.from_numpy(pooled[1][val_idx]))


def test_the_cache_is_built_once_and_refused_when_it_cannot_be_trusted(
    top_tagging_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = top_tagging_cache / f"nconst_{TOP_TAGGING_N}"
    stamps = {path.name: path.stat().st_mtime_ns for path in folder.iterdir()}
    for _ in range(2):
        _datamodule(top_tagging_cache).prepare_data()
    unchanged = {path.name: path.stat().st_mtime_ns for path in folder.iterdir()}
    source, fetched = top_tagging_cache / "source", top_tagging_cache / "fetched"
    source.mkdir()
    (source / "val.h5").write_bytes(b"not the record")
    monkeypatch.setattr(top_tagging, "RECORD_URL", f"file://{source}/{{name}}")
    monkeypatch.setattr(top_tagging, "RAW_FILES", {"val.h5": "0" * 32})

    assert unchanged == stamps
    with pytest.raises(ValueError, match="full_axis"):
        _datamodule(top_tagging_cache, full_axis=True).prepare_data()
    with pytest.raises(ValueError, match="md5"):
        top_tagging.download(fetched)
    assert not any(fetched.iterdir())
    meta = json.loads((folder / "meta.json").read_text()) | {"format_version": 0}
    (folder / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="format"):
        _datamodule(top_tagging_cache).prepare_data()


def test_a_preparation_waits_for_the_process_that_holds_the_lock(
    top_tagging_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The synthetic raw files pass the md5 check, so nothing is downloaded, and the
    # cache at N = 4 is missing.
    raw = top_tagging_cache / "raw"
    md5s = {name: top_tagging._md5(raw / name) for name in SPLITS.values()}
    monkeypatch.setattr(top_tagging, "RAW_FILES", md5s)
    datamodule = _datamodule(top_tagging_cache, nconstituents=4)

    with ThreadPoolExecutor(max_workers=1) as pool:
        with open(top_tagging_cache / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            preparation = pool.submit(datamodule.prepare_data)
            with pytest.raises(TimeoutError):
                preparation.result(timeout=1.0)
            assert not any(top_tagging_cache.glob("nconst_4*"))
        preparation.result(timeout=60.0)

    assert [path.name for path in top_tagging_cache.glob("nconst_4*")] == ["nconst_4"]
    assert (top_tagging_cache / "nconst_4" / "meta.json").is_file()


def test_the_lorentznet_schedule_warms_up_restarts_and_decays() -> None:
    # 35 epochs of 10 steps: warm-up to epoch 4, cosine cycles from epochs 4, 8 and 16,
    # decay from 32 off the rate at epoch 31, 15 epochs into the third cycle's 16.
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    factor = warmup_restarts_decay(optimizer, total_steps=350, epochs=35).lr_lambdas[0]
    last = 0.5 * (1.0 + math.cos(math.pi * 15 / 16))

    assert factor(20) == pytest.approx(0.5)
    assert factor(40) == factor(80) == factor(160) == 1.0
    assert factor(79) < 0.01
    assert factor(320) == pytest.approx(last / 2)
    assert factor(349) == pytest.approx(last / 8)


def _row_config(config_name: str, row: str, cache: Path, run_dir: Path) -> DictConfig:
    # The experiment on the synthetic cache and the CPU. TF32 is left off: it would
    # persist in the process and reach the tests that follow.
    overrides = (f"experiment=top_tagging/{row}_100", "trainer=cpu", "logger=none")
    cfg = compose_root(config_name, *overrides)
    with open_dict(cfg):
        cfg.data.cache_dir = str(cache)
        cfg.data.nconstituents = TOP_TAGGING_N
        cfg.data.batch_size = 16
        cfg.data.num_workers = 0
        cfg.data.pin_memory = False
        cfg.trainer.max_epochs = 2
        cfg.extras.matmul_precision = None
        cfg.extras.print_config = False
        cfg.extras.enforce_tags = False
    return point_into(cfg, run_dir)


@pytest.mark.parametrize(
    "row, subdir", [("deepsets", "pt_etarel_phirel"), ("lorentznet", "embedded")]
)
def test_a_top_tagging_row_trains_and_evaluates_into_its_own_table(
    row: str, subdir: str, top_tagging_cache: Path
) -> None:
    from src.eval import evaluate

    train_dir, eval_dir = top_tagging_cache / "train", top_tagging_cache / "eval"
    train_main_and_folds(_row_config("train.yaml", row, top_tagging_cache, train_dir))
    cfg = _row_config("eval.yaml", row, top_tagging_cache, eval_dir)
    with open_dict(cfg):
        cfg.checkpoint_manifest_path = str(train_dir)
    HydraConfig().set_config(cfg)

    metric_dict, _ = evaluate(cfg)

    table = eval_dir / "results" / "top_tagging" / subdir
    header, result = read_results_row(table)
    assert (table / "results_table_baseline.csv").is_file()
    assert header == results_header()
    assert (result["model"], result["nconstituents"], result["folds_used"]) == (
        row,
        str(TOP_TAGGING_N),
        "1",
    )
    expected = {"fold_1/performance/auc_top", "fold_1/invariance/so2_tv"}
    assert expected <= set(metric_dict)
    for label in ("main", "fold_1"):
        with np.load(eval_dir / "outputs" / f"{label}.npz") as stored:
            assert stored["logits_clean"].shape == (TOP_TAGGING_SIZES["test.h5"], 2)
    assert any((eval_dir / "plots").rglob("*.*"))


@pytest.mark.slow
@pytest.mark.skipif(
    "TOP_TAGGING_RAW" not in os.environ,
    reason="Export TOP_TAGGING_RAW to the directory of the record's raw files.",
)
def test_the_reader_reads_the_records_jets_in_falling_pt() -> None:
    path = Path(os.environ["TOP_TAGGING_RAW"]) / "val.h5"
    p4, flags = next(top_tagging.read_chunks(path, chunk=10_000))
    p4 = p4.astype(np.float64)
    pt = np.hypot(p4[..., 1], p4[..., 2])
    jet = p4.sum(axis=1)
    jet_pt = np.hypot(jet[:, 1], jet[:, 2])
    jet_eta = np.arcsinh(jet[:, 3] / jet_pt)

    assert p4.shape == (10_000, 200, 4) and set(np.unique(flags)) == {0, 1}
    assert abs(flags.mean() - 0.5) < 0.02
    assert (p4[..., 0] >= np.linalg.norm(p4[..., 1:], axis=-1) * (1 - 1e-3)).all()
    assert np.mean((jet_pt > 550) & (jet_pt < 650)) > 0.99
    assert np.mean(np.abs(jet_eta) < 2) > 0.99
    assert (np.diff(pt, axis=1) <= 0).all()
