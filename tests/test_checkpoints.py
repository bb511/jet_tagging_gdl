import logging
from pathlib import Path

import pytest

from src.utils.checkpoints import Manifest, ManifestRun, merge_manifests, write_manifest

MAIN = ManifestRun("main", None, "/ckpt/main.ckpt", 0.5)


def _fold(index: int) -> ManifestRun:
    return ManifestRun(f"fold_{index}", index, f"/ckpt/fold_{index}.ckpt", 0.5)


def _write(directory: Path, n_folds: int, *runs: ManifestRun) -> None:
    # As a process writes it: n_folds 0 for a main run alone, k for one of its folds.
    write_manifest(Manifest(n_folds, str(directory), "exp", list(runs)), directory)


def test_the_main_run_comes_first_and_the_folds_follow_by_index(tmp_path: Path) -> None:
    _write(tmp_path, 0, MAIN)
    for index in (10, 2, 1):
        _write(tmp_path / f"fold_{index}", 3, _fold(index))

    merged = merge_manifests(tmp_path)

    assert [run.fold_label for run in merged.runs] == ["main", "fold_1", "fold_2", "fold_10"]
    assert merged.n_folds == 3
    assert merged.hydra_output_dir == str(tmp_path)
    assert merged.experiment_name == "exp"


def test_a_fold_still_missing_is_counted_out_with_a_warning(tmp_path: Path, caplog) -> None:
    _write(tmp_path, 0, MAIN)
    _write(tmp_path / "fold_1", 2, _fold(1))

    with caplog.at_level(logging.WARNING):
        merged = merge_manifests(tmp_path)

    assert merged.n_folds == 1
    assert "still running" in caplog.text


def test_a_missing_main_manifest_is_refused(tmp_path: Path) -> None:
    _write(tmp_path / "fold_1", 2, _fold(1))

    with pytest.raises(FileNotFoundError, match="main run"):
        merge_manifests(tmp_path)


def test_a_fold_manifest_holding_more_than_its_fold_is_refused(tmp_path: Path) -> None:
    # A sequential k-fold manifest dropped into fold_1/ would count its folds twice over.
    _write(tmp_path, 0, MAIN)
    _write(tmp_path / "fold_1", 2, MAIN, _fold(1), _fold(2))

    with pytest.raises(ValueError, match="one fold"):
        merge_manifests(tmp_path)
