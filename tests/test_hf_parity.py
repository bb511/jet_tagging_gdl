"""Unit tests of the pure helpers in scripts/analysis/check_hf_parity.py.

The script is loaded by path rather than imported as a package: scripts/ is not one, and
loading it this way keeps the datamodules (and the hub) out of the test run.
"""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "check_hf_parity.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("check_hf_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


parity = _load_script()


def _entry(array: np.ndarray, path: Path) -> dict:
    """What a survey writes for one tensor: the file plus its summary.json entry."""
    np.save(path, array)

    return {
        "digest": parity.digest(array), "shape": list(array.shape), "dtype": str(array.dtype),
    }


# ----------------------------------------------------------------------
# digest
# ----------------------------------------------------------------------
def test_digest_is_stable_for_equal_arrays():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert parity.digest(a) == parity.digest(a.copy())


def test_digest_separates_dtype_shape_and_values():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert parity.digest(a) != parity.digest(a.astype(np.float64))
    assert parity.digest(a) != parity.digest(a.reshape(4, 3))
    assert parity.digest(a) != parity.digest(a + 1)


def test_digest_ignores_memory_layout():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert parity.digest(a) == parity.digest(np.asfortranarray(a))


# ----------------------------------------------------------------------
# compare_tensor
# ----------------------------------------------------------------------
def test_identical_tensors_compare_exact(tmp_path):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    left, right = _entry(a, tmp_path / "l.npy"), _entry(a, tmp_path / "r.npy")

    assert parity.compare_tensor(left, right, tmp_path / "l.npy", tmp_path / "r.npy") == (
        True, "exact",
    )


def test_tiny_deviation_compares_allclose(tmp_path):
    a = np.arange(12, dtype=np.float64).reshape(3, 4) / 10
    left = _entry(a, tmp_path / "l.npy")
    right = _entry(a + 5e-8, tmp_path / "r.npy")

    ok, note = parity.compare_tensor(left, right, tmp_path / "l.npy", tmp_path / "r.npy")

    assert ok
    assert note.startswith("allclose")


def test_real_deviation_fails(tmp_path):
    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    left = _entry(a, tmp_path / "l.npy")
    right = _entry(a + 0.5, tmp_path / "r.npy")

    ok, note = parity.compare_tensor(left, right, tmp_path / "l.npy", tmp_path / "r.npy")

    assert not ok
    assert "0.5" in note


def test_shape_mismatch_fails(tmp_path):
    left = _entry(np.zeros((3, 4)), tmp_path / "l.npy")
    right = _entry(np.zeros((2, 4)), tmp_path / "r.npy")

    ok, note = parity.compare_tensor(left, right, tmp_path / "l.npy", tmp_path / "r.npy")

    assert not ok
    assert note.startswith("shapes differ")


def test_chunked_reduction_sees_the_last_chunk(monkeypatch):
    monkeypatch.setattr(parity, "CHUNK", 2)
    a = np.zeros((5, 3))
    b = a.copy()
    b[4, 2] = 3.0

    assert parity.max_abs_diff(a, b) == 3.0
    assert not parity.allclose(a, b)


# ----------------------------------------------------------------------
# norm_params
# ----------------------------------------------------------------------
def test_norm_params_compare_on_names_kinds_and_scales():
    left = {"j1_pt": ["positive", 1.0], "j1_px": ["centered", 2.0]}
    assert parity.compare_norm_params(left, dict(left)) == (True, "")
    assert not parity.compare_norm_params(left, {"j1_pt": ["positive", 1.0]})[0]
    assert not parity.compare_norm_params(left, {**left, "j1_px": ["positive", 2.0]})[0]
    assert not parity.compare_norm_params(left, {**left, "j1_px": ["centered", 2.5]})[0]


# ----------------------------------------------------------------------
# case and route translation
# ----------------------------------------------------------------------
def test_case_overrides_render_booleans_lowercase():
    case = {"n_folds": 5, "keep_full_features_at_test": True}

    assert parity.case_overrides(case) == [
        "data.n_folds=5",
        "data.keep_full_features_at_test=true",
    ]


@pytest.mark.parametrize("case", parity.CASES)
def test_shipped_cases_translate_to_data_overrides(case):
    overrides = parity.case_overrides(case)
    assert len(overrides) == len(case)
    assert all(override.startswith("data.") and "=" in override for override in overrides)


def test_route_overrides_carry_the_paths_and_skip_an_absent_revision():
    args = argparse.Namespace(
        legacy_dir="/data/raw", repo="/data/mirror", revision=None, hf_cache="/data/hf",
    )
    routes = parity.route_overrides(args)

    assert routes["legacy"] == ["data=hls4ml", "data.data_dir=/data/raw"]
    assert routes["hf"] == [
        "data=hls4ml_hf", "data.repo_id=/data/mirror", "data.cache_dir=/data/hf",
    ]


def test_route_overrides_pin_a_revision_when_given():
    args = argparse.Namespace(
        legacy_dir="/data/raw", repo="podagiu/x", revision="abc123", hf_cache="/data/hf",
    )

    assert parity.route_overrides(args)["hf"][-1] == "data.revision=abc123"
