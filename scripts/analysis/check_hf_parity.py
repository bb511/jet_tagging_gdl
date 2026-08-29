#!/usr/bin/env python
"""Checks that the two data routes end at the same tensors.

`data=hls4ml` reads the Zenodo h5 files and caches its own .npy arrays; `data=hls4ml_hf`
reads the published record through the loader shipped inside it. Each route is surveyed in
a child process, so the wall-clock time and the peak RSS reported for it are its own and
the two sets of tensors never sit in memory together. Equal sha256 over shape, dtype and
bytes means the tensors are identical; when the digests disagree the saved arrays are
compared elementwise so the deviation can be sized.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rootutils

# Sets PROJECT_ROOT, which configs/paths/default.yaml reads. src/train.py does the same.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
SPLITS = ("train", "val", "test")
COMMON_OVERRIDES = ["data.num_workers=0", "data.pin_memory=False"]

# The three combinations that produce distinct tensors: plain split, k-fold pooling with
# the full feature set kept at test, and symm_loss (which never reduces the features).
CASES = [
    {"n_folds": 1, "keep_full_features_at_test": False},
    {"n_folds": 5, "keep_full_features_at_test": True},
    {"method": "symm_loss", "n_folds": 1},
]

RTOL, ATOL = 1e-6, 1e-7
CHUNK = 4096


def report(label: str, ok: bool, note: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  ' + note if note else ''}", flush=True)

    return ok


def announce(message: str) -> None:
    print(f"== {message}", flush=True)


def digest(array: np.ndarray) -> str:
    """One array's shape, dtype and bytes, hashed without copying it."""
    array = np.ascontiguousarray(array)
    hashed = hashlib.sha256(f"{array.shape}|{array.dtype}|".encode())
    hashed.update(memoryview(array.reshape(-1)).cast("B"))

    return hashed.hexdigest()


# ----------------------------------------------------------------------
# Survey: one route, one child process
# ----------------------------------------------------------------------
def _instantiate(overrides: list[str]):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    with initialize_config_dir(config_dir=str(CONFIGS), version_base=None):
        cfg = compose(config_name="train", overrides=overrides)

    return instantiate(cfg.data)


def _drive(datamodule, n_folds: int) -> None:
    """The sequence src/train.py runs before the first fold trains."""
    datamodule.prepare_data()
    datamodule.n_folds = n_folds
    datamodule.set_fold(0)
    datamodule.setup("fit")
    datamodule.setup("test")


def _dump(dataset, split: str, out_dir: Path) -> dict:
    """Digest and save both tensors, so a mismatch can be diagnosed afterwards."""
    entries = {}
    for kind, tensor in (("x", dataset.X), ("y", dataset.y)):
        array = tensor.detach().cpu().numpy()
        np.save(out_dir / f"{split}_{kind}.npy", array)
        entries[kind] = {
            "digest": digest(array), "shape": list(array.shape), "dtype": str(array.dtype),
        }

    return entries


def _meta(datamodule, seconds: float) -> dict:
    return {
        "n_train_samples": int(datamodule.n_train_samples),
        "n_val_samples": int(datamodule.n_val_samples),
        "feature_names": [str(name) for name in datamodule.feature_names],
        "norm_params": {
            str(name): [str(kind), float(scale)]
            for name, (kind, scale) in datamodule.norm_params.items()
        },
        "seconds": round(seconds, 1),
        # ru_maxrss is kB on Linux (bytes on macOS); the surveys run on Linux.
        "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }


def run_survey(overrides: list[str], n_folds: int, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    datamodule = _instantiate(overrides)
    started = time.perf_counter()
    _drive(datamodule, n_folds)
    summary = {s: _dump(getattr(datamodule, f"data_{s}"), s, out_dir) for s in SPLITS}
    summary.update(_meta(datamodule, time.perf_counter() - started))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return 0


# ----------------------------------------------------------------------
# Driver: the overrides each route and case needs
# ----------------------------------------------------------------------
def route_overrides(args: argparse.Namespace) -> dict[str, list[str]]:
    mirrored = ["data=hls4ml_hf", f"data.repo_id={args.repo}", f"data.cache_dir={args.hf_cache}"]
    if args.revision:
        mirrored.append(f"data.revision={args.revision}")

    return {"legacy": ["data=hls4ml", f"data.data_dir={args.legacy_dir}"], "hf": mirrored}


def case_overrides(case: dict) -> list[str]:
    """A case is a set of data.* overrides; hydra's grammar wants lowercase booleans."""
    return [
        f"data.{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in case.items()
    ]


def survey_route(route: str, overrides: list[str], n_folds: int, out_dir: Path) -> dict | None:
    """Run one survey as a child process; None when it failed."""
    announce(f"surveying {route}: {' '.join(overrides)}")
    argv = [
        sys.executable, str(Path(__file__).resolve()), "survey",
        "--out", str(out_dir), "--n-folds", str(n_folds), "--overrides", *overrides,
    ]
    if subprocess.run(argv).returncode != 0:
        return None

    return json.loads((out_dir / "summary.json").read_text())


# ----------------------------------------------------------------------
# Driver: comparison
# ----------------------------------------------------------------------
def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Chunked: the train tensor is ~1 GB, and a mismatch must not need three copies of it."""
    return max(
        (float(np.abs(a[i : i + CHUNK].astype(np.float64) - b[i : i + CHUNK]).max())
         for i in range(0, len(a), CHUNK)),
        default=0.0,
    )


def allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return all(
        np.allclose(a[i : i + CHUNK], b[i : i + CHUNK], rtol=RTOL, atol=ATOL)
        for i in range(0, len(a), CHUNK)
    )


def compare_tensor(left: dict, right: dict, left_path: Path, right_path: Path) -> tuple[bool, str]:
    if left["digest"] == right["digest"]:
        return True, "exact"

    # Only worth reading once the digests disagree, and then memory-mapped.
    a, b = (np.load(path, mmap_mode="r") for path in (left_path, right_path))
    if a.shape != b.shape:
        return False, f"shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"

    close, delta = allclose(a, b), max_abs_diff(a, b)

    return close, f"{'allclose' if close else 'differs'}, max |diff| = {delta:.3g}"


def compare_norm_params(left: dict, right: dict) -> tuple[bool, str]:
    if list(left) != list(right):
        return False, "different feature names"

    kinds = [name for name in left if left[name][0] != right[name][0]]
    if kinds:
        return False, f"kinds differ: {kinds}"

    scales = [np.allclose([left[n][1]], [right[n][1]], rtol=RTOL, atol=ATOL) for n in left]

    return all(scales), "" if all(scales) else "scales differ"


def compare_meta(left: dict, right: dict) -> bool:
    ok = report("n_train_samples", left["n_train_samples"] == right["n_train_samples"])
    ok &= report("n_val_samples", left["n_val_samples"] == right["n_val_samples"])
    ok &= report("feature_names", left["feature_names"] == right["feature_names"])
    passed, note = compare_norm_params(left["norm_params"], right["norm_params"])

    return ok & report("norm_params", passed, note)


def compare_case(work_dir: Path, summaries: dict) -> bool:
    left, right = summaries["legacy"], summaries["hf"]
    ok = True
    for split in SPLITS:
        for kind in ("x", "y"):
            paths = [work_dir / route / f"{split}_{kind}.npy" for route in ("legacy", "hf")]
            passed, note = compare_tensor(left[split][kind], right[split][kind], *paths)
            ok &= report(f"{split} {kind}", passed, note)

    return ok & compare_meta(left, right)


# ----------------------------------------------------------------------
# Driver: entry
# ----------------------------------------------------------------------
def run_case(index: int, routes: dict, work_root: Path, rows: list) -> bool:
    case = CASES[index]
    announce(f"case {index}: {case}")
    work_dir = work_root / f"case{index}"
    summaries = {
        route: survey_route(
            route, overrides + case_overrides(case) + COMMON_OVERRIDES,
            int(case.get("n_folds", 1)), work_dir / route,
        )
        for route, overrides in routes.items()
    }
    if any(summary is None for summary in summaries.values()):
        return report(f"case {index} surveys completed", False)

    rows += [(index, route, summary) for route, summary in summaries.items()]

    return compare_case(work_dir, summaries)


def timings(rows: list) -> None:
    announce("wall clock and peak RSS per route")
    print(f"  {'case':<6}{'route':<8}{'seconds':>10}{'max RSS/MB':>12}", flush=True)
    for index, route, summary in rows:
        print(
            f"  {index:<6}{route:<8}{summary['seconds']:>10.1f}{summary['max_rss_mb']:>12.1f}",
            flush=True,
        )


def run_driver(args: argparse.Namespace) -> int:
    routes = route_overrides(args)
    chosen = args.cases if args.cases else list(range(len(CASES)))
    rows: list = []
    # A list, not a generator: every case is surveyed even after one has failed.
    ok = all([run_case(index, routes, Path(args.work), rows) for index in chosen])
    timings(rows)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED", flush=True)

    return 0 if ok else 1


def parse_driver(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--legacy-dir", required=True, help="data.data_dir of the legacy route")
    parser.add_argument("--repo", required=True, help="hub id, or a local mirror directory")
    parser.add_argument("--revision", help="record revision to pin the hf route to")
    parser.add_argument("--hf-cache", required=True, help="where the record builds its caches")
    parser.add_argument("--work", required=True, help="scratch root for the surveys")
    parser.add_argument(
        "--cases", type=int, nargs="*", choices=range(len(CASES)),
        help=f"subset of {list(range(len(CASES)))}",
    )

    return parser.parse_args(argv)


def parse_survey(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="one route's survey, run as a child process")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-folds", type=int, required=True)
    parser.add_argument("--overrides", nargs="*", default=[])

    return parser.parse_args(argv)


def main() -> int:
    if sys.argv[1:2] == ["survey"]:
        args = parse_survey(sys.argv[2:])

        return run_survey(args.overrides, args.n_folds, Path(args.out))

    return run_driver(parse_driver(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
