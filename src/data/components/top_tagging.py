"""The top-quark tagging reference data set, converted once into arrays the datamodule
serves as it serves the hls4ml record.

The record (Zenodo 2603256, arXiv:1902.09914) holds 1 211 000 training, 403 000
validation and 404 000 test jets: anti-kT R = 0.8 jets of p_T in [550, 650] GeV and
pseudorapidity within (-2, 2), each with the four-momenta of its leading 200
particle-flow constituents, zero-padded, and the flag ``is_signal_new`` (1 for top, 0
for QCD). The conversion keeps the leading N constituents by p_T as the nine `COLUMNS`,
divides each column by a scale fitted on the training split as the hls4ml record's
loader fits it, and writes ``nconst_<N>/`` under the cache root:

- ``x_{train,validation,test}.npy``: float32 (jets, N, 9), padded slots all zero;
- ``y_{train,validation,test}.npy``: float32 one-hot (jets, 2), column 0 QCD, 1 top;
- ``meta.json``: the provenance, the counts per split and the scales, written last.
"""

import hashlib
import json
import os
import shutil
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch

from src.data.components.augmentation import (
    compute_eta,
    compute_phi,
    compute_pt,
    rebase_on_kept_axis,
)
from src.symmetry import CLASS_NAMES_OF
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

RECORD_URL = "https://zenodo.org/records/2603256/files/{name}?download=1"
# The md5 of each raw file, from https://zenodo.org/api/records/2603256.
RAW_FILES = {
    "train.h5": "45663819f47c13724f67eb0fd80bfa5c",
    "val.h5": "dca4b7248027618f041f9baa86d360fc",
    "test.h5": "13163479dee30a5fe546e4536cc3d04d",
}
# The datamodule's name of each split, and the raw file holding it.
SPLITS = {"train": "train.h5", "validation": "val.h5", "test": "test.h5"}
MAX_CONSTITUENTS = 200
# The four-momenta of the raw files, constituent by constituent.
RAW_COLUMNS = [
    f"{c}_{i}" for i in range(MAX_CONSTITUENTS) for c in ("E", "PX", "PY", "PZ")
]

COLUMNS = ("px", "py", "pz", "e", "pt", "eta", "phi", "etarel", "phirel")
# Scaled by their maximum, as the hls4ml record scales its non-negative columns; every
# other column lives on both sides of zero and is scaled by half its range.
POSITIVE_COLUMNS = ("pt", "e")
CLASS_NAMES = CLASS_NAMES_OF["top_tagging"]

META = "meta.json"
# Raise it whenever the layout of a cache changes, so that an old cache is refused.
FORMAT_VERSION = 1

# One read of a raw file: the four-momenta (jets, 200, 4) as [E, px, py, pz] in GeV,
# and the flags (jets,).
Chunk = tuple[np.ndarray, np.ndarray]


class SplitTensors(NamedTuple):
    """One split as the datamodule reads it: x (jets, N, 9) and one-hot y (jets, 2)."""

    x: torch.Tensor
    y: torch.Tensor


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _stream(url: str, out: Path) -> str:
    """Copy *url* into *out* and return the md5 of what was copied."""
    digest = hashlib.md5(usedforsecurity=False)
    # B310 warns of file:// URLs: RECORD_URL is https, and the tests use file:// on
    # purpose.
    with urllib.request.urlopen(url, timeout=600) as response:  # nosec B310
        with open(out, "wb") as handle:
            for block in iter(lambda: response.read(1 << 20), b""):
                digest.update(block)
                handle.write(block)
    return digest.hexdigest()


def _fetch(name: str, path: Path) -> None:
    """Download raw file *name* to *path* through a partial file of this process.

    :raises ValueError: if the md5 of the download is not the record's; nothing is left
        at *path*.
    """
    partial = path.with_name(f"{name}.{os.getpid()}.partial")
    url = RECORD_URL.format(name=name)
    log.info(f"Downloading {url} to {path}")
    try:
        found = _stream(url, partial)
        if found != RAW_FILES[name]:
            raise ValueError(f"{name}: md5 {found}, the record's is {RAW_FILES[name]}.")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def download(raw_dir: str | Path) -> None:
    """Download every raw file absent from *raw_dir* or whose md5 is not the
    record's."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in RAW_FILES.items():
        path = raw_dir / name
        if not (path.is_file() and _md5(path) == md5):
            _fetch(name, path)


def count_jets(path: Path) -> int:
    """The number of jets in raw file *path*, read off the HDF5 metadata alone."""
    with pd.HDFStore(path, mode="r") as store:
        return int(store.get_storer("table").nrows)


def read_chunks(path: Path, chunk: int = 100_000) -> Iterator[Chunk]:
    """Yield the jets of raw file *path* in their stored order, *chunk* at a time.

    pandas reads the file through PyTables, which it imports on the first read.
    """
    for start in range(0, count_jets(path), chunk):
        frame = pd.read_hdf(path, "table", start=start, stop=start + chunk)
        p4 = frame[RAW_COLUMNS].to_numpy(np.float32).reshape(len(frame), -1, 4)
        yield p4, frame["is_signal_new"].to_numpy()


def _columns(p4: np.ndarray) -> np.ndarray:
    """The nine `COLUMNS` (jets, N, 9) in float64 of the kept constituents *p4*.

    etarel and phirel are measured from the axis of the kept constituents, as the
    datamodule's rebase measures them; every column of a padded slot is zero.
    """
    e, px, py, pz = torch.from_numpy(p4).unbind(dim=-1)
    zero = torch.zeros_like(e)
    angles = (compute_pt(px, py), compute_eta(px, py, pz), compute_phi(px, py))
    x = torch.stack([px, py, pz, e, *angles, zero, zero], dim=-1) * (e > 0)[..., None]
    return rebase_on_kept_axis(x, COLUMNS).numpy()


def _leading(p4: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """The leading *n* constituents of every jet by p_T, in float64, and the number of
    jets whose stored order the sort changed.

    The sort is stable, so that the padded slots, all at p_T = 0, keep their order and
    move no jet into the count.
    """
    pt = np.hypot(p4[..., 1].astype(np.float64), p4[..., 2])
    order = np.argsort(-pt, axis=1, kind="stable")
    resorted = int((order != np.arange(p4.shape[1])).any(axis=1).sum())
    kept = np.take_along_axis(p4, order[:, :n, None], axis=1)
    return kept.astype(np.float64), resorted


@dataclass
class _Summary:
    """What the conversion records of one split, the multiplicities being those of the
    untruncated jets."""

    n_jets: int = 0
    class_counts: np.ndarray = field(default_factory=lambda: np.zeros(2, np.int64))
    jets_resorted: int = 0
    jets_truncated: int = 0
    constituents: int = 0
    max_multiplicity: int = 0
    # Per column, over the kept constituents, padded slots included.
    low: np.ndarray = field(default_factory=lambda: np.full(len(COLUMNS), np.inf))
    high: np.ndarray = field(default_factory=lambda: np.full(len(COLUMNS), -np.inf))

    def add(
        self, p4: np.ndarray, x: np.ndarray, labels: np.ndarray, resorted: int
    ) -> None:
        """Count one chunk: *p4* its raw jets (jets, 200, 4), *x* their stored
        constituents (jets, N, 9) and *labels* their class indices (jets,)."""
        multiplicity = (p4[..., 0] > 0).sum(axis=1)
        self.n_jets += len(x)
        self.class_counts += np.bincount(labels, minlength=2)
        self.jets_resorted += resorted
        self.jets_truncated += int((multiplicity > x.shape[1]).sum())
        self.constituents += int(multiplicity.sum())
        self.max_multiplicity = max(self.max_multiplicity, int(multiplicity.max()))
        self.low = np.minimum(self.low, x.min(axis=(0, 1)))
        self.high = np.maximum(self.high, x.max(axis=(0, 1)))

    def record(self) -> dict[str, object]:
        return {
            "n_jets": self.n_jets,
            "class_counts": dict(zip(CLASS_NAMES, self.class_counts.tolist())),
            "jets_resorted": self.jets_resorted,
            "jets_truncated": self.jets_truncated,
            "mean_multiplicity": self.constituents / self.n_jets,
            "max_multiplicity": self.max_multiplicity,
        }


def _labels(flags: np.ndarray) -> np.ndarray:
    """The class indices of *flags*, which must be 0 (QCD) or 1 (top)."""
    if not np.isin(flags, (0, 1)).all():
        raise ValueError(f"is_signal_new holds {np.unique(flags)}, expected 0 and 1.")
    return flags.astype(np.int64)


def _fill(x: np.ndarray, labels: np.ndarray, chunks: Iterator[Chunk]) -> _Summary:
    """Fill *x* (jets, N, 9) with the kept constituents of *chunks*, unscaled, and
    *labels* (jets,) with their class indices."""
    summary = _Summary()
    for p4, flags in chunks:
        kept, resorted = _leading(p4, x.shape[1])
        rows = slice(summary.n_jets, summary.n_jets + len(p4))
        x[rows], labels[rows] = _columns(kept), _labels(flags)
        summary.add(p4, x[rows], labels[rows], resorted)
    return summary


def _write_split(
    path: Path, folder: Path, split: str, n: int, read: Callable, count: Callable
) -> _Summary:
    """Write *split* from raw file *path* into *folder*, the constituents unscaled, and
    return what was seen."""
    n_jets = count(path)
    shape = (n_jets, n, len(COLUMNS))
    x = np.lib.format.open_memmap(folder / f"x_{split}.npy", "w+", np.float32, shape)
    labels = np.empty(n_jets, dtype=np.int64)
    summary = _fill(x, labels, read(path))
    if summary.n_jets != n_jets:
        raise ValueError(f"{path}: read {summary.n_jets} jets of {n_jets}.")
    x.flush()
    np.save(folder / f"y_{split}.npy", np.eye(2, dtype=np.float32)[labels])
    return summary


def _scale(name: str, low: float, high: float) -> tuple[str, float]:
    if name in POSITIVE_COLUMNS:
        return "positive", high
    return "centered", 0.5 * (high - low)


def _fit_scales(train: _Summary) -> dict[str, tuple[str, float]]:
    """One (kind, scale) per column from the training split, as the hls4ml record's
    loader fits them (``fit_physics_norm``)."""
    scales = {
        name: _scale(name, float(low), float(high))
        for name, low, high in zip(COLUMNS, train.low, train.high)
    }
    if not all(scale > 0 for _, scale in scales.values()):
        raise ValueError(f"A column is constant on the training split: {scales}.")
    return scales


def _divide(path: Path, scales: dict[str, tuple[str, float]], chunk: int) -> None:
    """Divide the stored columns of *path* by their scales in place, *chunk* jets at a
    time."""
    x = np.load(path, mmap_mode="r+")
    factors = np.array([scales[name][1] for name in COLUMNS])
    for start in range(0, len(x), chunk):
        x[start : start + chunk] /= factors
    x.flush()


def _write_meta(
    folder: Path,
    n: int,
    summaries: dict[str, _Summary],
    scales: dict[str, tuple[str, float]],
) -> None:
    """Write ``meta.json``: the provenance, the counts of every split and the scales."""
    meta = {
        "format_version": FORMAT_VERSION,
        "source": "https://zenodo.org/records/2603256",
        "files": RAW_FILES,
        "nconstituents": n,
        "columns": list(COLUMNS),
        "class_names": list(CLASS_NAMES),
        "splits": {split: summary.record() for split, summary in summaries.items()},
        "norm_params": scales,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (folder / META).write_text(json.dumps(meta, indent=2) + "\n")


def _publish(partial: Path, target: Path) -> None:
    """Rename the finished *partial* directory to *target*, or drop it if another
    process published a complete cache there first."""
    try:
        os.rename(partial, target)
    except OSError:
        if not (target / META).is_file():
            raise
        shutil.rmtree(partial)


def _build(
    folder: Path, raw_dir: Path, n: int, read: Callable, count: Callable, chunk: int
) -> None:
    """Write the whole cache into *folder*, ``meta.json`` last."""
    summaries = {
        split: _write_split(raw_dir / name, folder, split, n, read, count)
        for split, name in SPLITS.items()
    }
    scales = _fit_scales(summaries["train"])
    for split in SPLITS:
        _divide(folder / f"x_{split}.npy", scales, chunk)
    _write_meta(folder, n, summaries, scales)


def convert(
    raw_dir: str | Path,
    cache_dir: str | Path,
    n: int,
    read: Callable[[Path], Iterator[Chunk]] = read_chunks,
    count: Callable[[Path], int] = count_jets,
    chunk: int = 100_000,
) -> Path:
    """Convert the raw files in *raw_dir* into ``nconst_<n>/`` under *cache_dir*.

    The cache is written into a partial directory of this process and renamed into place
    once complete, so that a cache with ``meta.json`` is whole and two processes
    converting at once leave one cache.

    :param read: yields the chunks of one raw file, `read_chunks` or a stand-in.
    :param count: the number of jets in one raw file, which sizes its arrays.
    :return: the cache directory.
    """
    if not 0 < n <= MAX_CONSTITUENTS:
        raise ValueError(f"nconstituents must lie in 1..{MAX_CONSTITUENTS}, not {n}.")
    target = Path(cache_dir) / f"nconst_{n}"
    partial = target.with_name(f"{target.name}.{os.getpid()}.partial")
    partial.mkdir(parents=True)
    try:
        _build(partial, Path(raw_dir), n, read, count, chunk)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    _publish(partial, target)
    log.info(f"Converted the top-tagging record into {target}")
    return target


def read_meta(folder: Path) -> dict:
    """The ``meta.json`` of the cache in *folder*.

    :raises ValueError: if the cache was written in another format.
    """
    meta = json.loads((folder / META).read_text())
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"{folder} holds a cache of format {meta.get('format_version')}, and this "
            f"code reads format {FORMAT_VERSION}: convert into a fresh cache_dir."
        )
    return meta


def prepare(cache_root: str | Path, n: int) -> Path:
    """Build ``nconst_<n>/`` under *cache_root*, downloading the raw files into ``raw/``
    first; a cache that exists is checked and kept."""
    target = Path(cache_root) / f"nconst_{n}"
    if (target / META).is_file():
        read_meta(target)
        return target
    download(Path(cache_root) / "raw")
    return convert(Path(cache_root) / "raw", cache_root, n)


class TopTaggingRecord:
    """The cache of one constituent count, through the surface the hls4ml datamodule
    reads off the hls4ml record's loader: `feature_names`, `norm_params`, `prepare` and
    `load`.

    :param cache_root: holds ``raw/`` and one ``nconst_<N>/`` per constituent count.
    :param nconstituents: the N of the cache served.
    """

    def __init__(self, cache_root: str | Path, nconstituents: int) -> None:
        self.cache_root = Path(cache_root)
        self.nconstituents = nconstituents

    @property
    def cache_folder(self) -> Path:
        return self.cache_root / f"nconst_{self.nconstituents}"

    @property
    def feature_names(self) -> list[str]:
        return list(COLUMNS)

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        """The (kind, scale) of every column, None until `prepare` has built the
        cache."""
        if not (self.cache_folder / META).is_file():
            return None
        params = read_meta(self.cache_folder)["norm_params"]
        return {name: (kind, scale) for name, (kind, scale) in params.items()}

    def prepare(self) -> None:
        """Build the cache unless it exists, through `prepare`."""
        prepare(self.cache_root, self.nconstituents)

    def load(self, split: str) -> SplitTensors:
        """One split as copy-on-write memory maps, which the processes of one host share
        through the page cache until a tensor is written to."""
        if split not in SPLITS:
            raise ValueError(
                f"Unknown split {split!r}, expected one of {list(SPLITS)}."
            )
        read_meta(self.cache_folder)
        x, y = (
            torch.from_numpy(np.load(self.cache_folder / name, mmap_mode="c"))
            for name in (f"x_{split}.npy", f"y_{split}.npy")
        )
        return SplitTensors(x, y)
