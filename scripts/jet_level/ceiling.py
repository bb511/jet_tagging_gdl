#!/usr/bin/env python3
"""The expert jet-level ceiling: an MLP on the record's jet-level observables.

Beside its constituents, the record carries 53 observables per jet, computed upstream
from the whole jet: its p_T, rapidity, mass and multiplicity, N-subjettiness, energy
correlation functions and groomed masses. This script measures how well they tag the
five classes under the protocol of the networks' finals, so that its rows join the same
tables and the same comparisons. Fold i is the one that
`src.data.hls4ml_datamodule.fold_indices` selects from the pooled training and
validation jets with the ``kfold_seed`` of ``configs/data/hls4ml.yaml``. The MLP is
fitted on the fold's training jets, standardised by their own moments, stopped on its
validation jets, and scored on the whole test split by the macro accuracy and
`src.callbacks.invariance.clean_metrics`. Two rows are fitted: ``jet_level_all53``, and
``jet_level_noeta52`` without the jet rapidity, which a row exact under the beam boost
cannot see.

Each row is appended to ``results/jet_level/results_table_baseline.csv`` by
`src.utils.results_table.ResultsTable`, under the networks' header, with
``nconstituents`` 150, since the observables belong to the whole jet. They are never
transformed, so every invariance column holds NaN, and they are computed upstream, so
the FLOP and eBOP columns do too. The per-jet test logits go to
``<run-dir>/outputs/<row>/fold_<i>.npz`` and a JSON sidecar per row to
``<run-dir>/<row>.json``.

The folds select the networks' jets only if the parquet shards read here hold them in
the order of the loader's cache, which the networks read. Before anything is fitted, a
guard compares the class and the constituent count of every jet with that cache at 150
constituents; without the cache it refuses to run rather than build one.

A full run, then a smoke run that leaves ``results/`` alone::

    python scripts/jet_level/ceiling.py --device cuda:0
    python scripts/jet_level/ceiling.py --quick --run-dir <scratch directory>
"""

import argparse
import itertools
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import rootutils

# Before any other import: .env carries HF_HUB_CACHE, which huggingface_hub reads at
# import time, and PROJECT_ROOT is set to the root found here.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sklearn
import torch
import torch.nn.functional as F
import torchmetrics
from huggingface_hub import snapshot_download
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.functional.classification import multiclass_accuracy

from src.callbacks.invariance import clean_metrics
from src.data.hls4ml_datamodule import REPO_ID, HLS4MLDataModule, fold_indices
from src.symmetry import CLASS_NAMES, TOTAL_VARIATION
from src.utils.results_table import ResultsTable

ROOT = Path(os.environ["PROJECT_ROOT"])

# Each row and the observables it leaves out.
ROWS = {"jet_level_all53": (), "jet_level_noeta52": ("j_eta",)}
N_OBSERVABLES = 53
# The rows' label in the tables: the observables belong to the whole jet, which the
# networks see at 150 constituents. The order guard reads the loader's cache there too.
NCONSTITUENTS = 150
N_FOLDS = 5
# The networks' seed (configs/train.yaml); it fixes the initialisation, the batch order
# and the --quick subsamples.
SEED = 12345
SUBDIR = "jet_level"
SPLITS = ("train", "validation", "test")
# The loader's cache of one constituent count, complete once prepare() has run.
CACHE_FILES = (
    *(f"{array}_{split}.npy" for array in ("x", "y") for split in SPLITS),
    "norm_params.npz",
)


@dataclass(frozen=True)
class Recipe:
    """One MLP: its hidden widths, the dropout after each, and Adam's learning rate."""

    widths: tuple[int, ...]
    dropout: float
    lr: float


# The point the 2026-10-05 scratch run selected on the validation split for both rows.
DEFAULT = Recipe((256, 128, 64), 0.0, 1e-3)
# The grid of that run, which --select searches again.
GRID = tuple(
    Recipe(widths, dropout, lr)
    for widths, dropout, lr in itertools.product(
        ((64, 32, 32), (256, 128, 64)), (0.0, 0.1), (1e-3, 3e-4)
    )
)
BATCH = 1024
MAX_EPOCHS = 100
# ReduceLROnPlateau and early stopping, both on the macro validation accuracy.
LR_FACTOR = 0.1
LR_PATIENCE = 5
STOP_PATIENCE = 12
# Jets per forward pass at inference.
EVAL_CHUNK = 65536
QUICK_JETS = 20_000
QUICK_EPOCHS = 3

# One split on the host: the observables (jets, observables) in float32 and the class
# indices (jets,).
Split = tuple[np.ndarray, np.ndarray]
# The same on the device, standardised.
Pair = tuple[torch.Tensor, torch.Tensor]
# The per-observable mean and standard deviation, float64, shape (observables,) each.
Scale = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Run:
    """What every row of one invocation shares."""

    device: torch.device
    epochs: int
    quick: bool
    select: bool
    kfold_seed: int
    run_dir: Path
    results_dir: Path
    # The provenance every sidecar repeats: seeds, record, guard, software, machine.
    meta: dict = field(default_factory=dict)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def data_config() -> DictConfig:
    """The networks' data node, read for its pinned ``revision`` and its
    ``kfold_seed``."""
    return OmegaConf.load(ROOT / "configs" / "data" / "hls4ml.yaml")


def shard_dir(revision: str) -> Path:
    """The directory of the record's parquet shards at *revision*, in the hub cache."""
    root = snapshot_download(REPO_ID, repo_type="dataset", revision=revision)
    return Path(root) / "data"


def _shards(data_dir: Path, split: str) -> list[Path]:
    """The shards of *split*, sorted as the record's loader sorts them, which fixes the
    order of the jets."""
    found = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not found:
        raise FileNotFoundError(f"No {split} shards under {data_dir}")
    return found


def read_split(data_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The observables of *split* (jets, 53) in the record's column order, the class
    indices (jets,) and the observables' names."""
    paths = _shards(data_dir, split)
    names = [name for name in pq.read_schema(paths[0]).names if name.startswith("j_")]
    tables = [pq.read_table(path, columns=[*names, "label"]) for path in paths]
    table = pa.concat_tables(tables)
    x = np.stack([table[name].to_numpy() for name in names], axis=1)
    return x.astype(np.float32), table["label"].to_numpy().astype(np.int64), names


def constituent_counts(data_dir: Path, split: str) -> np.ndarray:
    """The constituents each jet of *split* carries in the shards, shape (jets,)."""
    paths = _shards(data_dir, split)
    columns = [pq.read_table(path, columns=["j1_pt"])["j1_pt"] for path in paths]
    lengths = [pc.list_value_length(column).to_numpy() for column in columns]
    return np.concatenate(lengths).astype(np.int64)


def read_record(data_dir: Path) -> tuple[dict[str, Split], list[str]]:
    """Every split's observables and class indices, and the observables' names."""
    data: dict[str, Split] = {}
    for split in SPLITS:
        x, y, names = read_split(data_dir, split)
        if x.shape[1] != N_OBSERVABLES or not np.isfinite(x).all():
            raise RuntimeError(
                f"{split}: {x.shape[1]} observables, expected {N_OBSERVABLES} finite ones"
            )
        data[split] = (x, y)
    return data, names


def _compare(split: str, name: str, loaded: np.ndarray, stored: np.ndarray) -> None:
    """Raise unless one per-jet quantity of *split* agrees jet by jet."""
    if loaded.shape != stored.shape:
        raise RuntimeError(
            f"{split}: the loader holds {len(loaded)} jets, the shards {len(stored)}"
        )
    wrong = np.flatnonzero(loaded != stored)
    if wrong.size:
        raise RuntimeError(
            f"{split}: the loader and the shards disagree on the {name} of {wrong.size} "
            f"jets, the first at index {wrong[0]}, so the folds would not select the "
            "networks' jets"
        )


def check_split(
    split: str,
    loaded: tuple[np.ndarray, np.ndarray],
    stored: tuple[np.ndarray, np.ndarray],
) -> None:
    """Raise unless the loader's jets of *split* carry the shards' class indices and
    constituent counts, jet by jet.

    :param loaded: the loader's class indices and real-constituent counts, (jets,) each.
    :param stored: the same read off the shards, the counts capped at the loader's
        constituent count.
    :raises RuntimeError: at the first quantity that differs.
    """
    for name, ours, theirs in zip(("labels", "constituent counts"), loaded, stored):
        _compare(split, name, ours, theirs)


def _require_cache(folder: Path) -> None:
    """Raise unless the loader's cache in *folder* is complete."""
    missing = [name for name in CACHE_FILES if not (folder / name).is_file()]
    if missing:
        raise RuntimeError(
            f"No complete loader cache at {folder} (missing {missing}). Pass the "
            "--cache-dir of an existing one, or build it with a training at the same "
            "constituent count."
        )


def cached_record(revision: str, cache_dir: Path, nconstituents: int):
    """The record's loader at *nconstituents*, refused unless its cache is complete.

    The record is reached through the private ``_ensure_record`` because
    ``prepare_data``, the public route, would build a missing cache, several gigabytes
    beside the networks' own; the guard reads a cache and never builds one.
    """
    datamodule = HLS4MLDataModule(
        revision=revision, cache_dir=str(cache_dir), nconstituents=nconstituents
    )
    record = datamodule._ensure_record()
    _require_cache(record.cache_folder)
    return record


def loaded_jets(record, split: str) -> tuple[np.ndarray, np.ndarray]:
    """The class indices and the real-constituent counts of the loader's cached
    *split*."""
    data = record.load(split)
    # A padded slot is an all-zero row, which the loader's normalisation, a division,
    # keeps at zero.
    counts = (data.x != 0).any(dim=-1).sum(dim=1)
    return data.y.argmax(dim=1).numpy(), counts.numpy()


def check_order(
    data: dict[str, Split],
    data_dir: Path,
    revision: str,
    cache_dir: Path,
    nconstituents: int,
) -> str:
    """Raise unless the loader's cache at *nconstituents* holds the shards' jets in the
    shards' order; return the outcome, for the sidecars."""
    record = cached_record(revision, cache_dir, nconstituents)
    for split in SPLITS:
        counts = np.minimum(constituent_counts(data_dir, split), nconstituents)
        check_split(split, loaded_jets(record, split), (data[split][1], counts))
        log(f"order guard: the {len(counts)} {split} jets match the loader's")
    return f"passed against {record.cache_folder}"


def columns_of(names: list[str], row: str) -> list[int]:
    """The indices of the observables *row* reads: every one it does not leave out."""
    return [index for index, name in enumerate(names) if name not in ROWS[row]]


def pooled(splits: dict[str, Split]) -> Split:
    """The training then the validation jets, the order the datamodule pools them in."""
    (x_train, y_train), (x_val, y_val) = splits["train"], splits["validation"]
    return np.concatenate([x_train, x_val]), np.concatenate([y_train, y_val])


def row_splits(data: dict[str, Split], columns: list[int]) -> dict[str, Split]:
    """Every split restricted to *columns*, and the pool of two under ``"pool"``."""
    splits = {split: (x[:, columns], y) for split, (x, y) in data.items()}
    return splits | {"pool": pooled(splits)}


def _take(split: Split, indices: np.ndarray) -> Split:
    return split[0][indices], split[1][indices]


def _subsampled(indices: tuple[np.ndarray, ...], quick: bool) -> tuple[np.ndarray, ...]:
    """*indices* whole, or under ``--quick`` QUICK_JETS of each, drawn from SEED."""
    if not quick:
        return tuple(indices)
    rng = np.random.default_rng(SEED)
    picks = [rng.choice(i, min(QUICK_JETS, len(i)), replace=False) for i in indices]
    return tuple(np.sort(pick) for pick in picks)


def fold_split(splits: dict[str, Split], fold: int, run: Run) -> tuple[Split, Split]:
    """The training and the validation jets of fold *fold*, counted from one."""
    pool = splits["pool"]
    indices = fold_indices(len(pool[1]), N_FOLDS, run.kfold_seed, fold - 1)
    train_idx, val_idx = _subsampled(indices, run.quick)
    return _take(pool, train_idx), _take(pool, val_idx)


def record_split(splits: dict[str, Split], run: Run) -> tuple[Split, Split]:
    """The record's own training and validation jets, on which ``--select`` runs."""
    training, validation = splits["train"], splits["validation"]
    indices = (np.arange(len(training[1])), np.arange(len(validation[1])))
    train_idx, val_idx = _subsampled(indices, run.quick)
    return _take(training, train_idx), _take(validation, val_idx)


def standardiser(x_train: np.ndarray) -> Scale:
    """The float64 mean and standard deviation of the training jets, per observable.

    A constant observable (``j_ptfrac`` is 1 on every jet) keeps a unit scale, so that
    it is centred rather than divided by zero.
    """
    x = x_train.astype(np.float64)
    std = x.std(axis=0)
    return x.mean(axis=0), np.where(std > 0, std, 1.0)


def tensors(x: np.ndarray, y: np.ndarray, scale: Scale, device: torch.device) -> Pair:
    """*x* standardised by *scale*, in float32, and *y*, both on *device*."""
    mean, std = scale
    standardised = ((x - mean) / std).astype(np.float32)
    return torch.from_numpy(standardised).to(device), torch.from_numpy(y).to(device)


def build_mlp(n_inputs: int, recipe: Recipe) -> nn.Sequential:
    """Linear, ReLU and dropout per hidden width, then the five logits."""
    layers: list[nn.Module] = []
    for width in recipe.widths:
        layers += [nn.Linear(n_inputs, width), nn.ReLU(), nn.Dropout(recipe.dropout)]
        n_inputs = width
    return nn.Sequential(*layers, nn.Linear(n_inputs, len(CLASS_NAMES)))


def macro_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """The fraction of each class's jets called correctly, averaged over the classes, as
    the networks' ``test/acc`` and ``val/acc`` are (torchmetrics' default average)."""
    n_classes = len(CLASS_NAMES)
    accuracy = multiclass_accuracy(
        logits, labels, num_classes=n_classes, average="macro"
    )
    return float(accuracy)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """The logits of every jet (jets, 5), EVAL_CHUNK jets at a time."""
    model.eval()
    return torch.cat([model(chunk) for chunk in x.split(EVAL_CHUNK)])


@dataclass
class Best:
    """The best validation accuracy so far, its epoch counted from one, and its
    weights."""

    acc: float = -1.0
    epoch: int = 0
    state: dict[str, torch.Tensor] = field(default_factory=dict)

    def update(self, model: nn.Module, acc: float, epoch: int) -> None:
        """Keep *model*'s weights if *acc* beats the best so far; a tie keeps the
        earlier epoch."""
        if acc > self.acc:
            self.acc, self.epoch = acc, epoch
            self.state = {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_epoch(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    generator: torch.Generator,
) -> None:
    """One pass over the training jets in an order drawn from *generator*."""
    model.train()
    order = torch.randperm(len(y), device=y.device, generator=generator)
    for batch in order.split(BATCH):
        loss = F.cross_entropy(model(x[batch]), y[batch])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()


def _optimisers(
    model: nn.Module, lr: float
) -> tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
    """Adam, and the plateau schedule on the macro validation accuracy."""
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimiser, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
    )
    return optimiser, scheduler


def train(
    model: nn.Module, lr: float, training: Pair, validation: Pair, epochs: int
) -> tuple[Best, int]:
    """Train for *epochs*, or until STOP_PATIENCE epochs pass without a better macro
    validation accuracy; return the best epoch and the count of epochs run."""
    optimiser, scheduler = _optimisers(model, lr)
    generator = torch.Generator(device=training[1].device).manual_seed(SEED)
    best = Best()
    for epoch in range(1, epochs + 1):
        train_epoch(model, optimiser, *training, generator)
        acc = macro_accuracy(predict(model, validation[0]), validation[1])
        scheduler.step(acc)
        best.update(model, acc, epoch)
        if epoch - best.epoch >= STOP_PATIENCE:
            break
    return best, epoch


def fit(
    recipe: Recipe, training: Pair, validation: Pair, epochs: int
) -> tuple[nn.Module, Best, int]:
    """Fit one MLP from SEED and restore the weights of its best validation epoch."""
    torch.manual_seed(SEED)
    model = build_mlp(training[0].shape[1], recipe).to(training[0].device)
    best, epochs_run = train(model, recipe.lr, training, validation, epochs)
    model.load_state_dict(best.state)
    return model, best, epochs_run


def fit_split(
    recipe: Recipe, training: Split, validation: Split, run: Run
) -> tuple[nn.Module, Scale, dict]:
    """Fit on *training*, standardised by its own moments, and stop on *validation*;
    return the model, the standardisation, and what the fit did."""
    scale, start = standardiser(training[0]), time.perf_counter()
    pairs = [tensors(*split, scale, run.device) for split in (training, validation)]
    model, best, epochs_run = fit(recipe, *pairs, run.epochs)
    seconds = round(time.perf_counter() - start, 1)
    jets = {"train": len(training[1]), "validation": len(validation[1])}
    summary = {"val_acc": best.acc, "best_epoch": best.epoch, "epochs_run": epochs_run}
    return model, scale, summary | {"jets": jets, "seconds": seconds}


def save_outputs(path: Path, logits: np.ndarray, labels: np.ndarray) -> None:
    """The per-jet test outputs, in the layout of the networks' ``.npz`` less the
    transformed logits and the draws."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, logits_clean=logits.astype(np.float32), labels=labels.astype(np.int8)
    )


def run_fold(
    row: str, splits: dict[str, Split], recipe: Recipe, fold: int, run: Run
) -> tuple[dict[str, float], dict]:
    """Fit fold *fold* (counted from one), score the whole test split and save its
    logits; return the fold's metrics, as eval.py collects them, and what its fit
    did."""
    model, scale, summary = fit_split(recipe, *fold_split(splits, fold, run), run)
    x_test, y_test = splits["test"]
    logits = predict(model, tensors(x_test, y_test, scale, run.device)[0]).cpu()
    accuracy = macro_accuracy(logits, torch.from_numpy(y_test))
    save_outputs(
        run.run_dir / "outputs" / row / f"fold_{fold}.npz", logits.numpy(), y_test
    )
    log(f"{row} fold {fold}: {summary}, test accuracy {accuracy:.4f}")
    metrics = {"test/acc": accuracy} | clean_metrics(logits.numpy(), y_test)
    return metrics, summary | {"fold": fold, "test_acc": accuracy}


def select_recipe(splits: dict[str, Split], run: Run) -> tuple[Recipe, list[dict]]:
    """The point of GRID with the best macro validation accuracy on the record's own
    split, and every point with its accuracy."""
    training, validation = record_split(splits, run)
    grid = []
    for recipe in GRID:
        summary = fit_split(recipe, training, validation, run)[2]
        grid.append(asdict(recipe) | {"val_acc": summary["val_acc"]})
        log(f"grid {recipe}: validation accuracy {summary['val_acc']:.4f}")
    best = int(np.argmax([point["val_acc"] for point in grid]))
    return GRID[best], grid


def write_row(results_dir: Path, model: str, folds: list[dict[str, float]]) -> Path:
    """Append *model*'s row to the jet-level baseline table under *results_dir*, and
    return the table's path.

    The folds carry no invariance key and no cost, so every invariance column holds a
    NaN per fold, ``selected_inv_key`` reads ``invariance/mean_tv``, and the FLOP and
    eBOP columns hold NaN.
    """
    table = ResultsTable(
        output_path=str(results_dir / "results_table.csv"),
        method="augmentation",
        model_name=model,
        augmentation_scheme="none",
        nconstituents=NCONSTITUENTS,
        subdir=SUBDIR,
        invariance_metric=TOTAL_VARIATION,
    )
    table.write(folds)
    return table.output_path


def write_sidecar(path: Path, record: dict) -> None:
    """Write one row's record as JSON, replacing an earlier one of the same run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")


def run_row(row: str, data: dict[str, Split], names: list[str], run: Run) -> None:
    """Fit the five folds of *row*, append its results row and write its sidecar."""
    columns = columns_of(names, row)
    splits = row_splits(data, columns)
    recipe, grid = select_recipe(splits, run) if run.select else (DEFAULT, None)
    log(f"{row}: {len(columns)} observables, {recipe}")
    fits = [run_fold(row, splits, recipe, fold, run) for fold in range(1, N_FOLDS + 1)]
    table = write_row(run.results_dir, row, [metrics for metrics, _ in fits])
    record = {"row": row, "columns": [names[i] for i in columns], "table": str(table)}
    record |= {"recipe": asdict(recipe), "grid": grid, "folds": [f for _, f in fits]}
    write_sidecar(run.run_dir / f"{row}.json", record | run.meta | {"ended": _now()})


def provenance(device: torch.device, config: DictConfig, guard: str) -> dict:
    """What every sidecar repeats: the seeds, the record, the order guard's outcome, the
    software and the machine."""
    modules = (np, pa, sklearn, torch, torchmetrics)
    versions = {module.__name__: module.__version__ for module in modules}
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    meta = {"seed": SEED, "kfold_seed": int(config.kfold_seed)}
    meta |= {"revision": config.revision, "order_guard": guard, "versions": versions}
    return meta | {"host": platform.node(), "device": str(device), "gpu": gpu}


def settings(args: argparse.Namespace, config: DictConfig, meta: dict) -> Run:
    """The invocation's shared settings.

    ``--quick`` shortens every fit and moves the table under the run directory, so that
    no smoke run reaches ``results/``.
    """
    return Run(
        device=torch.device(args.device),
        epochs=QUICK_EPOCHS if args.quick else MAX_EPOCHS,
        quick=args.quick,
        select=args.select,
        kfold_seed=int(config.kfold_seed),
        run_dir=args.run_dir,
        results_dir=args.run_dir / "results" if args.quick else ROOT / "results",
        meta=meta,
    )


def _add_choices(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", nargs="+", choices=list(ROWS), default=list(ROWS))
    parser.add_argument(
        "--select",
        action="store_true",
        help="choose the recipe from the grid of eight on the record's own split",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"smoke run: {QUICK_JETS} training and validation jets per fit, "
        f"{QUICK_EPOCHS} epochs, the whole test split, the table under "
        "<run-dir>/results/",
    )


def _add_locations(parser: argparse.ArgumentParser) -> None:
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data" / "hf",
        help="parent of the loader's caches, whose nconst_150 the order guard reads",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "logs" / "jet_level" / stamp,
        help="where the per-jet outputs and the sidecars go",
    )


def parse_args() -> argparse.Namespace:
    """The command line; the default paths hang off PROJECT_ROOT."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_choices(parser)
    _add_locations(parser)
    return parser.parse_args()


def main() -> None:
    """Check the order of the jets, then fit, score and tabulate every row asked for."""
    args, started = parse_args(), _now()
    config = data_config()
    data_dir = shard_dir(config.revision)
    data, names = read_record(data_dir)
    guard = check_order(data, data_dir, config.revision, args.cache_dir, NCONSTITUENTS)
    meta = provenance(torch.device(args.device), config, guard) | {"started": started}
    run = settings(args, config, meta)
    for row in args.rows:
        run_row(row, data, names, run)


if __name__ == "__main__":
    main()
