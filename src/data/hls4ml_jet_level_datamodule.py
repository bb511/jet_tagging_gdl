"""The record's jet-level observables, served as jets of one slot.

Beside its constituents, every jet of the record carries 53 observables computed
upstream from the whole jet: its p_T, rapidity, mass and multiplicity, N-subjettiness,
energy correlation functions and groomed masses. T
The folds of `src.data.hls4ml_datamodule.fold_indices` select the same jets for the
networks and for these rows.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.components.dataset import HLS4MLDataset
from src.data.hls4ml_datamodule import fold_indices, record_snapshot
from src.symmetry import CLASS_NAMES
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

# The record's jet-level columns carry this prefix, its constituent columns j1_.
PREFIX = "j_"

# One split as read off the shards: the observables (jets, features) in float64 and the
# class indices (jets,), in the order of the logit columns.
Split = tuple[np.ndarray, np.ndarray]
# The mean and the standard deviation of every observable read, float64, (features,).
Scale = tuple[np.ndarray, np.ndarray]


def shard_paths(directory: Path, split: str) -> list[Path]:
    """The parquet shards of *split*, sorted as the record's loader sorts them, which
    fixes the order of the jets."""
    paths = sorted(Path(directory).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards under {directory}.")
    return paths


def _read_split(directory: Path, split: str, columns: list[str]) -> Split:
    """The observables *columns* and the class indices of every jet of *split*."""
    tables = [
        pq.read_table(path, columns=[*columns, "label"])
        for path in shard_paths(directory, split)
    ]
    table = pa.concat_tables(tables)
    x = np.stack([table[name].to_numpy() for name in columns], axis=1)
    if not np.isfinite(x).all():
        raise ValueError(f"The {split} observables hold a value that is not finite.")
    return x.astype(np.float64), table["label"].to_numpy().astype(np.int64)


def _standardiser(x_train: np.ndarray) -> Scale:
    """The per-observable mean and standard deviation of the training jets.

    A constant observable (``j_ptfrac`` is 1 on every jet) keeps a unit scale, so that
    it is centred rather than divided by zero.
    """
    std = x_train.std(axis=0)
    return x_train.mean(axis=0), np.where(std > 0, std, 1.0)


def _tensors(
    split: Split, scale: Scale, n_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The standardised jets (jets, 1, features) and the one-hot labels (jets,
    n_classes)."""
    (x, labels), (mean, std) = split, scale
    jets = torch.from_numpy(((x - mean) / std).astype(np.float32)).unsqueeze(1)
    return jets, torch.eye(n_classes)[torch.from_numpy(labels)]


def _take(split: Split, indices: np.ndarray) -> Split:
    return split[0][indices], split[1][indices]


def _validate_regime(method: str, augmentation_scheme: str) -> None:
    """Refuse every regime but the baseline: no transform acts on the observables."""
    if (method, augmentation_scheme) != ("augmentation", "none"):
        raise ValueError(
            "The jet-level observables are never transformed, so only the baseline "
            "(data.method=augmentation, data.augmentation_scheme=none) runs on them, "
            f"not method={method!r}, augmentation_scheme={augmentation_scheme!r}."
        )


def _loader(dataset: HLS4MLDataset, num_workers: int, pin_memory: bool) -> DataLoader:
    """One element per batch, from the dataset's own batching.

    Workers persist across epochs, so that their copy of the shuffler advances; fresh
    forks would replay the first permutation every epoch.
    """
    return DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=lambda batch: batch[0],
    )


class JetLevelDataModule(LightningDataModule):
    """Serve the record's jet-level observables, standardised.

    Every split is standardised with the mean and the standard deviation.
    These are derived from the training module of the data.

    :param revision: the record's commit, as pinned in ``configs/data/hls4ml.yaml``.
    :param shards_dir: the directory of the ``{train,validation,test}-*.parquet``
        shards; None reads the hub snapshot of *revision*.
    :param exclude: observables the row leaves out.
    :param method: must be ``augmentation``; kept for the configs that interpolate it.
    :param augmentation_scheme: must be ``none``.
    :param transform_kwargs: unused, kept for the configs that interpolate it.
    :param nconstituents: the label of the row in the results tables; the observables
        belong to the whole jet.
    :param n_folds: 1 serves the record's own training and validation splits; k >= 2
        pools them and serves fold ``fold_idx`` (counted from zero) of k.
    """

    # The batches have the row's observables alone, nothing for a probe to transform.
    keep_full_features_at_test = False
    norm_scales = None
    class_names = CLASS_NAMES

    def __init__(
        self,
        revision: str | None = None,
        shards_dir: str | None = None,
        exclude: Sequence[str] = (),
        method: str = "augmentation",
        augmentation_scheme: str = "none",
        transform_kwargs: dict[str, float | bool] | None = None,
        nconstituents: int = 150,
        batch_size: int = 2048,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_folds: int = 1,
        kfold_seed: int = 42,
        fold_idx: int = 0,
    ) -> None:
        super().__init__()
        _validate_regime(method, augmentation_scheme)
        self.revision = revision
        self.shards_dir = shards_dir
        self.exclude = list(exclude)
        self.nconstituents = nconstituents
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.n_folds = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx = fold_idx

        # Filled on first use.
        self._directory: Path | None = None
        self._feature_names: list[str] | None = None
        self._scale: Scale | None = None
        self.data_train: HLS4MLDataset | None = None
        self.data_val: HLS4MLDataset | None = None
        self.data_test: HLS4MLDataset | None = None

    @property
    def shards_directory(self) -> Path:
        """The directory of the shards, got from the hub."""
        if self._directory is None:
            self._directory = Path(
                self.shards_dir or Path(record_snapshot(self.revision)) / "data"
            )
        return self._directory

    @property
    def feature_names(self) -> list[str]:
        """Every jet-level observable of the record, in the shards' column order."""
        if self._feature_names is None:
            schema = pq.read_schema(shard_paths(self.shards_directory, "train")[0])
            self._feature_names = [n for n in schema.names if n.startswith(PREFIX)]
        return self._feature_names

    @property
    def features(self) -> list[str]:
        """The observables the row reads: every one but those of ``exclude``.

        :raises ValueError: when ``exclude`` names a column the shards do not have.
        """
        unknown = [name for name in self.exclude if name not in self.feature_names]
        if unknown:
            raise ValueError(
                f"exclude names {unknown}, absent from the observables "
                f"{self.feature_names}."
            )
        return [name for name in self.feature_names if name not in self.exclude]

    @property
    def scale(self) -> Scale:
        """The standardisation, fixed on the record's own training split."""
        if self._scale is None:
            self._scale = _standardiser(self._read("train")[0])
        return self._scale

    def setup(self, stage: str | None = None) -> None:
        """Build *stage* (``fit`` or ``test``, None for both)."""
        if stage in (None, "fit") and self.data_train is None:
            self._setup_fit()
        if stage in (None, "test") and self.data_test is None:
            # Every test jet is scored, as for the networks.
            self.data_test = self._dataset(self._read("test"), drop_last=False)
            log.info(f"Test: x_test {tuple(self.data_test.X.shape)}")

    def train_dataloader(self) -> DataLoader:
        return _loader(self.data_train, self.num_workers, self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return _loader(self.data_val, self.num_workers, self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return _loader(self.data_test, self.num_workers, self.pin_memory)

    def _read(self, split: str) -> Split:
        return _read_split(self.shards_directory, split, self.features)

    def _setup_fit(self) -> None:
        """Build the training and validation datasets of the current fold."""
        train, validation = self._read("train"), self._read("validation")
        if self._scale is None:
            self._scale = _standardiser(train[0])
        if self.n_folds >= 2:
            train, validation = self._fold(train, validation)
        shuffler = torch.Generator().manual_seed(torch.initial_seed())
        self.data_train = self._dataset(train, shuffler=shuffler)
        self.data_val = self._dataset(validation)
        log.info(f"Fit: {len(train[1])} train, {len(validation[1])} val jets")

    def _fold(self, train: Split, validation: Split) -> tuple[Split, Split]:
        """The training and validation halves of fold ``fold_idx`` of the pooled
        training then validation jets."""
        pool = tuple(np.concatenate(arrays) for arrays in zip(train, validation))
        train_idx, val_idx = fold_indices(
            len(pool[1]), self.n_folds, self.kfold_seed, self.fold_idx
        )
        log.info(f"Fold {self.fold_idx + 1}/{self.n_folds} of {len(pool[1])} jets")
        return _take(pool, train_idx), _take(pool, val_idx)

    def _dataset(self, split: Split, **kwargs) -> HLS4MLDataset:
        jets, labels = _tensors(split, self.scale, len(self.class_names))
        return HLS4MLDataset(jets, labels, batch_size=self.batch_size, **kwargs)
