"""The hls4ml dataset loader.

The record, https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p,
contains its own processing pipeline that is being applied here. See the docs there for
more details about the dataset and processing pipeline.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.data.components.augmentation import (
    Transform,
    make_transform,
    rebase_on_kept_axis,
    resolve_feature_indices,
)
from src.data.components.dataset import HLS4MLDataset
from src.symmetry import CLASS_NAMES, SCHEMES
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

REPO_ID = "fastmachinelearning/hls4ml_lhc_jets_150p"

# How a run acquires a symmetry. The schemes it can do so under are `src.symmetry.SCHEMES`.
SUPPORTED_METHODS = ("augmentation", "symm_loss")

# Loader processes for validation and test.
EVAL_WORKERS = 2

# Jets per rebasing chunk.
REBASE_CHUNK = 65536

warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


def record_snapshot(revision: str | None) -> str:
    """The root of the record's hub snapshot at *revision*, None meaning the head."""
    return snapshot_download(REPO_ID, repo_type="dataset", revision=revision)


def fold_indices(
    n_jets: int, n_folds: int, seed: int, fold_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """The training and validation indices of fold *fold_idx*, counted from zero, of
    *n_jets* pooled jets.

    The split depends on the pool's length and the seed alone, so a script that pools
    the same jets in the same order selects the same folds as the datamodule.
    """
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(folds.split(np.arange(n_jets)))[fold_idx]


class HLS4MLDataModule(LightningDataModule):
    """Load the HF dataset, optionally augmented online and optionally k-folded.

    Every split is rebased on the axis of the constituents kept as it is loaded, because
    the record measures etarel, phirel, deltaR and ptrel from the untruncated jet's axis,
    whereas every transform downstream measures them from the batch it is handed.
    `full_axis` keeps the record's columns instead, for the test of what that axis is
    worth. It is refused outside the baseline regime, whose batches no transform
    touches; the probe still transforms, so the invariance figures of such a run mean
    nothing.

    :param revision: string with a HF repository commit pin; None goes to HEAD.
    :param cache_dir: the local cache path ``paths.base_data_dir``;
        the hub snapshot itself is set by ``HF_HUB_CACHE`` env variable.
    :param method: the ``augmentation`` transforms the training batches under symm groups,
        ``symm_loss`` leaves them alone.
    :param augmentation_scheme: the symmetry group, one of `src.symmetry.SCHEMES`;
        only used here when method is augmentation.
    :param transform_kwargs: the knobs of `~src.data.components.augmentation.make_transform`,
        shared with the SCL and the invariance probe, whose configs interpolate this node.
    :param nconstituents: how many constituents to keep by leading pT for each jet.
    :param features: the data features that the model consumes, ordered.
    :param n_folds: setting this to 1 uses the record's own train and validation splits;
        k >= 2 pools them and splits them again, with `fold_idx` choosing which fold is served.
    :param kfold_seed: fixes the kfold splitting so its reproducible.
    :param fold_idx: the fold served, counted from zero, which train.py and eval.py derive
        from their `fold` knob; ignored at n_folds 1.
    :param full_axis: keep the record's axis columns rather than rebasing them.
    :param test_split: ``validation`` serves at test time the validation jets that fit
        validates on, those of fold `fold_idx` under folds, so that a checkpoint is
        selected without the test split.
    """

    # Test batches keep all 16 record columns whatever the model reads, so that the
    # invariance probe finds the momenta it transforms.
    keep_full_features_at_test = True
    class_names = CLASS_NAMES

    def __init__(
        self,
        revision: str | None = None,
        cache_dir: str = "data/hf",
        method: str = "augmentation",
        augmentation_scheme: str = "none",
        transform_kwargs: dict[str, float | bool] | None = None,
        nconstituents: int = 16,
        features: tuple[str, ...] = ("px", "py", "pz", "e"),
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_folds: int = 1,
        kfold_seed: int = 42,
        fold_idx: int = 0,
        full_axis: bool = False,
        test_split: str = "test",
    ) -> None:
        super().__init__()
        _validate_run(method, augmentation_scheme)
        if full_axis and (method, augmentation_scheme) != ("augmentation", "none"):
            raise ValueError("full_axis runs in the baseline regime alone")
        if test_split not in ("test", "validation"):
            raise ValueError(f"test_split must be test or validation, not {test_split}")

        self.revision = revision
        self.cache_dir = cache_dir

        # data processing
        self.method = method
        self.augmentation_scheme = augmentation_scheme
        self.transform_kwargs = dict(transform_kwargs or {})

        self.nconstituents = nconstituents
        self.features = list(features)
        self.full_axis = full_axis
        self.test_split = test_split

        # dataloader hyperparameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # folding
        self.n_folds = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx = fold_idx

        # Filled at instantiation.
        self.record = None
        self.data_train: HLS4MLDataset | None = None
        self.data_val: HLS4MLDataset | None = None
        self.data_test: HLS4MLDataset | None = None

    def prepare_data(self) -> None:
        """Fetch the record and run its pipeline; cached, so reruns are cheap."""
        self._ensure_record().prepare()

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets of *stage* (``fit`` or ``test``, None for both)."""
        indices = self.feature_indices()

        if stage in (None, "fit") and self.data_train is None:
            self._setup_fit(indices)
        if stage in (None, "test") and self.data_test is None:
            self._setup_test()

    @property
    def feature_names(self) -> list[str]:
        """The record's columns, in the order its loader stores them."""
        return list(self._ensure_record().feature_names)

    @property
    def norm_scales(self) -> dict[str, float] | None:
        """Per-feature factors that return the normalised values to GeV, None until the
        record has its normalisation."""
        params = None if self.record is None else self.record.norm_params
        return (
            None
            if params is None
            else {name: scale for name, (_, scale) in params.items()}
        )

    def feature_indices(self) -> list[int] | None:
        """The record columns of ``features``, None when they are every column in the
        record's order."""
        names = self.feature_names
        indices = resolve_feature_indices(self.features, names)

        return None if indices == list(range(len(names))) else indices

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.data_train, self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.data_val, self._eval_workers)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.data_test, self._eval_workers)

    @property
    def _eval_workers(self) -> int:
        """Loader processes for a pass that never transforms; 0 workers stays 0."""
        return min(self.num_workers, EVAL_WORKERS)

    def load_split(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The jets of the record's own *split*, every column, rebased as setup()
        rebases them, and their labels; setup("fit") pools and re-splits them under
        folds."""
        return self._load(split)

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        return [
            f"paths.root_dir={root}",
            f"data.nconstituents={self.nconstituents}",
            f"paths.base_data_dir={self.cache_dir}",
        ]

    def _ensure_record(self):
        """Fetch the record and instantiate the pipeline it comes with.

        Because the record comes with its own dataloader, we need to insert the root
        path of the that dataloader into sys. That's what 'root' refers to here.'
        """
        if self.record is not None:
            return self.record

        root = record_snapshot(self.revision)
        if root not in sys.path:
            sys.path.insert(0, root)

        self.record = instantiate(
            compose_record_config(root, self._record_overrides(root)).data
        )
        log.info(f"Record {REPO_ID} at {root}, cache {self.record.cache_folder}")

        return self.record

    def _read(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The jets of the record's own *split*, every column as the loader stores them,
        and their one-hot labels."""
        data = self._ensure_record().load(split)
        return data.x, data.y

    def _load(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the jets and the labels of *split*, rebased on the kept axis unless
        `full_axis` is set."""
        x, y = self._read(split)
        x, scales = x.clone(), self.norm_scales
        if not self.full_axis:
            for chunk in x.split(REBASE_CHUNK):
                chunk.copy_(rebase_on_kept_axis(chunk, self.feature_names, scales))
        return x, y

    def _setup_fit(self, indices: list[int] | None) -> None:
        """Build the training and validation datasets for the current fold."""
        x_train, y_train, x_val, y_val = self._fit_arrays()
        x_val = _reduce(x_val, indices, self._keeps_full_width)
        self.data_val = self._dataset(x_val, y_val)
        self.data_train = self._train_dataset(x_train, y_train, indices)
        log.info(
            f"Fit: x_train {tuple(self.data_train.X.shape)}, x_val {tuple(x_val.shape)}"
        )

    def _train_dataset(
        self, x: torch.Tensor, y: torch.Tensor, indices: list[int] | None
    ) -> HLS4MLDataset:
        """The shuffled training dataset, transformed online if the run augments."""
        transform = self._train_transform(indices)
        shuffler = torch.Generator().manual_seed(torch.initial_seed())
        # Keep either the transformed set of features or the whole set of features.
        x = _reduce(x, indices, self._keeps_full_width or transform is not None)
        return self._dataset(x, y, shuffler=shuffler, transform=transform)

    def _setup_test(self) -> None:
        """Build the test dataset at the record's full width, as
        `keep_full_features_at_test` states."""
        if self.test_split == "validation":
            _, _, x_test, y_test = self._fit_arrays()
        else:
            x_test, y_test = self._load("test")
        # Every test jet is scored, so that each row of the results counts the same jets.
        self.data_test = self._dataset(x_test, y_test, drop_last=False)
        log.info(f"Test: x_test {tuple(x_test.shape)}, y_test {tuple(y_test.shape)}")

    @property
    def _keeps_full_width(self) -> bool:
        """Keep all the features if the methods is symmetry loss."""
        return self.method == "symm_loss"

    def _train_transform(self, indices: list[int] | None) -> Transform | None:
        """The online augmentation of the training batches, None if the run has none."""
        if self.method != "augmentation" or self.augmentation_scheme == "none":
            return None

        transform = make_transform(
            self.augmentation_scheme,
            self.feature_names,
            self.norm_scales,
            **self.transform_kwargs,
        )

        return transform if indices is None else _reduce_after(transform, indices)

    def _fit_arrays(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The training and validation tensors for the current fold, with
        normalisation."""
        x_train, y_train = self._load("train")
        x_val, y_val = self._load("validation")
        if self.n_folds <= 1:
            return x_train, y_train, x_val, y_val

        x_all, y_all = torch.cat([x_train, x_val]), torch.cat([y_train, y_val])
        # Freed before the folds are sliced out of the pool, lowering the peak by a copy
        # of both splits.
        del x_train, x_val
        log.info(f"Pooled training and validation splits: {len(x_all)} jets")

        return self._kfold_split(x_all, y_all)

    def _kfold_split(
        self, x_all: torch.Tensor, y_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the pooled jets into the training and validation halves of fold_idx."""
        train_idx, val_idx = fold_indices(
            len(x_all), self.n_folds, self.kfold_seed, self.fold_idx
        )
        log.info(
            f"Fold {self.fold_idx + 1}/{self.n_folds}: "
            f"{len(train_idx)} train, {len(val_idx)} val jets"
        )
        train_idx, val_idx = torch.from_numpy(train_idx), torch.from_numpy(val_idx)

        return x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx]

    def _dataset(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> HLS4MLDataset:
        return HLS4MLDataset(x.float(), y.float(), batch_size=self.batch_size, **kwargs)

    def _make_dataloader(self, dataset: HLS4MLDataset, num_workers: int) -> DataLoader:
        """Make the loader for each of the train, validation, and test data.

        The dataset batches internally. An element of this loader is one batch. The
        workers draw independently and hence persist throughout the epoch.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: batch[0],
        )


# Helper methods
# --------------
def compose_record_config(root: str, overrides: list) -> DictConfig:
    """Compose the configuration tree that is shipped with the record."""
    config_dir = str(Path(root) / "configs")
    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose("config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
        if outer is not None:
            GlobalHydra.instance().initialize(outer)


def _reduce_after(transform: Transform, indices: list[int]) -> Transform:
    """Wrapper for the data augmentation transformations.

    :param transform: The data augmentation method that performs the transformation on
        the data; takes all the features of the data set.
    :param indices: List of integers that specified the list of features that the model
        actually uses in its training.
    """

    def reduced(x: torch.Tensor) -> torch.Tensor:
        return transform(x)[:, :, indices]

    return reduced


def _validate_run(method: str, augmentation_scheme: str) -> None:
    """Reject a configuration the data pipeline cannot run, before downloading."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}, expected one of {list(SUPPORTED_METHODS)}."
        )
    if augmentation_scheme not in SCHEMES:
        raise ValueError(
            f"Unsupported augmentation_scheme {augmentation_scheme!r}, expected one of "
            f"{list(SCHEMES)}."
        )


def _reduce(
    x: torch.Tensor, indices: list[int] | None, keep_full: bool
) -> torch.Tensor:
    return x if keep_full or indices is None else x[:, :, indices]
