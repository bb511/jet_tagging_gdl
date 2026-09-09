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

from src.data.components.augmentation import Transform, feature_index, make_transform
from src.data.components.dataset import HLS4MLDataset
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

REPO_ID = "fastmachinelearning/hls4ml_lhc_jets_150p"

# Supported symmetry methods and augmentation schemes.
SUPPORTED_METHODS = ("augmentation", "symm_loss")
SUPPORTED_SCHEMES = ("none", "so2", "so3", "lorentz", "lorentz_iso")

# Loader processes for validation and test. They only gather, where the training loader may also
# run the online transform, and every worker is a fork whose private memory grows to hundreds of
# MB as Python's refcounting breaks copy-on-write. Four NGT pods were OOMKilled on 2026-09-09
# carrying 32 persistent workers per run at 150 constituents, half of them validating.
EVAL_WORKERS = 2

warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


class HLS4MLDataModule(LightningDataModule):
    """Load the HF dataset, optionally augmented online and optionally k-folded.

    :param revision: string with a HF repository commit pin; None goes to HEAD.
    :param cache_dir: the local cache path ``paths.base_data_dir``;
        the hub snapshot itself is set by ``HF_HUB_CACHE`` env variable.
    :param method: the ``augmentation`` transforms the training batches under symm groups,
        ``symm_loss`` leaves them alone.
    :param augmentation_scheme: the symmetry group, one of the `SUPPORTED_SCHEMES` above;
        only used here when method is augmentation.
    :param transform_kwargs: the knobs of `~src.data.components.augmentation.make_transform`,
        shared with the SCL and the invariance probe, whose configs interpolate this node.
    :param nconstituents: how many constituents to keep by leading pT for each jet.
    :param features: the data features that the model consumes, ordered.
    :param n_folds: setting this to 1 uses the record's own train and validation splits;
        k >= 2 pools them and splits them again, with `fold_idx` choosing which fold is served.
    :param kfold_seed: fixes the kfold splitting so its reproducible.
    :param fold_idx: the fold served, counted from zero, which `set_fold` moves and train.py
        derives from its `fold` knob; ignored at n_folds 1.
    :param keep_full_features_at_test: keep all the data features in the test batches,
        in case they are required at test time to compute quantities of interest.
    """

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
        keep_full_features_at_test: bool = True,
    ) -> None:
        super().__init__()
        _validate_run(method, augmentation_scheme)

        self.revision = revision
        self.cache_dir = cache_dir

        # data processing
        self.method = method
        self.augmentation_scheme = augmentation_scheme
        self.transform_kwargs = dict(transform_kwargs or {})

        self.nconstituents = nconstituents
        self.features = list(features)

        # dataloader hyperparameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # folding
        self.n_folds = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx = fold_idx if n_folds >= 2 else 0
        self.keep_full_features_at_test = keep_full_features_at_test

        # Filled at instantiation.
        self.record = None
        self.feature_names: list[str] | None = None
        self.data_train: HLS4MLDataset | None = None
        self.data_val: HLS4MLDataset | None = None
        self.data_test: HLS4MLDataset | None = None

    def prepare_data(self) -> None:
        """Fetch the record and run its pipeline; cached, so reruns are cheap."""
        self._ensure_record().prepare()

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets of *stage* (``fit``, ``validate`` or ``test``, None for all)."""
        record = self._ensure_record()
        indices = self.feature_indices()

        if stage in (None, "fit") and self.data_train is None:
            self._setup_fit(record, indices)
        if stage in (None, "validate") and self.data_val is None:
            self._setup_validate(record, indices)
        if stage in (None, "test", "predict") and self.data_test is None:
            self._setup_test(record, indices)

    @property
    def norm_params(self) -> dict[str, tuple[str, float]] | None:
        """Get the normalisation parameters used in normalising the data."""
        return self.record.norm_params if self.record is not None else None

    @property
    def norm_scales(self) -> dict[str, float] | None:
        """Per-feature factors that return the normalised values to GeV."""
        params = self.norm_params
        return (
            None
            if params is None
            else {name: scale for name, (_, scale) in params.items()}
        )

    def feature_indices(self) -> list[int] | None:
        """Maps the list of features you give it to the indices in the actual data."""
        names = list(self._ensure_record().feature_names)
        indices = [_column(feature, names) for feature in self.features]

        return None if indices == list(range(len(names))) else indices

    def set_fold(self, fold_idx: int) -> None:
        """Set which fold will be served next."""
        if self.n_folds >= 2 and not 0 <= fold_idx < self.n_folds:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range for n_folds={self.n_folds}."
            )

        self.fold_idx = fold_idx if self.n_folds >= 2 else 0
        self.data_train = None
        self.data_val = None

    @property
    def n_train_samples(self) -> int:
        """Jets in the training half of the current fold, 0 before :meth:`setup`."""
        return 0 if self.data_train is None else self.data_train.n

    @property
    def n_val_samples(self) -> int:
        """Jets in the validation half of the current fold, 0 before :meth:`setup`."""
        return 0 if self.data_val is None else self.data_val.n

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

    def _record_root(self) -> str:
        """Where the data is stored on the machine."""
        return snapshot_download(REPO_ID, repo_type="dataset", revision=self.revision)

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        # nconstituents None and <= 0 both mean 'keep them all' here.
        return [
            f"paths.root_dir={root}",
            f"data.nconstituents={self.nconstituents if self.nconstituents else 0}",
            f"paths.base_data_dir={self.cache_dir}",
        ]

    def _ensure_record(self):
        """Fetch the record and instantiate the pipeline it comes with.

        Because the record comes with its own dataloader, we need to insert the root
        path of the that dataloader into sys. That's what 'root' refers to here.'
        """
        if self.record is not None:
            return self.record

        root = self._record_root()
        if root not in sys.path:
            sys.path.insert(0, root)

        self.record = instantiate(
            compose_record_config(root, self._record_overrides(root)).data
        )
        self.feature_names = list(self.record.feature_names)
        log.info(f"Record {REPO_ID} at {root}, cache {self.record.cache_folder}")

        return self.record

    def _setup_fit(self, record, indices: list[int] | None) -> None:
        """Build the training and validation datasets for the current fold."""
        x_train, y_train, x_val, y_val = self._fit_arrays(record)
        transform = self._train_transform(indices)
        shuffler = torch.Generator()
        shuffler.manual_seed(torch.initial_seed())

        # Keep either the transformed set of features or the whole set of features.
        x_train = _reduce(
            x_train, indices, self._keeps_full_width or transform is not None
        )
        x_val = _reduce(x_val, indices, self._keeps_full_width)

        self.data_train = self._dataset(
            x_train, y_train, shuffler=shuffler, transform=transform
        )
        self.data_val = self._dataset(x_val, y_val)
        log.info(f"Fit: x_train {tuple(x_train.shape)}, x_val {tuple(x_val.shape)}")

    def _setup_validate(self, record, indices: list[int] | None) -> None:
        split = record.load("validation")
        x_val = _reduce(split.x, indices, self._keeps_full_width)
        self.data_val = self._dataset(x_val, split.y)

    def _setup_test(self, record, indices: list[int] | None) -> None:
        split = record.load("test")
        keep_full = self._keeps_full_width or self.keep_full_features_at_test
        x_test = _reduce(split.x, indices, keep_full)
        self.data_test = self._dataset(x_test, split.y)
        log.info(f"Test: x_test {tuple(x_test.shape)}, y_test {tuple(split.y.shape)}")

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
        self, record
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The training and validation tensors for the current fold, with normalisation."""
        train, val = record.load("train"), record.load("validation")
        if self.n_folds <= 1:
            return train.x, train.y, val.x, val.y

        x_all, y_all = torch.cat([train.x, val.x]), torch.cat([train.y, val.y])
        log.info(f"Pooled training and validation splits: {len(x_all)} jets")

        return self._kfold_split(x_all, y_all)

    def _kfold_split(
        self, x_all: torch.Tensor, y_all: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the pooled jets into the training and validation halves of fold_idx."""
        folds = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed)
        train_idx, val_idx = list(folds.split(np.arange(len(x_all))))[self.fold_idx]
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

        The dataset batches internally. An element of this loader is one batch.
        The workers draw independently and hence persist throughout the epoch.
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
    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            config_dir=str(Path(root) / "configs"), version_base=None
        ):
            return compose("config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
        if outer is not None:
            GlobalHydra.instance().initialize(outer)


def _reduce_after(transform: Transform, indices: list[int]) -> Transform:
    """Wrapper for the data augmentation transformations.

    :param transform: The data augmentation method that performs the transformation on the data;
        takes all the features of the data set.
    :param indices: List of integers that specified the list of features that the model actually
        uses in its training.
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
    if augmentation_scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported augmentation_scheme {augmentation_scheme!r}, expected one of "
            f"{list(SUPPORTED_SCHEMES)}."
        )


def _column(feature: str, names: list[str]) -> int:
    idx = feature_index(names, feature)
    if idx is None:
        raise ValueError(
            f"Feature {feature!r} is absent from the record columns {names}."
        )

    return idx


def _reduce(
    x: torch.Tensor, indices: list[int] | None, keep_full: bool
) -> torch.Tensor:
    return x if keep_full or indices is None else x[:, :, indices]
