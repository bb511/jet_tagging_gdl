"""The only data route of this project: the published hls4ml jet record.

The record, https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p, carries its
own pipeline under ``loader/`` and the configuration tree that drives it under ``configs/``, so no
reading, restricting or normalising stage runs here: the record is downloaded, its own
configuration is composed, and the tensors it hands back are wrapped in the datasets the rest of
the project trains on.

Batches are ``(x, y)``, with ``x`` of shape (jets, constituents, features), float32, and ``y`` a
one-hot vector of shape (jets, 5). Padded constituents are all-zero rows. The width of ``x`` is the
contract the width guard in ``JetTaggingModule.forward`` is written against: training and
validation batches carry ``len(features)`` columns, or all 16 of the record under ``symm_loss``;
test batches carry all 16 under ``symm_loss`` or ``keep_full_features_at_test``, and
``len(features)`` otherwise.
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

# Duplicated from src.algorithms.SUPPORTED_METHODS and
# src.algorithms.losses.symmetry.SUPPORTED_GROUPS, rather than imported, so that the data
# package does not depend on the algorithms package.
SUPPORTED_METHODS = ("augmentation", "symm_loss")
SUPPORTED_SCHEMES = ("none", "so2", "so3", "lorentz", "lorentz_iso")

# Lightning warns that an IterableDataset defining __len__ is unreliable under several
# workers. It is exact here: the workers stride through one grid of batches, so their
# counts add up to len(dataset).
warnings.filterwarnings(
    "ignore",
    message=r"Your `IterableDataset` has `__len__` defined\..*",
    category=UserWarning,
)


def compose_record_config(root: str, overrides: list) -> DictConfig:
    """Compose the record's own configuration tree, leaving the caller's Hydra intact.

    initialize_config_dir refuses to run while a global Hydra is live, which it is inside any
    hydra.main application, and restores only the state it saw at its own entry, that is the state
    after the clear below, so the running application's Hydra is set aside and put back by hand.
    """
    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(Path(root) / "configs"), version_base=None):
            return compose("config", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()
        if outer is not None:
            GlobalHydra.instance().initialize(outer)


def _reduce_after(transform: Transform, indices: list[int]) -> Transform:
    """Wrap *transform* so that the batch it returns is narrowed to the model's columns.

    A transform reads px, py and pz whatever the model consumes, so it is handed the full record
    width and the reduction happens on its output.
    """

    def reduced(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x_sym, keep_mask = transform(x)
        return x_sym[:, :, indices], keep_mask

    return reduced


def _validate_run(method: str, augmentation_scheme: str) -> None:
    """Reject a configuration the pipeline cannot run, before anything is downloaded."""
    if method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method {method!r}, expected one of {list(SUPPORTED_METHODS)}."
        )
    if augmentation_scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"Unsupported augmentation_scheme {augmentation_scheme!r}, expected one of "
            f"{list(SUPPORTED_SCHEMES)}."
        )


class HLS4MLDataModule(LightningDataModule):
    """Serve the published jets, optionally augmented online and optionally k-folded.

    Nothing reaches the hub until :meth:`prepare_data`, so instantiating this class is offline.

    :param repo_id: the hub repository, or an existing local directory of that name, which is how a
        mirror is tested before it is uploaded.
    :param revision: a commit to pin, or None to follow the head. The record ships executable code,
        so a run executes whatever is at this commit.
    :param cache_dir: the record's ``paths.base_data_dir``, where its restricted and normalised
        caches are built; the hub snapshot itself goes to ``HF_HUB_CACHE``.
    :param method:``augmentation`` transforms the training batches here, ``symm_loss`` leaves them
        alone and the model penalises the change in its logits under a transform of its own.
    :param augmentation_scheme: the symmetry group, one of :data:`SUPPORTED_SCHEMES`, where
        ``lorentz`` boosts along the beam and ``lorentz_iso`` in a random direction. Under
        ``symm_loss`` it names the group the model draws from and is never rewritten here, so
        ``${data.augmentation_scheme}`` and the results table agree on what a run did.
    :param lorentz_eta_max: half-width of the uniform rapidity range of the boosts, along the beam
        for ``lorentz``, in a random direction for ``lorentz_iso``.
    :param nconstituents: leading constituents by pT to keep, 150 being all the record carries. A
        change of this value builds a new cache directory.
    :param features: the record columns the model consumes, in the order it expects them.
    :param n_folds: 1 uses the record's own train and validation splits; k >= 2 pools them and
        splits them again, with :meth:`set_fold` choosing which fold is served.
    :param kfold_seed: fixed, so that two runs' per-fold numbers are comparable.
    :param keep_full_features_at_test: keep the full record width in the test batches, so that the
        invariance probes find px, py and pz whatever the model reads.
    """

    def __init__(
        self,
        repo_id: str = "fastmachinelearning/hls4ml_lhc_jets_150p",
        revision: str | None = None,
        cache_dir: str = "data/hf",
        method: str = "augmentation",
        augmentation_scheme: str = "none",
        lorentz_eta_max: float = 1.0,
        nconstituents: int = 16,
        features: tuple[str, ...] = ("px", "py", "pz", "e"),
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        n_folds: int = 1,
        kfold_seed: int = 42,
        keep_full_features_at_test: bool = True,
    ) -> None:
        super().__init__()
        _validate_run(method, augmentation_scheme)

        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.method = method
        self.augmentation_scheme = augmentation_scheme
        self.lorentz_eta_max = lorentz_eta_max
        self.nconstituents = nconstituents
        self.features = list(features)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.n_folds = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx = 0
        self.keep_full_features_at_test = keep_full_features_at_test

        # Filled once the record is instantiated, which is the first time it is needed.
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
        """The record's fitted normalisation, None until :meth:`prepare_data` has run."""
        return self.record.norm_params if self.record is not None else None

    @property
    def norm_scales(self) -> dict[str, float] | None:
        """Per-column factors that return the stored values to GeV."""
        params = self.norm_params
        return None if params is None else {name: scale for name, (_, scale) in params.items()}

    def feature_indices(self) -> list[int] | None:
        """Columns of :attr:`features` inside the record's 16, in the order requested.

        None means that the two agree column for column, so nothing has to be sliced.
        """
        names = list(self._ensure_record().feature_names)
        indices = [self._column(feature, names) for feature in self.features]

        return None if indices == list(range(len(names))) else indices

    def set_fold(self, fold_idx: int) -> None:
        """Select the fold the next :meth:`setup` serves, and force it to rebuild.

        The test split is never folded, so the test dataset survives the switch.
        """
        if self.n_folds >= 2 and not 0 <= fold_idx < self.n_folds:
            raise ValueError(f"fold_idx {fold_idx} is out of range for n_folds={self.n_folds}.")

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
        return self._make_dataloader(self.data_train)

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.data_val)

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader(self.data_test)

    def _record_root(self) -> str:
        """Where the record lives on this machine, fetching it if it is not here yet."""
        local = Path(self.repo_id)
        if local.is_dir():
            return str(local.resolve())

        return snapshot_download(self.repo_id, repo_type="dataset", revision=self.revision)

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        # nconstituents None and <= 0 both mean 'keep them all' here; the record spells
        # that as 0, since a Hydra override cannot carry None through as an int.
        return [
            f"paths.root_dir={root}",
            f"data.nconstituents={self.nconstituents if self.nconstituents else 0}",
            f"paths.base_data_dir={self.cache_dir}",
        ]

    def _ensure_record(self):
        """Fetch the record and instantiate the pipeline it ships, once."""
        if self.record is not None:
            return self.record

        root = self._record_root()
        # The record's _target_s name loader.*, so its root has to be importable.
        if root not in sys.path:
            sys.path.insert(0, root)

        self.record = instantiate(compose_record_config(root, self._record_overrides(root)).data)
        self.feature_names = list(self.record.feature_names)
        log.info(f"Record {self.repo_id} at {root}, cache {self.record.cache_folder}")

        return self.record

    def _setup_fit(self, record, indices: list[int] | None) -> None:
        """Build the training and validation datasets for the current fold."""
        x_train, y_train, x_val, y_val = self._fit_arrays(record)
        transform = self._train_transform(indices)
        shuffler = torch.Generator()
        # Deterministic after lightning.seed_everything, and advanced by every epoch's
        # permutation, so the epochs of one run differ from each other.
        shuffler.manual_seed(torch.initial_seed())
        # A transform is handed the full record width whatever the model consumes, so the
        # training split keeps it and the reduction happens on the transform's output.
        x_train = self._reduce(x_train, indices, self._keeps_full_width or transform is not None)
        x_val = self._reduce(x_val, indices, self._keeps_full_width)
        self.data_train = self._dataset(x_train, y_train, shuffler=shuffler, transform=transform)
        self.data_val = self._dataset(x_val, y_val)
        log.info(f"Fit: x_train {tuple(x_train.shape)}, x_val {tuple(x_val.shape)}")

    def _setup_validate(self, record, indices: list[int] | None) -> None:
        split = record.load("validation")
        x_val = self._reduce(split.x, indices, self._keeps_full_width)
        self.data_val = self._dataset(x_val, split.y)

    def _setup_test(self, record, indices: list[int] | None) -> None:
        split = record.load("test")
        keep_full = self._keeps_full_width or self.keep_full_features_at_test
        x_test = self._reduce(split.x, indices, keep_full)
        self.data_test = self._dataset(x_test, split.y)
        log.info(f"Test: x_test {tuple(x_test.shape)}, y_test {tuple(split.y.shape)}")

    @property
    def _keeps_full_width(self) -> bool:
        """The SCL transforms inside the model, from all 16 columns, so it keeps them."""
        return self.method == "symm_loss"

    def _train_transform(self, indices: list[int] | None) -> Transform | None:
        """The online augmentation of the training batches, None if the run has none."""
        if self.method != "augmentation" or self.augmentation_scheme == "none":
            return None

        transform = make_transform(
            self.augmentation_scheme, self.feature_names, self.norm_scales, self.lorentz_eta_max
        )

        return transform if indices is None else _reduce_after(transform, indices)

    def _fit_arrays(self, record) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The training and validation tensors for the current fold, normalisation baked in."""
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
        # KFold reads the sample count alone, so the pool itself never leaves torch.
        train_idx, val_idx = list(folds.split(np.arange(len(x_all))))[self.fold_idx]
        log.info(
            f"Fold {self.fold_idx + 1}/{self.n_folds}: "
            f"{len(train_idx)} train, {len(val_idx)} val jets"
        )
        train_idx, val_idx = torch.from_numpy(train_idx), torch.from_numpy(val_idx)

        return x_all[train_idx], y_all[train_idx], x_all[val_idx], y_all[val_idx]

    def _dataset(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> HLS4MLDataset:
        return HLS4MLDataset(x.float(), y.float(), batch_size=self.batch_size, **kwargs)

    def _make_dataloader(self, dataset: HLS4MLDataset) -> DataLoader:
        # The dataset batches internally, so one item of this loader is one batch.
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=lambda batch: batch[0],
        )

    @staticmethod
    def _column(feature: str, names: list[str]) -> int:
        idx = feature_index(names, feature)
        if idx is None:
            raise ValueError(f"Feature {feature!r} is absent from the record columns {names}.")

        return idx

    @staticmethod
    def _reduce(x: torch.Tensor, indices: list[int] | None, keep_full: bool) -> torch.Tensor:
        return x if keep_full or indices is None else x[:, :, indices]
