# Lightning data module for the record published at:
# https://huggingface.co/datasets/fastmachinelearning/hls4ml_lhc_jets_150p
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from huggingface_hub import snapshot_download
from omegaconf import DictConfig

from src.data.components.dataset import HLS4MLDataset
from src.data.hls4ml_datamodule import HLS4MLDataModule

log = logging.getLogger(__name__)


def compose_record_config(root: str, overrides: list) -> DictConfig:
    """Compose the record's own configuration tree, leaving the caller's Hydra intact.

    initialize_config_dir refuses to run while a global Hydra is live, which it is
    inside any hydra.main application, and it restores only the state it saw at its own
    entry, which is the state after the clear below. The running application's Hydra is
    therefore set aside here and put back by hand. Composing outside an application,
    as the comparison script does, takes the same path with nothing to restore.
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


class HLS4MLHFDataModule(HLS4MLDataModule):
    """Train on the published record, through the loader that ships inside it.

    The record carries its own pipeline under loader/ and the configuration tree that
    drives it under configs/, so none of the raw stages of HLS4MLDataModule run here:
    the record is downloaded, its own configuration is composed, and the tensors it
    hands back are wrapped in the datasets the rest of the project expects. Everything
    downstream of setup (k-fold state, dataloaders, feature reduction, the online
    augmentation transform and its diagnostic) is inherited from the parent, which
    keeps the two routes behaviourally identical; if the raw route is ever deleted,
    those helpers move here.

    :param repo_id: The data set repository on the HuggingFace hub. An existing local
        directory is read directly instead, which is how the mirror is tested before
        it is uploaded.
    :param revision: A commit of that repository to pin, or None to follow its head.
    :param cache_dir: Where the record's derived caches go, its paths.base_data_dir.
        None leaves the record's own default, which honours HLS4ML_JETS_CACHE.
    """

    def __init__(
        self,
        repo_id: str = "fastmachinelearning/hls4ml_lhc_jets_150p",
        revision: str | None = None,
        cache_dir: str | None = None,
        **kwargs,
    ) -> None:
        # The parent's raw and processed paths are placeholders: both are replaced by
        # the record's own once it is known, which is the first time it is needed.
        super().__init__(data_dir=cache_dir or "data/hf", **kwargs)

        self.repo_id = repo_id
        self.revision = revision
        self.cache_dir = cache_dir
        self.record = None

    def prepare_data(self) -> None:
        """Fetch the record and run its pipeline. Cached, so reruns are cheap."""
        self._ensure_record().prepare()

    def setup(self, stage: str | None = None) -> None:
        """Build the datasets out of the record's tensors, with no raw file in sight."""
        record = self._ensure_record()
        self.norm_params = record.norm_params
        indices = self._get_feature_indices(Path(self.processed_data_dir))
        # symm_loss builds its symmetric partner inside the model, which needs the full
        # on-disk feature set whatever subset the model itself consumes.
        keep_full = self.method == "symm_loss"

        if stage in (None, "fit") and self.data_train is None:
            self._setup_fit(record, indices, keep_full)
        if stage in (None, "validate") and self.data_val is None:
            self._setup_validate(record, indices, keep_full)
        if stage in (None, "test", "predict") and self.data_test is None:
            self._setup_test(record, indices, keep_full)

        self._maybe_plot_overlap(stage)

    def _record_root(self) -> str:
        """Where the record lives on this machine."""
        local = Path(self.repo_id)
        if local.is_dir():
            return str(local.resolve())

        return snapshot_download(
            self.repo_id, repo_type="dataset", revision=self.revision
        )

    def _record_overrides(self, root: str) -> list[str]:
        """What to override when composing the record's own configuration."""
        # nconstituents None and <= 0 both mean 'keep them all' here; the record spells
        # that as 0, since a Hydra override cannot carry None through as an int.
        overrides = [
            f"paths.root_dir={root}",
            f"data.nconstituents={self.nconstituents if self.nconstituents else 0}",
        ]
        if self.cache_dir is not None:
            overrides.append(f"paths.base_data_dir={self.cache_dir}")

        return overrides

    def _ensure_record(self):
        """Fetch the record and instantiate the pipeline it ships, once."""
        if self.record is not None:
            return self.record

        root = self._record_root()
        # The record's _target_s name loader.*, so its root has to be importable.
        if root not in sys.path:
            sys.path.insert(0, root)

        self.record = instantiate(compose_record_config(root, self._record_overrides(root)).data)
        # The inherited feature reduction and diagnostics read the norm_params.npz that
        # the record writes beside its cached arrays, in this project's own format.
        self.processed_data_dir = str(self.record.cache_folder)
        self.feature_names = list(self.record.feature_names)
        self.pt_idx = self.record.pt_idx
        log.info(f"[record] {self.repo_id} at {root}, cache {self.processed_data_dir}")

        return self.record

    def _setup_fit(self, record, indices: np.ndarray | None, keep_full: bool) -> None:
        x_train, y_train, x_val, y_val = self._fit_arrays(record)
        transform = (
            self._make_batch_transform() if self.augmentation_scheme != "none" else None
        )
        shuffler = torch.Generator()
        shuffler.manual_seed(torch.initial_seed())
        # A transform needs the full feature set to compute so2/so3/lorentz and reduces
        # to self.features itself afterwards, so the train tensor is left wide for it.
        self.data_train = self._dataset(
            self._reduce(x_train, indices, keep_full or transform is not None),
            y_train, shuffler=shuffler, transform=transform,
        )
        self.data_val = self._dataset(self._reduce(x_val, indices, keep_full), y_val)

    def _setup_validate(self, record, indices: np.ndarray | None, keep_full: bool) -> None:
        split = record.load("validation")
        x_val = self._reduce(split.x.numpy(), indices, keep_full)
        self.data_val = self._dataset(x_val, split.y.numpy())

    def _setup_test(self, record, indices: np.ndarray | None, keep_full: bool) -> None:
        split = record.load("test")
        keep = keep_full or self.keep_full_features_at_test
        x_test = self._reduce(split.x.numpy(), indices, keep)
        log.info(f"[setup] Test - x_test: {x_test.shape}, y_test: {tuple(split.y.shape)}")
        self.data_test = self._dataset(x_test, split.y.numpy())

    def _fit_arrays(self, record) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The train/val pair for this fold. Normalisation is already baked in."""
        train, val = record.load("train"), record.load("validation")
        x_train, y_train = train.x.numpy(), train.y.numpy()
        x_val, y_val = val.x.numpy(), val.y.numpy()
        if self.n_folds <= 1:
            return x_train, y_train, x_val, y_val

        x_all = np.concatenate([x_train, x_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)
        log.info(f"[setup] Full training pool: {len(x_all)} jets")

        return self._kfold_split(x_all, y_all)

    def _dataset(self, x: np.ndarray, y: np.ndarray, **kwargs) -> HLS4MLDataset:
        return HLS4MLDataset(
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            batch_size=self.batch_size,
            **kwargs,
        )

    @staticmethod
    def _reduce(x: np.ndarray, indices: np.ndarray | None, keep_full: bool) -> np.ndarray:
        return x if keep_full or indices is None else x[:, :, indices]

    def _maybe_plot_overlap(self, stage: str | None) -> None:
        """The parent's optional overlap diagnostic, on the record's cached arrays."""
        if (
            stage not in (None, "fit")
            or self._overlap_plot_done
            or self.method != "augmentation"
            or self.augmentation_scheme == "none"
            or self.feature_names is None
        ):
            return

        save_dir = Path(__file__).resolve().parents[2] / "logs" / "plots"
        self._plot_online_aug_diagnostic(save_dir=str(save_dir), n_batches=20)
        self._overlap_plot_done = True
