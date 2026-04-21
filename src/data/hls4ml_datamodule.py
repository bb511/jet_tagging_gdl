"""
Data module for HLS4ML jet classification.

Three training methods are supported:
1. augmentation  — offline augmentation baked into processed arrays
2. symm_loss     — no augmentation; symmetry enforced via loss term
3. another_method — placeholder for future methods (same pipeline as symm_loss for now)

K-fold cross validation is supported via n_folds > 1 in the config.
Call set_fold(fold_idx) before each fold's setup() in train.py.
"""

from pathlib import Path
from typing import Any, Optional, Tuple, Dict

import logging
import os
import tarfile

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import torch
import wget
from lightning import LightningDataModule
from torch.utils.data import TensorDataset, DataLoader

try:
    from IPython.display import display as ipy_display
except Exception:
    ipy_display = None

from src.utils.utils import plot_features_overlap_from_datasets
from src.utils.augmentation import apply_augmentation

log = logging.getLogger(__name__)

CENTERED_FEATURES = {
    "j1_px", "j1_py", "j1_pz",
    "j1_eta", "j1_etarel", "j1_etarot",
    "j1_phi", "j1_phirel", "j1_phirot",
    "j1_costheta", "j1_costhetarel",
}

POSITIVE_FEATURES = {
    "j1_pt", "j1_ptrel",
    "j1_e",  "j1_erel",
    "j1_deltaR",
}

# Methods that use the baseline (non-augmented) pipeline
_BASELINE_METHODS = {"symm_loss", "another_method"}


class HLS4MLDataModule(LightningDataModule):
    """Data module with a step-by-step raw->processed pipeline.

    Pipeline order:
    1. ``prepare_data`` downloads/extracts train and val archives (if missing).
    2. Raw H5 data is loaded into numpy arrays.
    3. Data is preprocessed (nconst restriction + pt sorting).
    4. [augmentation only] Train split is augmented (so2/so3/lorentz).
    5. Physics normalisation is applied and arrays are saved as ``.npy`` files.

    K-fold usage in train.py:
        datamodule.prepare_data()
        for fold_idx in range(cfg.data.n_folds):
            datamodule.set_fold(fold_idx)
            datamodule.setup(stage="fit")
            trainer.fit(model, datamodule)
            trainer.test(model, datamodule)
    """

    def __init__(
        self,
        data_dir: str = "/cluster/scratch/agogia/datasets/",
        method: str = "augmentation",
        batch_size: int = 256,
        num_workers: int = 7,
        pin_memory: bool = False,
        nconstituents: int = 32,
        features: Optional[Any] = None,
        augmentation_scheme: str = "lorentz",
        augmentation_seed: Optional[int] = None,
        rotation_nbr_rot: int = 8,
        rotation_shuffle: bool = True,
        lorentz_nbr_rot: int = 8,
        lorentz_eta_max: float = 2.0,
        n_folds: int = 1,
        kfold_seed: int = 42,
    ) -> None:
        super().__init__()

        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.pin_memory    = pin_memory
        self.nconstituents = nconstituents
        self.features      = features if features is not None else ["px", "py", "pz", "etarel", "phirel"]

        # --- Validate method ---
        self.method = method.lower()
        if self.method not in {"augmentation"} | _BASELINE_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. "
                f"Expected one of: ['augmentation', 'symm_loss', 'another_method']."
            )

        # --- Validate & resolve augmentation scheme ---
        requested_scheme = augmentation_scheme.lower()
        if requested_scheme not in {"none", "so2", "so3", "lorentz"}:
            raise ValueError(
                f"Unsupported augmentation_scheme='{augmentation_scheme}'. "
                "Expected one of: ['none', 'so2', 'so3', 'lorentz']."
            )

        if self.method in _BASELINE_METHODS:
            if requested_scheme != "none":
                log.warning(
                    f"method='{self.method}' forces augmentation_scheme='none' "
                    f"(got '{requested_scheme}'). Overriding."
                )
            self.augmentation_scheme = "none"
        else:
            self.augmentation_scheme = requested_scheme

        self._rng              = np.random.default_rng(augmentation_seed)
        self.rotation_nbr_rot  = rotation_nbr_rot
        self.rotation_shuffle  = rotation_shuffle
        self.lorentz_nbr_rot   = lorentz_nbr_rot
        self.lorentz_eta_max   = lorentz_eta_max

        # --- K-fold config ---
        self.n_folds    = n_folds
        self.kfold_seed = kfold_seed
        self.fold_idx   = 0      # current fold; updated by set_fold()

        # --- Paths ---
        self.data_path    = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        nconst_tag = (
            "full"
            if self.nconstituents is None or self.nconstituents <= 0
            else f"nconst_{self.nconstituents}"
        )
        # Canonical layout (method-aware):
        #   <data_dir>/processed/<method>/<augmentation_scheme>/<nconst_tag>/
        # This prevents collisions between augmentation and symm_loss pipelines.
        self.processed_data_dir = os.path.join(
            self.data_path,
            "processed",
            self.method,
            self.augmentation_scheme,
            nconst_tag,
        )
        # Backward-compatibility layout used in older runs:
        #   <data_dir>/processed/<augmentation_scheme>/<nconst_tag>/
        self.legacy_processed_data_dir = os.path.join(
            self.data_path,
            "processed",
            self.augmentation_scheme,
            nconst_tag,
        )

        self.raw_data_url     = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        self.raw_data_url_val = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"

        # Cached raw arrays
        self.x_train_raw   = None
        self.y_train_raw   = None
        self.x_val_raw     = None
        self.y_val_raw     = None
        self.x_test_raw    = None
        self.y_test_raw    = None
        self.feature_names = None
        self.pt_idx        = None
        self.norm_params   = None

        # Dataset references — reset by set_fold() between folds
        self.data_train = None
        self.data_val   = None
        self.data_test  = None

        self._overlap_plot_done = False
        self._skip_prepare_data = False

    # ------------------------------------------------------------------
    # K-fold interface
    # ------------------------------------------------------------------
    def set_fold(self, fold_idx: int) -> None:
        """Switch to a new fold. No-op for n_folds=1 (main training)."""
        if self.n_folds <= 1:
            # For main training, do not change fold or log
            self.fold_idx = 0
            return
        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range for n_folds={self.n_folds}."
            )
        self.fold_idx   = fold_idx
        self.data_train = None   # force setup() to reload
        self.data_val   = None
        self.data_test  = None
        # Only log for actual KFold runs, not for main training (fold_idx==0)
        if fold_idx != 0:
            log.info(f"[set_fold] Switched to fold {fold_idx + 1}/{self.n_folds}.")

    # ------------------------------------------------------------------
    # Convenience properties used by train.py for logging
    # ------------------------------------------------------------------
    @property
    def n_train_samples(self) -> int:
        return len(self.data_train) if self.data_train is not None else 0

    @property
    def n_val_samples(self) -> int:
        return len(self.data_val) if self.data_val is not None else 0

    # ------------------------------------------------------------------
    # Public pipeline entry point
    # ------------------------------------------------------------------
    def _processed_arrays_exist(self, processed_data_dir: str) -> bool:
        processed_dir = Path(processed_data_dir)
        return all(
            (processed_dir / fname).exists()
            for fname in ("x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy")
        )

    def _resolve_processed_dir(self) -> str:
        """Resolve processed dir, preferring canonical method-aware path.

        If only legacy arrays exist, reuse them for compatibility with older runs.
        """
        if self._processed_arrays_exist(self.processed_data_dir):
            return self.processed_data_dir
        if self._processed_arrays_exist(self.legacy_processed_data_dir):
            log.warning(
                "Using legacy processed directory layout: "
                f"{self.legacy_processed_data_dir}. "
                "Consider regenerating arrays into the method-aware path: "
                f"{self.processed_data_dir}"
            )
            return self.legacy_processed_data_dir
        return self.processed_data_dir

    def prepare_data(self) -> None:
        if self._skip_prepare_data:
            return

        resolved_dir = self._resolve_processed_dir()
        if self._processed_arrays_exist(resolved_dir):
            # Stick to whichever existing layout was resolved so setup() reads the same files.
            self.processed_data_dir = resolved_dir
            print(
                f"Processed arrays already exist in {resolved_dir}. "
                "Skipping preparation."
            )
            return

        # Step 1: Download
        raw_path = Path(self.raw_data_dir)
        has_tar  = any(raw_path.glob("*.tar.gz"))
        has_h5   = any(raw_path.rglob("*.h5"))
        if not has_tar and not has_h5:
            self._download_data_from_url(self.raw_data_url, val=False)
            self._download_data_from_url(self.raw_data_url_val, val=True)
        else:
            print("Raw data already exists. Skipping download.")

        # Step 2: Load raw H5
        (
            self.x_train_raw,
            self.x_val_raw,
            self.y_train_raw,
            self.y_val_raw,
            self.feature_names,
            self.pt_idx,
        ) = self._load_all_h5_files(self.raw_data_dir, isTest=False)

        # Step 3: Method-specific pipeline
        if self.method == "augmentation":
            self._pipeline_augmentation(
                self.raw_data_dir, self.processed_data_dir,
                self.nconstituents, self.features,
            )
        elif self.method in _BASELINE_METHODS:
            self._pipeline_baseline(
                self.raw_data_dir, self.processed_data_dir,
                self.nconstituents, self.features,
            )
        else:
            raise ValueError(f"Unsupported method '{self.method}'.")

    # ------------------------------------------------------------------
    # Pipeline A: augmentation
    # ------------------------------------------------------------------
    def _pipeline_augmentation(
        self, raw_data_dir, processed_data_dir, nconstituents, features
    ) -> None:
        log.info("[augmentation] Starting augmentation pipeline.")

        x_train_sorted = self._restrict_and_sort_by_pt(self.x_train_raw, self.pt_idx, nconstituents)
        x_val_sorted   = self._restrict_and_sort_by_pt(self.x_val_raw,   self.pt_idx, nconstituents)

        x_train_aug, y_train_aug = self._setup_augmentation_schemes(x_train_sorted, self.y_train_raw)

        # Fit norm on unaugmented train split
        norm_params  = self.fit_physics_norm(x_train_sorted, self.feature_names)
        x_train_aug  = self.apply_physics_norm(x_train_aug,  self.feature_names, norm_params)
        x_val_norm   = self.apply_physics_norm(x_val_sorted, self.feature_names, norm_params)
        self.norm_params = norm_params

        log.info(f"[augmentation] train: {x_train_aug.shape}, val: {x_val_norm.shape}")
        self._save_processed_arrays(
            processed_data_dir, x_train_aug, y_train_aug,
            x_val_norm, self.y_val_raw, norm_params,
        )

    # ------------------------------------------------------------------
    # Pipeline B: symm_loss / another_method
    # ------------------------------------------------------------------
    def _pipeline_baseline(
        self, raw_data_dir, processed_data_dir, nconstituents, features
    ) -> None:
        log.info(f"[{self.method}] Starting baseline pipeline (no augmentation).")

        x_train_sorted = self._restrict_and_sort_by_pt(self.x_train_raw, self.pt_idx, nconstituents)
        x_val_sorted   = self._restrict_and_sort_by_pt(self.x_val_raw,   self.pt_idx, nconstituents)

        norm_params  = self.fit_physics_norm(x_train_sorted, self.feature_names)
        x_train_norm = self.apply_physics_norm(x_train_sorted, self.feature_names, norm_params)
        x_val_norm   = self.apply_physics_norm(x_val_sorted,  self.feature_names, norm_params)
        self.norm_params = norm_params

        log.info(f"[{self.method}] train: {x_train_norm.shape}, val: {x_val_norm.shape}")
        self._save_processed_arrays(
            processed_data_dir, x_train_norm, self.y_train_raw,
            x_val_norm, self.y_val_raw, norm_params,
        )

    # ------------------------------------------------------------------
    # Download / extract
    # ------------------------------------------------------------------
    def _download_data_from_url(self, url: str, val: bool = False) -> None:
        split_name    = "val" if val else "train"
        expected_glob = "jetImage_*_150p_*.h5"
        os.makedirs(self.raw_data_dir, exist_ok=True)
        split_dir = Path(self.raw_data_dir) / split_name

        if (split_dir.exists() and any(split_dir.glob(expected_glob))) or \
                any(Path(self.raw_data_dir).glob(expected_glob)):
            print(f"{split_name.capitalize()} data already present. Skipping download.")
            return

        filename = os.path.join(self.raw_data_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {url} ...")
            wget.download(url, filename)
            print("\nDownload complete.")

        print(f"Extracting {filename} ...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=self.raw_data_dir)
        print("Extraction complete.")

    # ------------------------------------------------------------------
    # Raw H5 load
    # ------------------------------------------------------------------
    def _load_all_h5_files(
        self,
        data_dir: str,
        x_name: str = "jetConstituentList",
        y_name: str = "jets",
        isTest: bool = False,
    ) -> Tuple:
        subfolder = "val" if isTest else "train"
        h5_files  = sorted((Path(data_dir) / subfolder).glob("jetImage_*_150p_*.h5"))
        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {data_dir}/{subfolder}")

        all_x, all_y  = [], []
        feature_names = None
        pt_idx        = None

        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                x = np.asarray(f[x_name])
                y = np.asarray(f[y_name])
                if feature_names is None:
                    feature_names = [
                        feat.decode("utf-8")
                        for feat in f["particleFeatureNames"]
                        if feat != b"j1_pdgid"
                    ]
                    pt_idx = feature_names.index("j1_pt")
                all_x.append(x)
                all_y.append(y[:, -6:-1])

        x_data = np.concatenate(all_x, axis=0)
        y_data = np.concatenate(all_y, axis=0)

        if not isTest:
            x_data, x_val, y_data, y_val = train_test_split(
                x_data, y_data, test_size=0.2, random_state=42
            )
            print(f"Train raw: {x_data.shape}, Val raw: {x_val.shape}")
            self._show_class_distribution(y_data, y_data.sum(axis=0).astype(int))
            return x_data, x_val, y_data, y_val, feature_names, pt_idx
        else:
            print(f"Test raw: {x_data.shape}")
            self._show_class_distribution(y_data, y_data.sum(axis=0).astype(int))
            return x_data, y_data, feature_names, pt_idx

    def _show_class_distribution(self, y_data, class_counts) -> None:
        class_df = pd.DataFrame({
            "class_idx": np.arange(y_data.shape[1]),
            "count":     class_counts,
            "percent":   100 * class_counts / class_counts.sum(),
        })
        if ipy_display is not None:
            ipy_display(class_df)
        else:
            print(class_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------
    def _restrict_and_sort_by_pt(self, x_data, pt_idx, nconstituents) -> np.ndarray:
        target = x_data.shape[1] if (nconstituents is None or nconstituents <= 0) else nconstituents
        if x_data.shape[1] < target:
            padding = np.zeros(
                (x_data.shape[0], target - x_data.shape[1], x_data.shape[2]),
                dtype=x_data.dtype,
            )
            x_data = np.concatenate((x_data, padding), axis=1)
        else:
            x_data = x_data[:, :target, :]
        sort_idx = np.argsort(x_data[:, :, pt_idx], axis=1)[:, ::-1]
        return np.take_along_axis(x_data, sort_idx[:, :, np.newaxis], axis=1)

    def _setup_augmentation_schemes(self, x_train, y_train) -> Tuple[np.ndarray, np.ndarray]:
        return apply_augmentation(
            scheme=self.augmentation_scheme,
            x_train=x_train,
            y_train=y_train,
            feature_names=self.feature_names,
            rng=self._rng,
            rotation_nbr_rot=self.rotation_nbr_rot,
            rotation_shuffle=self.rotation_shuffle,
            lorentz_nbr_rot=self.lorentz_nbr_rot,
            lorentz_eta_max=self.lorentz_eta_max,
        )

    def fit_physics_norm(self, x_train, feature_names) -> Dict[str, Tuple[str, float]]:
        params: Dict[str, Tuple[str, float]] = {}
        for i, name in enumerate(feature_names):
            vals = x_train[:, :, i]
            vmin, vmax = float(vals.min()), float(vals.max())
            if name in CENTERED_FEATURES:
                params[name] = ("centered", 0.5 * (vmax - vmin))
            elif name in POSITIVE_FEATURES:
                params[name] = ("positive", vmax)
            else:
                raise ValueError(f"Feature '{name}' not categorised in CENTERED or POSITIVE sets.")
        return params

    def apply_physics_norm(self, x, feature_names, params, eps=1e-8) -> np.ndarray:
        x_norm = x.copy()
        for i, name in enumerate(feature_names):
            kind, scale = params[name]
            if kind in {"centered", "positive"}:
                x_norm[:, :, i] /= (scale + eps)
            else:
                raise ValueError(f"Unknown normalisation kind '{kind}' for feature '{name}'.")
        return x_norm

    def _save_processed_arrays(
        self, processed_data_dir, x_train, y_train, x_val, y_val, norm_params=None
    ) -> None:
        os.makedirs(processed_data_dir, exist_ok=True)
        np.save(os.path.join(processed_data_dir, "x_train.npy"), x_train)
        np.save(os.path.join(processed_data_dir, "y_train.npy"), y_train)
        np.save(os.path.join(processed_data_dir, "x_val.npy"),   x_val)
        np.save(os.path.join(processed_data_dir, "y_val.npy"),   y_val)
        if norm_params is not None:
            self._save_norm_params(processed_data_dir, norm_params)

    def _save_norm_params(self, processed_data_dir, params) -> None:
        names  = np.array(list(params.keys()), dtype="U64")
        kinds  = np.array([params[n][0] for n in names], dtype="U16")
        scales = np.array([params[n][1] for n in names], dtype=np.float32)
        np.savez(
            os.path.join(processed_data_dir, "norm_params.npz"),
            feature_names=names, kinds=kinds, scales=scales,
        )

    def _get_feature_indices(self, processed_dir: Path) -> Optional[np.ndarray]:
        npz_path = processed_dir / "norm_params.npz"
        if not npz_path.exists():
            return None
        all_names = list(np.load(npz_path, allow_pickle=False)["feature_names"])
        indices = []
        for feat in self.features:
            if feat in all_names:
                indices.append(all_names.index(feat))
            elif f"j1_{feat}" in all_names:
                indices.append(all_names.index(f"j1_{feat}"))
            else:
                raise ValueError(
                    f"Requested feature '{feat}' not found in saved names: {all_names}"
                )
        return np.array(indices)

    # ------------------------------------------------------------------
    # K-fold split helper
    # ------------------------------------------------------------------
    def _kfold_split(
        self, x_all: np.ndarray, y_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split the full processed pool into train/val for self.fold_idx.

        The saved x_train + x_val are concatenated to form the full pool.
        KFold then splits this pool. The test set (from separate H5 val
        files) is always kept separate and is never part of the pool.
        """
        kf     = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.kfold_seed)
        splits = list(kf.split(x_all))
        train_idx, val_idx = splits[self.fold_idx]

        log.info(
            f"[KFold] Fold {self.fold_idx + 1}/{self.n_folds} — "
            f"train: {len(train_idx)}, val: {len(val_idx)}"
        )
        return (
            x_all[train_idx], y_all[train_idx],
            x_all[val_idx],   y_all[val_idx],
        )

    # ------------------------------------------------------------------
    # Lightning setup
    # ------------------------------------------------------------------
    def setup(self, stage: Optional[str] = None) -> None:
        processed_dir = Path(self.processed_data_dir)

        # Reload norm_params + feature_names from disk on every worker
        npz_path = processed_dir / "norm_params.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=False)
            self.feature_names = list(npz["feature_names"])
            self.norm_params   = {
                name: (kind, float(scale))
                for name, kind, scale in zip(npz["feature_names"], npz["kinds"], npz["scales"])
            }

        # Load x_val_raw for InvarianceMetricCallback
        if self.x_val_raw is None:
            try:
                _, x_val, _, _, feat_names, pt_idx = self._load_all_h5_files(
                    self.raw_data_dir, isTest=False
                )
                self.x_val_raw     = x_val
                self.feature_names = feat_names
                self.pt_idx        = pt_idx
                log.info(f"[setup] Loaded x_val_raw: {self.x_val_raw.shape}")
            except Exception as e:
                log.warning(f"[setup] Could not load x_val_raw: {e}")

        # Validate processed files exist
        missing = [
            str(processed_dir / f)
            for f in ("x_train.npy", "y_train.npy", "x_val.npy", "y_val.npy")
            if not (processed_dir / f).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Processed arrays missing — run prepare_data() first. Missing: {missing}"
            )

        feature_indices = self._get_feature_indices(processed_dir)

        # --- Fit stage: build train + val datasets ---
        if stage in (None, "fit") and self.data_train is None:
            x_train_disk = np.load(processed_dir / "x_train.npy", allow_pickle=False)
            y_train_disk = np.load(processed_dir / "y_train.npy", allow_pickle=False)
            x_val_disk   = np.load(processed_dir / "x_val.npy",   allow_pickle=False)
            y_val_disk   = np.load(processed_dir / "y_val.npy",   allow_pickle=False)

            if self.n_folds > 1:
                # Merge processed train+val pool, then re-split for this fold.
                # Normalisation is already baked in — no re-fitting needed.
                x_all = np.concatenate([x_train_disk, x_val_disk], axis=0)
                y_all = np.concatenate([y_train_disk, y_val_disk], axis=0)
                log.info(f"[setup] Full training pool: {len(x_all)} jets")
                x_train, y_train, x_val, y_val = self._kfold_split(x_all, y_all)
            else:
                # Single split — use the disk arrays as-is
                x_train, y_train = x_train_disk, y_train_disk
                x_val,   y_val   = x_val_disk,   y_val_disk

            if feature_indices is not None:
                x_train = x_train[:, :, feature_indices]
                x_val   = x_val[:,   :, feature_indices]

            if self.n_folds > 1:
                log.info(
                    f"==================================================="
                    f"[setup] FOLD {self.fold_idx + 1}/{self.n_folds} — "
                    f"x_train: {x_train.shape}, x_val: {x_val.shape}"
                    f"==================================================="
                )

            self.data_train = TensorDataset(
                torch.from_numpy(x_train).float(),
                torch.from_numpy(y_train).float(),
            )
            self.data_val = TensorDataset(
                torch.from_numpy(x_val).float(),
                torch.from_numpy(y_val).float(),
            )

        if stage in (None, "validate") and self.data_val is None:
            x_val = np.load(processed_dir / "x_val.npy", allow_pickle=False)
            y_val = np.load(processed_dir / "y_val.npy", allow_pickle=False)
            if feature_indices is not None:
                x_val = x_val[:, :, feature_indices]
            self.data_val = TensorDataset(
                torch.from_numpy(x_val).float(),
                torch.from_numpy(y_val).float(),
            )

        if stage in (None, "test", "predict") and self.data_test is None:
            x_test, y_test, _, test_pt_idx = self._load_all_h5_files(
                self.raw_data_dir, isTest=True
            )
            x_test = self._restrict_and_sort_by_pt(x_test, test_pt_idx, self.nconstituents)

            # 2. normalise on FULL feature set (norm_params keys match full feature_names)
            if self.norm_params is None or self.feature_names is None:
                raise RuntimeError(
                    "norm_params or feature_names is None during test setup. "
                    "Ensure norm_params.npz was loaded at the top of setup()."
                )
            x_test = self.apply_physics_norm(x_test, self.feature_names, self.norm_params)

            # 3. THEN select the subset of features the model actually uses
            if feature_indices is not None:
                x_test = x_test[:, :, feature_indices]

            log.info(f"[setup] Test — x_test: {x_test.shape}, y_test: {y_test.shape}")
            self.data_test = TensorDataset(
                torch.from_numpy(x_test).float(),
                torch.from_numpy(y_test).float(),
)
        # Optional overlap diagnostic plot (augmentation only)
        if (
            stage in (None, "fit")
            and not self._overlap_plot_done
            and self.augmentation_scheme != "none"
            and self.feature_names is not None
        ):
            try:
                plot_features_overlap_from_datasets(
                    datasets_dir=self.data_path,
                    feature_names=self.feature_names,
                    class_idx=0,
                    augmentation_scheme=self.augmentation_scheme,
                    method=self.method,
                    nconstituents=self.nconstituents,
                    split="train",
                    save_dir=os.path.join(
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "logs", "plots",
                    ),
                )
            except FileNotFoundError:
                log.warning("Skipping overlap plot — baseline 'none' processed files not found.")
            self._overlap_plot_done = True

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if self.method == "augmentation":
            return DataLoader(
                dataset=self.data_train, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True,
            )
        elif self.method in _BASELINE_METHODS:
            return DataLoader(
                dataset=self.data_train, batch_size=self.batch_size,
                num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True,
            )
        else:
            raise ValueError(f"Unsupported method '{self.method}' for train_dataloader.")

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=False,
        )

    # ------------------------------------------------------------------
    # Lightning boilerplate
    # ------------------------------------------------------------------
    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_module = HLS4MLDataModule(data_dir=str(project_root / "datasets"))
