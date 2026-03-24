from pathlib import Path
from typing import Any, Optional, Tuple

import os
import tarfile

import h5py
import numpy as np
import pandas as pd
import torch
import wget
from lightning import LightningDataModule
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset, DistributedSampler
from torch import Tensor
from typing import Dict
try:
    from IPython.display import display as ipy_display
except Exception:
    ipy_display = None
from src.utils.utils import plot_features_overlap_from_datasets

CENTERED_FEATURES = {
    "j1_px",
    "j1_py",
    "j1_pz",
    "j1_eta",
    "j1_etarel",
    "j1_etarot",
    "j1_phi",
    "j1_phirel",
    "j1_phirot",
    "j1_costheta",
    "j1_costhetarel",
}

POSITIVE_FEATURES = {
    "j1_pt",
    "j1_ptrel",
    "j1_e",
    "j1_erel",
    "j1_deltaR",
}


class HLS4MLDataModule(LightningDataModule):
    """Data module with a step-by-step raw->processed pipeline.

    Pipeline order:
    1. ``prepare_data`` downloads/extracts train and val archives (if missing).
    2. Raw H5 data is loaded into numpy arrays.
    3. Data is preprocessed (nconst restriction + pt sorting).
    4. Train split is optionally augmented (none/so2/so3/lorentz).
    5. Processed arrays are saved as ``.npy`` files.
    """

    def __init__(
        self,
        data_dir: str = "../datasets/",
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
    ) -> None:
        super().__init__()

        # Runtime/training config (kept for compatibility with existing configs).
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.nconstituents = nconstituents
        self.features = features if features is not None else ["pt", "etarel", "phirel"]

        # Augmentation config.
        self.augmentation_scheme = augmentation_scheme.lower()
        if self.augmentation_scheme not in {"none", "so2", "so3", "lorentz"}:
            raise ValueError(
                f"Unsupported augmentation_scheme='{augmentation_scheme}'. "
                "Expected one of: ['none', 'so2', 'so3', 'lorentz']."
            )
        self._rng = np.random.default_rng(augmentation_seed)
        self.rotation_nbr_rot = rotation_nbr_rot
        self.rotation_shuffle = rotation_shuffle
        self.lorentz_nbr_rot = lorentz_nbr_rot
        self.lorentz_eta_max = lorentz_eta_max

        # Paths.
        self.data_path = data_dir
        self.raw_data_dir = os.path.join(data_dir, "raw")
        nconst_tag = "full" if self.nconstituents is None or self.nconstituents <= 0 else f"nconst_{self.nconstituents}"
        self.processed_data_dir = os.path.join(data_dir, "processed", self.augmentation_scheme, nconst_tag)

        # Dataset URLs.
        self.raw_data_url = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        self.raw_data_url_val = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"

        # Cached raw data/state (filled in prepare_data).
        self.x_train_raw = None
        self.y_train_raw = None
        self.x_val_raw = None
        self.y_val_raw = None
        self.feature_names = None
        self.pt_idx = None
        self.norm_params = None
        self._overlap_plot_done = False

    # ---------------------------------------------------------------------
    # Public pipeline
    # ---------------------------------------------------------------------
    def prepare_data(self) -> None:
        """Run the full data-preparation pipeline once."""
        "1. Download the tar files and extract the h5 files from them (if not already done)."
        self._download_data_from_url(self.raw_data_url, val=False)
        self._download_data_from_url(self.raw_data_url_val, val=True)

        "2. Load raw H5 data into numpy arrays (cached in self attributes). and store them in raw/train and raw/val subdirs."
        self.x_train_raw, self.y_train_raw, self.feature_names, self.pt_idx = self._load_all_h5_files(
            self.raw_data_dir, isVal=False
        )
        self.x_val_raw, self.y_val_raw, _, _ = self._load_all_h5_files(self.raw_data_dir, isVal=True)

        "3. Feed the raw data into the preprocessing pipeline. This includes nconst restriction, pt sorting, augmentation, and physics normalization. The final processed arrays are saved to disk and also cached in self attributes for potential direct use by the model."
        self._pre_proc_raw_data(self.raw_data_dir, self.processed_data_dir, self.nconstituents, self.features)

    # ---------------------------------------------------------------------
    # Step 1: Download/extract
    # ---------------------------------------------------------------------
    def _download_data_from_url(self, url: str, val: bool = False) -> None:
        split_name = "val" if val else "train"
        expected_glob = "jetImage_7_150p_*.h5" if val else "jetImage_0_150p_*.h5"

        os.makedirs(self.raw_data_dir, exist_ok=True)
        split_dir = Path(self.raw_data_dir) / split_name

        # Just check in the expected split subdir first, then fallback to checking the entire raw_data_dir. This way we avoid false positives from other files in the raw_data_dir that might match the glob pattern but aren't actually extracted data.
        split_has_data = split_dir.exists() and any(split_dir.glob(expected_glob))
        raw_has_data = any(Path(self.raw_data_dir).glob(expected_glob))
        if split_has_data or raw_has_data:
            print(f"{split_name.capitalize()} data already present in {self.raw_data_dir}. Skipping download and extraction.")
            return

        # If we get here, it means the expected data files aren't present, so we proceed with download and extraction.
        filename = os.path.join(self.raw_data_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading data from {url} to {filename}...")
            wget.download(url, filename)
            print("\nDownload complete.")
        else:
            print(f"File {filename} already exists. Skipping download.")

        print(f"Extracting {filename} to {self.raw_data_dir}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=self.raw_data_dir)
        print("Extraction complete.")

    # ---------------------------------------------------------------------
    # Step 2: Raw load
    # ---------------------------------------------------------------------
    def _load_all_h5_files(
        self,
        data_dir: str,
        x_name: str = "jetConstituentList",
        y_name: str = "jets",
        isVal: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, list, int]:
        all_x, all_y = [], []
        if isVal:
            h5_files = sorted((Path(data_dir) / "val").glob("jetImage_7_150p_0_10000.h5"))
        else:
            h5_files = sorted((Path(data_dir) / "train").glob("jetImage_0_150p_0_10000.h5"))

        if not h5_files:
            raise FileNotFoundError(f"No H5 files found in {data_dir}")

        feature_names = None
        pt_idx = None

        for h5_file in h5_files:
            with h5py.File(h5_file, "r") as f:
                x = np.asarray(f[x_name])
                y = np.asarray(f[y_name])

                if feature_names is None:
                    particle_features = f["particleFeatureNames"]
                    feature_names = [feat.decode("utf-8") for feat in particle_features if feat != b"j1_pdgid"]
                    pt_idx = feature_names.index("j1_pt")

                all_x.append(x)
                all_y.append(y[:, -6:-1])

        x_data = np.concatenate(all_x, axis=0)
        y_data = np.concatenate(all_y, axis=0)

        split_name = "Val" if isVal else "Train"
        print(f"{split_name} raw shape:", x_data.shape)
        print(f"{split_name} labels shape:", y_data.shape)
        class_counts = y_data.sum(axis=0).astype(int)
        print(f"{split_name} class counts:", class_counts)
        self._show_class_distribution(y_data, class_counts)

        return x_data, y_data, feature_names, pt_idx

    def _show_class_distribution(self, y_data: np.ndarray, class_counts: np.ndarray) -> None:
        class_df = pd.DataFrame(
            {
                "class_idx": np.arange(y_data.shape[1]),
                "count": class_counts,
                "percent": 100 * class_counts / class_counts.sum(),
            }
        )
        if ipy_display is not None:
            ipy_display(class_df)
        else:
            print(class_df.to_string(index=False))

    # ---------------------------------------------------------------------
    # Step 3/4/5: preprocess, augment, save
    # ---------------------------------------------------------------------
    def _pre_proc_raw_data(
        self,
        raw_data_dir: str,
        processed_data_dir: str,
        nconstituents: int,
        features: list,
    ) -> None:
        # Keep lazy fallback for direct method calls.
        if self.x_train_raw is None or self.y_train_raw is None or self.feature_names is None or self.pt_idx is None:
            self.x_train_raw, self.y_train_raw, self.feature_names, self.pt_idx = self._load_all_h5_files(
                raw_data_dir, isVal=False
            )
        if self.x_val_raw is None or self.y_val_raw is None:
            self.x_val_raw, self.y_val_raw, _, _ = self._load_all_h5_files(raw_data_dir, isVal=True)

        x_train_sorted = self._restrict_and_sort_by_pt(self.x_train_raw, self.pt_idx, nconstituents)
        x_val_sorted = self._restrict_and_sort_by_pt(self.x_val_raw, self.pt_idx, nconstituents)

        x_train_aug, y_train_aug = self._setup_augmentation_schemes(x_train_sorted, self.y_train_raw)
        if self.feature_names is None:
            raise ValueError("feature_names is not available; cannot run normalization.")

        norm_params = self.fit_physics_norm(x_train_sorted, self.feature_names)
        x_train_aug = self.apply_physics_norm(x_train_aug, self.feature_names, norm_params)
        x_val_sorted = self.apply_physics_norm(x_val_sorted, self.feature_names, norm_params)
        self.norm_params = norm_params
        print("Applied physics normalization using train split statistics.")

        print("Processed train shape:", x_train_aug.shape)
        print("Processed train labels shape:", y_train_aug.shape)
        print("Processed val shape:", x_val_sorted.shape)
        print("Processed val labels shape:", self.y_val_raw.shape)

        self._save_processed_arrays(
            processed_data_dir=processed_data_dir,
            x_train=x_train_aug,
            y_train=y_train_aug,
            x_val=x_val_sorted,
            y_val=self.y_val_raw,
            norm_params=norm_params,
        )

    def _restrict_and_sort_by_pt(self, x_data: np.ndarray, pt_idx: int, nconstituents: int) -> np.ndarray:
        target_nconstituents = x_data.shape[1] if nconstituents is None or nconstituents <= 0 else nconstituents

        if x_data.shape[1] < target_nconstituents:
            padding = np.zeros(
                (x_data.shape[0], target_nconstituents - x_data.shape[1], x_data.shape[2]),
                dtype=x_data.dtype,
            )
            x_data = np.concatenate((x_data, padding), axis=1)
        else:
            x_data = x_data[:, :target_nconstituents, :]

        pt_values = x_data[:, :, pt_idx]
        sort_indices = np.argsort(pt_values, axis=1)[:, ::-1]
        return np.take_along_axis(x_data, sort_indices[:, :, np.newaxis], axis=1)

    def _save_processed_arrays(
        self,
        processed_data_dir: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        norm_params: Optional[Dict[str, Tuple[str, float]]] = None,
    ) -> None:
        os.makedirs(processed_data_dir, exist_ok=True)
        np.save(os.path.join(processed_data_dir, "x_train.npy"), x_train)
        np.save(os.path.join(processed_data_dir, "y_train.npy"), y_train)
        np.save(os.path.join(processed_data_dir, "x_val.npy"), x_val)
        np.save(os.path.join(processed_data_dir, "y_val.npy"), y_val)
        if norm_params is not None:
            self._save_norm_params(processed_data_dir, norm_params)

    def fit_physics_norm(
        self,
        x_train: np.ndarray,
        feature_names: list,
    ) -> Dict[str, Tuple[str, float]]:
        """Fit normalization parameters on training data only."""
        params: Dict[str, Tuple[str, float]] = {}
        for i, name in enumerate(feature_names):
            vals = x_train[:, :, i]
            vmin = float(vals.min())
            vmax = float(vals.max())

            if name in CENTERED_FEATURES:
                scale = 0.5 * (vmax - vmin)
                params[name] = ("centered", scale)
            elif name in POSITIVE_FEATURES:
                params[name] = ("positive", vmax)
            else:
                raise ValueError(f"Feature {name} not categorized")
        return params

    def apply_physics_norm(
        self,
        x: np.ndarray,
        feature_names: list,
        params: Dict[str, Tuple[str, float]],
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Apply previously fitted normalization."""
        x_norm = x.copy()
        for i, name in enumerate(feature_names):
            kind, scale = params[name]
            if kind in {"centered", "positive"}:
                x_norm[:, :, i] /= (scale + eps)
            else:
                raise ValueError(f"Unknown normalization kind '{kind}' for feature '{name}'")
        return x_norm

    def _save_norm_params(self, processed_data_dir: str, params: Dict[str, Tuple[str, float]]) -> None:
        feature_names = np.array(list(params.keys()), dtype="U64")
        kinds = np.array([params[name][0] for name in feature_names], dtype="U16")
        scales = np.array([params[name][1] for name in feature_names], dtype=np.float32)
        np.savez(os.path.join(processed_data_dir, "norm_params.npz"), feature_names=feature_names, kinds=kinds, scales=scales)
    

    # ---------------------------------------------------------------------
    # Augmentation orchestration
    # ---------------------------------------------------------------------
    def _setup_augmentation_schemes(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.augmentation_scheme == "none":
            return x_train, y_train
        if self.augmentation_scheme == "so2":
            return self._apply_so2_augmentation(x_train, y_train)
        if self.augmentation_scheme == "so3":
            return self._apply_so3_augmentation(x_train, y_train)
        if self.augmentation_scheme == "lorentz":
            return self._apply_lorentz_augmentation(x_train, y_train)
        raise ValueError(f"Unknown augmentation scheme: {self.augmentation_scheme}")

    def _feature_index(self, *candidate_names: str) -> Optional[int]:
        if self.feature_names is None:
            return None

        index_map = {name: idx for idx, name in enumerate(self.feature_names)}
        for name in candidate_names:
            if name in index_map:
                return index_map[name]
            prefixed = f"j1_{name}"
            if prefixed in index_map:
                return index_map[prefixed]
        return None

    # ---------------------------------------------------------------------
    # Rotation augmentation (SO2 / SO3)
    # ---------------------------------------------------------------------
    def create_rot_mat_regular(self, num_rotations: int) -> np.ndarray:
        if num_rotations <= 0:
            return np.empty((0, 3, 3), dtype=np.float32)

        rot_matrices = []
        grid_size = max(1, int(np.ceil(np.cbrt(num_rotations))))
        angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)

        for theta in angles:
            for phi in angles:
                for psi in angles:
                    rz = np.array(
                        [
                            [np.cos(theta), -np.sin(theta), 0.0],
                            [np.sin(theta), np.cos(theta), 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                    ry = np.array(
                        [
                            [np.cos(phi), 0.0, np.sin(phi)],
                            [0.0, 1.0, 0.0],
                            [-np.sin(phi), 0.0, np.cos(phi)],
                        ],
                        dtype=np.float32,
                    )
                    rx = np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, np.cos(psi), -np.sin(psi)],
                            [0.0, np.sin(psi), np.cos(psi)],
                        ],
                        dtype=np.float32,
                    )
                    rot_matrices.append(rz @ ry @ rx)

        return np.asarray(rot_matrices[:num_rotations], dtype=np.float32)

    def create_so2_rot_matrices_regular(self, nbr_rot: int) -> np.ndarray:
        if nbr_rot <= 0:
            return np.empty((0, 3, 3), dtype=np.float32)

        thetas = np.linspace(0.0, 2.0 * np.pi, nbr_rot, endpoint=False)
        rot_mats = np.zeros((nbr_rot, 3, 3), dtype=np.float32)
        rot_mats[:, 0, 0] = np.cos(thetas)
        rot_mats[:, 0, 1] = -np.sin(thetas)
        rot_mats[:, 1, 0] = np.sin(thetas)
        rot_mats[:, 1, 1] = np.cos(thetas)
        rot_mats[:, 2, 2] = 1.0
        return rot_mats

    def _apply_so2_augmentation(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotation_mats = self.create_so2_rot_matrices_regular(self.rotation_nbr_rot)
        return self._augment_with_rotation_mats(x_train, y_train, rotation_mats, space="so2")

    def _apply_so3_augmentation(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rotation_mats = self.create_rot_mat_regular(self.rotation_nbr_rot)
        return self._augment_with_rotation_mats(x_train, y_train, rotation_mats, space="so3")

    def _augment_with_rotation_mats(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        rotation_mats: np.ndarray,
        space: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_all = [x_data]
        y_all = [y_data]

        for rotation_matrix in rotation_mats:
            if space == "so2":
                x_aug = self._apply_so2_single_rotation(x_data, rotation_matrix)
            else:
                x_aug = self._apply_so3_single_rotation(x_data, rotation_matrix)
            x_all.append(x_aug)
            y_all.append(y_data)

        x_out = np.concatenate(x_all, axis=0)
        y_out = np.concatenate(y_all, axis=0)
        if self.rotation_shuffle:
            indices = self._rng.permutation(x_out.shape[0])
            x_out = x_out[indices]
            y_out = y_out[indices]
        return x_out, y_out

    def _apply_so2_single_rotation(self, x_train: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        px_idx = self._feature_index("j1_px", "px")
        py_idx = self._feature_index("j1_py", "py")
        pz_idx = self._feature_index("j1_pz", "pz")
        pt_idx = self._feature_index("j1_pt", "pt")
        phi_idx = self._feature_index("j1_phi", "phi")

        if None in (px_idx, py_idx, pz_idx, pt_idx, phi_idx):
            print("SO(2) augmentation requested but required px/py/pz/pt/phi features were not found. Returning unaugmented train data.")
            return x_train

        x_new = x_train.copy()
        vec = np.stack(
            [
                x_train[:, :, px_idx],
                x_train[:, :, py_idx],
                x_train[:, :, pz_idx],
            ],
            axis=-1,
        )
        vec_rot = np.einsum("ij,ncj->nci", rotation_matrix, vec, optimize=True)

        px_r = vec_rot[:, :, 0]
        py_r = vec_rot[:, :, 1]
        pz_r = vec_rot[:, :, 2]
        pt_r, _, phi_r = self.px_py_pz_to_pt_eta_phi(px_r, py_r, pz_r)

        x_new[:, :, px_idx] = px_r
        x_new[:, :, py_idx] = py_r
        x_new[:, :, pz_idx] = pz_r
        x_new[:, :, pt_idx] = pt_r
        x_new[:, :, phi_idx] = phi_r
        return x_new

    def _apply_so3_single_rotation(self, x_train: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        px_idx = self._feature_index("j1_px", "px")
        py_idx = self._feature_index("j1_py", "py")
        pz_idx = self._feature_index("j1_pz", "pz")
        pt_idx = self._feature_index("j1_pt", "pt")
        eta_idx = self._feature_index("j1_eta", "eta")
        phi_idx = self._feature_index("j1_phi", "phi")
        etarel_idx = self._feature_index("j1_etarel", "etarel")
        phirel_idx = self._feature_index("j1_phirel", "phirel")
        delta_r_idx = self._feature_index("j1_deltaR", "deltaR")
        ptrel_idx = self._feature_index("j1_ptrel", "ptrel")

        if None in (
            px_idx,
            py_idx,
            pz_idx,
            pt_idx,
            eta_idx,
            phi_idx,
            etarel_idx,
            phirel_idx,
            delta_r_idx,
            ptrel_idx,
        ):
            print("SO(3) augmentation requested but required features were not found. Returning unaugmented train data.")
            return x_train

        x_new = x_train.copy()
        pad_mask = x_train[:, :, pt_idx] > 0

        vec = np.stack(
            [
                x_train[:, :, px_idx],
                x_train[:, :, py_idx],
                x_train[:, :, pz_idx],
            ],
            axis=-1,
        )
        vec = vec * pad_mask[:, :, None]
        vec_rot = np.einsum("ij,ncj->nci", rotation_matrix, vec, optimize=True)

        px_r = vec_rot[:, :, 0]
        py_r = vec_rot[:, :, 1]
        pz_r = vec_rot[:, :, 2]

        pt_r, eta_r, phi_r = self.px_py_pz_to_pt_eta_phi(px_r, py_r, pz_r)
        jet_px = (px_r * pad_mask).sum(axis=1, keepdims=True)
        jet_py = (py_r * pad_mask).sum(axis=1, keepdims=True)
        jet_pz = (pz_r * pad_mask).sum(axis=1, keepdims=True)
        jet_pt, jet_eta, jet_phi = self.px_py_pz_to_pt_eta_phi(jet_px, jet_py, jet_pz)

        eta_rel = eta_r - jet_eta
        phi_rel = self.wrap_phi(phi_r - jet_phi)
        delta_r = np.sqrt(eta_rel ** 2 + phi_rel ** 2)
        pt_rel = pt_r / np.where(jet_pt > 0, jet_pt, 1.0)

        x_new[:, :, px_idx] = np.where(pad_mask, px_r, x_train[:, :, px_idx])
        x_new[:, :, py_idx] = np.where(pad_mask, py_r, x_train[:, :, py_idx])
        x_new[:, :, pz_idx] = np.where(pad_mask, pz_r, x_train[:, :, pz_idx])
        x_new[:, :, pt_idx] = np.where(pad_mask, pt_r, x_train[:, :, pt_idx])
        x_new[:, :, eta_idx] = np.where(pad_mask, eta_r, x_train[:, :, eta_idx])
        x_new[:, :, phi_idx] = np.where(pad_mask, phi_r, x_train[:, :, phi_idx])
        x_new[:, :, etarel_idx] = np.where(pad_mask, eta_rel, x_train[:, :, etarel_idx])
        x_new[:, :, phirel_idx] = np.where(pad_mask, phi_rel, x_train[:, :, phirel_idx])
        x_new[:, :, delta_r_idx] = np.where(pad_mask, delta_r, x_train[:, :, delta_r_idx])
        x_new[:, :, ptrel_idx] = np.where(pad_mask, pt_rel, x_train[:, :, ptrel_idx])
        return x_new

    # ---------------------------------------------------------------------
    # Lorentz augmentation
    # ---------------------------------------------------------------------
    def create_lorentz_boosts(self, nbr_rot: int, eta_max: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate random pure Lorentz boosts acting on (E, px, py, pz)."""
        if nbr_rot <= 0:
            return np.empty((0, 4, 4), dtype=np.float32)

        rng = np.random.default_rng(seed) if seed is not None else self._rng
        mats = np.zeros((nbr_rot, 4, 4), dtype=np.float32)

        for i in range(nbr_rot):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)

            eta = rng.uniform(-eta_max, eta_max)
            beta = np.tanh(eta) * direction
            bx, by, bz = beta
            beta2 = bx * bx + by * by + bz * bz

            if beta2 < 1e-12:
                mats[i] = np.eye(4, dtype=np.float32)
                continue

            gamma = 1.0 / np.sqrt(1.0 - beta2)
            gb = (gamma - 1.0) / beta2

            L = np.eye(4, dtype=np.float32)
            L[0, 0] = gamma
            L[0, 1:] = -gamma * beta
            L[1:, 0] = -gamma * beta
            L[1:, 1:] += gb * np.outer(beta, beta)
            mats[i] = L

        return mats

    def _apply_lorentz_augmentation(self, x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.feature_names is None:
            return x_train, y_train

        required = {
            "j1_px",
            "j1_py",
            "j1_pz",
            "j1_e",
            "j1_pt",
            "j1_eta",
            "j1_phi",
            "j1_etarel",
            "j1_phirel",
            "j1_deltaR",
            "j1_ptrel",
        }
        missing = [name for name in required if name not in self.feature_names]
        if missing:
            print(
                f"Lorentz augmentation requested but missing required features: {missing}. "
                "Returning unaugmented train data."
            )
            return x_train, y_train

        lorentz_mats = self.create_lorentz_boosts(
            nbr_rot=self.lorentz_nbr_rot,
            eta_max=self.lorentz_eta_max,
        )
        return self.augment_dataset(x_train, y_train, lorentz_mats, feature_names=self.feature_names)

    def augment_dataset(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        lorentz_mats: np.ndarray,
        feature_names: list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply multiple Lorentz boosts and concatenate with original."""
        x_all = [x_data]
        y_all = [y_data]

        for L in lorentz_mats:
            x_aug = self.apply_boost_preserve_phi(x_data, L, feature_names)
            x_all.append(x_aug)
            y_all.append(y_data)

        return np.concatenate(x_all, axis=0), np.concatenate(y_all, axis=0)

    def apply_boost_preserve_phi(self, x_data: np.ndarray, L: np.ndarray, feature_names: list) -> np.ndarray:
        x_new = x_data.copy()

        def idx(name: str) -> int:
            return feature_names.index(name)

        px_idx = idx("j1_px")
        py_idx = idx("j1_py")
        pz_idx = idx("j1_pz")
        e_idx = idx("j1_e")

        pt_idx = idx("j1_pt")
        eta_idx = idx("j1_eta")
        phi_idx = idx("j1_phi")

        etarel_idx = idx("j1_etarel")
        phirel_idx = idx("j1_phirel")
        delta_r_idx = idx("j1_deltaR")
        ptrel_idx = idx("j1_ptrel")

        pad_mask = x_data[:, :, pt_idx] > 0

        p4 = np.stack(
            [
                x_data[:, :, e_idx],
                x_data[:, :, px_idx],
                x_data[:, :, py_idx],
                x_data[:, :, pz_idx],
            ],
            axis=-1,
        )
        p4 = p4 * pad_mask[:, :, None]

        p4b = np.einsum("ij,ncj->nci", L, p4, optimize=True)

        e_b = p4b[:, :, 0]
        px_b = p4b[:, :, 1]
        py_b = p4b[:, :, 2]
        pz_b = p4b[:, :, 3]

        px = x_data[:, :, px_idx] * pad_mask
        py = x_data[:, :, py_idx] * pad_mask

        jet_px_orig = px.sum(axis=1, keepdims=True)
        jet_py_orig = py.sum(axis=1, keepdims=True)
        jet_phi_orig = np.arctan2(jet_py_orig, jet_px_orig)

        jet_px = (px_b * pad_mask).sum(axis=1, keepdims=True)
        jet_py = (py_b * pad_mask).sum(axis=1, keepdims=True)
        jet_phi_boost = np.arctan2(jet_py, jet_px)

        theta = jet_phi_orig - jet_phi_boost
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        px_r = cos_t * px_b - sin_t * py_b
        py_r = sin_t * px_b + cos_t * py_b
        pz_r = pz_b
        e_r = e_b

        px_r *= pad_mask
        py_r *= pad_mask
        pz_r *= pad_mask
        e_r *= pad_mask

        energy_mask = (e_r > 0) & pad_mask

        pt = self.compute_pt(px_r, py_r)
        phi = self.compute_phi(px_r, py_r)
        eta = self.compute_eta(px_r, py_r, pz_r)

        x_new[:, :, px_idx] = np.where(energy_mask, px_r, x_data[:, :, px_idx])
        x_new[:, :, py_idx] = np.where(energy_mask, py_r, x_data[:, :, py_idx])
        x_new[:, :, pz_idx] = np.where(energy_mask, pz_r, x_data[:, :, pz_idx])
        x_new[:, :, e_idx] = np.where(energy_mask, e_r, x_data[:, :, e_idx])

        x_new[:, :, pt_idx] = np.where(energy_mask, pt, x_data[:, :, pt_idx])
        x_new[:, :, eta_idx] = np.where(energy_mask, eta, x_data[:, :, eta_idx])
        x_new[:, :, phi_idx] = np.where(energy_mask, phi, x_data[:, :, phi_idx])

        jet_px = (px_r * pad_mask).sum(axis=1, keepdims=True)
        jet_py = (py_r * pad_mask).sum(axis=1, keepdims=True)
        jet_pz = (pz_r * pad_mask).sum(axis=1, keepdims=True)

        jet_pt = self.compute_pt(jet_px, jet_py)
        jet_phi = self.compute_phi(jet_px, jet_py)
        jet_eta = self.compute_eta(jet_px, jet_py, jet_pz)

        eta_rel = eta - jet_eta
        phi_rel = self.wrap_phi(phi - jet_phi)
        delta_r = np.sqrt(eta_rel ** 2 + phi_rel ** 2)
        pt_rel = pt / np.where(jet_pt > 0, jet_pt, 1.0)

        x_new[:, :, etarel_idx] = np.where(energy_mask, eta_rel, x_data[:, :, etarel_idx])
        x_new[:, :, phirel_idx] = np.where(energy_mask, phi_rel, x_data[:, :, phirel_idx])
        x_new[:, :, delta_r_idx] = np.where(energy_mask, delta_r, x_data[:, :, delta_r_idx])
        x_new[:, :, ptrel_idx] = np.where(energy_mask, pt_rel, x_data[:, :, ptrel_idx])

        return x_new

    # ---------------------------------------------------------------------
    # Numeric helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_pt(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return np.sqrt(px ** 2 + py ** 2)

    @staticmethod
    def compute_phi(px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return np.arctan2(py, px)

    @staticmethod
    def compute_eta(px: np.ndarray, py: np.ndarray, pz: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        return 0.5 * np.log((p + pz + eps) / (p - pz + eps))

    @staticmethod
    def wrap_phi(dphi: np.ndarray) -> np.ndarray:
        return (dphi + np.pi) % (2 * np.pi) - np.pi

    def px_py_pz_to_pt_eta_phi(
        self,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pt = self.compute_pt(px, py)
        eta = self.compute_eta(px, py, pz)
        phi = self.compute_phi(px, py)
        return pt, eta, phi

    # ---------------------------------------------------------------------
    # Commented Lightning template code (kept for future reference)
    # ---------------------------------------------------------------------
    # @property
    # def num_classes(self) -> int:
    #     """Get the number of classes.
    #
    #     :return: The number of MNIST classes (10).
    #     """
    #     return 10

    def _get_feature_indices(self, processed_dir: Path) -> np.ndarray:
        """Return column indices for the features requested in self.features.

        Loads full feature names from norm_params.npz and matches against
        self.features, trying both bare names (e.g. 'pt') and prefixed names
        (e.g. 'j1_pt').
        """
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
                    f"Requested feature '{feat}' not found in saved feature names: {all_names}"
                )
        return np.array(indices)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
    
        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.
    
        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        processed_dir = Path(self.processed_data_dir)
        x_train_path = processed_dir / "x_train.npy"
        y_train_path = processed_dir / "y_train.npy"
        x_val_path = processed_dir / "x_val.npy"
        y_val_path = processed_dir / "y_val.npy"

        required_paths = [x_train_path, y_train_path, x_val_path, y_val_path]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Processed arrays are missing. Run `prepare_data()` first. "
                f"Missing files: {missing_paths}"
            )

        if stage in (None, "fit") and (not hasattr(self, "data_train") or self.data_train is None):
            x_train = np.load(x_train_path, allow_pickle=False)
            y_train = np.load(y_train_path, allow_pickle=False)
            x_val = np.load(x_val_path, allow_pickle=False)
            y_val = np.load(y_val_path, allow_pickle=False)

            self.data_train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
            self.data_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

        if stage in (None, "validate") and (not hasattr(self, "data_val") or self.data_val is None):
            x_val = np.load(x_val_path, allow_pickle=False)
            y_val = np.load(y_val_path, allow_pickle=False)
            self.data_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

        if stage in (None, "test", "predict") and (not hasattr(self, "data_test") or self.data_test is None):
            if not hasattr(self, "data_val") or self.data_val is None:
                x_val = np.load(x_val_path, allow_pickle=False)
                y_val = np.load(y_val_path, allow_pickle=False)
                self.data_val = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
            # No separate test split yet; re-use validation split for test/predict.
            self.data_test = self.data_val

        # Optional diagnostic plot: only when augmentation is enabled and feature metadata is available.
        should_plot = (
            stage in (None, "fit")
            and not self._overlap_plot_done
            and self.augmentation_scheme != "none"
            and self.feature_names is not None
        )
        if should_plot:
            plot_features_overlap_from_datasets(
                datasets_dir=self.data_path,
                feature_names=self.feature_names,
                class_idx=0,
                augmentation_scheme=self.augmentation_scheme,
                nconstituents=self.nconstituents,
                split="train",
            )
            self._overlap_plot_done = True


    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.
        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.
    
        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.
    
        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.
    
        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.
    
        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    data_module = HLS4MLDataModule(data_dir=str(project_root / "datasets"))
    data_module.prepare_data()
    data_module.setup(stage="fit")
