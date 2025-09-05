# Handles the data importing and small preprocessing for the interaction network.
# This is a modified version of Patrick Odagiu's code available at https://github.com/Nicole-Nobili/distillinginvariances/blob/main/jetid_experiments/data.py

import os
from pathlib import Path
import wget
import tarfile

import h5py
import numpy as np
from typing import List, Tuple

import data.standardization as standardization
import data.plot_data as plot_data
from data.compute_jet_info import compute_center_xy, compute_center_xyz, compute_center_pt_eta_phi, compute_center_energy, compute_center_xyz_pt_eta_phi
import data.compute_jet_info as cji

class HLS4MLData150(object):
    """Data class for importing and processing the jet data.

    The raw data is available at https://zenodo.org/records/3602260.
    See Moreno et. al. 2019 - JEDI-net: a jet identification algorithm for a full
    description of this data set.

    Args:
        root: The root directory of the data. It should contain a 'raw' and 'processed'
            folder with raw and processed data. Otherwise, these will be generated.
        nconst: The number of constituents the jet data should be sampled down to.
            The raw number of constituents is 150.
        feats: Which feature selection scheme should be applied. 'ptetaphi' for getting
            the transverse momentum, pseudo-rapidity, and azimuthal angle of each
            cosntituents for every jet. Otherwise, 'all' gets all the features of
            each constituents.
        norm: What kind of normalisation to apply to the features of the data.
            Currently implemented: minmax, robust, or standard.
        train: Whether to import the training data (True) or validation data (False)
            of this data set.
    """

    def __init__(self, 
                root: str, # Path to the root directory
                data_name: str, # Name of the data
                norm: str, # Normalization type
                train: bool # Whether to import the training data or validation data
                ):
        # Setting up the paths
        self.root = Path(root)
        self.data_name = data_name
        self.data_path = self.root / "data_hls4ml"
        self.raw_data_path = self.data_path / "raw_data"
        self.processed_data_path = self.data_path / self.data_name
        
        # Setting up data requirements
        self.norm = norm
        self.train = train
        self.type = "train" if self.train else "val"
        
        # Setting up the data URLs
        self.train_url = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        self.test_url = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"

        # Get data specificities
        self.feats, self.nconst = self._parse_data_name()
        self.original_feats = self.feats
        self.data_index, self.pt_index = self._get_data_index()

        self._setup_directories()
        self._get_raw_data()

        self.preproc_output_name = f"{self.data_name}_{self.type}_{self.nconst}const.npy"
        self.proc_output_name = f"{self.data_name}_{self.type}_{self.norm}_{self.nconst}const.npy"
        
        self.min_pt = 2
        self.x_pro = None
        self.y_pro = None
        self._get_processed_data()
        
    def _setup_directories(self):
        """
        Set up the necessary directories for storing data.
        This method creates the following directories:
        - `data_path`: The main data directory.
        - `raw_data_path`: The directory for storing raw data.
        - `processed_data_path`: The directory for storing processed data.
        If the directories already exist, this method does nothing.
        """
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

    def _parse_data_name(self):
        """Parses the data name and returns relevant information.
            tuple: A tuple containing the following information:
                - feats (str): The feature string.
                - nconst (int): The value of nconst.
            The data name should be in the format 'feats_jet_center_nconst'.
        """
        
        print(tcols.WARNING + 'Data name:', self.data_name + '\n' + 'Be sure it is in the format "features_jet_center_nconst" if you want to include jet center or "features_nconst" if you do not want to include jet center' + tcols.ENDC)
        
        parts = self.data_name.split('_')
        nconst = int(parts[-1])
        
        assert nconst > 0 and nconst < 151, f'{tcols.WARNING}Number of constituents should be a positive integer less than 151{tcols.ENDC}'
        
        feats = '_'.join(parts[:-1])
        
        return feats, nconst
    
    def _get_data_index(self):
        """Returns the index of the data to be used based on the feature selection."""
        feature_map = {
            'px': 0, 'py': 1, 'pz': 2, 'E': 3, 'Erel': 4, 'pt': 5, 'ptrel': 6,
            'eta': 7, 'etarel': 8, 'etarot': 9, 'phi': 10, 'phirel': 11, 'phirot': 12,
            'deltaR': 13, 'costheta': 14, 'costhetarel': 15, 'pdgid': 16
        }
        if self.feats == 'allfeats':
            return list(range(17)), 5
        
        elif 'jet_center' in self.feats or 'ptrelinv' in self.feats or 'etarelinv' in self.feats or 'phirelinv' in self.feats:
            feats = ['px', 'py', 'pz', 'E', 'Erel', 'pt', 'ptrel', 'eta', 'etarel', 'phi', 'phirel']
            data_index = [feature_map[feat] for feat in feats]
            return [feature_map[feat] for feat in feats], 5
        else:
            selected_feats = self.feats.split('_')
            if 'pt' not in selected_feats:
                selected_feats.append('pt')
        
            data_index = [feature_map[feat] for feat in selected_feats]
            data_index = sorted(data_index) #making sure the features are in the right order for the h5 files
            pt_index = data_index.index(5)
            
            return data_index, pt_index

    def _get_raw_data(self):
        """Downloads the jet data if it does not already exist on disk."""
        if self._check_raw_data_exists():
            print(f"{tcols.OKGREEN}Raw data already exists. Skipping download.{tcols.ENDC}")
            return

        print(f"{tcols.OKBLUE}Start downloading data{tcols.ENDC}")
        try:
            self._download_data()
        except Exception as e:
            print(f"{tcols.FAIL}Data could not be downloaded. Error: {str(e)}")
            print("Please check your internet connection and that the data URL is reachable.{tcols.ENDC}")
            exit(1)

    def _check_raw_data_exists(self) -> bool:
        """
        Checks if both training and validation data exist in the raw_data_path.
        
        Returns:
            bool: True if both training and validation data exist, False otherwise.
        """
        if not self.raw_data_path.is_dir():
            return False
        
        train_path = self.raw_data_path / "train"
        val_path = self.raw_data_path / "val"
        
        if not (train_path.is_dir() and val_path.is_dir()):
            return False
        
        # Check if there are files in both directories
        train_files = list(train_path.glob('*.h5'))
        val_files = list(val_path.glob('*.h5'))
        
        return len(train_files) > 0 and len(val_files) > 0
    
    def _download_data(self):
        """Downloads and extracts the jet data."""
        url = self.train_url if self.train else self.test_url
        file_name = "hls4ml_LHCjet_150p_train.tar.gz" if self.train else "hls4ml_LHCjet_150p_val.tar.gz"
        
        data_file_path = self.raw_data_path / file_name
        wget.download(url, out=str(data_file_path))
        
        print(f"\nExtracting {file_name}...")
        with tarfile.open(data_file_path, "r:gz") as tar:
            tar.extractall(path=str(self.raw_data_path))
        
        os.remove(data_file_path)
        print(f"{tcols.OKGREEN}Data downloaded and extracted successfully.{tcols.ENDC}")
        
    def _compute_center(self, x_data: np.ndarray):
        """Computes the jet center based on the feature selection."""
        if self.feats == 'px_py_pz_jet_center':
            return compute_center_xyz(x_data)
        if self.feats == 'px_py_jet_center':
            return compute_center_xy(x_data)
        elif self.feats == 'pt_eta_phi_jet_center':
            return compute_center_pt_eta_phi(x_data)
        elif self.feats == 'px_py_pz_E_jet_center':
            return compute_center_energy(x_data)
        else:
            raise ValueError(f"Feature selection {self.feats} not valid for jet center computation Feel free to add your own jet info computation function in the compute_jet_info.py file.")
        
    def _compute_inv_feat(self, x_data: np.ndarray):
        if self.feats == 'ptrelinv_etarelinv_phirelinv':
            return cji.compute_ptrelinv_etarelinv_phirelinv(x_data)
        elif self.feats == 'px_py_pz_etarelinv_phirelinv':
            return cji.compute_xyz_etarelinv_phirelinv(x_data)
        else:
            raise ValueError(f"Feature selection {self.feats} not valid for relative features computation. Feel free to add your own jet info computation function in the compute_jet_info.py file.")
        
    def _preproc_raw_data(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_data, y_data = self._cut_transverse_momentum(x_data, y_data)
        x_data = self._restrict_nb_constituents(x_data)
        
        if "jet_center" in self.data_name:
            x_data = self._compute_center(x_data)
        elif "ptrelinv" in self.data_name or "etarelinv" in self.data_name or "phirelinv" in self.data_name:
            x_data = self._compute_inv_feat(x_data)
        else: # as jet_center already automatically removes it if needed
            x_data = self._remove_pt_if_not_original(x_data)
        
        print(f"Preprocessed data shape: {x_data.shape}")
        
        proc_dir = self.processed_data_path
        np.save(proc_dir / f"x_preproc_{self.preproc_output_name}", x_data)
        np.save(proc_dir / f"y_preproc_{self.preproc_output_name}", y_data)

        return x_data, y_data
    
    def _cut_transverse_momentum(self, x_data: np.ndarray, y_data: np.ndarray = None) -> Tuple[List[np.ndarray], np.ndarray]:
        """Reject constituents that are below a certain transverse momentum.
        If a jet has no constituents with a momentum above the given threshold, then
        the whole jet is removed.

        Args:
            x_data: Array containing the unprocessed data.
            y_data: Array containing the target.

        Returns:
            The processed data list of arrays together with the processed target.
        """
        boolean_mask = x_data[:, :, self.pt_index] > self.min_pt
        structure_memory = boolean_mask.sum(axis=1)
        x_data = np.split(x_data[boolean_mask], np.cumsum(structure_memory)[:-1])
        x_data = [jet_const for jet_const in x_data if jet_const.size > 0]
        print(f"Number of jets after transverse momentum cut: {len(x_data)}")
        print(f"Shape of first jet after cut: {x_data[0].shape}")
        
        if y_data is not None:
            y_data = y_data[structure_memory > 0]
            return x_data, y_data
        return x_data

    def _restrict_nb_constituents(self, x_data: List[np.ndarray]) -> np.ndarray:
        """Restrict the number of constituents for each jet."""
        restricted_data = []
        for jet in x_data:
            if jet.shape[0] >= self.nconst:
                restricted_data.append(jet[:self.nconst])
            else:
                padding = np.zeros((self.nconst - jet.shape[0], jet.shape[1]))
                restricted_data.append(np.vstack((jet, padding)))
        
        return np.array(restricted_data)

    def _remove_pt_if_not_original(self, x_data: np.ndarray) -> np.ndarray:
        """Removes the 'pt' column if it was not in the original feature selection."""
        if 'pt' not in self.original_feats.split('_') and self.original_feats != 'allfeats':
            # Remove the 'pt' column
            return np.delete(x_data, self.pt_index, axis=2)
        return x_data
    
    def _boundary_check(self, x_data: np.ndarray):
        n_features = x_data.shape[2]  # Number of features in the last dimension
        # Initialize lists to store global min and max for each feature
        global_min = []
        global_max = []
        
        selected_feats = self.feats.split('_')
        if "jet_center" in self.feats:
            selected_feats = selected_feats[:-2]  # Remove 'jet_center' features from the list
        # Iterate over each feature
        for feature_idx in range(n_features):
            feature_data = x_data[:, :, feature_idx]  # Extract data for the current feature
            global_min.append(np.min(feature_data))  # Compute global minimum
            global_max.append(np.max(feature_data))  # Compute global maximum
        # Print results
        for i, feature_name in enumerate(selected_feats):
            print(f"Feature '{feature_name}': Min = {global_min[i]}, Max = {global_max[i]}")

    def _get_processed_data(self):
        if not self._check_processed_data_exists():
            if not self._check_preprocessed_data_exists():
                print(f'{tcols.OKBLUE}Importing raw data{tcols.ENDC}')
                self.x_raw, self.y_raw = self._import_raw_data()
                print(f'{tcols.OKBLUE}Checking the global features boundaries for all constituents{tcols.ENDC}')
                self._boundary_check(self.x_raw)
                print(f'{tcols.OKBLUE}Preprocessing raw data{tcols.ENDC}')
                self.x_pre, self.y_pre = self._preproc_raw_data(self.x_raw, self.y_raw)
                del self.x_raw
                del self.y_raw
            self.x_pro, self.y_pro = self._process_data(self.x_pre, self.y_pre)
            print(f'{tcols.OKBLUE}Checking the features boundaries for the selected constituents{tcols.ENDC}')
            self._boundary_check(self.x_pro)
            
            self._plot_data(self.x_pro, self.y_pro)
            del self.x_pre
            del self.y_pre
            
    def _plot_data(self, x_data, y_data):
        """Plots the normalised data."""
        print(f"{tcols.OKBLUE}Plotting data...{tcols.ENDC}")
        plots_folder = self.data_path / f"plots_{self.norm}_{self.nconst}const_{self.feats}"
        plots_folder.mkdir(parents=True, exist_ok=True)

        plot_data.constituent_number(plots_folder, x_data, self.type)
        plot_data.normalised_data(plots_folder, x_data, y_data, self.type, self.feats)
        print(f"{tcols.OKGREEN}Data plotting completed. Plots saved in {plots_folder}{tcols.ENDC}")


    def _check_processed_data_exists(self):
        """Checks if the processed data already exists."""
        proc_folder = self.processed_data_path
        x_proc_file = proc_folder / f"x_{self.proc_output_name}"
        y_proc_file = proc_folder / f"y_{self.proc_output_name}"

        if x_proc_file.is_file() and y_proc_file.is_file():
            self.x_pro = np.load(x_proc_file)
            self.y_pro = np.load(y_proc_file)
            return True

        return False

    def _check_preprocessed_data_exists(self):
        """Checks if the preprocessed data already exists."""
        proc_folder = self.processed_data_path
        x_preproc_file = proc_folder / f"x_preproc_{self.preproc_output_name}"
        y_preproc_file = proc_folder / f"y_preproc_{self.preproc_output_name}"

        if x_preproc_file.is_file() and y_preproc_file.is_file():
            self.x_pre = np.load(x_preproc_file)
            self.y_pre = np.load(y_preproc_file)
            return True

        return False

    def _import_raw_data(self):
        """Imports the raw data from the h5 files. It imports only the needed features."""
        path = self.raw_data_path / "train" if self.train else self.raw_data_path / "val"
        dfiles = list(file for file in path.iterdir() if file.is_file())
        data = h5py.File(dfiles[0])
        x_data = data["jetConstituentList"][:, :, self.data_index]
        y_data = data["jets"][:, -6:-1]

        for file_path in dfiles[1:]:
            print(file_path)
            data = h5py.File(file_path)
            add_x_data = data["jetConstituentList"][:, :, self.data_index]
            add_y_data = data["jets"][:, -6:-1]
            
            x_data = np.concatenate((x_data, add_x_data), axis=0)
            y_data = np.concatenate((y_data, add_y_data), axis=0)
        print(f'{tcols.OKGREEN}Data imported successfully{tcols.ENDC}')
        print(f'{tcols.OKBLUE}x_data shape: {x_data.shape}{tcols.ENDC}')
        print(f'{tcols.OKBLUE}y_data shape: {y_data.shape}{tcols.ENDC}')
        return x_data, y_data

    def _process_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """Processes the data by normalizing it and saving it to disk."""
        print(f'{tcols.OKBLUE}Applying {self.norm} normalization{tcols.ENDC}')
        
        if self.norm == "nonorm":
            proc_folder = self.processed_data_path
            np.save(proc_folder / f"x_{self.proc_output_name}", x_data)
            np.save(proc_folder / f"y_{self.proc_output_name}", y_data)
            return x_data, y_data
        else:
            raise ValueError(f"Normalization {self.norm} not supported anymore in this class. Please use 'nonorm' instead.")
    #     else:
    #         if not self.train:
    #             x_data_train = self._load_preproc_train_data()
    #             norm_params = standardization.fit_standardisation(self.norm, x_data_train)
    #             del x_data_train
    #         else:
    #             norm_params = standardization.fit_standardisation(self.norm, x_data)

    #         x_data = standardization.apply_standardisation(x_data, norm_params)

    #         proc_folder = self.processed_data_path
    #         np.save(proc_folder / f"x_{self.proc_output_name}", x_data)
    #         np.save(proc_folder / f"y_{self.proc_output_name}", y_data)

    #         return x_data, y_data
    #     # --------- not used anymore ------------
        
    # # --------- not used anymore ------------
    # def _load_preproc_train_data(self):
    #     preproc_file_name = f"x_preproc_{self.data_name}_train_{self.nconst}const.npy"
    #     try:
    #         x_data_train = np.load(self.processed_data_path / preproc_file_name)
    #     except OSError as e:
    #         print(f"Process the training data with {self.nconst} const before the test data.")
    #         print("Exiting...")
    #         exit(1)

    #     return x_data_train
    # # --------- not used anymore ------------


class tcols:
    """Pretty terminal colors ooooo."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    
class HLS4MLDataJets(HLS4MLData150):
    def __init__(self, 
                root: str, # Path to the root directory 
                train: bool,
                jet_name:str = "px_py_pz", # Name of the jet features
                data_name: str = "px_py_pz_150", # Name of the data
                norm: str = 'nonorm', # Normalization type
                 # Whether to import the training data or validation data
                ):
        super().__init__(root, data_name, norm, train)
        self.jet_name = jet_name
        
        self.jet_data_path = self.data_path / f"jet_{self.jet_name}"
        self.jet_output_name = f"{self.jet_name}_{self.type}.npy"
        self.jet_data_path.mkdir(parents=True, exist_ok=True)
        
        self.jet = None
        self._get_jet_data()
            
    def _check_jet_data_exists(self):
        """Checks if the jet data already exists."""
        jet_folder = self.jet_data_path
        jet_file = jet_folder / f"jet_{self.jet_output_name}"
        
        if jet_file.is_file():
            print(f"{tcols.OKGREEN}Jet data already exists. Loading from {jet_file}{tcols.ENDC}")
            self.jet = np.load(jet_file)
            return True
        return False
    
    def _get_jet_data(self):
        if not self._check_jet_data_exists():
            if self.jet_name == "px_py_pz":
                jet_data = np.sum(self.x_pro, axis=1)
            else:
                raise ValueError(f"Jet name {self.jet_name} not recognized. Please use 'px_py_pz' or other valid jet feature names.")
            jet_dir = self.jet_data_path
            np.save(jet_dir / f"jet_{self.jet_output_name}", jet_data)
            self.jet = jet_data
            
    
class HLS4MLDataJets_old(HLS4MLData150):
    """
    Handles compute/save/load of jet centers for any raw-x_data.
    """
    def __init__(self, 
                root: str, # Path to the root directory
                jet_name: str, # Name of the jet features
                train: bool # Whether to import the training data or validation data
                ):
        
    # Setting up the paths
        self.root = Path(root)
        self.jet_name = jet_name
        
        self.train = train
        self.type = "train" if self.train else "val"
        self.nconst = 150
        
        self.data_path = self.root / "data_hls4ml"
        self.raw_data_path = self.data_path / "raw_data"
        self.jet_data_path = self.data_path / f"jet_{self.jet_name}"
        self.jet_output_name = f"{self.jet_name}_{self.type}.npy"
        
        # Setting up data requirements
        self.train = train
        self.type = "train" if self.train else "val"
        
        # Setting up the data URLs
        self.train_url = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        self.test_url = "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"

        # Get data specificities
        self.data_index, self.pt_index = self._get_data_index()
        self._setup_directories()
        self._get_raw_data()
        
        self.min_pt = 2
        self.jet = None
        self._get_jet_data()
        
    def _setup_directories(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.jet_data_path.mkdir(parents=True, exist_ok=True)
        
    def _check_jet_data_exists(self):
        """Checks if the jet data already exists."""
        jet_folder = self.jet_data_path
        jet_file = jet_folder / f"jet_{self.jet_output_name}"
        
        if jet_file.is_file():
            print(f"{tcols.OKGREEN}Jet data already exists. Loading from {jet_file}{tcols.ENDC}")
            self.jet = np.load(jet_file)
            return True
        return False
        
    def _get_data_index(self):
        """Returns the index of the data to be used based on the feature selection."""
        feature_map = {
            'px': 0, 'py': 1, 'pz': 2, 'E': 3, 'Erel': 4, 'pt': 5, 'ptrel': 6,
            'eta': 7, 'etarel': 8, 'etarot': 9, 'phi': 10, 'phirel': 11, 'phirot': 12,
            'deltaR': 13, 'costheta': 14, 'costhetarel': 15, 'pdgid': 16
        }
        
        feats = ['px', 'py', 'pz', 'E', 'Erel', 'pt', 'ptrel', 'eta', 'etarel', 'phi', 'phirel']
        data_index = [feature_map[feat] for feat in feats]
        return data_index, 5
       
    def _compute_jet_data(self, x_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_data = self._cut_transverse_momentum(x_data)
        x_data = self._restrict_nb_constituents(x_data)
        jet_data = self.compute(x_data)
        print (f"Jet data shape: {jet_data.shape}")
        jet_dir = self.jet_data_path
        np.save(jet_dir / f"jet_{self.jet_output_name}", jet_data)
        return jet_data
        
    def compute(self, x_raw: np.ndarray) -> np.ndarray:
        if self.jet_name == 'px_py_pz':
            return compute_center_xyz(x_raw, separated=True)
        if self.jet_name == 'pt_eta_phi':
            return compute_center_pt_eta_phi(x_raw, separated=True)
        if self.jet_name == 'px_py_pz_pt_eta_phi':
            return compute_center_xyz_pt_eta_phi(x_raw, separated=True)
        raise ValueError(f"Unknown jet_center feat: {self.jet_name}")

    def _get_jet_data(self):
        if not self._check_jet_data_exists():
            print(f'{tcols.OKBLUE}Importing raw data{tcols.ENDC}')
            self.x_raw, _ = self._import_raw_data()
            print(f'{tcols.OKBLUE}Computing the jet data{tcols.ENDC}')
            self.jet = self._compute_jet_data(self.x_raw)
            del self.x_raw
            
    
    
if __name__ == "__main__":
    data = HLS4MLData150(root=".", data_name="px_py_pz_E_pt_ptrel_jet_center_150", norm="nonorm", train=True)

