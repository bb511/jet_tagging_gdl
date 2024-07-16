# Handles the data importing and small preprocessing for the interaction network.
# This is a modified version of Patrick Odagiu's code available at https://github.com/Nicole-Nobili/distillinginvariances/blob/main/jetid_experiments/data.py

import os
from pathlib import Path
import wget
import tarfile

import h5py
import numpy as np
import torch

import standardization
import data_plots


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

    def __init__(self, root: str, data_name: str, norm: str, train: bool):
        self.data_from_name = {
            'px' : 0,
            'py' : 1,
            'pz' : 2,
            'E' : 3,
            'Erel' : 4,
            'pt' : 5,
            'ptrel' : 6,
            'eta' : 7,
            'etarel' : 8,
            'etarot' : 9,
            'phi' : 10,
            'phirel' : 11,
            'phirot' : 12,
            'deltaR' : 13,
            'cos(theta)' : 14,
            'cos(thetarel)' : 15,
            'pdgid' : 16,
            'jet-center' : 17
        }
        
        self.root = Path(root)
        self.data_name = data_name
        self.data_path = self.root / "preprocess" / self.data_name
        print(f"The data will be saved in {self.data_path} and the raw data should be in {self.root / 'raw'}")
        self.data_index, self.nconst, self.feats = self._get_data_from_name(data_name)
        self.norm = norm
        if self.data_name == 'allfeats':
            print(f'{tcols.WARNING}Warning: Importing all the feature can be memory intensive. This might lead to a OOM (out of memory error){tcols.ENDC}')

        self.train = train
        self.type = "train" if self.train else "val"
        self.min_pt = 2.0

        self.train_url = (
            "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz"
        )
        self.test_url = (
            "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz"
        )

        self.preproc_output_name = f"{self.data_name}_{self.type}_{self.nconst}const_{self.feats}.npy"
        self.proc_output_name = (
            f"{self.data_name}_{self.type}_{self.norm}_{self.nconst}const_{self.feats}.npy"
        )
        
        self.data_file_dir = self._get_raw_data()
        self.x_pro = None
        self.y_pro = None
        self._get_processed_data()
    
    def _get_data_from_name(self, data_name:str):
        """Get the index of the data from the name of the data."""
        data_names = data_name.split('_')
        if data_names[0] == 'allfeats':
            assert len(data_names) == 2, f'{tcols.WARNING}allfeats should not have any other data name, only number of features{tcols.ENDC}'
            nbr_const = data_names[-1]
            return [i for i in range(17)], int(nbr_const), 17
        else:
            if data_names[-2] == 'jet-center':
                assert data_names[:-2].all() in self.data_from_name.keys(), f'{tcols.WARNING}Data name {data_name} not valid{tcols.ENDC}. Choose from {self.data_from_name.keys()}'
            else:
                assert data_names[:-1].all() in self.data_from_name.keys(), f'{tcols.WARNING}Data name {data_name} not valid{tcols.ENDC}. Choose from {self.data_from_name.keys()}'
            
            num_const = data_names[-1] if data_names[-2] != 'jet-center' else data_names[-1]+1
            
            assert int(num_const)>0, f'{tcols.WARNING}Number of constituents should be a positive integer{tcols.ENDC}'
            assert (int(num_const)<=150 if data_names[-2] != 'jet_center' else int(num_const)<=151), f'{tcols.WARNING}Number of constituents should be less than 150{tcols.ENDC}'
            
            num_feats = len(data_names[:-1]) if data_names[-2] != 'jet-center' else len(data_names[:-2])
            
            data_index = []
            for name in data_names[:-1]:
                try:
                    data_index.append(self.data_from_name[name])
                except KeyError:
                    print(f'{tcols.WARNING}Data name {name} not valid{tcols.ENDC}. Choose from {self.data_from_name.keys()}')
                    exit(1)
            data_index = data_index.sort()
        return data_index, int(num_const), int(num_feats)

    def _get_raw_data(self):
        """Downloads and unzips the raw data if it does not exist."""
        if not self._check_raw_data_exists():
            print(f"{tcols.OKBLUE}Start dowloading data{tcols.ENDC}")
            try:
                self._download_data()
            except:
                print(f"{tcols.FAIL}Data could not be downloaded. Please check your internet connection and that https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz is available{tcols.ENDC}")
                exit(1)
        print(f'{tcols.OKGREEN}Raw data is available{tcols.ENDC}')
        return self.root / "raw" 

    def _check_raw_data_exists(self)->bool:
        """Checks if the data exists in the given root dir or needs to be downloaded."""
        if self.root.is_dir():
            data_dir = self.root / "raw"
            if data_dir.is_dir():
                if any(data_dir.iterdir()):
                    return True

        return False

    def _check_processed_data_exists(self):
        """Checks if the processed data exists or needs to be re-processed."""
        if self.root.is_dir():
            proc_folder = self.root / "processed"
            x_proc_file = proc_folder / f"x_{self.proc_output_name}"
            y_proc_file = proc_folder / f"y_{self.proc_output_name}"

            if x_proc_file.is_file() and y_proc_file.is_file():
                self.x_pro = np.load(x_proc_file)
                self.y_pro = np.load(y_proc_file)
                return 1

        return 0

    def _check_preprocessed_data_exists(self):
        """Checks if the preprocessed data exisits or needs to be re-processed."""
        print('Check preprocess')
        if self.root.is_dir():
            proc_folder = self.root / "processed"
            x_preproc_file = proc_folder / f"x_preproc_{self.preproc_output_name}"
            y_preproc_file = proc_folder / f"y_preproc_{self.preproc_output_name}"

            if x_preproc_file.is_file() and y_preproc_file.is_file():
                self.x_pre = np.load(x_preproc_file)
                self.y_pre = np.load(y_preproc_file)
                return 1

        return 0

    def _download_data(self):
        """Downloads the jet data if it does not already exist on disk."""
        raw_dir = self.root / "raw"
        if not raw_dir.is_dir():
            os.makedirs(raw_dir)

        if self.train:
            data_file_path = wget.download(self.train_url, out=str(raw_dir))
        else:
            data_file_path = wget.download(self.test_url, out=str(raw_dir))

        print("")
        data_tar = tarfile.open(data_file_path, "r:gz")
        data_tar.extractall(str(raw_dir))
        data_tar.close()
        os.remove(data_file_path)
        
    def _compute_center(self, x_data):
        """Compute the jet centers in (jet_px, jet_py, jet_pz) from p_T, p_T_rel
            eta, eta_rel and phi, phi_rel"""
            
        #compute the pT jet center
        if x_data[:,4].all() != 0:
            jet_pT = np.mean(x_data[:,3]/x_data[:,4])
        else:# to take into account zero padded values
            count = 0
            jet_pT = 0
            for i in range(x_data.shape[0]):
                if x_data[i,4] != 0:
                    jet_pT += x_data[i,3]/x_data[i,4]#pT/pT_rel
                    count += 1
            assert count > 0
            jet_pT = jet_pT/count # take the average of the non zero values
        
        #compute the eta jet center
        if x_data[:,6].all() != 0:
            jet_eta = np.mean(x_data[:,5]-x_data[:,6])#eta - eta_rel
        else: # to take into account zero padded values
            count = 0
            jet_eta = 0
            for i in range(x_data.shape[0]):
                if x_data[i,6] != 0:
                    jet_eta += x_data[i,5]-x_data[i,6] #eta - eta_rel
                    count += 1
            assert count > 0
            jet_eta = jet_eta/count
        
        #compute the phi jet center
        if x_data[:,8].all() != 0:
            jet_phi = np.mean(x_data[:,7]-x_data[:,8])#phi - phi_rel
        else:# to take into account zero padded values
            count = 0
            jet_phi = 0
            for i in range(x_data.shape[0]):
                if x_data[i,8] != 0:
                    jet_phi += x_data[i,7]-x_data[i,8]#phi_jet = phi - phi_rel
                    count += 1
            assert count > 0
            jet_phi = jet_phi/count
                    
        # For jet_px, jet_py, jet_pz calculation decomment the following lines
        if self.feats == 'px_py_pz_jet_center':
            jet_px = jet_pT*np.cos(jet_phi)
            jet_py = jet_pT*np.sin(jet_phi)
            jet_pz = jet_pT*np.sinh(jet_eta)

            jet_center = np.array((jet_px.item(), jet_py.item(), jet_pz.item()))
            jet_center.squeeze()
            jet_center = np.expand_dims(jet_center, axis=0)
            jet_center = np.concatenate((x_data[:,0:3], jet_center), axis=0)
            
            return jet_center
        
        # For jet_pT, jet_eta, jet_phi calculation
        elif self.feats == 'pt_eta_phi_jet_center':
            jet_center = np.array((jet_pT.item(), jet_eta.item(), jet_phi.item()))
            jet_center.squeeze()
            jet_center = np.expand_dims(jet_center, axis=0)
            jet_center = np.concatenate((x_data[:,[3,5,7]], jet_center), axis=0)
            return jet_center
        
        else:
            print(f'{tcols.WARNING}Feature selection not valid{tcols.ENDC}. Should be either px_py_pz_jet_center or pt_eta_phi_jet_center')
            exit(1)

    def _compute_center_energy(self, x_data):
        """Compute the jet centers in (jet_px, jet_py, jet_pz jet_E) from p_T, p_T_rel,
        E, E_rel, eta, eta_rel and phi, phi_rel"""
        
        #compute the pT jet center
        if x_data[:,6].all() != 0:
            jet_pT = np.mean(x_data[:,5]/x_data[:,6])
        else:
            count = 0
            jet_pT = 0
            for i in range(x_data.shape[0]):
                if x_data[i,6] != 0:
                    jet_pT += x_data[i,5]/x_data[i,6]#pT/pT_rel
                    count += 1
            assert count > 0
            jet_pT = jet_pT/count # take the average of the non zero values

        #compute the eta jet center
        if x_data[:,8].all() != 0:
            jet_eta = np.mean(x_data[:,7]-x_data[:,8])#eta - eta_rel
        else: # to take into account zero padded values
            count = 0
            jet_eta = 0
            for i in range(x_data.shape[0]):
                if x_data[i,8] != 0:
                    jet_eta += x_data[i,7]-x_data[i,8] #eta - eta_rel
                    count += 1
            assert count > 0
            jet_eta = jet_eta/count

        #compute the phi jet center
        if x_data[:,10].all() != 0:
            jet_phi = np.mean(x_data[:,9]-x_data[:,10])#phi - phi_rel
        else:
            count = 0
            jet_phi = 0
            for i in range(x_data.shape[0]):
                if x_data[i,10] != 0:
                    jet_phi += x_data[i,9]-x_data[i,10]#phi_jet = phi - phi_rel
                    count += 1
            assert count > 0
            jet_phi = jet_phi/count
            
        # compute the energy jet center
        if x_data[:,4].all() != 0:
            jet_E = np.mean(x_data[:,3]/x_data[:,4])# E_jet = E/E_rel
        else:
            count = 0
            jet_E = 0
            for i in range(x_data.shape[0]):
                if x_data[i,4] > 1e-10: # to avoid division by zero and some numerical issues
                    jet_E += x_data[i,3]/x_data[i,4]#E_jet = E/E_rel
                    count += 1
            assert count > 0, 'There are no jets with non zero E_rel'
            jet_E = jet_E/count
            
        
        # For jet_px, jet_py, jet_pz calculation decomment the following lines
        if self.feats == 'px_py_pz_E_jet_center':
            jet_px = jet_pT*np.cos(jet_phi)
            jet_py = jet_pT*np.sin(jet_phi)
            jet_pz = jet_pT*np.sinh(jet_eta)
            jet_E = jet_E

            jet_center = np.array((jet_px.item(), jet_py.item(), jet_pz.item(), jet_E.item()))
            jet_center.squeeze()
            jet_center = np.expand_dims(jet_center, axis=0)
            jet_center = np.concatenate((x_data[:,0:4], jet_center), axis=0)
            return jet_center
        else:
            print(f'{tcols.WARNING}Feature selection not valid{tcols.ENDC}. Should be px_py_pz_E_jet_center')
            exit(1)
    
    def _import_raw_data(self):
        """Imports the raw data files into a numpy array."""
        dfiles = list(file for file in self.data_file_dir.iterdir() if file.is_file())
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

        return x_data, y_data

    def _preproc_raw_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """Sets the minimum transverse momentum of the jet constituents and their nb."""
        x_data, y_data = self._cut_transverse_momentum(x_data, y_data)
        x_data = self._restrict_nb_constituents(x_data)
        # add the jet center to the constituents (comment out if not needed)
        if self.feats in ['px_py_pz_jet_center', 'pt_eta_phi_jet_center', 'px_py_pz_E_jet_center']:
            x_data_jet = np.zeros((x_data.shape[0], x_data.shape[1]+1,4))
            for i in range(x_data.shape[0]):
                if self.feats in ['px_py_pz_jet_center', 'pt_eta_phi_jet_center']:
                    x_data_jet[i] = self._compute_center(x_data[i])
                elif self.feats == 'px_py_pz_E_jet_center':
                    x_data_jet[i] = self._compute_center_energy(x_data[i])
            x_data = x_data_jet
        proc_dir = self.root / "processed"
        if not proc_dir.is_dir():
            os.makedirs(proc_dir)
        np.save(proc_dir / f"x_preproc_{self.preproc_output_name}", x_data)
        np.save(proc_dir / f"y_preproc_{self.preproc_output_name}", y_data)

        return x_data, y_data

    def _get_processed_data(self):
        """Imports the processed data if it exists. If not, generates it."""
        if not self._check_processed_data_exists():
            if not self._check_preprocessed_data_exists():
                print(f'{tcols.OKBLUE}Importing raw data{tcols.ENDC}')
                self.x_raw, self.y_raw = self._import_raw_data()
                print(f'{tcols.OKBLUE}Preprocessing raw data{tcols.ENDC}')
                self.x_pre, self.y_pre = self._preproc_raw_data(self.x_raw, self.y_raw)
                del self.x_raw
                del self.y_raw
            self.x_pro, self.y_pro = self._process_data(self.x_pre, self.y_pre)
            del self.x_pre
            del self.y_pre

    def _load_preproc_train_data(self):
        """Loads preprocessed train data to infer normalisation parameters from."""
        preproc_file_name = f"x_preproc_train_{self.nconst}const.npy"
        try:
            x_data_train = np.load(self.root / "processed" / preproc_file_name)
        except OSError as e:
            print(
                f"Process the training data with {self.nconst} const and"
                + f"{self.min_pt} minimum pt before the test data."
            )
            print("Exiting...")
            exit(1)

        return x_data_train

    def _process_data(self, x_data: np.ndarray, y_data: np.ndarray):
        """Selects features normalizes data that was preprocessed."""
        x_data = self._select_features(x_data)
        if not self.train:
            x_data_train = self._load_preproc_train_data()
            x_data_train = self._select_features(x_data_train)
            norm_params = standardization.fit_standardisation(self.norm, x_data_train)
            del x_data_train
        else:
            norm_params = standardization.fit_standardisation(self.norm, x_data)

        x_data = standardization.apply_standardisation(self.norm, x_data, norm_params)
        print(x_data.shape)
        self._plot_data(x_data, y_data)

        proc_folder = self.root / "processed"
        np.save(proc_folder / f"x_{self.proc_output_name}", x_data)
        np.save(proc_folder / f"y_{self.proc_output_name}", y_data)
        del x_data
        del y_data

        return (
            np.load(proc_folder / f"x_{self.proc_output_name}"),
            np.load(proc_folder / f"y_{self.proc_output_name}"),
        )

    def _plot_data(self, x_data, y_data):
        """Plots the normalised data."""
        print("Plotting data...")
        plots_folder = self.root / f"plots_{self.norm}_{self.nconst}const_{self.feats}"
        if not plots_folder.is_dir():
            os.makedirs(plots_folder)

        data_plots.constituent_number(plots_folder, x_data, self.type)
        data_plots.normalised_data(plots_folder, x_data, y_data, self.type, self.feats)

    


    def _select_features(self, data: np.ndarray):
        """Selects (px, py, pz) features from the numpy jet array."""
        print(data.shape)
        if self.feats in ['px_py_pz', 'px_py_pz_jet_center', 'pt_eta_rel_phi_rel','pt_eta_phi', 'pt_eta_phi_jet_center']:
            return data[:, :, [0,1,2]]
        elif self.feats in ['px_py_pz_eta_rel_phi_rel']:
            return data[:, :, [0,1,2,4,5]]
        elif self.feats in ['px_py_pz_E','px_py_pz_E_jet_center']:
            return data[:, :, [0,1,2,3]]
        elif self.feats == 'allfeats':
            return data[:, :, :] #(px, py, pz, E, Erel, pT, ptrel, eta, etarel, etarot, phi, phirel, phirot, deltaR, cos(theta), cos(thetarel), pdgid)
        else:
            print(f'{tcols.WARNING}Feature selection not valid{tcols.ENDC}. Should be in {self._POSSIBLE_DATA}')
            print('Exiting...')
            exit(1)

    def _cut_transverse_momentum(self, x_data: np.ndarray, y_data: np.ndarray):
        """Reject constituents that are below a certain transverse momentum.

        If a jet has no constituents with a momentum above the given threshold, then
        the whole jet is removed.

        Args:
            x_data: Array containing the unprocessed data.
            y_data: Array containing the target.
            min_pt: The minimum transverse momentum an event can have (GeV).

        Returns:
            The processed data array together with the processed target.
        """
        if self.feats in ['px_py_pz', 'px_py_pz_jet_center', 'px_py_pz_eta_rel_phi_rel']:
            pT_index = 3
        elif self.feats in ['pt_eta_phi', 'pt_eta_rel_phi_rel', 'pt_eta_phi_jet_center']:
            pT_index = 0
        elif self.feats == 'allfeats':
            pT_index = 5
        elif self.feats in ['px_py_pz_E_jet_center', 'px_py_pz_E']:
            pT_index = 4
        else:
            print(f'{tcols.WARNING}Feature selection not valid{tcols.ENDC}. Should be in {self._POSSIBLE_DATA}')
            exit(1)
        boolean_mask = x_data[:, :, pT_index] > self.min_pt
        structure_memory = boolean_mask.sum(axis=1)
        x_data = np.split(x_data[boolean_mask, :], np.cumsum(structure_memory)[:-1])
        x_data = [jet_const for jet_const in x_data if jet_const.size > 0]
        y_data = y_data[structure_memory > 0]

        return x_data, y_data

    def _restrict_nb_constituents(self, x_data: np.ndarray):
        """Force each jet to have an equal number of constituents.

        If the jet has more, then the ones after the given number are discarded. If the
        jet has less than the number of max constituents, then it is padded with 0
        values.

        Args:
            x_data: Data array to be processed.

        Returns:
            The data array with a fixed number of constituents per jet.
        """
        for jet in range(len(x_data)):
            if x_data[jet].shape[0] >= self.nconst:
                x_data[jet] = x_data[jet][: self.nconst, :]
            else:
                padding_length = self.nconst - x_data[jet].shape[0]
                x_data[jet] = np.pad(x_data[jet], ((0, padding_length), (0, 0)))

        return np.array(x_data)

    def get_torch_dataset(self):
        """Convert numpy array data to torch tensor and make a tensor data set."""
        x = torch.Tensor(self.x_pro)
        y = torch.Tensor(self.y_pro)
        return torch.utils.data.TensorDataset(x, y)


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