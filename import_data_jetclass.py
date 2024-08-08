import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import random

import time
PATH = "/mnt/c/Users/bonvi/Desktop/jetclass_data/"

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


class ChunkSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        return iter(random.sample(range(len(self.data_source)), self.num_samples))

    def __len__(self):
        return self.num_samples
class LargeNumpyDataset(Dataset):
    def __init__(self, x_file, x_jet_file, y_file, feature_set):
        self.x = np.load(x_file, mmap_mode='r')
        self.x_jet = np.load(x_jet_file, mmap_mode='r')
        self.y = np.load(y_file, mmap_mode='r')
        self.feature_set = feature_set

    def __len__(self):
        return len(self.x)
    
    
    #@timing_decorator
    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx].copy())
        x_jet = torch.from_numpy(self.x_jet[idx].copy())
        y = torch.from_numpy(self.y[idx].copy())
        #unique_labels = np.unique(self.y)
        #if np.any(unique_labels < 0) or np.any(unique_labels >=9):
        #    raise ValueError(f"Invalid labels found. Labels should be in range [0, {9-1}]")
        
        features = self.select_features(x, x_jet)
        return  features, y

    def select_features(self, x, x_jet):
        px, py, pz, E, pt, eta, phi = x.unbind(dim=-1)
        pt_jet, eta_jet, phi_jet, E_jet = x_jet.unbind(dim=-1)
        
        eta_rel = eta - eta_jet
        phi_rel = phi - phi_jet
        
        px_jet = pt_jet * torch.cos(phi_jet)
        py_jet = pt_jet * torch.sin(phi_jet)
        pz_jet = pt_jet * torch.sinh(eta_jet)
        
        if self.feature_set == "px_py_pz":
            return torch.stack([px, py, pz], dim=-1)
        elif self.feature_set == "px_py_pz_E":
            return torch.stack([px, py, pz, E], dim=-1)
        elif self.feature_set == "pt_eta_rel_phi_rel":
            return torch.stack([pt, eta_rel, phi_rel], dim=-1)
        elif self.feature_set == "px_py_pz_eta_rel_phi_rel":
            return torch.stack([px, py, pz, eta_rel, phi_rel], dim=-1)
        elif self.feature_set == "px_py_pz_jet_center":
            particle_features = torch.stack([px, py, pz], dim=-1)
            jet_features = torch.stack([px_jet, py_jet, pz_jet], dim=-1)
            return torch.cat([particle_features, jet_features.unsqueeze(0)], dim=0)
        elif self.feature_set == "px_py_pz_E_jet_center":
            particle_features = torch.stack([px, py, pz, E], dim=-1)
            jet_features = torch.stack([px_jet, py_jet, pz_jet, E_jet], dim=-1)
            return torch.cat([particle_features, jet_features.unsqueeze(0)], dim=0)
        else:
            raise ValueError(f"Unsupported feature set: {self.feature_set}")

def import_data(feature_set: str, valid: bool = False):
    if valid:
        x_file = PATH + "x_valid_norm.npy"
        x_jet_file = PATH + "jet_valid_norm.npy"
        y_file = PATH + "y_valid_norm.npy"
    else:
        x_file = PATH + "x_train_norm.npy"
        x_jet_file = PATH + "jet_train_norm.npy"
        y_file = PATH + "y_train_norm.npy"

    dataset = LargeNumpyDataset(x_file, x_jet_file, y_file, feature_set)
    sampler = ChunkSampler(dataset, 10**7)
    print(f"Loaded {len(dataset)} samples")

    return dataset, sampler

def select_data(name: str, feature_set: str):
    """Function to select the data"""
    dataset_train, sampler_train = import_data(feature_set, valid=False)
    dataset_validate, _ = import_data(feature_set, valid=True)
    
    return dataset_train, sampler_train, dataset_validate