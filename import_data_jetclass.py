import numpy as np
import torch
from torch.utils.data import TensorDataset
import math

def import_data_without_prepossesing(feature_set: str, valid: bool = False):
    """
    Function to import the data as a tensor dataset
    
    Args:
    feature_set (str): The combination of features to use
    valid (bool): Whether to load validation data
    
    Returns:
    TensorDataset: Dataset containing the selected features and labels
    """
    PATH = "/cluster/scratch/bonvinma/data/"
    
    if valid:
        x_file = PATH + "x_valid_norm.npy"
        x_jet_file = PATH + "jet_valid_norm.npy"
        y_file = PATH + "y_valid_norm.npy"
    else:
        x_file = PATH + "x_train_norm.npy"
        x_jet_file = PATH + "jet_train_norm.npy"
        y_file = PATH + "y_train_norm.npy"

    x = torch.tensor(np.load(x_file))
    x_jet = torch.tensor(np.load(x_jet_file))
    y = torch.tensor(np.load(y_file))
    
    print(f"x shape: {x.shape}")
    print(f"x_jet shape: {x_jet.shape}")
    print(f"y shape: {y.shape}")
    
    # Extract individual features
    px, py, pz, E, pt, eta, phi = x.unbind(dim=-1)
    pt_jet, eta_jet, phi_jet, E_jet = x_jet.unbind(dim=-1)
    
    # Calculate relative coordinates
    eta_rel = eta - eta_jet.unsqueeze(1)
    phi_rel = phi - phi_jet.unsqueeze(1)
    
    # Calculate jet cartesian coordinates
    px_jet = pt_jet * torch.cos(phi_jet)
    py_jet = pt_jet * torch.sin(phi_jet)
    pz_jet = pt_jet * torch.sinh(eta_jet)
    
    # Select features based on the specified combination
    if feature_set == "px_py_pz":
        features = torch.stack([px, py, pz], dim=-1)
    elif feature_set == "px_py_pz_E":
        features = torch.stack([px, py, pz, E], dim=-1)
    elif feature_set == "pt_eta_rel_phi_rel":
        features = torch.stack([pt, eta_rel, phi_rel], dim=-1)
    elif feature_set == "px_py_pz_eta_rel_phi_rel":
        features = torch.stack([px, py, pz, eta_rel, phi_rel], dim=-1)
    elif feature_set == "px_py_pz_jet_center":
        particle_features = torch.stack([px, py, pz], dim=-1)
        jet_features = torch.stack([px_jet, py_jet, pz_jet], dim=-1).unsqueeze(1)
        features = torch.cat([particle_features, jet_features], dim=1)
    elif feature_set == "px_py_pz_E_jet_center":
        particle_features = torch.stack([px, py, pz, E], dim=-1)
        jet_features = torch.stack([px_jet, py_jet, pz_jet, E_jet], dim=-1).unsqueeze(1)
        features = torch.cat([particle_features, jet_features], dim=1)
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    
    print(f"The data has shape: {features.shape}")
    
    return TensorDataset(features, y)

def select_data(name: str, feature_set: str):
    """Function to select the data"""
    sel_data = {
        "deepset": import_data_without_prepossesing,
        "vector_neuron": import_data_without_prepossesing,
        "vn_etaphi_rel": import_data_without_prepossesing,
        "vector_neuron_lorentz": import_data_without_prepossesing,
        #"vn_dgcnn": import_data_without_prepossesing,
        #"dgcnn": import_data_without_prepossesing,
        "transformer": import_data_without_prepossesing,
    }
    
    dataset_func = sel_data.get(name)
    if dataset_func is None:
        raise ValueError(f"The model {name} is not implemented!")
    
    dataset_train = dataset_func(feature_set, valid=False)
    dataset_validate = dataset_func(feature_set, valid=True)
    
    return dataset_train, dataset_validate