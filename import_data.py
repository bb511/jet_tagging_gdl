import numpy as np

import torch 

#from torch_geometric.data import Data # to create the graph data

import util
import data.get_data_hls4ml as get_data_hls4ml

import torch
from typing import Tuple

def select_data(name: str, data_name: str) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    """
    Selects and imports the training and validation data based on the given parameters.
    
    Args:
    name (str): Name of the model (not used in data selection, but kept for consistency with the main script).
    data_name (str): Name of the data to be imported, including feature selection and number of constituents.
    
    Returns:
    Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]: Training and validation datasets.
    """
    print(f"Importing data: {data_name}")
    
    # Initialize the data class for training data
    train_data = get_data_hls4ml.HLS4MLData150(root=".", data_name=data_name, norm="robust", train=True)
    
    # Initialize the data class for validation data
    val_data = get_data_hls4ml.HLS4MLData150(root=".", data_name=data_name, norm="robust", train=False)
    
    x_train = torch.tensor(train_data.x_pro, dtype=torch.float32)
    y_train = torch.tensor(train_data.y_pro, dtype=torch.float32)
    
    x_val = torch.tensor(val_data.x_pro, dtype=torch.float32)
    y_val = torch.tensor(val_data.y_pro, dtype=torch.float32)
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    return train_dataset, val_dataset
