import numpy as np

import torch 

from torch_geometric.data import Data # to create the graph data

import util

def select_data(name : str, data_name : str):
    """Function to select the data"""
    sel_data = {
        "deepset": import_data_without_prepossesing,
        "vector_neuron": import_data_without_prepossesing,
        "vector_neuron_lorentz": import_data_without_prepossesing,
        "vn_dgcnn": import_data_without_prepossesing,
        "dgcnn": import_data_without_prepossesing,
        "graph_net": import_data_graph_net,
        "msg_pass_edgeconv" : import_data_graph,
    }
    PATH = "/mnt/c/users/bonvi/Documents/deepset/data/"
    sel_path = {
        "pt_eta_rel_phi_rel_8_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/ptetaphi",8),
        "pt_eta_rel_phi_rel_16_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/ptetaphi",16),
        "pt_eta_rel_phi_rel_32_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/ptetaphi",32),
        "px_py_pz_8_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/pxpypz",8),
        "px_py_pz_16_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/pxpypz",16),
        "px_py_pz_32_jetclass": (PATH+"jetclass/pxpypzenergyptetaphi_ptetaphienergy_200/pxpypz",32),
    }
    
    data_path = sel_path.get(data_name, None)
    if data_path is None:
        raise ValueError(
            f"The data {data_name} is not download! Please download it using the train.py script"
            )
    dataset_train = sel_data.get(name, None)(data_path)
    
    if dataset_train is None:
        raise ValueError(
            f"The model {name} not implemented!"
            )
    dataset_validate = sel_data.get(name, None)(data_path, valid = True)
    if dataset_train is None:
        raise ValueError(
            f"The model {name} not implemented!"
            )
    return dataset_train, dataset_validate


def import_data_without_prepossesing(tup:tuple, valid = False):
    """Function to import the data as such in the data set"""
    path = tup[0]
    if valid:    
        x = path + f"concatenated_x_pt_etarel_phirel_32_0.npy"
        y = path + f"y_concatenated_x_pt_etarel_phirel_32_0.npy"
    else:
        x = path + f"concatenated_x_pt_etarel_phirel_32_1.npy"
        y = path + f"y_concatenated_x_pt_etarel_phirel_32_1.npy"
    
    
    x =  torch.Tensor(np.load(x))
    y =  torch.Tensor(np.load(y))
    
    print('The data has shape: ',x.shape)
    
    return torch.utils.data.TensorDataset(x[:,:tup[1], :], y)


def point_clouds_to_fully_connected_graphs(point_clouds):
    """
    Convert point clouds to fully connected graphs.
    
    Args:
    point_clouds (torch.Tensor): Point cloud data with shape (num_graphs, num_points, 3)
    
    Returns:
    list: List of PyG Data objects representing fully connected graphs
    """
    graphs = []
    for point_cloud in point_clouds:
        num_nodes = point_cloud.shape[0]
        
        edge_index = torch.combinations(torch.arange(num_nodes), r=2).t().contiguous()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)# bidirectional edges
        
        graph = Data(x=point_cloud, edge_index=edge_index)
        graphs.append(graph)
    
    return graphs



def import_data_graph(tup : tuple, valid : bool = False):
    """Function to import the graph data"""
    path = tup[0]
    if valid:
        x = path + f"x_val_robust_{tup[1]}const_ptetaphi.npy"
        y = path + f"y_val_robust_{tup[1]}const_ptetaphi.npy"
    else:
        x = path + f"x_train_robust_{tup[1]}const_ptetaphi.npy"
        y = path + f"y_train_robust_{tup[1]}const_ptetaphi.npy"
    
    

    x = torch.Tensor(np.load(x))
    graph_x = point_clouds_to_fully_connected_graphs(x)
    y = torch.Tensor(np.load(y))
    
    data = []
    for i in range(len(graph_x)):
        data.append((graph_x[i],y[i]))
    return data

def import_data_graph_net(tup: tuple, valid:bool = False):
    """Function that import the data for the graph net"""
    path = tup[0]
    if valid:
        x = path + f"x_val_robust_{tup[1]}const_ptetaphi.npy"
        y = path + f"y_val_robust_{tup[1]}const_ptetaphi.npy"  
    else:
        x = path + f"x_train_robust_{tup[1]}const_ptetaphi.npy"
        y = path + f"y_train_robust_{tup[1]}const_ptetaphi.npy"
    
    x = torch.Tensor(np.load(x))
    y = torch.Tensor(np.load(y))
    
    return x,y