import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network_new.deepsets import DeepSetsInvariant
from network_new.vector_neurons import Vect_Neuron, Vect_Neuron_breakSO3
from network_new.vector_neurons_so2 import SO2VectNeuron, SO2VectNeuron_v2, SO2VectNeuron_v3, SO2VectNeuron_v4, SO2VectNeuron_v5
from network_new.transfomer import Transformer

# from network.deepsets import DeepSetsInvariant
# from network.vector_neurons import Vect_Neuron
# from network.transfomer import Transformer

def get_model(name : str, nbr_const : int, nbr_feat : int, coef_norm : dict, model_hyperparamter: dict):
    """Function that choose the model to train"""
    parameter = model_hyperparamter[f'{name}']
    print(name)
    sel_model = {
        "deepset": lambda: DeepSetsInvariant(nbr_const, nbr_feat, coef_norm, **parameter),
        "vector_neuron": lambda:Vect_Neuron(nbr_const, nbr_feat, coef_norm, **parameter),
        "transformer": lambda: Transformer(nbr_const, nbr_feat, coef_norm, **parameter),
        "vector_neuron_so2": lambda: SO2VectNeuron_v4(nbr_const, nbr_feat, coef_norm, **parameter)
        # "vn_etaphi_rel": lambda: Vect_Neuron_etaphi_rel(nbr_const, nbr_feat, **parameter),
        # "vector_neuron_lorentz": lambda:Vect_Neuron_Lorentz(nbr_const, nbr_feat,**parameter),
        #"vn_dgcnn": lambda: VN_DGCNN(nbr_const, nbr_feat,**parameter),
        #"dgcnn": lambda: DGCNN(nbr_const, nbr_feat,**parameter),
        #"graph_net": lambda: MultiLayerEdgeConv(nbr_const, nbr_feat,**parameter),
    }
    model = sel_model.get(name, lambda: None)()
    if model is None:
        raise ValueError(
            f"The model {name} not implemented! Please choose among {list(sel_model.keys())}"
            )
    return model