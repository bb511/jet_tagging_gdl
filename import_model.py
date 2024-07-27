import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network.deepsets import DeepSetsInvariant
from network.vector_neurons import Vect_Neuron
from network.vector_neurons import Vect_Neuron_etaphi_rel
from network.vector_neurons_lorentz import Vect_Neuron_Lorentz
#from network.vn_dgcnn import VN_DGCNN 
#from network.dgcnn import DGCNN
from network.transfomer import Transformer
#from network.graph_net import MultiLayerEdgeConv

import util
import pickle

def get_model(name : str, nbr_const : int, nbr_feat : int, model_hyperparamter: dict):
    """Function that choose the model to train"""
    parameter = model_hyperparamter[f'{name}']
    print(name)
    sel_model = {
        "deepset": lambda: DeepSetsInvariant(nbr_const, nbr_feat,**parameter),
        "vector_neuron": lambda:Vect_Neuron(nbr_const, nbr_feat,**parameter),
        "vn_etaphi_rel": lambda: Vect_Neuron_etaphi_rel(nbr_const, nbr_feat, **parameter),
        "vector_neuron_lorentz": lambda:Vect_Neuron_Lorentz(nbr_const, nbr_feat,**parameter),
        #"vn_dgcnn": lambda: VN_DGCNN(nbr_const, nbr_feat,**parameter),
        #"dgcnn": lambda: DGCNN(nbr_const, nbr_feat,**parameter),
        "transformer": lambda: Transformer(nbr_const, nbr_feat,**parameter),
        #"graph_net": lambda: MultiLayerEdgeConv(nbr_const, nbr_feat,**parameter),
    }
    model = sel_model.get(name, lambda: None)()
    if model is None:
        raise ValueError(
            f"The model {name} not implemented! Please choose among {list(sel_model.keys())}"
            )
    return model