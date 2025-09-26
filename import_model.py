import numpy as np

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from network.deepsets import DeepSetsInvariant, DeepSetsInvariantIRC, L_EFN, L_PFN
from network.vector_neurons import Vect_Neuron, Vect_Neuron_breakSO3, SO2VectNeuron, VectNeuron_and_Linear
from network.transfomer import Transformer

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
        "vector_neuron_so2": lambda: SO2VectNeuron(nbr_const, nbr_feat, coef_norm, **parameter),
        "deepset_irc": lambda: DeepSetsInvariantIRC(nbr_const, nbr_feat, coef_norm, **parameter),
        "l_pfn": lambda: L_PFN(nbr_const, nbr_feat, coef_norm, **parameter),
        "l_efn": lambda: L_EFN(nbr_const, nbr_feat, coef_norm, **parameter),
    }
    model = sel_model.get(name, lambda: None)()
    if model is None:
        raise ValueError(
            f"The model {name} not implemented! Please choose among {list(sel_model.keys())}"
            )
    return model