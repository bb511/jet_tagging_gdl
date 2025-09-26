# DeepSets implementation from DeepSets - Zaheer et. al.
# This is slightly modified from Patrick code

import numpy as np

import torch
import torch.nn as nn
from deel import torchlip

def activ_string_to_torch(activ: str):
    """Converts activation name from string to a torch object."""
    activations = {
        "relu": lambda: nn.ReLU(inplace=True),
        "tanh": lambda: nn.Tanh(),
        "sigmoid": lambda: nn.Sigmoid(),
    }
    activation = activations.get(activ, lambda: None)()
    if activation is None:
        raise ValueError(
            f"Activation {activation} not implemented! Go to deepsets.py and add it."
        )
    return activation


class DeepSetsInvariant(nn.Module):
    """DeepSets permutation invariant.

    For details on how this network works see Zaheer et. al. 2018 - DeepSets.
    The difference between this network and its equivariant counterpart is that it does
    not use any of the permutation equivariant layers in the rho and phi networks.

    Args:
        input_dim: The input dimension of the features.
        phi_layers: The layers of the phi MLP, applied on the input and before the
            aggregation operation.
        rho_layers: The layers of the rho MLP, applied after the aggregation operation.
        activ: The activation function to use between the layers of both phi and rho.
        aggreg: The aggregation operation performed over the 1st dimension (node level)
            of the data. This lies between the two MLPs.
        dropout: The dropout rate for the dropout applied to the layers of rho.
        output_dim: The output dimension, equal to the number of classes in the dataset.
    """

    def __init__(
        self,
        nbr_const: int,
        nbr_feat: int,
        coef_norm: dict, 
        phi_layers: list,
        rho_layers: str,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
        mask: bool,
    ):
        super(DeepSetsInvariant, self).__init__()
        self.activ = activ
        self.input_dim = nbr_feat
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)
        self.mask=mask
        
        # Store normalization method
        
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization not supported anymore inside the model")

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        phi_layers = [self.input_dim] + self.phi_layers[:]
        for nlayer in range(len(phi_layers) - 1):
            layer = nn.Linear(phi_layers[nlayer], phi_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)
        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")
            
        rho_layers = [self.phi_layers[-1]] + self.rho_layers[:] + [self.output_dim]  # copy
        for nlayer in range(len(rho_layers) - 1):
            layer = nn.Linear(rho_layers[nlayer], rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(rho_layers) - 2:
                break
            rho.append(activ)
        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {"mean": lambda: torch.mean, "max": lambda: torch.max, "sum": lambda: torch.sum}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation

    def forward(self, x):
        x = x.to(torch.float32)
        phi_output = self.phi(x)
        if self.mask:
            mask = (x[:, :, 0] > 0).float().unsqueeze(-1)
            phi_output = phi_output * mask
        agg_output = self.agg(phi_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)
        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)
        return output
    
class DeepSetsInvariantIRC(nn.Module):
    def __init__(
        self,
        nbr_const: int,
        nbr_feat: int,
        coef_norm: dict, 
        phi_layers: list,
        rho_layers: str,
        activ: str,
        dropout: float,
        output_dim: int,
    ):
        super(DeepSetsInvariantIRC, self).__init__()
        self.activ = activ
        self.input_dim = nbr_feat
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = torch.sum
        self.rho = self._construct_rho(dropout)
        
        # Store normalization method
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization not supported anymore inside the model")

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        phi_layers = [self.input_dim-1] + self.phi_layers[:]
        for nlayer in range(len(phi_layers) - 1):
            layer = nn.Linear(phi_layers[nlayer], phi_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)

        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")
            
        rho_layers = [self.phi_layers[-1]] + self.rho_layers[:] + [self.output_dim]  # copy
        for nlayer in range(len(rho_layers) - 1):
            layer = nn.Linear(rho_layers[nlayer], rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(rho_layers) - 2:
                break
            rho.append(activ)
        return rho     

    def forward(self, x):
        x = x.to(torch.float32)
        p_t = x[:, :, 0]/torch.sum(x[:, :, 0], dim=1, keepdim=True)
        eta_phi = x[:, :, 1:3]
        phi_output = self.phi(eta_phi)
        irc_output = p_t.unsqueeze(-1) * phi_output
        agg_output = self.agg(irc_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)

        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)
        return output
    
class L_EFN (nn.Module):
    def __init__(
        self,
        nbr_const: int,
        nbr_feat: int,
        coef_norm: dict, 
        phi_layers: list,
        rho_layers: str,
        activ: str,
        dropout: float,
        output_dim: int,
        l: float
    ):
        super(L_EFN, self).__init__()
        self.activ = activ
        self.input_dim = nbr_feat
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = torch.sum
        self.rho = self._construct_rho(dropout)
        self.l = l
        
        # Store normalization method
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization not supported anymore inside the model")

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = torchlip.Sequential()
        phi_layers = [self.input_dim-1] + self.phi_layers[:]
        for nlayer in range(len(phi_layers) - 1):
            layer = torchlip.SpectralLinear(phi_layers[nlayer], phi_layers[nlayer + 1], k_coef_lip=self.l)
            activ = nn.ReLU()
            phi.append(layer)
            phi.append(activ)
        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = torchlip.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")
            
        rho_layers = [self.phi_layers[-1]] + self.rho_layers[:] + [self.output_dim]  # copy
        for nlayer in range(len(rho_layers) - 1):
            layer = torchlip.SpectralLinear(rho_layers[nlayer], rho_layers[nlayer + 1], k_coef_lip=self.l)
            activ = nn.ReLU()
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(rho_layers) - 2:
                break
            rho.append(activ)       
        return rho     

    def forward(self, x):
        x = x.to(torch.float32)
        p_t = x[:, :, 0]/torch.sum(x[:, :, 0], dim=1, keepdim=True)
        eta_phi = x[:, :, 1:3]
        #print(x)
        phi_output = self.phi(eta_phi)
        irc_output = p_t.unsqueeze(-1) * phi_output
        agg_output = self.agg(irc_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)
        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)
        return output
    
class L_PFN (nn.Module):
    def __init__(
        self,
        nbr_const: int,
        nbr_feat: int,
        coef_norm: dict, 
        phi_layers: list,
        rho_layers: str,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
        mask: bool,
        l: float
    ):
        super(L_PFN, self).__init__()
        self.activ = activ
        self.input_dim = nbr_feat
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)
        self.mask=mask
        self.l = l
        
        # Store normalization method
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization not supported anymore inside the model")

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = torchlip.Sequential()
        phi_layers = [self.input_dim] + self.phi_layers[:]
        for nlayer in range(len(phi_layers) - 1):
            layer = torchlip.SpectralLinear(phi_layers[nlayer], phi_layers[nlayer + 1], k_coef_lip=self.l)
            activ = nn.ReLU()
            phi.append(layer)
            phi.append(activ)
        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = torchlip.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")
            
        rho_layers = [self.phi_layers[-1]] + self.rho_layers[:] + [self.output_dim]  # copy
        for nlayer in range(len(rho_layers) - 1):
            layer = torchlip.SpectralLinear(rho_layers[nlayer], rho_layers[nlayer + 1], k_coef_lip=self.l)
            activ = nn.ReLU()
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(rho_layers) - 2:
                break
            rho.append(activ)
        return rho     
    
    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {"mean": lambda: torch.mean, "max": lambda: torch.max, "sum": lambda: torch.sum}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )
        return aggregation

    def forward(self, x):
        x = x.to(torch.float32)
        phi_output = self.phi(x)
        if self.mask:
            mask = (x[:, :, 0] > 0).float().unsqueeze(-1)
            phi_output = phi_output * mask
        #print(x)
        agg_output = self.agg(phi_output, dim=1)
        if isinstance(agg_output, tuple):
            agg_output = agg_output[0]
        rho_output = self.rho(agg_output)

        return rho_output

    @torch.no_grad()
    def predict(self, x) -> np.ndarray:
        self.eval()
        output = self.forward(x)

        return output
    
