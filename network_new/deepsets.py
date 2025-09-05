# DeepSets implementation from DeepSets - Zaheer et. al.
# This is slightly modified from Patrick code

import numpy as np

import torch
import torch.nn as nn



class PermutationEquivariantMean(nn.Module):
    """Permutation equivariant layer using the mean operation.

    Args:
        input_dim: The input dimension of this layer.
        output_dim: The output dimension of this layer.
    """

    def __init__(self, input_dim, output_dim):
        super(PermutationEquivariantMean, self).__init__()
        # Two linear layers, denoted gamma and lambda in the paper.
        self.Gamma = nn.Linear(input_dim, output_dim)
        self.Lambda = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x_mean = x.mean(1, keepdim=True)
        x_mean = self.Lambda(x_mean)
        x = self.Gamma(x)

        x = x - x_mean

        return x


class PermutationEquivariantMax(nn.Module):
    """Permutation equivariant layer using the maximum operation.

    Args:
        input_dim: The input dimension of this layer.
        output_dim: The output dimension of this layer.
    """

    def __init__(self, input_dim, output_dim):
        super(PermutationEquivariantMax, self).__init__()
        # Two linear layers, denoted gamma and lambda in the paper.
        self.Gamma = nn.Linear(input_dim, output_dim)
        self.Lambda = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        x_max = x.max(1, keepdim=True)
        x_max = self.Lambda(x_max)
        x = self.Gamma(x)

        x = x - x_max

        return x


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


class DeepSetsEquivariant(nn.Module):
    """DeepSets permutation equivariant.

    For details on how this network works see Zaheer et. al. 2018 - DeepSets.
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
        input_dim: int,
        phi_layers: list,
        rho_layers: list,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
    ):
        super(DeepSetsEquivariant, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        self.phi_layers.insert(0, self.input_dim)
        for nlayer in range(len(self.phi_layers) - 1):
            layer = PermutationEquivariantMean(
                self.phi_layers[nlayer], self.phi_layers[nlayer + 1]
            )
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)

        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")

        self.rho_layers.insert(0, self.phi_layers[-1])
        self.rho_layers.append(self.output_dim)
        for nlayer in range(len(self.rho_layers) - 1):
            layer = nn.Linear(self.rho_layers[nlayer], self.rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(self.rho_layers) - 2:
                break
            rho.append(activ)

        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {
            "mean": lambda: torch.mean,
            "max": lambda: torch.max,
        }
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation

    def forward(self, x):
        phi_output = self.phi(x)
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
        
        # Store normalization method
        
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            self.norm_method = coef_norm.get('scaler_0', {}).get('type', 'none')
        
        # Register normalization coefficients as buffers
        if self.norm_method != 'none':
            n_features = nbr_feat
            for i in range(n_features):
                feature_params = coef_norm.get(f'scaler_{i}', {})
                
                if self.norm_method == 'minmax':
                    self.register_buffer(f'norm_min_{i}', torch.tensor(feature_params.get('min', 0.0), dtype=torch.float32))
                    self.register_buffer(f'norm_scale_{i}', torch.tensor(feature_params.get('scale', 1.0), dtype=torch.float32))
                    self.register_buffer(f'norm_data_min_{i}', torch.tensor(feature_params.get('data_min', 0.0), dtype=torch.float32))
                    self.register_buffer(f'norm_data_max_{i}', torch.tensor(feature_params.get('data_max', 1.0), dtype=torch.float32))
                
                elif self.norm_method == 'standard':
                    self.register_buffer(f'norm_mean_{i}', torch.tensor(feature_params.get('mean', 0.0), dtype=torch.float32))
                    self.register_buffer(f'norm_scale_{i}', torch.tensor(feature_params.get('scale', 1.0), dtype=torch.float32))
                    self.register_buffer(f'norm_var_{i}', torch.tensor(feature_params.get('var', 1.0), dtype=torch.float32))
                
                elif self.norm_method == 'robust':
                    self.register_buffer(f'norm_center_{i}', torch.tensor(feature_params.get('center', 0.0), dtype=torch.float32))
                    self.register_buffer(f'norm_scale_{i}', torch.tensor(feature_params.get('scale', 1.0), dtype=torch.float32))

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        phi_layers = [self.input_dim] + self.phi_layers[:]
        for nlayer in range(len(phi_layers) - 1):
            layer = nn.Linear(phi_layers[nlayer], phi_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)
        # self.phi_layers.insert(0, self.input_dim)
        # for nlayer in range(len(self.phi_layers) - 1):
        #     layer = nn.Linear(self.phi_layers[nlayer], self.phi_layers[nlayer + 1])
        #     activ = activ_string_to_torch(self.activ)
        #     phi.append(layer)
        #     phi.append(activ)

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
            
        # self.rho_layers.insert(0, self.phi_layers[-1])
        # self.rho_layers.append(self.output_dim)
        # for nlayer in range(len(self.rho_layers) - 1):
        #     layer = nn.Linear(self.rho_layers[nlayer], self.rho_layers[nlayer + 1])
        #     activ = activ_string_to_torch(self.activ)
        #     if dropout > 0 and dropout < 1:
        #         rho.append(nn.Dropout(p=dropout))
        #     rho.append(layer)
        #     if nlayer == len(self.rho_layers) - 2:
        #         break
        #     rho.append(activ)
        
        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {"mean": lambda: torch.mean, "max": lambda: torch.max}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation
    
    def normalize_input(self, x):
        """
        Normalize input data using stored normalization parameters
        x: input tensor of shape [batch_size, n_constituents, n_features]
        """
        if self.norm_method == 'none':
            return x
        
        normalized_x = x.clone()
        
        for i in range(self.input_dim):
            if self.norm_method == 'minmax':
                min_val = getattr(self, f'norm_min_{i}')
                scale = getattr(self, f'norm_scale_{i}')
                normalized_x[:, :, i] = (x[:, :, i] - min_val) * scale
                
            elif self.norm_method == 'standard':
                mean = getattr(self, f'norm_mean_{i}')
                scale = getattr(self, f'norm_scale_{i}')
                normalized_x[:, :, i] = (x[:, :, i] - mean) / scale
                
            elif self.norm_method == 'robust':
                center = getattr(self, f'norm_center_{i}')
                scale = getattr(self, f'norm_scale_{i}')
                normalized_x[:, :, i] = (x[:, :, i] - center) / scale
                
        return normalized_x

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.normalize_input(x)
        #print(x)
        phi_output = self.phi(x)
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
    
class DeepSetsSO3Invariant(nn.Module):
    """DeepSets permutation invariant with SO(3) equivariant layers.

    Deepsets with phi layer being SO(3)
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
        input_dim: int,
        phi_layers: list,
        rho_layers: str,
        activ: str,
        aggreg: str,
        dropout: float,
        output_dim: int,
    ):
        super(DeepSetsInvariant, self).__init__()
        self.activ = activ
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.phi_layers = phi_layers
        self.rho_layers = rho_layers

        self.phi = self._construct_phi()
        self.agg = self._get_aggregation(aggreg)
        self.rho = self._construct_rho(dropout)

    def _construct_phi(self) -> nn.Sequential:
        """Builds the first MLP before the aggregation, called phi in the paper."""
        phi = nn.Sequential()
        self.phi_layers.insert(0, self.input_dim)
        for nlayer in range(len(self.phi_layers) - 1):
            layer = nn.Linear(self.phi_layers[nlayer], self.phi_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            phi.append(layer)
            phi.append(activ)

        return phi

    def _construct_rho(self, dropout: float) -> nn.Sequential:
        """Builds the second MLP after the aggregation, called rho in the paper."""
        rho = nn.Sequential()
        if dropout < 0 or dropout > 1:
            print(f"Given dropout rate {dropout} is invalid! Building model w/o.")

        self.rho_layers.insert(0, self.phi_layers[-1])
        self.rho_layers.append(self.output_dim)
        for nlayer in range(len(self.rho_layers) - 1):
            layer = nn.Linear(self.rho_layers[nlayer], self.rho_layers[nlayer + 1])
            activ = activ_string_to_torch(self.activ)
            if dropout > 0 and dropout < 1:
                rho.append(nn.Dropout(p=dropout))
            rho.append(layer)
            if nlayer == len(self.rho_layers) - 2:
                break
            rho.append(activ)
        return rho

    def _get_aggregation(self, aggreg: str):
        """Gets the desired aggregation function specified through the aggr string."""
        aggregations = {"mean": lambda: torch.mean, "max": lambda: torch.max}
        aggregation = aggregations.get(aggreg, lambda: None)()
        if aggregation is None:
            raise ValueError(
                f"Aggr {aggregation} not implemented! Go to deepsets.py and add it."
            )

        return aggregation

    def forward(self, x):
        
        phi_output = self.phi(x)
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