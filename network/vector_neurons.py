# adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

import numpy as np
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.manual_seed_all(15)


class VNLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(VNLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.linear = nn.Linear(dim_in, dim_out, bias = False)
    def forward(self, x):
        x = self.linear(x.transpose(1,-1)).transpose(1,-1)
        #print(x.shape, x.nonzero().shape) # check if the output is sparse
        return x
    
    
class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out
    
class VNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        norm = torch.norm(x, dim=2) + 10e-6
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x

    

class VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLU(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out
    

class VNGeometricMixing(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNGeometricMixing, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.VNLinearandLReLU_1 = VNLinearAndLeakyReLU(self.in_channels, self.out_channels,negative_slope =0)
        self.VNLinearandLReLU_2 = VNLinearAndLeakyReLU(self.in_channels, self.out_channels, negative_slope =0)
    
    def forward(self, x):
        x1 = self.VNLinearandLReLU_1(x)
        x2 = self.VNLinearandLReLU_2(x)
        
        m = x1.transpose(1,-1) @ x2
        
        x = x @ m
        return x
class CircularSlidingWindowWeightedMean(nn.Module):
    def __init__(self, window_size=8):
        super(CircularSlidingWindowWeightedMean, self).__init__()
        self.window_size = window_size
        
        # Create decreasing weights
        weights = torch.linspace(1, 0, steps=window_size)
        weights = weights.flip(0)  # Reverse the order to have decreasing weights
        self.register_buffer('weights', weights.unsqueeze(0).unsqueeze(2))

    def forward(self, x):
        padding = self.window_size - 1
        x_padded = torch.cat([x, x[:, :padding, :]], dim=1)
        
       
        unfolded = x_padded.unfold(dimension=1, size=self.window_size, step=1)
        
        # Apply weights to the unfolded tensor
        weighted_unfolded = unfolded * self.weights.reshape(1,1,-1)
        
        # Compute weighted mean
        windowed_weighted_mean = weighted_unfolded.sum(dim=-1) / self.weights.sum()
       
        return windowed_weighted_mean
    
class CircularSlidingWindowMean(nn.Module):
    def __init__(self, window_size=8):
        super(CircularSlidingWindowMean, self).__init__()
        self.window_size = window_size

    def forward(self, x):
        padding = self.window_size - 1
        x_padded = torch.cat([x, x[:, :padding, :]], dim=1)
        
        unfolded = x_padded.unfold(dimension=1, size=self.window_size, step=1)
        windowed_mean = unfolded.mean(dim=-1)
        
        return windowed_mean
    

class Vect_Neuron(nn.Module):
    def __init__(self,
                nbr_const: int, #nbr of constituents 
                nbr_feat: int, #nbr of features 
                out_channel: int, #nbr of classes
                hidden: int, #hidden layer size 
                num_layers: int, #number of layers 
                add_mlp: int = False, #add mlp in the end
                ):
        super(Vect_Neuron, self).__init__()
        assert nbr_feat == 3, "The number of features must be 3"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp
        
        self.inp = VNLinearAndLeakyReLU(self.in_channel, self.hidden,4)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden,4) for _ in range(self.num_layers-1)])
        #self.mixing1 = VNGeometricMixing(self.hidden, self.hidden)
        self.mixing2 = VNGeometricMixing(self.hidden, self.hidden)
        self.sliding_window = CircularSlidingWindowWeightedMean(24)
                                                                                
        
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, self.hidden, 4)
        if add_mlp:
            self.out1 = nn.Linear(self.hidden**2, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden**2, self.out_channel)
                
    def forward(self, x):
        x = self.sliding_window(x)
        #print(f'Number of zero value in the input {x.is_null().sum()}')
        x = self.inp(x)
        #x = self.mixing1(x)
        for module in self.nn:
            x = module(x)
        #x = self.mixing2(x)
        x = self.before_scalar(x)
        inv_x = x@(x.transpose(1,-1))
        x = torch.flatten(inv_x, start_dim = 1, end_dim = -1)
        if self.add_mlp:
            out = self.out1(x)
            out = self.act(out)
            for module in self.mlp_out:
                out = module(out)
                out = self.act(out)
            out = self.out2(out)
        else:
            out = self.out(x)
        return out
    
class Vect_Neuron_etaphi_rel(nn.Module):
    def __init__(self, nbr_const, nbr_feat, out_channel, hidden, num_layers, hidden_mlp, num_layers_mlp):
        super(Vect_Neuron_etaphi_rel, self).__init__()
        assert nbr_feat == 5, "The number of features must be 3"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.hidden_mlp = hidden_mlp
        self.num_layers_mlp = num_layers_mlp
        
        self.inp = VNLinearAndLeakyReLU(self.in_channel, self.hidden,4)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden,4) for _ in range(self.num_layers-1)])
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, 16, 4)
        self.out1 = nn.Linear(16**2 + nbr_const * 2, self.hidden_mlp)
        self.mlp_out = nn.ModuleList([nn.Linear(self.hidden_mlp, self.hidden_mlp) for _ in range(self.num_layers_mlp-1)])
        self.act = nn.ReLU()   
        self.out2 = nn.Linear(self.hidden_mlp, self.out_channel)
                
    def forward(self, x):
        #split the input tensor into two tensors px_py_pz and eta_phi        
        px_py_pz = x[:,:,0:3]
        eta_phi = x[:,:,3:5]
        
        #px_py_pz goes through the vector neuron
        y = px_py_pz.unsqueeze(dim = -1)
        y = self.inp(y)
        for module in self.nn:
            y = module(y)
        y = y.squeeze(dim = -1)
        y = self.before_scalar(y)
        inv_x = y@(y.transpose(1,-1))
        
        #concatenate the two tensors px_py_pz and eta_phi
        inv_x = torch.flatten(inv_x, start_dim = 1, end_dim = -1)
        eta_phi = torch.flatten(eta_phi, start_dim = 1, end_dim = -1)
        x = torch.concat((inv_x, eta_phi), dim = 1)
        
        
        # finial mlp
        out = self.out1(x)
        for module in self.mlp_out:
            out = module(out)
            out = self.act(out)
        out = self.out2(out)
        return out
    
    
EPS = 1e-6