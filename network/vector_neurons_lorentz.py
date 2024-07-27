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

def normsq4(p):
    """Function to calculate the Minkowski norm of a matrix of four-vectors"""
    dot = dot4(p, p)
    norm = torch.diagonal(dot, dim1=-2, dim2=-1)
    norm = torch.sqrt(norm)
    print("norm", norm.isnan().sum())
    return norm

def dot4(p1, p2):
    """Function to calculate the Minkowski dot product of two matrices of four-vectors"""
    Lambda = torch.diag(torch.tensor([1.0,1.0,1.0,-1.0])).to(torch.device('cuda'))
    Lambda = Lambda.unsqueeze(0)
    
    dot = p1 @ Lambda @ p2.transpose(-1,1) 
    
    return dot
    

class VNLeakyReLULorentz(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLULorentz, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 4, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        #d_norm_sq = normsq4(d).unsqueeze(-1)                                                                                                                                                                         
        dotprod = dot4(x, d)#.unsqueeze(-1)#
        dotprod = torch.diagonal(dotprod, dim1=-2, dim2=-1).unsqueeze(-1)
        
        mask = (dotprod >= 0).float()
        
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-dotprod*d))#/(d_norm_sq+EPS)
        
        return x_out

class VNBatchNormLorentz(nn.Module):
    def __init__(self, num_features, dim):
        super(VNBatchNormLorentz, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 4, N_samples, ...]
        '''
        norm = normsq4(x)
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        
        return x
    
    
class VNGeometricMixingLorentz(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNGeometricMixingLorentz, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
                
        self.Lambda = torch.diag(torch.tensor([1.0,1.0,1.0,-1.0])).to(torch.device('cuda')).unsqueeze(0)

        self.VNLinearandLReLU_1 = VNLinearAndLeakyReLULorentz(self.in_channels, self.out_channels)
        self.VNLinearandLReLU_2 = VNLinearAndLeakyReLULorentz(self.in_channels, self.out_channels)
    
    def forward(self, x):
        x1 = self.VNLinearandLReLU_1(x)
        x2 = self.VNLinearandLReLU_2(x)

        m = self.Lambda @ x1.transpose(1,-1) @ x2
        
        x = x @ m
        return x
    
class VNLinearAndLeakyReLULorentz(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, share_nonlinearity=False, use_batchnorm='none', negative_slope=0.2):
        super(VNLinearAndLeakyReLULorentz, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.negative_slope = negative_slope
        
        self.linear = VNLinear(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLULorentz(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNormLorentz(out_channels, dim=dim)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 4, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # InstanceNorm
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out

class Vect_Neuron_Lorentz(nn.Module):
    def __init__(self, nbr_const, nbr_feat, out_channel, hidden, num_layers):
        super(Vect_Neuron_Lorentz, self).__init__()
        assert nbr_feat == 4, "The number of features must be 4"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        
        self.inp = VNLinearAndLeakyReLULorentz(self.in_channel, self.hidden)#
        self.nn = nn.ModuleList([VNLinearAndLeakyReLULorentz(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        self.before_scalar = VNLinearAndLeakyReLULorentz(self.hidden, self.hidden)
        self.out = nn.Linear(self.hidden**2, self.out_channel)
                
    def forward(self, x):
        x = self.inp(x)
        for module in self.nn:
            x = module(x)
        invariant_x = dot4(x, x)
        x = torch.flatten(invariant_x, start_dim = 1, end_dim = -1)
        out = self.out(x)
        return out



EPS = 1e-6
