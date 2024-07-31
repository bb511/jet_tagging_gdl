# adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

import numpy as np
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.manual_seed_all(15)

import math

def dot4(p1, p2):
    """Function to calculate the Minkowski dot product of two matrices of four-vectors"""
    p1_space, p1_time = p1[..., :3], p1[..., 3]
    p2_space, p2_time = p2[..., :3], p2[..., 3]
    
    space_dot = torch.sum(p1_space * p2_space, dim=-1)
    time_dot = p1_time * p2_time
    
    return space_dot - time_dot
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
        dotprod = dot4(x, d).unsqueeze(-1)
        
        mask = (dotprod >= 0).float()
        
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-dotprod*d))
        
        return x_out
    
    
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
        self.out = nn.Linear(self.hidden, self.out_channel)
                
    def forward(self, x):
        x = self.inp(x)
        for module in self.nn:
            x = module(x)
        invariant_x = dot4(x, x)
        x = torch.flatten(invariant_x, start_dim = 1, end_dim = -1)
        out = self.out(x)
        return out

def normsq4(p):
    """Function to calculate the Minkowski norm of a matrix of four-vectors"""
    norm = dot4(p, p)
    assert norm.all() > 0, "The norm is smaller than zero"

    eps = 1e-8
    norm = torch.clamp(norm, min=eps)
    
    norm = torch.sqrt(norm)
    assert norm.all()
    return norm
    
class VNLinearPositiv(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(VNLinearPositiv, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.weight = nn.Parameter(torch.Tensor(dim_out, dim_in))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        with torch.no_grad():
            self.weight.abs_()

    def forward(self, x):
        positive_weight = self.weight.abs()
        x = F.linear(x.transpose(1, -1), positive_weight).transpose(1, -1)
        return x
    
class PositiveLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PositiveLinear, self).__init__(in_features, out_features, bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        with torch.no_grad():
            self.weight.abs_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.abs(), self.bias)
    
    
class VNLeakyReLULorentzPositiv(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLULorentzPositiv, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = PositiveLinear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = PositiveLinear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 4, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        d_norm_sq = normsq4(d).unsqueeze(-1) 
        dotprod = dot4(x, d).unsqueeze(-1)        
        
        mask = (dotprod >= 0).float()
        
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-dotprod*d)/(d_norm_sq))#
        
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
        x = x / (norm) * norm_bn
        
        return x
    
class VNLinearAndLeakyReLULorentzPositiv(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, share_nonlinearity=False, use_batchnorm='none', negative_slope=0.2):
        super(VNLinearAndLeakyReLULorentzPositiv, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.negative_slope = negative_slope
        
        self.linear = VNLinearPositiv(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLULorentzPositiv(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        # BatchNorm
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNormLorentz(out_channels, dim=dim)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 4, N_samples, ...]
        '''
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        x_out = self.leaky_relu(x)
        return x_out

class Vect_Neuron_LorentzPositiv(nn.Module):
    def __init__(self, nbr_const, nbr_feat, out_channel, hidden, num_layers):
        super(Vect_Neuron_LorentzPositiv, self).__init__()
        assert nbr_feat == 4, "The number of features must be 4"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        
        self.inp = VNLinearAndLeakyReLULorentzPositiv(self.in_channel, self.hidden)#
        self.nn = nn.ModuleList([VNLinearAndLeakyReLULorentzPositiv(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        self.before_scalar = VNLinearAndLeakyReLULorentzPositiv(self.hidden, self.hidden)
        self.out = nn.Linear(self.hidden, self.out_channel)
                
    def forward(self, x):
        x = self.inp(x)
        for module in self.nn:
            x = module(x)
        invariant_x = dot4(x, x)
        x = torch.flatten(invariant_x, start_dim = 1, end_dim = -1)
        out = self.out(x)
        return out

import torch
import torch.nn as nn

class HatLayer(torch.nn.Module):
    def __init__(self, algebra_type='so31'):
        super(HatLayer, self).__init__()
        if algebra_type == 'so31':
            # Generators for SO(3,1)
            E01 = torch.tensor([[0, 1, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
            E02 = torch.tensor([[0, 0, 1, 0],
                                [0, 0, 0, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 0]])
            E03 = torch.tensor([[0, 0, 0, 1],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [1, 0, 0, 0]])
            E12 = torch.tensor([[0, 0, 0, 0],
                                [0, 0, -1, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]])
            E13 = torch.tensor([[0, 0, 0, 0],
                                [0, 0, 0, -1],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0]])
            E23 = torch.tensor([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, -1],
                                [0, 0, 1, 0]])

            E_bases = torch.stack([E01, E02, E03, E12, E13, E23], dim=0)  # [6,4,4]
            self.register_buffer('E_bases', E_bases)
        else:
            raise ValueError('Invalid algebra type for the hat operation')

    def forward(self, v):
        """
        v: a tensor of arbitrary shape with the last dimension of size 6
        """
        return (v[..., None, None]*self.E_bases).sum(dim=-3)

def killingform(x, y, algebra_type='so31'):
    if algebra_type == 'so31':
        # Killing form for SO(3,1) is proportional to tr(xy)
        return torch.einsum('...ij,...ji->...', x, y)
    else:
        raise ValueError('Invalid algebra type for the Killing form')

def vee(x, algebra_type='so31'):
    if algebra_type == 'so31':
        # Extract the 6 independent components from the 4x4 matrix
        v = torch.zeros(*x.shape[:-2], 6, device=x.device)
        v[..., 0] = x[..., 0, 1]
        v[..., 1] = x[..., 0, 2]
        v[..., 2] = x[..., 0, 3]
        v[..., 3] = x[..., 1, 2]
        v[..., 4] = x[..., 1, 3]
        v[..., 5] = x[..., 2, 3]
        return v
    else:
        raise ValueError('Invalid algebra type for the vee operation')

class LNKillingRelu(nn.Module):
    def __init__(self, in_channels, algebra_type='so31', share_nonlinearity=False, leaky_relu=False, negative_slope=0.2):
        super(LNKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        if share_nonlinearity:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.HatLayer = HatLayer(algebra_type)
        self.algebra_type = algebra_type
        self.leaky_relu = leaky_relu
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, F, 6, N]
        '''
        x_out = torch.zeros_like(x)
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        x = x.transpose(2, -1)
        d = d.transpose(2, -1)
        x_hat = self.HatLayer(x)
        d_hat = self.HatLayer(d)
        kf_xd = killingform(x_hat, d_hat, self.algebra_type)
        
        if self.leaky_relu:
            mask = (kf_xd <= 0).float()
            x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(-kf_xd)*d))
        else:
            x_out = torch.where(kf_xd <= 0, x, x - (-kf_xd) * d)
        x_out = x_out.transpose(2, -1)
        return x_out

class LNLieBracket(nn.Module):
    def __init__(self, in_channels, algebra_type='so31', share_nonlinearity=False):
        super(LNLieBracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type
        if share_nonlinearity:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)
        self.HatLayer = HatLayer(algebra_type)

    def forward(self, x):
        '''
        x: point features of shape [B, F, 6, N]
        '''
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)
        d = d.transpose(2, -1)
        d2 = d2.transpose(2,-1)
        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        return x_out

EPS = 1e-6
