# adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

import numpy as np
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.manual_seed_all(15)

class VNMaxPool(nn.Module):
    def __init__(self, in_channels):
        super(VNMaxPool, self).__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max
def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)
class VNStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNLinearAndLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearAndLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0
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
    
class VNLinearPermutationEquivariant(nn.Module):
    def __init__(self, dim_in, dim_out, expressivity=False):
        super(VNLinearPermutationEquivariant, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        if expressivity:
            self.param_lambda = nn.Parameter(torch.randn(dim_in)).to(torch.device('cuda')) 
        else:
            self.param_lambda = nn.Parameter(torch.randn(1)).to(torch.device('cuda'))
        
        self.param_theta = nn.Parameter(torch.randn(1)).to(torch.device('cuda'))
        
        if expressivity:
            self.w = self.define_weights()
        else:
            self.w = torch.full((dim_in, dim_out), self.param_theta.item()) + torch.zeros(dim_in, dim_out).fill_diagonal_(self.param_lambda.item())
            #self.w.requires_grad = True
        self.w = self.w.to(torch.device('cuda'))
        
    def define_weights(self):
        weight = torch.zeros(self.dim_in, self.dim_out).to(torch.device('cuda'))
        lambda_i = self.param_lambda
        for i in range(self.dim_in):
            weight[i] = lambda_i
            lambda_i = torch.roll(lambda_i, 1)# shift the lambda_i to the right
        if self.dim_in < self.dim_out:
            weight[:,self.dim_in:] = self.param_theta.item()
        return weight
    
    def forward(self, x):
        #print(self.w)
        return (x.transpose(1,-1)@ self.w.transpose(1,-1)).transpose(1,-1)
    
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
    
class VNLeakyReLUPermutationEquivariant(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(VNLeakyReLUPermutationEquivariant, self).__init__()
        self.param_lambda = nn.Parameter(torch.randn(1))
        self.param_theta = nn.Parameter(torch.randn(1))
        
        if share_nonlinearity == True:
            self.w = torch.full((in_channels, 1), self.param_theta.item()) + torch.zeros(in_channels, 1).fill_diagonal_(self.param_lambda.item())
        else:
            self.w = torch.full((in_channels, in_channels), self.param_theta.item()) + torch.zeros(in_channels, in_channels).fill_diagonal_(self.param_lambda.item())
        self.w.requires_grad = True
        self.w = self.w.to(torch.device('cuda'))

        self.negative_slope = negative_slope
    def forward(self, x):
        d = (x.transpose(1,-1)@self.w.transpose(1,-1)).transpose(1,-1)
        dotprod = (x*d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d))
        return x_out
    

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
    
class VNLinearPermutationEquivariantAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=4, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(VNLinearPermutationEquivariantAndLeakyReLU, self).__init__()
        self.dim = dim
        self.share_nonlinearity = share_nonlinearity
        self.negative_slope = negative_slope
        
        self.linear = VNLinearPermutationEquivariant(in_channels, out_channels)
        self.leaky_relu = VNLeakyReLUPermutationEquivariant(out_channels, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        self.batch_norm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, dim=dim)
    def forward(self, x):
        x = self.linear(x)
        if self.batch_norm != 'none':
            x = self.batchnorm(x)
        x_out = self.leaky_relu(x)
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
class Vect_Neuron_Big(nn.Module):
    def __init__(self, nbr_const, nbr_feat, out_channel, hidden, num_layers):
        super(Vect_Neuron_Big, self).__init__()
        assert nbr_feat == 3, "The number of features must be 3"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden # set up hidden as 64
        self.num_layers = num_layers
        
        self.inp = VNLinear(self.in_channel, self.hidden)
        self.act = VNLeakyReLU(self.hidden)
        self.batchnorm = VNBatchNorm(self.hidden, dim=3)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden,4) for _ in range(self.num_layers-1)])
        self.linear1 = nn.Linear(4096,256)
        self.linear2 = nn.Linear(256,256)
        self.out = nn.Linear(256, self.out_channel)
                
    def forward(self, x):
        #x = x.to(torch.float32)
        x = x.unsqueeze(dim = -1)
        x = self.inp(x)
        x = self.batchnorm(x)
        x = self.act(x)
        for module in self.nn:
            x = module(x)
        x = x.squeeze(dim = -1)
        inv_x = x@(x.transpose(1,-1))
        x = torch.flatten(inv_x, start_dim = 1, end_dim = -1)
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.out(x)
        return out
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
        self.mixing1 = VNGeometricMixing(self.hidden, self.hidden)
        self.mixing2 = VNGeometricMixing(self.hidden, self.hidden)
                                                                                
        
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, self.hidden, 4)
        if add_mlp:
            self.out1 = nn.Linear(self.hidden**2, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden**2, self.out_channel)
                
    def forward(self, x):
        #x = x.to(torch.float32)
        #x = x.unsqueeze(dim = -1)
        x = self.inp(x)
        #x = self.mixing1(x)
        for module in self.nn:
            #x = self.mixing_channel(x)
            x = module(x)
        #x = x.squeeze(dim = -1)
        x = self.mixing2(x)
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
    
def max_px(x):
    """Find the maximum px in the input tensor x and return the value of px, py, pz of the particle with the maximum px"""
    x_max = torch.zeros(x.shape[0], x.shape[2]).to(torch.device('cuda'))
    for i in range(x.shape[0]):
        max_px = -100000
        index = None
        for j in range(x.shape[1]):
            if x[i,j,0] > max_px:
                max_px = x[i,j,0]
                index = j
        if index == None:
            print("No particle found")
            return None
        else:
            x_max[i] = x[i, index, :]
    return x_max
class Vect_Neuron_DeepSet(nn.Module):
    def __init__(self, nbr_const, nbr_feat, out_channel, hidden, num_layers):
        super(Vect_Neuron_DeepSet, self).__init__()
        assert nbr_feat == 3, "The number of features must be 3" 
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        
        # phi net which is a vector neuron
        self.in_phi = VNLinear(1, self.hidden)#PermutationEquivariant
        self.act_phi = VNLeakyReLU(self.hidden, share_nonlinearity=False)
        self.nn_phi = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden,8) for _ in range(self.num_layers-1)])
        self.out_phi = VNLinearAndLeakyReLU(self.hidden, self.hidden,8)
        self.batchnorm_phi = VNBatchNorm(self.hidden, dim=4)
        
        
        #self.in_phi = nn.Linear(3, self.hidden)
        #self.act_phi = nn.ReLU()
        #self.nn_phi = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        #self.out_phi = nn.Linear(self.hidden, 3)
        
        #self.in_rho = VNLinearPermutationEquivariant(3, self.hidden)
        #self.act_rho = VNLeakyReLUPermutationEquivariant(self.hidden)
        #self.nn_rho = nn.ModuleList([VNLinearPermutationEquivariantAndLeakyReLU(self.hidden, self.hidden,8) for _ in range(self.num_layers-1)])
        #self.out_rho = nn.Linear(256, self.out_channel)
        #self.batchnorm_rho = VNBatchNorm(self.hidden, dim=3)
        
        # rho net which is a mlp
        self.in_rho = nn.Linear(9, self.hidden)
        self.act_rho = nn.ReLU()
        self.nn_rho = nn.ModuleList([nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers-1)])
        self.out_rho = nn.Linear(self.hidden, self.out_channel)
        
    def forward(self, x):
        #phi net
        #x = x.to(torch.float32)
        x = x.unsqueeze(dim = -1)
        x = x.transpose(1,-1)
        x = self.in_phi(x)
        x = self.batchnorm_phi(x)
        x = self.act_phi(x)
        for module in self.nn_phi:
            x = module(x)
        x = self.out_phi(x)
        x = x.transpose(1,-1)
        #x = x.squeeze()
        #print(x.shape)
        x = x@(x.transpose(2,-1))
        #print(x.shape)
        
        #aggregate the output of phi net
        #x = torch.mean(x, dim = 1) # dim = 1 as we want the constituent to be permutation invariant
        x = torch.mean(x, dim = 1)
        #x = max_px(x)
        #rho net
        #x = x.unsqueeze(dim = -1)
        x = x.flatten(start_dim = 1, end_dim = -1)
        x = self.in_rho(x)
        #x = self.batchnorm_rho(x)
        x = self.act_rho(x)
        for module in self.nn_rho:
            x = module(x)
            x = self.act_rho(x)
        #x = x.squeeze(dim = -1)
        #inv_x = x@(x.transpose(1,-1))
        #x = torch.flatten(inv_x, start_dim = 1, end_dim = -1)
        out = self.out_rho(x)
        #out = torch.sum(out, dim = 1)

        return out
    
EPS = 1e-6