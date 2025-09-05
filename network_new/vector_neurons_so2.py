import torch
import torch.nn as nn
import torch.nn.functional as F
from .vector_neurons import Vect_Neuron, VNLinear, VNLeakyReLU, VNBatchNorm, VNLinearAndLeakyReLU

EPS = 1e-6

class SO2VNLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super(SO2VNLinear, self).__init__()
        self.linear_xy = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_z = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        x_xy = x[:, :, :2]
        x_z = x[:, :, 2:]
        x_xy = self.linear_xy(x_xy.transpose(1, -1)).transpose(1, -1)
        x_z = self.linear_z(x_z.transpose(1, -1)).transpose(1, -1)
        return torch.cat([x_xy, x_z], dim=2)
    
class SO2VNScalarModulatedLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.linear_xy = nn.Linear(dim_in, dim_out, bias=False)
        self.linear_z  = nn.Linear(dim_in, dim_out, bias=False)
        self.z_gate    = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.GELU()
        )

    def forward(self, x):
        """
        x: [B, C, 3]  (xy + z)
        """
        x_xy = x[:, :, :2]       # [B, C, 2]
        x_z  = x[:, :, 2:]       # [B, C, 1]

        # Linear on XY
        x_xy_out = self.linear_xy(x_xy.transpose(1, -1)).transpose(1, -1)  # [B, C, out]

        # z-conditioned gate: one per feature
        gate = self.z_gate(x_z.transpose(1, -1)).transpose(1, -1)  # [B, C, out]
        x_xy_out = x_xy_out * gate  # [B, C, out]

        # Linear on z as usual
        x_z_out = self.linear_z(x_z.transpose(1, -1)).transpose(1, -1)  # [B, C, out]
        return torch.cat([x_xy_out, x_z_out], dim=2)  # [B, C, out + out]

class SO2VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super(SO2VNLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        if share_nonlinearity:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        x_xy = x[:, :, :2]
        x_z = x[:, :, 2:]
        d = self.map_to_dir(x_xy.transpose(1, -1)).transpose(1, -1)
        dotprod = (x_xy * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True) + EPS
        x_xy_out = self.negative_slope * x_xy + (1 - self.negative_slope) * (mask * x_xy + (1 - mask) * (x_xy - (dotprod / d_norm_sq) * d))
        x_z_out = F.leaky_relu(x_z, negative_slope=self.negative_slope)
        return torch.cat([x_xy_out, x_z_out], dim=2)

class SO2VNBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(SO2VNBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)
        # self.bn_z = nn.BatchNorm1d(num_features)

    def forward(self, x):
        x_xy = x[:, :, :2]
        x_z = x[:, :, 2:]
        norm_xy = torch.norm(x_xy, dim=2) + EPS
        norm_bn = self.bn(norm_xy).unsqueeze(2)
        norm_xy = norm_xy.unsqueeze(2)  # [B, C, 1]
        x_xy = x_xy / norm_xy * norm_bn
        
        # abs_z = torch.abs(x_z) 
        # abs_z_bn = self.bn_z(abs_z)                  # [B, C]
        # x_z = torch.sign(x_z) * abs_z_bn                # Rescale with BN-ed magnitude
        x_z = x_z / norm_xy * norm_bn
        
        return torch.cat([x_xy, x_z], dim=2)

class SO2VNLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, use_batchnorm='norm', negative_slope=0.2):
        super(SO2VNLinearAndLeakyReLU, self).__init__()
        self.linear = SO2VNLinear(in_channels, out_channels)
        self.leaky_relu = SO2VNLeakyReLU(out_channels, share_nonlinearity, negative_slope)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = SO2VNBatchNorm(out_channels, 3)

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        return self.leaky_relu(x)
    
class SO2VNModulatedLinearAndLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, use_batchnorm='none', negative_slope=0.2):
        super(SO2VNModulatedLinearAndLeakyReLU, self).__init__()
        self.linear = SO2VNScalarModulatedLinear(in_channels, out_channels)
        self.leaky_relu = SO2VNLeakyReLU(out_channels, share_nonlinearity, negative_slope)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm != 'none':
            self.batchnorm = VNBatchNorm(out_channels, 3)

    def forward(self, x):
        x = self.linear(x)
        if self.use_batchnorm != 'none':
            x = self.batchnorm(x)
        return self.leaky_relu(x)

class SO2VectNeuron(nn.Module):
    def __init__(self, nbr_const, nbr_feat, coef_norm, out_channel, hidden, num_layers, add_mlp=False):
        super(SO2VectNeuron, self).__init__()
        assert nbr_feat == 3, "Input must be 3D vector features (px, py, pz)"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp

        self.inp = SO2VNLinearAndLeakyReLU(self.in_channel, self.hidden, 2)
        self.nn = nn.ModuleList([SO2VNLinearAndLeakyReLU(self.hidden, self.hidden, 2) for _ in range(self.num_layers - 1)])
        self.before_scalar = SO2VNLinearAndLeakyReLU(self.hidden, self.hidden, 2)

        if add_mlp:
            self.out1 = nn.Linear(self.hidden ** 2 + self.hidden, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden ** 2 + self.hidden, self.out_channel)

    def forward(self, x):
        x = self.inp(x)
        for layer in self.nn:
            x = layer(x)
        x = self.before_scalar(x)
        x_xy = x[:, :, :2]
        x_z = x[:, :, 2:]
        inv_x = x_xy @ x_xy.transpose(1, -1) 
        inv_x = torch.flatten(inv_x, start_dim=1, end_dim=-1)
        x_z = torch.flatten(x_z, start_dim = 1, end_dim = -1)
        x = torch.concat((inv_x, x_z), dim = 1)
        if self.add_mlp:
            x = self.act(self.out1(x))
            for layer in self.mlp_out:
                x = self.act(layer(x))
            return self.out2(x)
        else:
            return self.out(x)


class SO2VectNeuron_v2(nn.Module):
    def __init__(self, nbr_const, nbr_feat, coef_norm, out_channel, hidden, num_layers, add_mlp=False):
        super(SO2VectNeuron_v2, self).__init__()
        assert nbr_feat == 3, "Input must be 3D vector features (px, py, pz)"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp

        self.inp = VNLinearAndLeakyReLU(self.in_channel, self.hidden, 2)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden) for _ in range(self.num_layers - 1)])
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, self.hidden)

        if add_mlp:
            self.out1 = nn.Linear(self.hidden ** 2 + nbr_const, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden ** 2 + nbr_const, self.out_channel)

    def forward(self, x):
        px_py = x[:,:,0:2]
        pz = x[:,:,2]
        
        #px_py_pz goes through the vector neuron
        y = px_py.unsqueeze(dim = -1)
        y = self.inp(y)
        for module in self.nn:
            y = module(y)
        y = y.squeeze(dim = -1)
        y = self.before_scalar(y)
        inv_x = y@(y.transpose(1,-1))
        
        # x = self.inp(x)
        # for layer in self.nn:
        #     x = layer(x)
        # x = self.before_scalar(x)
        # x_xy = x[:, :, :2]
        # x_z = x[:, :, 2:]
        # inv_x = x_xy @ x_xy.transpose(1, -1) 
        inv_x = torch.flatten(inv_x, start_dim=1, end_dim=-1)
        pz = torch.flatten(pz, start_dim = 1, end_dim = -1)
        x = torch.concat((inv_x, pz), dim = 1)
        if self.add_mlp:
            x = self.act(self.out1(x))
            for layer in self.mlp_out:
                x = self.act(layer(x))
            return self.out2(x)
        else:
            return self.out(x)
        
        
class SO2VectNeuron_v3(nn.Module):
    def __init__(self, nbr_const, nbr_feat, coef_norm, out_channel, hidden, num_layers, add_mlp=False):
        super(SO2VectNeuron_v3, self).__init__()
        assert nbr_feat == 3, "Input must be 3D vector features (px, py, pz)"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp

        self.inp = SO2VNModulatedLinearAndLeakyReLU(self.in_channel, self.hidden)
        self.nn = nn.ModuleList([
            SO2VNModulatedLinearAndLeakyReLU(self.hidden, self.hidden)
            for _ in range(self.num_layers - 1)
        ])
        self.before_scalar = SO2VNModulatedLinearAndLeakyReLU(self.hidden, self.hidden)

        # Gating layer for z modulation
       # self.z_gating = nn.Linear(self.hidden, self.hidden**2)

        if add_mlp:
            self.out1 = nn.Linear(self.hidden ** 2+self.hidden, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden ** 2 +self.hidden, self.out_channel)

    def forward(self, x):
        x = self.inp(x)
        for layer in self.nn:
            x = layer(x)
        x = self.before_scalar(x)

        x_xy = x[:, :, :2]  # [B, C, 2]
        x_z  = x[:, :, 2:]  # [B, C, 1]

        # Compute Gram matrix invariant from (x, y)
        inv_x = torch.matmul(x_xy, x_xy.transpose(1, 2))  # [B, C, C]
        inv_x = inv_x.flatten(1)  # [B, C*C]

        # z-gated modulation of the invariant
        z_flat = x_z.flatten(1)  # [B, C]
        # z_proj = self.z_gating(z_flat)  # [B, C*C]
        # inv_x_mod = inv_x * torch.sigmoid(z_proj)  # gated invariant
        # x_out = inv_x_mod

        # Final feature vector
        x_out = torch.cat([inv_x, z_flat], dim=1)  # [B, C*C + C]

        if self.add_mlp:
            x_out = self.act(self.out1(x_out))
            for layer in self.mlp_out:
                x_out = self.act(layer(x_out))
            return self.out2(x_out)
        else:
            return self.out(x_out)


################################################################################################################################################

class SO2VectNeuron_v4(nn.Module):
    def __init__(self, nbr_const, nbr_feat, coef_norm, out_channel, hidden, num_layers, add_mlp=False):
        super(SO2VectNeuron_v4, self).__init__()
        assert nbr_feat == 3, "Input must be 3D vector features (px, py, pz)"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp

        self.inp = VNLinearAndLeakyReLU(self.in_channel, self.hidden)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden) for _ in range(self.num_layers - 1)])
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, self.hidden)

        if add_mlp:
            self.out1 = nn.Linear(self.hidden ** 2 + self.hidden, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden ** 2 + self.hidden, self.out_channel)

    def forward(self, x):
        
        x = self.inp(x)
        for module in self.nn:
            x = module(x)
        x = self.before_scalar(x)
        px_py = x[:,:,0:2]
        pz = x[:,:,2]
        inv_x = px_py@(px_py.transpose(1,-1)) #invariant layer
        inv_x = torch.flatten(inv_x, start_dim = 1, end_dim = -1)
        pz = torch.flatten(pz, start_dim = 1, end_dim = -1)
        x = torch.concat((inv_x, pz), dim = 1)
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
    
class SO2VectNeuron_v5(nn.Module):
    def __init__(self, nbr_const, nbr_feat, coef_norm, out_channel, hidden, num_layers, add_mlp=False):
        super(SO2VectNeuron_v5, self).__init__()
        assert nbr_feat == 3, "Input must be 3D vector features (px, py, pz)"
        self.in_channel = nbr_const
        self.out_channel = out_channel
        self.hidden = hidden
        self.num_layers = num_layers
        self.add_mlp = add_mlp

        self.inp = VNLinearAndLeakyReLU(self.in_channel, self.hidden)
        self.nn = nn.ModuleList([VNLinearAndLeakyReLU(self.hidden, self.hidden) for _ in range(self.num_layers - 1)])
        self.before_scalar = VNLinearAndLeakyReLU(self.hidden, self.hidden)
        self.replace_inv = nn.Linear(self.hidden*3, self.hidden ** 2)

        if add_mlp:
            self.out1 = nn.Linear(self.hidden ** 2 + self.hidden, 128)
            self.mlp_out = nn.ModuleList([nn.Linear(128, 128) for _ in range(4)])
            self.out2 = nn.Linear(128, self.out_channel)
            self.act = nn.ReLU()
        else:
            self.out = nn.Linear(self.hidden ** 2, self.out_channel)

    def forward(self, x):
        
        x = self.inp(x)
        for module in self.nn:
            x = module(x)
        x = self.before_scalar(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = F.relu(self.replace_inv(x))
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