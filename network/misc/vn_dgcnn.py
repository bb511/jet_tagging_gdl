# adapted from https://github.com/FlyingGiraffe/vnn/blob/master/models/vn_layers.py

import numpy as np
import matplotlib.pyplot as plt


import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.cuda.manual_seed_all(15)

from torch.utils.checkpoint import checkpoint

from network.vector_neurons import VNLinearAndLeakyReLU
from network.misc.util_vn import VNStdFeature, VNMaxPool, mean_pool

def knn(x, k):
    inner = 2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx + inner - xx.transpose(2, 1) # as ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2<x_i, x_j>

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)
        else:          # fixed knn graph with input point coordinates
            idx = knn(x_coord, k=k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1) # index of the k nearest neighbors

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 4, 1, 2).contiguous() # as x_i' = aggregation of (h_w(x_i-x_j, x_j))
  
    return feature




class VN_DGCNN(nn.Module):
    def __init__(self, nbr_const:int, nbr_feat:int, n_knn:int, pooling:str,  output=5):
        super(VN_DGCNN, self).__init__()
        self.nbr_const = nbr_const
        self.nbr_feat = nbr_feat
        self.n_knn = n_knn
        
        self.conv1 = VNLinearAndLeakyReLU(2, 8//3, dim=5)#64//3
        self.conv2 = VNLinearAndLeakyReLU(8//3*2, 8//3, dim=5)#64//3, #64//3
        self.conv3 = VNLinearAndLeakyReLU(8//3*2, 16//3, dim=5)#64//3, 128//3
        self.conv4 = VNLinearAndLeakyReLU(16//3*2, 32//3, dim=5)#128//3, 256//3

        self.conv5 = VNLinearAndLeakyReLU(32//3+16//3+8//3*2, 64//3, dim=4, share_nonlinearity=True)#256//3 + 128//3 + 64//3*2, 1024//3
        
        self.std_feature = VNStdFeature(64//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((64//3)*12, 32)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(16, output)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(8//3)
            self.pool2 = VNMaxPool(8//3)
            self.pool3 = VNMaxPool(16//3)
            self.pool4 = VNMaxPool(32//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
        else:
            raise ValueError('Pooling must be either max or mean')

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x= x.transpose(-1, 1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        trans_feat = None
        return x#, trans_feat
    
class VN_DGCNN_checkpoint(nn.Module):
    def __init__(self, nbr_const:int, nbr_feat:int, n_knn:int, pooling:str,  output=5):
        super(VN_DGCNN_checkpoint, self).__init__()
        self.nbr_const = nbr_const
        self.nbr_feat = nbr_feat
        self.n_knn = n_knn
        
        self.conv1 = VNLinearAndLeakyReLU(2, 64//3, dim=5)
        self.conv2 = VNLinearAndLeakyReLU(64//3*2, 64//3, dim=5)
        self.conv3 = VNLinearAndLeakyReLU(64//3*2, 128//3, dim=5)
        self.conv4 = VNLinearAndLeakyReLU(128//3*2, 256//3, dim=5)

        self.conv5 = VNLinearAndLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output)
        
        if pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool
        else:
            raise ValueError('Pooling must be either max or mean')

    def _block1(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        return self.pool1(x)

    def _block2(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv2(x)
        return self.pool2(x)

    def _block3(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv3(x)
        return self.pool3(x)

    def _block4(self, x):
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv4(x)
        return self.pool4(x)

    def _final_blocks(self, x):
        batch_size = x.size(0)
        x = self.conv5(x)
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)

    def forward(self, x):
        x = x.unsqueeze(-1).transpose(-1, 1)
        
        x1 = checkpoint(self._block1, x)
        x2 = checkpoint(self._block2, x1)
        x3 = checkpoint(self._block3, x2)
        x4 = checkpoint(self._block4, x3)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = checkpoint(self._final_blocks, x)
        
        return x