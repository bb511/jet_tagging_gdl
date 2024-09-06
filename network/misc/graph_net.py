# this file contains the implementation of the graph neural network models
# most of them are adapted from the PyTorch Geometric tutorials and documentation
# https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


#from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing
from torch_geometric.utils import add_self_loops, degree


import matplotlib.pyplot as plt

from import_data import select_data

import util

class GCNClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        # Node embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        print(x.shape)
        # Global pooling
        x = global_mean_pool(x, batch)
        print(x.shape)
        
        # Classification
        x = self.lin(x)
        return x
    
class MultiLayerMessagePassingClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(MultiLayerMessagePassingClassifier, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(MPLayer(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(MPLayer(hidden_channels, hidden_channels))
        
        self.convs.append(MPLayer(hidden_channels, out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        x = self.convs[-1](x, edge_index)
        
        x = global_mean_pool(x, batch)
        return x

class MPLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPLayer, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.fill_(0)
    
    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        x = self.lin(x)
        
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.propagate(edge_index, x=x, norm=norm)
        
        out = out + self.bias
        
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j
    
class MultiLayerEdgeConv(torch.nn.Module):
    def __init__(self, nbr_const, nbr_feat, hidden_channels, out_channels, num_layers):
        super(MultiLayerEdgeConv, self).__init__()
        in_channels = nbr_feat
        self.convs = torch.nn.ModuleList()
        self.convs.append(EdgeConvLayer(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(EdgeConvLayer(hidden_channels, hidden_channels))
        
        self.convs.append(EdgeConvLayer(hidden_channels, out_channels))
        
    def forward(self, x, edge_index, batch):
        mask = (x.sum(dim=-1) != 0) 
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, mask)
            x = F.relu(x)
        
        x = self.convs[-1](x, edge_index, mask)
        
        x = x * mask.unsqueeze(-1)
        
        x = global_mean_pool(x, batch)
        return x, mask

class EdgeConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvLayer, self).__init__(aggr='max')  # Using max aggregation as in the original EdgeConv
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, out_channels)
        )
    
    def forward(self, x, edge_index, mask):
        return self.propagate(edge_index, x=x)* mask.unsqueeze(-1)
    
    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)