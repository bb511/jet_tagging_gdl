import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self,
                nbr_const:int, # number of constituents
                nbr_feat:int, # number of features
                lifting_dim:int, # dimension of the lifting layer
                num_layers:int, # number of layers
                d_models:int, # dimension of the model
                nhead:int, # number of heads
                output_size:int, # output size
                norm = None, # normalization layer
                
                ):
        self.nbr_const = nbr_const
        self.nbr_feat = nbr_feat
                
        self.lifting_dim = lifting_dim 
        self.num_layers = num_layers
        self.d_models = d_models
        self.nhead = nhead
        
        self.output_size = output_size
        self.norm = norm

        self.lifting_layer = nn.Linear(nbr_feat, lifting_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_models, nhead=self.nhead)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.fc = nn.Linear(self.lifting_dim, self.output_size)
        
    def forward(self, x, x_mask):
        x = self.lifting_layer(x)
        x = self.transformer_encoder(x, x_mask)
        x = self.fc(x)
        return x