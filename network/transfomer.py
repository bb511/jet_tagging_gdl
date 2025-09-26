# Transformer, this implementation is based on the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
# and the PyTorch implementation (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
# This has been suggested by Maximilian Herde (but I am the one to blame for wrong usage)
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self,                                                
                 nbr_const: int,  # number of constituents
                 nbr_feat: int,  # number of features
                 coef_norm: dict,
                 lifting_dim: int,  # dimension of the lifting layer
                 num_layers: int,  # number of layers
                 d_models: int,  # dimension of the model
                 nhead: int,  # number of heads
                 output_size: int,  # output size
                 norm=None,  # normalization layer
                 bias: bool = True,
                 use_mask: bool = True,
                 fast_path: bool = False,
                 fill_masked: bool = False,
                ):
        super(Transformer, self).__init__()
        
        self.nbr_const = nbr_const
        self.nbr_feat = nbr_feat
        
        self.lifting_dim = lifting_dim
        self.num_layers = num_layers
        self.d_models = d_models
        self.nhead = nhead
        
        self.output_size = output_size
        self.norm = norm
        self.use_mask = use_mask
        self.fill_masked = fill_masked
        
        # Store normalization method
        if coef_norm is None:
            self.norm_method = 'none'
        else:
            raise NotImplementedError("Normalization inside the model is not supported anymore.")

        # Lifting layer for each constituent
        self.lifting_layer = nn.Linear(nbr_feat, d_models, bias=bias)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_models, nhead=self.nhead, batch_first=True, bias=bias)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, enable_nested_tensor=fast_path)
        self.fc = nn.Linear(self.d_models * self.nbr_const, self.output_size)
        
    def forward(self, x):     
        # Create mask based on zero values
        mask = (x.sum(dim=-1) == 0)  # (batch, number of constituents)
        x = self.lifting_layer(x)
        
        if self.use_mask:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Explicitly zero out outputs at masked positions to prevent garbage in FC
        if self.use_mask and self.fill_masked:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
        x = x.reshape(x.size(0), -1)  # (batch, number of constituents * d_models)
        x = self.fc(x)
        
        return x
    
