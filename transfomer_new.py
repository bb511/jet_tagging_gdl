# Transformer, this implementation is based on the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762)
# and the PyTorch implementation (https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
# This has been suggested by Maximilian Herde (but I am the one to blame for wrong usage)
import torch
import torch.nn as nn

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
        
        # Store normalization method
        self.norm_method = coef_norm.get('method', 'none')
        
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

        # Lifting layer for each constituent
        self.lifting_layer = nn.Linear(nbr_feat, d_models)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_models, nhead=self.nhead, batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(self.d_models * self.nbr_const, self.output_size)
        
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
        # Create mask based on zero values
        mask = (x.sum(dim=-1) == 0)  # (batch, number of constituents)
        
        x = self.lifting_layer(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        x = x.reshape(x.size(0), -1)  # (batch, number of constituents * d_models)
        
        x = self.fc(x)
        
        return x