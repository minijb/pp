import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict
from typing import Optional, Tuple, Union

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
 
    def attention(self, x: torch.Tensor, need_weights=False, attn_mask: torch.Tensor = None):
        attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else attn_mask
        return self.attn(x, x, x, need_weights=need_weights, attn_mask=attn_mask)
    
    def save_attention_map(self, attn):
        self.attn_map = attn     
        
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients 
        
    def forward(self, x: torch.Tensor, save_attn=False, attn_mask: torch.Tensor = None):
        attn_output, attn_output_weights = self.attention(self.ln_1(x), need_weights=save_attn, attn_mask=attn_mask)
        
#         if save_attn:
#             self.save_attention_map(attn_output_weights)
#             attn_output_weights.register_hook(self.save_attn_gradients)        
            
        x = x + attn_output
        x_ffn = self.mlp(self.ln_2(x))
        x = x + x_ffn
        return x
    

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask=None):
        for i,block in enumerate(self.resblocks):
            if i==self.layers-1:                    
                x = block(x, save_attn=True, attn_mask=attn_mask)
            else:
                x = block(x, attn_mask=attn_mask)    
        return x
