import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.util import scaled_dot_product_attention 

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None, weights=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.dv = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model, d_model)
        
        self.attn_pdrop = attn_pdrop if attn_pdrop is not None else 0.0
        
        self.dropout = nn.Dropout(self.attn_pdrop)

        if weights is not None:
            self.load_weights(weights)


    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project and split into heads
        queries = self.query_projection(x).view(batch_size, seq_len, self.num_heads, self.dk)
        keys = self.key_projection(x).view(batch_size, seq_len, self.num_heads, self.dk)
        values = self.value_projection(x).view(batch_size, seq_len, self.num_heads, self.dv)
        
        # Transpose to get dimensions batch_size, num_heads, seq_len, dk or dv
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Create mask for causal attention
        mask = torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimension
        
        # Scaled dot-product attention
        attn_output = scaled_dot_product_attention(queries, keys, values, mask, self.attn_pdrop)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final output projection
        output = self.output_projection(attn_output)
        return output

    def load_weights(self, weights):
        """ Load weights from a dictionary """
        for i in range(self.num_heads):
            self.query_projection.weight.data[:, i*self.dk:(i+1)*self.dk] = weights[f"q_heads.{i}.weight"].t()
            self.key_projection.weight.data[:, i*self.dk:(i+1)*self.dk] = weights[f"k_heads.{i}.weight"].t()
            self.value_projection.weight.data[:, i*self.dk:(i+1)*self.dk] = weights[f"v_heads.{i}.weight"].t()
        
        self.output_projection.weight.data = weights["output_proj.weight"]

        # Assuming no bias in the linear layers, or setting them to zero if they exist.
        if self.query_projection.bias is not None:
            self.query_projection.bias.data.fill_(0)
        if self.key_projection.bias is not None:
            self.key_projection.bias.data.fill_(0)
        if self.value_projection.bias is not None:
            self.value_projection.bias.data.fill_(0)
        if self.output_projection.bias is not None:
            self.output_projection.bias.data.fill_(0)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, gain=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps 
        self.gain = nn.Parameter(torch.ones(d_model)) if gain is None else gain 

    def forward(self, x: torch.FloatTensor):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_scaled = self.gain * x_normed
        return x_scaled

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.activation = gelu 

    def forward(self, x: torch.FloatTensor):
        x = self.activation(self.w1(x))
        x = self.w2(x)
        return x

