import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.util import scaled_dot_product_attention 

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float = None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attn_dropout = nn.Dropout(attn_pdrop) if attn_pdrop is not None else nn.Identity()
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(attn_output)
    
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

