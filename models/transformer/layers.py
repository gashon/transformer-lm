import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.transformer.util import scaled_dot_product_attention, softmax

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dk = self.dv = d_model // num_heads

        # Define the projection matrices
        self.q_heads = nn.Parameter(torch.Tensor(d_model, d_model))
        self.k_heads = nn.Parameter(torch.Tensor(d_model, d_model))
        self.v_heads = nn.Parameter(torch.Tensor(d_model, d_model))
        self.output_proj = nn.Parameter(torch.Tensor(d_model, d_model))

        # Dropout (optional)
        self.dropout = nn.Dropout(attn_pdrop) if attn_pdrop is not None else None

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights using a sensible strategy, e.g., Glorot initialization
        nn.init.xavier_uniform_(self.q_heads)
        nn.init.xavier_uniform_(self.k_heads)
        nn.init.xavier_uniform_(self.v_heads)
        nn.init.xavier_uniform_(self.output_proj)

    def forward(self, x) -> torch.FloatTensor:
        batch_size, seq_len, _ = x.shape

        # Projection matrices
        q = torch.matmul(x, self.q_heads.t()).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        k = torch.matmul(x, self.k_heads.t()).view(batch_size, seq_len, self.num_heads, self.dk).transpose(1, 2)
        v = torch.matmul(x, self.v_heads.t()).view(batch_size, seq_len, self.num_heads, self.dv).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        print("HIT", softmax)
        attn_probs = softmax(attn_scores, dim=-1)
        if self.dropout:
            attn_probs = self.dropout(attn_probs)

        # Compute the final output
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return torch.matmul(attn_output, self.output_proj.t())

    def load_weights(self, weights):
        self.q_heads.data = torch.cat([weights[f'q_heads.{i}.weight'] for i in range(self.num_heads)], dim=0)
        self.k_heads.data = torch.cat([weights[f'k_heads.{i}.weight'] for i in range(self.num_heads)], dim=0)
        self.v_heads.data = torch.cat([weights[f'v_heads.{i}.weight'] for i in range(self.num_heads)], dim=0)

        self.output_proj.data.copy_(weights['output_proj.weight'])


    
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

