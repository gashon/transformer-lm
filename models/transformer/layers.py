import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer.util import scaled_dot_product_attention, softmax, gelu

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, attn_pdrop=None, residual_pdrop=None):
        super(TransformerBlock, self).__init__()
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.mha = CausalMultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.dropout1 = nn.Dropout(residual_pdrop, inplace=False) if residual_pdrop is not None else nn.Identity()

        self.norm2 = RMSNorm(d_model, eps=1e-5)
        self.ff = PositionWiseFeedForward(d_model, d_ff)
        self.dropout2 = nn.Dropout(residual_pdrop, inplace=False) if residual_pdrop is not None else nn.Identity()

    def forward(self, x):
        rms_norm_output = self.norm1(x)
        mha_output = self.mha(rms_norm_output)
        dropout_output = self.dropout1(mha_output)
        attn_output = x + dropout_output

        rms_norm_output = self.norm2(attn_output)
        ff_output = self.ff(rms_norm_output)
        dropout_output = self.dropout2(ff_output)

        return attn_output + dropout_output
        

    def load_weights(self, weights):
        # Load weights for multi-head attention
        self.mha.q_heads.data.copy_(weights['attn.q_proj.weight'])
        self.mha.k_heads.data.copy_(weights['attn.k_proj.weight'])
        self.mha.v_heads.data.copy_(weights['attn.v_proj.weight'])
        self.mha.output_proj.data.copy_(weights['attn.output_proj.weight'])

        # Load weights for RMSNorm layers
        self.norm1.gain.data.copy_(weights['ln1.weight'])
        self.norm2.gain.data.copy_(weights['ln2.weight'])

        # Load weights for feed-forward network
        self.ff.w1.weight.data.copy_(weights['ffn.w1.weight'])
        self.ff.w2.weight.data.copy_(weights['ffn.w2.weight'])


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

        self.attn_pdrop = attn_pdrop

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
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1) > 0
        attn_output = scaled_dot_product_attention(q, k, v, mask=causal_mask, pdrop=self.attn_pdrop)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return torch.matmul(attn_output, self.output_proj.transpose(0, 1))

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

