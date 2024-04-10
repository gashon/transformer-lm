import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float, gain=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps 
        self.gain = nn.Parameter(torch.ones(d_model)) if gain is None else gain 

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_scaled = self.gain * x_normed
        return x_scaled

