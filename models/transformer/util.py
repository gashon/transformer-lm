import torch
import torch.nn as nn

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

def gelu(features: torch.FloatTensor) -> torch.FloatTensor:
    """Given input features, return the output of applying a GELU activation function.

    Args:
        features: torch.FloatTensor
            Input features to apply GELU activation on.
            Shape is (batch_size, seq_len, d_model).

    Returns:
        gelu_features: torch.FloatTensor
            Output features after applying GELU activation.
            Shape is (batch_size, seq_len, d_model).
    """
    return 0.5 * features * (1 + torch.erf(features / torch.sqrt(torch.tensor(2.0))))

def softmax(scores: torch.Tensor, dim: int) -> torch.Tensor:
    """Given a tensor of scores, apply softmax activation along the specified dimension.

    Args:
        scores: torch.Tensor
            Tensor of scores to apply softmax on.
        dim: int
            Dimension to apply softmax on.

    Returns:
        probabilities: torch.Tensor
            Tensor of probabilities after applying softmax.
    """
    # Subtract the maximum value for numerical stability
    max_scores = scores.max(dim=dim, keepdim=True).values
    exp_scores = torch.exp(scores - max_scores)

    # Compute softmax scores
    probabilities = exp_scores / exp_scores.sum(dim=dim, keepdim=True)
    return probabilities

