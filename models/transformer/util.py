from typing import Optional
import numpy as np
import math
import torch
import torch.nn.functional as F
from collections.abc import Callable
from typing import Optional

def scaled_dot_product_attention(q: torch.FloatTensor, k: torch.FloatTensor, v: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None, pdrop: Optional[float]= 0.0):
    """Compute scaled dot-product attention.
        
    Args:
        q: torch.FloatTensor
            Query tensor of shape (batch_size, seq_len_q, d_model).
        k: torch.FloatTensor
            Key tensor of shape (batch_size, seq_len_k, d_model).
        v: torch.FloatTensor
            Value tensor of shape (batch_size, seq_len_v, d_model).
        mask: torch.BoolTensor
            Mask tensor of shape (batch_size, seq_len_q, seq_len_k).
        pdrop: float
            Dropout probability.

    Returns:
        output: torch.FloatTensor

    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())

    # Apply mask to the scores
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)

    if pdrop and pdrop > 0.0:
        attn_weights = F.dropout(attn_weights, p=pdrop)

    output = torch.matmul(attn_weights, v)

    return output 

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

def softmax(scores: torch.FloatTensor, dim: int) -> torch.FloatTensor:
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

def cross_entropy_loss(logits: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
    """Given a tensor of logits and a tensor of targets, compute the cross-entropy loss.

    Args:
        logits: torch.FloatTensor
            Tensor of logits from the model.
            Shape is (batch_size, seq_len, vocab_size).
        targets: torch.LongTensor
            Tensor of targets.
            Shape is (batch_size, seq_len).

    Returns:
        loss: torch.FloatTensor
            Scalar tensor representing the cross-entropy loss.
    """
    # Ensure logits are float32 for numerical stability
    logits = logits.float()

    # Subtract the maximum value from logits along vocab dimension for numerical stability
    max_logits = torch.max(logits, dim=1, keepdim=True)[0]
    logits = logits - max_logits

    # Compute the exponentials of the logits and their sum
    exp_logits = torch.exp(logits)
    sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)

    # Compute softmax probabilities
    softmax_probs = exp_logits / sum_exp_logits

    # Gather the softmax probabilities for the correct classes
    target_probs = softmax_probs.gather(1, targets.unsqueeze(1)).squeeze()

    # Compute the negative log likelihood
    neg_log_likelihood = -torch.log(target_probs + 1e-9)  # Adding a small value to prevent log(0)

    # Return the average loss over the batch
    return torch.mean(neg_log_likelihood)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps <= 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                lr = group['lr']
                epsilon = group['epsilon']
                weight_decay = group['weight_decay']

                state['step'] += 1

                # AdamW update
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1

                p.data.addcdiv_(m, v.sqrt().add_(epsilon), value=-step_size)
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss

def cosine_learning_rate_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    """
    Calculate the learning rate at a given iteration using a cosine annealing schedule with a warm-up period.

    Args:
        it: int
            Iteration number to get learning rate for.
        max_learning_rate: float
            alpha_max, the maximum learning rate for the cosine learning rate schedule (with warmup).
        min_learning_rate: float
            alpha_min, the minimum / final learning rate for the cosine learning rate schedule (with warmup).
        warmup_iters: int
            T_w, the number of iterations to linearly warm-up the learning rate.
        cosine_cycle_iters: int
            T_c, the total number of iterations over which the cosine schedule should run.

    Returns:
        float: The computed learning rate at the specified iteration.
    """
    if it < warmup_iters:
        # Linear warm-up phase from 0 to max_learning_rate
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine annealing phase from max_learning_rate to min_learning_rate
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    else:
        # Post-annealing phase, constant at min_learning_rate
        return min_learning_rate

def clip_gradients(parameters, max_norm):
    """
    Clip the gradients of the parameters to the specified maximum l2-norm.

    Args:
        parameters: Iterable[torch.nn.Parameter]
            An iterable of parameters whose gradients need to be clipped.
        max_norm: float
            The maximum norm for the gradients.

    Returns:
        None; the function modifies gradients in-place.
    """
    # Calculate the total norm of all parameters
    total_norm = torch.sqrt(sum(torch.sum(p.grad.data ** 2) for p in parameters if p.grad is not None) + 1e-6)

    # Scale down gradients if the total norm exceeds the max_norm
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)


