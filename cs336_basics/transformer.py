import torch
from torch import Tensor

# Softmax.
def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    exp_x = torch.exp(in_features)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Linear Model. weight * input
def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.T

# embdedding. integer list -> vector
def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]

# RMSNorm. (substitution of LayerNorm)
def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    rms = torch.sqrt(in_features.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (in_features / rms) * weights

# SiLU = x * sigma(x).
def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)

# SwiGLU(x) = (SiLU(x @ W1.T) ⊙ (x @ W3.T)) @ W2.T
def run_swiglu(d_model: int, d_ff: int, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor, in_features: Tensor) -> Tensor:
    gate = run_silu(in_features @ w1_weight.T)
    up   = in_features @ w3_weight.T
    return (gate * up) @ w2_weight.T
