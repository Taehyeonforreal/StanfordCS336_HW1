import torch
from torch import Tensor

# SiLU = x * sigma(x)
def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)

# Softmax
def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    exp_x = torch.exp(in_features)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

# Linear Model. weight * input
def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    return in_features @ weights.T

# embdedding, integer list -> vector
def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    return weights[token_ids]
