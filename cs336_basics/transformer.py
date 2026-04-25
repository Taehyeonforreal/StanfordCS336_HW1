import torch
from torch import Tensor
import torch.nn as nn

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


# RoPE Class
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # 주파수 계산 후 버퍼로 저장 (학습 안 되는 상수)
        i = torch.arange(0, d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * i / d_k))  # (d_k/2,)
        
        # 모든 position에 대해 미리 cos, sin 계산
        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len,)
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (max_seq_len, d_k/2)
        
        # register_buffer: 모델 저장/로드에 포함되지만 학습은 안 됨
        self.register_buffer('cos', torch.cos(angles))  # (max_seq_len, d_k/2)
        self.register_buffer('sin', torch.sin(angles))  # (max_seq_len, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions: (..., seq_len)
        # cos, sin을 position에 맞게 슬라이싱
        cos = self.cos[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin[token_positions]  # (..., seq_len, d_k/2)
        
        # 짝수/홀수 인덱스 분리
        x_even = x[..., 0::2]  # (..., seq_len, d_k/2)
        x_odd  = x[..., 1::2]  # (..., seq_len, d_k/2)
        
        # 회전 적용
        out_even = x_even * cos - x_odd * sin
        out_odd  = x_even * sin + x_odd * cos
        
        # 다시 합치기
        out = torch.stack([out_even, out_odd], dim=-1)
        return out.flatten(-2)  # (..., seq_len, d_k)

def run_rope(d_k: int, theta: float, max_seq_len: int, in_query_or_key: Tensor, token_positions: Tensor) -> Tensor:
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return rope(in_query_or_key, token_positions)
