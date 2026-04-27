import os
import torch
from torch import Tensor
from typing import Iterable
from cs336_basics.transformer import run_softmax
import math

# Cross Entropy
def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # input : run_transformer_lm 출력, target : 실제 다음 단어

    # 1. softmax로 확률 변환
    probs = run_softmax(inputs, dim=-1)
    
    # 2. 각 예시에서 정답 위치의 확률만 꺼내서, -log 취하기
    batch_size = inputs.size(0)
    correct_probs = torch.zeros(batch_size)
    losses = torch.zeros(batch_size)
    for i in range(batch_size):
        correct_word_id = targets[i]           # i번째 예시의 정답 단어 ID
        correct_probs[i] = probs[i][correct_word_id]  # 그 단어의 확률
        losses[i] = -torch.log(correct_probs[i])
    
    # 3. -log 취하고 평균
    return losses.mean()


# Gradient Clipping, 그래디언트 폭주 방지
def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # parameters는 generator 이기에, list로 만들어 재사용 할 수 있게
    params = list(parameters)
    
    # 1. 모든 파라미터의 gradient를 모아서 전체 L2 norm 계산
    total_norm_sq = 0.0
    for param in params:
        if param.grad is not None:
            total_norm_sq += param.grad.pow(2).sum().item()
    total_norm = total_norm_sq ** 0.5

    # 2. norm이 max_l2_norm을 넘으면 scaling
    # scaling을 하면, max_l2_norm을 넘던 것을 딱 max_l2_norm으로 맞춤
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)  # 1e-6은 0으로 나누기 방지
        for param in params:
            if param.grad is not None:
                param.grad.mul_(scale)



# AdamW, 클래스만 반환하기
def get_adamw_cls():
    return torch.optim.AdamW

# Learning Rate Sceduling, cosine annealing schedule 사용.
# 식은 Assignment instruction에 나와있는거 보기
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    
    # 구간 1: Warmup
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    
    # 구간 3: Cosine 이후
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    # 구간 2: Cosine decay
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_value = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + (max_learning_rate - min_learning_rate) * cosine_value


# bat
