import math

import torch
import torch.nn as nn
import torch.nn.init as init

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, but got d_k={d_k}")
        
        self.theta = theta
        self.d_k = d_k

        pair_ids = torch.arange(d_k // 2, dtype=torch.float32, device=device)  # 0,1,2,...
        inv_freq = 1.0 / (self.theta ** (2 * pair_ids / d_k))

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # 0,1,2,...
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)
        

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_k]
        # token_positions: [batch, seq_len]
        cos = self.cos_cached[token_positions]   # [batch, seq_len, d_k // 2]
        sin = self.sin_cached[token_positions]   # [batch, seq_len, d_k // 2]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

