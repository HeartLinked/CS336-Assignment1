import math

import torch
import torch.nn as nn
import torch.nn.init as init

from cs336_basics.rope import RoPE

class TransformerLM(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, theta, max_seq_len, device=None):
        super().__init__()
        head_dim = d_model // num_heads

        self.rope_cache = RoPE(
            theta=theta,
            d_k=head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                rope_cache=self.rope_cache,
            )
            for _ in range(num_layers)
        ])