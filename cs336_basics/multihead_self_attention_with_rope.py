import math

import einops
import torch
import torch.nn as nn
import torch.nn.init as init

from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention


class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_cache: RoPE):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_cache = rope_cache

        self.weight_q = Linear(d_model, d_model)
        self.weight_k = Linear(d_model, d_model)
        self.weight_v = Linear(d_model, d_model)
        self.weight_o = Linear(d_model, d_model)

    def forward(self, 
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor: 
        batch_size, seq_len, _ = x.shape

        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
            token_positions = token_positions.unsqueeze(0).expand(batch_size, seq_len)
        # token_positions: [batch, seq_len]

        # QKV projection
        q = self.weight_q(x)
        k = self.weight_k(x)
        v = self.weight_v(x)

        # split heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, num_heads, seq_len, head_dim]

        # expand token_positions to match q/k batch dims
        rope_positions = token_positions.unsqueeze(1).expand(batch_size, self.num_heads, seq_len)
        # rope_positions: [batch, num_heads, seq_len]

        # apply RoPE to Q/K 
        if self.rope_cache is not None:
            q = self.rope_cache(q, rope_positions)
            k = self.rope_cache(k, rope_positions)

        causal_mask = torch.tril(
           torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attn_out = scaled_dot_product_attention(
            q, k, v,
            mask=causal_mask,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.weight_o(attn_out)
        return out