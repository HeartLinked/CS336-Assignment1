import math

import torch
import torch.nn as nn
import torch.nn.init as init

from cs336_basics.rope import RoPE
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGlu
from cs336_basics.rope import RoPE
from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttentionWithRope
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, rope_cache: RoPE, device = None):
        super().__init__()
        self.rope_cache = rope_cache
        self.norm1 = RMSNorm(d_model=d_model, device = device)
        self.attn = MultiHeadSelfAttentionWithRope(d_model, num_heads, rope_cache)
        self.norm2 = RMSNorm(d_model=d_model, device = device)
        self.ff = SwiGlu(d_model, d_ff, device=device)
        
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
