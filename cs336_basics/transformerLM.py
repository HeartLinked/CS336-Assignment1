import math

import torch
import torch.nn as nn
import torch.nn.init as init
from jaxtyping import Bool, Float, Int

from cs336_basics.rope import RoPE
from cs336_basics.transformer_block import TransformerBlock

from cs336_basics.bpe import my_run_train_bpe
from cs336_basics.tokenizer import my_get_tokenizer
from cs336_basics.linear import Linear
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGlu
from cs336_basics.rope import RoPE
from cs336_basics.multihead_self_attention_with_rope import MultiHeadSelfAttentionWithRope
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
    ):
        super().__init__()

        head_dim = d_model // num_heads
        rope_cache = RoPE(
            d_k=head_dim,
            theta=rope_theta,
            max_seq_len=context_length,
            device=device,
        )

        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope_cache=rope_cache,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, device=device)

    def forward(
        self,
        in_indices: torch.Tensor,
    ) -> Float[torch.Tensor, " batch_size sequence_length vocab_size"]:
        x = self.token_embeddings(in_indices) 

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)               
        return logits