import math

import torch
import torch.nn as nn
import torch.nn.init as init

# finally, we want : E: (vocab_size, d_model)
# this is a embedding matrix.
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device = None, dtype = None):
        super().__init__()
        # output = x @ W^T
        self.weight = nn.Parameter(
            torch.empty(vocab_size, d_model, device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(
            self.weight, 
            mean = 0.0,
            std = 1.0,
            a = -3.0, 
            b = 3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        return self.weight[token_ids]

