import math

from cs336_basics.linear import Linear
import torch
import torch.nn as nn
import torch.nn.init as init

class SwiGlu(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        # output = x @ W^T
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor :
        gate = self.w1(input)
        value = self.w3(input)
        gate = gate * torch.sigmoid(gate)
        hidden = gate * value
        out = self.w2(hidden)
        return out
        

