import math

import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        # out = x @ W^T
        # shape of weight -> out_features, in_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device = device, dtype = dtype)
        )
        std = math.sqrt(2 / (in_features + out_features))

        torch.nn.init.trunc_normal_(
            self.weight, 
            mean = 0.0,
            std = std,
            a = -3.0 * std,
            b = 3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return x @ self.weight.transpose(-1, -2)
        return x @ self.weight.transpose(0, 1)
