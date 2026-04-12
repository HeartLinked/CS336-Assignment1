

import torch
from torch import Tensor

def my_run_cross_entropy(
    inputs: Tensor, targets: Tensor
) -> Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs: shape (batch_size, vocab_size)
        targets: shape (batch_size,)

    Returns:
        scalar tensor: average cross-entropy loss
    """
    row_max = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - row_max

    logsumexp = shifted.exp().sum(dim=-1).log()

    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = logsumexp - target_logits
    return loss.mean()