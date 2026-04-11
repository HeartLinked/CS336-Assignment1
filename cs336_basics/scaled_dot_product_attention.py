import math

import torch
import einops
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init

from cs336_basics.rope import RoPE
from jaxtyping import Bool, Float, Int

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... keys d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    # queries 表示“有多少个 query 位置要发起注意力查询”。
    # 在 self-attention 里，它通常就是序列长度 seq_len
    # keys 表示“有多少个可被关注的位置”。
    # 在 self-attention 里，通常：queries = seq_len, keys = seq_len
    Q_mul_k = einops.einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    # 注意它的形状：对每个query查询，有一组keys（对应的打分），我们softmax处理的就是这一组打分
    d_k = K.shape[-1]
    scores = Q_mul_k / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    # 所以这里是对最后一维keys进行softmax 
    attn = torch.softmax(scores, dim=-1)
    output = einops.einsum(attn, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return output