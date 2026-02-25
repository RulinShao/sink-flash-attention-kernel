"""
Decode-time attention for Sink Flash Attention.

For inference decode (single query token against cached KV), the cache manager
already provides a contiguous [sink, window] KV tensor. This module wraps the
existing sink_flash_attention kernel for the N_q=1 case, and can be replaced
with an optimized Triton kernel later.

Note: During decode the KV tensor from the cache is contiguous and already
contains only valid tokens (no gap between sink and window). We use standard
causal attention (no sink/window mask) since the cache linearization handles
the masking implicitly — all returned KV tokens should be attended to.
"""

import torch
import math


def sink_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Single-query attention for decode step.

    The KV cache has already been linearized by SinkCacheLayer.get_kv() into
    [sink_tokens, window_tokens] order. All tokens in the KV are valid and
    should be attended to (no masking needed — the cache handles eviction).

    Args:
        q: [B, H_q, 1, D] — single query token
        k: [B, H_kv, N_kv, D] — cached keys (sink + window, contiguous)
        v: [B, H_kv, N_kv, D] — cached values

    Returns:
        output: [B, H_q, 1, D]
    """
    B, H_q, _, D = q.shape
    H_kv = k.shape[1]
    N_kv = k.shape[2]

    scale = 1.0 / math.sqrt(D)

    # GQA: expand KV heads to match Q heads
    if H_q != H_kv:
        groups = H_q // H_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # q: [B, H_q, 1, D], k: [B, H_q, N_kv, D]
    # scores: [B, H_q, 1, N_kv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # No masking needed: all KV tokens are valid (cache handles eviction)
    attn = torch.softmax(scores, dim=-1)

    # output: [B, H_q, 1, D]
    output = torch.matmul(attn.to(v.dtype), v)

    return output
