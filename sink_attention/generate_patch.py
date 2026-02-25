"""
Monkey-patch HuggingFace model.generate() to use sink flash attention with KV caching.

Usage:
    from sink_attention import patch_for_generation

    cache = patch_for_generation(model, num_sink=4, window_size=4096)
    output = model.generate(input_ids, past_key_values=cache, max_new_tokens=1024)

    # When done, restore original attention:
    from sink_attention.generate_patch import unpatch_generation
    unpatch_generation()

How it works:
    Patches transformers' _flash_attention_forward() to intercept attention calls.
    Detects prefill (N_q > 1) vs decode (N_q == 1):
      - Prefill: uses sink_flash_attention kernel with full sequence
      - Decode: uses sink_decode_attention (simple matmul, KV from cache)
    Returns a SinkAttentionCache instance to pass as past_key_values to generate().
"""

import torch
from typing import Optional

from .sink_flash_attention import sink_flash_attention
from .decode_kernel import sink_decode_attention
from .cache import SinkAttentionCache

# Store original function for fallback and restoration
_original_flash_attention_forward = None

# Global config for the generation patch
_GENERATION_CONFIG = {
    "num_sink": 4,
    "window_size": 4096,
    "enabled": False,
}


def _is_packed(position_ids: torch.Tensor) -> bool:
    """Check if position_ids indicate packed/variable-length sequences."""
    if position_ids is None:
        return False
    if position_ids.dim() < 2:
        return False
    if position_ids.size(1) <= 1:
        return False
    diffs = position_ids[:, 1:] - position_ids[:, :-1]
    return (diffs < 0).any().item()


def _generation_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool = True,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: Optional[bool] = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    implementation: Optional[str] = None,
    **kwargs,
):
    """Drop-in replacement for transformers' _flash_attention_forward.

    Routes to sink_flash_attention for prefill or sink_decode_attention for
    single-token decode. Falls back to original FA2 for unsupported cases.

    Input layout from transformers: [B, N, H, D]
    Kernel layout: [B, H, N, D]
    """
    # Fall back to original FA for cases we don't support
    has_varlen = all(
        x is not None for x in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )
    has_packed = (
        position_ids is not None
        and query_states.size(0) > 0
        and _is_packed(position_ids)
    )

    if has_varlen or has_packed or not is_causal:
        return _original_flash_attention_forward(
            query_states, key_states, value_states, attention_mask, query_length,
            is_causal=is_causal, dropout=dropout, position_ids=position_ids,
            softmax_scale=softmax_scale, sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask, softcap=softcap,
            deterministic=deterministic, cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k, max_length_q=max_length_q,
            max_length_k=max_length_k, target_dtype=target_dtype,
            implementation=implementation, **kwargs,
        )

    num_sink = _GENERATION_CONFIG["num_sink"]
    window_size = _GENERATION_CONFIG["window_size"]

    # Transpose: [B, N, H, D] -> [B, H, N, D]
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    N_q = q.shape[2]
    N_kv = k.shape[2]

    if N_q > 1:
        # Prefill: use the full sink flash attention kernel
        # During prefill with KV cache, N_q == N_kv (full sequence)
        out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
    else:
        # Decode: single query against cached KV
        # KV comes from the cache (already has sink + window concatenated by
        # the model's attention layer using past_key_values)
        # Use simple attention — all KV tokens are valid
        out = sink_decode_attention(q, k, v)

    # Transpose back: [B, H, N, D] -> [B, N, H, D]
    out = out.transpose(1, 2).contiguous()
    return out


def patch_for_generation(
    model=None,
    num_sink: int = 4,
    window_size: int = 4096,
) -> SinkAttentionCache:
    """Patch HuggingFace model for generation with sink attention caching.

    Args:
        model: HuggingFace model (unused, kept for API clarity). The patch is
               applied globally to transformers' flash attention forward.
        num_sink: Number of sink tokens to keep permanently.
        window_size: Sliding window size for recent tokens.

    Returns:
        SinkAttentionCache instance to pass as past_key_values to generate().
    """
    global _original_flash_attention_forward

    _GENERATION_CONFIG["num_sink"] = num_sink
    _GENERATION_CONFIG["window_size"] = window_size
    _GENERATION_CONFIG["enabled"] = True

    # Patch transformers' _flash_attention_forward
    import transformers.modeling_flash_attention_utils as fa_utils

    _original_flash_attention_forward = fa_utils._flash_attention_forward
    fa_utils._flash_attention_forward = _generation_flash_attention_forward

    # Also patch the integrations module if present
    try:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _generation_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    # Return a fresh cache for the caller to use with generate()
    cache = SinkAttentionCache(num_sink=num_sink, window_size=window_size)
    return cache


def unpatch_generation():
    """Restore original flash attention function."""
    global _original_flash_attention_forward
    if _original_flash_attention_forward is None:
        return

    import transformers.modeling_flash_attention_utils as fa_utils
    fa_utils._flash_attention_forward = _original_flash_attention_forward

    try:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _original_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    _GENERATION_CONFIG["enabled"] = False
    _original_flash_attention_forward = None
