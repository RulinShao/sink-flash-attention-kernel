"""
Monkey-patch verl to use sink flash attention for GRPO training.

Usage:
    # In your training script, before model initialization:
    from sink_attention.verl_patch import patch_verl_with_sink_attention
    patch_verl_with_sink_attention(num_sink=4, window_size=4096)

    # Then run verl training as normal -- all attention calls will use sink FA.

Or from command line:
    python -c "from sink_attention.verl_patch import patch_verl_with_sink_attention; patch_verl_with_sink_attention()" && python -m verl.trainer ...

How it works:
    verl calls transformers' _flash_attention_forward() for all attention layers.
    This patch replaces that function with our sink flash attention kernel.
    The replacement handles:
    - Layout transpose: transformers uses [B, N, H, D], our kernel uses [B, H, N, D]
    - Ulysses sequence parallelism: the patch is applied BEFORE the Ulysses wrapper,
      so SP all-to-all happens around our kernel (same as with FA2)
    - GQA: our kernel handles GQA natively
    - Variable-length (packed) sequences: falls back to original FA for varlen
"""

import sys
import os
import math
import torch
from typing import Optional

# Add sink_attention to path
_SINK_ATTN_DIR = os.path.dirname(os.path.abspath(__file__))
if _SINK_ATTN_DIR not in sys.path:
    sys.path.insert(0, _SINK_ATTN_DIR)

from sink_flash_attention import sink_flash_attention

# Global config
_SINK_ATTENTION_CONFIG = {
    "num_sink": 4,
    "window_size": 4096,
    "enabled": False,
}

# Store original function for fallback
_original_flash_attention_forward = None


def _sink_flash_attention_forward(
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
    """
    Drop-in replacement for transformers' _flash_attention_forward that uses
    sink flash attention.

    Handles the layout difference:
    - transformers: [B, N, H, D]
    - sink_flash_attention: [B, H, N, D]
    """
    # Fall back to original FA for cases we don't support:
    # 1. Variable-length (packed) sequences with cu_seqlens
    # 2. Non-causal attention
    # 3. Attention mask present (padding)
    has_varlen = all(x is not None for x in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k))
    has_packed = position_ids is not None and query_states.size(0) > 0 and _is_packed(position_ids)

    if has_varlen or has_packed or not is_causal or attention_mask is not None:
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

    num_sink = _SINK_ATTENTION_CONFIG["num_sink"]
    window_size = _SINK_ATTENTION_CONFIG["window_size"]

    # Transpose: [B, N, H, D] -> [B, H, N, D]
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    # Call our kernel
    out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)

    # Transpose back: [B, H, N, D] -> [B, N, H, D]
    out = out.transpose(1, 2).contiguous()

    return out


def _is_packed(position_ids: torch.Tensor) -> bool:
    """Check if position_ids indicate packed/variable-length sequences."""
    if position_ids is None:
        return False
    if position_ids.dim() < 2:
        return False
    # Packed sequences have position_ids that reset to 0 mid-sequence
    # Check if any position is less than its predecessor
    if position_ids.size(1) <= 1:
        return False
    diffs = position_ids[:, 1:] - position_ids[:, :-1]
    return (diffs < 0).any().item()


def patch_verl_with_sink_attention(num_sink: int = 4, window_size: int = 4096):
    """
    Monkey-patch verl to use sink flash attention.

    Call this BEFORE initializing the verl trainer/model.

    Args:
        num_sink: Number of sink tokens (default: 4)
        window_size: Sliding window size (default: 4096)
    """
    global _original_flash_attention_forward

    _SINK_ATTENTION_CONFIG["num_sink"] = num_sink
    _SINK_ATTENTION_CONFIG["window_size"] = window_size
    _SINK_ATTENTION_CONFIG["enabled"] = True

    # Patch transformers' _flash_attention_forward
    import transformers.modeling_flash_attention_utils as fa_utils

    _original_flash_attention_forward = fa_utils._flash_attention_forward
    fa_utils._flash_attention_forward = _sink_flash_attention_forward

    # Also patch the module-level import that verl uses
    try:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _sink_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    print(f"[SinkAttention] Patched verl with sink attention: "
          f"num_sink={num_sink}, window_size={window_size}")


def unpatch_verl():
    """Restore original flash attention."""
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

    _SINK_ATTENTION_CONFIG["enabled"] = False
    print("[SinkAttention] Restored original flash attention")
