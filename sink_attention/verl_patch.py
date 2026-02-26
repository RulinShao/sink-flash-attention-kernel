"""
Monkey-patch verl to use sink flash attention for training (SFT / GRPO).

Handles the gpt-oss ``s_aux`` (learnable attention sink) mechanism that FA2
silently drops through **kwargs, as well as per-layer ``sliding_window``.

Usage:
    # In your training script, before model initialization:
    from sink_attention.verl_patch import patch_verl_with_sink_attention
    patch_verl_with_sink_attention()

    # Then run verl training as normal -- all attention calls will use
    # the sink flash attention kernel with proper s_aux handling.

Call chain after patching (with Ulysses SP):
    gpt-oss → flash_attention_forward → _ulysses_flash_attention_forward
    → SP all-to-all → _sink_flash_attention_forward → our kernel → SP all-to-all back

Without SP:
    gpt-oss → flash_attention_forward → _sink_flash_attention_forward → our kernel
"""

import torch
from typing import Optional

from .sink_flash_attention import sink_flash_attention
from .decode_kernel import sink_decode_attention

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
    sink flash attention with s_aux support.

    Handles the layout difference:
    - transformers: [B, N, H, D]
    - sink_flash_attention: [B, H, N, D]

    And extracts gpt-oss's s_aux from **kwargs.
    """
    # Extract gpt-oss specific kwargs
    s_aux = kwargs.pop("s_aux", None)

    # Fall back to original FA for cases we don't support:
    # 1. Variable-length (packed) sequences with cu_seqlens
    # 2. Non-causal attention
    # 3. Attention mask present (padding)
    # 4. softcap (not implemented in our kernel)
    has_varlen = all(x is not None for x in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k))
    has_packed = position_ids is not None and query_states.size(0) > 0 and _is_packed(position_ids)

    # Fall back for cases we can't handle at all
    N_q = query_states.shape[1]
    N_kv = key_states.shape[1]

    if has_varlen or has_packed or not is_causal or attention_mask is not None or softcap is not None:
        # Restore s_aux to kwargs for fallback (original FA ignores it harmlessly)
        if s_aux is not None:
            kwargs["s_aux"] = s_aux
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

    # Decode step: N_q=1, N_kv=large (KV cache during model.generate())
    # Use FlashDecoding kernel which parallelizes over KV blocks and
    # correctly handles s_aux without materializing the full attention matrix.
    if N_q != N_kv:
        H_q = query_states.shape[2]

        # Handle s_aux for decode (typically no SP during inference)
        s_aux_local = None
        if s_aux is not None:
            H_total = s_aux.shape[0]
            if H_total == H_q:
                s_aux_local = s_aux
            elif H_total > H_q and H_total % H_q == 0:
                try:
                    from verl.utils.ulysses import get_ulysses_sequence_parallel_rank
                    sp_rank = get_ulysses_sequence_parallel_rank()
                except (ImportError, RuntimeError):
                    sp_rank = 0
                    if torch.distributed.is_initialized():
                        sp_size = H_total // H_q
                        sp_rank = torch.distributed.get_rank() % sp_size
                s_aux_local = s_aux[sp_rank * H_q : (sp_rank + 1) * H_q]

        # Transpose: [B, N, H, D] -> [B, H, N, D]
        q = query_states.transpose(1, 2).contiguous()
        k = key_states.transpose(1, 2).contiguous()
        v = value_states.transpose(1, 2).contiguous()

        out = sink_decode_attention(q, k, v, s_aux=s_aux_local)

        # Transpose back: [B, H, N, D] -> [B, N, H, D]
        return out.transpose(1, 2).contiguous()

    # Input shape: [B, N, H, D] (transformers convention)
    B, N, H_q, D = query_states.shape
    H_kv = key_states.shape[2]

    # Handle s_aux with Ulysses SP: s_aux is [H_total] but we may have
    # only H_q = H_total / sp_size heads on this rank after head scattering.
    s_aux_local = None
    if s_aux is not None:
        H_total = s_aux.shape[0]
        if H_total == H_q:
            # No SP or already sliced
            s_aux_local = s_aux
        elif H_total > H_q and H_total % H_q == 0:
            # Ulysses SP: slice s_aux for this rank's heads
            try:
                from verl.utils.ulysses import get_ulysses_sequence_parallel_rank
                sp_rank = get_ulysses_sequence_parallel_rank()
            except (ImportError, RuntimeError):
                # Fallback: try standard distributed rank within SP group
                sp_rank = 0
                if torch.distributed.is_initialized():
                    sp_size = H_total // H_q
                    sp_rank = torch.distributed.get_rank() % sp_size
            s_aux_local = s_aux[sp_rank * H_q : (sp_rank + 1) * H_q]
        else:
            # Shape mismatch -- skip s_aux
            s_aux_local = None

    # Determine window size for this layer
    # sliding_window=None means full attention (use N as window)
    if sliding_window is not None:
        window_size = sliding_window
    else:
        window_size = N  # full causal attention

    # Transpose: [B, N, H, D] -> [B, H, N, D]
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    # Call our kernel with num_sink=0 (gpt-oss uses s_aux, not positional sinks)
    out = sink_flash_attention(
        q, k, v,
        num_sink=0,
        window_size=window_size,
        s_aux=s_aux_local,
    )

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


def patch_verl_with_sink_attention():
    """
    Monkey-patch transformers to use sink flash attention with s_aux support.

    Call this BEFORE initializing the verl trainer/model.

    Patches _flash_attention_forward at all locations where it's referenced:
    1. transformers.modeling_flash_attention_utils (primary)
    2. transformers.integrations.flash_attention (used by ALL_ATTENTION_FUNCTIONS)
    3. verl.models.transformers.monkey_patch (used by Ulysses SP wrapper)
    """
    global _original_flash_attention_forward

    # Guard: only patch once to avoid saving the patched function as "original"
    # (which would cause infinite recursion in fallback paths)
    if _original_flash_attention_forward is not None:
        return

    # Patch transformers' _flash_attention_forward
    import transformers.modeling_flash_attention_utils as fa_utils

    _original_flash_attention_forward = fa_utils._flash_attention_forward
    fa_utils._flash_attention_forward = _sink_flash_attention_forward

    # Also patch the module-level import that flash_attention_forward uses
    try:
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _sink_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    # Patch the reference in verl's monkey_patch module if already imported.
    # This ensures _ulysses_flash_attention_forward (Ulysses SP wrapper)
    # calls our kernel instead of the original FA.
    try:
        import verl.models.transformers.monkey_patch as verl_mp
        verl_mp._flash_attention_forward = _sink_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    print("[SinkAttention] Patched with s_aux-aware sink flash attention kernel")
    print("[SinkAttention]   - gpt-oss s_aux: extracted from kwargs per layer")
    print("[SinkAttention]   - sliding_window: respected per layer (None=full causal)")
    print("[SinkAttention]   - Ulysses SP: s_aux auto-sliced for local heads")


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

    try:
        import verl.models.transformers.monkey_patch as verl_mp
        verl_mp._flash_attention_forward = _original_flash_attention_forward
    except (ImportError, AttributeError):
        pass

    print("[SinkAttention] Restored original flash attention")
