"""
Triton FlashDecoding kernel with s_aux (learnable attention sink) support.

For decode (N_q=1, N_kv=large), parallelizes over KV blocks instead of Q blocks:
  Phase 1 (Triton): Each program computes partial softmax for one KV block
  Phase 2 (PyTorch): Reduces partial results, incorporating s_aux

This avoids materializing the full [B, H_q, 1, N_kv] attention matrix in HBM,
using only O(num_splits * D) intermediate storage per batch-head.

Memory footprint for B=1, H_q=64, N_kv=128K, D=128, BLOCK_N=256:
  M_partial: 64 * 512 * 4B = 128 KB
  L_partial: 128 KB
  O_partial: 64 * 512 * 128 * 4B = 16 MB
  Total: ~16.3 MB (vs ~50 MB for materialized attention matrix)
"""

import torch
import triton
import triton.language as tl
import math


# ============================================================================
# Phase 1: Split-KV Triton Kernel
# ============================================================================

@triton.jit
def _decode_split_kv_kernel(
    Q, K, V,
    M_partial, L_partial, O_partial,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_mb, stride_mh,
    stride_lb, stride_lh,
    stride_opb, stride_oph, stride_ops, stride_opd,
    N_KV,
    scale,
    NUM_KV_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
):
    """
    Phase 1: Each program computes partial attention for one KV block.

    Grid: (num_splits, B * H_q)

    For its assigned KV block [kv_start, kv_start + BLOCK_N):
      1. Load single query vector q [D]
      2. Load K block [BLOCK_N, D], compute scores [BLOCK_N]
      3. Compute partial softmax: m_partial, l_partial
      4. Load V block [BLOCK_N, D], compute weighted sum o_partial [D]
      5. Store (m_partial, l_partial, o_partial) for Phase 2 reduction
    """
    pid_split = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // NUM_Q_HEADS
    pid_h = pid_bh % NUM_Q_HEADS
    pid_kv_h = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    # Load single query vector [D]
    offs_d = tl.arange(0, D)
    q = tl.load(
        Q + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    ).to(tl.float32)
    q = q * scale

    # KV range for this split
    kv_start = pid_split * BLOCK_N
    offs_n = kv_start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_KV

    # Load K block [BLOCK_N, D]
    k_ptrs = (K + pid_b * stride_kb + pid_kv_h * stride_kh
              + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

    # Scores: [BLOCK_N] = sum_d(q[d] * k[n, d])
    scores = tl.sum(q[None, :] * k, axis=1)
    scores = tl.where(mask_n, scores, float('-inf'))

    # Partial softmax
    m_partial = tl.max(scores)
    m_safe = tl.where(m_partial > float('-inf'), m_partial, 0.0)
    p = tl.exp(scores - m_safe)
    p = tl.where(mask_n, p, 0.0)
    l_partial = tl.sum(p)

    # Weighted sum of V: [D]
    v_ptrs = (V + pid_b * stride_vb + pid_kv_h * stride_vh
              + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
    o_partial = tl.sum(p[:, None] * v, axis=0)  # [D]

    # Store partials
    # M_partial, L_partial: [B, H_q, num_splits] (contiguous, stride_ms=1)
    # O_partial: [B, H_q, num_splits, D]
    tl.store(
        M_partial + pid_b * stride_mb + pid_h * stride_mh + pid_split,
        m_partial,
    )
    tl.store(
        L_partial + pid_b * stride_lb + pid_h * stride_lh + pid_split,
        l_partial,
    )
    tl.store(
        O_partial + pid_b * stride_opb + pid_h * stride_oph
        + pid_split * stride_ops + offs_d * stride_opd,
        o_partial,
    )


# ============================================================================
# Public API
# ============================================================================

def sink_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    s_aux: torch.Tensor = None,
) -> torch.Tensor:
    """
    FlashDecoding-style single-query attention with s_aux support.

    Parallelizes over KV blocks (Phase 1 Triton kernel) then reduces
    partial results in PyTorch (Phase 2), incorporating s_aux as a
    virtual KV split with m=s_aux, l=1, o=0.

    Args:
        q: [B, H_q, 1, D] — single query token
        k: [B, H_kv, N_kv, D] — cached keys
        v: [B, H_kv, N_kv, D] — cached values
        s_aux: [H_q] — learnable attention sink per head, or None

    Returns:
        output: [B, H_q, 1, D]
    """
    B, H_q, N_q, D = q.shape
    H_kv = k.shape[1]
    N_kv = k.shape[2]

    assert N_q == 1, f"sink_decode_attention requires N_q=1, got {N_q}"
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be divisible by H_kv ({H_kv})"
    # D must be power of 2 for tl.arange
    assert D & (D - 1) == 0 and D >= 16, f"D={D} must be a power of 2 >= 16"

    scale = 1.0 / math.sqrt(D)

    # Choose BLOCK_N to fit K[BLOCK_N, D] + V[BLOCK_N, D] in SRAM
    # ~2 * BLOCK_N * D * 2 bytes (bf16/fp16)
    if D <= 64:
        BLOCK_N = 512
    elif D <= 128:
        BLOCK_N = 256
    else:
        BLOCK_N = 128

    num_splits = triton.cdiv(N_kv, BLOCK_N)

    # Squeeze q from [B, H_q, 1, D] to [B, H_q, D]
    q_flat = q.squeeze(2).contiguous()
    k = k.contiguous()
    v = v.contiguous()

    # Allocate partial buffers
    M_partial = torch.empty(B, H_q, num_splits, device=q.device, dtype=torch.float32)
    L_partial = torch.empty(B, H_q, num_splits, device=q.device, dtype=torch.float32)
    O_partial = torch.empty(B, H_q, num_splits, D, device=q.device, dtype=torch.float32)

    # ---- Phase 1: Split-KV Triton kernel ----
    grid = (num_splits, B * H_q)
    _decode_split_kv_kernel[grid](
        q_flat, k, v,
        M_partial, L_partial, O_partial,
        # Q strides (squeezed: [B, H_q, D])
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # M_partial strides ([B, H_q, num_splits])
        M_partial.stride(0), M_partial.stride(1),
        # L_partial strides
        L_partial.stride(0), L_partial.stride(1),
        # O_partial strides ([B, H_q, num_splits, D])
        O_partial.stride(0), O_partial.stride(1), O_partial.stride(2), O_partial.stride(3),
        # Runtime params
        N_KV=N_kv,
        scale=scale,
        # Constexpr params
        NUM_KV_HEADS=H_kv,
        NUM_Q_HEADS=H_q,
        BLOCK_N=BLOCK_N,
        D=D,
    )

    # ---- Phase 2: Reduction in PyTorch ----
    # Partial tensors are small: O(B * H_q * num_splits * D) floats
    # For B=1, H_q=64, num_splits=512, D=128: ~16 MB total

    # Incorporate s_aux as a virtual KV split: m=s_aux, l=1.0, o=0
    # This exactly matches the prefill kernel's initialization:
    #   m_i = s_aux, l_i = 1.0  →  exp(s_aux) in the softmax denominator
    if s_aux is not None:
        s_aux_f32 = s_aux.float()
        s_aux_m = s_aux_f32[None, :, None].expand(B, -1, 1)   # [B, H_q, 1]
        s_aux_l = torch.ones(B, H_q, 1, device=q.device, dtype=torch.float32)
        s_aux_o = torch.zeros(B, H_q, 1, D, device=q.device, dtype=torch.float32)
        M_partial = torch.cat([s_aux_m, M_partial], dim=2)
        L_partial = torch.cat([s_aux_l, L_partial], dim=2)
        O_partial = torch.cat([s_aux_o, O_partial], dim=2)

    # Stable online softmax reduction across all splits
    m_global = M_partial.max(dim=2, keepdim=True).values           # [B, H_q, 1]
    alpha = torch.exp(M_partial - m_global)                        # [B, H_q, S]
    alpha = alpha.masked_fill(M_partial == float('-inf'), 0.0)     # handle -inf
    L_scaled = L_partial * alpha                                   # [B, H_q, S]
    L_global = L_scaled.sum(dim=2, keepdim=True).clamp(min=1e-8)  # [B, H_q, 1]
    O_scaled = O_partial * alpha.unsqueeze(-1)                     # [B, H_q, S, D]
    O_global = O_scaled.sum(dim=2) / L_global.squeeze(2).unsqueeze(-1)  # [B, H_q, D]

    return O_global.unsqueeze(2).to(q.dtype)  # [B, H_q, 1, D]
