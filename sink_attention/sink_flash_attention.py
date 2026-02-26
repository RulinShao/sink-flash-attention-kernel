"""
Sink Flash Attention: A Triton kernel for Flash Attention with Attention Sink support.

Implements the attention pattern from "Efficient Streaming Language Models with Attention Sinks"
(Xiao et al., 2023) as a fused Triton kernel compatible with A100-class GPUs (where FA2 runs).

Attention pattern for query at position i:
  - Always attend to tokens [0, num_sink)           (sink tokens)
  - Attend to tokens [max(num_sink, i - W + 1), i]  (sliding window, causal)

The per-element mask is: valid(i, j) = causal(j <= i) AND (j < num_sink OR j >= i - W + 1)

v2: Two-range iteration -- processes only sink blocks + window blocks,
    skipping the gap entirely. O(num_sink + window_size) work per query block
    instead of O(N) loop iterations with runtime skip.

Supports:
  - Forward + backward pass (training)
  - MHA, GQA, MQA
  - fp16, bf16
  - Causal masking
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _build_sink_window_mask(
    offs_m, offs_n, mask_m, mask_n,
    num_sink: tl.constexpr, window_size: tl.constexpr,
):
    """Build the combined sink + sliding window + causal mask."""
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    sink_mask = offs_n[None, :] < num_sink
    window_mask = offs_n[None, :] >= (offs_m[:, None] - window_size + 1)
    return causal_mask & (sink_mask | window_mask) & mask_m[:, None] & mask_n[None, :]


@triton.jit
def _fwd_inner(
    q, k_base, v_base, acc, m_i, l_i,
    offs_m, offs_d, mask_m,
    stride_kn, stride_kd, stride_vn, stride_vd,
    block_n_start, block_n_end,
    num_sink: tl.constexpr, window_size: tl.constexpr,
    N: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_BLOCKS_LIMIT: tl.constexpr,
):
    """Process a range of KV blocks in the forward pass."""
    for block_idx in range(NUM_BLOCKS_LIMIT):
        block_n = block_n_start + block_idx
        if block_n < block_n_end:
            kv_start = block_n * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k))
            valid_mask = _build_sink_window_mask(offs_m, offs_n, mask_m, mask_n,
                                                  num_sink, window_size)
            s = tl.where(valid_mask, s, float('-inf'))

            # Online softmax with NaN-safe handling
            row_max = tl.max(s, axis=1)
            has_valid = row_max > float('-inf')
            m_new = tl.where(has_valid, tl.maximum(m_i, row_max), m_i)
            alpha = tl.where(m_i > float('-inf'), tl.exp(m_i - m_new), 0.0)
            m_new_safe = tl.where(m_new > float('-inf'), m_new, 0.0)
            p = tl.exp(s - m_new_safe[:, None])
            p = tl.where(valid_mask, p, 0.0)

            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
            acc += tl.dot(p.to(v.dtype), v)

            m_i = m_new

    return acc, m_i, l_i


# ============================================================================
# Forward Kernel
# ============================================================================

@triton.jit
def _sink_flash_attn_fwd_kernel(
    Q, K, V, O,
    LSE,
    S_AUX,  # Learnable sink scalar per Q-head [H_q], or null pointer
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_lseb, stride_lseh, stride_lsen,
    N: tl.constexpr,
    D: tl.constexpr,
    num_sink: tl.constexpr,
    window_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SINK_BLOCKS: tl.constexpr,
    MAX_WINDOW_BLOCKS: tl.constexpr,
    USE_S_AUX: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // NUM_Q_HEADS
    pid_h = pid_bh % NUM_Q_HEADS
    pid_kv_h = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N
    q_start = pid_m * BLOCK_M

    # Load Q
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    # Default initialisation for online softmax (no s_aux)
    m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # s_aux initialization: pre-load the learnable sink scalar into the
    # online softmax state so it contributes exp(s_aux) to the denominator
    # without contributing to the weighted sum of V (no associated value).
    if USE_S_AUX:
        s_aux_val = tl.load(S_AUX + pid_h)  # scalar per Q head
        # Override: set m_i = s_aux, l_i = 1.0  (exp(s_aux - s_aux) = 1)
        m_i = tl.full([BLOCK_M], value=0.0, dtype=tl.float32) + s_aux_val
        l_i = tl.full([BLOCK_M], value=1.0, dtype=tl.float32)

    k_base = K + pid_b * stride_kb + pid_kv_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_kv_h * stride_vh

    # ---- Range 1: Sink blocks [0, NUM_SINK_BLOCKS) ----
    acc, m_i, l_i = _fwd_inner(
        q, k_base, v_base, acc, m_i, l_i,
        offs_m, offs_d, mask_m,
        stride_kn, stride_kd, stride_vn, stride_vd,
        0, NUM_SINK_BLOCKS,
        num_sink, window_size, N, BLOCK_N,
        NUM_SINK_BLOCKS,
    )

    # ---- Range 2: Window blocks ----
    # Window start for earliest query in this block
    win_key_start = q_start - window_size + 1
    # Clamp to not overlap with sink region
    win_key_start = tl.maximum(win_key_start, num_sink)
    # First KV block in window
    win_block_start = win_key_start // BLOCK_N
    # Don't re-process sink blocks
    win_block_start = tl.maximum(win_block_start, NUM_SINK_BLOCKS)
    # Last KV block: contains q_start + BLOCK_M - 1 (last query, causal)
    win_block_end = (tl.minimum(q_start + BLOCK_M, N) - 1) // BLOCK_N + 1

    acc, m_i, l_i = _fwd_inner(
        q, k_base, v_base, acc, m_i, l_i,
        offs_m, offs_d, mask_m,
        stride_kn, stride_kd, stride_vn, stride_vd,
        win_block_start, win_block_end,
        num_sink, window_size, N, BLOCK_N,
        MAX_WINDOW_BLOCKS,
    )

    # Normalize
    l_i = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_i[:, None]

    # Store O
    o_ptrs = O + pid_b * stride_ob + pid_h * stride_oh + \
             offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])

    # Store LSE
    lse = m_i + tl.log(l_i)
    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsen
    tl.store(lse_ptrs, lse, mask=mask_m)


# ============================================================================
# Backward Kernel - dK, dV
# ============================================================================

@triton.jit
def _sink_flash_attn_bwd_dkdv_kernel(
    Q, K, V, O, DO, DK, DV,
    LSE, D_arr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N: tl.constexpr,
    D: tl.constexpr,
    num_sink: tl.constexpr,
    window_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_Q_BLOCKS: tl.constexpr,
):
    """Compute dK and dV. Grid: (num_kv_blocks, B * H_q).

    For each KV block, we iterate over Q blocks that attend to it.
    - If this is a sink block: all Q blocks from (kv_block) to end attend to it.
    - If this is a window block: only Q blocks within window_size attend to it.
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // NUM_Q_HEADS
    pid_h = pid_bh % NUM_Q_HEADS
    pid_kv_h = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    mask_n = offs_n < N
    kv_start = pid_n * BLOCK_N

    k_ptrs = K + pid_b * stride_kb + pid_kv_h * stride_kh + \
             offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_b * stride_vb + pid_kv_h * stride_vh + \
             offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)

    is_sink_block = kv_start < num_sink

    # Determine Q block range that attends to this KV block
    if is_sink_block:
        # Sink blocks: all Q blocks where q >= kv_start (causal)
        q_block_start = kv_start // BLOCK_M
        q_block_end = NUM_Q_BLOCKS
    else:
        # Window blocks: Q blocks where kv_start is in their window
        # A query q attends to key k if: k >= q - window_size + 1 AND k <= q
        # For KV block [kv_start, kv_start+BLOCK_N-1], the max q that can attend
        # to the last key is: kv_start + BLOCK_N - 1 + window_size - 1
        q_block_start = kv_start // BLOCK_M
        max_q = kv_start + BLOCK_N + window_size - 2
        q_block_end = tl.minimum(max_q // BLOCK_M + 1, NUM_Q_BLOCKS)

    for block_m in range(NUM_Q_BLOCKS):
        if block_m >= q_block_start and block_m < q_block_end:
            q_start = block_m * BLOCK_M
            offs_m = q_start + tl.arange(0, BLOCK_M)
            mask_m = offs_m < N

            q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
                     offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
            q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            q = (q * scale).to(q.dtype)

            do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + \
                      offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
            do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

            lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsen
            lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

            d_ptrs = D_arr + pid_b * stride_db + pid_h * stride_dh + offs_m * stride_dn
            Di = tl.load(d_ptrs, mask=mask_m, other=0.0)

            s = tl.dot(q, tl.trans(k))
            valid_mask = _build_sink_window_mask(offs_m, offs_n, mask_m, mask_n,
                                                  num_sink, window_size)

            p = tl.exp(s - lse[:, None])
            p = tl.where(valid_mask, p, 0.0)

            dv += tl.dot(tl.trans(p.to(do.dtype)), do)

            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - Di[:, None])
            ds = tl.where(valid_mask, ds, 0.0)

            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)

    dk_ptrs = DK + pid_b * stride_dkb + pid_h * stride_dkh + \
              offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + pid_b * stride_dvb + pid_h * stride_dvh + \
              offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk.to(DK.dtype.element_ty), mask=mask_n[:, None])
    tl.store(dv_ptrs, dv.to(DV.dtype.element_ty), mask=mask_n[:, None])


# ============================================================================
# Backward Kernel - dQ
# ============================================================================

@triton.jit
def _sink_flash_attn_bwd_dq_kernel(
    Q, K, V, O, DO, DQ,
    LSE, D_arr,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_dob, stride_doh, stride_don, stride_dod,
    stride_dqb, stride_dqh, stride_dqn, stride_dqd,
    stride_lseb, stride_lseh, stride_lsen,
    stride_db, stride_dh, stride_dn,
    N: tl.constexpr,
    D: tl.constexpr,
    num_sink: tl.constexpr,
    window_size: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_SINK_BLOCKS: tl.constexpr,
    MAX_WINDOW_BLOCKS: tl.constexpr,
):
    """Compute dQ. Grid: (num_q_blocks, B * H_q).
    Same two-range structure as forward.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // NUM_Q_HEADS
    pid_h = pid_bh % NUM_Q_HEADS
    pid_kv_h = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mask_m = offs_m < N
    q_start = pid_m * BLOCK_M

    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + \
             offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    q = (q * scale).to(q.dtype)

    do_ptrs = DO + pid_b * stride_dob + pid_h * stride_doh + \
              offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

    lse_ptrs = LSE + pid_b * stride_lseb + pid_h * stride_lseh + offs_m * stride_lsen
    lse = tl.load(lse_ptrs, mask=mask_m, other=0.0)

    d_ptrs = D_arr + pid_b * stride_db + pid_h * stride_dh + offs_m * stride_dn
    Di = tl.load(d_ptrs, mask=mask_m, other=0.0)

    dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    k_base = K + pid_b * stride_kb + pid_kv_h * stride_kh
    v_base = V + pid_b * stride_vb + pid_kv_h * stride_vh

    # ---- Range 1: Sink blocks ----
    for block_idx in range(NUM_SINK_BLOCKS):
        kv_start = block_idx * BLOCK_N
        offs_n = kv_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k))
        valid_mask = _build_sink_window_mask(offs_m, offs_n, mask_m, mask_n,
                                              num_sink, window_size)
        p = tl.exp(s - lse[:, None])
        p = tl.where(valid_mask, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - Di[:, None])
        ds = tl.where(valid_mask, ds, 0.0)
        dq += tl.dot(ds.to(k.dtype), k)

    # ---- Range 2: Window blocks ----
    win_key_start = q_start - window_size + 1
    win_key_start = tl.maximum(win_key_start, num_sink)
    win_block_start = win_key_start // BLOCK_N
    win_block_start = tl.maximum(win_block_start, NUM_SINK_BLOCKS)
    win_block_end = (tl.minimum(q_start + BLOCK_M, N) - 1) // BLOCK_N + 1

    for block_idx in range(MAX_WINDOW_BLOCKS):
        block_n = win_block_start + block_idx
        if block_n < win_block_end:
            kv_start = block_n * BLOCK_N
            offs_n = kv_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < N

            k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
            v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k))
            valid_mask = _build_sink_window_mask(offs_m, offs_n, mask_m, mask_n,
                                                  num_sink, window_size)
            p = tl.exp(s - lse[:, None])
            p = tl.where(valid_mask, p, 0.0)
            dp = tl.dot(do, tl.trans(v))
            ds = p * (dp - Di[:, None])
            ds = tl.where(valid_mask, ds, 0.0)
            dq += tl.dot(ds.to(k.dtype), k)

    # Apply scale
    dq = dq * scale
    dq_ptrs = DQ + pid_b * stride_dqb + pid_h * stride_dqh + \
              offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=mask_m[:, None])


# ============================================================================
# PyTorch Autograd Wrapper
# ============================================================================

class SinkFlashAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, num_sink, window_size, s_aux=None):
        B, H_q, N, D = q.shape
        H_kv = k.shape[1]
        assert k.shape == (B, H_kv, N, D)
        assert v.shape == (B, H_kv, N, D)
        assert H_q % H_kv == 0

        use_s_aux = s_aux is not None
        if use_s_aux:
            assert s_aux.shape == (H_q,), f"s_aux shape must be [H_q={H_q}], got {s_aux.shape}"
            s_aux = s_aux.contiguous().float()

        scale = 1.0 / math.sqrt(D)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        o = torch.empty_like(q)
        lse = torch.empty(B, H_q, N, device=q.device, dtype=torch.float32)

        # Block sizes -- tuned on H200 (227KB SRAM per SM)
        # Smaller BLOCK_N gives better parallelism for forward
        BLOCK_M = 64
        BLOCK_N = 32
        if D <= 64:
            BLOCK_M = 128
            BLOCK_N = 64

        # Backward uses larger BLOCK_N for fewer loop iterations in dK/dV
        BLOCK_M_BWD = BLOCK_M
        BLOCK_N_BWD = 64
        if D <= 64:
            BLOCK_N_BWD = 64

        NUM_SINK_BLOCKS = triton.cdiv(num_sink, BLOCK_N) if num_sink > 0 else 0
        # Max window blocks any Q block needs: ceil((window_size + BLOCK_M) / BLOCK_N)
        MAX_WINDOW_BLOCKS = triton.cdiv(window_size + BLOCK_M, BLOCK_N)

        grid = (triton.cdiv(N, BLOCK_M), B * H_q)

        # S_AUX pointer: use the actual tensor if provided, else pass a dummy
        s_aux_ptr = s_aux if use_s_aux else q  # dummy pointer, won't be loaded

        _sink_flash_attn_fwd_kernel[grid](
            q, k, v, o, lse,
            s_aux_ptr,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N=N, D=D,
            num_sink=num_sink,
            window_size=window_size,
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_KV_HEADS=H_kv, NUM_Q_HEADS=H_q,
            NUM_SINK_BLOCKS=NUM_SINK_BLOCKS,
            MAX_WINDOW_BLOCKS=MAX_WINDOW_BLOCKS,
            USE_S_AUX=use_s_aux,
        )

        ctx.save_for_backward(q, k, v, o, lse, s_aux if use_s_aux else torch.empty(0, device=q.device))
        ctx.num_sink = num_sink
        ctx.window_size = window_size
        ctx.scale = scale
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        ctx.BLOCK_M_BWD = BLOCK_M_BWD
        ctx.BLOCK_N_BWD = BLOCK_N_BWD
        ctx.use_s_aux = use_s_aux

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, s_aux_saved = ctx.saved_tensors
        num_sink = ctx.num_sink
        window_size = ctx.window_size
        scale = ctx.scale
        use_s_aux = ctx.use_s_aux
        BLOCK_M = ctx.BLOCK_M_BWD
        BLOCK_N = ctx.BLOCK_N_BWD

        B, H_q, N, D = q.shape
        H_kv = k.shape[1]

        do = do.contiguous()
        D_arr = (do.float() * o.float()).sum(dim=-1)  # [B, H_q, N]

        dq = torch.empty_like(q)
        dk = torch.empty(B, H_q, N, D, device=k.device, dtype=k.dtype)
        dv = torch.empty(B, H_q, N, D, device=v.device, dtype=v.dtype)

        NUM_KV_BLOCKS = triton.cdiv(N, BLOCK_N)
        NUM_Q_BLOCKS = triton.cdiv(N, BLOCK_M)
        NUM_SINK_BLOCKS = triton.cdiv(num_sink, BLOCK_N) if num_sink > 0 else 0
        MAX_WINDOW_BLOCKS = triton.cdiv(window_size + BLOCK_M, BLOCK_N)

        # dK, dV
        grid_dkdv = (NUM_KV_BLOCKS, B * H_q)
        _sink_flash_attn_bwd_dkdv_kernel[grid_dkdv](
            q, k, v, o, do, dk, dv,
            lse, D_arr,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D_arr.stride(0), D_arr.stride(1), D_arr.stride(2),
            N=N, D=D,
            num_sink=num_sink, window_size=window_size,
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_KV_HEADS=H_kv, NUM_Q_HEADS=H_q,
            NUM_Q_BLOCKS=NUM_Q_BLOCKS,
        )

        # dQ
        grid_dq = (NUM_Q_BLOCKS, B * H_q)
        _sink_flash_attn_bwd_dq_kernel[grid_dq](
            q, k, v, o, do, dq,
            lse, D_arr,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            D_arr.stride(0), D_arr.stride(1), D_arr.stride(2),
            N=N, D=D,
            num_sink=num_sink, window_size=window_size,
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_KV_HEADS=H_kv, NUM_Q_HEADS=H_q,
            NUM_SINK_BLOCKS=NUM_SINK_BLOCKS,
            MAX_WINDOW_BLOCKS=MAX_WINDOW_BLOCKS,
        )

        if H_q != H_kv:
            groups = H_q // H_kv
            dk = dk.view(B, H_kv, groups, N, D).sum(dim=2)
            dv = dv.view(B, H_kv, groups, N, D).sum(dim=2)

        # Gradient for s_aux (learnable sink parameter)
        # The s_aux contributes exp(s_aux_h) / sum_j(exp(s_j)) to the softmax
        # normalization. Its gradient is:
        #   ds_aux_h = -sum_{b,n} exp(s_aux_h - lse_{b,h,n}) * D_{b,h,n}
        # where D_{b,h,n} = sum_d(dO_{b,h,n,d} * O_{b,h,n,d})
        ds_aux = None
        if use_s_aux:
            s_aux = s_aux_saved  # [H_q]
            # exp(s_aux_h - lse_{b,h,n}): softmax weight of the sink
            # Shape: s_aux[H_q] -> [1, H_q, 1], lse[B, H_q, N]
            sink_prob = torch.exp(s_aux[None, :, None] - lse)  # [B, H_q, N]
            # ds_aux_h = -sum_{b,n} sink_prob_{b,h,n} * D_{b,h,n}
            ds_aux = -(sink_prob * D_arr).sum(dim=(0, 2))  # [H_q]

        return dq, dk, dv, None, None, ds_aux


def sink_flash_attention(q, k, v, num_sink=4, window_size=512, s_aux=None):
    """
    Flash Attention with Attention Sink support.

    Args:
        q: Query tensor [B, H_q, N, D]
        k: Key tensor [B, H_kv, N, D]
        v: Value tensor [B, H_kv, N, D]
        num_sink: Number of sink tokens (default: 4)
        window_size: Sliding window size (default: 512)
        s_aux: Optional learnable sink scalar per Q-head [H_q].
               When provided, initializes the online softmax with
               m_i = s_aux, l_i = 1, causing exp(s_aux) to contribute
               to the softmax denominator without affecting the value sum.
               This implements the gpt-oss "attention sink" mechanism.

    Returns:
        Output tensor [B, H_q, N, D]
    """
    return SinkFlashAttentionFunc.apply(q, k, v, num_sink, window_size, s_aux)
