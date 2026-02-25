# Sink Flash Attention: Design Document

## Problem Statement

Flash Attention 2 does not support attention sinks. Flash Attention 3 does, but
only runs on B200 GPUs. We need a Triton kernel that adds sink attention support
to A100-class GPUs for long-context training in verl with FSDP and sequence
parallelism.

## Background

### Attention Sinks (Xiao et al., 2023)

In autoregressive LLMs, the first few tokens receive disproportionately high
attention scores regardless of semantic relevance. These are "attention sinks."
When using sliding window attention, evicting sink tokens causes significant
quality degradation. The fix: always keep the first `k` tokens in the attention
window.

### Attention Pattern

For query at position `i`, it attends to:
```
sink region:   [0, num_sink)                              -- always
window region: [max(num_sink, i - window_size + 1), i]    -- causal sliding window
```

Visually (S=sink, W=window, .=masked):
```
Position:  0 1 2 3 4 5 6 7 8 9 ...
Query 0:   W
Query 1:   S W
Query 2:   S S W
Query 3:   S S . W
Query 4:   S S . . W
Query 5:   S S . . W W
Query 6:   S S . . . W W
Query 7:   S S . . . . W W
...
```
(num_sink=2, window_size=2)

## Architecture

### Kernel Design

The kernel follows the Flash Attention 2 algorithm (Dao, 2023) with two key
modifications:

1. **Two-phase KV iteration**: Instead of iterating over all KV blocks
   sequentially, the kernel processes KV blocks in two phases:
   - Phase 1: Sink blocks `[0, ceil(num_sink/BLOCK_N))`
   - Phase 2: Window blocks `[first_window_block, last_window_block]`
   Blocks between sink and window regions are skipped entirely, providing
   computational savings proportional to the gap size.

2. **Combined masking**: Each phase applies the appropriate mask:
   - Sink phase: `causal AND (key_pos < num_sink)`
   - Window phase: `causal AND (key_pos >= query_pos - window_size + 1)`

### Online Softmax Across Non-Contiguous Blocks

The online softmax algorithm naturally handles the gap between sink and window
regions. The running max `m_i` and sum `l_i` are maintained across both phases.
When processing resumes at the window blocks, the existing softmax state
correctly rescales the sink contributions.

### Three Kernels

1. **Forward kernel** (`_sink_flash_attn_fwd_kernel`): Computes output `O` and
   log-sum-exp `LSE` (needed for backward). Grid: `(ceil(N/BLOCK_M), B*H_q)`.

2. **Backward dK/dV kernel** (`_sink_flash_attn_bwd_dkdv_kernel`): For each KV
   block, iterates over Q blocks that attend to it. Grid:
   `(ceil(N/BLOCK_N), B*H_q)`.

3. **Backward dQ kernel** (`_sink_flash_attn_bwd_dq_kernel`): For each Q block,
   iterates over KV blocks it attends to (same two-phase pattern as forward).
   Grid: `(ceil(N/BLOCK_M), B*H_q)`.

### GQA/MQA Support

The kernel supports grouped-query attention natively. Q heads are mapped to KV
heads via `kv_head = q_head // (num_q_heads // num_kv_heads)`. In the backward
pass, dK and dV are computed per Q head and then summed across the group.

## Computational Complexity

Standard causal attention: `O(N^2 * D)` FLOPs per head.

Sink + sliding window attention: `O(N * (num_sink + window_size) * D)` FLOPs per
head. This is linear in `N` when `num_sink` and `window_size` are fixed constants.

For a sequence of length 32K with num_sink=4 and window_size=4096:
- Standard: 32K * 32K = 1B attention pairs
- Sink+window: 32K * 4100 = 131M attention pairs
- **~7.8x reduction** in compute

## Integration with verl FSDP + Sequence Parallelism

### Overview

verl uses FSDP for data parallelism with optional sequence parallelism (SP) for
long sequences. With SP, the sequence dimension `N` is split across `P` devices,
each holding a chunk of size `N/P`.

### Challenge

Sink tokens are at positions `[0, num_sink)`. With sequence parallelism, these
reside on rank 0 only. But ALL ranks need them for attention computation.

### Solution: Sink Token Broadcast

Before the attention computation:

```python
def prepare_sink_kv_for_sp(k, v, num_sink, sp_group):
    """
    Broadcast sink token KV pairs to all ranks in the SP group.

    Args:
        k, v: Local KV tensors [B, H, N_local, D]
        num_sink: Number of sink tokens
        sp_group: Process group for sequence parallelism

    Returns:
        k_with_sinks, v_with_sinks: KV tensors with sink tokens prepended
    """
    rank = torch.distributed.get_rank(sp_group)
    world_size = torch.distributed.get_world_size(sp_group)

    # Rank 0 extracts sink KV
    if rank == 0:
        sink_k = k[:, :, :num_sink].contiguous()
        sink_v = v[:, :, :num_sink].contiguous()
    else:
        sink_k = torch.empty(k.shape[0], k.shape[1], num_sink, k.shape[3],
                             device=k.device, dtype=k.dtype)
        sink_v = torch.empty_like(sink_k)

    # Broadcast from rank 0
    torch.distributed.broadcast(sink_k, src=0, group=sp_group)
    torch.distributed.broadcast(sink_v, src=0, group=sp_group)

    if rank == 0:
        # Rank 0 already has sinks in its local chunk
        return k, v
    else:
        # Other ranks prepend sink KV
        k_with_sinks = torch.cat([sink_k, k], dim=2)
        v_with_sinks = torch.cat([sink_v, v], dim=2)
        return k_with_sinks, v_with_sinks
```

Then adjust the kernel call: non-rank-0 devices have `num_sink` extra tokens at
the beginning of their local KV, and should set `num_sink=num_sink` in the kernel
call. The window is applied to the LOCAL positions after the prepended sinks.

### Ring Attention Integration

For ring attention (context parallelism), where KV chunks are passed around a
ring:

1. **Before the ring**: Extract sink KV from rank 0, broadcast to all ranks.
2. **During the ring**: Each rank always includes the sink KV in its attention
   computation, plus whatever KV chunk it currently holds from the ring.
3. **Kernel call**: For each ring step, call the kernel with the combined
   `[sink_kv, ring_chunk_kv]` as the KV input. Set `num_sink` appropriately.

### Gradient Accumulation

In the backward pass with SP, gradients for sink KV are computed on all ranks.
These must be all-reduced (summed) back to rank 0:

```python
def reduce_sink_kv_grads(dk, dv, num_sink, sp_group):
    """All-reduce sink KV gradients back to rank 0."""
    rank = torch.distributed.get_rank(sp_group)

    if rank != 0:
        sink_dk = dk[:, :, :num_sink].contiguous()
        sink_dv = dv[:, :, :num_sink].contiguous()
    else:
        sink_dk = dk[:, :, :num_sink].clone()
        sink_dv = dv[:, :, :num_sink].clone()

    torch.distributed.all_reduce(sink_dk, op=torch.distributed.ReduceOp.SUM, group=sp_group)
    torch.distributed.all_reduce(sink_dv, op=torch.distributed.ReduceOp.SUM, group=sp_group)

    if rank == 0:
        dk[:, :, :num_sink] = sink_dk
        dv[:, :, :num_sink] = sink_dv

    return dk, dv
```

## Performance Considerations

### Block Sizes

- Default: `BLOCK_M=64, BLOCK_N=64` (good for D <= 128)
- For D=256: `BLOCK_M=32, BLOCK_N=32` (to fit in SRAM)
- These match FA2's defaults for A100

### Memory

- Forward: O(N) for output + LSE (no N^2 materializations)
- Backward: O(N) for dQ, dK, dV, D_arr

### Potential Optimizations (Future Work)

1. **Block skipping in backward**: The dK/dV kernel currently iterates over all
   Q blocks for each KV block. For window blocks, only nearby Q blocks attend to
   them -- this loop can be bounded more tightly.

2. **Split-K for dK/dV with GQA**: When many Q heads map to one KV head,
   parallelize the accumulation across Q heads using atomic adds.

3. **Persistent kernels**: For very long sequences, a persistent kernel that
   reuses thread blocks across multiple Q rows could improve occupancy.

4. **Paged KV cache**: For inference with dynamic batching, support paged KV
   cache layouts (block tables).

## File Structure

```
sink_attention/
├── __init__.py                  # Public API
├── sink_flash_attention.py      # Triton kernels + PyTorch wrapper
├── test_sink_attention.py       # Correctness tests
└── docs/
    └── design.md                # This document
```

## Usage

```python
from sink_attention import sink_flash_attention

# Standard usage
output = sink_flash_attention(
    q,                # [B, H_q, N, D]
    k,                # [B, H_kv, N, D]
    v,                # [B, H_kv, N, D]
    num_sink=4,       # first 4 tokens are sinks
    window_size=4096, # sliding window of 4096
)

# Fully differentiable -- use in training
loss = output.sum()
loss.backward()  # dQ, dK, dV computed via Triton backward kernels
```
