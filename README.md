# Sink Flash Attention

A Triton kernel implementing [Flash Attention](https://arxiv.org/abs/2307.08691) with [Attention Sink](https://arxiv.org/abs/2309.17453) support for training on A100/H200 GPUs.

Flash Attention 2 doesn't support sink attention. Flash Attention 3 does, but only runs on B200. This kernel fills the gap with a pure Triton implementation that works on any GPU Triton supports.

## TL;DR

```bash
pip install sink-flash-attention
```

**As a standalone kernel:**

```python
from sink_attention import sink_flash_attention

# Drop-in replacement for attention -- supports forward + backward
output = sink_flash_attention(
    q, k, v,          # [B, H, N, D] -- supports GQA (H_q != H_kv)
    num_sink=4,        # always attend to first 4 tokens
    window_size=4096,  # sliding window for the rest
)
```

**Monkey-patch verl / any HuggingFace model (one line):**

```python
from sink_attention import patch_verl_with_sink_attention
patch_verl_with_sink_attention(num_sink=4, window_size=4096)

# That's it. All attention layers now use sink attention.
# Works with verl GRPO, FSDP, Ulysses sequence parallelism.
# No other code changes needed.
```

## Why not Flash Attention 2?

FA2 does not implement sink attention. You cannot get sink attention behavior from FA2 -- it only supports full causal or sliding window (without sinks). Using FA2 with `sliding_window` will drop the sink tokens and degrade model quality.

FA3 supports sink attention but requires B200 GPUs. This kernel gives you sink attention on A100/H200.

## Installation

```bash
pip install sink-flash-attention
```

From source:

```bash
pip install git+https://github.com/RulinShao/sink-flash-attention-kernel.git
```

## What is Sink Attention?

In autoregressive LLMs, the first few tokens receive disproportionately high attention scores regardless of their semantic content ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)). These "attention sinks" are critical for model quality. Standard sliding window attention evicts them, causing quality degradation.

Sink attention fixes this by always keeping the first `k` tokens in the attention window:

```
For query at position i, attend to:
  - Tokens [0, num_sink)                           -- always (sinks)
  - Tokens [max(num_sink, i - window_size + 1), i] -- sliding window (causal)

Mask visualization (S=sink, W=window, .=masked, num_sink=2, window_size=3):

         Keys: 0 1 2 3 4 5 6 7 8 9
Query 0:       W
Query 1:       S W
Query 2:       S S W
Query 3:       S S W W
Query 4:       S S . W W
Query 5:       S S . . W W
Query 6:       S S . . W W W
Query 7:       S S . . . W W W
Query 8:       S S . . . . W W W
Query 9:       S S . . . . . W W W
```

## Design

### Two-Range Iteration

The core optimization: instead of iterating over all N/BLOCK_N KV blocks with runtime skip (O(N) loop iterations), the kernel uses two tight loops:

1. **Sink loop**: `range(NUM_SINK_BLOCKS)` -- processes `ceil(num_sink / BLOCK_N)` blocks
2. **Window loop**: `range(MAX_WINDOW_BLOCKS)` with runtime start/end -- processes `ceil((window_size + BLOCK_M) / BLOCK_N)` blocks

The gap between sink and window regions is never touched. Total work per query block: **O(num_sink + window_size)** instead of O(N).

```
KV blocks:  [SINK] [SINK] [....skip....] [WIN] [WIN] [WIN] [WIN]
                                ↑
                    Never loaded or processed
```

### Online Softmax Across Non-Contiguous Blocks

The online softmax (Milakov & Gimelshein, 2018) naturally handles the gap between sink and window regions. The running max `m_i` and sum `l_i` carry state from the sink phase into the window phase. NaN-safe guards handle edge cases where a block has no valid entries for some query rows (`-inf - (-inf)` would produce NaN without protection).

### Three Kernels

| Kernel | Grid | Description |
|--------|------|-------------|
| `_sink_flash_attn_fwd_kernel` | `(ceil(N/BLOCK_M), B*H_q)` | Forward pass. Two-range KV iteration. Stores output O and log-sum-exp LSE. |
| `_sink_flash_attn_bwd_dkdv_kernel` | `(ceil(N/BLOCK_N), B*H_q)` | Backward dK/dV. For each KV block, iterates over Q blocks that attend to it. Sink blocks iterate over all subsequent Q blocks; window blocks iterate over a bounded range. |
| `_sink_flash_attn_bwd_dq_kernel` | `(ceil(N/BLOCK_M), B*H_q)` | Backward dQ. Same two-range structure as forward. Applies `scale` factor at the end. |

### Unified Mask

All three kernels share a single mask function:

```python
valid(i, j) = causal(j <= i) AND (j < num_sink OR j >= i - window_size + 1)
```

This avoids the bug of treating sink and window as mutually exclusive regions -- a single KV block can contain both sink positions and window positions.

### Supported Configurations

- **Attention**: MHA, GQA, MQA (any H_q/H_kv ratio where H_q % H_kv == 0)
- **Dtypes**: fp16, bf16
- **Head dims**: 64, 128, 256
- **Edge cases**: num_sink=0 (pure sliding window), window_size=1 (sink + self only)

## Performance

Benchmarked on NVIDIA H200 (143GB), torch 2.10.0, triton 3.6.0.

Config: B=1, H_q=32, H_kv=8 (GQA 4:1), D=128, num_sink=4, window_size=4096.

### Forward Pass

![Forward pass latency and memory comparison](docs/performance.png)

### Training (Forward + Backward)

![Training latency and memory comparison](docs/training.png)

### Sink FA vs Eager (Materialized) Attention

Like Flash Attention, this kernel uses tiled online softmax and never materializes the N×N attention matrix. This gives O(N) memory instead of O(N^2).

| N | Sink FA | Eager Attention | Speedup | Mem Sink | Mem Eager |
|------:|--------:|--------------------:|--------:|---------:|-----------------:|
| 512 | 0.07 ms | 0.10 ms | 1.5x | 4 MB | 38 MB |
| 1,024 | 0.12 ms | 0.29 ms | 2.4x | 9 MB | 144 MB |
| 2,048 | 0.27 ms | 1.07 ms | 3.9x | 17 MB | 558 MB |
| 4,096 | 0.80 ms | 4.70 ms | 5.8x | 34 MB | **2,198 MB** |
| 8,192 | 2.09 ms | 20.20 ms | **9.5x** | 68 MB | **8,724 MB** |
| 16,384 | 4.67 ms | OOM | -- | 136 MB | OOM |
| 32,768 | 9.85 ms | OOM | -- | 273 MB | OOM |

At N=8192 the kernel uses **128x less memory** than eager attention.

### Sink FA vs Flash Attention 2 (via SDPA)

**FA2 does not support sink attention.** The comparison below is against FA2 running *full causal* attention -- a different and incorrect attention pattern for sink attention use cases. FA2 cannot be used as an alternative; it computes the wrong result. This comparison shows when our kernel (computing the correct sparse pattern) becomes faster than FA2 (computing the wrong dense pattern).

**Forward only:**

| N | Sink FA | FA2 (full causal) | vs FA2 | Note |
|------:|--------:|-----------------:|--------:|------|
| 4,096 | 0.76 ms | 0.52 ms | 0.7x | FA2 faster (CUDA vs Triton) |
| 8,192 | 1.88 ms | 1.75 ms | 0.9x | Approaching crossover |
| **16,384** | **4.18 ms** | **6.63 ms** | **1.6x** | **Sink FA wins** |
| **32,768** | **8.77 ms** | **25.83 ms** | **2.9x** | **Sink FA dominant** |

**Training (forward + backward):**

| N | Sink FA | FA2 (full causal) | vs FA2 | Mem Sink | Mem FA2 |
|------:|--------:|-----------------:|--------:|---------:|---------:|
| 4,096 | 2.81 ms | 2.09 ms | 0.7x | 269 MB | 337 MB |
| 8,192 | 7.28 ms | 6.89 ms | 0.9x | 538 MB | 673 MB |
| **16,384** | **16.39 ms** | **24.63 ms** | **1.5x** | **1,076 MB** | **1,346 MB** |

**Crossover at ~N=10-12K.** Below this, FA2's hand-optimized CUDA is faster per-FLOP despite doing O(N^2) work. Above this, the O(N * W) scaling wins.

### Scaling

| N | FLOPs reduction vs full causal | Measured speedup vs FA2 |
|------:|----------------------------:|----------------------:|
| 8,192 | 2.0x | 0.9x |
| 16,384 | 4.0x | 1.6x |
| 32,768 | 8.0x | 2.9x |
| 65,536 | 16.0x | ~5-6x (projected) |

### Numerical Accuracy

Compared against fp32 eager attention as the reference. The Triton kernel matches to within fp16/bf16 precision.

**Forward pass** (Triton fp16 vs eager fp32):

| Config | Max Abs Error | Mean Abs Error | Cosine Similarity |
|--------|-------------:|---------------:|------------------:|
| MHA fp16 N=256 D=64 | 9.77e-4 | 2.6e-5 | 1.00000 |
| MHA fp16 N=1024 D=64 | 9.77e-4 | 1.9e-5 | 1.00000 |
| MHA fp16 N=2048 D=64 | 9.77e-4 | 1.5e-5 | 1.00000 |
| GQA 4:1 fp16 D=128 N=512 | 1.95e-3 | 3.5e-5 | 1.00000 |
| MHA bf16 N=512 | 7.81e-3 | 2.0e-4 | 0.99999 |

**Backward pass** (Triton fp16 gradients vs eager fp32 gradients):

| Config | dQ Max Error | dK Max Error | dV Max Error |
|--------|------------:|------------:|------------:|
| MHA N=128 sink=4 win=32 | 1.66e-3 | 1.96e-3 | 1.94e-3 |
| GQA N=256 sink=4 win=64 | 1.17e-3 | 2.98e-3 | 4.16e-3 |
| MHA D=128 N=256 | 1.47e-3 | 1.94e-3 | 2.48e-3 |

All errors within expected fp16 precision bounds. Cosine similarity is 1.0 across all configurations.

## Sequence Parallelism (verl FSDP)

When using sequence parallelism, sink tokens (on rank 0) must be broadcast to all ranks:

```python
from sink_attention import SinkAttentionSPWrapper

attn = SinkAttentionSPWrapper(num_sink=4, window_size=4096, sp_group=sp_group)
output = attn(q_local, k_local, v_local)
```

Or use the lower-level functions:

```python
from sink_attention import prepare_sink_kv_for_sp, reduce_sink_kv_grads

# Forward: broadcast sink KV to all ranks
k_local, v_local = prepare_sink_kv_for_sp(k_local, v_local, num_sink, sp_group)
output = sink_flash_attention(q_local, k_local, v_local, num_sink, window_size)

# Backward: reduce sink KV gradients
dk_local, dv_local = reduce_sink_kv_grads(dk_local, dv_local, num_sink, sp_group)
```

See `docs/design.md` for details on ring attention integration.

## Running Tests

```bash
pip install -e .

# Basic correctness (requires GPU)
python tests/test_sink_attention.py

# Extended tests + benchmarks
python tests/benchmark.py
```

## Project Structure

```
sink_attention/
├── __init__.py               Public API
├── sink_flash_attention.py   Triton kernels + PyTorch autograd wrapper
├── sp_utils.py               Sequence parallelism utilities
└── verl_patch.py             verl/HuggingFace monkey patch
tests/
├── test_sink_attention.py    Correctness tests (11 configs)
├── benchmark.py              Extended tests (29 configs) + benchmarks
├── numerical_accuracy.py     Numerical accuracy measurement
└── tune_block_sizes.py       Block size tuning sweep
docs/
└── design.md                 Detailed design document
pyproject.toml                Package configuration
```

## Limitations and Future Work

- **Triton vs CUDA gap**: Our kernel is ~50-70% of FA2's per-FLOP throughput. A CUDA implementation would close this, moving the crossover from ~N=10-12K down to ~N=6-8K.
- **dK/dV backward for sink blocks**: Sink blocks iterate over all subsequent Q blocks. A segmented approach could bound this.
- **Block size tuning**: Tuned for H200 (BLOCK_M=64, BLOCK_N=32 for D=128). A100 may benefit from different settings.
- **Paged KV cache**: Not yet supported. Needed for inference with dynamic batching.

## References

- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (Xiao et al., 2023)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Dao, 2023)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018)
