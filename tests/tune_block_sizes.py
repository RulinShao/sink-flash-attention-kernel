"""
Block size tuning for H200/H100.

H200 has 228KB shared memory per SM (vs 164KB on A100).
Larger blocks can improve occupancy and reduce loop iterations.

This script sweeps BLOCK_M x BLOCK_N combinations and measures forward pass latency.
"""

import torch
import triton
import math
import time

from sink_attention.sink_flash_attention import (
    _sink_flash_attn_fwd_kernel,
    _build_sink_window_mask,
    _fwd_inner,
)


def benchmark_forward(B, H_q, H_kv, N, D, num_sink, window_size, BLOCK_M, BLOCK_N, warmup=5, repeat=20):
    """Benchmark forward pass with specific block sizes."""
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16).contiguous()
    o = torch.empty_like(q)
    lse = torch.empty(B, H_q, N, device=q.device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(D)
    NUM_SINK_BLOCKS = triton.cdiv(num_sink, BLOCK_N) if num_sink > 0 else 0
    MAX_WINDOW_BLOCKS = triton.cdiv(window_size + BLOCK_M, BLOCK_N)
    grid = (triton.cdiv(N, BLOCK_M), B * H_q)

    def run():
        _sink_flash_attn_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            lse.stride(0), lse.stride(1), lse.stride(2),
            N=N, D=D,
            num_sink=num_sink, window_size=window_size,
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            NUM_KV_HEADS=H_kv, NUM_Q_HEADS=H_q,
            NUM_SINK_BLOCKS=NUM_SINK_BLOCKS,
            MAX_WINDOW_BLOCKS=MAX_WINDOW_BLOCKS,
        )

    # Warmup (also triggers JIT compilation)
    try:
        for _ in range(warmup):
            run()
        torch.cuda.synchronize()
    except Exception as e:
        return None  # Block size not supported (e.g., too large for SRAM)

    # Benchmark
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SRAM per SM: {torch.cuda.get_device_properties(0).shared_memory_per_block_optin // 1024}KB")
    print()

    B, H_q, H_kv = 1, 32, 8
    num_sink, window_size = 4, 4096

    # Block size candidates
    block_configs = [
        # (BLOCK_M, BLOCK_N)
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
    ]

    for D in [128]:
        for N in [4096, 8192, 16384, 32768]:
            print(f"D={D}, N={N}, sink={num_sink}, window={window_size}")
            print(f"  {'BLOCK_M':>7} x {'BLOCK_N':>7} | {'Time (ms)':>10} | {'vs best':>8}")
            print(f"  {'-'*7}-x-{'-'*7}-+-{'-'*10}-+-{'-'*8}")

            results = []
            for bm, bn in block_configs:
                t = benchmark_forward(B, H_q, H_kv, N, D, num_sink, window_size, bm, bn)
                if t is not None:
                    results.append((bm, bn, t))

            if not results:
                print("  No valid configurations found")
                continue

            best_time = min(r[2] for r in results)
            for bm, bn, t in results:
                ratio = t / best_time
                marker = " <-- best" if t == best_time else ""
                print(f"  {bm:>7} x {bn:>7} | {t:10.3f} | {ratio:7.2f}x{marker}")
            print()
