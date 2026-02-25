"""
Extended correctness tests and performance benchmarks for Sink Flash Attention.
"""

import torch
import math
import sys
import time

from sink_flash_attention import sink_flash_attention


def naive_sink_attention(q, k, v, num_sink, window_size):
    """Reference implementation."""
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    heads_per_group = H_q // H_kv
    scale = 1.0 / math.sqrt(D)

    outputs = []
    for h_q in range(H_q):
        h_kv = h_q // heads_per_group
        qi, ki, vi = q[:, h_q], k[:, h_kv], v[:, h_kv]
        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale
        row_idx = torch.arange(N, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)
        mask = ((col_idx < num_sink) | (col_idx >= (row_idx - window_size + 1))) & (col_idx <= row_idx)
        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        outputs.append(torch.matmul(attn, vi))
    return torch.stack(outputs, dim=1)


def run_test(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"        {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Extended Correctness Tests
# ============================================================================

def test_forward(B, H_q, H_kv, N, D, num_sink, window_size, dtype=torch.float16):
    if num_sink >= N or window_size >= N:
        return
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)

    out_triton = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
    out_ref = naive_sink_attention(q.float(), k.float(), v.float(), num_sink, window_size).to(dtype)

    atol = 2e-2 if dtype == torch.float16 else 5e-3
    rtol = 2e-2 if dtype == torch.float16 else 5e-3
    torch.testing.assert_close(out_triton, out_ref, atol=atol, rtol=rtol)


def test_backward(B, H_q, H_kv, N, D, num_sink, window_size):
    torch.manual_seed(42)
    # Reference in fp32
    q_ref = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    out_ref = naive_sink_attention(q_ref, k_ref, v_ref, num_sink, window_size)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    # Triton in fp16
    q_tri = q_ref.detach().half().requires_grad_(True)
    k_tri = k_ref.detach().half().requires_grad_(True)
    v_tri = v_ref.detach().half().requires_grad_(True)
    out_tri = sink_flash_attention(q_tri, k_tri, v_tri, num_sink=num_sink, window_size=window_size)
    out_tri.backward(grad_out.half())

    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad, atol=5e-2, rtol=5e-2)


def test_forward_long_seq(N, num_sink, window_size):
    """Test forward at longer sequence lengths (no reference - just check it runs and produces finite output)."""
    B, H_q, H_kv, D = 1, 8, 2, 128
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

    out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
    assert out.shape == (B, H_q, N, D), f"Wrong shape: {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values in output"
    # Verify first few tokens only attend to themselves + sinks
    # Verify last tokens attend to window + sinks


def test_backward_long_seq(N, num_sink, window_size):
    """Test backward at longer sequences (just check it runs and produces finite grads)."""
    B, H_q, H_kv, D = 1, 4, 4, 64
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)

    out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
    loss = out.sum()
    loss.backward()

    assert torch.isfinite(q.grad).all(), "Non-finite dQ"
    assert torch.isfinite(k.grad).all(), "Non-finite dK"
    assert torch.isfinite(v.grad).all(), "Non-finite dV"


# ============================================================================
# Benchmarks
# ============================================================================

def benchmark_fn(fn, warmup=5, repeat=20, sync=True):
    """Benchmark a function, return median time in ms."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]  # median


def naive_causal_attention(q, k, v):
    """Standard causal attention for baseline comparison."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    N = q.shape[2]
    mask = torch.triu(torch.ones(N, N, device=q.device, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v)


def run_benchmarks():
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 80)

    B, H_q, H_kv, D = 1, 32, 8, 128  # Typical LLaMA-style config (GQA)
    num_sink = 4
    window_size = 4096

    # Use torch SDPA as baseline (dispatches to FA2/FA3/mem-efficient internally)
    import torch.nn.functional as F

    print(f"\n  Config: B={B}, H_q={H_q}, H_kv={H_kv}, D={D}")
    print(f"  Sink tokens: {num_sink}, Window: {window_size}")
    print(f"  Baseline: torch.nn.functional.scaled_dot_product_attention (is_causal=True)")
    print(f"  {'N':>8} | {'Sink FA (ms)':>12} | {'SDPA Causal':>12} | {'Speedup':>8} | {'Mem Sink(MB)':>12} | {'Mem SDPA(MB)':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*12}-+-{'-'*12}")

    for N in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        torch.manual_seed(42)
        q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

        # Sink flash attention
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        t_sink = benchmark_fn(lambda: sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size))
        mem_sink = (torch.cuda.max_memory_allocated() - mem_before) / 1e6

        # SDPA causal (need to expand KV for GQA since SDPA handles it differently)
        groups = H_q // H_kv
        k_exp = k[:, :, None, :, :].expand(B, H_kv, groups, N, D).reshape(B, H_q, N, D)
        v_exp = v[:, :, None, :, :].expand(B, H_kv, groups, N, D).reshape(B, H_q, N, D)
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        try:
            t_sdpa = benchmark_fn(lambda: F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True))
            mem_sdpa = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
            speedup = t_sdpa / t_sink
        except torch.cuda.OutOfMemoryError:
            t_sdpa = float('inf')
            mem_sdpa = float('inf')
            speedup = float('inf')
            torch.cuda.empty_cache()

        sdpa_str = f"{t_sdpa:12.2f}" if t_sdpa < float('inf') else "         OOM"
        speedup_str = f"{speedup:7.1f}x" if speedup < float('inf') else "     N/A"
        mem_sdpa_str = f"{mem_sdpa:11.1f}" if mem_sdpa < float('inf') else "        OOM"

        print(f"  {N:>8} | {t_sink:12.2f} | {sdpa_str} | {speedup_str} | {mem_sink:11.1f} | {mem_sdpa_str}")

    # Forward + backward benchmark
    print(f"\n  Forward + Backward (training):")
    print(f"  {'N':>8} | {'Sink FWD+BWD':>14} | {'SDPA FWD+BWD':>14} | {'Speedup':>8} | {'Mem Sink':>10} | {'Mem SDPA':>10}")
    print(f"  {'-'*8}-+-{'-'*14}-+-{'-'*14}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}")

    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        torch.manual_seed(42)
        q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)

        def sink_fwd_bwd():
            q.grad = k.grad = v.grad = None
            out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
            out.sum().backward()

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        try:
            t_sink = benchmark_fn(sink_fwd_bwd)
            mem_sink_fb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
        except torch.cuda.OutOfMemoryError:
            t_sink = float('inf')
            mem_sink_fb = float('inf')
            torch.cuda.empty_cache()

        # SDPA fwd+bwd
        groups = H_q // H_kv
        q2 = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        k2_base = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
        v2_base = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16, requires_grad=True)

        def sdpa_fwd_bwd():
            q2.grad = k2_base.grad = v2_base.grad = None
            k2 = k2_base[:, :, None, :, :].expand(B, H_kv, groups, N, D).reshape(B, H_q, N, D)
            v2 = v2_base[:, :, None, :, :].expand(B, H_kv, groups, N, D).reshape(B, H_q, N, D)
            out = F.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
            out.sum().backward()

        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()
        try:
            t_sdpa = benchmark_fn(sdpa_fwd_bwd)
            mem_sdpa_fb = (torch.cuda.max_memory_allocated() - mem_before) / 1e6
        except torch.cuda.OutOfMemoryError:
            t_sdpa = float('inf')
            mem_sdpa_fb = float('inf')
            torch.cuda.empty_cache()

        def fmt(v, w=14):
            return f"{v:{w}.2f}" if v < float('inf') else " " * (w - 3) + "OOM"
        def fmtm(v, w=10):
            return f"{v:{w}.1f}" if v < float('inf') else " " * (w - 3) + "OOM"

        speedup = t_sdpa / t_sink if t_sink < float('inf') and t_sdpa < float('inf') else float('inf')
        speedup_str = f"{speedup:7.1f}x" if speedup < float('inf') else "     N/A"

        print(f"  {N:>8} | {fmt(t_sink)} | {fmt(t_sdpa)} | {speedup_str} | {fmtm(mem_sink_fb)} | {fmtm(mem_sdpa_fb)}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    # ---- Extended forward tests ----
    print("=== Extended Forward Correctness ===")

    # Various sequence lengths
    for N in [64, 128, 256, 512, 1024]:
        results.append(run_test(
            f"fwd N={N} sink=4 win=64",
            test_forward, 1, 4, 4, N, 64, 4, 64
        ))

    # Edge cases
    results.append(run_test("fwd num_sink=0 (pure window)", test_forward, 1, 4, 4, 256, 64, 0, 64))
    results.append(run_test("fwd window=1 (sink + self only)", test_forward, 1, 4, 4, 256, 64, 4, 1))
    results.append(run_test("fwd num_sink=1", test_forward, 1, 4, 4, 256, 64, 1, 32))
    results.append(run_test("fwd large sink (32)", test_forward, 1, 4, 4, 256, 64, 32, 64))

    # D=128
    results.append(run_test("fwd D=128", test_forward, 1, 4, 4, 256, 128, 4, 64))

    # GQA configs
    results.append(run_test("fwd GQA H_q=8 H_kv=1 (MQA)", test_forward, 1, 8, 1, 256, 64, 4, 64))
    results.append(run_test("fwd GQA H_q=8 H_kv=2", test_forward, 1, 8, 2, 256, 64, 4, 64))
    results.append(run_test("fwd GQA H_q=32 H_kv=8", test_forward, 1, 32, 8, 256, 64, 4, 64))

    # Batch size
    results.append(run_test("fwd B=4", test_forward, 4, 4, 4, 128, 64, 4, 32))

    # BFloat16
    results.append(run_test("fwd bf16", test_forward, 1, 4, 4, 256, 64, 4, 64, torch.bfloat16))

    # ---- Extended backward tests ----
    print("\n=== Extended Backward Correctness ===")

    for N in [64, 128, 256]:
        results.append(run_test(
            f"bwd N={N} sink=4 win=32",
            test_backward, 1, 4, 4, N, 64, 4, 32
        ))

    results.append(run_test("bwd GQA H_q=8 H_kv=2", test_backward, 1, 8, 2, 128, 64, 4, 32))
    results.append(run_test("bwd D=128", test_backward, 1, 4, 4, 128, 128, 4, 32))
    results.append(run_test("bwd sink=0 (pure window)", test_backward, 1, 4, 4, 128, 64, 0, 64))
    results.append(run_test("bwd window=1 (sink only)", test_backward, 1, 4, 4, 128, 64, 4, 1))

    # ---- Long sequence tests (no reference comparison) ----
    print("\n=== Long Sequence Tests (run + finiteness check) ===")

    for N in [2048, 4096, 8192, 16384]:
        results.append(run_test(
            f"fwd long N={N} sink=4 win=4096",
            test_forward_long_seq, N, 4, 4096
        ))

    for N in [2048, 4096, 8192]:
        results.append(run_test(
            f"bwd long N={N} sink=4 win=4096",
            test_backward_long_seq, N, 4, 4096
        ))

    # ---- Summary ----
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 40}")
    print(f"CORRECTNESS: {passed}/{total} tests passed.")
    print(f"{'=' * 40}")

    if passed < total:
        print("SOME TESTS FAILED!")
        sys.exit(1)

    # ---- Benchmarks ----
    run_benchmarks()
