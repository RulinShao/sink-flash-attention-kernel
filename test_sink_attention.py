"""
Correctness tests for Sink Flash Attention.

Compares the Triton kernel output against a naive PyTorch implementation
that explicitly constructs the sink + sliding window attention mask.
"""

import torch
import math
import sys

from sink_flash_attention import sink_flash_attention


def naive_sink_attention(q, k, v, num_sink, window_size):
    """
    Reference implementation using explicit mask construction.
    q, k, v: [B, H, N, D]
    """
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    heads_per_group = H_q // H_kv

    scale = 1.0 / math.sqrt(D)

    outputs = []
    for h_q in range(H_q):
        h_kv = h_q // heads_per_group
        qi = q[:, h_q]
        ki = k[:, h_kv]
        vi = v[:, h_kv]

        scores = torch.matmul(qi, ki.transpose(-2, -1)) * scale

        row_idx = torch.arange(N, device=q.device).unsqueeze(1)
        col_idx = torch.arange(N, device=q.device).unsqueeze(0)

        causal = col_idx <= row_idx
        sink = col_idx < num_sink
        window = col_idx >= (row_idx - window_size + 1)
        mask = (sink | window) & causal

        scores = scores.masked_fill(~mask.unsqueeze(0), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.matmul(attn, vi)
        outputs.append(out)

    return torch.stack(outputs, dim=1)


def test_forward_correctness(B, H_q, H_kv, N, D, num_sink, window_size):
    """Test that forward pass matches naive implementation."""
    if num_sink >= N or window_size >= N:
        return

    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

    out_triton = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)

    out_ref = naive_sink_attention(q.float(), k.float(), v.float(), num_sink, window_size)
    out_ref = out_ref.half()

    torch.testing.assert_close(out_triton, out_ref, atol=1e-2, rtol=1e-2)


def test_backward_correctness(B, H_q, H_kv, N, D, num_sink, window_size):
    """Test that gradients are correct via comparison with naive implementation."""
    torch.manual_seed(42)

    q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    out_ref = naive_sink_attention(q_ref, k_ref, v_ref, num_sink, window_size)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    q_tri = q.detach().half().requires_grad_(True)
    k_tri = k.detach().half().requires_grad_(True)
    v_tri = v.detach().half().requires_grad_(True)

    out_tri = sink_flash_attention(q_tri, k_tri, v_tri, num_sink=num_sink, window_size=window_size)
    out_tri.backward(grad_out.half())

    torch.testing.assert_close(q_tri.grad.float(), q_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_tri.grad.float(), k_ref.grad, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_tri.grad.float(), v_ref.grad, atol=5e-2, rtol=5e-2)


def test_degenerate_full_attention():
    """When window_size >= N and num_sink=0, should match standard causal attention."""
    B, H, N, D = 1, 4, 128, 64
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out = sink_flash_attention(q, k, v, num_sink=0, window_size=N)

    scale = 1.0 / math.sqrt(D)
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    mask = torch.triu(torch.ones(N, N, device='cuda'), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out_ref = torch.matmul(attn, v.float()).half()

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


def test_sink_only():
    """When window_size=1, only sink tokens + self should be attended to."""
    B, H, N, D = 1, 2, 64, 64
    num_sink = 4
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=1)
    out_ref = naive_sink_attention(q.float(), k.float(), v.float(), num_sink, window_size=1).half()

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


def test_memory_efficiency():
    """Verify the kernel doesn't materialize the full N x N attention matrix."""
    B, H, N, D = 1, 4, 4096, 128
    num_sink = 4
    window_size = 256

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.max_memory_allocated()

    out = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)

    mem_after = torch.cuda.max_memory_allocated()
    mem_used = mem_after - mem_before

    full_attn_mem = B * H * N * N * 2
    assert mem_used < full_attn_mem * 0.25, \
        f"Memory usage {mem_used / 1e6:.1f}MB seems too high (full attn would be {full_attn_mem / 1e6:.1f}MB)"

    print(f"  Memory: {mem_used / 1e6:.1f}MB vs full attention {full_attn_mem / 1e6:.1f}MB "
          f"({mem_used / full_attn_mem * 100:.1f}%)")


def run_test(name, fn, *args, **kwargs):
    """Run a single test with error reporting."""
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


if __name__ == "__main__":
    print("Running smoke tests...")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    results.append(run_test("degenerate full attention", test_degenerate_full_attention))
    results.append(run_test("sink only", test_sink_only))
    results.append(run_test("memory efficiency", test_memory_efficiency))

    # Forward correctness - multiple configs
    for B, H_q, H_kv, N, D, ns, ws in [
        (1, 4, 4, 128, 64, 4, 32),
        (1, 4, 4, 256, 64, 4, 64),
        (1, 8, 2, 256, 64, 4, 64),   # GQA
        (2, 4, 4, 128, 64, 1, 64),   # num_sink=1
        (1, 4, 4, 256, 128, 4, 64),  # D=128
        (1, 4, 4, 512, 64, 16, 128), # larger N, more sinks
    ]:
        results.append(run_test(
            f"forward B={B} H_q={H_q} H_kv={H_kv} N={N} D={D} sink={ns} win={ws}",
            test_forward_correctness, B, H_q, H_kv, N, D, ns, ws
        ))

    # Backward correctness
    for B, H_q, H_kv, N, D, ns, ws in [
        (1, 4, 4, 128, 64, 4, 32),
        (1, 4, 2, 128, 64, 4, 64),   # GQA backward
    ]:
        results.append(run_test(
            f"backward B={B} H_q={H_q} H_kv={H_kv} N={N} D={D} sink={ns} win={ws}",
            test_backward_correctness, B, H_q, H_kv, N, D, ns, ws
        ))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed.")
    if passed < total:
        sys.exit(1)
