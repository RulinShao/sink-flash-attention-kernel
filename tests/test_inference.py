"""
Inference tests for sink flash attention.

Tests cover:
  - Decode correctness: single-step decode via cache + decode kernel matches
    naive full-sequence attention
  - Multi-step decode: sequential generation produces same result as batched
  - End-to-end generation with a HuggingFace model
"""

import torch
import math
import sys


def naive_sink_attention_full(q, k, v, num_sink, window_size):
    """Reference implementation for full-sequence sink attention.

    q, k, v: [B, H, N, D]
    Returns: [B, H, N, D]
    """
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    heads_per_group = H_q // H_kv
    scale = 1.0 / math.sqrt(D)

    outputs = []
    for h_q in range(H_q):
        h_kv = h_q // heads_per_group
        qi = q[:, h_q]  # [B, N, D]
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


def test_decode_correctness():
    """Verify decode produces the same output as the last row of full attention.

    Process: build full sequence, compute full sink attention, compare last
    token's output with decode using cache.
    """
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    B, H_q, H_kv, D = 1, 4, 4, 64
    num_sink, window_size = 4, 8
    N = 20

    torch.manual_seed(42)
    q_full = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float32)
    k_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32)
    v_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32)

    # Full attention reference for last token
    ref_out = naive_sink_attention_full(q_full, k_full, v_full, num_sink, window_size)
    ref_last = ref_out[:, :, -1:, :]  # [B, H_q, 1, D]

    # Simulate cache: prefill first N-1 tokens, then decode last token
    layer = SinkCacheLayer(num_sink, window_size)
    layer.update(k_full[:, :, :N-1, :], v_full[:, :, :N-1, :])

    # Decode last token
    k_cached, v_cached = layer.update(k_full[:, :, N-1:N, :], v_full[:, :, N-1:N, :])
    q_last = q_full[:, :, N-1:N, :]

    out_decode = sink_decode_attention(q_last, k_cached, v_cached)

    torch.testing.assert_close(out_decode, ref_last, atol=1e-4, rtol=1e-4)


def test_decode_correctness_gqa():
    """Decode correctness with GQA (H_q != H_kv)."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    B, H_q, H_kv, D = 1, 8, 2, 64
    num_sink, window_size = 4, 8
    N = 16

    torch.manual_seed(123)
    q_full = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float32)
    k_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32)
    v_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32)

    ref_out = naive_sink_attention_full(q_full, k_full, v_full, num_sink, window_size)
    ref_last = ref_out[:, :, -1:, :]

    layer = SinkCacheLayer(num_sink, window_size)
    layer.update(k_full[:, :, :N-1, :], v_full[:, :, :N-1, :])
    k_cached, v_cached = layer.update(k_full[:, :, N-1:N, :], v_full[:, :, N-1:N, :])
    q_last = q_full[:, :, N-1:N, :]

    out_decode = sink_decode_attention(q_last, k_cached, v_cached)

    torch.testing.assert_close(out_decode, ref_last, atol=1e-4, rtol=1e-4)


def test_multistep_decode():
    """Multi-step decode: verify each step matches full attention's corresponding output."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    B, H_q, H_kv, D = 1, 4, 4, 64
    num_sink, window_size = 2, 4
    prefill_len = 6
    decode_steps = 8

    torch.manual_seed(7)
    N_total = prefill_len + decode_steps
    q_all = torch.randn(B, H_q, N_total, D, device='cuda', dtype=torch.float32)
    k_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=torch.float32)
    v_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=torch.float32)

    layer = SinkCacheLayer(num_sink, window_size)
    layer.update(k_all[:, :, :prefill_len, :], v_all[:, :, :prefill_len, :])

    for step in range(decode_steps):
        pos = prefill_len + step
        q_step = q_all[:, :, pos:pos+1, :]
        k_step = k_all[:, :, pos:pos+1, :]
        v_step = v_all[:, :, pos:pos+1, :]

        k_cached, v_cached = layer.update(k_step, v_step)
        out_decode = sink_decode_attention(q_step, k_cached, v_cached)

        # Reference: full attention up to pos+1
        ref_out = naive_sink_attention_full(
            q_all[:, :, :pos+1, :],
            k_all[:, :, :pos+1, :],
            v_all[:, :, :pos+1, :],
            num_sink, window_size,
        )
        ref_step = ref_out[:, :, pos:pos+1, :]

        torch.testing.assert_close(
            out_decode, ref_step, atol=1e-4, rtol=1e-4,
            msg=f"Mismatch at decode step {step} (position {pos})",
        )


def test_decode_with_eviction():
    """Verify correctness when window buffer wraps and evicts old tokens."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    B, H_q, H_kv, D = 1, 2, 2, 32
    num_sink, window_size = 2, 3
    prefill_len = 4
    # Decode enough steps to wrap the window buffer multiple times
    decode_steps = 10

    torch.manual_seed(99)
    N_total = prefill_len + decode_steps
    q_all = torch.randn(B, H_q, N_total, D, device='cuda', dtype=torch.float32)
    k_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=torch.float32)
    v_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=torch.float32)

    layer = SinkCacheLayer(num_sink, window_size)
    layer.update(k_all[:, :, :prefill_len, :], v_all[:, :, :prefill_len, :])

    for step in range(decode_steps):
        pos = prefill_len + step
        q_step = q_all[:, :, pos:pos+1, :]
        k_step = k_all[:, :, pos:pos+1, :]
        v_step = v_all[:, :, pos:pos+1, :]

        k_cached, v_cached = layer.update(k_step, v_step)
        out_decode = sink_decode_attention(q_step, k_cached, v_cached)

        ref_out = naive_sink_attention_full(
            q_all[:, :, :pos+1, :],
            k_all[:, :, :pos+1, :],
            v_all[:, :, :pos+1, :],
            num_sink, window_size,
        )
        ref_step = ref_out[:, :, pos:pos+1, :]

        torch.testing.assert_close(
            out_decode, ref_step, atol=1e-4, rtol=1e-4,
            msg=f"Mismatch at decode step {step} (pos {pos}, window wraps={step >= window_size})",
        )


def test_decode_fp16():
    """Decode correctness in fp16 (lower precision)."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    B, H_q, H_kv, D = 1, 4, 4, 64
    num_sink, window_size = 4, 8
    N = 20

    torch.manual_seed(42)
    q_full = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16)
    k_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
    v_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

    ref_out = naive_sink_attention_full(
        q_full.float(), k_full.float(), v_full.float(), num_sink, window_size
    )
    ref_last = ref_out[:, :, -1:, :].half()

    layer = SinkCacheLayer(num_sink, window_size)
    layer.update(k_full[:, :, :N-1, :], v_full[:, :, :N-1, :])
    k_cached, v_cached = layer.update(k_full[:, :, N-1:N, :], v_full[:, :, N-1:N, :])
    q_last = q_full[:, :, N-1:N, :]

    out_decode = sink_decode_attention(q_last, k_cached, v_cached)

    torch.testing.assert_close(out_decode, ref_last, atol=1e-2, rtol=1e-2)


def test_generation_e2e():
    """End-to-end test with HuggingFace model.generate()."""
    from sink_attention.generate_patch import patch_for_generation, unpatch_generation

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("  SKIP: transformers not available")
        return True

    model_candidates = [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ]

    model = None
    tokenizer = None
    for model_name in model_candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
            ).cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
            )
            print(f"  Loaded {model_name}")
            break
        except Exception:
            continue

    if model is None:
        print("  SKIP: No cached model available")
        return True

    # Patch for generation
    cache = patch_for_generation(model, num_sink=4, window_size=128)

    try:
        input_ids = torch.randint(0, 1000, (1, 32), device='cuda')

        # Generate with cache
        with torch.no_grad():
            output = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=16,
                do_sample=False,
            )

        assert output.shape[1] > input_ids.shape[1], "No tokens generated"
        print(f"  Generated {output.shape[1] - input_ids.shape[1]} tokens")
        print(f"  Output shape: {output.shape}")

    finally:
        unpatch_generation()
        del model
        torch.cuda.empty_cache()

    return True


def run_test(name, fn):
    """Run a single test with error reporting."""
    try:
        result = fn()
        if result is False:
            print(f"  FAIL: {name}")
            return False
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"        {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running inference tests...")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print()

    results = []

    # Decode correctness tests
    results.append(run_test("decode correctness (MHA)", test_decode_correctness))
    results.append(run_test("decode correctness (GQA)", test_decode_correctness_gqa))
    results.append(run_test("multi-step decode", test_multistep_decode))
    results.append(run_test("decode with eviction", test_decode_with_eviction))
    results.append(run_test("decode fp16", test_decode_fp16))

    # End-to-end
    results.append(run_test("generation e2e", test_generation_e2e))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed.")
    if passed < total:
        sys.exit(1)
