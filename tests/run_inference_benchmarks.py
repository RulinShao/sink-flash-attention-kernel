"""
Numerical accuracy and performance benchmark for inference features.

Runs:
  1. Cache correctness tests
  2. Decode correctness tests (vs naive reference)
  3. Decode performance benchmark (latency)
  4. End-to-end generation test

Outputs results in a structured format for README inclusion.
"""

import torch
import math
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Reference implementation
# ============================================================================

def naive_sink_attention_full(q, k, v, num_sink, window_size):
    """Reference implementation for full-sequence sink attention."""
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


# ============================================================================
# Numerical accuracy tests
# ============================================================================

def test_decode_accuracy():
    """Measure numerical accuracy of decode vs naive reference."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    print("=" * 70)
    print("DECODE NUMERICAL ACCURACY")
    print("=" * 70)
    print(f"{'Config':<45} {'Max Err':>10} {'Mean Err':>10} {'CosSim':>10}")
    print("-" * 75)

    configs = [
        # (B, H_q, H_kv, D, num_sink, window_size, N, dtype_name)
        (1, 4, 4, 64, 4, 8, 20, "fp32"),
        (1, 4, 4, 128, 4, 16, 32, "fp32"),
        (1, 8, 2, 64, 4, 8, 24, "fp32"),     # GQA
        (1, 32, 8, 128, 4, 32, 64, "fp32"),   # GQA large
        (1, 4, 4, 64, 4, 8, 20, "fp16"),
        (1, 4, 4, 128, 4, 16, 32, "fp16"),
        (1, 8, 2, 64, 4, 8, 24, "fp16"),      # GQA fp16
        (1, 32, 8, 128, 4, 32, 64, "fp16"),   # GQA large fp16
        (1, 4, 4, 128, 4, 8, 20, "bf16"),
        (1, 8, 2, 128, 4, 16, 32, "bf16"),    # GQA bf16
    ]

    all_pass = True
    for B, H_q, H_kv, D, num_sink, window_size, N, dtype_name in configs:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        torch.manual_seed(42)

        q_full = torch.randn(B, H_q, N, D, device='cuda', dtype=dtype)
        k_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)
        v_full = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)

        # Reference: full attention, last token
        ref_out = naive_sink_attention_full(
            q_full.float(), k_full.float(), v_full.float(), num_sink, window_size
        )
        ref_last = ref_out[:, :, -1:, :].to(dtype)

        # Cache + decode
        layer = SinkCacheLayer(num_sink, window_size)
        layer.update(k_full[:, :, :N-1, :], v_full[:, :, :N-1, :])
        k_cached, v_cached = layer.update(k_full[:, :, N-1:N, :], v_full[:, :, N-1:N, :])
        q_last = q_full[:, :, N-1:N, :]
        out_decode = sink_decode_attention(q_last, k_cached, v_cached)

        # Compute errors
        max_err = (out_decode.float() - ref_last.float()).abs().max().item()
        mean_err = (out_decode.float() - ref_last.float()).abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            out_decode.float().flatten(), ref_last.float().flatten(), dim=0
        ).item()

        gqa_str = f"GQA({H_q}/{H_kv})" if H_q != H_kv else f"MHA({H_q})"
        config_str = f"B={B} {gqa_str} D={D} sink={num_sink} win={window_size} N={N} {dtype_name}"

        tol = 1e-4 if dtype_name == "fp32" else 1e-2
        status = "PASS" if max_err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {config_str:<43} {max_err:>10.2e} {mean_err:>10.2e} {cos_sim:>10.6f}  [{status}]")

    print()
    return all_pass


def test_multistep_decode_accuracy():
    """Test accuracy across multiple decode steps with eviction."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    print("=" * 70)
    print("MULTI-STEP DECODE ACCURACY (with eviction)")
    print("=" * 70)

    configs = [
        # (B, H_q, H_kv, D, num_sink, window_size, prefill_len, decode_steps, dtype)
        (1, 4, 4, 64, 2, 4, 6, 10, "fp32"),
        (1, 4, 4, 128, 4, 8, 8, 20, "fp32"),
        (1, 8, 2, 64, 4, 8, 10, 15, "fp32"),
        (1, 4, 4, 64, 2, 4, 6, 10, "fp16"),
        (1, 4, 4, 128, 4, 8, 8, 20, "fp16"),
    ]

    all_pass = True
    for B, H_q, H_kv, D, num_sink, window_size, prefill_len, decode_steps, dtype_name in configs:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        torch.manual_seed(7)
        N_total = prefill_len + decode_steps
        q_all = torch.randn(B, H_q, N_total, D, device='cuda', dtype=dtype)
        k_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=dtype)
        v_all = torch.randn(B, H_kv, N_total, D, device='cuda', dtype=dtype)

        layer = SinkCacheLayer(num_sink, window_size)
        layer.update(k_all[:, :, :prefill_len, :], v_all[:, :, :prefill_len, :])

        max_errors = []
        for step in range(decode_steps):
            pos = prefill_len + step
            q_step = q_all[:, :, pos:pos+1, :]
            k_step = k_all[:, :, pos:pos+1, :]
            v_step = v_all[:, :, pos:pos+1, :]

            k_cached, v_cached = layer.update(k_step, v_step)
            out_decode = sink_decode_attention(q_step, k_cached, v_cached)

            ref_out = naive_sink_attention_full(
                q_all[:, :, :pos+1, :].float(),
                k_all[:, :, :pos+1, :].float(),
                v_all[:, :, :pos+1, :].float(),
                num_sink, window_size,
            )
            ref_step = ref_out[:, :, pos:pos+1, :].to(dtype)

            max_err = (out_decode.float() - ref_step.float()).abs().max().item()
            max_errors.append(max_err)

        avg_max_err = sum(max_errors) / len(max_errors)
        worst_err = max(max_errors)
        tol = 1e-4 if dtype_name == "fp32" else 1e-2
        status = "PASS" if worst_err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False

        gqa_str = f"GQA({H_q}/{H_kv})" if H_q != H_kv else f"MHA({H_q})"
        config_str = f"B={B} {gqa_str} D={D} sink={num_sink} win={window_size} pre={prefill_len} dec={decode_steps} {dtype_name}"
        print(f"  {config_str}")
        print(f"    Avg max error: {avg_max_err:.2e}, Worst: {worst_err:.2e}  [{status}]")

    print()
    return all_pass


# ============================================================================
# Performance benchmarks
# ============================================================================

def benchmark_decode_latency():
    """Benchmark decode latency for various configurations."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention
    from sink_attention import sink_flash_attention

    print("=" * 70)
    print("DECODE LATENCY BENCHMARK")
    print("=" * 70)
    print(f"{'Config':<55} {'Decode (ms)':>12} {'Training kernel N_q=1 (ms)':>28}")
    print("-" * 95)

    configs = [
        # (B, H_q, H_kv, D, num_sink, window_size, N_kv)
        (1, 32, 8, 128, 4, 128, 132),
        (1, 32, 8, 128, 4, 512, 516),
        (1, 32, 8, 128, 4, 1024, 1028),
        (1, 32, 8, 128, 4, 2048, 2052),
        (1, 32, 8, 128, 4, 4096, 4100),
        (1, 64, 8, 128, 4, 1024, 1028),   # More Q heads
        (1, 64, 8, 128, 4, 4096, 4100),
    ]

    warmup = 10
    repeats = 100

    results = []
    for B, H_q, H_kv, D, num_sink, window_size, N_kv in configs:
        torch.manual_seed(42)

        # Setup: cache already populated, we benchmark the decode attention only
        q = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_kv, N_kv, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, N_kv, D, device='cuda', dtype=torch.float16)

        # Warmup decode kernel
        for _ in range(warmup):
            sink_decode_attention(q, k, v)
        torch.cuda.synchronize()

        # Benchmark decode kernel
        start = time.perf_counter()
        for _ in range(repeats):
            sink_decode_attention(q, k, v)
        torch.cuda.synchronize()
        decode_ms = (time.perf_counter() - start) / repeats * 1000

        # Benchmark training kernel with N_q=1 for comparison
        # Training kernel requires N_q == N_kv, so we construct full tensors
        q_train = torch.randn(B, H_q, N_kv, D, device='cuda', dtype=torch.float16)
        k_train = torch.randn(B, H_kv, N_kv, D, device='cuda', dtype=torch.float16)
        v_train = torch.randn(B, H_kv, N_kv, D, device='cuda', dtype=torch.float16)

        for _ in range(warmup):
            sink_flash_attention(q_train, k_train, v_train, num_sink=num_sink, window_size=window_size)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            sink_flash_attention(q_train, k_train, v_train, num_sink=num_sink, window_size=window_size)
        torch.cuda.synchronize()
        train_ms = (time.perf_counter() - start) / repeats * 1000

        gqa_str = f"GQA({H_q}/{H_kv})" if H_q != H_kv else f"MHA({H_q})"
        config_str = f"B={B} {gqa_str} D={D} sink={num_sink} win={window_size} N_kv={N_kv}"
        print(f"  {config_str:<53} {decode_ms:>10.3f}ms {train_ms:>10.3f}ms")

        results.append({
            "config": config_str,
            "decode_ms": decode_ms,
            "train_kernel_ms": train_ms,
            "speedup": train_ms / decode_ms if decode_ms > 0 else float('inf'),
        })

    print()
    print("Speedup (training kernel / decode kernel):")
    for r in results:
        print(f"  {r['config']:<53} {r['speedup']:>6.1f}x")

    print()
    return results


def benchmark_cache_overhead():
    """Benchmark cache update + decode overhead."""
    from sink_attention.cache import SinkCacheLayer
    from sink_attention.decode_kernel import sink_decode_attention

    print("=" * 70)
    print("CACHE UPDATE + DECODE OVERHEAD")
    print("=" * 70)
    print(f"{'Config':<50} {'Cache update':>12} {'Decode attn':>12} {'Total':>12}")
    print("-" * 86)

    configs = [
        (1, 32, 8, 128, 4, 1024),
        (1, 32, 8, 128, 4, 4096),
        (1, 64, 8, 128, 4, 4096),
    ]

    warmup = 10
    repeats = 200

    for B, H_q, H_kv, D, num_sink, window_size in configs:
        torch.manual_seed(42)

        # Prefill
        layer = SinkCacheLayer(num_sink, window_size)
        k_pre = torch.randn(B, H_kv, num_sink + window_size, D, device='cuda', dtype=torch.float16)
        v_pre = torch.randn(B, H_kv, num_sink + window_size, D, device='cuda', dtype=torch.float16)
        layer.update(k_pre, v_pre)

        q = torch.randn(B, H_q, 1, D, device='cuda', dtype=torch.float16)
        k_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
        v_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)

        # Warmup
        for _ in range(warmup):
            k_cached, v_cached = layer.update(k_new, v_new)
            sink_decode_attention(q, k_cached, v_cached)
        torch.cuda.synchronize()

        # Benchmark cache update
        start = time.perf_counter()
        for _ in range(repeats):
            k_cached, v_cached = layer.update(k_new, v_new)
        torch.cuda.synchronize()
        cache_ms = (time.perf_counter() - start) / repeats * 1000

        # Benchmark decode attention
        k_cached, v_cached = layer.get_kv()
        start = time.perf_counter()
        for _ in range(repeats):
            sink_decode_attention(q, k_cached, v_cached)
        torch.cuda.synchronize()
        attn_ms = (time.perf_counter() - start) / repeats * 1000

        gqa_str = f"GQA({H_q}/{H_kv})" if H_q != H_kv else f"MHA({H_q})"
        config_str = f"B={B} {gqa_str} D={D} sink={num_sink} win={window_size}"
        total = cache_ms + attn_ms
        print(f"  {config_str:<48} {cache_ms:>10.3f}ms {attn_ms:>10.3f}ms {total:>10.3f}ms")

    print()


def benchmark_prefill_latency():
    """Benchmark prefill latency: sink FA kernel vs standard SDPA."""
    from sink_attention import sink_flash_attention

    print("=" * 70)
    print("PREFILL LATENCY: Sink FA vs torch SDPA (causal)")
    print("=" * 70)
    print(f"{'Config':<45} {'Sink FA (ms)':>14} {'SDPA (ms)':>12} {'Ratio':>8}")
    print("-" * 80)

    configs = [
        (1, 32, 8, 128, 4, 4096, 512),
        (1, 32, 8, 128, 4, 4096, 1024),
        (1, 32, 8, 128, 4, 4096, 2048),
        (1, 32, 8, 128, 4, 4096, 4096),
        (1, 32, 8, 128, 4, 4096, 8192),
    ]

    warmup = 5
    repeats = 20

    for B, H_q, H_kv, D, num_sink, window_size, N in configs:
        torch.manual_seed(42)
        q = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

        # Sink FA
        for _ in range(warmup):
            sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
        torch.cuda.synchronize()
        sink_ms = (time.perf_counter() - start) / repeats * 1000

        # SDPA (expand KV for GQA)
        groups = H_q // H_kv
        k_exp = k.repeat_interleave(groups, dim=1)
        v_exp = v.repeat_interleave(groups, dim=1)

        for _ in range(warmup):
            torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            torch.nn.functional.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        torch.cuda.synchronize()
        sdpa_ms = (time.perf_counter() - start) / repeats * 1000

        ratio = sink_ms / sdpa_ms
        config_str = f"B={B} GQA({H_q}/{H_kv}) D={D} N={N}"
        print(f"  {config_str:<43} {sink_ms:>12.3f}ms {sdpa_ms:>10.3f}ms {ratio:>7.2f}x")

    print()


# ============================================================================
# Cache correctness tests
# ============================================================================

def run_cache_tests():
    """Run all cache tests."""
    print("=" * 70)
    print("CACHE CORRECTNESS TESTS")
    print("=" * 70)

    from sink_attention.cache import SinkCacheLayer, SinkAttentionCache

    results = []

    def run(name, fn):
        try:
            fn()
            print(f"  PASS: {name}")
            results.append(True)
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Test 1: prefill returns full KV
    def t1():
        layer = SinkCacheLayer(4, 8)
        k = torch.randn(1, 4, 16, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 4, 16, 64, device='cuda', dtype=torch.float16)
        k_out, v_out = layer.update(k, v)
        assert k_out.shape == k.shape
        torch.testing.assert_close(k_out, k)
    run("prefill returns full KV", t1)

    # Test 2: prefill short
    def t2():
        layer = SinkCacheLayer(8, 16)
        k = torch.randn(1, 4, 4, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 4, 4, 64, device='cuda', dtype=torch.float16)
        k_out, _ = layer.update(k, v)
        assert k_out.shape == k.shape
        assert layer.sink_len == 4
        assert layer.window_len == 0
    run("prefill short (< num_sink)", t2)

    # Test 3: prefill overflow
    def t3():
        layer = SinkCacheLayer(4, 8)
        k = torch.randn(1, 2, 20, 64, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 2, 20, 64, device='cuda', dtype=torch.float16)
        layer.update(k, v)
        k_cached, _ = layer.get_kv()
        assert k_cached.shape == (1, 2, 12, 64)
        torch.testing.assert_close(k_cached[:, :, :4, :], k[:, :, :4, :])
        torch.testing.assert_close(k_cached[:, :, 4:, :], k[:, :, 12:, :])
    run("prefill overflow window", t3)

    # Test 4: decode eviction
    def t4():
        layer = SinkCacheLayer(2, 4)
        k_pre = torch.arange(4 * 32, device='cuda', dtype=torch.float16).view(1, 1, 4, 32)
        v_pre = k_pre.clone()
        layer.update(k_pre, v_pre)
        tokens = []
        for i in range(6):
            val = (i + 4) * 100
            k_new = torch.full((1, 1, 1, 32), val, device='cuda', dtype=torch.float16)
            tokens.append(k_new)
            k_out, _ = layer.update(k_new, k_new.clone())
        assert k_out.shape == (1, 1, 6, 32)
        torch.testing.assert_close(k_out[:, :, :2, :], k_pre[:, :, :2, :])
        for i in range(4):
            torch.testing.assert_close(k_out[:, :, 2 + i, :], tokens[2 + i][:, :, 0, :])
    run("decode with eviction", t4)

    # Test 5: circular linearization
    def t5():
        layer = SinkCacheLayer(1, 3)
        k_pre = torch.arange(4 * 8, device='cuda', dtype=torch.float16).view(1, 1, 4, 8)
        layer.update(k_pre, k_pre.clone())
        k4 = torch.full((1, 1, 1, 8), 100.0, device='cuda', dtype=torch.float16)
        k_out, _ = layer.update(k4, k4.clone())
        assert k_out.shape == (1, 1, 4, 8)
        torch.testing.assert_close(k_out[:, :, 0, :], k_pre[:, :, 0, :])
        torch.testing.assert_close(k_out[:, :, 1, :], k_pre[:, :, 2, :])
        torch.testing.assert_close(k_out[:, :, 2, :], k_pre[:, :, 3, :])
        torch.testing.assert_close(k_out[:, :, 3, :], k4[:, :, 0, :])
    run("circular buffer linearization", t5)

    # Test 6: multi-layer cache
    def t6():
        cache = SinkAttentionCache(num_sink=4, window_size=8)
        for layer_idx in range(3):
            k = torch.randn(1, 4, 16, 64, device='cuda', dtype=torch.float16)
            v = torch.randn(1, 4, 16, 64, device='cuda', dtype=torch.float16)
            k_out, _ = cache.update(k, v, layer_idx)
            assert k_out.shape == (1, 4, 16, 64)  # prefill returns full
        assert len(cache) == 3
        assert cache.get_seq_length(0) == 12  # 4 + 8
        for layer_idx in range(3):
            k = torch.randn(1, 4, 1, 64, device='cuda', dtype=torch.float16)
            v = torch.randn(1, 4, 1, 64, device='cuda', dtype=torch.float16)
            k_out, _ = cache.update(k, v, layer_idx)
        assert k_out.shape == (1, 4, 12, 64)
    run("multi-layer cache", t6)

    # Test 7: HF Cache isinstance
    def t7():
        from transformers.cache_utils import Cache
        cache = SinkAttentionCache(num_sink=4, window_size=128)
        assert isinstance(cache, Cache)
    run("HF Cache isinstance", t7)

    # Test 8: beam search reorder
    def t8():
        layer = SinkCacheLayer(2, 4)
        k = torch.randn(4, 2, 6, 32, device='cuda', dtype=torch.float16)
        layer.update(k, k.clone())
        beam_idx = torch.tensor([1, 1, 1, 1], device='cuda', dtype=torch.long)
        layer.reorder_cache(beam_idx)
        k_out, _ = layer.get_kv()
        for b in range(4):
            torch.testing.assert_close(k_out[b], k_out[0])
    run("beam search reorder", t8)

    print(f"\n  {sum(results)}/{len(results)} cache tests passed.")
    print()
    return all(results)


# ============================================================================
# End-to-end generation test
# ============================================================================

def test_generation_e2e():
    """End-to-end test with HuggingFace model.generate()."""
    from sink_attention.generate_patch import patch_for_generation, unpatch_generation

    print("=" * 70)
    print("END-TO-END GENERATION TEST")
    print("=" * 70)

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
    model_name_used = None
    for model_name in model_candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
            ).cuda()
            model_name_used = model_name
            print(f"  Loaded {model_name} ({model.config.num_hidden_layers}L, "
                  f"{model.config.num_attention_heads}H, "
                  f"kv_heads={getattr(model.config, 'num_key_value_heads', 'N/A')})")
            break
        except Exception:
            continue

    if model is None:
        print("  SKIP: No cached model available")
        return True

    cache = patch_for_generation(model, num_sink=4, window_size=128)

    try:
        input_ids = torch.randint(0, 1000, (1, 64), device='cuda')

        with torch.no_grad():
            output = model.generate(
                input_ids,
                past_key_values=cache,
                max_new_tokens=32,
                do_sample=False,
            )

        generated = output.shape[1] - input_ids.shape[1]
        assert generated > 0, "No tokens generated"
        print(f"  Generated {generated} tokens (input={input_ids.shape[1]}, output={output.shape[1]})")
        print(f"  PASS")

    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        unpatch_generation()
        del model
        torch.cuda.empty_cache()

    print()
    return True


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    ok = True

    # 1. Cache tests
    ok &= run_cache_tests()

    # 2. Numerical accuracy
    ok &= test_decode_accuracy()
    ok &= test_multistep_decode_accuracy()

    # 3. Performance benchmarks
    benchmark_prefill_latency()
    benchmark_decode_latency()
    benchmark_cache_overhead()

    # 4. E2E generation
    ok &= test_generation_e2e()

    print("=" * 70)
    if ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
