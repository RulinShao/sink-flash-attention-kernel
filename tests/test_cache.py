"""
Tests for SinkCacheLayer and SinkAttentionCache.

Tests cover:
  - Prefill with various sequence lengths (returns full KV for kernel masking)
  - Internal cache state after prefill
  - Sequential decode with eviction (returns linearized [sink, window])
  - Circular buffer linearization correctness
  - GQA compatibility
  - Multi-layer cache
  - Beam search reordering
"""

import torch
import sys


def test_prefill_returns_full_kv():
    """Prefill should return the FULL input KV (kernel handles masking)."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 4, 64
    num_sink, window_size = 4, 8
    N = 16

    layer = SinkCacheLayer(num_sink, window_size)

    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)

    # Prefill returns the FULL input KV unchanged
    assert k_out.shape == (B, H, N, D), f"Expected full KV shape, got {k_out.shape}"
    torch.testing.assert_close(k_out, k)
    torch.testing.assert_close(v_out, v)


def test_prefill_short():
    """Prefill with fewer tokens than num_sink — cache state correct."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 4, 64
    num_sink, window_size = 8, 16
    N = 4  # fewer than num_sink

    layer = SinkCacheLayer(num_sink, window_size)

    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)

    # Prefill returns full input
    assert k_out.shape == (B, H, N, D)
    torch.testing.assert_close(k_out, k)

    # Internal state: only N sink slots populated
    assert layer.sink_len == N
    assert layer.window_len == 0
    assert layer.get_seq_length() == N

    # get_kv should return only populated sink slots
    k_cached, v_cached = layer.get_kv()
    assert k_cached.shape == (B, H, N, D)
    torch.testing.assert_close(k_cached, k)


def test_prefill_exact_sink():
    """Prefill with exactly num_sink tokens."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 2, 64
    num_sink, window_size = 4, 8
    N = 4

    layer = SinkCacheLayer(num_sink, window_size)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)

    # Full KV returned
    assert k_out.shape == (B, H, N, D)
    torch.testing.assert_close(k_out, k)

    # Cache state
    assert layer.sink_len == num_sink
    assert layer.window_len == 0


def test_prefill_sink_plus_window():
    """Prefill with tokens filling both sink and part of window."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 2, 64
    num_sink, window_size = 4, 8
    N = 10  # 4 sink + 6 window

    layer = SinkCacheLayer(num_sink, window_size)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)

    # Prefill returns full input
    assert k_out.shape == (B, H, N, D)
    torch.testing.assert_close(k_out, k)

    # Cache state
    assert layer.sink_len == num_sink
    assert layer.window_len == 6

    # get_kv returns sink + window in order
    k_cached, v_cached = layer.get_kv()
    assert k_cached.shape == (B, H, N, D)
    torch.testing.assert_close(k_cached[:, :, :num_sink, :], k[:, :, :num_sink, :])
    torch.testing.assert_close(k_cached[:, :, num_sink:, :], k[:, :, num_sink:, :])


def test_prefill_overflow_window():
    """Prefill with more non-sink tokens than window_size — cache keeps last window_size."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 2, 64
    num_sink, window_size = 4, 8
    N = 20  # 4 sink + 16 non-sink, but window only holds 8

    layer = SinkCacheLayer(num_sink, window_size)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)

    # Prefill returns full input
    assert k_out.shape == (B, H, N, D)
    torch.testing.assert_close(k_out, k)

    # Cache state: stores sink + last window_size tokens
    assert layer.sink_len == num_sink
    assert layer.window_len == window_size

    # get_kv returns truncated version
    k_cached, v_cached = layer.get_kv()
    assert k_cached.shape == (B, H, num_sink + window_size, D)
    torch.testing.assert_close(k_cached[:, :, :num_sink, :], k[:, :, :num_sink, :])
    torch.testing.assert_close(k_cached[:, :, num_sink:, :], k[:, :, N - window_size:, :])


def test_decode_sequential():
    """Decode one token at a time, verify correctness before eviction."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 2, 64
    num_sink, window_size = 2, 4

    layer = SinkCacheLayer(num_sink, window_size)

    # Prefill with 4 tokens (2 sink + 2 window)
    k_pre = torch.randn(B, H, 4, D, device='cuda', dtype=torch.float16)
    v_pre = torch.randn(B, H, 4, D, device='cuda', dtype=torch.float16)
    layer.update(k_pre, v_pre)

    assert layer.window_len == 2
    assert layer.get_seq_length() == 4

    # Decode 2 more tokens (still no eviction, window can hold 4)
    tokens = []
    for i in range(2):
        k_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        v_new = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        tokens.append((k_new, v_new))
        k_out, v_out = layer.update(k_new, v_new)

    assert layer.window_len == 4
    assert layer.get_seq_length() == 6
    assert k_out.shape == (B, H, num_sink + 4, D)

    # Verify sink tokens preserved
    torch.testing.assert_close(k_out[:, :, :num_sink, :], k_pre[:, :, :num_sink, :])


def test_decode_with_eviction():
    """Decode past window capacity and verify eviction + circular linearization."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 1, 32
    num_sink, window_size = 2, 4

    layer = SinkCacheLayer(num_sink, window_size)

    # Prefill: 2 sink + 2 window tokens
    k_pre = torch.arange(4 * D, device='cuda', dtype=torch.float16).view(B, H, 4, D)
    v_pre = k_pre.clone()
    layer.update(k_pre, v_pre)

    # Decode 6 more tokens (window will wrap multiple times)
    all_decode_tokens = []
    for i in range(6):
        val = (i + 4) * 100  # distinct values
        k_new = torch.full((B, H, 1, D), val, device='cuda', dtype=torch.float16)
        v_new = k_new.clone()
        all_decode_tokens.append(k_new)
        k_out, v_out = layer.update(k_new, v_new)

    # Window should be full (4 tokens), total = 2 sink + 4 window = 6
    assert k_out.shape == (B, H, 6, D)
    assert layer.window_len == 4

    # Sink tokens should be preserved
    torch.testing.assert_close(k_out[:, :, :2, :], k_pre[:, :, :2, :])

    # Window should have the last 4 decoded tokens in order
    # Decoded tokens 2,3,4,5 (0-indexed from decode start)
    for w_idx in range(4):
        decode_idx = 2 + w_idx  # last 4 of 6 decoded tokens
        torch.testing.assert_close(
            k_out[:, :, 2 + w_idx, :],
            all_decode_tokens[decode_idx][:, :, 0, :],
        )


def test_circular_buffer_linearization():
    """Verify circular buffer is correctly linearized after wrapping."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 1, 1, 8
    num_sink, window_size = 1, 3

    layer = SinkCacheLayer(num_sink, window_size)

    # Prefill: 1 sink + 3 window = 4 tokens
    k_pre = torch.arange(4 * D, device='cuda', dtype=torch.float16).view(B, H, 4, D)
    v_pre = k_pre.clone()
    layer.update(k_pre, v_pre)

    # Now window has [token1, token2, token3], write_pos=0 (full, wrapped)
    assert layer.window_len == 3

    # Decode token 4 — should evict token1
    k4 = torch.full((B, H, 1, D), 100.0, device='cuda', dtype=torch.float16)
    k_out, _ = layer.update(k4, k4.clone())

    # Should be: [sink, token2, token3, token4]
    assert k_out.shape == (B, H, 4, D)
    # Token at position 0 is sink
    torch.testing.assert_close(k_out[:, :, 0, :], k_pre[:, :, 0, :])
    # Remaining 3 are window in chronological order
    torch.testing.assert_close(k_out[:, :, 1, :], k_pre[:, :, 2, :])  # token2
    torch.testing.assert_close(k_out[:, :, 2, :], k_pre[:, :, 3, :])  # token3
    torch.testing.assert_close(k_out[:, :, 3, :], k4[:, :, 0, :])     # token4


def test_gqa_shapes():
    """Verify cache works with GQA (H_q > H_kv)."""
    from sink_attention.cache import SinkCacheLayer

    B, H_kv, D = 2, 4, 64
    num_sink, window_size = 4, 8
    N = 16

    layer = SinkCacheLayer(num_sink, window_size)

    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)

    k_out, v_out = layer.update(k, v)
    # Prefill returns full input
    assert k_out.shape == (B, H_kv, N, D)

    # Decode
    k_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
    v_new = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
    k_out, v_out = layer.update(k_new, v_new)
    assert k_out.shape[0] == B
    assert k_out.shape[1] == H_kv
    # Should have sink + window tokens
    expected_window = min(N - num_sink + 1, window_size)
    assert k_out.shape[2] == num_sink + expected_window


def test_multi_layer_cache():
    """Test SinkAttentionCache with multiple layers."""
    from sink_attention.cache import SinkAttentionCache

    cache = SinkAttentionCache(num_sink=4, window_size=8)

    B, H_kv, D = 1, 4, 64
    N = 16

    # Prefill 3 layers — prefill returns full KV
    for layer_idx in range(3):
        k = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float16)
        k_out, v_out = cache.update(k, v, layer_idx)
        # Prefill returns full input
        assert k_out.shape == (B, H_kv, N, D)

    assert len(cache) == 3
    # Internal cache state: 4 sink + 8 window = 12 (since 16-4=12 > 8, keeps last 8)
    assert cache.get_seq_length(0) == 4 + 8

    # Decode — returns [sink, window]
    for layer_idx in range(3):
        k = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H_kv, 1, D, device='cuda', dtype=torch.float16)
        k_out, v_out = cache.update(k, v, layer_idx)

    # After 1 decode: 4 sink + 8 window = 12 (one evicted, one added)
    assert cache.get_seq_length(0) == 4 + 8
    assert k_out.shape == (B, H_kv, 4 + 8, D)


def test_beam_search_reorder():
    """Test that beam reordering works correctly."""
    from sink_attention.cache import SinkCacheLayer

    B, H, D = 4, 2, 32
    num_sink, window_size = 2, 4

    layer = SinkCacheLayer(num_sink, window_size)

    # Give each batch a unique pattern
    k = torch.randn(B, H, 6, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, 6, D, device='cuda', dtype=torch.float16)
    layer.update(k, v)

    # Reorder: duplicate batch 1 to all positions
    beam_idx = torch.tensor([1, 1, 1, 1], device='cuda', dtype=torch.long)
    layer.reorder_cache(beam_idx)

    k_out, v_out = layer.get_kv()
    # All batches should now match batch 1
    for b in range(B):
        torch.testing.assert_close(k_out[b], k_out[0])


def test_seen_tokens_tracking():
    """Verify seen_tokens is tracked correctly across prefill and decode."""
    from sink_attention.cache import SinkAttentionCache

    cache = SinkAttentionCache(num_sink=2, window_size=4)
    B, H, D = 1, 2, 32

    # Prefill with 10 tokens
    k = torch.randn(B, H, 10, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, 10, D, device='cuda', dtype=torch.float16)
    cache.update(k, v, layer_idx=0)
    assert cache.seen_tokens == 10

    # 5 decode steps
    for _ in range(5):
        k = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, 1, D, device='cuda', dtype=torch.float16)
        cache.update(k, v, layer_idx=0)

    assert cache.seen_tokens == 15


def test_hf_cache_isinstance():
    """Verify SinkAttentionCache passes isinstance check for HF Cache."""
    from sink_attention.cache import SinkAttentionCache

    try:
        from transformers.cache_utils import Cache
        cache = SinkAttentionCache(num_sink=4, window_size=128)
        assert isinstance(cache, Cache), "SinkAttentionCache should be instance of transformers.Cache"
    except ImportError:
        pass  # Skip if transformers not installed


def run_test(name, fn):
    """Run a single test with error reporting."""
    try:
        fn()
        print(f"  PASS: {name}")
        return True
    except Exception as e:
        print(f"  FAIL: {name}")
        print(f"        {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running cache tests...")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print()

    results = []
    results.append(run_test("prefill returns full KV", test_prefill_returns_full_kv))
    results.append(run_test("prefill short (< num_sink)", test_prefill_short))
    results.append(run_test("prefill exact sink", test_prefill_exact_sink))
    results.append(run_test("prefill sink + window", test_prefill_sink_plus_window))
    results.append(run_test("prefill overflow window", test_prefill_overflow_window))
    results.append(run_test("decode sequential", test_decode_sequential))
    results.append(run_test("decode with eviction", test_decode_with_eviction))
    results.append(run_test("circular buffer linearization", test_circular_buffer_linearization))
    results.append(run_test("GQA shapes", test_gqa_shapes))
    results.append(run_test("multi-layer cache", test_multi_layer_cache))
    results.append(run_test("beam search reorder", test_beam_search_reorder))
    results.append(run_test("seen tokens tracking", test_seen_tokens_tracking))
    results.append(run_test("HF Cache isinstance", test_hf_cache_isinstance))

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed.")
    if passed < total:
        sys.exit(1)
