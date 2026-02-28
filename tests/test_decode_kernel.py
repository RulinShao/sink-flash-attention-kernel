"""
Tests for the Triton FlashDecoding kernel with s_aux support.

Compares against a naive PyTorch reference that materializes the full
attention matrix (correct but memory-hungry for long contexts).
"""

import torch
import math
import pytest

from sink_attention.decode_kernel import sink_decode_attention


# ============================================================================
# Reference implementation (PyTorch, materializes full attention matrix)
# ============================================================================

def reference_decode_attention(q, k, v, s_aux=None):
    """
    Naive single-query attention for correctness comparison.

    Args:
        q: [B, H_q, 1, D]
        k: [B, H_kv, N_kv, D]
        v: [B, H_kv, N_kv, D]
        s_aux: [H_q] or None

    Returns:
        output: [B, H_q, 1, D]
    """
    B, H_q, _, D = q.shape
    H_kv = k.shape[1]
    scale = 1.0 / math.sqrt(D)

    # GQA expansion
    if H_q != H_kv:
        groups = H_q // H_kv
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

    # scores: [B, H_q, 1, N_kv]
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale

    if s_aux is not None:
        # Add s_aux as extra column: [B, H_q, 1, 1+N_kv]
        s_aux_col = s_aux.float()[None, :, None, None].expand(B, -1, 1, 1)
        scores_aug = torch.cat([s_aux_col, scores], dim=-1)
        attn = torch.softmax(scores_aug, dim=-1)
        attn = attn[:, :, :, 1:]  # Drop s_aux column (no associated V)
    else:
        attn = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn.to(v.dtype), v)
    return output


# ============================================================================
# Tests
# ============================================================================

class TestDecodeNoSAux:
    """Test decode kernel without s_aux (standard attention)."""

    @pytest.mark.parametrize("N_kv", [64, 256, 1024, 4096])
    def test_basic(self, N_kv):
        B, H_q, H_kv, D = 1, 8, 8, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        out_kernel = sink_decode_attention(q, k, v)
        out_ref = reference_decode_attention(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)

    def test_fp16(self):
        B, H_q, H_kv, D, N_kv = 1, 8, 8, 128, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.float16)

        out_kernel = sink_decode_attention(q, k, v)
        out_ref = reference_decode_attention(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)

    def test_batch_size_2(self):
        B, H_q, H_kv, D, N_kv = 2, 8, 8, 128, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        out_kernel = sink_decode_attention(q, k, v)
        out_ref = reference_decode_attention(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("D", [64, 128, 256])
    def test_head_dims(self, D):
        B, H_q, H_kv, N_kv = 1, 4, 4, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        out_kernel = sink_decode_attention(q, k, v)
        out_ref = reference_decode_attention(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)


class TestDecodeGQA:
    """Test decode kernel with grouped-query attention (H_q != H_kv)."""

    @pytest.mark.parametrize("H_q,H_kv", [(16, 4), (32, 8), (8, 1)])
    def test_gqa(self, H_q, H_kv):
        B, D, N_kv = 1, 128, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        out_kernel = sink_decode_attention(q, k, v)
        out_ref = reference_decode_attention(q, k, v)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)


class TestDecodeWithSAux:
    """Test decode kernel with s_aux (learnable attention sink)."""

    @pytest.mark.parametrize("N_kv", [64, 256, 1024, 4096])
    def test_s_aux_correctness(self, N_kv):
        """Verify s_aux-aware decode matches reference."""
        B, H_q, H_kv, D = 1, 8, 8, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.bfloat16) * 2

        out_kernel = sink_decode_attention(q, k, v, s_aux=s_aux)
        out_ref = reference_decode_attention(q, k, v, s_aux=s_aux)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)

    def test_s_aux_gqa(self):
        """s_aux with GQA."""
        B, H_q, H_kv, D, N_kv = 1, 32, 8, 128, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.bfloat16) * 3

        out_kernel = sink_decode_attention(q, k, v, s_aux=s_aux)
        out_ref = reference_decode_attention(q, k, v, s_aux=s_aux)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)

    def test_s_aux_absorbs_mass(self):
        """Large s_aux should suppress attention to all KV tokens."""
        B, H_q, H_kv, D, N_kv = 1, 4, 4, 128, 256
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        # Without s_aux
        out_no_sink = sink_decode_attention(q, k, v, s_aux=None)

        # With large s_aux: attention is almost entirely absorbed by sink
        s_aux_large = torch.full((H_q,), 100.0, device="cuda", dtype=torch.bfloat16)
        out_large_sink = sink_decode_attention(q, k, v, s_aux=s_aux_large)

        # Output should be near zero (all attention mass goes to s_aux virtual key)
        assert out_large_sink.abs().max().item() < 0.01, (
            f"Large s_aux should suppress output, got max={out_large_sink.abs().max().item()}"
        )
        # And significantly different from no-sink output
        diff = (out_no_sink - out_large_sink).abs().max().item()
        assert diff > 0.01, f"s_aux should change output, diff={diff}"

    def test_s_aux_zero_is_noop_equivalent(self):
        """s_aux=0 should give similar (but not identical) results to no s_aux,
        since exp(0)=1 adds a constant denominator term."""
        B, H_q, H_kv, D, N_kv = 1, 4, 4, 128, 512
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)

        s_aux_zero = torch.zeros(H_q, device="cuda", dtype=torch.bfloat16)
        out_zero = sink_decode_attention(q, k, v, s_aux=s_aux_zero)
        out_none = sink_decode_attention(q, k, v, s_aux=None)

        # s_aux=0 adds exp(0)=1 to denominator, so output should be slightly
        # attenuated compared to no s_aux. They won't be identical.
        # Just check they're in the same ballpark.
        diff = (out_zero - out_none).abs().max().item()
        assert diff < 1.0, f"s_aux=0 vs None should be similar, diff={diff}"


class TestDecodeLongContext:
    """Test with long context lengths (memory efficiency)."""

    @pytest.mark.parametrize("N_kv", [8192, 16384])
    def test_long_kv(self, N_kv):
        B, H_q, H_kv, D = 1, 32, 8, 128
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.bfloat16) * 2

        out_kernel = sink_decode_attention(q, k, v, s_aux=s_aux)
        out_ref = reference_decode_attention(q, k, v, s_aux=s_aux)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=2e-2, rtol=2e-2)

    def test_non_block_aligned(self):
        """N_kv not divisible by BLOCK_N."""
        B, H_q, H_kv, D = 1, 8, 8, 128
        N_kv = 300  # Not aligned to BLOCK_N=256
        torch.manual_seed(42)
        q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(B, H_kv, N_kv, D, device="cuda", dtype=torch.bfloat16)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.bfloat16)

        out_kernel = sink_decode_attention(q, k, v, s_aux=s_aux)
        out_ref = reference_decode_attention(q, k, v, s_aux=s_aux)

        torch.testing.assert_close(out_kernel, out_ref.to(q.dtype), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


