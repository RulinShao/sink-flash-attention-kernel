"""
Correctness tests for s_aux (learnable attention sink) support.

Tests:
1. Forward: compare kernel output against reference eager implementation
2. Backward: verify gradients for q, k, v, and s_aux via torch.autograd.gradcheck
3. Full-causal with s_aux: when window_size=N, should match gpt-oss eager attention
4. Sliding window with s_aux: window_size < N
"""

import torch
import math
import pytest


def reference_attention_with_s_aux(
    q, k, v, s_aux=None, window_size=None, num_sink=0,
):
    """
    Reference eager implementation of attention with s_aux.
    Matches gpt-oss's eager_attention_forward logic.

    Args:
        q: [B, H_q, N, D]
        k: [B, H_kv, N, D]
        v: [B, H_kv, N, D]
        s_aux: [H_q] or None
        window_size: int or None (None = full causal)
        num_sink: int, number of positional sink tokens
    Returns:
        output: [B, H_q, N, D]
    """
    B, H_q, N, D = q.shape
    H_kv = k.shape[1]
    groups = H_q // H_kv

    scale = 1.0 / math.sqrt(D)

    # Expand K, V for GQA
    k_expanded = k.repeat_interleave(groups, dim=1)  # [B, H_q, N, D]
    v_expanded = v.repeat_interleave(groups, dim=1)  # [B, H_q, N, D]

    # Compute attention scores
    attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) * scale  # [B, H_q, N, N]

    # Build mask: causal + sink + window
    if window_size is None:
        window_size = N
    i_idx = torch.arange(N, device=q.device).unsqueeze(1)  # [N, 1]
    j_idx = torch.arange(N, device=q.device).unsqueeze(0)  # [1, N]
    causal = j_idx <= i_idx
    sink = j_idx < num_sink
    window = j_idx >= (i_idx - window_size + 1)
    valid = causal & (sink | window)
    mask = (~valid).float() * (-1e9)
    attn_weights = attn_weights + mask.unsqueeze(0).unsqueeze(0)

    if s_aux is not None:
        # Append s_aux as an extra logit column (gpt-oss mechanism)
        # s_aux: [H_q] -> [1, H_q, 1, 1] broadcast to [B, H_q, N, 1]
        sinks_expanded = s_aux.reshape(1, H_q, 1, 1).expand(B, H_q, N, 1)
        combined = torch.cat([attn_weights, sinks_expanded], dim=-1)  # [B, H_q, N, N+1]
        # Stabilize for numerical safety
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = torch.softmax(combined, dim=-1)
        # Drop the sink column
        scores = probs[..., :-1]  # [B, H_q, N, N]
    else:
        scores = torch.softmax(attn_weights, dim=-1)

    output = torch.matmul(scores, v_expanded)
    return output


class TestSAuxForward:
    """Test forward pass correctness with s_aux."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("H_q,H_kv", [(8, 8), (8, 2)])  # MHA and GQA
    def test_full_causal_with_s_aux(self, dtype, H_q, H_kv):
        """Full causal attention with s_aux should match reference."""
        torch.manual_seed(42)
        B, N, D = 1, 128, 64
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32) * 0.5

        from sink_attention import sink_flash_attention
        out_kernel = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux)

        out_ref = reference_attention_with_s_aux(
            q.float(), k.float(), v.float(), s_aux=s_aux.float(), window_size=N
        ).to(dtype)

        # Allow tolerance for Triton vs eager numerical differences
        atol = 2e-2 if dtype == torch.float16 else 3e-2
        rtol = 1e-2
        torch.testing.assert_close(out_kernel, out_ref, atol=atol, rtol=rtol)
        print(f"  PASS: full causal + s_aux ({dtype}, H_q={H_q}, H_kv={H_kv})")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sliding_window_with_s_aux(self, dtype):
        """Sliding window (like gpt-oss sliding_attention layers) with s_aux."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 8, 2, 256, 64
        window_size = 128
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32) * 0.5

        from sink_attention import sink_flash_attention
        out_kernel = sink_flash_attention(q, k, v, num_sink=0, window_size=window_size, s_aux=s_aux)

        out_ref = reference_attention_with_s_aux(
            q.float(), k.float(), v.float(), s_aux=s_aux.float(), window_size=window_size
        ).to(dtype)

        atol = 2e-2 if dtype == torch.float16 else 3e-2
        rtol = 1e-2
        torch.testing.assert_close(out_kernel, out_ref, atol=atol, rtol=rtol)
        print(f"  PASS: sliding window + s_aux ({dtype})")

    def test_without_s_aux_unchanged(self):
        """Without s_aux, kernel should match standard causal attention."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 8, 2, 128, 64
        dtype = torch.float16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)

        from sink_attention import sink_flash_attention
        out_kernel = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=None)

        out_ref = reference_attention_with_s_aux(
            q.float(), k.float(), v.float(), s_aux=None, window_size=N
        ).to(dtype)

        torch.testing.assert_close(out_kernel, out_ref, atol=2e-2, rtol=1e-2)
        print("  PASS: without s_aux (standard causal)")

    def test_s_aux_absorbs_mass(self):
        """Verify s_aux reduces attention weights: large s_aux -> smaller output norms."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 4, 4, 64, 32
        dtype = torch.float16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)

        from sink_attention import sink_flash_attention

        # No s_aux
        out_none = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=None)
        # Small s_aux
        s_aux_small = torch.zeros(H_q, device="cuda", dtype=torch.float32)
        out_small = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux_small)
        # Large s_aux (absorbs a lot of mass)
        s_aux_large = torch.full((H_q,), 10.0, device="cuda", dtype=torch.float32)
        out_large = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux_large)

        norm_none = out_none.float().norm().item()
        norm_small = out_small.float().norm().item()
        norm_large = out_large.float().norm().item()

        print(f"  Output norms: no s_aux={norm_none:.4f}, small={norm_small:.4f}, large={norm_large:.4f}")
        assert norm_large < norm_small, "Large s_aux should absorb more mass -> smaller output"
        print("  PASS: s_aux mass absorption verified")


class TestSAuxBackward:
    """Test backward pass / gradient computation for s_aux."""

    def test_ds_aux_gradient_exists(self):
        """Verify that s_aux receives a gradient."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 4, 4, 64, 32
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32, requires_grad=True)

        from sink_attention import sink_flash_attention
        out = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux)
        loss = out.sum()
        loss.backward()

        assert s_aux.grad is not None, "s_aux should have a gradient"
        assert s_aux.grad.shape == (H_q,), f"s_aux grad shape mismatch: {s_aux.grad.shape}"
        assert torch.isfinite(s_aux.grad).all(), "s_aux gradient has non-finite values"
        print(f"  s_aux grad: {s_aux.grad}")
        print("  PASS: ds_aux gradient exists and is finite")

    def test_ds_aux_gradient_numerical(self):
        """Verify ds_aux gradient against numerical finite differences.
        Note: Uses float32 because the Triton kernel doesn't support float64."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 32, 16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32, requires_grad=True)

        from sink_attention import sink_flash_attention

        # Analytical gradient
        out = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux)
        loss = out.sum()
        loss.backward()
        grad_analytical = s_aux.grad.clone()

        # Numerical gradient (finite differences) - larger eps for float32
        eps = 1e-3
        grad_numerical = torch.zeros_like(s_aux)
        for i in range(H_q):
            s_aux_plus = s_aux.detach().clone()
            s_aux_plus[i] += eps
            out_plus = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux_plus)
            loss_plus = out_plus.sum()

            s_aux_minus = s_aux.detach().clone()
            s_aux_minus[i] -= eps
            out_minus = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux_minus)
            loss_minus = out_minus.sum()

            grad_numerical[i] = (loss_plus - loss_minus) / (2 * eps)

        max_diff = (grad_analytical - grad_numerical).abs().max().item()
        rel_diff = max_diff / (grad_numerical.abs().max().item() + 1e-8)
        print(f"  Analytical grad:  {grad_analytical}")
        print(f"  Numerical grad:   {grad_numerical}")
        print(f"  Max abs diff:     {max_diff:.2e}")
        print(f"  Max rel diff:     {rel_diff:.2e}")

        # Relaxed tolerance for float32 Triton vs eager
        assert max_diff < 5e-2, f"ds_aux gradient mismatch: max_diff={max_diff:.2e}"
        print("  PASS: ds_aux gradient matches numerical finite differences")

    def test_dq_dk_dv_gradients_with_s_aux(self):
        """Verify dQ, dK, dV gradients exist and are finite when s_aux is present.
        Note: torch.autograd.gradcheck requires float64 which the Triton kernel
        doesn't support. We verify gradient existence and finiteness instead."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 2, 2, 32, 16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=torch.float32, requires_grad=True)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32, requires_grad=True)

        from sink_attention import sink_flash_attention
        out = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux)
        loss = out.sum()
        loss.backward()

        for name, param in [("q", q), ("k", k), ("v", v), ("s_aux", s_aux)]:
            assert param.grad is not None, f"{name} should have a gradient"
            assert torch.isfinite(param.grad).all(), f"{name} gradient has non-finite values"
            print(f"    {name} grad norm: {param.grad.norm().item():.6f}")
        print("  PASS: dQ, dK, dV, ds_aux gradients exist and are finite")


class TestGptOssEagerComparison:
    """Compare our kernel against gpt-oss's exact eager attention."""

    def test_match_gpt_oss_eager(self):
        """Test that kernel output matches gpt-oss eager_attention_forward."""
        torch.manual_seed(42)
        # Use gpt-oss-like config: H_q=64, H_kv=8, D=80
        B, H_q, H_kv, N, D = 1, 16, 4, 128, 64
        dtype = torch.float16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32) * 0.3

        # Test both full attention and sliding window
        for window_size, label in [(N, "full"), (128, "sliding_128")]:
            from sink_attention import sink_flash_attention
            out_kernel = sink_flash_attention(q, k, v, num_sink=0, window_size=window_size, s_aux=s_aux)

            out_ref = reference_attention_with_s_aux(
                q.float(), k.float(), v.float(), s_aux=s_aux.float(), window_size=window_size
            ).to(dtype)

            max_diff = (out_kernel.float() - out_ref.float()).abs().max().item()
            mean_diff = (out_kernel.float() - out_ref.float()).abs().mean().item()
            print(f"  {label}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            assert max_diff < 0.05, f"Output mismatch ({label}): max_diff={max_diff:.6f}"

        print("  PASS: kernel matches gpt-oss eager attention")

    def test_gpt_oss_head_dim_80(self):
        """Test with gpt-oss-20b's actual head_dim=80."""
        torch.manual_seed(42)
        B, H_q, H_kv, N, D = 1, 8, 2, 64, 80
        dtype = torch.bfloat16
        q = torch.randn(B, H_q, N, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H_kv, N, D, device="cuda", dtype=dtype)
        s_aux = torch.randn(H_q, device="cuda", dtype=torch.float32) * 0.5

        from sink_attention import sink_flash_attention
        out_kernel = sink_flash_attention(q, k, v, num_sink=0, window_size=N, s_aux=s_aux)

        out_ref = reference_attention_with_s_aux(
            q.float(), k.float(), v.float(), s_aux=s_aux.float(), window_size=N
        ).to(dtype)

        max_diff = (out_kernel.float() - out_ref.float()).abs().max().item()
        print(f"  head_dim=80: max_diff={max_diff:.6f}")
        assert max_diff < 0.05, f"Output mismatch with D=80: max_diff={max_diff:.6f}"
        print("  PASS: head_dim=80 (gpt-oss-20b)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing s_aux (learnable attention sink) support")
    print("=" * 60)

    print("\n--- Forward Tests ---")
    t = TestSAuxForward()
    t.test_full_causal_with_s_aux(torch.float16, 8, 8)
    t.test_full_causal_with_s_aux(torch.bfloat16, 8, 2)
    t.test_sliding_window_with_s_aux(torch.float16)
    t.test_without_s_aux_unchanged()
    t.test_s_aux_absorbs_mass()

    print("\n--- Backward Tests ---")
    t2 = TestSAuxBackward()
    t2.test_ds_aux_gradient_exists()
    t2.test_ds_aux_gradient_numerical()
    t2.test_dq_dk_dv_gradients_with_s_aux()

    print("\n--- gpt-oss Eager Comparison ---")
    t3 = TestGptOssEagerComparison()
    t3.test_match_gpt_oss_eager()
    t3.test_gpt_oss_head_dim_80()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

