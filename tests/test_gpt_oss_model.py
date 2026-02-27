#!/usr/bin/env python3
"""
End-to-end correctness test: compare gpt-oss-20b forward pass outputs
between eager attention (reference, uses s_aux correctly) and our
sink flash attention kernel (patched via verl_patch).

Usage:
    python tests/test_gpt_oss_model.py [--seq-len 512] [--num-tokens 3]
"""

import argparse
import torch
import time


def test_gpt_oss_eager_vs_kernel(seq_len=512, num_tokens=3):
    """Compare gpt-oss forward pass: eager (reference) vs sink kernel."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_name = "openai/gpt-oss-20b"
    print(f"Loading {model_name} config...")
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"  num_layers={config.num_hidden_layers}, "
          f"num_heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, "
          f"head_dim={getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)}, "
          f"sliding_window={config.sliding_window}")
    from collections import Counter
    print(f"  layer_types: {Counter(config.layer_types)}")

    # ---- Step 1: Run with eager attention (reference) ----
    print(f"\n[1/3] Loading model with eager attention...")
    t0 = time.time()
    model_eager = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model_eager.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Generate random input
    torch.manual_seed(42)
    input_ids = torch.randint(100, 50000, (1, seq_len), device=model_eager.device)

    print(f"  Forward pass (eager, seq_len={seq_len})...")
    with torch.no_grad():
        out_eager = model_eager(input_ids)
    logits_eager = out_eager.logits.clone()

    # Check s_aux values
    for i, layer in enumerate(model_eager.model.layers):
        sinks = layer.self_attn.sinks.data
        print(f"  Layer {i:2d} ({layer.attention_type:20s}): "
              f"s_aux mean={sinks.mean():.4f}, std={sinks.std():.4f}, "
              f"min={sinks.min():.4f}, max={sinks.max():.4f}")

    # Free eager model
    del model_eager
    torch.cuda.empty_cache()

    # ---- Step 2: Run with sink flash attention kernel ----
    print(f"\n[2/3] Patching with sink flash attention kernel...")
    from sink_attention.verl_patch import patch_verl_with_sink_attention, unpatch_verl
    patch_verl_with_sink_attention()

    print(f"  Loading model with flash_attention_2 (now patched to use our kernel)...")
    t0 = time.time()
    model_kernel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_kernel.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"  Forward pass (sink kernel, seq_len={seq_len})...")
    with torch.no_grad():
        out_kernel = model_kernel(input_ids.to(model_kernel.device))
    logits_kernel = out_kernel.logits.clone()

    # Free kernel model
    del model_kernel
    torch.cuda.empty_cache()
    unpatch_verl()

    # ---- Step 3: Run with FA2 (ignores s_aux -- current broken behavior) ----
    print(f"\n[3/3] Loading model with FA2 (no patch -- s_aux ignored)...")
    t0 = time.time()
    model_fa2 = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model_fa2.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"  Forward pass (FA2, seq_len={seq_len})...")
    with torch.no_grad():
        out_fa2 = model_fa2(input_ids.to(model_fa2.device))
    logits_fa2 = out_fa2.logits.clone()

    del model_fa2
    torch.cuda.empty_cache()

    # ---- Compare ----
    print(f"\n{'='*60}")
    print(f"COMPARISON (seq_len={seq_len})")
    print(f"{'='*60}")

    # Move to same device for comparison
    logits_eager = logits_eager.float().cpu()
    logits_kernel = logits_kernel.float().cpu()
    logits_fa2 = logits_fa2.float().cpu()

    # Kernel vs Eager (should be close -- both handle s_aux)
    diff_kernel_eager = (logits_kernel - logits_eager).abs()
    print(f"\nKernel vs Eager (both handle s_aux):")
    print(f"  Max abs diff:  {diff_kernel_eager.max().item():.6f}")
    print(f"  Mean abs diff: {diff_kernel_eager.mean().item():.6f}")
    print(f"  Cosine sim:    {torch.nn.functional.cosine_similarity(logits_kernel.flatten(), logits_eager.flatten(), dim=0).item():.6f}")

    # FA2 vs Eager (FA2 ignores s_aux -- should differ more)
    diff_fa2_eager = (logits_fa2 - logits_eager).abs()
    print(f"\nFA2 vs Eager (FA2 ignores s_aux):")
    print(f"  Max abs diff:  {diff_fa2_eager.max().item():.6f}")
    print(f"  Mean abs diff: {diff_fa2_eager.mean().item():.6f}")
    print(f"  Cosine sim:    {torch.nn.functional.cosine_similarity(logits_fa2.flatten(), logits_eager.flatten(), dim=0).item():.6f}")

    # Kernel vs FA2
    diff_kernel_fa2 = (logits_kernel - logits_fa2).abs()
    print(f"\nKernel vs FA2:")
    print(f"  Max abs diff:  {diff_kernel_fa2.max().item():.6f}")
    print(f"  Mean abs diff: {diff_kernel_fa2.mean().item():.6f}")

    # Token prediction comparison
    print(f"\nTop-1 token predictions (first {num_tokens} positions):")
    for pos in range(min(num_tokens, seq_len)):
        tok_eager = logits_eager[0, pos].argmax().item()
        tok_kernel = logits_kernel[0, pos].argmax().item()
        tok_fa2 = logits_fa2[0, pos].argmax().item()
        match_ke = "✓" if tok_eager == tok_kernel else "✗"
        match_fe = "✓" if tok_eager == tok_fa2 else "✗"
        print(f"  pos {pos}: eager={tok_eager}, kernel={tok_kernel} [{match_ke}], fa2={tok_fa2} [{match_fe}]")

    # Verdict
    print(f"\n{'='*60}")
    kernel_better = diff_kernel_eager.mean().item() < diff_fa2_eager.mean().item()
    if kernel_better:
        ratio = diff_fa2_eager.mean().item() / max(diff_kernel_eager.mean().item(), 1e-10)
        print(f"PASS: Kernel is {ratio:.1f}x closer to eager than FA2")
        print(f"  -> Kernel properly handles s_aux, FA2 does not")
    else:
        print(f"WARNING: Kernel is NOT closer to eager than FA2")
        print(f"  -> Something may be wrong with s_aux handling")
    print(f"{'='*60}")

    return kernel_better


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=5)
    args = parser.parse_args()

    success = test_gpt_oss_eager_vs_kernel(
        seq_len=args.seq_len,
        num_tokens=args.num_tokens,
    )
    exit(0 if success else 1)



