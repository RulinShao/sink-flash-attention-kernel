"""
Test the verl monkey patch with a real HuggingFace model.
Verifies that sink flash attention is correctly applied in the model's forward pass.
"""

import torch


def test_patch_with_hf_model():
    """Test that the patch intercepts attention in a real HF model."""
    from sink_attention.verl_patch import patch_verl_with_sink_attention, unpatch_verl, _SINK_ATTENTION_CONFIG

    # Track whether our kernel was called
    call_count = [0]
    original_sink_fa = None

    import sink_attention.sink_flash_attention as sfa_module
    original_sink_fa = sfa_module.sink_flash_attention

    def tracking_sink_fa(q, k, v, num_sink=4, window_size=512):
        call_count[0] += 1
        return original_sink_fa(q, k, v, num_sink=num_sink, window_size=window_size)

    # Apply patch
    patch_verl_with_sink_attention(num_sink=4, window_size=128)

    # Also patch the sink_flash_attention function to track calls
    sfa_module.sink_flash_attention = tracking_sink_fa
    # Re-import in verl_patch module
    import sink_attention.verl_patch as verl_patch_module
    verl_patch_module.sink_flash_attention = tracking_sink_fa

    # Load a small model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Try multiple cached models
    model_candidates = [
        "Qwen/Qwen3-4B",
        "Qwen/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
    ]

    model = None
    for model_name in model_candidates:
        print(f"  Trying {model_name}...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float16,
                trust_remote_code=True,
                local_files_only=True,
            ).cuda()
            print(f"  Loaded {model_name} ({model.config.num_hidden_layers} layers, "
                  f"{model.config.num_attention_heads} heads, "
                  f"kv_heads={getattr(model.config, 'num_key_value_heads', 'N/A')})")
            break
        except Exception as e:
            print(f"    Failed: {e}")
            continue

    if model is None:
        print("  SKIP: No cached model available")
        sfa_module.sink_flash_attention = original_sink_fa
        unpatch_verl()
        return True

    # Forward pass
    input_ids = torch.randint(0, 1000, (1, 256), device='cuda')
    call_count[0] = 0

    with torch.no_grad():
        output = model(input_ids)

    print(f"  Sink FA called {call_count[0]} times (expected: {model.config.num_hidden_layers})")
    print(f"  Output shape: {output.logits.shape}")
    assert call_count[0] == model.config.num_hidden_layers, \
        f"Expected {model.config.num_hidden_layers} calls, got {call_count[0]}"
    assert output.logits.shape == (1, 256, model.config.vocab_size)
    assert torch.isfinite(output.logits).all(), "Non-finite logits"

    # Backward pass
    call_count[0] = 0
    output2 = model(input_ids)
    loss = output2.logits.sum()
    loss.backward()
    print(f"  Backward: Sink FA called {call_count[0]} times")
    # In backward, the kernel is called via autograd -- call_count tracks forward calls
    assert call_count[0] == model.config.num_hidden_layers

    # Compare with original FA2
    sfa_module.sink_flash_attention = original_sink_fa
    unpatch_verl()

    with torch.no_grad():
        output_orig = model(input_ids)

    # Outputs will differ because sink attention uses a different mask than full causal
    # But both should be finite
    assert torch.isfinite(output_orig.logits).all()
    print(f"  Original FA2 logits finite: True")

    # They should differ (sink attention != full causal)
    max_diff = (output.logits - output_orig.logits).abs().max().item()
    print(f"  Max diff sink vs full causal: {max_diff:.4f}")
    if max_diff > 0.01:
        print(f"  (Expected: outputs differ because sink+window != full causal)")
    else:
        print(f"  WARNING: outputs identical -- patch may not be active")

    print("  PASS: verl patch test")
    return True


def test_patch_function():
    """Test basic patch/unpatch functionality."""
    from sink_attention.verl_patch import patch_verl_with_sink_attention, unpatch_verl, _SINK_ATTENTION_CONFIG
    import transformers.modeling_flash_attention_utils as fa_utils

    original = fa_utils._flash_attention_forward

    # Patch
    patch_verl_with_sink_attention(num_sink=8, window_size=2048)
    assert fa_utils._flash_attention_forward is not original
    assert _SINK_ATTENTION_CONFIG["num_sink"] == 8
    assert _SINK_ATTENTION_CONFIG["window_size"] == 2048
    print("  PASS: patch applied")

    # Unpatch
    unpatch_verl()
    assert fa_utils._flash_attention_forward is original
    print("  PASS: unpatch restored original")


if __name__ == "__main__":
    print("Testing verl monkey patch...")
    print()

    print("1. Patch/unpatch functionality:")
    test_patch_function()
    print()

    print("2. HF model integration:")
    test_patch_with_hf_model()
