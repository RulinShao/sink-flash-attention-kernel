"""Measure numerical accuracy of sink flash attention vs reference."""
import torch
import math
from sink_flash_attention import sink_flash_attention


def naive_sink_attention(q, k, v, num_sink, window_size):
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


print("Numerical Accuracy: Triton kernel vs fp32 reference (eager attention)")
print("=" * 85)
print(f"{'Config':>50} | {'Max Abs Err':>12} | {'Mean Abs Err':>12} | {'Cosine Sim':>10}")
print(f"{'-'*50}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

configs = [
    # (B, H_q, H_kv, N, D, num_sink, window_size, dtype, label)
    (1, 4, 4, 256, 64, 4, 64, torch.float16, "MHA fp16 N=256"),
    (1, 4, 4, 512, 64, 4, 64, torch.float16, "MHA fp16 N=512"),
    (1, 4, 4, 1024, 64, 4, 128, torch.float16, "MHA fp16 N=1024"),
    (1, 4, 4, 2048, 64, 4, 256, torch.float16, "MHA fp16 N=2048"),
    (1, 8, 2, 512, 64, 4, 64, torch.float16, "GQA 4:1 fp16 N=512"),
    (1, 32, 8, 512, 128, 4, 128, torch.float16, "GQA 4:1 fp16 D=128 N=512"),
    (1, 4, 4, 512, 64, 4, 64, torch.bfloat16, "MHA bf16 N=512"),
    (1, 4, 4, 512, 64, 0, 64, torch.float16, "pure window fp16 N=512"),
    (1, 4, 4, 512, 64, 16, 128, torch.float16, "16 sinks fp16 N=512"),
    (1, 4, 4, 512, 64, 4, 1, torch.float16, "sink+self only fp16 N=512"),
]

for B, H_q, H_kv, N, D, num_sink, window_size, dtype, label in configs:
    torch.manual_seed(42)
    q = torch.randn(B, H_q, N, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H_kv, N, D, device='cuda', dtype=dtype)

    out_triton = sink_flash_attention(q, k, v, num_sink=num_sink, window_size=window_size)
    out_ref = naive_sink_attention(q.float(), k.float(), v.float(), num_sink, window_size).to(dtype)

    diff = (out_triton.float() - out_ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_triton.float().reshape(-1).unsqueeze(0),
        out_ref.float().reshape(-1).unsqueeze(0)
    ).item()

    print(f"{label:>50} | {max_err:12.6f} | {mean_err:12.6f} | {cos_sim:10.8f}")

# Gradient accuracy
print()
print("Gradient Accuracy: Triton kernel (fp16) vs fp32 reference")
print("=" * 85)
print(f"{'Config':>50} | {'dQ MaxErr':>10} | {'dK MaxErr':>10} | {'dV MaxErr':>10}")
print(f"{'-'*50}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

grad_configs = [
    (1, 4, 4, 128, 64, 4, 32, "MHA N=128 sink=4 win=32"),
    (1, 4, 4, 256, 64, 4, 64, "MHA N=256 sink=4 win=64"),
    (1, 8, 2, 256, 64, 4, 64, "GQA N=256 sink=4 win=64"),
    (1, 4, 4, 256, 128, 4, 64, "MHA D=128 N=256 sink=4 win=64"),
]

for B, H_q, H_kv, N, D, num_sink, window_size, label in grad_configs:
    torch.manual_seed(42)
    q_ref = torch.randn(B, H_q, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn(B, H_kv, N, D, device='cuda', dtype=torch.float32, requires_grad=True)
    out_ref = naive_sink_attention(q_ref, k_ref, v_ref, num_sink, window_size)
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)

    q_tri = q_ref.detach().half().requires_grad_(True)
    k_tri = k_ref.detach().half().requires_grad_(True)
    v_tri = v_ref.detach().half().requires_grad_(True)
    out_tri = sink_flash_attention(q_tri, k_tri, v_tri, num_sink=num_sink, window_size=window_size)
    out_tri.backward(grad_out.half())

    dq_err = (q_tri.grad.float() - q_ref.grad).abs().max().item()
    dk_err = (k_tri.grad.float() - k_ref.grad).abs().max().item()
    dv_err = (v_tri.grad.float() - v_ref.grad).abs().max().item()

    print(f"{label:>50} | {dq_err:10.6f} | {dk_err:10.6f} | {dv_err:10.6f}")
