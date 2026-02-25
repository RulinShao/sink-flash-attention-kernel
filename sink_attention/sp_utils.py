"""
Sequence parallelism utilities for Sink Flash Attention.

When using sequence parallelism (SP) in FSDP training (e.g., verl), the sequence
dimension is split across devices. Each device holds a local chunk of size N/P
where P is the SP world size.

Problem: Sink tokens (positions [0, num_sink)) reside on rank 0, but ALL ranks
need them for attention. These utilities handle broadcasting sink KV pairs and
reducing their gradients.

Usage:
    # Before attention
    k_local, v_local = prepare_sink_kv_for_sp(k_local, v_local, num_sink, sp_group)

    # Attention (each rank uses adjusted num_sink and local positions)
    out = sink_flash_attention(q_local, k_local, v_local, num_sink=num_sink, window_size=W)

    # After backward
    dk_local, dv_local = reduce_sink_kv_grads(dk_local, dv_local, num_sink, sp_group)
"""

import torch
import torch.distributed as dist
from typing import Optional


def prepare_sink_kv_for_sp(
    k: torch.Tensor,
    v: torch.Tensor,
    num_sink: int,
    sp_group: dist.ProcessGroup,
    rank: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast sink token KV pairs from rank 0 to all ranks in the SP group,
    and prepend them to the local KV tensors on non-zero ranks.

    Args:
        k: Local key tensor [B, H_kv, N_local, D]
        v: Local value tensor [B, H_kv, N_local, D]
        num_sink: Number of sink tokens
        sp_group: Process group for sequence parallelism
        rank: Optional override for rank (for testing)

    Returns:
        (k_with_sinks, v_with_sinks): KV tensors with sink tokens available.
        - On rank 0: returns original tensors (sinks already at start)
        - On other ranks: returns tensors with sink KV prepended
    """
    if num_sink == 0:
        return k, v

    if rank is None:
        rank = dist.get_rank(sp_group)

    B, H_kv, N_local, D = k.shape

    # Extract or allocate sink KV buffers
    if rank == 0:
        sink_k = k[:, :, :num_sink].contiguous()
        sink_v = v[:, :, :num_sink].contiguous()
    else:
        sink_k = torch.empty(B, H_kv, num_sink, D, device=k.device, dtype=k.dtype)
        sink_v = torch.empty(B, H_kv, num_sink, D, device=v.device, dtype=v.dtype)

    # Broadcast from rank 0
    dist.broadcast(sink_k, src=dist.get_global_rank(sp_group, 0), group=sp_group)
    dist.broadcast(sink_v, src=dist.get_global_rank(sp_group, 0), group=sp_group)

    if rank == 0:
        # Rank 0 already has sinks at the start of its local chunk
        return k, v
    else:
        # Prepend sink KV to local chunk
        k_with_sinks = torch.cat([sink_k, k], dim=2)
        v_with_sinks = torch.cat([sink_v, v], dim=2)
        return k_with_sinks, v_with_sinks


def reduce_sink_kv_grads(
    dk: torch.Tensor,
    dv: torch.Tensor,
    num_sink: int,
    sp_group: dist.ProcessGroup,
    rank: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    All-reduce gradients for sink KV pairs across SP ranks.

    Each rank computes dK/dV for the sink tokens (since all ranks attend to them).
    These gradients must be summed back to rank 0.

    Args:
        dk: Local dK tensor [B, H_kv, N_local_with_sinks, D]
        dv: Local dV tensor [B, H_kv, N_local_with_sinks, D]
        num_sink: Number of sink tokens
        sp_group: Process group for sequence parallelism
        rank: Optional override for rank

    Returns:
        (dk, dv): Adjusted gradient tensors.
        - On rank 0: sink gradients are summed from all ranks
        - On other ranks: sink prefix is stripped (returns only local gradients)
    """
    if num_sink == 0:
        return dk, dv

    if rank is None:
        rank = dist.get_rank(sp_group)

    # Extract sink gradients from all ranks
    sink_dk = dk[:, :, :num_sink].contiguous()
    sink_dv = dv[:, :, :num_sink].contiguous()

    # Sum across all SP ranks
    dist.all_reduce(sink_dk, op=dist.ReduceOp.SUM, group=sp_group)
    dist.all_reduce(sink_dv, op=dist.ReduceOp.SUM, group=sp_group)

    if rank == 0:
        # Update sink gradients in place
        dk = dk.clone()
        dv = dv.clone()
        dk[:, :, :num_sink] = sink_dk
        dv[:, :, :num_sink] = sink_dv
        return dk, dv
    else:
        # Strip the prepended sink KV gradients, keep only local
        return dk[:, :, num_sink:], dv[:, :, num_sink:]


def get_local_position_offset(rank: int, n_local: int, num_sink: int) -> int:
    """
    Compute the global position offset for a given SP rank.

    When using sequence parallelism, each rank holds a contiguous chunk of the
    sequence. This function returns the global start position of the local chunk
    (excluding prepended sink tokens on non-zero ranks).

    Args:
        rank: SP rank
        n_local: Local sequence length (before prepending sinks)
        num_sink: Number of sink tokens

    Returns:
        Global position offset for the first non-sink token on this rank.
    """
    return rank * n_local


class SinkAttentionSPWrapper(torch.nn.Module):
    """
    Wrapper that handles sequence parallelism for sink flash attention.

    Usage:
        wrapper = SinkAttentionSPWrapper(num_sink=4, window_size=4096, sp_group=sp_group)
        output = wrapper(q_local, k_local, v_local)
    """

    def __init__(self, num_sink: int = 4, window_size: int = 4096,
                 sp_group: Optional[dist.ProcessGroup] = None):
        super().__init__()
        self.num_sink = num_sink
        self.window_size = window_size
        self.sp_group = sp_group

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        from .sink_flash_attention import sink_flash_attention

        if self.sp_group is None or dist.get_world_size(self.sp_group) == 1:
            # No SP, just call directly
            return sink_flash_attention(q, k, v, self.num_sink, self.window_size)

        # Broadcast sink KV to all ranks
        k_sp, v_sp = prepare_sink_kv_for_sp(k, v, self.num_sink, self.sp_group)

        # Run attention with local queries against (sinks + local) KV
        out = sink_flash_attention(q, k_sp, v_sp, self.num_sink, self.window_size)

        return out
