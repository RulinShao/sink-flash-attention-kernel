"""
KV Cache Manager for Sink Flash Attention inference.

Provides a per-layer cache with two separate buffers:
  - Sink buffer: [B, H_kv, num_sink, D] — fixed, never evicted
  - Window buffer: [B, H_kv, window_size, D] — circular buffer, oldest evicted when full

SinkAttentionCache wraps multiple layers and extends transformers.Cache for
compatibility with HuggingFace generate().

Note: all batch elements must have the same sequence length (e.g., batch size 1
or left-padded to equal length). The cache state (write_pos, window_len) is
shared across the batch dimension.
"""

import torch
from typing import Optional, Tuple, List, Any
from abc import ABC

try:
    from transformers.cache_utils import Cache as HFCache, CacheLayerMixin
    _HAS_HF_CACHE = True
except ImportError:
    _HAS_HF_CACHE = False
    HFCache = object
    CacheLayerMixin = ABC


class SinkCacheLayer(CacheLayerMixin if _HAS_HF_CACHE else object):
    """Per-layer KV cache with sink tokens + circular sliding window buffer.

    During prefill, the first `num_sink` tokens are stored in a fixed sink buffer
    and up to `window_size` of the remaining tokens go into the window buffer.
    The full input KV is returned during prefill (for the kernel to handle masking).

    During decode, new tokens are written to the window buffer at `write_pos`,
    which advances modularly, evicting the oldest token when full.
    The linearized [sink, window] KV is returned during decode.
    """

    def __init__(
        self,
        num_sink: int,
        window_size: int,
    ):
        if _HAS_HF_CACHE:
            super().__init__()
        self.num_sink = num_sink
        self.window_size = window_size

        # Buffers initialized lazily on first update
        self.sink_k: Optional[torch.Tensor] = None
        self.sink_v: Optional[torch.Tensor] = None
        self.window_k: Optional[torch.Tensor] = None
        self.window_v: Optional[torch.Tensor] = None

        # How many valid sink slots are populated (0..num_sink)
        self.sink_len = 0
        # How many valid tokens are in the window buffer (0..window_size)
        self.window_len = 0
        # Where the next decode token will be written
        self.write_pos = 0
        # Whether prefill has happened
        self.prefilled = False
        # Track total seen tokens for seq_length reporting
        self.seen_tokens = 0

    def lazy_initialization(self, key_states: torch.Tensor):
        """Initialize buffers from tensor metadata (shape, dtype, device)."""
        B, H_kv, _, D = key_states.shape
        dtype = key_states.dtype
        device = key_states.device

        self.sink_k = torch.zeros(B, H_kv, self.num_sink, D, dtype=dtype, device=device)
        self.sink_v = torch.zeros(B, H_kv, self.num_sink, D, dtype=dtype, device=device)
        self.window_k = torch.zeros(B, H_kv, self.window_size, D, dtype=dtype, device=device)
        self.window_v = torch.zeros(B, H_kv, self.window_size, D, dtype=dtype, device=device)
        self.is_initialized = True

    def _prefill(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Store prefill KV into cache and return the FULL input KV.

        During prefill, the sink_flash_attention kernel handles masking over the
        full sequence. We store the cache state for later decode but return the
        original full KV so Q and K have matching sequence lengths.

        Args:
            k, v: [B, H_kv, N, D] — full prefill sequence

        Returns:
            k, v: [B, H_kv, N, D] — unchanged input (kernel handles masking)
        """
        N = k.shape[2]
        self.seen_tokens = N

        if N <= self.num_sink:
            # Everything fits in sink buffer
            self.sink_k[:, :, :N, :] = k
            self.sink_v[:, :, :N, :] = v
            self.sink_len = N
            self.window_len = 0
            self.write_pos = 0
        else:
            # Store sink tokens
            self.sink_k.copy_(k[:, :, :self.num_sink, :])
            self.sink_v.copy_(v[:, :, :self.num_sink, :])
            self.sink_len = self.num_sink

            # Store window tokens (last window_size tokens after sink)
            non_sink = N - self.num_sink
            if non_sink <= self.window_size:
                self.window_k[:, :, :non_sink, :] = k[:, :, self.num_sink:, :]
                self.window_v[:, :, :non_sink, :] = v[:, :, self.num_sink:, :]
                self.window_len = non_sink
                self.write_pos = non_sink % self.window_size
            else:
                # More non-sink tokens than window_size: keep the last window_size
                start = N - self.window_size
                self.window_k.copy_(k[:, :, start:, :])
                self.window_v.copy_(v[:, :, start:, :])
                self.window_len = self.window_size
                self.write_pos = 0  # buffer is full, next write wraps

        self.prefilled = True

        # Return full KV so the kernel can apply its own masking
        return k, v

    def _decode(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append a single decode token and return linearized [sink, window] KV.

        Args:
            k, v: [B, H_kv, 1, D]

        Returns:
            k_out, v_out: [B, H_kv, sink_len + window_len, D]
        """
        self.seen_tokens += 1

        # Write to circular buffer
        self.window_k[:, :, self.write_pos, :] = k[:, :, 0, :]
        self.window_v[:, :, self.write_pos, :] = v[:, :, 0, :]

        self.write_pos = (self.write_pos + 1) % self.window_size
        self.window_len = min(self.window_len + 1, self.window_size)

        return self.get_kv()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache and return KV for attention.

        During prefill: returns full input KV (kernel handles masking).
        During decode: returns linearized [sink, window] KV.

        Args:
            key_states, value_states: [B, H_kv, N_new, D]
            cache_kwargs: unused, for HF CacheLayerMixin compatibility

        Returns:
            key_states, value_states for attention computation
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        if not self.prefilled:
            return self._prefill(key_states, value_states)
        else:
            N_new = key_states.shape[2]
            if N_new == 1:
                return self._decode(key_states, value_states)
            else:
                # Multi-token decode (e.g., speculative decoding)
                for i in range(N_new):
                    result = self._decode(
                        key_states[:, :, i:i+1, :],
                        value_states[:, :, i:i+1, :],
                    )
                return result

    def get_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return linearized KV: [B, H_kv, sink_len + window_len, D].

        The circular buffer is reordered so that tokens appear in chronological
        order: oldest first, newest last.
        """
        parts_k = [self.sink_k[:, :, :self.sink_len, :]]
        parts_v = [self.sink_v[:, :, :self.sink_len, :]]

        if self.window_len > 0:
            if self.window_len < self.window_size:
                # Buffer not yet full: just take first window_len entries
                parts_k.append(self.window_k[:, :, :self.window_len, :])
                parts_v.append(self.window_v[:, :, :self.window_len, :])
            else:
                # Buffer is full: linearize from write_pos (oldest) to write_pos-1 (newest)
                if self.write_pos == 0:
                    parts_k.append(self.window_k)
                    parts_v.append(self.window_v)
                else:
                    parts_k.append(torch.cat([
                        self.window_k[:, :, self.write_pos:, :],
                        self.window_k[:, :, :self.write_pos, :],
                    ], dim=2))
                    parts_v.append(torch.cat([
                        self.window_v[:, :, self.write_pos:, :],
                        self.window_v[:, :, :self.write_pos, :],
                    ], dim=2))

        k_out = torch.cat(parts_k, dim=2)
        v_out = torch.cat(parts_v, dim=2)
        return k_out, v_out

    def get_seq_length(self) -> int:
        """Return the number of valid KV tokens (sink + window)."""
        return self.sink_len + self.window_len

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        """Return (kv_length, kv_offset) for mask construction."""
        return self.get_seq_length(), 0

    def get_max_cache_shape(self) -> int:
        """Maximum number of KV tokens that can be stored."""
        return self.num_sink + self.window_size

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder cache for beam search."""
        if self.sink_k is not None:
            device = self.sink_k.device
            beam_idx = beam_idx.to(device)
            self.sink_k = self.sink_k.index_select(0, beam_idx)
            self.sink_v = self.sink_v.index_select(0, beam_idx)
            self.window_k = self.window_k.index_select(0, beam_idx)
            self.window_v = self.window_v.index_select(0, beam_idx)


class SinkAttentionCache(HFCache if _HAS_HF_CACHE else object):
    """Multi-layer KV cache for sink flash attention inference.

    Extends transformers.Cache for compatibility with model.generate().
    Each layer uses a SinkCacheLayer with sink + circular window buffers.
    """

    def __init__(
        self,
        num_sink: int = 4,
        window_size: int = 4096,
    ):
        self.num_sink = num_sink
        self.window_size = window_size
        self._seen_tokens = 0

        if _HAS_HF_CACHE:
            # Initialize HF Cache with our layer class
            super().__init__(
                layer_class_to_replicate=None,
                layers=[],
            )
        else:
            self.layers: List[SinkCacheLayer] = []

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, idx: int) -> SinkCacheLayer:
        return self.layers[idx]

    def __repr__(self) -> str:
        return (
            f"SinkAttentionCache(num_sink={self.num_sink}, "
            f"window_size={self.window_size}, "
            f"layers={len(self.layers)}, "
            f"seen_tokens={self._seen_tokens})"
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update a specific layer's cache and return KV for attention.

        Args:
            key_states: [B, H_kv, N_new, D]
            value_states: [B, H_kv, N_new, D]
            layer_idx: which layer to update
            cache_kwargs: unused, for HF compatibility

        Returns:
            key_states, value_states for attention computation
        """
        # Lazily create layers as needed
        while len(self.layers) <= layer_idx:
            self.layers.append(SinkCacheLayer(
                num_sink=self.num_sink,
                window_size=self.window_size,
            ))

        k_out, v_out = self.layers[layer_idx].update(key_states, value_states, cache_kwargs)

        # Track seen tokens from layer 0
        if layer_idx == 0:
            self._seen_tokens = self.layers[0].seen_tokens

        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return sequence length of the cache."""
        if layer_idx < len(self.layers):
            return self.layers[layer_idx].get_seq_length()
        return 0

    def get_max_cache_length(self) -> int:
        """Maximum number of KV tokens that can be stored."""
        return self.num_sink + self.window_size

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorder all layers' caches for beam search."""
        for layer in self.layers:
            layer.reorder_cache(beam_idx)

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens
