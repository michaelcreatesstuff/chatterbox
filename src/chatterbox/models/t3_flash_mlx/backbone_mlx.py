# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Llama backbone for the Chatterbox-Flash block-diffusion T3 decoder (MLX).

The parameter layout matches HF ``LlamaModel`` (and ``t3_mlx.modules.llama_mlx``)
so the PyTorch ``tfmr.*`` weights load by name. What differs from the
autoregressive backbone is the cache and the attention masks:

* The KV cache is preallocated per layer as ``(B, n_kv_heads, max_seq, head_dim)``
  and written with slice assignment. A tentative block step writes its K/V
  into ``[cs:cs+L]`` (overwriting the previous step) and attends over
  ``[:cs+L]``, so there is no crop, concatenation or reallocation per step.
* Prefix forwards are causal; block forwards see the full committed cache plus
  every position of their own block (bidirectional within the block), which is
  what the upstream torch SDPA engine does with its all-zero 4D mask.

Block forwards also have a functional, fixed-shape form
(:meth:`FlashLlamaMLX.block_forward`) for ``mx.compile``: the cache start is an
``mx.array``, K/V are written with ``mx.slice_update`` at that start, attention
runs over a static window ``[:attn_len]`` of the cache with a mask that hides
columns at or beyond the visible end, and the updated cache arrays are returned
instead of assigned. One compiled trace then serves every step of every block
whose window falls in the same ``attn_len`` bucket. Batched multi-chunk
decoding passes a per-row mask (:func:`batch_visibility_mask`) and per-row RoPE
offsets instead; the cache write position stays shared.
"""

import logging
from typing import List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from ..t3_mlx.modules.llama_mlx import LlamaConfigMLX, RMSNorm, RoPE, MLP

logger = logging.getLogger(__name__)

# Cache sizes grow in buckets of this many positions so that small changes in
# prompt length don't force a reallocation.
CACHE_BUCKET = 512

MaskType = Union[None, str, mx.array]


class FlashKVCache:
    """Preallocated per-layer KV cache, reused across ``generate()`` calls."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq: int,
        dtype: mx.Dtype,
    ):
        shape = (batch_size, n_kv_heads, max_seq, head_dim)
        self.keys: List[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(n_layers)
        ]
        self.values: List[mx.array] = [
            mx.zeros(shape, dtype=dtype) for _ in range(n_layers)
        ]
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.dtype = dtype
        mx.eval(self.keys, self.values)

    @property
    def nbytes(self) -> int:
        return sum(k.nbytes + v.nbytes for k, v in zip(self.keys, self.values))

    def fits(self, batch_size: int, needed: int, dtype: mx.Dtype) -> bool:
        return (
            self.batch_size == batch_size
            and self.max_seq >= needed
            and self.dtype == dtype
        )


def bucket_len(needed: int) -> int:
    """Round ``needed`` up to the next multiple of :data:`CACHE_BUCKET`."""
    return ((int(needed) + CACHE_BUCKET - 1) // CACHE_BUCKET) * CACHE_BUCKET


def fused_commit_mask(cache_start: int, len_commit: int, len_next: int) -> mx.array:
    """Boolean mask for the fused "commit + next block step 0" forward.

    Rows ``[0, len_commit)`` are the committed block: they see the cache and
    their own block. Rows ``[len_commit, len_commit + len_next)`` are the next
    (all-MASK) block: they see the cache, the committed block and themselves.
    """
    kv_len = cache_start + len_commit + len_next
    rows = mx.arange(len_commit + len_next)[:, None]
    cols = mx.arange(kv_len)[None, :]
    return (cols < cache_start + len_commit) | (rows >= len_commit)


def block_visibility_mask(
    cache_start: mx.array, n_rows: int, n_first: int, attn_len: int
) -> mx.array:
    """Boolean ``(n_rows, attn_len)`` mask for a fixed-shape block forward.

    Row ``r`` sees cache columns ``< cache_start + (n_first if r < n_first else
    n_rows)``. ``n_first == n_rows`` is a plain block step (every row sees the
    committed cache plus the whole block); ``n_first < n_rows`` is the fused
    commit + next block forward of :func:`fused_commit_mask`.
    """
    rows = mx.arange(n_rows)[:, None]
    ends = cache_start + mx.where(rows < n_first, n_first, n_rows)
    return mx.arange(attn_len)[None, :] < ends


def batch_visibility_mask(
    cache_start: mx.array, pad: mx.array, ends: mx.array, attn_len: int
) -> mx.array:
    """Per-row boolean ``(B, 1, 1, attn_len)`` mask for a batched block forward.

    Row ``b`` is left-padded by ``pad[b]`` cache positions and its block holds
    ``ends[b]`` real tokens, so it sees columns ``[pad[b], cache_start +
    ends[b])``: its own prefix, its committed blocks and its whole block.
    """
    cols = mx.arange(attn_len)[None, :]
    vis = (cols >= pad[:, None]) & (cols < cache_start + ends[:, None])
    return vis[:, None, None, :]


def padded_prefix_mask(pad: mx.array, length: int) -> mx.array:
    """Causal ``(B, 1, length, length)`` mask over left-padded prefixes.

    A real query sees the real (non-pad) columns up to itself. A pad query
    sees only itself, so no row is fully masked (its output is unused).
    """
    rows = mx.arange(length)[:, None]
    cols = mx.arange(length)[None, :]
    real = cols[None] >= pad[:, None, None]
    vis = ((cols <= rows)[None] & real) | (cols == rows)[None]
    return vis[:, None]


class FlashAttention(nn.Module):
    """Llama attention writing into a preallocated :class:`FlashKVCache`."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=bias
        )
        self.rope = RoPE(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: MaskType,
        cache: Optional[FlashKVCache],
        layer_idx: int,
    ) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        q, k = self.rope(q, k, offset=offset)

        if cache is not None:
            end = offset + L
            keys = cache.keys[layer_idx]
            values = cache.values[layer_idx]
            keys[:, :, offset:end, :] = k
            values[:, :, offset:end, :] = v
            k = keys[:, :, :end, :]
            v = values[:, :, :end, :]

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)

    def functional(
        self,
        x: mx.array,
        cache_start: mx.array,
        mask: mx.array,
        keys: mx.array,
        values: mx.array,
        attn_len: int,
        rope_offset: Optional[mx.array] = None,
    ):
        """Fixed-shape block attention; returns ``(out, new_keys, new_values)``.

        ``rope_offset`` (default ``cache_start``) may be a per-row ``(B,)``
        int32 array: rows then sit at different RoPE positions while their K/V
        share the cache write position ``cache_start``.
        """
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        offset = cache_start if rope_offset is None else rope_offset
        q, k = self.rope(q, k, offset=offset)
        start = cache_start.reshape(1)
        keys = mx.slice_update(keys, k, start, axes=(2,))
        values = mx.slice_update(values, v, start, axes=(2,))
        out = mx.fast.scaled_dot_product_attention(
            q,
            keys[:, :, :attn_len],
            values[:, :, :attn_len],
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), keys, values


class FlashTransformerBlock(nn.Module):
    """Pre-norm Llama block (attention + SwiGLU MLP)."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.self_attn = FlashAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x, offset, mask, cache, layer_idx):
        h = x + self.self_attn(self.input_layernorm(x), offset, mask, cache, layer_idx)
        return h + self.mlp(self.post_attention_layernorm(h))


class FlashLlamaMLX(nn.Module):
    """Llama stack with a preallocated cache and explicit attention masks."""

    def __init__(self, config: LlamaConfigMLX):
        super().__init__()
        self.config = config
        self.layers = [
            FlashTransformerBlock(config) for _ in range(config.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def make_cache(
        self, batch_size: int, max_seq: int, dtype: mx.Dtype
    ) -> FlashKVCache:
        cfg = self.config
        return FlashKVCache(
            cfg.num_hidden_layers,
            batch_size,
            cfg.num_key_value_heads,
            cfg.head_dim,
            max_seq,
            dtype,
        )

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: Optional[FlashKVCache] = None,
        offset: int = 0,
        mask: MaskType = None,
    ) -> mx.array:
        """Run the stack and return the final-norm hidden states ``(B, L, D)``.

        Args:
            inputs_embeds: ``(B, L, D)`` input embeddings.
            cache: KV cache to write ``[offset:offset+L]`` into and attend over
                ``[:offset+L]``. ``None`` attends only over the inputs.
            offset: Position of the first input (RoPE and cache write offset).
            mask: ``None`` (everything visible), ``"causal"`` or a boolean /
                additive array broadcastable to ``(B, H, L, offset+L)``.
        """
        h = inputs_embeds
        for i, layer in enumerate(self.layers):
            h = layer(h, offset, mask, cache, i)
        return self.norm(h)

    def block_forward(
        self,
        inputs_embeds: mx.array,
        cache_start: mx.array,
        keys: List[mx.array],
        values: List[mx.array],
        attn_len: int,
        n_first: Optional[int] = None,
        mask: Optional[mx.array] = None,
        rope_offset: Optional[mx.array] = None,
    ):
        """Functional block forward with fixed shapes (for ``mx.compile``).

        Writes K/V for ``[cache_start, cache_start + L)`` into copies of
        ``keys`` / ``values`` (in place when the caller has dropped its
        references, see :class:`~.t3_flash_mlx.T3FlashMLX`) and attends over
        ``[:attn_len]`` under :func:`block_visibility_mask`.

        Args:
            inputs_embeds: ``(B, L, D)``.
            cache_start: int32 0-d array, position of the first input.
            keys, values: Per-layer cache arrays ``(B, H_kv, max_seq, hd)``.
            attn_len: Static attended length, ``>= cache_start + L``.
            n_first: Rows of the first (committed) sub-block; ``None`` = ``L``.
            mask: Boolean mask broadcastable to ``(B, H, L, attn_len)``; replaces
                :func:`block_visibility_mask` (per-row masks of batched
                decoding, see :func:`batch_visibility_mask`).
            rope_offset: Per-row ``(B,)`` RoPE offsets; ``None`` = ``cache_start``.

        Returns:
            ``(hidden, new_keys, new_values)`` with final-norm ``hidden``.
        """
        L = inputs_embeds.shape[1]
        n_first = L if n_first is None else n_first
        if mask is None:
            mask = block_visibility_mask(cache_start, L, n_first, attn_len)
        h = inputs_embeds
        new_keys, new_values = [], []
        for layer, k, v in zip(self.layers, keys, values):
            a, k, v = layer.self_attn.functional(
                layer.input_layernorm(h),
                cache_start,
                mask,
                k,
                v,
                attn_len,
                rope_offset,
            )
            h = h + a
            h = h + layer.mlp(layer.post_attention_layernorm(h))
            new_keys.append(k)
            new_values.append(v)
        return self.norm(h), new_keys, new_values
