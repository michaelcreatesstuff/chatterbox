# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
GPT-2 backbone of Chatterbox-Turbo / Nano (MLX), with a double-buffered KV cache.

The parameter layout matches HF ``GPT2Model`` (``h.N.attn.c_attn``,
``h.N.mlp.c_fc``, ``wpe``, ``ln_f``), so the torch ``tfmr.*`` weights load by
name. HF stores its ``Conv1D`` weights as ``(in, out)``; they are transposed
into ``nn.Linear``'s ``(out, in)`` on load (see :func:`is_conv1d_weight`).
Blocks are pre-LN (``ln_1`` -> attention, ``ln_2`` -> MLP, both residual), the
MLP uses ``gelu_new`` (the tanh approximation) and the learned ``wpe`` is added
to the input embeddings at each token's absolute position. ``wte`` is not used:
T3 feeds ``inputs_embeds``.

Cache layout, per row: ``[cond (Lc) | pad | text | SOS | speech ...]``.

* Every T3 sequence starts with the same conditioning block (speaker
  projection + 375 prompt speech tokens) at positions ``0..Lc-1``, and
  attention is causal, so its K/V depend on the voice only. They are computed
  once per voice at batch 1 (:meth:`GPT2ModelMLX.cond_forward`) and copied
  into every row; the prefill then runs only each row's ``[text | SOS]``. The
  prefill is compute-bound and the conditioning is most of the prefix, so a
  batched prefill is several times cheaper this way.
* Batched rows left-pad their text after the cond block: row ``b``'s column
  ``c >= Lc`` holds position ``c - pad[b]``, and its masks hide columns
  ``[Lc, Lc + pad[b])``.

Double buffering (:class:`TurboKVCache` holds two full buffers): MLX updates
an array in place only when no queued kernel still holds it, so with pipelined
decoding (step ``k+1`` enqueued while step ``k`` runs) a step that writes the
buffer the step in flight reads makes a full copy of it. Step ``k`` instead
writes buffer ``k % 2``: its own column plus the previous step's column
(which went to the other buffer), then attends over that buffer with the fused
SDPA kernel. The caller waits for step ``k-2`` (the last reader of buffer
``k % 2``, normally finished already) before enqueuing step ``k``, so the GPU
always has the next step queued and every write stays in place.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .configs import GPT2ConfigMLX


def is_conv1d_weight(key: str) -> bool:
    """True for HF GPT-2 ``Conv1D`` weights, stored ``(in, out)``."""
    return key.startswith("tfmr.h.") and key.endswith(
        ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    )


class TurboKVCache:
    """Two per-layer KV buffers ``(B, H, max_seq, hd)``, reused across calls."""

    def __init__(
        self,
        n_layers: int,
        batch_size: int,
        n_heads: int,
        head_dim: int,
        max_seq: int,
        dtype: mx.Dtype,
    ):
        shape = (batch_size, n_heads, max_seq, head_dim)
        # buffers[i] = (keys, values), each a per-layer list
        self.buffers: List = [
            (
                [mx.zeros(shape, dtype) for _ in range(n_layers)],
                [mx.zeros(shape, dtype) for _ in range(n_layers)],
            )
            for _ in range(2)
        ]
        self.batch_size = batch_size
        self.max_seq = max_seq
        self.dtype = dtype
        mx.eval(self.buffers)

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for ks, vs in self.buffers for a in ks + vs)

    def fits(self, batch_size: int, needed: int, dtype: mx.Dtype) -> bool:
        return (
            self.batch_size == batch_size
            and self.max_seq >= needed
            and self.dtype == dtype
        )

    def take(self, i: int):
        """Remove and return buffer ``i`` so the caller holds its only reference."""
        buf = self.buffers[i]
        self.buffers[i] = None
        return buf


def prefill_mask(pad: mx.array, lc: int, length: int) -> mx.array:
    """Boolean ``(B, 1, L, Lc + L)`` mask for the rows' ``[pad | text | SOS]``.

    A real query sees the cond block and the real columns up to itself; a pad
    query sees the cond block and itself (its output is unused).
    """
    rows = mx.arange(length)[:, None]
    cols = mx.arange(lc + length)[None, :] - lc  # row-relative column
    real = cols[None] >= pad[:, None, None]
    own = ((cols <= rows)[None] & real) | (cols == rows)[None]
    vis = own | (cols < 0)[None]
    return vis[:, None]


def decode_mask(pad: mx.array, lc: int, col: mx.array, attn_len: int) -> mx.array:
    """Boolean ``(B, 1, 1, attn_len)`` mask of a decode step at column ``col``.

    Row ``b`` sees the cond block and columns ``[Lc + pad[b], col]``.
    """
    cols = mx.arange(attn_len)[None, :]
    vis = (cols < lc) | ((cols >= lc + pad[:, None]) & (cols <= col))
    return vis[:, None, None, :]


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2ConfigMLX):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        D = config.n_embd
        self.c_attn = nn.Linear(D, 3 * D)
        self.c_proj = nn.Linear(D, D)

    def qkv(self, x: mx.array):
        B, L, _ = x.shape
        qkv = self.c_attn(x).reshape(B, L, 3, self.n_head, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, hd)
        return qkv[0], qkv[1], qkv[2]

    def out(self, out: mx.array) -> mx.array:
        B, _, L, _ = out.shape
        return self.c_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))

    def cond(self, x: mx.array):
        """Causal attention over the cond block alone; returns ``(out, k, v)``."""
        q, k, v = self.qkv(x)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal"
        )
        return self.out(out), k, v

    def prefill(self, x, mask, lc, cond_k, cond_v, bufs):
        """Write cond + ``[0, L)`` K/V into every buffer and attend over ``[:Lc+L]``.

        ``bufs`` is a list of ``(keys, values)`` arrays of this layer; the
        updated pairs are returned in the same order.
        """
        B, L, _ = x.shape
        q, k, v = self.qkv(x)
        ck = mx.broadcast_to(cond_k, (B,) + cond_k.shape[1:])
        cv = mx.broadcast_to(cond_v, (B,) + cond_v.shape[1:])
        zero = mx.array([0], mx.int32)
        start = mx.array([lc], mx.int32)
        new = []
        for keys, values in bufs:
            keys = mx.slice_update(keys, ck, zero, axes=(2,))
            values = mx.slice_update(values, cv, zero, axes=(2,))
            keys = mx.slice_update(keys, k, start, axes=(2,))
            values = mx.slice_update(values, v, start, axes=(2,))
            new.append((keys, values))
        keys, values = new[0]
        out = mx.fast.scaled_dot_product_attention(
            q, keys[:, :, : lc + L], values[:, :, : lc + L], scale=self.scale, mask=mask
        )
        return self.out(out), new

    def decode(self, x, col, prev_col, prev_k, prev_v, mask, keys, values, attn_len):
        """One-token step: write ``prev`` at ``prev_col`` and own K/V at ``col``.

        Returns ``(out, own_k, own_v, keys, values)``.
        """
        q, k, v = self.qkv(x)
        keys = mx.slice_update(keys, prev_k, prev_col, axes=(2,))
        values = mx.slice_update(values, prev_v, prev_col, axes=(2,))
        keys = mx.slice_update(keys, k, col, axes=(2,))
        values = mx.slice_update(values, v, col, axes=(2,))
        out = mx.fast.scaled_dot_product_attention(
            q, keys[:, :, :attn_len], values[:, :, :attn_len], scale=self.scale, mask=mask
        )
        return self.out(out), k, v, keys, values


class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2ConfigMLX):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.inner_dim)
        self.c_proj = nn.Linear(config.inner_dim, config.n_embd)

    def __call__(self, x: mx.array) -> mx.array:
        # gelu_new = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        return self.c_proj(nn.gelu_approx(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2ConfigMLX):
        super().__init__()
        eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=eps)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=eps)
        self.mlp = GPT2MLP(config)


class GPT2ModelMLX(nn.Module):
    """GPT-2 stack fed with ``inputs_embeds`` (``wte`` is not used)."""

    def __init__(self, config: GPT2ConfigMLX):
        super().__init__()
        self.config = config
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = [GPT2Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def make_cache(self, batch_size: int, max_seq: int, dtype: mx.Dtype) -> TurboKVCache:
        cfg = self.config
        return TurboKVCache(
            cfg.n_layer, batch_size, cfg.n_head, cfg.head_dim, max_seq, dtype
        )

    def cond_forward(self, cond_embeds: mx.array):
        """K/V of the cond block ``(1, Lc, D)`` at positions ``0..Lc-1``.

        Returns per-layer ``(cond_keys, cond_values)``, each ``(1, H, Lc, hd)``.
        The block's own outputs are not needed (the prefix continues with
        text), so the last layer's residual and MLP are skipped.
        """
        Lc = cond_embeds.shape[1]
        h = cond_embeds + self.wpe.weight[:Lc].astype(cond_embeds.dtype)
        cond_k, cond_v = [], []
        for i, block in enumerate(self.h):
            a, k, v = block.attn.cond(block.ln_1(h))
            cond_k.append(k)
            cond_v.append(v)
            if i < len(self.h) - 1:
                h = h + a
                h = h + block.mlp(block.ln_2(h))
                mx.eval(h, k, v)  # bounds peak memory (see prefill)
        return cond_k, cond_v

    def prefill(
        self,
        inputs_embeds: mx.array,
        positions: mx.array,
        mask: mx.array,
        cond_k: List[mx.array],
        cond_v: List[mx.array],
        buffers: List,
        eval_layers: bool = False,
    ):
        """Forward of each row's ``[pad | text | SOS]`` after the cond block.

        Args:
            inputs_embeds: ``(B, L, D)``.
            positions: ``(B, L)`` int32 ``wpe`` positions.
            mask: From :func:`prefill_mask`.
            cond_k, cond_v: From :meth:`cond_forward`.
            buffers: List of ``(keys, values)`` per-layer lists; each gets the
                cond K/V at ``[0, Lc)`` and the rows' K/V at ``[Lc, Lc + L)``.
            eval_layers: Evaluate after every layer. MLX keeps a command
                buffer's inputs alive until it completes, so one lazy graph
                over all layers holds every layer's activations at once
                (~10x the peak memory of a per-layer eval at B=4).

        Returns:
            ``(hidden, new_buffers)`` with ``ln_f``-normed ``hidden``.
        """
        lc = int(cond_k[0].shape[2])
        h = inputs_embeds + self.wpe(positions).astype(inputs_embeds.dtype)
        per_buf = [([], []) for _ in buffers]
        for i, block in enumerate(self.h):
            layer_bufs = [(ks[i], vs[i]) for ks, vs in buffers]
            a, new = block.attn.prefill(
                block.ln_1(h), mask, lc, cond_k[i], cond_v[i], layer_bufs
            )
            del layer_bufs
            h = h + a
            h = h + block.mlp(block.ln_2(h))
            if eval_layers:
                mx.eval(h, new)
            for (ks, vs), (k, v) in zip(per_buf, new):
                ks.append(k)
                vs.append(v)
        return self.ln_f(h), per_buf

    def decode(
        self,
        inputs_embeds: mx.array,
        positions: mx.array,
        col: mx.array,
        prev_col: mx.array,
        prev_k: List[mx.array],
        prev_v: List[mx.array],
        mask: mx.array,
        keys: List[mx.array],
        values: List[mx.array],
        attn_len: int,
    ):
        """One-token forward over one buffer.

        Args:
            inputs_embeds: ``(B, 1, D)``.
            positions: ``(B,)`` int32 ``wpe`` positions.
            col, prev_col: ``(1,)`` int32 columns of this and the previous token.
            prev_k, prev_v: Per-layer ``(B, H, 1, hd)`` K/V of the previous
                token (the previous step's ``own`` outputs).
            mask: From :func:`decode_mask`.
            keys, values: Per-layer buffer arrays (window ``[:attn_len]``).
            attn_len: Static attended window.

        Returns:
            ``(hidden, own_k, own_v, new_keys, new_values)``.
        """
        h = inputs_embeds + self.wpe(positions)[:, None].astype(inputs_embeds.dtype)
        own_k, own_v, new_k, new_v = [], [], [], []
        layers = zip(self.h, prev_k, prev_v, keys, values)
        for block, pk, pv, k, v in layers:
            a, ok, ov, k, v = block.attn.decode(
                block.ln_1(h), col, prev_col, pk, pv, mask, k, v, attn_len
            )
            h = h + a
            h = h + block.mlp(block.ln_2(h))
            own_k.append(ok)
            own_v.append(ov)
            new_k.append(k)
            new_v.append(v)
        return self.ln_f(h), own_k, own_v, new_k, new_v


__all__ = [
    "GPT2ModelMLX",
    "TurboKVCache",
    "decode_mask",
    "is_conv1d_weight",
    "prefill_mask",
]
