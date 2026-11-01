# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX port of ``ChatterboxFlashT3`` — the Chatterbox-Flash block-diffusion T3.

Reference: ``chatterbox_flash.model.ChatterboxFlashT3.generate`` (chatterbox-flash
0.1.0, MIT, Resemble AI) with the torch SDPA engine. The decoder is the
Chatterbox T3 (Llama-520M, conditioning encoder, text / speech embeddings and
speech head) with one extra ``[MASK]`` row in ``speech_emb``.

Generation, for a single utterance:

1. The prefix ``[cond | text | SOS]`` runs causally and seeds the KV cache.
   With CFG (``cfg_scale > 0``) the batch has two rows: the conditional prefix
   and a null prefix of the same length (zeroed conditioning, zeroed text,
   real SOS). Both rows always share the same speech tokens.
2. Speech is decoded in blocks of ``block_size`` tokens. Each step embeds the
   block (``speech_emb`` + block-relative ``speech_pos_emb``) at RoPE positions
   ``cs..cs+bl-1``, attends over the committed cache plus the whole block, and
   takes shifted logits: block position ``i`` reads the hidden state at
   ``i-1``, where ``-1`` is ``shift_ctx``.
3. Up to ``num_steps`` steps per block unmask positions ranked by
   prior-calibrated PMI (see :mod:`.sampling_mlx`). The block stops early when
   nothing is masked or it holds EOS.
4. A commit forward of the clean block writes its KV and yields the next
   ``shift_ctx`` (its last hidden state). Generation stops after the first
   block that contains EOS.

Performance notes: everything in the loop stays on MLX; the only host sync per
step is a 2-element ``[n_mask, has_eos]`` read. The commit forward is the same
computation as the next tentative step (same tokens, same cache offset), so by
default the forward for the new tokens is enqueued *before* that read and
serves as either the next step or the commit (``pipeline=True``). The
alternative ``fuse_commit=True`` merges the commit with the next block's
step-0 forward into one ``2*bl``-token forward under a block-causal mask.

Compiled block forwards (``compiled=True``, the default): each block forward
(embedding, 30 layers, shift-by-one and speech head) is one ``mx.compile``d
function of fixed shapes. The cache start is an array, K/V go in with
``mx.slice_update`` and attention covers a static window ``[:attn_len]`` of
the preallocated cache, with ``attn_len`` rounded up to :attr:`attn_bucket`
positions. So one trace serves every step of every block in a window bucket,
and traces are reused across ``generate()`` calls. The cache arrays are passed
in and returned; the caller drops its references first so that MLX donates the
buffers and the update stays in place (no per-step cache copy).

Batched multi-chunk decoding (:meth:`T3FlashMLX.generate_batch`): a forward
costs nearly the same for 2 or 8 rows, so several chunks of one long text run
as rows of one batch. Prefixes are left-padded to a common length, so all rows
write K/V at one shared cache position (one ``slice_update``, in place); each
row gets its own RoPE offset (``mx.fast.rope`` takes a ``(B,)`` offset array)
and its own visibility mask. Chunks advance in lockstep block rounds.
"""

import logging
import math
import secrets
from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..t3.llama_configs import LLAMA_CONFIGS
from ..t3.modules.t3_config import T3Config
from ..t3_mlx.modules.cond_enc_mlx import T3CondEncMLX, T3CondMLX
from ..t3_mlx.modules.learned_pos_emb_mlx import LearnedPositionEmbeddingsMLX
from ..t3_mlx.modules.llama_mlx import LlamaConfigMLX
from .backbone_mlx import (
    FlashKVCache,
    FlashLlamaMLX,
    batch_visibility_mask,
    bucket_len,
    fused_commit_mask,
    padded_prefix_mask,
)
from .sampling_mlx import (
    StepFnCache,
    make_batch_step_fn,
    omnivoice_unmask_schedule,
    time_shift_quantiles,
)

logger = logging.getLogger(__name__)

# Compiled block forwards attend over the cache window rounded up to this many
# positions; each window size is one trace.
ATTN_BUCKET = 256

DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def resolve_dtype(dtype: Union[str, mx.Dtype]) -> mx.Dtype:
    if isinstance(dtype, str):
        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {list(DTYPES)}, got {dtype!r}")
        return DTYPES[dtype]
    return dtype


class T3FlashMLX(nn.Module):
    """Block-diffusion T3 decoder on MLX (Chatterbox-Flash)."""

    def __init__(self, hp: Optional[T3Config] = None, block_size: int = 16):
        super().__init__()
        if hp is None:
            hp = T3Config.english_only()
        if hp.input_pos_emb != "learned":
            raise NotImplementedError("T3FlashMLX expects learned position embeddings")
        self.hp = hp
        self.cfg = LlamaConfigMLX.from_dict(LLAMA_CONFIGS[hp.llama_config_name])
        self.dim = self.cfg.hidden_size
        self.block_size = int(block_size)

        self.tfmr = FlashLlamaMLX(self.cfg)
        self.cond_enc = T3CondEncMLX(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        # One extra row for the [MASK] token (id = speech_tokens_dict_size).
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size + 1, self.dim)
        self.text_pos_emb = LearnedPositionEmbeddingsMLX(
            hp.max_text_tokens + 2, self.dim
        )
        self.speech_pos_emb = LearnedPositionEmbeddingsMLX(
            hp.max_speech_tokens + 4, self.dim
        )
        self.speech_head = nn.Linear(self.dim, hp.speech_tokens_dict_size, bias=False)

        # Inference state (underscore names keep them out of parameters()).
        self._uncond_prior: Optional[mx.array] = None
        self._cache: Optional[FlashKVCache] = None
        # Second cache for pipelined compiled decoding (see get_twin_cache).
        self._twin: Optional[FlashKVCache] = None
        # Main + twin caches of batched multi-chunk decoding (generate_batch),
        # kept apart so single and batched calls don't reallocate each other's.
        self._batch_caches: List[FlashKVCache] = []
        self._step_fns = StepFnCache()
        self._block_fns: Dict[tuple, object] = {}
        self._quantize_bits: Optional[int] = None
        # Attended-window bucket of the compiled block forward (0 = whole cache).
        self.attn_bucket = ATTN_BUCKET

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def mask_token_id(self) -> int:
        return self.hp.speech_tokens_dict_size

    @property
    def dtype(self) -> mx.Dtype:
        return self.speech_emb.weight.dtype

    @property
    def cache_nbytes(self) -> int:
        caches = [self._cache, self._twin] + list(self._batch_caches)
        return sum(c.nbytes for c in caches if c is not None)

    def release_caches(self) -> None:
        """Free the KV caches and compiled traces; the next call rebuilds them.

        The caches are kept between calls and only grow (to the longest
        chunk times the largest batch seen), so a long document can leave
        several GB allocated.
        """
        self._cache = self._twin = None
        self._batch_caches = []
        self._block_fns = {}
        mx.clear_cache()

    # ------------------------------------------------------------------ #
    #  Weights
    # ------------------------------------------------------------------ #

    def load_torch_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """Load a ``ChatterboxFlashT3`` (or plain T3) torch state dict.

        Keys map 1:1 (``tfmr.layers.N.self_attn.q_proj.weight`` etc.). Torch-only
        entries (``text_head``, ``tfmr.embed_tokens``, rotary buffers) are
        dropped. A plain-T3 ``speech_emb`` with ``V`` rows gets a zero
        ``[MASK]`` row appended (only useful for tests).
        """
        from mlx.utils import tree_flatten

        if "model" in state_dict:  # legacy layout
            state_dict = state_dict["model"][0]

        expected = dict(tree_flatten(self.parameters()))
        weights = {}
        for k, v in state_dict.items():
            if k not in expected:
                continue
            arr = v.detach().cpu().float().numpy() if hasattr(v, "detach") else v
            weights[k] = mx.array(np.asarray(arr))

        emb = weights.get("speech_emb.weight")
        V = self.hp.speech_tokens_dict_size
        if emb is not None and emb.shape[0] == V:
            logger.info("Expanding speech_emb from %d to %d (adding [MASK]).", V, V + 1)
            emb = mx.concatenate([emb, mx.zeros((1, emb.shape[1]), emb.dtype)])
            weights["speech_emb.weight"] = emb

        missing = sorted(set(expected) - set(weights))
        dropped = sorted(set(state_dict) - set(weights))
        if dropped:
            logger.debug("Ignoring %d torch-only keys: %s", len(dropped), dropped[:8])
        if missing and strict:
            raise ValueError(f"Missing {len(missing)} T3 weights: {missing[:10]}")
        for k, v in weights.items():
            if tuple(v.shape) != tuple(expected[k].shape):
                raise ValueError(
                    f"Shape mismatch for {k}: checkpoint {tuple(v.shape)}, "
                    f"model {tuple(expected[k].shape)}"
                )
        self.load_weights(list(weights.items()), strict=False)
        self._block_fns = {}

    def finalize(
        self,
        dtype: Union[str, mx.Dtype] = "float16",
        quantize_bits: Optional[int] = None,
        group_size: int = 64,
    ) -> "T3FlashMLX":
        """Cast to ``dtype``, optionally quantize, and prime the PMI prior.

        Quantization (weight-only, ``bits`` 4 or 8) covers the Llama linears and
        the speech head; embeddings and the conditioning encoder stay in
        ``dtype``.
        """
        self.set_dtype(resolve_dtype(dtype))
        if quantize_bits is not None:
            if quantize_bits not in (4, 8):
                raise ValueError("quantize_bits must be None, 4 or 8")
            nn.quantize(
                self.tfmr,
                group_size=group_size,
                bits=quantize_bits,
                class_predicate=lambda _, m: isinstance(m, nn.Linear),
            )
            self.speech_head = nn.QuantizedLinear.from_linear(
                self.speech_head, group_size=group_size, bits=quantize_bits
            )
            self._quantize_bits = quantize_bits
        self.eval()
        mx.eval(self.parameters())
        self._cache = self._twin = None
        self._batch_caches = []
        # Compiled traces capture the weights; rebuild them after a change.
        self._block_fns = {}
        self.prime_uncond_block_prior()
        return self

    # ------------------------------------------------------------------ #
    #  Embeddings
    # ------------------------------------------------------------------ #

    def _speech_pos(self, n: int) -> mx.array:
        return self.speech_pos_emb.emb.weight[:n]

    def embed_speech_block(self, tokens: mx.array) -> mx.array:
        """``speech_emb(tokens) + speech_pos_emb(0..L-1)`` for ``(L,)`` tokens."""
        return self.speech_emb(tokens) + self._speech_pos(tokens.shape[-1])

    def prepare_conditioning(self, t3_cond: T3CondMLX) -> mx.array:
        """Conditioning embeddings ``(1, len_cond, D)``; ``t3_cond`` is not modified."""
        emb = t3_cond.cond_prompt_speech_emb
        tokens = t3_cond.cond_prompt_speech_tokens
        if tokens is not None and emb is None:
            emb = self.speech_emb(tokens) + self._speech_pos(tokens.shape[-1])
        cond = T3CondMLX(
            speaker_emb=t3_cond.speaker_emb,
            clap_emb=t3_cond.clap_emb,
            cond_prompt_speech_tokens=tokens,
            cond_prompt_speech_emb=emb,
            emotion_adv=t3_cond.emotion_adv,
        )
        return self.cond_enc(cond)

    def build_prefix(
        self, cond_emb: mx.array, text_tokens: mx.array, cfg: bool
    ) -> mx.array:
        """``[cond | text | SOS]`` (and the null row when ``cfg``) in model dtype."""
        if text_tokens.ndim == 1:
            text_tokens = text_tokens[None]
        T = text_tokens.shape[1]
        text_emb = self.text_emb(text_tokens) + self.text_pos_emb.emb.weight[:T]
        sos = mx.array([[self.hp.start_speech_token]])
        sos_emb = self.speech_emb(sos) + self._speech_pos(1)
        dtype = self.dtype
        cond_emb = cond_emb.astype(dtype)
        text_emb = text_emb.astype(dtype)
        sos_emb = sos_emb.astype(dtype)
        prefix = mx.concatenate([cond_emb, text_emb, sos_emb], axis=1)
        if cfg:
            # zero_all null prefix: conditioning and (unpadded) text zeroed
            null = mx.concatenate(
                [mx.zeros_like(cond_emb), mx.zeros_like(text_emb), sos_emb], axis=1
            )
            prefix = mx.concatenate([prefix, null], axis=0)
        return prefix

    # ------------------------------------------------------------------ #
    #  Forwards
    # ------------------------------------------------------------------ #

    def get_cache(self, batch_size: int, needed: int) -> FlashKVCache:
        """Reuse the preallocated cache if it fits, else grow it (512 buckets)."""
        dtype = self.dtype
        cache = self._cache
        if cache is not None and cache.fits(batch_size, needed, dtype):
            return cache
        size = bucket_len(needed)
        if cache is not None and cache.batch_size == batch_size:
            size = max(size, cache.max_seq)
        self._cache = self._twin = None
        del cache
        logger.debug("Allocating Flash KV cache: batch=%d max_seq=%d", batch_size, size)
        self._cache = self.tfmr.make_cache(batch_size, size, dtype)
        return self._cache

    def get_twin_cache(self) -> FlashKVCache:
        """A second cache shaped like :meth:`get_cache`'s, for pipelined decoding.

        MLX updates a buffer in place only when no queued kernel still reads
        it. A pipelined step enqueues forward ``k+1`` while forward ``k`` (which
        reads the block's K/V) is in flight, so writing the same cache would
        copy it on every step. Alternating two caches keeps every write in
        place; committed blocks are copied to the other cache (``bl`` rows).
        """
        main = self._cache
        twin = self._twin
        if twin is None or not (
            twin.fits(main.batch_size, main.max_seq, main.dtype)
            and twin.max_seq == main.max_seq
        ):
            self._twin = None
            del twin
            self._twin = self.tfmr.make_cache(main.batch_size, main.max_seq, main.dtype)
        return self._twin

    def copy_cache_rows(
        self,
        src: FlashKVCache,
        dst: FlashKVCache,
        start: int,
        n_rows: int,
        compile_fn: bool = True,
    ) -> None:
        """Copy K/V rows ``[start, start + n_rows)`` of every layer from ``src`` to ``dst``."""
        key = ("copy", int(n_rows), bool(compile_fn))
        fn = self._block_fns.get(key)
        if fn is None:

            def fn(src_k, src_v, dst_k, dst_v, start):
                out = []
                for s_, d_ in zip(src_k + src_v, dst_k + dst_v):
                    sizes = s_.shape[:2] + (n_rows,) + s_.shape[3:]
                    rows = mx.slice(s_, start, axes=(2,), slice_size=sizes)
                    out.append(mx.slice_update(d_, rows, start, axes=(2,)))
                return out

            if compile_fn:
                fn = mx.compile(fn)
            self._block_fns[key] = fn
        dst_k, dst_v = dst.keys, dst.values
        dst.keys = dst.values = None
        out = fn(src.keys, src.values, dst_k, dst_v, mx.array([start], mx.int32))
        del dst_k, dst_v
        n = len(out) // 2
        dst.keys, dst.values = out[:n], out[n:]

    def forward_prefix(self, prefix_emb: mx.array, cache: FlashKVCache) -> mx.array:
        """Causal prefix forward at offset 0; returns hidden ``(B, lp, D)``."""
        return self.tfmr(prefix_emb, cache, offset=0, mask="causal")

    def forward_block(
        self, block_emb: mx.array, cache: FlashKVCache, cache_start: int, mask=None
    ) -> mx.array:
        """Block forward at ``cache_start`` (bidirectional within the block)."""
        return self.tfmr(block_emb, cache, offset=cache_start, mask=mask)

    def shifted_logits(self, shift_ctx: mx.array, hidden: mx.array) -> mx.array:
        """Float32 logits ``(B, bl, V)`` for a block from its shifted hidden states."""
        bl = hidden.shape[1]
        shift_hidden = mx.concatenate([shift_ctx, hidden[:, : bl - 1]], axis=1)
        return self.speech_head(shift_hidden).astype(mx.float32)

    def attn_len_for(self, cache_start: int, n_rows: int, max_seq: int) -> int:
        """Static attended window for a compiled forward of ``n_rows`` at ``cache_start``."""
        bucket = int(self.attn_bucket)
        if bucket <= 0:
            return max_seq
        need = cache_start + n_rows
        return min(max_seq, ((need + bucket - 1) // bucket) * bucket)

    def block_fn(self, n_rows: int, n_first: int, attn_len: int, compile_fn=True):
        """Compiled ``(tokens, shift_ctx, cs, keys, values) -> (logits, ctx, keys, values)``.

        ``tokens`` ``(n_rows,)`` int32 are embedded with block-relative speech
        positions (restarting at ``n_first`` for a fused commit + next block),
        run through :meth:`FlashLlamaMLX.block_forward` at the int32 0-d
        ``cs``, and turned into float32 logits for the last block's
        ``n_rows - n_first`` (or ``n_rows``) positions with the shift-by-one
        rule. ``ctx`` is the hidden state of row ``n_first - 1``: the next
        ``shift_ctx`` once these tokens are committed.
        """
        key = (n_rows, n_first, attn_len, bool(compile_fn))
        fn = self._block_fns.get(key)
        if fn is not None:
            return fn
        n_next = n_rows - n_first

        def fn(tokens, shift_ctx, cache_start, keys, values):
            B = shift_ctx.shape[0]
            pos = self.speech_pos_emb.emb.weight
            if n_next:
                pos = mx.concatenate([pos[:n_first], pos[:n_next]])
            else:
                pos = pos[:n_rows]
            emb = (self.speech_emb(tokens) + pos).astype(self.dtype)
            emb = mx.broadcast_to(emb[None], (B,) + emb.shape)
            hidden, keys, values = self.tfmr.block_forward(
                emb, cache_start, keys, values, attn_len, n_first
            )
            if n_next:
                shifted = hidden[:, n_first - 1 : n_rows - 1]
            else:
                shifted = mx.concatenate([shift_ctx, hidden[:, : n_rows - 1]], axis=1)
            logits = self.speech_head(shifted).astype(mx.float32)
            return logits, hidden[:, n_first - 1 : n_first], keys, values

        if compile_fn:
            fn = mx.compile(fn)
        self._block_fns[key] = fn
        return fn

    def run_block_fn(
        self,
        cache: FlashKVCache,
        tokens: mx.array,
        shift_ctx: mx.array,
        cache_start: int,
        n_first: Optional[int] = None,
        compile_fn: bool = True,
    ):
        """Run :meth:`block_fn` on ``cache`` and swap in the updated K/V arrays.

        The cache drops its references before the call, so the old arrays are
        held only by the graph and MLX updates them in place.
        """
        n_rows = int(tokens.shape[0])
        n_first = n_rows if n_first is None else int(n_first)
        attn_len = self.attn_len_for(cache_start, n_rows, cache.max_seq)
        fn = self.block_fn(n_rows, n_first, attn_len, compile_fn)
        keys, values = cache.keys, cache.values
        cache.keys = cache.values = None
        try:
            logits, ctx, new_keys, new_values = fn(
                tokens, shift_ctx, mx.array(cache_start, dtype=mx.int32), keys, values
            )
        except Exception:
            cache.keys, cache.values = keys, values
            raise
        del keys, values
        cache.keys, cache.values = new_keys, new_values
        return logits, ctx

    def get_batch_caches(self, batch_size: int, needed: int) -> List[FlashKVCache]:
        """Main and twin caches ``(batch_size, max_seq)`` for :meth:`generate_batch`.

        Same reuse and 512-bucket growth as :meth:`get_cache`; a different
        ``batch_size`` (number of rows) reallocates both.
        """
        dtype = self.dtype
        caches = self._batch_caches
        if caches and caches[0].fits(batch_size, needed, dtype):
            return caches
        size = bucket_len(needed)
        if caches and caches[0].batch_size == batch_size:
            size = max(size, caches[0].max_seq)
        self._batch_caches = []
        del caches
        logger.debug(
            "Allocating batched Flash KV caches: batch=%d max_seq=%d", batch_size, size
        )
        self._batch_caches = [
            self.tfmr.make_cache(batch_size, size, dtype) for _ in range(2)
        ]
        return self._batch_caches

    def batch_block_fn(
        self, n_chunks: int, rows_per_chunk: int, attn_len: int, compile_fn=True
    ):
        """Compiled batched block forward for :meth:`generate_batch`.

        ``(tokens, shift_ctx, cs, pad, ends, keys, values) -> (logits, ctx,
        keys, values)`` with ``tokens`` ``(n_chunks, BS)`` int32 (chunk ``c``
        is embedded once and repeated over its ``rows_per_chunk`` rows),
        ``shift_ctx`` ``(B, 1, D)``, the shared int32 0-d cache start ``cs`` and
        per-row int32 ``pad`` / ``ends`` ``(B,)``. Row ``b`` sits at RoPE
        positions ``cs - pad[b] + j`` and sees cache columns ``[pad[b], cs +
        ends[b])`` (:func:`~.backbone_mlx.batch_visibility_mask`).
        """
        key = ("batch", n_chunks, rows_per_chunk, attn_len, bool(compile_fn))
        fn = self._block_fns.get(key)
        if fn is not None:
            return fn
        R = int(rows_per_chunk)

        def fn(tokens, shift_ctx, cache_start, pad, ends, keys, values):
            bl = tokens.shape[1]
            pos = self.speech_pos_emb.emb.weight[:bl]
            emb = (self.speech_emb(tokens) + pos).astype(self.dtype)
            if R > 1:
                emb = mx.repeat(emb, R, axis=0)
            mask = batch_visibility_mask(cache_start, pad, ends, attn_len)
            hidden, keys, values = self.tfmr.block_forward(
                emb,
                cache_start,
                keys,
                values,
                attn_len,
                mask=mask,
                rope_offset=cache_start - pad,
            )
            shifted = mx.concatenate([shift_ctx, hidden[:, : bl - 1]], axis=1)
            logits = self.speech_head(shifted).astype(mx.float32)
            return logits, hidden[:, bl - 1 : bl], keys, values

        if compile_fn:
            fn = mx.compile(fn)
        self._block_fns[key] = fn
        return fn

    @staticmethod
    def _run_on_cache(cache: FlashKVCache, fn, *args):
        """``fn(*args, keys, values)`` with the cache's references dropped first.

        ``fn`` returns ``(*outputs, new_keys, new_values)``; the new arrays
        replace the cache's, and ``outputs`` are returned. With no other
        reference left MLX donates the old buffers (in-place update).
        """
        keys, values = cache.keys, cache.values
        cache.keys = cache.values = None
        try:
            *outs, new_keys, new_values = fn(*args, keys, values)
        except Exception:
            cache.keys, cache.values = keys, values
            raise
        del keys, values
        cache.keys, cache.values = new_keys, new_values
        return outs

    def prime_uncond_block_prior(self) -> mx.array:
        """PMI denominator: mean softmax of a ``[SOS, MASK x BS]`` forward.

        No conditioning at all; positions ``0..BS``; causal attention (the
        reference calls the HF model without a mask). Logits for mask ``i``
        come from hidden ``i`` (SOS -> mask 0).
        """
        BS = self.block_size
        toks = mx.array(
            [self.hp.start_speech_token] + [self.mask_token_id] * BS, dtype=mx.int32
        )
        emb = self.embed_speech_block(toks)[None].astype(self.dtype)
        hidden = self.tfmr(emb, None, offset=0, mask="causal")
        logits = self.speech_head(hidden[:, :BS]).astype(mx.float32)
        prior = mx.softmax(logits, axis=-1).mean(axis=1)[0]
        mx.eval(prior)
        self._uncond_prior = prior
        return prior

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        t3_cond: Union[T3CondMLX, mx.array],
        text_tokens: mx.array,
        total_speech_len: int,
        *,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        seed: Optional[int] = None,
        fuse_commit: bool = False,
        pipeline: bool = True,
        compiled: bool = True,
        stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """Block-diffusion generation for one utterance.

        Args:
            t3_cond: Conditioning (``T3CondMLX``) or precomputed conditioning
                embeddings ``(1, len_cond, D)``.
            text_tokens: ``(1, T)`` or ``(T,)`` text ids with start/stop tokens.
            total_speech_len: Speech-token budget ``N``.
            fuse_commit: Merge each commit forward with the next block's step-0
                forward (one ``2*bl``-token forward).
            pipeline: Enqueue the forward for the new tokens before the per-step
                host read (ignored when ``fuse_commit``).
            compiled: Use the fixed-shape compiled block forward (see the module
                docstring); ``False`` runs the eager, variable-length forward.
                Both give the same tokens up to float rounding.
            stats: Optional dict that receives ``n_forwards``, ``n_blocks`` and
                ``steps_per_block``.

        Returns:
            int64 numpy array of speech ids, truncated before the first EOS
            (same as the reference; may still contain ``[MASK]`` / special ids,
            see :func:`trim_speech_tokens`).
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if self._uncond_prior is None:
            self.prime_uncond_block_prior()
        N = int(total_speech_len)
        if N <= 0:
            return np.zeros((0,), dtype=np.int64)
        BS = self.block_size
        K = int(num_steps)
        MASK = self.mask_token_id
        stop = self.hp.stop_speech_token
        num_blocks = math.ceil(N / BS)
        use_cfg = cfg_scale > 0
        pipeline = pipeline and not fuse_commit

        cond_emb = (
            t3_cond
            if isinstance(t3_cond, mx.array)
            else self.prepare_conditioning(t3_cond)
        )
        prefix = self.build_prefix(cond_emb, text_tokens, use_cfg)
        B, lp = prefix.shape[0], prefix.shape[1]
        cache = self.get_cache(B, lp + N + 64)

        step = self._step_fns.get(
            use_cfg=use_cfg,
            sample=temperature > 0,
            position_noise=position_temperature > 0.0,
            use_quantile=time_shift_tau > 0.0,
            mask_id=MASK,
            stop_id=stop,
        )
        prior = self._uncond_prior
        cfg_w = mx.array(float(cfg_scale), dtype=mx.float32)
        inv_temp = mx.array(1.0 / temperature if temperature > 0 else 1.0, mx.float32)
        q_arr = mx.array(time_shift_quantiles(K, time_shift_tau), dtype=mx.float32)
        last_flags = [mx.array(k == K - 1) for k in range(K)]
        key = mx.random.key(secrets.randbits(32) if seed is None else int(seed))

        def embed(blk):
            e = self.embed_speech_block(blk).astype(self.dtype)[None]
            return mx.broadcast_to(e, (B,) + e.shape[1:]) if B > 1 else e

        # Pipelined compiled forwards alternate between two caches (in-place
        # K/V writes while the previous forward is still in flight).
        caches = [cache, self.get_twin_cache()] if compiled and pipeline else [cache]
        turn = [0, 0]  # [cache for the next forward, cache of the last forward]

        def forward(blk, cs, ctx):
            """``(logits, next_ctx)`` for block tokens ``blk`` at ``cs``."""
            c = caches[turn[0]]
            turn[1] = turn[0]
            turn[0] = (turn[0] + 1) % len(caches)
            # A short tail block runs eagerly: one trace per tail length would
            # cost more than it saves on a few forwards.
            if compiled and blk.shape[0] == BS:
                return self.run_block_fn(c, blk, ctx, cs)
            h = self.forward_block(embed(blk), c, cs)
            bl = blk.shape[0]
            return self.shifted_logits(ctx, h), h[:, bl - 1 : bl]

        def forward_fused(blk, next_blk, cs):
            """Commit ``blk`` and run the next block's step 0 in one forward."""
            bl = blk.shape[0]
            if compiled and next_blk.shape[0] == BS:
                return self.run_block_fn(
                    cache, mx.concatenate([blk, next_blk]), shift_ctx, cs, n_first=bl
                )
            emb = mx.concatenate([embed(blk), embed(next_blk)], axis=1)
            mask = fused_commit_mask(cs, bl, next_blk.shape[0])
            fused = self.forward_block(emb, cache, cs, mask=mask)
            ctx = fused[:, bl - 1 : bl]
            return self.shifted_logits(ctx, fused[:, bl:]), ctx

        hidden = self.forward_prefix(prefix, cache)
        shift_ctx = hidden[:, lp - 1 : lp]
        del hidden
        if len(caches) > 1:
            self.copy_cache_rows(cache, caches[1], 0, lp, compile_fn=False)
        if compiled:
            # Finish the prefix first: a block forward queued behind it would
            # find the cache still referenced and copy it instead of updating.
            mx.eval(shift_ctx, [c.keys for c in caches], [c.values for c in caches])
        n_forwards = 1
        steps_per_block = []
        blocks = []
        out = None  # (logits, next_ctx) of the current block's tokens, if enqueued
        mask_blk = None

        for b in range(num_blocks):
            bs_ = b * BS
            bl = min(BS, N - bs_)
            cs = lp + bs_
            last_block = b == num_blocks - 1
            sched = mx.array(
                omnivoice_unmask_schedule(bl, K, omnivoice_schedule_t_shift),
                dtype=mx.int32,
            )
            if mask_blk is None or mask_blk.shape[0] != bl:
                mask_blk = mx.full((bl,), MASK, dtype=mx.int32)
            blk = mask_blk
            if out is None:
                out = forward(blk, cs, shift_ctx)
                n_forwards += 1

            eos = False
            for k in range(K):
                key, sub = mx.random.split(key)
                blk, flags = step(
                    out[0],
                    blk,
                    prior,
                    sub,
                    sched[k],
                    q_arr[k],
                    last_flags[k],
                    cfg_w,
                    inv_temp,
                )
                out = None
                if pipeline and (k < K - 1 or not last_block):
                    # Next step's forward, or the commit if this block is done.
                    mx.async_eval(blk, flags)
                    out = forward(blk, cs, shift_ctx)
                    n_forwards += 1
                    mx.async_eval(out)
                n_mask, has_eos = flags.tolist()
                eos = bool(has_eos)
                if n_mask == 0 or eos or k == K - 1:
                    break
                if out is None:
                    out = forward(blk, cs, shift_ctx)
                    n_forwards += 1
            steps_per_block.append(k + 1)
            blocks.append(blk)
            if eos or last_block:
                break

            # Commit the clean block -> next shift_ctx.
            if fuse_commit:
                bl_next = min(BS, N - bs_ - BS)
                next_blk = mx.full((bl_next,), MASK, dtype=mx.int32)
                logits, shift_ctx = forward_fused(blk, next_blk, cs)
                n_forwards += 1
                out = (logits, None)
                mask_blk = next_blk
            else:
                if out is None:
                    out = forward(blk, cs, shift_ctx)
                    n_forwards += 1
                shift_ctx = out[1]
                out = None
                if len(caches) > 1:
                    # The next forward uses the other cache: give it this block.
                    src = caches[turn[1]]
                    self.copy_cache_rows(src, caches[1 - turn[1]], cs, bl)
            if compiled:
                # Finish the commit (and the copy, which reads the commit's
                # cache) so the next writes to both caches stay in place.
                mx.eval(shift_ctx, [c.keys for c in caches], [c.values for c in caches])

        tokens = np.array(mx.concatenate(blocks)).astype(np.int64)
        eos_pos = np.nonzero(tokens == stop)[0]
        if len(eos_pos) > 0:
            tokens = tokens[: eos_pos[0]]
        if stats is not None:
            stats["n_forwards"] = n_forwards
            stats["n_blocks"] = len(steps_per_block)
            stats["steps_per_block"] = steps_per_block
        return tokens

    def generate_batch(
        self,
        t3_cond: Union[T3CondMLX, mx.array],
        text_tokens: Sequence[mx.array],
        total_speech_lens: Sequence[int],
        *,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        seeds: Optional[Sequence[Optional[int]]] = None,
        n_slots: Optional[int] = None,
        pipeline: bool = True,
        compile_fn: bool = True,
        stats: Optional[Dict] = None,
    ) -> List[np.ndarray]:
        """Decode several utterances (chunks) concurrently as rows of one batch.

        Chunk ``c`` owns rows ``[c * R, (c + 1) * R)`` (``R = 2`` with CFG:
        conditional, null). Prefixes are left-padded to the longest one, so
        every row writes its K/V at the same cache position; a per-row RoPE
        offset keeps each row at its own true positions and a per-row mask
        hides its padding. Each chunk keeps the single-chunk algorithm of
        :meth:`generate`: its own budget, schedule, PRNG key, early exit and
        EOS stop.

        Blocks run in lockstep rounds: in round ``b`` every live chunk decodes
        its block ``b``. A chunk whose block finishes early idles (its tokens
        are frozen, its rows are still computed) until the round ends; the
        round's last forward then commits every chunk at once. A chunk that
        hits EOS or its budget becomes dead rows; the batch is not compacted.
        A chunk's last block shorter than ``block_size`` is padded with
        non-mask tokens that its real positions cannot see.

        Under deterministic settings (``temperature=0``,
        ``position_temperature=0``) each chunk gives the tokens it gives alone
        with :meth:`generate` (up to float rounding of the larger batch). The
        PRNG keys are split exactly as in :meth:`generate`, so seeded sampling
        matches too, except on a short last block (the noise is drawn at
        ``block_size`` shape instead of the block's length) and wherever
        rounding moves a sampled token.

        Args:
            t3_cond: Conditioning shared by all chunks (``T3CondMLX`` or
                ``(1, len_cond, D)`` embeddings).
            text_tokens: Per chunk ``(1, T)`` or ``(T,)`` text ids.
            total_speech_lens: Per-chunk speech-token budgets.
            seeds: Per-chunk seeds (``None`` entries draw a random seed).
            n_slots: Pad the batch to this many chunks with dead slots, so that
                calls of different sizes share one cache and one trace.
            pipeline: As in :meth:`generate`.
            compile_fn: ``False`` runs the block forward and step uncompiled.
            stats: Optional dict that receives ``n_forwards``, ``n_rounds``,
                ``n_blocks`` and ``steps_per_block`` (per-chunk lists).

        Returns:
            Per-chunk int64 numpy arrays of speech ids, truncated before the
            first EOS (as :meth:`generate`).
        """
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if self._uncond_prior is None:
            self.prime_uncond_block_prior()
        n_real = len(text_tokens)
        if len(total_speech_lens) != n_real:
            raise ValueError("text_tokens and total_speech_lens differ in length")
        if seeds is None:
            seeds = [None] * n_real
        n = max(n_real, int(n_slots or 0), 1)
        Ns = [max(0, int(x)) for x in total_speech_lens] + [0] * (n - n_real)
        BS = self.block_size
        K = int(num_steps)
        MASK = self.mask_token_id
        stop = self.hp.stop_speech_token
        FILL = 0  # pad token of a short last block: neither MASK nor EOS
        use_cfg = cfg_scale > 0
        R = 2 if use_cfg else 1
        n_blocks = [math.ceil(N / BS) for N in Ns]
        if max(n_blocks) == 0:
            return [np.zeros((0,), dtype=np.int64) for _ in range(n_real)]

        cond_emb = (
            t3_cond
            if isinstance(t3_cond, mx.array)
            else self.prepare_conditioning(t3_cond)
        )
        texts = list(text_tokens) + [text_tokens[0]] * (n - n_real)
        prefixes = [self.build_prefix(cond_emb, t, use_cfg) for t in texts]
        lp = max(p.shape[1] for p in prefixes)
        pads = []
        for i, p in enumerate(prefixes):
            pad = lp - p.shape[1]
            if pad:
                zeros = mx.zeros((p.shape[0], pad, p.shape[2]), dtype=p.dtype)
                prefixes[i] = mx.concatenate([zeros, p], axis=1)
            pads += [pad] * R
        prefix = mx.concatenate(prefixes, axis=0)
        pad_arr = mx.array(pads, dtype=mx.int32)
        B = n * R
        caches = self.get_batch_caches(B, lp + max(n_blocks) * BS + 64)
        if not pipeline:
            caches = caches[:1]

        step = self._step_fns.get_batch(
            n,
            R,
            use_cfg=use_cfg,
            sample=temperature > 0,
            position_noise=position_temperature > 0.0,
            use_quantile=time_shift_tau > 0.0,
            mask_id=MASK,
            stop_id=stop,
        )
        if not compile_fn:
            step = make_batch_step_fn(
                n,
                R,
                use_cfg=use_cfg,
                sample=temperature > 0,
                position_noise=position_temperature > 0.0,
                use_quantile=time_shift_tau > 0.0,
                mask_id=MASK,
                stop_id=stop,
            )
        prior = self._uncond_prior
        cfg_w = mx.array(float(cfg_scale), dtype=mx.float32)
        inv_temp = mx.array(1.0 / temperature if temperature > 0 else 1.0, mx.float32)
        q_arr = mx.array(time_shift_quantiles(K, time_shift_tau), dtype=mx.float32)
        last_flags = [mx.array(k == K - 1) for k in range(K)]
        keys = [
            mx.random.key(secrets.randbits(32) if sd is None else int(sd))
            for sd in list(seeds) + [0] * (n - n_real)
        ]

        turn = [0, 0]  # [cache for the next forward, cache of the last forward]

        def forward(blk, cs, ctx, ends):
            c = caches[turn[0]]
            turn[1] = turn[0]
            turn[0] = (turn[0] + 1) % len(caches)
            attn_len = self.attn_len_for(cs, BS, c.max_seq)
            fn = self.batch_block_fn(n, R, attn_len, compile_fn)
            cs_arr = mx.array(cs, dtype=mx.int32)
            return self._run_on_cache(c, fn, blk, ctx, cs_arr, pad_arr, ends)

        # Left-padded causal prefix forward for every row at once.
        main = caches[0]
        (hidden,) = self._run_on_cache(
            main,
            lambda emb, ks, vs: self.tfmr.block_forward(
                emb,
                mx.array(0, dtype=mx.int32),
                ks,
                vs,
                lp,
                mask=padded_prefix_mask(pad_arr, lp),
                rope_offset=-pad_arr,
            ),
            prefix,
        )
        shift_ctx = hidden[:, lp - 1 : lp]
        del hidden
        if len(caches) > 1:
            self.copy_cache_rows(main, caches[1], 0, lp, compile_fn=False)
        mx.eval(shift_ctx, [c.keys for c in caches], [c.values for c in caches])

        n_forwards = 1
        n_rounds = 0
        steps_per_block: List[List[int]] = [[] for _ in range(n)]
        blocks: List[List[mx.array]] = [[] for _ in range(n)]
        alive = [nb > 0 for nb in n_blocks]
        out = None

        for b in range(max(n_blocks)):
            in_round = [c for c in range(n) if alive[c]]
            if not in_round:
                break
            n_rounds += 1
            bs_ = b * BS
            cs = lp + bs_
            bls = [min(BS, Ns[c] - bs_) if alive[c] else 0 for c in range(n)]
            blk0 = np.full((n, BS), FILL, dtype=np.int32)
            sched = np.zeros((n, K), dtype=np.int32)
            for c in in_round:
                blk0[c, : bls[c]] = MASK
                sched[c] = omnivoice_unmask_schedule(
                    bls[c], K, omnivoice_schedule_t_shift
                )
            blk = mx.array(blk0)
            sched = mx.array(sched)
            ends = mx.array(
                [bls[c] if alive[c] else BS for c in range(n) for _ in range(R)],
                dtype=mx.int32,
            )
            # Some chunk continues after this round: its commit is needed.
            needs_commit = any(b < n_blocks[c] - 1 for c in in_round)
            active = [c in in_round for c in range(n)]
            eos = [False] * n
            if out is None:
                out = forward(blk, cs, shift_ctx, ends)
                n_forwards += 1

            for k in range(K):
                subs = []
                for c in range(n):
                    if active[c]:
                        keys[c], sub = mx.random.split(keys[c])
                        subs.append(sub)
                    else:
                        subs.append(keys[c])
                blk, flags = step(
                    out[0],
                    blk,
                    prior,
                    mx.stack(subs),
                    sched[:, k],
                    q_arr[k],
                    last_flags[k],
                    cfg_w,
                    inv_temp,
                    mx.array(active),
                )
                out = None
                if pipeline and (k < K - 1 or needs_commit):
                    # Next step's forward, or the commit if the round is done.
                    mx.async_eval(blk, flags)
                    out = forward(blk, cs, shift_ctx, ends)
                    n_forwards += 1
                    mx.async_eval(out)
                for c, (n_mask, has_eos) in enumerate(flags.tolist()):
                    if not active[c]:
                        continue
                    eos[c] = bool(has_eos)
                    if n_mask == 0 or eos[c] or k == K - 1:
                        active[c] = False
                        steps_per_block[c].append(k + 1)
                if not any(active):
                    break
                if out is None:
                    out = forward(blk, cs, shift_ctx, ends)
                    n_forwards += 1

            for c in in_round:
                blocks[c].append(blk[c, : bls[c]])
                if eos[c] or b == n_blocks[c] - 1:
                    alive[c] = False
            if not any(alive):
                break

            # Commit every chunk's clean block -> next shift_ctx.
            if out is None:
                out = forward(blk, cs, shift_ctx, ends)
                n_forwards += 1
            shift_ctx = out[1]
            out = None
            if len(caches) > 1:
                src = caches[turn[1]]
                self.copy_cache_rows(src, caches[1 - turn[1]], cs, BS)
            mx.eval(shift_ctx, [c.keys for c in caches], [c.values for c in caches])

        results = []
        for c in range(n_real):
            if not blocks[c]:
                results.append(np.zeros((0,), dtype=np.int64))
                continue
            tokens = np.array(mx.concatenate(blocks[c])).astype(np.int64)
            eos_pos = np.nonzero(tokens == stop)[0]
            if len(eos_pos) > 0:
                tokens = tokens[: eos_pos[0]]
            results.append(tokens)
        if stats is not None:
            stats["n_forwards"] = n_forwards
            stats["n_rounds"] = n_rounds
            stats["n_blocks"] = [len(s) for s in steps_per_block[:n_real]]
            stats["steps_per_block"] = steps_per_block[:n_real]
        return results


def trim_speech_tokens(tokens: np.ndarray, hp: T3Config) -> np.ndarray:
    """Cut at the first EOS and drop ids S3Gen can't vocode (``>= SOS``).

    Mirrors ``ChatterboxFlashTTS._trim_to_eos``.
    """
    tokens = np.asarray(tokens)
    eos = np.nonzero(tokens == hp.stop_speech_token)[0]
    if len(eos) > 0:
        tokens = tokens[: eos[0]]
    return tokens[tokens < hp.start_speech_token].astype(np.int64)
