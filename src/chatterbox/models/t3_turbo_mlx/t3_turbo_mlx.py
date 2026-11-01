# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX port of the Chatterbox-Turbo / Nano T3 (GPT-2 backbone, autoregressive).

Reference: upstream ``T3`` with a GPT-2 config (``is_gpt``) and
``T3.inference_turbo`` (resemble-ai/chatterbox master). Turbo uses GPT-2
medium (1024 / 24 layers / 16 heads), Nano GPT-2 small (768 / 12 / 12).
Compared with the Llama T3:

* no text / speech position embeddings; the backbone adds its learned ``wpe``
  at each token's absolute position (conditioning included);
* the conditioning is the speaker projection plus the 375 prompt speech
  tokens through ``speech_emb`` (no perceiver, no emotion, no CFG);
* ``speech_head`` has a bias;
* the text is fed as tokenized (upstream adds no start / stop text token).

Generation (:meth:`T3TurboMLX.generate_batch`, ``generate`` is one row):

1. The conditioning block's K/V are computed once per voice
   (:meth:`T3TurboMLX.cond_kv`) and copied into every row; each row's
   ``[text | SOS]`` then runs once (prefill) and its last hidden state gives
   the first token. Upstream's first repetition-penalty ``input_ids`` is the
   SOS token; later steps penalise the tokens generated so far (not SOS). The
   first token never stops generation, as upstream.
2. Each later token is one compiled step: embed the token, add ``wpe``, run
   the stack over one of the two cache buffers (:mod:`.gpt2_mlx`), apply the
   speech head and the HF processor chain, sample (:mod:`.sampling_mlx`).
   Everything stays on the device; the host only enqueues steps.
3. Steps are pipelined two deep with ``mx.async_eval``: before enqueuing step
   ``k`` the host reads step ``k-2``'s "all rows finished" flag, which is
   normally computed already, so the GPU never waits for the host. A
   finished row keeps emitting the stop token; tokens are trimmed at the
   first stop afterwards, and at most two steps past the last EOS are wasted.

Batched decoding: an AR step at small batch reads every weight once per step
(memory-bandwidth bound), so several chunks of a long text decode as rows of
one batch for well under B times the cost of one; the per-row cost is the
attention over the row's own K/V. Texts are left-padded to a common length;
each row has its own ``wpe`` positions and attention mask.
"""

import logging
import secrets
from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..t3_mlx.modules.cond_enc_mlx import T3CondEncMLX, T3CondMLX
from .configs import GPT2ConfigMLX, TurboT3Config
from .gpt2_mlx import (
    GPT2ModelMLX,
    TurboKVCache,
    decode_mask,
    is_conv1d_weight,
    prefill_mask,
)
from .sampling_mlx import onehot_mask, sample_tokens, sampling_params

logger = logging.getLogger(__name__)

# Cache sizes grow in buckets of this many positions so that small changes in
# prompt length don't force a reallocation.
CACHE_BUCKET = 256
# Compiled decode steps attend over the cache window rounded up to this many
# positions; each window size is one trace.
ATTN_BUCKET = 256
# Batched prefills over more tokens than this evaluate layer by layer.
PREFILL_EVAL_TOKENS = 512
# Steps in flight: step k waits for step k - PIPELINE_DEPTH (see gpt2_mlx).
PIPELINE_DEPTH = 2
# upstream inference_turbo max_gen_len
MAX_GEN_LEN = 1000

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


def _bucket(n: int, size: int) -> int:
    return ((int(n) + size - 1) // size) * size


def trim_at_stop(tokens: np.ndarray, stop_id: int) -> np.ndarray:
    """Upstream's stop rule on one row ``[t0, t1, ...]``.

    Generation stops at the first stop token *after* ``t0`` (the first token is
    never checked) and that stop token is dropped.
    """
    tokens = np.asarray(tokens)
    if len(tokens) <= 1:
        return tokens.astype(np.int64)
    stops = np.nonzero(tokens[1:] == stop_id)[0]
    if len(stops):
        tokens = tokens[: 1 + stops[0]]
    return tokens.astype(np.int64)


class T3TurboMLX(nn.Module):
    """Chatterbox-Turbo / Nano T3 decoder on MLX."""

    def __init__(
        self,
        nano: bool = False,
        hp: Optional[TurboT3Config] = None,
    ):
        super().__init__()
        if hp is None:
            hp = TurboT3Config(nano=nano)
        self.hp = hp
        self.cfg = GPT2ConfigMLX.from_dict(hp.gpt2_config)
        self.dim = self.cfg.hidden_size

        self.tfmr = GPT2ModelMLX(self.cfg)
        self.cond_enc = T3CondEncMLX(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)
        self.speech_head = nn.Linear(self.dim, hp.speech_tokens_dict_size, bias=True)

        # Inference state (underscore names keep them out of parameters()).
        self._cache: Optional[TurboKVCache] = None
        # (source, cond_keys, cond_values) of the last conditioning seen.
        self._cond: Optional[tuple] = None
        self._fns: Dict[tuple, object] = {}
        self._quantize_bits: Optional[int] = None
        # Attended-window bucket of the compiled decode step (0 = whole cache).
        self.attn_bucket = ATTN_BUCKET

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def dtype(self) -> mx.Dtype:
        return self.speech_emb.weight.dtype

    @property
    def cache_nbytes(self) -> int:
        return 0 if self._cache is None else self._cache.nbytes

    def release_caches(self) -> None:
        """Free the KV caches and compiled traces; the next call rebuilds them."""
        self._cache = None
        self._cond = None
        self._fns = {}
        mx.clear_cache()

    # ------------------------------------------------------------------ #
    #  Weights
    # ------------------------------------------------------------------ #

    def load_torch_state_dict(self, state_dict: Dict, strict: bool = True) -> None:
        """Load an upstream Turbo / Nano T3 state dict (``t3_turbo_v1.safetensors``).

        Keys map 1:1 (``tfmr.h.N.attn.c_attn.weight`` etc.). HF ``Conv1D``
        weights are transposed to ``nn.Linear`` layout. Unused entries
        (``tfmr.wte``, ``text_head``, attention-bias buffers) are dropped.
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
            arr = np.asarray(arr)
            if is_conv1d_weight(k):
                arr = np.ascontiguousarray(arr.T)
            weights[k] = mx.array(arr)

        missing = sorted(set(expected) - set(weights))
        dropped = sorted(set(state_dict) - set(weights))
        if dropped:
            logger.debug("Ignoring %d unused keys: %s", len(dropped), dropped[:8])
        if missing and strict:
            raise ValueError(f"Missing {len(missing)} T3 weights: {missing[:10]}")
        for k, v in weights.items():
            if tuple(v.shape) != tuple(expected[k].shape):
                raise ValueError(
                    f"Shape mismatch for {k}: checkpoint {tuple(v.shape)}, "
                    f"model {tuple(expected[k].shape)}"
                )
        self.load_weights(list(weights.items()), strict=False)
        self._cond = None
        self._fns = {}

    def finalize(
        self,
        dtype: Union[str, mx.Dtype] = "float16",
        quantize_bits: Optional[int] = None,
        group_size: int = 64,
    ) -> "T3TurboMLX":
        """Cast to ``dtype`` and optionally quantize (weight-only, 4 or 8 bits).

        Quantization covers the GPT-2 linears and the speech head; embeddings,
        ``wpe``, layer norms and the conditioning encoder stay in ``dtype``.
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
        self._cache = None
        self._cond = None
        # Compiled traces capture the weights; rebuild them after a change.
        self._fns = {}
        return self

    # ------------------------------------------------------------------ #
    #  Embeddings
    # ------------------------------------------------------------------ #

    def prepare_conditioning(self, t3_cond: T3CondMLX) -> mx.array:
        """Conditioning embeddings ``(1, len_cond, D)``; ``t3_cond`` is not modified.

        Prompt speech tokens go through ``speech_emb`` only (no position
        embedding on the GPT-2 models).
        """
        emb = t3_cond.cond_prompt_speech_emb
        tokens = t3_cond.cond_prompt_speech_tokens
        if tokens is not None and emb is None:
            emb = self.speech_emb(tokens)
        cond = T3CondMLX(
            speaker_emb=t3_cond.speaker_emb,
            clap_emb=t3_cond.clap_emb,
            cond_prompt_speech_tokens=tokens,
            cond_prompt_speech_emb=emb,
            emotion_adv=t3_cond.emotion_adv,
        )
        return self.cond_enc(cond)

    def build_row(self, text_tokens: mx.array) -> mx.array:
        """``[text | SOS]`` ``(1, T + 1, D)`` in model dtype (``wpe`` not added)."""
        if text_tokens.ndim == 1:
            text_tokens = text_tokens[None]
        dtype = self.dtype
        text_emb = self.text_emb(text_tokens).astype(dtype)
        sos = mx.array([[self.hp.start_speech_token]])
        sos_emb = self.speech_emb(sos).astype(dtype)
        return mx.concatenate([text_emb, sos_emb], axis=1)

    def cond_kv(self, t3_cond: Union[T3CondMLX, mx.array]):
        """Shared conditioning K/V ``(cond_keys, cond_values)`` for ``t3_cond``.

        Cached by identity: passing the same ``T3CondMLX`` or embedding array
        again (the pipeline keeps one per voice) reuses the K/V.
        """
        if self._cond is not None and self._cond[0] is t3_cond:
            return self._cond[1], self._cond[2]
        cond_emb = (
            t3_cond
            if isinstance(t3_cond, mx.array)
            else self.prepare_conditioning(t3_cond)
        )
        ck, cv = self.tfmr.cond_forward(cond_emb.astype(self.dtype))
        mx.eval(ck, cv)
        self._cond = (t3_cond, ck, cv)
        return ck, cv

    # ------------------------------------------------------------------ #
    #  Cache and compiled functions
    # ------------------------------------------------------------------ #

    def get_cache(self, batch_size: int, needed: int) -> TurboKVCache:
        """Reuse the preallocated cache if it fits, else grow it (256 buckets)."""
        dtype = self.dtype
        cache = self._cache
        if cache is not None and cache.fits(batch_size, needed, dtype):
            return cache
        size = _bucket(needed, CACHE_BUCKET)
        if cache is not None and cache.batch_size == batch_size:
            size = max(size, cache.max_seq)
        self._cache = None
        del cache
        logger.debug("Allocating Turbo KV cache: batch=%d max_seq=%d", batch_size, size)
        self._cache = self.tfmr.make_cache(batch_size, size, dtype)
        return self._cache

    def attn_len_for(self, col: int, max_seq: int) -> int:
        """Static attended window for a decode step writing column ``col``."""
        bucket = int(self.attn_bucket)
        if bucket <= 0:
            return max_seq
        return min(max_seq, _bucket(col + 1, bucket))

    def decode_fn(
        self, batch_size: int, lc: int, attn_len: int, flags: dict, compile_fn=True
    ):
        """Compiled single-token decode step + sampling.

        ``(tok, col, prev_col, pad, step, caps, presence, finished, keys,
        temperature, top_p, penalty, prev_k, prev_v, buf_k, buf_v) -> (tok, presence,
        finished, keys, all_done, own_k, own_v, buf_k, buf_v)``. ``tok`` /
        ``pad`` are ``(B,)`` int32, ``col`` / ``prev_col`` ``(1,)`` int32
        cache columns of the input token and the previous one; row ``b``'s
        token sits at ``wpe`` position ``col - pad[b]``. ``all_done`` is the
        0-d "every row finished" flag the host polls two steps later. A row
        whose ``caps[b]`` decode steps are used up (``step >= caps[b]``, with
        ``step`` the ``(1,)`` int32 step index) is finished like at EOS.
        """
        key = (
            "decode",
            batch_size,
            lc,
            attn_len,
            tuple(sorted(flags.items())),
            compile_fn,
        )
        fn = self._fns.get(key)
        if fn is not None:
            return fn
        stop = self.hp.stop_speech_token

        def fn(
            tok,
            col,
            prev_col,
            pad,
            step,
            caps,
            presence,
            finished,
            keys,
            temp,
            top_p,
            penalty,
            prev_k,
            prev_v,
            buf_k,
            buf_v,
        ):
            finished = finished | (step >= caps)
            emb = self.speech_emb(tok)[:, None].astype(self.dtype)
            mask = decode_mask(pad, lc, col, attn_len)
            hidden, own_k, own_v, buf_k, buf_v = self.tfmr.decode(
                emb,
                col - pad,
                col,
                prev_col,
                prev_k,
                prev_v,
                mask,
                buf_k,
                buf_v,
                attn_len,
            )
            logits = self.speech_head(hidden[:, 0]).astype(mx.float32)
            tok, presence, finished, keys = sample_tokens(
                logits,
                presence,
                finished,
                keys,
                temp,
                top_p,
                penalty,
                stop_id=stop,
                **flags,
            )
            return (
                tok,
                presence,
                finished,
                keys,
                finished.all(),
                own_k,
                own_v,
                buf_k,
                buf_v,
            )

        if compile_fn:
            fn = mx.compile(fn)
        self._fns[key] = fn
        return fn

    # ------------------------------------------------------------------ #
    #  Generation
    # ------------------------------------------------------------------ #

    def generate(
        self,
        t3_cond: Union[T3CondMLX, mx.array],
        text_tokens: mx.array,
        *,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = MAX_GEN_LEN,
        seed: Optional[int] = None,
        compile_fn: bool = True,
        stats: Optional[Dict] = None,
    ) -> np.ndarray:
        """Decode one utterance; returns int64 speech ids without the stop token.

        Same output contract as upstream ``inference_turbo`` (which may return
        a leading stop id when it is the first token; the pipeline drops ids
        ``>= 6561`` afterwards).
        """
        return self.generate_batch(
            t3_cond,
            [text_tokens],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_gen_len=max_gen_len,
            seeds=[seed],
            compile_fn=compile_fn,
            stats=stats,
        )[0]

    def generate_batch(
        self,
        t3_cond: Union[T3CondMLX, mx.array],
        text_tokens: Sequence[mx.array],
        *,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_gen_len: int = MAX_GEN_LEN,
        seeds: Optional[Sequence[Optional[int]]] = None,
        max_gen_lens: Optional[Sequence[int]] = None,
        n_slots: Optional[int] = None,
        compile_fn: bool = True,
        stats: Optional[Dict] = None,
    ) -> List[np.ndarray]:
        """Decode several utterances (chunks) concurrently, one row each.

        Each row follows :meth:`generate`'s algorithm with its own PRNG key,
        repetition-penalty state and stop; under ``top_k=1`` (greedy) or with
        the same seed a row gives the tokens it gives alone (up to float
        rounding of the larger batch). The batch runs until every row has
        stopped (or ``max_gen_len``), so one runaway row holds the others.

        Args:
            t3_cond: Conditioning shared by all rows (``T3CondMLX`` or
                ``(1, len_cond, D)`` embeddings; see :meth:`cond_kv`).
            text_tokens: Per row ``(1, T)`` or ``(T,)`` text ids.
            seeds: Per-row seeds (``None`` entries draw a random seed).
            max_gen_lens: Per-row decode-step caps (each at most
                ``max_gen_len``): a row stops after ``1 + cap`` tokens, as if
                it had hit EOS, so a row that never emits EOS can't hold the
                batch to ``max_gen_len``. ``None`` = ``max_gen_len`` for all.
            n_slots: Pad the batch to this many rows with dead rows, so calls
                of different sizes share one cache and one trace.
            compile_fn: ``False`` runs the decode step uncompiled.
            stats: Optional dict that receives ``n_steps`` (decode steps
                run), ``prefix_len`` (cond + padded text + SOS) and
                ``batch_size``.

        Returns:
            Per-row int64 numpy arrays (see :func:`trim_at_stop`).
        """
        n_real = len(text_tokens)
        if n_real == 0:
            return []
        if seeds is None:
            seeds = [None] * n_real
        B = max(n_real, int(n_slots or 0))
        V = self.hp.speech_tokens_dict_size
        stop = self.hp.stop_speech_token
        max_gen_len = max(0, int(max_gen_len))
        if max_gen_lens is None:
            max_gen_lens = [max_gen_len] * n_real
        if len(max_gen_lens) != n_real:
            raise ValueError("text_tokens and max_gen_lens differ in length")
        caps = [min(max_gen_len, max(0, int(c))) for c in max_gen_lens]
        caps += [0] * (B - n_real)
        max_gen_len = max(caps)
        caps_arr = mx.array(caps, dtype=mx.int32)

        cond_k, cond_v = self.cond_kv(t3_cond)
        lc = int(cond_k[0].shape[2])
        texts = list(text_tokens) + [text_tokens[0]] * (B - n_real)
        rows = [self.build_row(t) for t in texts]
        lr = max(r.shape[1] for r in rows)
        pads = []
        for i, r in enumerate(rows):
            pad = lr - r.shape[1]
            if pad:
                zeros = mx.zeros((1, pad, r.shape[2]), dtype=r.dtype)
                rows[i] = mx.concatenate([zeros, r], axis=1)
            pads.append(pad)
        prefix = mx.concatenate(rows, axis=0)
        pad_arr = mx.array(pads, dtype=mx.int32)
        col0 = lc + lr  # cache column of the first decode step's input token
        cache = self.get_cache(B, col0 + max_gen_len + 1)

        # Prefill: each row's [pad | text | SOS] after the cond block, written
        # into both buffers.
        positions = mx.maximum(mx.arange(lc, col0)[None, :] - pad_arr[:, None], 0)
        buffers = [cache.take(0), cache.take(1)]
        try:
            hidden, buffers = self.tfmr.prefill(
                prefix,
                positions,
                prefill_mask(pad_arr, lc, lr),
                cond_k,
                cond_v,
                buffers,
                eval_layers=B * lr > PREFILL_EVAL_TOKENS,
            )
        finally:
            cache.buffers = buffers
        del buffers
        logits = self.speech_head(hidden[:, -1]).astype(mx.float32)
        del hidden

        params, flags = sampling_params(temperature, top_k, top_p, repetition_penalty)
        rng = mx.stack(
            [
                mx.random.key(secrets.randbits(32) if sd is None else int(sd))
                for sd in list(seeds) + [0] * (B - n_real)
            ]
        )
        # First token: upstream's input_ids is SOS; it never stops the row.
        sos = mx.full((B,), self.hp.start_speech_token, dtype=mx.int32)
        tok, _, _, rng = sample_tokens(
            logits,
            onehot_mask(sos, V),
            mx.zeros((B,), mx.bool_),
            rng,
            *params,
            stop_id=stop,
            **flags,
        )
        presence = onehot_mask(tok, V)
        finished = mx.array([i >= n_real for i in range(B)])
        tok = mx.where(finished, mx.array(stop, mx.int32), tok)
        steps = [tok]
        # Sync once: steps 0 and 1 write the buffers the prefill wrote, which
        # would otherwise still be held by the prefill in flight (a copy each).
        mx.eval(tok, presence, finished, rng, cache.buffers)

        # Step 0 has no previous token to copy: it writes a dummy at its own
        # column, which its own K/V then overwrite.
        n_layers = len(self.tfmr.h)
        _, H, _, hd = cond_k[0].shape
        zero_kv = mx.zeros((B, H, 1, hd), self.dtype)
        prev_k = prev_v = [zero_kv] * n_layers
        done_flags: List[mx.array] = []
        n_steps = 0
        fn = None
        fn_len = -1
        for k in range(max_gen_len):
            if k >= PIPELINE_DEPTH and bool(done_flags[k - PIPELINE_DEPTH].item()):
                break
            col = col0 + k
            attn_len = self.attn_len_for(col, cache.max_seq)
            if attn_len != fn_len:
                fn = self.decode_fn(B, lc, attn_len, flags, compile_fn)
                fn_len = attn_len
            col_arr = mx.array([col], mx.int32)
            prev_arr = mx.array([col - 1 if k else col], mx.int32)
            buf_k, buf_v = cache.take(k % 2)
            try:
                (tok, presence, finished, rng, done, prev_k, prev_v, buf_k, buf_v) = fn(
                    tok,
                    col_arr,
                    prev_arr,
                    pad_arr,
                    mx.array([k], mx.int32),
                    caps_arr,
                    presence,
                    finished,
                    rng,
                    *params,
                    prev_k,
                    prev_v,
                    buf_k,
                    buf_v,
                )
            finally:
                cache.buffers[k % 2] = (buf_k, buf_v)
            del buf_k, buf_v
            steps.append(tok)
            done_flags.append(done)
            n_steps += 1
            mx.async_eval(
                tok, presence, finished, rng, done, prev_k, prev_v, cache.buffers[k % 2]
            )

        out = np.array(mx.stack(steps, axis=1)).astype(np.int64)  # (B, 1 + n_steps)
        if stats is not None:
            stats["n_steps"] = n_steps
            stats["prefix_len"] = col0
            stats["batch_size"] = B
        return [trim_at_stop(out[i], stop) for i in range(n_real)]
