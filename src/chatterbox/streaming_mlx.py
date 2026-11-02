# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Token-level streaming helpers shared by the MLX pipelines.

- :class:`_WindowVocoder` vocodes one chunk on growing token windows while T3
  still decodes it.
- :class:`_ChunkJoiner` is an incremental
  :func:`~chatterbox.generation_utils.crossfade_chunks`.
- :class:`MLXVocoderHost` wraps an MLX ``S3Token2WavMLX`` and its reference
  dict in the interface that :class:`_WindowVocoder` reads.

``ChatterboxTurboMLX.generate_stream`` (:mod:`chatterbox.tts_turbo_mlx`) and
``ChatterboxMultilingualTTSMLX.generate_stream`` (:mod:`chatterbox.mtl_tts_mlx`)
use them. :mod:`chatterbox.tts_turbo_mlx` re-exports every name it used to
define.
"""

from typing import Iterator, List, Optional

import numpy as np

try:
    import mlx.core as mx
except ImportError as e:  # pragma: no cover - Apple Silicon only
    raise ImportError(
        "MLX is not installed (it requires Apple Silicon). Install it with:\n"
        "  pip install 'mlx>=0.32.2'"
    ) from e

import torch

# upstream models/s3gen/const.py (not in this fork's const.py)
S3GEN_SIL = 4299
# Speech ids S3Gen can vocode are < this (SOS = 6561, EOS = 6562).
SPEECH_VOCAB = 6561

# --- Streaming (generate_stream) ------------------------------------------ #
# One S3 token is 2 mel frames of 480 samples at 24 kHz (40 ms).
SAMPLES_PER_TOKEN = 960
# S3Gen's pre-lookahead: a non-final flow call drops the last 3 tokens' mel.
S3GEN_PRE_LOOKAHEAD = 3
STREAM_FIRST_CHUNK_TOKENS = 25  # first window: 1 s of audio
STREAM_CHUNK_TOKENS = 50  # second window; later windows double ...
STREAM_MAX_CHUNK_TOKENS = 200  # ... up to this
STREAM_CONTEXT_TOKENS = 50  # tokens re-vocoded left of a window's new audio
STREAM_LOOKAHEAD_TOKENS = 5  # tokens decoded right of a window's emitted end
# Raised-cosine crossfade between consecutive windows of one chunk.
WINDOW_XFADE = SAMPLES_PER_TOKEN
# T3 ids are read in blocks of this many tokens.
STREAM_BLOCK_TOKENS = 4


def ref_dict_to_mlx(ref_dict: dict) -> dict:
    """Torch S3Gen reference dict -> the MLX vocoder's layout."""
    prompt_feat = mx.array(ref_dict["prompt_feat"].detach().float().cpu().numpy())
    prompt_token_len = ref_dict["prompt_token_len"]
    if torch.is_tensor(prompt_token_len):
        prompt_token_len = int(prompt_token_len.reshape(-1)[0].item())
    return {
        "prompt_token": mx.array(ref_dict["prompt_token"].detach().cpu().numpy()),
        "prompt_token_len": int(prompt_token_len),
        "prompt_feat": prompt_feat,
        "prompt_feat_len": prompt_feat.shape[1],
        "embedding": mx.array(ref_dict["embedding"].detach().float().cpu().numpy()),
    }


class _ChunkJoiner:
    """Incremental :func:`~chatterbox.generation_utils.crossfade_chunks`.

    With more than one chunk, every chunk loses its first 20 ms (S3Gen's
    silent fade-in), consecutive chunks overlap by ``overlap_duration`` with
    linear fades, and the last 3 ms fade to 10 %. The last ``overlap``
    samples of the output are held back until the next chunk (or
    :meth:`finish`) decides how they end. One chunk passes through as is.
    """

    def __init__(self, n_chunks: int, sr: int, overlap_duration: float):
        multi = n_chunks > 1
        self.trim = sr // 50 if multi else 0
        self.ov = int(overlap_duration * sr) if multi else 0
        self.tail = sr // 333 if multi else 0
        self.fade_out = np.linspace(1, 0, self.ov, dtype=np.float32)
        self.fade_in = np.linspace(0, 1, self.ov, dtype=np.float32)
        self._empty = np.zeros((0,), dtype=np.float32)
        self.held = self._empty  # output end, kept for the next crossfade
        self.head = self._empty  # current chunk's start, waiting to crossfade
        self.ready = self._empty  # output released by begin_chunk
        self.to_trim = 0
        self.mixing = False

    def begin_chunk(self) -> None:
        if self.mixing:  # last chunk was too short to crossfade
            self.held = np.concatenate([self.held, self.head])
            self.head = self._empty
        if self.ov and len(self.held) > self.ov:
            self.ready = np.concatenate([self.ready, self.held[: -self.ov]])
            self.held = self.held[-self.ov :]
        self.to_trim = self.trim
        self.mixing = self.ov > 0 and len(self.held) == self.ov

    def push(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        cut = min(self.to_trim, len(x))
        x = x[cut:]
        self.to_trim -= cut
        if self.mixing:
            self.head = np.concatenate([self.head, x])
            if len(self.head) < self.ov:
                return self._take_ready()
            mixed = self.held * self.fade_out + self.head[: self.ov] * self.fade_in
            x = np.concatenate([mixed, self.head[self.ov :]])
            self.held = self.head = self._empty
            self.mixing = False
        y = np.concatenate([self._take_ready(), self.held, x])
        if self.ov:
            k = min(self.ov, len(y))
            self.held, y = y[len(y) - k :], y[: len(y) - k]
        return y

    def finish(self) -> np.ndarray:
        y = np.concatenate([self._take_ready(), self.held, self.head])
        self.held = self.head = self._empty
        self.mixing = False
        n = min(self.tail, len(y))
        if n:
            y = y.copy()
            y[-n:] *= np.linspace(1.0, 0.1, n, dtype=np.float32)
        return y

    def _take_ready(self) -> np.ndarray:
        y, self.ready = self.ready, self._empty
        return y


class _WindowVocoder:
    """Vocode one chunk on growing token windows while T3 still decodes it.

    Window ``w`` vocodes the tokens ``[start, n)`` decoded so far, where
    ``start`` is ``context_tokens`` left of the audio already emitted. S3Gen
    runs non-final (it drops the last 3 tokens' mel), and the window emits
    audio up to ``lookahead_tokens`` before ``n``. The last
    :data:`WINDOW_XFADE` samples of a window are crossfaded with the same
    samples of the next window. The next window reuses the previous window's
    HiFiGAN source signal up to that crossfade end (CosyVoice's
    ``cache_source``), and every window reads the flow noise of a frame from
    one per-chunk noise buffer, so overlapping samples match closely. The
    final window (T3 finished, the 3 silence tokens appended) runs S3Gen in
    final mode and emits the rest. A chunk that ends before its first window
    is vocoded once, exactly as :meth:`ChatterboxTurboMLX.generate` does.

    ``tts`` is the vocoder host. It has ``s3gen`` (an MLX ``S3Token2WavMLX``
    with ``hift_inference`` and ``trim_fade``), ``_ref_dict`` (the MLX
    reference dict), ``mlx_cache_limit`` (bytes), ``_vocode(tokens,
    n_cfm_timesteps, seed)`` and ``_token2mel(tokens, finalize,
    n_cfm_timesteps, noise)``. :class:`~chatterbox.tts_turbo_mlx.ChatterboxTurboMLX`
    is one; :class:`MLXVocoderHost` is a stand-alone one.
    """

    def __init__(
        self,
        tts,
        *,
        seed: Optional[int],
        n_cfm_timesteps: int,
        max_tokens: int,
        first_chunk_tokens: int,
        chunk_tokens: int,
        context_tokens: int,
        lookahead_tokens: int,
    ):
        self.tts = tts
        self.seed = seed
        self.n_cfm = n_cfm_timesteps
        self.max_tokens = max_tokens
        self.size = first_chunk_tokens
        self.chunk_tokens = chunk_tokens
        self.context = context_tokens
        self.lookahead = lookahead_tokens
        self.tokens: List[int] = []
        self.emitted = 0  # chunk samples already emitted
        self.tail: Optional[np.ndarray] = None  # last window's [emitted, +X)
        self.src = None  # last window's HiFiGAN source (1, 1, N)
        self.src_start = 0  # its first chunk sample
        self.n_windows = 0
        self._noise = None
        x = WINDOW_XFADE
        t = np.arange(x, dtype=np.float32) / x
        self.fade_in = (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)
        self.fade_out = (1.0 - self.fade_in).astype(np.float32)

    def feed(self, ids: np.ndarray) -> Iterator[np.ndarray]:
        """Add T3 ids; yield the audio of every window they complete."""
        ids = np.asarray(ids).reshape(-1)
        self.tokens.extend(int(t) for t in ids[ids < SPEECH_VOCAB])
        while len(self.tokens) >= self._target():
            before = self._progress()
            yield self._window(final=False)
            if self._progress() <= before:  # no progress (short vocoder output)
                break
            self.size = (
                self.chunk_tokens
                if self.n_windows == 1
                else min(2 * self.size, max(self.chunk_tokens, STREAM_MAX_CHUNK_TOKENS))
            )

    def finish(self) -> np.ndarray:
        """Append the silence tokens and return the chunk's remaining audio."""
        self.tokens.extend([S3GEN_SIL] * 3)
        return self._window(final=True)

    def _progress(self) -> int:
        return self.emitted + (0 if self.tail is None else len(self.tail))

    def _target(self) -> int:
        return self._progress() // SAMPLES_PER_TOKEN + self.size + self.lookahead

    def _seed_window(self) -> None:
        if self.seed is not None:
            sd = int(self.seed) + 7919 * self.n_windows
            mx.random.seed(sd % 2**32)

    def _noise_slice(self, start_frame: int, n_frames: int):
        ref = self.tts._ref_dict
        if 2 * int(ref["prompt_token_len"]) != int(ref["prompt_feat_len"]):
            return None
        if self._noise is None:
            shape = (1, int(ref["prompt_feat"].shape[-1]), 2 * (self.max_tokens + 8))
            key = None
            if self.seed is not None:
                key = mx.random.key((int(self.seed) + 0x5EED) % 2**32)
            self._noise = mx.random.normal(shape, key=key)
        if start_frame + n_frames > self._noise.shape[2]:
            return None
        return self._noise[:, :, start_frame : start_frame + n_frames]

    def _window(self, final: bool) -> np.ndarray:
        tts = self.tts
        T, X = SAMPLES_PER_TOKEN, WINDOW_XFADE
        tokens = np.asarray(self.tokens, dtype=np.int64)
        n = len(tokens)
        if final and self.n_windows == 0:
            self.n_windows = 1
            return tts._vocode(tokens, self.n_cfm, self.seed)

        start = 0 if self.emitted == 0 else max(0, self.emitted // T - self.context)
        win = tokens[start:]
        n_frames = 2 * len(win) - (0 if final else 2 * S3GEN_PRE_LOOKAHEAD)
        self._seed_window()
        noise = self._noise_slice(2 * start, n_frames)
        mel = tts._token2mel(win, final, self.n_cfm, noise)

        cache = None
        if self.src is not None:
            lo = start * T - self.src_start
            hi = self.emitted + X - self.src_start
            hi = min(hi, self.src.shape[2], lo + mel.shape[-1] * (T // 2))
            if hi > lo >= 0:
                cache = self.src[:, :, lo:hi]
        wav, src = tts.s3gen.hift_inference(mel, cache)
        wav = np.array(wav.astype(mx.float32)).reshape(-1)
        trim_fade = getattr(tts.s3gen, "trim_fade", None)
        if start == 0 and trim_fade is not None:
            fade = np.array(trim_fade, dtype=np.float32).reshape(-1)
            if len(wav) >= len(fade):
                wav[: len(fade)] *= fade
        if mx.get_cache_memory() > tts.mlx_cache_limit:
            mx.clear_cache()
        self.n_windows += 1

        w0 = start * T
        end = w0 + len(wav)
        cut = end if final else min(end, (n - self.lookahead) * T)
        seg = wav[max(0, self.emitted - w0) : max(0, cut - w0)].copy()
        if self.tail is not None:
            k = min(len(self.tail), len(seg))
            seg[:k] = self.tail[:k] * self.fade_out[:k] + seg[:k] * self.fade_in[:k]
            if k < len(self.tail):  # window shorter than the crossfade
                seg = np.concatenate([seg, self.tail[k:]])
        self.tail = None
        if final or len(seg) <= X:
            out = seg
            self.emitted += len(seg)
        else:
            out, self.tail = seg[:-X], seg[-X:].copy()
            self.emitted += len(out)
        self.src, self.src_start = src, w0
        return out


class MLXVocoderHost:
    """An MLX S3Gen and its reference dict, in the interface :class:`_WindowVocoder` reads.

    ``_vocode`` and ``_token2mel`` do what the methods of the same name on
    :class:`~chatterbox.tts_turbo_mlx.ChatterboxTurboMLX` do. ``_vocode``
    takes an extra ``finalize`` (default ``True``, as Turbo).
    """

    def __init__(self, s3gen, ref_dict: dict, mlx_cache_limit: int):
        self.s3gen = s3gen
        self._ref_dict = ref_dict
        self.mlx_cache_limit = int(mlx_cache_limit)

    def _vocode(
        self,
        speech_tokens: np.ndarray,
        n_cfm_timesteps: Optional[int],
        seed: Optional[int],
        finalize: bool = True,
    ) -> np.ndarray:
        if seed is not None:
            mx.random.seed(int(seed))
        wav, _ = self.s3gen.inference(
            mx.array(np.asarray(speech_tokens)[None].astype(np.int32)),
            self._ref_dict,
            finalize=finalize,
            n_cfm_timesteps=n_cfm_timesteps,
        )
        wav = np.array(wav.astype(mx.float32)).reshape(-1)
        if mx.get_cache_memory() > self.mlx_cache_limit:
            mx.clear_cache()
        return wav

    def _token2mel(self, tokens: np.ndarray, finalize: bool, n_cfm_timesteps, noise):
        """S3Gen flow only (tokens -> mel ``(1, 80, F)``), optionally with given noise."""
        from .models.s3gen_mlx.s3gen_mlx import S3Token2MelMLX

        toks = mx.array(np.asarray(tokens)[None].astype(np.int32))
        if noise is not None and isinstance(self.s3gen, S3Token2MelMLX):
            mel = S3Token2MelMLX.__call__(
                self.s3gen,
                toks,
                self._ref_dict,
                finalize,
                n_cfm_timesteps=n_cfm_timesteps,
                noised_mels=noise,
            )
            mx.eval(mel)
            return mel
        return self.s3gen.flow_inference(
            toks, self._ref_dict, finalize, n_cfm_timesteps=n_cfm_timesteps
        )
