# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Chatterbox-Turbo / Nano TTS pipeline with the GPT-2 T3 on MLX.

- T3: :class:`~chatterbox.models.t3_turbo_mlx.T3TurboMLX` (autoregressive,
  compiled single-token steps, on-device sampling)
- S3Gen: the meanflow-distilled vocoder on MLX (``S3Token2WavMLX(meanflow=True)``
  from ``s3gen_meanflow.safetensors``), 2 CFM steps as upstream
- Conditioning (voice encoder, S3 prompt tokens, S3Gen reference dict) runs in
  PyTorch on the CPU once per voice. The checkpoint's ``conds.pt`` is the
  default voice.

Parity notes vs. upstream ``chatterbox.tts_turbo.ChatterboxTurboTTS``: same
text normalization (:func:`punc_norm`), GPT-2 tokenizer, sampling chain
(temperature, top-k 1000, top-p 0.95, repetition penalty 1.2, applied in that
order), ``max_gen_len`` 1000, conditioning (reference > 5 s, -27 LUFS, 15 s /
10 s segments), token post-processing (ids ``>= 6561`` dropped, three
``S3GEN_SIL`` tokens appended) and Perth watermark. With
``split_sentences=False`` a call is the upstream single-shot generation.

With ``split_sentences=True`` (the default) text over
:data:`~chatterbox.tts_flash_mlx.MAX_CHUNK_WORDS` words is cut into sentence
groups (:func:`~chatterbox.tts_flash_mlx.plan_chunks`); up to
``max_batch_chunks`` groups decode concurrently as rows of one T3 batch, are
vocoded one by one and crossfaded. The watermark is applied once, to the
final waveform.
"""

import logging
import math
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .generation_utils import count_words, crossfade_chunks
from .models.utils import TORCH_LOCK, materialize_mlx_state
from .tts_flash_mlx import (
    SPEECH_BUDGET_PAD,
    SPEECH_TOKENS_PER_WORD,
    _accepts,
    _locked,
    _require_meanflow,
    plan_chunks,
)

try:
    import mlx.core as mx
except ImportError as e:  # pragma: no cover - Apple Silicon only
    raise ImportError(
        "MLX is not installed (it requires Apple Silicon). Install it with:\n"
        "  pip install 'mlx>=0.32.2'"
    ) from e

import librosa
import torch
from safetensors.torch import load_file

from .models.s3gen import S3GEN_SR
from .models.s3tokenizer import S3_SR
from .models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from .models.t3_turbo_mlx import T3TurboMLX
from .models.t3_turbo_mlx.t3_turbo_mlx import MAX_GEN_LEN
from .models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"
NANO_REPO_ID = "ResembleAI/chatterbox-nano"
VE_FILENAME = "ve.safetensors"
S3GEN_FILENAME = "s3gen_meanflow.safetensors"
CONDS_FILENAME = "conds.pt"


def t3_filename(nano: bool) -> str:
    return "t3_nano_v1.safetensors" if nano else "t3_turbo_v1.safetensors"


# Files a release needs (the plain s3gen.safetensors is not used).
ALLOW_PATTERNS = [
    VE_FILENAME,
    S3GEN_FILENAME,
    CONDS_FILENAME,
    "t3_*_v1.safetensors",
    "*.json",
    "*.txt",
]

# upstream models/s3gen/const.py (not in this fork's const.py)
S3GEN_SIL = 4299
# Speech ids S3Gen can vocode are < this (SOS = 6561, EOS = 6562).
SPEECH_VOCAB = 6561
TEXT_VOCAB = 50276

DEFAULT_MAX_BATCH_CHUNKS = 8

# Per-chunk decode cap with split_sentences: 1.5x the Flash budget estimate
# (12.5 tokens per word + 25) plus 25 tokens. Over 272 chunks with 4 real
# voices, no chunk used more than 0.88x the estimate (12.2 tokens per word at
# most), so the cap only cuts a chunk that loops or never emits EOS. Without
# it such a chunk would hold its whole batch group to max_gen_len.
CHUNK_CAP_FACTOR = 1.5
CHUNK_CAP_PAD = 25


def chunk_speech_cap(text: str, max_gen_len: int = 1000) -> int:
    """Decode-step cap for one chunk, from its word count (see above)."""
    flash = math.ceil(SPEECH_TOKENS_PER_WORD * count_words(text) + SPEECH_BUDGET_PAD)
    return min(int(max_gen_len), int(CHUNK_CAP_FACTOR * flash + CHUNK_CAP_PAD))


def punc_norm(text: str) -> str:
    """Upstream ``tts_turbo.punc_norm`` (copied verbatim in behaviour)."""
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def normalize_loudness(
    wav: np.ndarray, sr: int, target_lufs: float = -27
) -> np.ndarray:
    """Upstream ``ChatterboxTurboTTS.norm_loudness``."""
    try:
        import pyloudnorm as ln

        meter = ln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        gain_db = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)
        if math.isfinite(gain_linear) and gain_linear > 0.0:
            # A numpy float64 gain would promote a float32 wav to float64,
            # which the S3 tokenizer's float32 mel filters reject.
            wav = (wav * gain_linear).astype(wav.dtype, copy=False)
    except Exception as e:
        logger.warning("Error in norm_loudness, skipping: %s", e)
    return wav


def postprocess_speech_tokens(tokens: np.ndarray) -> np.ndarray:
    """Drop ids S3Gen can't vocode (``>= 6561``) and append 3 ``S3GEN_SIL``."""
    tokens = np.asarray(tokens).reshape(-1)
    tokens = tokens[tokens < SPEECH_VOCAB].astype(np.int64)
    return np.concatenate([tokens, np.full(3, S3GEN_SIL, dtype=np.int64)])


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


class ChatterboxTurboMLX:
    """Chatterbox-Turbo (or Nano) text-to-speech with the T3 decoder on MLX."""

    ENC_COND_LEN = 15 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3TurboMLX,
        s3gen,
        ve: VoiceEncoder,
        tokenizer,
        *,
        pt_s3gen=None,
        conds: Optional[dict] = None,
        nano: bool = False,
        mlx_cache_limit_gb: float = 2.0,
    ):
        """
        Args:
            t3: Finalized :class:`T3TurboMLX`.
            s3gen: MLX ``S3Token2WavMLX(meanflow=True)``.
            ve: Voice encoder (PyTorch, CPU).
            tokenizer: HF GPT-2 tokenizer of the release (``pad = eos``).
            pt_s3gen: Torch ``S3Gen(meanflow=True)`` for conditioning only
                (``embed_ref`` and its S3 tokenizer).
            conds: Builtin voice, the ``torch.load`` of ``conds.pt``
                (``{"t3": {...}, "gen": {...}}``); it becomes the default
                conditioning.
            nano: Label only (logging).
            mlx_cache_limit_gb: Release MLX's buffer cache after vocoding once
                it holds more than this (the vocoder leaves GBs of buffers
                behind per new token length).
        """
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.pt_s3gen = pt_s3gen
        self.model_label = "Nano" if nano else "Turbo"
        self.mlx_cache_limit = int(mlx_cache_limit_gb * 1024**3)
        self.last_timings: dict = {}
        self._lock = threading.RLock()
        self._watermarker = None

        # Per-voice conditioning (set by prepare_conditionals / conds.pt)
        self._t3_cond_emb: Optional[mx.array] = None
        self._ref_dict = None

        # MLX streams are per-thread; evaluate lazily-built state here.
        materialize_mlx_state(self.t3, self.s3gen)
        if conds is not None:
            self.set_conditionals_from_dict(conds)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        nano: bool = False,
        dtype: str = "float16",
        quantize_bits: Optional[int] = None,
        repo_id: Optional[str] = None,
        mlx_cache_limit_gb: float = 2.0,
    ) -> "ChatterboxTurboMLX":
        """Download a Turbo (or Nano) release from the Hugging Face Hub and load it."""
        import os

        from huggingface_hub import snapshot_download

        local_path = snapshot_download(
            repo_id=repo_id or (NANO_REPO_ID if nano else REPO_ID),
            token=os.getenv("HF_TOKEN") or None,
            allow_patterns=ALLOW_PATTERNS,
        )
        return cls.from_local(
            local_path,
            nano=nano,
            dtype=dtype,
            quantize_bits=quantize_bits,
            mlx_cache_limit_gb=mlx_cache_limit_gb,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Union[str, Path],
        nano: bool = False,
        dtype: str = "float16",
        quantize_bits: Optional[int] = None,
        mlx_cache_limit_gb: float = 2.0,
    ) -> "ChatterboxTurboMLX":
        """Load from a directory holding a Turbo / Nano release."""
        ckpt_dir = Path(ckpt_dir)
        from transformers import AutoTokenizer

        from .models.s3gen import S3Gen
        from .models.s3gen_mlx.convert_weights import convert_and_load_weights
        from .models.s3gen_mlx.s3gen_mlx import S3Token2WavMLX

        _require_meanflow(S3Gen, "chatterbox.models.s3gen.S3Gen")
        _require_meanflow(S3Token2WavMLX, "S3Token2WavMLX")

        with TORCH_LOCK:
            logger.info("Loading voice encoder...")
            ve = VoiceEncoder()
            ve.load_state_dict(load_file(ckpt_dir / VE_FILENAME))
            ve.to("cpu").eval()

            logger.info("Loading meanflow S3Gen (PyTorch, conditioning only)...")
            pt_s3gen = S3Gen(meanflow=True)
            res = pt_s3gen.load_state_dict(
                load_file(ckpt_dir / S3GEN_FILENAME), strict=False
            )
            if res.missing_keys or res.unexpected_keys:
                logger.warning(
                    "S3Gen load: %d missing, %d unexpected keys",
                    len(res.missing_keys),
                    len(res.unexpected_keys),
                )
            pt_s3gen.to("cpu").eval()
            # Only embed_ref (mel extractor, speaker encoder, tokenizer) runs in
            # torch; its flow and HiFiGAN would be a second copy of the vocoder.
            del pt_s3gen.flow, pt_s3gen.mel2wav

        logger.info(
            "Loading %s T3 into MLX (dtype=%s, quantize_bits=%s)...",
            "Nano" if nano else "Turbo",
            dtype,
            quantize_bits,
        )
        t3 = T3TurboMLX(nano=nano)
        t3.load_torch_state_dict(load_file(ckpt_dir / t3_filename(nano)))
        t3.finalize(dtype, quantize_bits=quantize_bits)

        logger.info("Loading meanflow S3Gen MLX vocoder...")
        s3gen = S3Token2WavMLX(meanflow=True)
        weights = convert_and_load_weights(str(ckpt_dir / S3GEN_FILENAME))
        s3gen.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=False)
        s3gen.eval()

        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if len(tokenizer) != TEXT_VOCAB:
            logger.warning("Tokenizer len %d != %d", len(tokenizer), TEXT_VOCAB)

        conds = None
        builtin_voice = ckpt_dir / CONDS_FILENAME
        if builtin_voice.exists():
            conds = torch.load(builtin_voice, map_location="cpu", weights_only=True)

        # Weight conversion leaves ~2 GB in MLX's buffer cache.
        mx.clear_cache()
        return cls(
            t3,
            s3gen,
            ve,
            tokenizer,
            pt_s3gen=pt_s3gen,
            conds=conds,
            nano=nano,
            mlx_cache_limit_gb=mlx_cache_limit_gb,
        )

    # ------------------------------------------------------------------ #
    #  Conditioning
    # ------------------------------------------------------------------ #

    @_locked
    def release_caches(self) -> None:
        """Free T3's KV cache, compiled traces and MLX's buffer cache.

        Weights and conditioning stay; the next :meth:`generate` rebuilds
        the cache.
        """
        self.t3.release_caches()

    def _set_conditionals(self, speaker_emb, prompt_tokens, ref_dict) -> None:
        """Store T3 conditioning embeddings and the MLX S3Gen reference dict."""
        t3_cond = T3CondMLX(
            speaker_emb=mx.array(np.asarray(speaker_emb, dtype=np.float32)),
            cond_prompt_speech_tokens=(
                None
                if prompt_tokens is None
                else mx.array(np.asarray(prompt_tokens).astype(np.int32))
            ),
        )
        cond_emb = self.t3.prepare_conditioning(t3_cond)
        mx.eval(cond_emb)
        self._t3_cond_emb = cond_emb
        self._ref_dict = ref_dict_to_mlx(ref_dict)
        materialize_mlx_state(self._ref_dict)

    @_locked
    def set_conditionals_from_dict(self, conds: dict) -> None:
        """Use a saved voice (the ``torch.load`` of an upstream ``conds.pt``)."""
        t3 = conds["t3"]
        tokens = t3.get("cond_prompt_speech_tokens")
        if t3.get("cond_prompt_speech_emb") is not None:
            logger.warning(
                "conds.pt cond_prompt_speech_emb ignored; re-embedding tokens"
            )
        self._set_conditionals(
            t3["speaker_emb"].float().cpu().numpy(),
            None if tokens is None else torch.atleast_2d(tokens).cpu().numpy(),
            conds["gen"],
        )

    @_locked
    def prepare_conditionals(
        self,
        wav_fpath: Union[str, Path],
        exaggeration: float = 0.0,
        norm_loudness: bool = True,
    ) -> None:
        """Build speaker / prompt conditioning from a reference wav (as upstream).

        The reference must be longer than 5 s. It is normalized to -27 LUFS
        (``norm_loudness``); the first 15 s at 16 kHz give the 375 T3 prompt
        tokens, the whole 16 kHz wav the VE embedding and the first 10 s at
        24 kHz the S3Gen reference dict. ``exaggeration`` is unused by Turbo.
        """
        s3gen_ref_wav, _sr = librosa.load(str(wav_fpath), sr=S3GEN_SR)
        if len(s3gen_ref_wav) / _sr <= 5.0:
            raise ValueError("Audio prompt must be longer than 5 seconds!")
        if norm_loudness:
            s3gen_ref_wav = normalize_loudness(s3gen_ref_wav, _sr)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]

        with TORCH_LOCK, torch.inference_mode():
            ref_dict = self.pt_s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device="cpu")
            plen = self.t3.hp.speech_cond_prompt_len
            prompt_tokens = None
            if plen:
                tokens, _ = self.pt_s3gen.tokenizer.forward(
                    [ref_16k_wav[: self.ENC_COND_LEN]], max_len=plen
                )
                prompt_tokens = torch.atleast_2d(tokens).cpu().numpy()
            ve_embed = torch.from_numpy(
                self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
            )
            ve_embed = ve_embed.mean(axis=0, keepdim=True).float().numpy()

        self._set_conditionals(ve_embed, prompt_tokens, ref_dict)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def _encode_text(self, text: str) -> mx.array:
        """``punc_norm`` + the release tokenizer, as upstream (no BOS / EOS)."""
        enc = self.tokenizer(
            punc_norm(text), return_tensors="np", padding=True, truncation=True
        )
        ids = np.asarray(enc["input_ids"]).reshape(1, -1)
        return mx.array(ids.astype(np.int32))

    def _vocode(
        self, speech_tokens: np.ndarray, n_cfm_timesteps: int, seed
    ) -> np.ndarray:
        if seed is not None:
            mx.random.seed(int(seed))
        kwargs = {"finalize": True}
        if _accepts(self.s3gen.inference, "n_cfm_timesteps"):
            kwargs["n_cfm_timesteps"] = n_cfm_timesteps
        wav, _ = self.s3gen.inference(
            mx.array(speech_tokens[None].astype(np.int32)), self._ref_dict, **kwargs
        )
        wav = np.array(wav.astype(mx.float32)).reshape(-1)
        if mx.get_cache_memory() > self.mlx_cache_limit:
            mx.clear_cache()
        return wav

    def _watermark(self, wav: np.ndarray) -> np.ndarray:
        # Perth runs torch ops; with autograd on it leaves tensors with a
        # grad_fn behind on every call.
        with TORCH_LOCK, torch.inference_mode():
            if self._watermarker is None:
                import perth

                self._watermarker = perth.PerthImplicitWatermarker()
            out = self._watermarker.apply_watermark(wav, sample_rate=self.sr)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def _groups(self, n: int, max_batch: int) -> List[range]:
        """Consecutive equal-size chunk groups of at most ``max_batch`` chunks."""
        n_groups = math.ceil(n / max_batch)
        size = math.ceil(n / n_groups)
        return [range(g, min(n, g + size)) for g in range(0, n, size)]

    @_locked
    def generate(
        self,
        text: str,
        *,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        seed: Optional[int] = None,
        split_sentences: bool = True,
        max_batch_chunks: int = DEFAULT_MAX_BATCH_CHUNKS,
        overlap_duration: float = 0.05,
        language: str = "en",
        watermark: bool = True,
        n_cfm_timesteps: int = 2,
        max_gen_len: int = MAX_GEN_LEN,
        audio_prompt_path: Optional[Union[str, Path]] = None,
        norm_loudness: bool = True,
        exaggeration: float = 0.0,
        cfg_weight: float = 0.0,
        min_p: float = 0.0,
    ) -> np.ndarray:
        """Synthesize ``text``; returns a 1-D float32 waveform at :attr:`sr`.

        Sampling defaults are upstream's. Timings land in :attr:`last_timings`
        (``t3_s``, ``s3gen_s``, ``watermark_s``, ``n_speech_tokens``,
        ``n_chunks``, ``n_groups``, ``n_steps``, ``batch_sizes``).

        Args:
            seed: Seeds T3 sampling (chunk ``i`` uses ``seed + i``) and the
                vocoder noise.
            split_sentences: Cut long text into sentence groups and crossfade
                them; each chunk is capped at :func:`chunk_speech_cap` tokens.
                ``False`` is upstream's single-shot generation (one T3 pass of
                up to ``max_gen_len`` tokens, about 40 s of audio).
            max_batch_chunks: Decode up to this many chunks concurrently as
                rows of one T3 batch; the chunks are split into equal-size
                consecutive groups. ``1`` decodes them one by one.
            watermark: Apply the Perth watermark (upstream always does).
            audio_prompt_path: Reference voice; runs :meth:`prepare_conditionals`
                first (with ``norm_loudness``).
            exaggeration, cfg_weight, min_p: Not supported by Turbo (warned
                and ignored, as upstream).
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, norm_loudness=norm_loudness)
        if self._t3_cond_emb is None:
            raise ValueError("Call prepare_conditionals() first.")
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning(
                "CFG, min_p and exaggeration are not supported by the %s version "
                "and will be ignored.",
                self.model_label,
            )
        if not text or not text.strip():
            self.last_timings = {"n_chunks": 0}
            return np.zeros((0,), dtype=np.float32)

        pieces = plan_chunks(text, language=language) if split_sentences else [text]
        seeds = [None if seed is None else int(seed) + i for i in range(len(pieces))]
        max_batch = max(1, int(max_batch_chunks))
        t3_kwargs = dict(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_gen_len=max_gen_len,
        )

        t3_s = s3gen_s = 0.0
        n_tokens = n_steps = 0
        batch_sizes = []
        wavs: List[np.ndarray] = []
        groups = self._groups(len(pieces), max_batch)
        for group in groups:
            texts = [self._encode_text(pieces[i]) for i in group]
            stats: dict = {}
            t0 = time.perf_counter()
            # Every group runs at the first group's size (dead rows pad a
            # smaller one), so all groups share one cache and one trace. A
            # dead row costs little: a step at small batch is bandwidth-bound.
            tokens = self.t3.generate_batch(
                self._t3_cond_emb,
                texts,
                seeds=[seeds[i] for i in group],
                max_gen_lens=(
                    [chunk_speech_cap(pieces[i], max_gen_len) for i in group]
                    if split_sentences
                    else None
                ),
                n_slots=len(groups[0]),
                stats=stats,
                **t3_kwargs,
            )
            t3_s += time.perf_counter() - t0
            n_steps += stats.get("n_steps", 0)
            batch_sizes.append(stats.get("batch_size", len(group)))
            for i, toks in zip(group, tokens):
                speech = postprocess_speech_tokens(toks)
                n_tokens += len(speech) - 3
                if len(speech) == 3:
                    logger.warning(
                        "Turbo T3 produced no speech tokens for %r", pieces[i][:60]
                    )
                t1 = time.perf_counter()
                wavs.append(self._vocode(speech, n_cfm_timesteps, seeds[i]))
                s3gen_s += time.perf_counter() - t1

        # A chunk too short to crossfade (crossfade_chunks trims 20 ms and
        # overlaps ``overlap_duration`` at each end) is dropped.
        min_len = int(self.sr * (2 * overlap_duration + 0.02)) + 1
        nonempty = [w for w in wavs if len(w) >= min_len]
        if not nonempty:
            nonempty = [w for w in wavs if len(w)][:1]
        if len(nonempty) > 1:
            out = crossfade_chunks(nonempty, self.sr, overlap_duration)
            wav = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        else:
            wav = nonempty[0] if nonempty else np.zeros((0,), dtype=np.float32)
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)

        watermark_s = 0.0
        if watermark and len(wav):
            t2 = time.perf_counter()
            wav = self._watermark(wav)
            watermark_s = time.perf_counter() - t2

        self.last_timings = {
            "t3_s": t3_s,
            "s3gen_s": s3gen_s,
            "watermark_s": watermark_s,
            "n_speech_tokens": n_tokens,
            "n_chunks": len(pieces),
            "n_groups": len(groups),
            "n_steps": n_steps,
            "batch_sizes": batch_sizes,
        }
        return wav
