# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Chatterbox-Flash TTS pipeline with the block-diffusion T3 on MLX.

- T3: :class:`~chatterbox.models.t3_flash_mlx.T3FlashMLX` (fully MLX loop)
- S3Gen: meanflow-distilled vocoder, 2 CFM steps by default, either PyTorch
  (``s3gen_backend="mps"``, MPS when available) or MLX (``s3gen_backend="mlx"``)
- Conditioning (voice encoder, S3 prompt tokens, S3Gen reference dict) runs in
  PyTorch once per voice.

Parity notes vs. upstream ``chatterbox_flash.tts.ChatterboxFlashTTS``: same text
normalization (``en_us_cleaner``), token budget (``max(6 * n_text_tokens,
300)``), conditioning and EOS trimming when ``split_sentences=False``. Upstream
Flash does not watermark its output, so neither does this pipeline.

With ``split_sentences=True`` (the default) long text is cut into sentence
groups of at most :data:`MAX_CHUNK_WORDS` words, each group gets a speech-token
budget from its word count (:func:`speech_budget_for_words`), and the chunk
waveforms are crossfaded. Up to ``max_batch_chunks`` chunks are decoded
concurrently as rows of one T3 batch (:meth:`T3FlashMLX.generate_batch`), then
vocoded one by one. See :meth:`ChatterboxFlashMLX.generate`.
"""

import inspect
import logging
import functools
import math
import threading
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .generation_utils import (
    TARGET_WORDS_PER_CHUNK,
    count_words,
    crossfade_chunks,
    split_text_intelligently,
)
from .models.utils import TORCH_LOCK, materialize_mlx_state

try:
    import mlx.core as mx
except ImportError as e:  # pragma: no cover - Apple Silicon only
    raise ImportError(
        "MLX is not installed (it requires Apple Silicon). Install it with:\n"
        "  pip install 'mlx>=0.32.2'"
    ) from e

import librosa
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3gen import S3GEN_SR
from .models.s3tokenizer import S3_SR
from .models.t3.modules.t3_config import T3Config
from .models.t3_flash_mlx import T3FlashMLX, en_us_cleaner, trim_speech_tokens
from .models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-flash"
T3_FILENAME = "t3_flash.safetensors"
S3GEN_FILENAME = "s3gen.safetensors"
VE_FILENAME = "ve.safetensors"
TOKENIZER_FILENAME = "tokenizer.json"

S3GEN_BACKENDS = ("mps", "mlx")

# Early-decoding presets. The model card's "alpha" is ``time_shift_tau`` in the
# code: the card's alpha = 0 row is "no early decoding", which is what
# ``time_shift_tau = 0`` means (chatterbox_flash/model.py, generate() docstring:
# "``0.0`` disables early decoding"). The card quotes alpha = 0.5 as its default
# and alpha = 0.75 as the faster setting, while the code and eval scripts
# default to 0.1 ("paper"); generate() keeps the code default.
# Block length D. A forward is compute-bound at these sizes, so a longer
# block costs little more per step. Measured on an M4 (fp16, medium text):
# D = 16 RTF 0.43, D = 24 0.375 at the same WER, D = 32 0.29 but WER rises
# from 0.02 to 0.11. Pass block_size=16 for the paper configuration.
DEFAULT_BLOCK_SIZE = 24

EARLY_DECODING_PRESETS = {
    "off": 0.0,  # card alpha = 0: fixed K steps per block
    "paper": 0.1,  # chatterbox_flash.tts / model.py default
    "balanced": 0.5,  # card "D = 16, alpha = 0.5 (default)"
    "fast": 0.75,  # card "alpha = 0.75"
}

# Sentence chunking (split_sentences=True). The cap matches tts_mlx
# generate_long (TARGET_WORDS_PER_CHUNK); the floor matches mtl_tts_mlx
# MIN_CHUNK_WORDS. Flash garbles very short inputs (2 of 3 seeds on an
# 11-word sentence), so short chunks are merged into a neighbour.
MAX_CHUNK_WORDS = TARGET_WORDS_PER_CHUNK
MIN_CHUNK_WORDS = 8

# Per-chunk speech-token budget: S3 tokens run at 25 per second. Good Flash
# reads of the benchmark texts take 9.2-10.2 tokens per word (about 150 words
# per minute). 12.5 tokens per word is 120 words per minute, 25 % slower than
# that, plus 1 s (25 tokens) for leading / trailing silence. Anything past the
# budget is the runaway tail (repeated last clause) seen with the upstream
# budget, which is pure truncation here: the budget only sets how many blocks
# are decoded, never what an earlier block contains.
SPEECH_TOKENS_PER_WORD = 12.5
SPEECH_BUDGET_PAD = 25


def speech_len_for_text_tokens(n_text_tokens: int) -> int:
    """Speech-token budget per utterance (``chatterbox_flash.tts``)."""
    return max(n_text_tokens * 6, 300)


def speech_budget_for_words(n_words: int, block_size: int = 16) -> int:
    """Per-chunk speech-token budget from a word count (see the module notes).

    ``ceil(12.5 * n_words + 25)`` rounded up to whole blocks, at least 4 blocks.
    """
    n = math.ceil(SPEECH_TOKENS_PER_WORD * max(0, n_words) + SPEECH_BUDGET_PAD)
    n = max(n, 4 * block_size)
    return ((n + block_size - 1) // block_size) * block_size


def _locked(method):
    """Run ``method`` under the instance lock.

    A call reuses the instance's KV caches and conditioning in place, so two
    threads on one instance must not overlap (they segfault inside MLX).
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


def plan_chunks(
    text: str,
    max_words: int = MAX_CHUNK_WORDS,
    min_words: int = MIN_CHUNK_WORDS,
    language: str = "en",
    splitter=None,
) -> List[str]:
    """Split ``text`` into sentence groups of at most ``max_words`` words.

    Text within the cap stays one chunk. Otherwise sentences are grouped with
    ``splitter`` (``tts_mlx``'s long-form grouping); a group under
    ``min_words`` (a short sentence, or a short tail) is merged into its
    shorter neighbour even if that crosses the cap.
    """
    if splitter is None:
        splitter = split_text_intelligently
    text = text.strip()
    if count_words(text) <= max_words:
        return [text]
    chunks = [c.strip() for c in splitter(text, max_words, language) if c.strip()]
    i = 0
    while len(chunks) > 1 and i < len(chunks):
        if count_words(chunks[i]) >= min_words:
            i += 1
            continue
        if i == 0:
            j = 1
        elif i == len(chunks) - 1:
            j = i - 1
        else:
            prev_w = count_words(chunks[i - 1])
            next_w = count_words(chunks[i + 1])
            j = i - 1 if prev_w <= next_w else i + 1
        lo, hi = min(i, j), max(i, j)
        chunks[lo : hi + 1] = [chunks[lo] + " " + chunks[hi]]
        i = lo
    return chunks or [text]


def _accepts(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _require_meanflow(cls, where: str):
    """Raise unless some ``__init__`` in ``cls``'s MRO names a ``meanflow`` arg."""

    def _named(c):
        try:
            return "meanflow" in inspect.signature(c.__init__).parameters
        except (TypeError, ValueError):
            return False

    if not any(_named(c) for c in cls.__mro__ if "__init__" in vars(c)):
        raise RuntimeError(
            f"{where} has no `meanflow` flag. Chatterbox-Flash ships a "
            "meanflow-distilled S3Gen that needs `S3Gen(meanflow=True)`; update "
            "chatterbox.models.s3gen / s3gen_mlx to a version with meanflow support."
        )


class ChatterboxFlashMLX:
    """Chatterbox-Flash text-to-speech with the T3 decoder on MLX."""

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3FlashMLX,
        s3gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        *,
        s3gen_backend: str = "mlx",
        device: str = "cpu",
        pt_s3gen=None,
        fuse_commit: bool = False,
        mlx_cache_limit_gb: float = 2.0,
    ):
        """
        Args:
            t3: Finalized :class:`T3FlashMLX`.
            s3gen: Torch ``S3Gen(meanflow=True)`` (``"mps"``) or MLX
                ``S3Token2WavMLX(meanflow=True)`` (``"mlx"``).
            ve: Voice encoder (PyTorch, CPU).
            tokenizer: English text tokenizer.
            s3gen_backend: ``"mlx"`` (default; vocodes about 2x faster than
                MPS on an M4) or ``"mps"``.
            device: Torch device of the torch S3Gen.
            pt_s3gen: Torch S3Gen used for conditioning (defaults to ``s3gen``
                for the torch backend).
            fuse_commit: Default for fusing each block commit with the next
                block's first step.
            mlx_cache_limit_gb: Release MLX's buffer cache after vocoding once
                it holds more than this. Every new token length leaves
                1.5-6 GB of vocoder buffers behind, and unbounded the cache
                reaches ~22 GB and pushes the machine into swap.
        """
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.s3gen_backend = s3gen_backend
        self.device = device
        self.pt_s3gen = pt_s3gen if pt_s3gen is not None else s3gen
        self.fuse_commit = fuse_commit
        self.mlx_cache_limit = int(mlx_cache_limit_gb * 1024**3)
        self.last_timings: dict = {}
        self._lock = threading.RLock()

        # Per-voice conditioning (set by prepare_conditionals)
        self._t3_cond_emb: Optional[mx.array] = None
        self._ref_dict = None

        # MLX streams are per-thread; evaluate lazily-built state here.
        materialize_mlx_state(self.t3, self.s3gen if s3gen_backend == "mlx" else None)

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = REPO_ID,
        dtype: str = "float16",
        quantize_bits: Optional[int] = None,
        s3gen_backend: str = "mlx",
        fuse_commit: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> "ChatterboxFlashMLX":
        """Download a Chatterbox-Flash release from the Hugging Face Hub and load it."""
        local_path = None
        for fname in (VE_FILENAME, T3_FILENAME, S3GEN_FILENAME, TOKENIZER_FILENAME):
            local_path = hf_hub_download(repo_id=repo_id, filename=fname)
        return cls.from_local(
            Path(local_path).parent,
            dtype=dtype,
            quantize_bits=quantize_bits,
            s3gen_backend=s3gen_backend,
            fuse_commit=fuse_commit,
            block_size=block_size,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Union[str, Path],
        dtype: str = "float16",
        quantize_bits: Optional[int] = None,
        s3gen_backend: str = "mlx",
        fuse_commit: bool = False,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> "ChatterboxFlashMLX":
        """Load from a directory holding the four Chatterbox-Flash files.

        ``block_size`` is the diffusion block length ``D``. The model card
        reports D = 16 (paper default), 24 and 32; a larger block puts more
        tokens in each forward at nearly the same cost. See
        :data:`DEFAULT_BLOCK_SIZE` for why the default is 24.
        """
        if s3gen_backend not in S3GEN_BACKENDS:
            raise ValueError(
                f"s3gen_backend must be one of {S3GEN_BACKENDS}, got {s3gen_backend!r}"
            )
        ckpt_dir = Path(ckpt_dir)
        from .models.s3gen import S3Gen

        _require_meanflow(S3Gen, "chatterbox.models.s3gen.S3Gen")

        with TORCH_LOCK:
            logger.info("Loading voice encoder...")
            ve = VoiceEncoder()
            ve.load_state_dict(load_file(ckpt_dir / VE_FILENAME))
            ve.to("cpu").eval()

            device = "cpu"
            if s3gen_backend == "mps" and torch.backends.mps.is_available():
                device = "mps"
            logger.info("Loading meanflow S3Gen (PyTorch on %s)...", device)
            s3_state = load_file(ckpt_dir / S3GEN_FILENAME)
            pt_s3gen = S3Gen(meanflow=True)
            pt_s3gen.load_state_dict(s3_state, strict=False)
            pt_s3gen.to(device).eval()

        logger.info(
            "Loading Flash T3 into MLX (dtype=%s, quantize_bits=%s)...",
            dtype,
            quantize_bits,
        )
        t3 = T3FlashMLX(T3Config.english_only(), block_size=block_size)
        t3.load_torch_state_dict(load_file(ckpt_dir / T3_FILENAME))
        t3.finalize(dtype, quantize_bits=quantize_bits)

        s3gen = pt_s3gen
        if s3gen_backend == "mlx":
            from .models.s3gen_mlx.convert_weights import convert_and_load_weights
            from .models.s3gen_mlx.s3gen_mlx import S3Token2WavMLX

            _require_meanflow(S3Token2WavMLX, "S3Token2WavMLX")
            logger.info("Loading meanflow S3Gen MLX vocoder...")
            s3gen = S3Token2WavMLX(meanflow=True)
            weights = convert_and_load_weights(str(ckpt_dir / S3GEN_FILENAME))
            s3gen.load_weights(
                [(k, mx.array(v)) for k, v in weights.items()], strict=False
            )
            s3gen.eval()
            # The torch S3Gen now only runs embed_ref (mel extractor, speaker
            # encoder, tokenizer); its flow and HiFiGAN are a second copy of
            # the vocoder (135M params, ~540 MB fp32).
            with TORCH_LOCK:
                del pt_s3gen.flow, pt_s3gen.mel2wav

        tokenizer = EnTokenizer(str(ckpt_dir / TOKENIZER_FILENAME))
        # Weight conversion leaves ~2 GB in MLX's buffer cache.
        mx.clear_cache()
        return cls(
            t3,
            s3gen,
            ve,
            tokenizer,
            s3gen_backend=s3gen_backend,
            device=device,
            pt_s3gen=pt_s3gen,
            fuse_commit=fuse_commit,
        )

    # ------------------------------------------------------------------ #
    #  Conditioning
    # ------------------------------------------------------------------ #

    @_locked
    def release_caches(self) -> None:
        """Free T3's KV caches, compiled traces and MLX's buffer cache.

        Weights and conditioning stay; the next :meth:`generate` rebuilds
        the caches. Call it after a long document to return the memory.
        """
        self.t3.release_caches()

    @_locked
    def prepare_conditionals(
        self, wav_fpath: Union[str, Path], exaggeration: float = 0.5
    ):
        """Build speaker / prompt conditioning from a reference wav.

        Same segmentation as upstream: the first 6 s (16 kHz) give the T3 prompt
        speech tokens and the VE embedding input, the first 10 s (24 kHz) give
        the S3Gen reference dict.
        """
        s3gen_ref_wav, _sr = librosa.load(str(wav_fpath), sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]

        with TORCH_LOCK, torch.inference_mode():
            ref_dict = self.pt_s3gen.embed_ref(
                s3gen_ref_wav, S3GEN_SR, device=self.device
            )
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

        if self.s3gen_backend == "mlx":
            prompt_feat = mx.array(ref_dict["prompt_feat"].float().cpu().numpy())
            ref_dict = {
                "prompt_token": mx.array(ref_dict["prompt_token"].cpu().numpy()),
                "prompt_token_len": int(ref_dict["prompt_token_len"][0].item()),
                "prompt_feat": prompt_feat,
                "prompt_feat_len": prompt_feat.shape[1],
                "embedding": mx.array(ref_dict["embedding"].float().cpu().numpy()),
            }

        t3_cond = T3CondMLX(
            speaker_emb=mx.array(ve_embed),
            cond_prompt_speech_tokens=(
                None if prompt_tokens is None else mx.array(prompt_tokens)
            ),
            emotion_adv=float(exaggeration),
        )
        cond_emb = self.t3.prepare_conditioning(t3_cond)
        mx.eval(cond_emb)
        self._t3_cond_emb = cond_emb
        self._ref_dict = ref_dict
        materialize_mlx_state(self._ref_dict)

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def _encode_text(self, text: str, normalize_text: bool = True) -> mx.array:
        if normalize_text:
            text = en_us_cleaner(text)
        with TORCH_LOCK:
            ids = self.tokenizer.text_to_tokens(text).cpu().numpy().reshape(-1)
        hp = self.t3.hp
        ids = np.concatenate([[hp.start_text_token], ids, [hp.stop_text_token]])
        return mx.array(ids[None].astype(np.int32))

    def _vocode(
        self, speech_tokens: np.ndarray, n_cfm_timesteps: int, seed
    ) -> np.ndarray:
        if self.s3gen_backend == "mlx":
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

        with TORCH_LOCK, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(int(seed))
            wav, _ = self.s3gen.inference(
                speech_tokens=torch.from_numpy(speech_tokens).to(self.device),
                ref_dict=self._ref_dict,
                n_cfm_timesteps=n_cfm_timesteps,
            )
            return wav.detach().float().cpu().numpy().reshape(-1)

    def _encode_with_budget(
        self,
        text: str,
        budget: Optional[int],
        normalize_text: bool,
        max_speech_tokens: Optional[int],
    ):
        """Text ids and the speech-token budget for one chunk.

        ``budget=None`` uses the upstream budget from the text-token count.
        """
        text_tokens = self._encode_text(text, normalize_text=normalize_text)
        total = speech_len_for_text_tokens(int(text_tokens.shape[1]))
        if budget is not None:
            total = min(total, budget)
        if max_speech_tokens is not None:
            total = min(total, max_speech_tokens)
        return text_tokens, total

    def _vocode_chunk(self, text: str, tokens, n_cfm_timesteps: int, seed):
        """Trim and vocode one chunk's T3 output; returns ``(wav, n_tokens, s)``."""
        speech_tokens = trim_speech_tokens(tokens, self.t3.hp)
        t1 = time.perf_counter()
        if len(speech_tokens) == 0:
            logger.warning("Flash T3 produced no speech tokens for %r", text[:60])
            wav = np.zeros((0,), dtype=np.float32)
        else:
            wav = self._vocode(speech_tokens, n_cfm_timesteps, seed)
        s3gen_s = time.perf_counter() - t1
        return wav.astype(np.float32, copy=False), len(speech_tokens), s3gen_s

    def _generate_chunk(
        self,
        text: str,
        budget: Optional[int],
        *,
        seed: Optional[int],
        n_cfm_timesteps: int,
        normalize_text: bool,
        max_speech_tokens: Optional[int],
        t3_kwargs: dict,
    ):
        """One T3 + vocoder pass; returns ``(wav, timings)``.

        ``budget=None`` uses the upstream budget from the text-token count.
        """
        text_tokens, total = self._encode_with_budget(
            text, budget, normalize_text, max_speech_tokens
        )
        stats: dict = {}
        t0 = time.perf_counter()
        tokens = self.t3.generate(
            self._t3_cond_emb, text_tokens, total, seed=seed, stats=stats, **t3_kwargs
        )
        t3_s = time.perf_counter() - t0
        wav, n_tok, s3gen_s = self._vocode_chunk(text, tokens, n_cfm_timesteps, seed)
        timings = {
            "t3_s": t3_s,
            "s3gen_s": s3gen_s,
            "n_speech_tokens": int(n_tok),
            "speech_budget": int(total),
            "n_forwards": stats.get("n_forwards", 0),
            "n_blocks": stats.get("n_blocks", 0),
            "batch_rounds": stats.get("n_blocks", 0),
            "steps_per_block": stats.get("steps_per_block", []),
        }
        return wav, timings

    def _generate_group(
        self,
        texts: List[str],
        budgets: List[int],
        *,
        seeds: List[Optional[int]],
        n_slots: int,
        n_cfm_timesteps: int,
        normalize_text: bool,
        max_speech_tokens: Optional[int],
        t3_kwargs: dict,
    ):
        """Batched T3 pass over ``texts``, then one vocoder pass per chunk.

        Returns ``(wavs, timings)``. T3 time and forwards are for the whole
        group: the first chunk's timings carry them, the others carry zero,
        so sums over chunks stay correct.
        """
        enc = [
            self._encode_with_budget(t, b, normalize_text, max_speech_tokens)
            for t, b in zip(texts, budgets)
        ]
        kwargs = {k: v for k, v in t3_kwargs.items() if k != "fuse_commit"}
        stats: dict = {}
        t0 = time.perf_counter()
        tokens = self.t3.generate_batch(
            self._t3_cond_emb,
            [e[0] for e in enc],
            [e[1] for e in enc],
            seeds=seeds,
            n_slots=n_slots,
            stats=stats,
            **kwargs,
        )
        t3_s = time.perf_counter() - t0
        wavs, timings = [], []
        for i, (text, toks, seed) in enumerate(zip(texts, tokens, seeds)):
            wav, n_tok, s3gen_s = self._vocode_chunk(text, toks, n_cfm_timesteps, seed)
            wavs.append(wav)
            timings.append(
                {
                    "t3_s": t3_s if i == 0 else 0.0,
                    "s3gen_s": s3gen_s,
                    "n_speech_tokens": int(n_tok),
                    "speech_budget": int(enc[i][1]),
                    "n_forwards": stats["n_forwards"] if i == 0 else 0,
                    "n_blocks": stats["n_blocks"][i],
                    "batch_rounds": stats["n_rounds"] if i == 0 else 0,
                    "steps_per_block": stats["steps_per_block"][i],
                }
            )
        return wavs, timings

    @_locked
    def generate(
        self,
        text: str,
        *,
        num_steps: int = 10,
        temperature: float = 0.6,
        time_shift_tau: float = 0.1,
        omnivoice_schedule_t_shift: float = 0.5,
        cfg_scale: float = 1.0,
        position_temperature: float = 5.0,
        n_cfm_timesteps: int = 2,
        seed: Optional[int] = None,
        normalize_text: bool = True,
        max_speech_tokens: Optional[int] = None,
        fuse_commit: Optional[bool] = None,
        preset: Optional[str] = None,
        split_sentences: bool = True,
        overlap_duration: float = 0.05,
        language: str = "en",
        max_batch_chunks: int = 4,
    ) -> np.ndarray:
        """Synthesize ``text``; returns a 1-D float32 waveform at :attr:`sr`.

        Decoding defaults are the paper / upstream ``tts.generate`` defaults.
        Timings and decoder statistics land in :attr:`last_timings` (summed
        over chunks, plus ``n_chunks`` and ``batch_rounds``: the number of
        T3 block rounds, which is the block count when chunks run one by one).

        Args:
            time_shift_tau: Early-decoding strength; the model card calls it
                alpha (0 = off, card default 0.5, fast 0.75; code default 0.1).
            preset: Name in :data:`EARLY_DECODING_PRESETS`; overrides
                ``time_shift_tau``.
            split_sentences: Cut text over :data:`MAX_CHUNK_WORDS` words into
                sentence groups (short groups merged), budget each chunk from
                its word count (:func:`speech_budget_for_words`, never above the
                upstream budget) and crossfade the chunks. ``False`` is the
                upstream single-shot behaviour with the upstream budget.
            overlap_duration: Crossfade between chunks, in seconds.
            language: Sentence-splitter language.
            max_batch_chunks: Decode up to this many chunks concurrently as
                rows of one T3 batch (:meth:`T3FlashMLX.generate_batch`).
                Consecutive chunks are grouped into equal-size groups. ``1``
                decodes them one by one with :meth:`T3FlashMLX.generate`.
                Seeded sampling then matches one-by-one decoding except
                through float rounding of the larger batch, or in a last block
                cut short by ``max_speech_tokens`` or the upstream budget
                (its noise is drawn at full block shape). ``fuse_commit``
                applies to one-by-one decoding only.
        """
        if self._t3_cond_emb is None:
            raise ValueError("Call prepare_conditionals() first.")
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        if not text.strip():
            self.last_timings = {"n_chunks": 0}
            return np.zeros((0,), dtype=np.float32)
        if preset is not None:
            if preset not in EARLY_DECODING_PRESETS:
                raise ValueError(
                    f"preset must be one of {list(EARLY_DECODING_PRESETS)}, "
                    f"got {preset!r}"
                )
            time_shift_tau = EARLY_DECODING_PRESETS[preset]
        fuse = self.fuse_commit if fuse_commit is None else fuse_commit
        t3_kwargs = dict(
            num_steps=num_steps,
            temperature=temperature,
            time_shift_tau=time_shift_tau,
            omnivoice_schedule_t_shift=omnivoice_schedule_t_shift,
            cfg_scale=cfg_scale,
            position_temperature=position_temperature,
            fuse_commit=fuse,
        )
        chunk_kwargs = dict(
            n_cfm_timesteps=n_cfm_timesteps,
            normalize_text=normalize_text,
            max_speech_tokens=max_speech_tokens,
            t3_kwargs=t3_kwargs,
        )

        if not split_sentences:
            wav, t = self._generate_chunk(text, None, seed=seed, **chunk_kwargs)
            wavs, chunks_t = [wav], [t]
        else:
            # Conditioning and the T3 KV cache / compiled traces are reused
            # across chunks; only the text changes.
            pieces = plan_chunks(text, language=language)
            budgets, seeds = [], []
            for i, piece in enumerate(pieces):
                spoken = en_us_cleaner(piece) if normalize_text else piece
                budgets.append(
                    speech_budget_for_words(count_words(spoken), self.t3.block_size)
                )
                seeds.append(None if seed is None else int(seed) + i)
            wavs, chunks_t = [], []
            max_batch = max(1, int(max_batch_chunks))
            if max_batch == 1 or len(pieces) == 1:
                for piece, budget, chunk_seed in zip(pieces, budgets, seeds):
                    wav, t = self._generate_chunk(
                        piece, budget, seed=chunk_seed, **chunk_kwargs
                    )
                    wavs.append(wav)
                    chunks_t.append(t)
            else:
                # Equal-size groups; a smaller group is padded with dead slots
                # so every group shares one batched cache and one trace. A
                # leftover single chunk runs unbatched instead: a forward
                # costs nearly linearly in rows, so a dead slot is not free.
                n_groups = math.ceil(len(pieces) / max_batch)
                size = math.ceil(len(pieces) / n_groups)
                for g in range(0, len(pieces), size):
                    if len(pieces) - g == 1:
                        wav, t = self._generate_chunk(
                            pieces[g], budgets[g], seed=seeds[g], **chunk_kwargs
                        )
                        wavs.append(wav)
                        chunks_t.append(t)
                        continue
                    w, t = self._generate_group(
                        pieces[g : g + size],
                        budgets[g : g + size],
                        seeds=seeds[g : g + size],
                        n_slots=size,
                        **chunk_kwargs,
                    )
                    wavs += w
                    chunks_t += t

        # A chunk that ends after a token or two is too short to crossfade:
        # crossfade_chunks trims 20 ms and overlaps ``overlap_duration`` at
        # each end, and a shorter chunk breaks its placement.
        min_len = int(self.sr * (2 * overlap_duration + 0.02)) + 1
        nonempty = [w for w in wavs if len(w) >= min_len]
        if not nonempty:
            nonempty = [w for w in wavs if len(w)][:1]
        if len(nonempty) > 1:
            out = crossfade_chunks(nonempty, self.sr, overlap_duration)
            wav = out.numpy() if hasattr(out, "numpy") else np.asarray(out)
        else:
            wav = nonempty[0] if nonempty else np.zeros((0,), dtype=np.float32)

        steps = [s for t in chunks_t for s in t["steps_per_block"]]
        self.last_timings = {
            "t3_s": sum(t["t3_s"] for t in chunks_t),
            "s3gen_s": sum(t["s3gen_s"] for t in chunks_t),
            "n_speech_tokens": sum(t["n_speech_tokens"] for t in chunks_t),
            "speech_budget": sum(t["speech_budget"] for t in chunks_t),
            "n_forwards": sum(t["n_forwards"] for t in chunks_t),
            "n_blocks": sum(t["n_blocks"] for t in chunks_t),
            "batch_rounds": sum(t["batch_rounds"] for t in chunks_t),
            "avg_steps_per_block": float(np.mean(steps)) if steps else 0.0,
            "n_chunks": len(chunks_t),
        }
        return np.asarray(wav, dtype=np.float32).reshape(-1)
