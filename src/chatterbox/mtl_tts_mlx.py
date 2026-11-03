# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
MLX-optimized Multilingual TTS pipeline for Chatterbox.
Provides significant performance improvements on Apple Silicon (M1/M2/M3/M4).

This implementation uses a hybrid approach:
- MLX for T3 (autoregressive text-to-speech token generation) - main performance win
- S3Gen (vocoder), selected with ``vocoder=`` at load time:

  - ``"pt"`` (default, as in 1.2.0): the multilingual S3Gen in PyTorch on MPS,
    5 midpoint CFM steps.
  - ``"mlx"``: the same S3Gen weights on MLX (``S3Token2WavMLX``), 5 steps.
  - ``"meanflow"``: the meanflow-distilled S3Gen of Chatterbox-Turbo on MLX,
    2 steps. It needs ``s3gen_meanflow.safetensors`` from the
    ``ResembleAI/chatterbox-turbo`` snapshot (``meanflow_path=``).

  Voice conditioning (``prepare_conditionals``) always uses the PyTorch S3Gen's
  encoders.

:meth:`ChatterboxMultilingualTTSMLX.generate_stream` yields audio pieces while
T3 still decodes (token windows, as Turbo's ``generate_stream``).

Supports 23 languages including English, Spanish, French, German, Japanese, Chinese, etc.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional, List, Dict, Union
import functools
import logging
import re
import os
import threading
import time

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    _mlx_import_error = (
        "MLX is not installed (it requires Apple Silicon). Install it with:\n"
        "  pip install chatterbox-mlx\n"
        "or manually:\n"
        "  pip install 'mlx>=0.32.2'"
    )

if not MLX_AVAILABLE:
    raise ImportError(_mlx_import_error)

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors

from .models.t3_mlx.t3_mlx import T3MLX
from .models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from .models.t3.modules.t3_config import T3Config
from .models.t3.modules.cond_enc import T3Cond

# Use PyTorch S3Gen for now (hybrid approach - T3 in MLX, S3Gen in PyTorch/MPS).
# S3Gen itself is imported where it's constructed: its diffusers dependency
# imports transformers, which the MLX path shouldn't pay for at import time.
from .models.s3gen import S3GEN_SR

if TYPE_CHECKING:
    from .models.s3gen import S3Gen
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.voice_encoder import VoiceEncoder
from .models.tokenizers import MTLTokenizer
from .models.utils import TORCH_LOCK, clear_device_memory, materialize_mlx_state

# Shared generation utilities
from .generation_utils import (
    count_words,
    SPACY_AVAILABLE,
    split_into_sentences,
    get_adaptive_chunks,
    split_text_intelligently,
    crossfade_chunks,
    estimate_max_tokens,
    print_generation_plan,
    print_chunk_generating,
    print_chunk_completed,
    print_generation_complete,
    print_crossfading,
)

from .streaming_mlx import (
    S3GEN_PRE_LOOKAHEAD,
    S3GEN_SIL,
    STREAM_BLOCK_TOKENS,
    STREAM_CONTEXT_TOKENS,
    STREAM_FIRST_CHUNK_TOKENS,
    STREAM_LOOKAHEAD_TOKENS,
    MLXVocoderHost,
    _ChunkJoiner,
    _WindowVocoder,
    ref_dict_to_mlx,
)

import librosa

from .watermarking import PerthWatermarkMixin, StreamWatermark

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox"

# --- Vocoder choice ------------------------------------------------------- #
VOCODERS = ("pt", "mlx", "meanflow")
# CFM steps when ``n_cfm_timesteps`` is None ("pt": the PyTorch flow default, 5).
DEFAULT_CFM_STEPS = {"pt": None, "mlx": 5, "meanflow": 2}
# The meanflow vocoder ships with Chatterbox-Turbo, not with this model.
MEANFLOW_REPO_ID = "ResembleAI/chatterbox-turbo"
MEANFLOW_FILENAME = "s3gen_meanflow.safetensors"

# --- Streaming (generate_stream) ---------------------------------------- #
# Second token window. Turbo uses 50 (STREAM_CHUNK_TOKENS). Measured on the
# multilingual model (fp16 T3 + meanflow, M4, 24 replies x 3 runs): with 50
# the second window of the first sentence left only ~0.09 s of playback
# slack and gapped in 2-8 of 24 replies (0.01-0.37 s); with 40, 0 gaps in
# 2 runs, same first-audio time (0.60 s) and quality.
MTL_STREAM_CHUNK_TOKENS = 40

# --- T3 precision ---------------------------------------------------------- #
T3_DTYPES = {
    "fp32": "float32",
    "float32": "float32",
    "fp16": "float16",
    "float16": "float16",
}
T3_QUANT_GROUP_SIZE = 64


def _check_vocoder(vocoder: str) -> str:
    if vocoder not in VOCODERS:
        raise ValueError(f"vocoder must be one of {VOCODERS}, got {vocoder!r}")
    return vocoder


def resolve_meanflow_path(
    meanflow_path: Optional[Union[str, Path]], ckpt_dir: Optional[Path] = None
) -> Path:
    """Locate ``s3gen_meanflow.safetensors`` for ``vocoder="meanflow"``.

    ``meanflow_path`` is the file itself or a directory that holds it (for
    example the local ``ResembleAI/chatterbox-turbo`` snapshot). With
    ``None``, ``ckpt_dir`` is searched. Nothing is downloaded here; get the
    file with ``huggingface_hub.hf_hub_download("ResembleAI/chatterbox-turbo",
    "s3gen_meanflow.safetensors")`` or use :meth:`from_pretrained`.
    """
    candidates = []
    if meanflow_path is not None:
        p = Path(meanflow_path)
        candidates.append(p / MEANFLOW_FILENAME if p.is_dir() else p)
    elif ckpt_dir is not None:
        candidates.append(Path(ckpt_dir) / MEANFLOW_FILENAME)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"vocoder='meanflow' needs {MEANFLOW_FILENAME} from {MEANFLOW_REPO_ID} "
        f"(looked in: {', '.join(str(c) for c in candidates) or 'nowhere'}). "
        "Pass meanflow_path=<file or Turbo snapshot dir>, e.g. "
        f"huggingface_hub.hf_hub_download({MEANFLOW_REPO_ID!r}, {MEANFLOW_FILENAME!r})."
    )


def load_mlx_vocoder(weights_path: Union[str, Path], meanflow: bool):
    """An MLX ``S3Token2WavMLX`` with the weights of a PyTorch S3Gen checkpoint."""
    from .models.s3gen_mlx.convert_weights import convert_and_load_weights
    from .models.s3gen_mlx.s3gen_mlx import S3Token2WavMLX

    voc = S3Token2WavMLX(meanflow=meanflow)
    weights = convert_and_load_weights(str(weights_path))
    voc.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=False)
    voc.eval()
    mx.eval(voc.parameters())
    # Weight conversion leaves GBs in MLX's buffer cache.
    mx.clear_cache()
    return voc


def apply_t3_precision(
    t3: T3MLX, t3_dtype: str = "float32", quantize_bits: Optional[int] = None
) -> T3MLX:
    """Cast T3 to ``t3_dtype`` and optionally quantize it (4 or 8 bits, group 64).

    ``"float32"`` without quantization leaves T3 untouched (the 1.2.0
    weights). Otherwise ``T3MLX.finalize(dtype=..., quantize_bits=...)`` does
    the work; a T3 without ``finalize`` is cast with ``set_dtype`` and its
    transformer quantized with ``mlx.nn.quantize``.
    """
    if t3_dtype not in T3_DTYPES:
        raise ValueError(f"t3_dtype must be one of {list(T3_DTYPES)}, got {t3_dtype!r}")
    if quantize_bits not in (None, 4, 8):
        raise ValueError("quantize_bits must be None, 4 or 8")
    dtype = T3_DTYPES[t3_dtype]
    if dtype == "float32" and quantize_bits is None:
        return t3
    finalize = getattr(t3, "finalize", None)
    if callable(finalize):
        finalize(dtype=dtype, quantize_bits=quantize_bits)
    else:
        import mlx.nn as nn

        if dtype != "float32":
            t3.set_dtype(getattr(mx, dtype))
        if quantize_bits is not None:
            nn.quantize(t3.tfmr, group_size=T3_QUANT_GROUP_SIZE, bits=quantize_bits)
        t3.eval()
    mx.eval(t3.parameters())
    return t3


def _exclusive(method):
    """Run ``method`` under the instance lock; refuse while a stream is open."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            self._check_no_stream()
            return method(self, *args, **kwargs)

    return wrapper


# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


# Minimum words per generated chunk. Sentences shorter than this are
# merged with their neighbours before synthesis.
#
# generate() synthesizes each sentence independently, and estimate_max_tokens
# floors the budget at 80 tokens -- 3.2s at S3_TOKEN_RATE = 25Hz -- for short
# text. A 3-word sentence needs about 1s, leaving ~2s of budget the model
# fills with invented speech when EOS does not fire promptly.
#
# Measured, 240 generations per condition (single voice, English, ASR-scored
# insertions against the reference):
#
#     2-word chunk    10.0%      hallucination rate
#     3-word chunk    33.8%
#     4-word chunk    10.0%
#     9-word chunk     1.2%
#
# Per item containing three short chunks: 44.7% -> 3.7%.
#
# generate_long() never had this problem because get_adaptive_chunks groups
# sentences to ~50 words; only the short-form path was exposed.
MIN_CHUNK_WORDS = 8
MAX_CHUNK_WORDS = 30

_TERMINATOR_RE = re.compile(r"[.!?。！？]+[\"”]*$")
_FULLWIDTH_TERMINATORS = frozenset("。！？")


def merge_short_sentences(
    sentences: List[str],
    min_words: int = MIN_CHUNK_WORDS,
    max_words: int = MAX_CHUNK_WORDS,
) -> List[str]:
    """Group adjacent sentences until each group reaches ``min_words``.

    Within a group every sentence but the last has its terminal ``.!?``
    replaced by a comma (``。！？`` by a full-width ``，``), so the group
    reads as one utterance and is synthesized as a single generation. No words are added, removed or
    reordered.

    Sentences already at or above ``min_words`` pass through untouched,
    and no group exceeds ``max_words`` -- the goal is to lift tiny chunks
    out of the hallucination-prone range, not to maximize chunk size.
    """
    if len(sentences) < 2:
        return sentences

    groups: List[List[str]] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        words = count_words(sentence)
        if current and (
            current_words >= min_words or current_words + words > max_words
        ):
            groups.append(current)
            current, current_words = [], 0
        current.append(sentence)
        current_words += words

    if current:
        # Never leave a short tail as its own chunk -- that is the exact
        # case this function exists to prevent.
        if groups and current_words < min_words:
            tail_room = sum(count_words(s) for s in groups[-1]) + current_words
            if tail_room <= max_words:
                groups[-1] = groups[-1] + current
                current = []
        if current:
            groups.append(current)

    merged = [_join_sentence_group(g) for g in groups]
    if len(merged) != len(sentences):
        logger.debug(
            "merged %d sentences into %d chunk(s): %s -> %s words",
            len(sentences),
            len(merged),
            [count_words(s) for s in sentences],
            [count_words(s) for s in merged],
        )
    return merged


def _join_sentence_group(group: List[str]) -> str:
    """Comma-join a group so the model treats it as one utterance."""
    if len(group) == 1:
        return group[0]
    parts = []
    for sentence in group[:-1]:
        terminator = _TERMINATOR_RE.search(sentence)
        comma = (
            "，"
            if terminator and _FULLWIDTH_TERMINATORS.intersection(terminator.group())
            else ","
        )
        stripped = _TERMINATOR_RE.sub("", sentence).rstrip()
        if stripped:
            parts.append(stripped + comma)
    parts.append(group[-1])
    return " ".join(parts)


def punc_norm(text: str, debug: bool = True) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset
    """
    original_text = text

    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (
            """, '"'),
        (""",
            '"',
        ),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Clean up leading quotes and spaces
    # This handles sentences that start with: '" "No entendía...' -> 'No entendía...'
    while text and text[0] in "\"'":
        text = text[1:].lstrip()

    # Clean up trailing quotes and spaces in various patterns
    # Remove trailing whitespace first
    text = text.rstrip()

    # Remove standalone quotes at end (with or without spaces before them)
    # This handles: '." "' -> '."' and then '."' -> '.'
    while text and text[-1] in "\"'":
        text = text[:-1].rstrip()

    # Now handle punctuation followed by quotes: '."' -> '.'
    text = re.sub(r'([.!?,])["\']+$', r"\1", text)
    text = re.sub(r'["\']+([.!?,])$', r"\1", text)

    # Clean up double punctuation (e.g., ".." or ",," but not "...")
    text = re.sub(r"([.!?,])\1+", r"\1", text)

    # Clean up double spaces that may have been introduced
    text = " ".join(text.split())

    # Skip empty text after cleanup
    if len(text) == 0:
        return ""

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "、", "，", "。", "？", "！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    # DEBUG: Log punc_norm changes
    if debug and original_text != text:
        logger.info(f"[punc_norm DEBUG] Input:  {repr(original_text[:200])}")
        logger.info(f"[punc_norm DEBUG] Output: {repr(text[:200])}")

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen (same structure as mtl_tts.py)
    """

    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(t3=self.t3.__dict__, gen=self.gen)
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs["t3"]), kwargs["gen"])


class ChatterboxMultilingualTTSMLX(PerthWatermarkMixin):
    """
    MLX-optimized Multilingual Chatterbox TTS pipeline.

    Uses a hybrid approach for best performance on Apple Silicon:
    - MLX for T3 (autoregressive token generation) - main speedup
    - PyTorch/MPS for S3Gen (vocoder) - fast on Apple Silicon

    Supports 23 languages including English, Spanish, French, German,
    Japanese, Chinese, Korean, Arabic, Hindi, etc.
    """

    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3MLX,
        s3gen: "S3Gen",
        ve: VoiceEncoder,
        tokenizer: MTLTokenizer,
        device: str = "mps",
        conds: Conditionals = None,
        *,
        vocoder: str = "pt",
        s3gen_mlx=None,
        mlx_cache_limit_gb: float = 2.0,
    ):
        """
        Initialize hybrid MLX/PyTorch multilingual TTS pipeline.

        Args:
            t3: T3MLX model (MLX)
            s3gen: S3Gen (PyTorch). With ``vocoder="pt"`` it vocodes; it
                always provides the conditioning encoders (``embed_ref``, S3
                tokenizer).
            ve: Voice encoder (PyTorch)
            tokenizer: Multilingual text tokenizer
            device: PyTorch device for S3Gen ("mps" or "cpu")
            conds: Optional pre-computed conditioning
            vocoder: ``"pt"`` (PyTorch S3Gen, default), ``"mlx"`` or
                ``"meanflow"`` (``s3gen_mlx`` vocodes).
            s3gen_mlx: MLX ``S3Token2WavMLX`` for ``vocoder="mlx"``
                (``meanflow=False``, multilingual ``s3gen.pt`` weights) or
                ``"meanflow"`` (``meanflow=True``, Turbo's
                ``s3gen_meanflow.safetensors``).
            mlx_cache_limit_gb: With an MLX vocoder, release MLX's buffer
                cache after a vocode once it holds more than this.
        """
        self.sr = S3GEN_SR  # Sample rate: 24kHz
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        # NOTE (watermark): generate(), generate_long() and generate_stream()
        # take ``watermark`` (default False). The default keeps the output of
        # every release up to 1.3.0 and of upstream mtl_tts.py, which builds
        # a watermarker but never applies it. With ``watermark=True`` the
        # Perth watermark is applied as Turbo does: once to the whole output,
        # or per stream piece with the previous audio as left context. The
        # watermarker (``self.watermarker``) is built on first use, not here,
        # so an engine that never watermarks pays no load time or memory.
        self._watermarker = None

        self.vocoder = _check_vocoder(vocoder)
        if vocoder != "pt":
            if s3gen_mlx is None:
                raise ValueError(f"vocoder={vocoder!r} needs s3gen_mlx")
            if bool(getattr(s3gen_mlx, "meanflow", vocoder == "meanflow")) != (
                vocoder == "meanflow"
            ):
                raise ValueError(
                    f"vocoder={vocoder!r} does not match s3gen_mlx.meanflow="
                    f"{getattr(s3gen_mlx, 'meanflow', None)}"
                )
        self.s3gen_mlx = s3gen_mlx
        self.mlx_cache_limit = int(mlx_cache_limit_gb * 1024**3)
        self.last_timings: dict = {}
        # One call at a time per instance (calls share T3's state and the
        # conditioning). generate_stream holds it while its generator is open.
        self._lock = threading.RLock()
        self._stream_active = 0
        # MLX copy of conds.gen for the MLX vocoders, rebuilt when conds change.
        self._mlx_ref_src = None
        self._mlx_ref = None

        # Evaluate lazily-built MLX state now, so the model can be used from
        # threads other than the one that loaded it (MLX streams are per-thread).
        materialize_mlx_state(self.t3, self.conds, self.s3gen_mlx)

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Return dictionary of supported language codes and names."""
        return SUPPORTED_LANGUAGES.copy()

    @classmethod
    def from_pretrained(
        cls,
        cache_dir: Optional[str] = None,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
        *,
        vocoder: str = "pt",
        meanflow_path: Optional[Union[str, Path]] = None,
        t3_dtype: str = "float32",
        quantize_bits: Optional[int] = None,
        mlx_cache_limit_gb: float = 2.0,
    ) -> "ChatterboxMultilingualTTSMLX":
        """
        Load pre-trained multilingual Chatterbox TTS model with MLX optimization.

        Args:
            cache_dir: Optional cache directory for model files
            t3_config: Optional T3 configuration (default: multilingual)
            use_default_speaker: Whether to load default speaker conditioning
            device: PyTorch device for S3Gen ("mps" or "cpu")
            vocoder, meanflow_path, t3_dtype, quantize_bits, mlx_cache_limit_gb:
                As :meth:`from_local`. With ``vocoder="meanflow"`` and no
                ``meanflow_path``, this call also downloads
                ``s3gen_meanflow.safetensors`` from ``ResembleAI/chatterbox-turbo``.

        Returns:
            ChatterboxMultilingualTTSMLX instance
        """
        _check_vocoder(vocoder)
        logger.info("Loading Chatterbox Multilingual TTS with MLX optimization...")

        # Download model files from HuggingFace
        ckpt_dir = Path(
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_mtl23ls_v2.safetensors",
                    "s3gen.pt",
                    "grapheme_mtl_merged_expanded_v1.json",
                    "conds.pt",
                    "Cangjie5_TC.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )

        if vocoder == "meanflow" and meanflow_path is None:
            from huggingface_hub import hf_hub_download

            meanflow_path = hf_hub_download(
                MEANFLOW_REPO_ID,
                MEANFLOW_FILENAME,
                cache_dir=cache_dir,
                token=os.getenv("HF_TOKEN") or None,
            )

        return cls.from_local(
            ckpt_dir,
            t3_config,
            use_default_speaker,
            device,
            vocoder=vocoder,
            meanflow_path=meanflow_path,
            t3_dtype=t3_dtype,
            quantize_bits=quantize_bits,
            mlx_cache_limit_gb=mlx_cache_limit_gb,
        )

    @classmethod
    def from_local(
        cls,
        ckpt_dir: Path,
        t3_config: Optional[T3Config] = None,
        use_default_speaker: bool = True,
        device: str = "mps",
        *,
        vocoder: str = "pt",
        meanflow_path: Optional[Union[str, Path]] = None,
        t3_dtype: str = "float32",
        quantize_bits: Optional[int] = None,
        mlx_cache_limit_gb: float = 2.0,
    ) -> "ChatterboxMultilingualTTSMLX":
        """
        Load model from local checkpoint directory.

        The defaults load the 1.2.0 model: fp32 T3 and the PyTorch S3Gen.

        Args:
            ckpt_dir: Path to checkpoint directory
            t3_config: Optional T3 configuration
            use_default_speaker: Whether to load default speaker
            device: PyTorch device for S3Gen
            vocoder: ``"pt"`` (default): PyTorch S3Gen on ``device``, 5 steps.
                ``"mlx"``: the same ``s3gen.pt`` weights on MLX, 5 steps.
                ``"meanflow"``: Turbo's meanflow S3Gen on MLX, 2 steps
                (fastest; speaker similarity about 0.01-0.02 lower). With an
                MLX vocoder the PyTorch S3Gen keeps only its conditioning
                encoders (its flow and HiFiGAN are dropped).
            meanflow_path: ``s3gen_meanflow.safetensors``, or a directory that
                holds it (the local ``ResembleAI/chatterbox-turbo``
                snapshot). Only for ``vocoder="meanflow"``; ``None`` looks in
                ``ckpt_dir``. Nothing is downloaded (see
                :func:`resolve_meanflow_path`).
            t3_dtype: ``"float32"`` (default) or ``"float16"`` T3 weights
                (``"fp32"`` / ``"fp16"`` also accepted). fp16 roughly halves
                the T3 time per token on the fast decode loop.
            quantize_bits: ``None`` (default), 8 or 4: weight-only quantization
                of T3 (group size 64). Off by default: q8 is not yet
                validated on every language.
            mlx_cache_limit_gb: See :meth:`__init__`.

        Returns:
            ChatterboxMultilingualTTSMLX instance
        """
        ckpt_dir = Path(ckpt_dir)
        _check_vocoder(vocoder)
        meanflow_file = (
            resolve_meanflow_path(meanflow_path, ckpt_dir)
            if vocoder == "meanflow"
            else None
        )
        if t3_dtype not in T3_DTYPES:
            raise ValueError(
                f"t3_dtype must be one of {list(T3_DTYPES)}, got {t3_dtype!r}"
            )
        if quantize_bits not in (None, 4, 8):
            raise ValueError("quantize_bits must be None, 4 or 8")

        # Check MPS availability
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = "cpu"

        # Always load to CPU first for non-CUDA devices
        map_location = torch.device("cpu") if device in ["cpu", "mps"] else None

        # Load voice encoder (PyTorch)
        logger.info("Loading voice encoder...")
        ve = VoiceEncoder()
        ve.load_state_dict(
            torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=map_location)
        )
        ve.to(device).eval()

        # Initialize T3 MLX model with multilingual config
        logger.info("Initializing T3 MLX model (multilingual)...")
        if t3_config is None:
            t3_config = T3Config.multilingual()

        t3 = T3MLX(hp=t3_config)

        # Load T3 multilingual weights and convert to MLX
        logger.info("Loading T3 multilingual weights into MLX...")
        t3_ckpt = ckpt_dir / "t3_mtl23ls_v2.safetensors"
        pt_state = load_safetensors(t3_ckpt)
        if "model" in pt_state.keys():
            pt_state = pt_state["model"][0]

        # Convert PyTorch weights to MLX format
        mlx_state = {}
        for k, v in pt_state.items():
            mlx_state[k] = mx.array(v.cpu().numpy())

        # Load weights into T3MLX with strict=False to ignore unused embed_tokens
        t3.load_weights(list(mlx_state.items()), strict=False)
        t3.eval()  # Set to eval mode to disable dropout
        logger.info("✓ T3 multilingual weights loaded successfully")
        if T3_DTYPES[t3_dtype] != "float32" or quantize_bits is not None:
            logger.info(
                f"T3 precision: dtype={t3_dtype}, quantize_bits={quantize_bits}"
            )
            del mlx_state, pt_state
            apply_t3_precision(t3, t3_dtype, quantize_bits)

        # Load S3Gen (PyTorch/MPS) - hybrid approach
        logger.info(f"Loading S3Gen vocoder (PyTorch on {device})...")
        from .models.s3gen import S3Gen

        s3gen = S3Gen()
        s3gen.load_state_dict(
            torch.load(
                ckpt_dir / "s3gen.pt", weights_only=True, map_location=map_location
            )
        )
        s3gen_mlx = None
        if vocoder != "pt":
            # Only embed_ref (mel extractor, speaker encoder, S3 tokenizer)
            # runs in torch; its flow and HiFiGAN would be a second vocoder.
            del s3gen.flow, s3gen.mel2wav
            weights = meanflow_file if vocoder == "meanflow" else ckpt_dir / "s3gen.pt"
            logger.info(f"Loading {vocoder} S3Gen vocoder (MLX) from {weights}...")
            s3gen_mlx = load_mlx_vocoder(weights, meanflow=vocoder == "meanflow")
        s3gen.to(device).eval()
        logger.info("✓ S3Gen loaded successfully")

        # Load multilingual tokenizer
        logger.info("Loading multilingual tokenizer...")
        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))

        # Load default conditioning if requested
        conds = None
        if use_default_speaker:
            logger.info("Loading default speaker conditioning...")
            cond_path = ckpt_dir / "conds.pt"
            if cond_path.exists():
                conds = Conditionals.load(cond_path, map_location=map_location).to(
                    device
                )
                logger.info("✓ Default speaker conditioning loaded")

        logger.info("✅ ChatterboxMultilingualTTSMLX loaded successfully!")
        logger.info("   T3: MLX (Apple Silicon optimized)")
        if vocoder == "pt":
            logger.info(f"   S3Gen: PyTorch on {device}")
        else:
            logger.info(
                f"   S3Gen: MLX ({vocoder}); conditioning in PyTorch on {device}"
            )
        logger.info(f"   Supported languages: {len(SUPPORTED_LANGUAGES)}")

        return cls(
            t3=t3,
            s3gen=s3gen,
            ve=ve,
            tokenizer=tokenizer,
            device=device,
            conds=conds,
            vocoder=vocoder,
            s3gen_mlx=s3gen_mlx,
            mlx_cache_limit_gb=mlx_cache_limit_gb,
        )

    def _check_no_stream(self) -> None:
        # Another thread waits on the lock; this one re-enters it. A stream
        # suspended mid-decode shares T3's state and the conditioning.
        if self._stream_active:
            raise RuntimeError(
                "A generate_stream() generator is open on this instance; "
                "exhaust or close() it first."
            )

    def _mlx_ref_dict(self) -> dict:
        """``self.conds.gen`` in the MLX vocoder's layout (cached per conds)."""
        gen = self.conds.gen
        if self._mlx_ref is None or self._mlx_ref_src is not gen:
            with TORCH_LOCK:
                ref = ref_dict_to_mlx(gen)
            materialize_mlx_state(ref)
            self._mlx_ref_src, self._mlx_ref = gen, ref
        return self._mlx_ref

    def _vocoder_host(self) -> MLXVocoderHost:
        return MLXVocoderHost(
            self.s3gen_mlx, self._mlx_ref_dict(), self.mlx_cache_limit
        )

    def _cfm_steps(self, n_cfm_timesteps: Optional[int]) -> Optional[int]:
        if n_cfm_timesteps is not None:
            return int(n_cfm_timesteps)
        return DEFAULT_CFM_STEPS[self.vocoder]

    def _vocode_mlx(
        self, speech_tokens: np.ndarray, seed: Optional[int], n_cfm_timesteps
    ) -> np.ndarray:
        """Vocode one chunk's speech ids (``< 6561``) with the MLX vocoder.

        ``"meanflow"`` appends 3 silence tokens and runs S3Gen in final mode,
        as Turbo. ``"mlx"`` runs non-final with no silence tokens, as the
        PyTorch path (its lookahead trim drops the last 3 tokens' mel).
        """
        tokens = np.asarray(speech_tokens).reshape(-1).astype(np.int64)
        steps = self._cfm_steps(n_cfm_timesteps)
        host = self._vocoder_host()
        if self.vocoder == "meanflow":
            tokens = np.concatenate([tokens, np.full(3, S3GEN_SIL, dtype=np.int64)])
            return host._vocode(tokens, steps, seed, finalize=True)
        return host._vocode(tokens, steps, seed, finalize=False)

    @_exclusive
    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5):
        """
        Prepare conditioning from a reference audio file.

        Args:
            wav_fpath: Path to reference audio file
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
        """
        # PyTorch/MPS is not thread-safe; see TORCH_LOCK.
        with TORCH_LOCK:
            # Load reference wav
            s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
            ref_16k_wav = librosa.resample(
                s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR
            )

            s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
            s3gen_ref_dict = self.s3gen.embed_ref(
                s3gen_ref_wav, S3GEN_SR, device=self.device
            )

            # Speech cond prompt tokens
            t3_cond_prompt_tokens = None
            if plen := self.t3.hp.speech_cond_prompt_len:
                s3_tokzr = self.s3gen.tokenizer
                # Limit audio length more aggressively for memory safety
                safe_audio_len = min(len(ref_16k_wav), 6 * S3_SR)
                limited_audio = ref_16k_wav[:safe_audio_len]

                # Memory cleanup before tokenization
                clear_device_memory()

                # Use smaller max_len to be extra safe
                safe_max_len = min(plen, 150)
                t3_cond_prompt_tokens, _ = s3_tokzr.forward(
                    [limited_audio], max_len=safe_max_len
                )
                t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(
                    self.device
                )

                # More memory cleanup after tokenization
                clear_device_memory()

            # Voice-encoder speaker embedding
            ve_embed = torch.from_numpy(
                self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR)
            )
            ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

            t3_cond = T3Cond(
                speaker_emb=ve_embed,
                cond_prompt_speech_tokens=t3_cond_prompt_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

            self.conds = Conditionals(t3_cond, s3gen_ref_dict)
            materialize_mlx_state(self.conds)

    @_exclusive
    def generate(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
        seed: Optional[int] = None,
        n_cfm_timesteps: Optional[int] = None,
        watermark: bool = False,
    ) -> torch.Tensor:
        """
        Generate speech from text using hybrid MLX/PyTorch pipeline.

        By default, splits text into sentences for optimal generation quality.

        Args:
            text: Input text to synthesize
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Optional path to reference audio for voice cloning
            exaggeration: Emotion exaggeration factor (0.0 to 1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            max_new_tokens: Maximum tokens to generate (auto-estimated if None)
            show_progress: Whether to show token-level progress bar (default True)
            seed: Random seed for deterministic generation. If None (default),
                  uses non-deterministic generation. Set to an integer (e.g., 42)
                  for reproducible results.
            n_cfm_timesteps: S3Gen CFM steps; None (default) uses the
                  vocoder's default (5, or 2 for ``"meanflow"``).
            watermark: Apply the Perth watermark once, to the final waveform
                  (as Turbo). Default False: upstream does not apply it.

        Returns:
            Generated audio waveform as torch tensor

        :attr:`last_timings` is set: ``wall_s`` (generation),
        ``watermark_s`` and ``n_chunks``.
        """
        import time as _time

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare conditioning if audio prompt provided
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        self._set_exaggeration(exaggeration)

        # Normalize text
        text = punc_norm(text)

        # Split text into sentences for optimal generation
        lang = language_id.lower() if language_id else "en"
        sentences = self._plan_sentences(text, lang)

        total_words = count_words(text)
        num_chunks = len(sentences)
        lang_name = SUPPORTED_LANGUAGES.get(lang, language_id)

        # Print generation plan
        print_generation_plan(
            total_words,
            sentences,
            "per-sentence",
            prefix=f"[MLX Multilingual - {lang_name}] ",
        )

        # Generate audio for each sentence
        if len(sentences) == 1:
            # Single sentence - generate directly
            print_chunk_generating(0, 1, sentences[0])
            gen_start = _time.time()

            # Ensure deterministic RNG even when user doesn't provide seed
            # This prevents chip-specific artifacts (M1 vs M4) from MLX global RNG state
            import time as _time_module

            internal_seed = (
                seed if seed is not None else int(_time_module.time() * 1000) % (2**31)
            )

            result = self._generate_single(
                sentences[0],
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,  # Use caller's preference
                seed=internal_seed,
                n_cfm_timesteps=n_cfm_timesteps,
            )

            gen_time = _time.time() - gen_start
            audio_duration = (
                result.shape[-1] / self.sr
                if hasattr(result, "shape")
                else len(result) / self.sr
            )
            print_chunk_completed(0, 1, gen_time, audio_duration)
            print_generation_complete(
                gen_time, audio_duration, 1, prefix="[MLX Multilingual] "
            )

            # Store generation time for external use (e.g., benchmarking)
            self.last_generation_time = gen_time

            clear_device_memory()
            return self._finish_generate(result, watermark, gen_time, 1)

        # Multiple sentences - generate each and crossfade
        audio_chunks = []
        total_start = _time.time()

        # Ensure deterministic RNG for all chunks, even when user doesn't provide seed
        # This prevents chip-specific artifacts (M1 vs M4) from accumulated RNG state
        # If no seed provided, generate one based on current time for session consistency
        import time as _time_module

        internal_seed = (
            seed if seed is not None else int(_time_module.time() * 1000) % (2**31)
        )

        for i, sentence in enumerate(sentences):
            print_chunk_generating(i, num_chunks, sentence)
            chunk_start = _time.time()

            # Increment seed for each chunk to avoid repetition
            # Always use a seed (either user-provided or internally generated)
            chunk_seed = internal_seed + i

            chunk_audio = self._generate_single(
                sentence,
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=False,  # Suppress token progress, observability handles chunk progress
                seed=chunk_seed,
                n_cfm_timesteps=n_cfm_timesteps,
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            audio_chunks.append(chunk_audio)

        total_time = _time.time() - total_start

        # Crossfade chunks together
        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, 0.01)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[MLX Multilingual] "
        )

        # Store generation time for external use (e.g., benchmarking)
        self.last_generation_time = total_time

        clear_device_memory()

        return self._finish_generate(
            (
                torch.from_numpy(result_np).unsqueeze(0)
                if isinstance(result_np, np.ndarray)
                else result.unsqueeze(0)
            ),
            watermark,
            total_time,
            num_chunks,
        )

    def _finish_generate(
        self, wav: torch.Tensor, watermark: bool, wall_s: float, n_chunks: int
    ) -> torch.Tensor:
        """Watermark ``generate``'s ``(1, N)`` output if asked; set :attr:`last_timings`."""
        watermark_s = 0.0
        if watermark and wav.shape[-1]:
            t0 = time.perf_counter()
            marked = self._watermark(wav.detach().cpu().numpy().reshape(-1))
            wav = torch.from_numpy(marked).unsqueeze(0).to(wav.dtype)
            watermark_s = time.perf_counter() - t0
        self.last_timings = {
            "wall_s": wall_s,
            "watermark_s": watermark_s,
            "n_chunks": n_chunks,
        }
        return wav

    def _set_exaggeration(self, exaggeration: float) -> None:
        """Rebuild ``conds.t3`` when ``exaggeration`` differs from its emotion value."""
        with TORCH_LOCK:
            if float(exaggeration) != float(self.conds.t3.emotion_adv[0, 0, 0].item()):
                _cond = self.conds.t3
                self.conds.t3 = T3Cond(
                    speaker_emb=_cond.speaker_emb,
                    cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                    emotion_adv=exaggeration * torch.ones(1, 1, 1),
                ).to(device=self.device)

    @staticmethod
    def _plan_sentences(text: str, lang: str) -> List[str]:
        """generate()'s chunk plan for normalized ``text``: sentences, short ones merged."""
        if SPACY_AVAILABLE:
            sentences = split_into_sentences(text, lang=lang)
        else:
            sentences = [text]

        # Clean up each sentence (remove orphaned quotes from splitting)
        sentences = [punc_norm(s, debug=False) for s in sentences]
        # Filter out empty sentences
        sentences = [s for s in sentences if s and s.strip()]

        # Merge short sentences so no chunk is small enough to hallucinate.
        # Runs after punc_norm, so quotes are already stripped and the
        # merge cannot strand one mid-text.
        return merge_short_sentences(sentences, MIN_CHUNK_WORDS)

    def generate_stream(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        *,
        n_cfm_timesteps: Optional[int] = None,
        overlap_duration: float = 0.01,
        first_chunk_tokens: int = STREAM_FIRST_CHUNK_TOKENS,
        chunk_tokens: int = MTL_STREAM_CHUNK_TOKENS,
        context_tokens: int = STREAM_CONTEXT_TOKENS,
        lookahead_tokens: int = STREAM_LOOKAHEAD_TOKENS,
        watermark: bool = False,
    ) -> Iterator[np.ndarray]:
        """Synthesize ``text`` and yield 1-D float32 pieces at :attr:`sr` as they are ready.

        The pieces concatenate to one waveform; play them in order. The
        arguments shared with :meth:`generate` mean the same, and the text is
        cut into the same sentence chunks. Chunk ``i`` uses ``seed + i``.
        Chunks decode one after another and join with the same crossfade as
        :meth:`generate` (``overlap_duration``, default 10 ms).

        * With an MLX vocoder (``"meanflow"``, recommended, or ``"mlx"``)
          every chunk is vocoded on growing token windows while T3 still
          decodes it, as Turbo's ``generate_stream``: the first window after
          ``first_chunk_tokens`` tokens (plus ``lookahead_tokens``), the next
          after ``chunk_tokens`` more (default 40; Turbo uses 50), then
          windows double up to 200 tokens. Each window re-vocodes
          ``context_tokens`` tokens on its left (see
          :class:`~chatterbox.streaming_mlx._WindowVocoder`). A chunk ends
          with 3 silence tokens and a final-mode S3Gen call (as Turbo).
          A chunk that ends before its first window is vocoded once.
        * ``first_chunk_tokens=0``, or ``vocoder="pt"``: each chunk is
          yielded whole when it is done; the pieces then equal
          :meth:`generate` with the same seed.

        With ``"meanflow"`` a windowed chunk matches a whole-chunk vocode of
        the same tokens closely but not bit for bit (window joins). With
        ``"mlx"`` (5 steps) the windows are slower than real time on typical
        hardware and playback can gap; prefer ``"meanflow"`` for streaming.

        Time to first audio needs a T3 that yields tokens while it decodes
        (``T3MLX.generate_stream``). Without one, each chunk's tokens are
        decoded first and then fed to the windows (same audio, later).

        ``watermark`` (default False, as :meth:`generate`) applies the Perth
        watermark to each piece, with the end of the previous unwatermarked
        audio as left context (as Turbo's ``generate_stream``). The pieces
        keep their count and lengths.

        The generator holds the instance lock until it is exhausted or
        closed: other threads wait, and calling :meth:`generate`,
        :meth:`generate_long`, :meth:`generate_stream` or
        :meth:`prepare_conditionals` of this instance from the loop raises
        ``RuntimeError``.

        :attr:`last_timings` is set when the generator ends: ``t3_s`` and
        ``s3gen_s`` (windowed chunks), ``wall_s``, ``first_audio_s`` (call to
        first piece), ``watermark_s``, ``n_speech_tokens`` (windowed chunks),
        ``n_chunks``, ``n_pieces``, ``n_windows`` (S3Gen window calls) and
        ``vocoder``.
        """
        with self._lock:
            self._check_no_stream()
            first_chunk_tokens = int(first_chunk_tokens)
            if first_chunk_tokens and first_chunk_tokens < 2:
                raise ValueError("first_chunk_tokens must be 0 or >= 2")
            if lookahead_tokens < S3GEN_PRE_LOOKAHEAD:
                raise ValueError(f"lookahead_tokens must be >= {S3GEN_PRE_LOOKAHEAD}")
            if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
                supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
                raise ValueError(
                    f"Unsupported language_id '{language_id}'. "
                    f"Supported languages: {supported_langs}"
                )
            t_start = time.perf_counter()
            if audio_prompt_path:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            elif self.conds is None:
                raise ValueError(
                    "Please `prepare_conditionals` first or specify `audio_prompt_path`"
                )
            if not text or not text.strip():
                self.last_timings = {"n_chunks": 0}
                return
            self._set_exaggeration(exaggeration)

            lang = language_id.lower() if language_id else "en"
            sentences = self._plan_sentences(punc_norm(text), lang)
            internal_seed = (
                seed if seed is not None else int(time.time() * 1000) % (2**31)
            )
            n = len(sentences)
            joiner = _ChunkJoiner(n, self.sr, overlap_duration)
            stream_wm = StreamWatermark(self._watermark, self.sr)
            t = {
                "t3_s": 0.0,
                "s3gen_s": 0.0,
                "wall_s": 0.0,
                "first_audio_s": None,
                "watermark_s": 0.0,
                "n_speech_tokens": 0,
                "n_chunks": n,
                "n_pieces": 0,
                "n_windows": 0,
                "vocoder": self.vocoder,
            }

            def out(wav: np.ndarray, last: bool) -> np.ndarray:
                y = joiner.push(wav)
                if last:
                    y = np.concatenate([y, joiner.finish()])
                if watermark and len(y):
                    t0 = time.perf_counter()
                    # Perth sees the previous (unwatermarked) audio as context.
                    y = stream_wm(y)
                    t["watermark_s"] += time.perf_counter() - t0
                if len(y):
                    t["n_pieces"] += 1
                    if t["first_audio_s"] is None:
                        t["first_audio_s"] = time.perf_counter() - t_start
                return y

            sample = dict(
                cfg_weight=cfg_weight,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            windowed = bool(first_chunk_tokens) and self.vocoder != "pt"
            self._stream_active += 1
            try:
                for i, sentence in enumerate(sentences):
                    chunk_seed = internal_seed + i
                    last = i == n - 1
                    joiner.begin_chunk()
                    if windowed:
                        yield from self._stream_chunk(
                            sentence,
                            language_id,
                            chunk_seed,
                            max_new_tokens,
                            sample,
                            n_cfm_timesteps,
                            t,
                            out,
                            last=last,
                            first_chunk_tokens=first_chunk_tokens,
                            chunk_tokens=max(1, int(chunk_tokens)),
                            context_tokens=max(0, int(context_tokens)),
                            lookahead_tokens=int(lookahead_tokens),
                        )
                        continue
                    wav = self._generate_single(
                        sentence,
                        language_id=language_id,
                        max_new_tokens=max_new_tokens,
                        show_progress=False,
                        seed=chunk_seed,
                        n_cfm_timesteps=n_cfm_timesteps,
                        **sample,
                    )
                    y = out(wav.numpy().reshape(-1), last=last)
                    if len(y):
                        yield y
            finally:
                self._stream_active -= 1
                t["wall_s"] = time.perf_counter() - t_start
                self.last_timings = t
                self.last_generation_time = t["wall_s"]
        if self.vocoder == "pt":
            clear_device_memory()

    def _stream_chunk(
        self,
        text: str,
        language_id: str,
        seed: int,
        max_new_tokens: Optional[int],
        sample: dict,
        n_cfm_timesteps: Optional[int],
        t: dict,
        out,
        *,
        last: bool,
        first_chunk_tokens: int,
        chunk_tokens: int,
        context_tokens: int,
        lookahead_tokens: int,
    ) -> Iterator[np.ndarray]:
        """Token-window streaming of one chunk (see :meth:`generate_stream`)."""
        text, text_tokens_mx, t3_cond_mx, max_new_tokens = self._prepare_t3_inputs(
            text, language_id, sample["cfg_weight"], max_new_tokens, seed
        )
        voc = _WindowVocoder(
            self._vocoder_host(),
            seed=seed,
            n_cfm_timesteps=self._cfm_steps(n_cfm_timesteps),
            max_tokens=max_new_tokens + 1,
            first_chunk_tokens=first_chunk_tokens,
            chunk_tokens=chunk_tokens,
            context_tokens=context_tokens,
            lookahead_tokens=lookahead_tokens,
        )
        blocks = self._t3_token_blocks(
            t3_cond=t3_cond_mx,
            text_tokens=text_tokens_mx,
            max_new_tokens=max_new_tokens,
            seed=seed,
            **sample,
        )
        all_ids = []
        try:
            t0 = time.perf_counter()
            for ids in blocks:
                ids = np.asarray(ids).reshape(-1).astype(np.int64)
                all_ids.append(ids)
                t["t3_s"] += time.perf_counter() - t0
                t1 = time.perf_counter()
                wavs = list(voc.feed(ids))
                t["s3gen_s"] += time.perf_counter() - t1
                for wav in wavs:
                    y = out(wav, last=False)
                    if len(y):
                        yield y
                t0 = time.perf_counter()
            t["t3_s"] += time.perf_counter() - t0
        finally:
            # Close T3's generator here (on this thread), not at garbage collection.
            close = getattr(blocks, "close", None)
            if close is not None:
                close()

        ids = np.concatenate(all_ids) if all_ids else np.zeros((0,), np.int64)
        with TORCH_LOCK:
            # The same EOS warning and debug output as generate().
            n_speech = len(self._postprocess_speech_tokens(ids, text))
        t["n_speech_tokens"] += n_speech
        t1 = time.perf_counter()
        wav = voc.finish()
        t["s3gen_s"] += time.perf_counter() - t1
        t["n_windows"] += voc.n_windows
        y = out(wav, last=last)
        if len(y):
            yield y

    def _t3_token_blocks(
        self, *, block_tokens: int = STREAM_BLOCK_TOKENS, seed=None, **t3_kwargs
    ) -> Iterator[np.ndarray]:
        """T3 speech ids of one chunk, in blocks of ``block_tokens``, while T3 decodes.

        Uses ``T3MLX.generate_tokens`` (the fast decode loop). It samples from
        MLX's global RNG, which :meth:`_prepare_t3_inputs` seeded. A T3
        without it decodes the whole chunk with ``inference`` first; its ids
        are then cut into blocks (same ids, no early audio).

        Between blocks the window vocoder re-seeds MLX's RNG for its own
        noise. So with windows, T3 samples after the first window differ
        from :meth:`generate` with the same seed; they stay deterministic
        for a given seed.
        """
        block_tokens = max(1, int(block_tokens))
        gen_tokens = getattr(self.t3, "generate_tokens", None)
        if not callable(gen_tokens):
            yield from self._t3_blocks_after_decode(block_tokens, seed, t3_kwargs)
            return
        tokens = gen_tokens(**t3_kwargs, show_progress=False)
        try:
            block = []
            for tok in tokens:
                block.append(int(tok))
                if len(block) >= block_tokens:
                    yield np.asarray(block, dtype=np.int64)
                    block = []
            if block:
                yield np.asarray(block, dtype=np.int64)
        finally:
            close = getattr(tokens, "close", None)
            if close is not None:
                close()

    def _t3_blocks_after_decode(
        self, block_tokens: int, seed, t3_kwargs: dict
    ) -> Iterator[np.ndarray]:
        with torch.inference_mode():
            toks = self.t3.inference(**t3_kwargs, seed=seed, show_progress=False)
        ids = np.array(toks[0] if len(toks.shape) > 1 else toks)
        ids = ids.reshape(-1).astype(np.int64)
        for i in range(0, len(ids), block_tokens):
            yield ids[i : i + block_tokens]

    def _generate_single(
        self,
        text: str,
        language_id: str,
        cfg_weight: float = 0.5,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        show_progress: bool = True,
        seed: Optional[int] = None,
        n_cfm_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).

        Args:
            text: Input text (single sentence/chunk)
            language_id: Language code
            cfg_weight: Classifier-free guidance weight
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            repetition_penalty: Repetition penalty factor
            min_p: Minimum probability threshold
            top_p: Nucleus sampling threshold
            show_progress: Whether to show token-level progress bar
            seed: Random seed for deterministic generation
            n_cfm_timesteps: S3Gen CFM steps (None: the vocoder's default)

        Returns:
            Generated audio waveform as torch tensor
        """
        text, text_tokens_mx, t3_cond_mx, max_new_tokens = self._prepare_t3_inputs(
            text, language_id, cfg_weight, max_new_tokens, seed
        )

        # Generate speech tokens with T3 MLX
        with torch.inference_mode():
            speech_tokens_mx = self.t3.inference(
                t3_cond=t3_cond_mx,
                text_tokens=text_tokens_mx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=show_progress,
                seed=seed,
            )

        # Extract conditional batch (first one)
        speech_tokens = (
            speech_tokens_mx[0] if len(speech_tokens_mx.shape) > 1 else speech_tokens_mx
        )

        # Convert back to PyTorch for S3Gen
        speech_tokens_np = np.array(speech_tokens).astype(np.int64)
        with TORCH_LOCK:
            speech_tokens_pt = self._postprocess_speech_tokens(speech_tokens_np, text)
            if self.vocoder == "pt":
                wav = self._vocode_pt(speech_tokens_pt, seed, n_cfm_timesteps)
            else:
                speech_ids = speech_tokens_pt.numpy()
        if self.vocoder != "pt":
            # MLX vocoder: outside TORCH_LOCK (no torch work).
            wav = self._vocode_mlx(speech_ids, seed, n_cfm_timesteps)

        if os.getenv("CHATTERBOX_DEBUG"):
            wav_duration = len(wav) / self.sr
            print(
                f"[DEBUG] S3Gen output: {len(wav)} samples = {wav_duration:.2f}s @ {self.sr}Hz"
            )
            print(f"[DEBUG] Text preview: {text[:60]}...")

        return torch.from_numpy(wav).unsqueeze(0)

    def _prepare_t3_inputs(
        self,
        text: str,
        language_id: str,
        cfg_weight: float,
        max_new_tokens: Optional[int],
        seed: Optional[int],
    ):
        """Seed the RNGs, normalize and tokenize ``text``, convert the T3 conditioning.

        Returns ``(text, text_tokens_mx, t3_cond_mx, max_new_tokens)``.
        """
        # Seed all RNGs if seed is provided (for both T3 MLX and S3Gen PyTorch)
        # Torch work only; T3 runs on MLX, outside the lock.
        with TORCH_LOCK:
            if seed is not None:
                # Set all seeds
                mx.random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    # MPS backend seed setting
                    torch.mps.manual_seed(seed)

                # Set deterministic behavior for PyTorch operations
                torch.use_deterministic_algorithms(
                    False
                )  # Some ops don't support deterministic
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

                # Optional debug logging
                if os.getenv("CHATTERBOX_DEBUG"):
                    print(f"[SEED] Setting seed={seed}")

            # Normalize and tokenize
            text = punc_norm(text)

            # Estimate max_new_tokens if not provided
            if max_new_tokens is None:
                max_new_tokens = estimate_max_tokens(text, self.t3.hp.max_speech_tokens)

            text_tokens = self.tokenizer.text_to_tokens(
                text, language_id=language_id.lower() if language_id else None
            ).to(self.device)

            # Add start/end tokens for CFG
            if cfg_weight > 0.0:
                text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

            sot = self.t3.hp.start_text_token
            eot = self.t3.hp.stop_text_token
            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)

            # Convert to MLX for T3 inference
            text_tokens_mx = mx.array(text_tokens.cpu().numpy())

            # Convert T3 conditioning to MLX
            t3_cond_mx = T3CondMLX(
                speaker_emb=(
                    mx.array(self.conds.t3.speaker_emb.cpu().numpy())
                    if self.conds.t3.speaker_emb is not None
                    else None
                ),
                cond_prompt_speech_tokens=(
                    mx.array(self.conds.t3.cond_prompt_speech_tokens.cpu().numpy())
                    if self.conds.t3.cond_prompt_speech_tokens is not None
                    else None
                ),
                emotion_adv=float(self.conds.t3.emotion_adv[0, 0, 0].item()),
            )
        return text, text_tokens_mx, t3_cond_mx, max_new_tokens

    def _postprocess_speech_tokens(self, speech_tokens_np: np.ndarray, text: str):
        """T3 ids -> the speech ids S3Gen vocodes (CPU tensor). Call under TORCH_LOCK."""
        speech_tokens_pt = torch.from_numpy(speech_tokens_np)

        # DEBUG: Log token counts before filtering
        # Missing EOS means generation ran to the token cap instead of
        # deciding it was finished -- the signature of a hallucinated
        # tail. Warn unconditionally: the absence of the existing
        # "EOS detected" info log is not something a bulk run can grep
        # for, so this failure was previously silent in production.
        if not (speech_tokens_pt == 6562).any().item():
            logger.warning(
                "no EOS token: generation used all %d tokens for %d-word text "
                "(%r) -- output may contain hallucinated audio",
                len(speech_tokens_pt),
                count_words(text),
                text[:60],
            )

        if os.getenv("CHATTERBOX_DEBUG"):
            print(f"[DEBUG] T3 generated {len(speech_tokens_pt)} tokens")
            print(
                f"[DEBUG] Token stats: min={speech_tokens_pt.min().item()}, max={speech_tokens_pt.max().item()}"
            )
            # Check for SOS/EOS
            SOS, EOS = 6561, 6562
            has_sos = (speech_tokens_pt == SOS).any().item()
            has_eos = (speech_tokens_pt == EOS).any().item()
            if has_sos:
                sos_pos = (speech_tokens_pt == SOS).nonzero(as_tuple=True)[0][0].item()
                print(f"[DEBUG] SOS (6561) found at position {sos_pos}")
            if has_eos:
                eos_pos = (speech_tokens_pt == EOS).nonzero(as_tuple=True)[0][0].item()
                print(f"[DEBUG] EOS (6562) found at position {eos_pos}")
                print(
                    f"[DEBUG] Tokens after EOS: {len(speech_tokens_pt) - eos_pos - 1}"
                )
            else:
                print(
                    "[DEBUG] WARNING: No EOS token found! Generation may have hit max_new_tokens"
                )
            # Count special tokens
            num_special = (speech_tokens_pt >= 6561).sum().item()
            print(f"[DEBUG] Special tokens (>=6561): {num_special}")

        # Drop invalid tokens (SOS/EOS)
        speech_tokens_pt = drop_invalid_tokens(speech_tokens_pt)

        if os.getenv("CHATTERBOX_DEBUG"):
            print(f"[DEBUG] After drop_invalid_tokens: {len(speech_tokens_pt)} tokens")

        # Filter out tokens >= 6561 (special tokens)
        speech_tokens_pt = speech_tokens_pt[speech_tokens_pt < 6561]

        if os.getenv("CHATTERBOX_DEBUG"):
            print(
                f"[DEBUG] After filtering special tokens: {len(speech_tokens_pt)} tokens"
            )
            print(
                f"[DEBUG] Final token range: min={speech_tokens_pt.min().item()}, max={speech_tokens_pt.max().item()}"
            )

        return speech_tokens_pt

    def _vocode_pt(self, speech_tokens_pt, seed: Optional[int], n_cfm_timesteps):
        """The 1.2.0 vocoder: PyTorch S3Gen, non-final. Call under TORCH_LOCK."""
        speech_tokens_pt = speech_tokens_pt.to(self.device)

        # Re-seed PyTorch RNG right before S3Gen to ensure determinism
        # This is necessary because T3 generation may have consumed random numbers
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Generate waveform with S3Gen (PyTorch/MPS)
        kwargs = {}
        if n_cfm_timesteps is not None:
            kwargs["n_cfm_timesteps"] = int(n_cfm_timesteps)
        with torch.inference_mode():
            wav, sources = self.s3gen.inference(
                speech_tokens=speech_tokens_pt,
                ref_dict=self.conds.gen,
                finalize=False,  # Enable lookahead trimming to prevent extra audio generation
                **kwargs,
            )
            if sources is not None:
                del sources
            wav = wav.squeeze(0).detach().cpu().numpy()
        return wav

    def _split_text_intelligently(
        self, text: str, language_id: str, target_words_per_chunk: int = 50
    ) -> List[str]:
        """
        Split text at sentence/phrase boundaries for chunked generation.
        Uses shared utility with spacy support.
        """
        return split_text_intelligently(
            text,
            target_words_per_chunk,
            lang=language_id.lower() if language_id else "en",
        )

    def _crossfade_chunks(
        self, chunks: List[torch.Tensor], overlap_duration: float = 0.1
    ) -> np.ndarray:
        """
        Concatenate audio chunks with crossfading.
        Uses shared optimized utility.
        """
        result = crossfade_chunks(chunks, self.sr, overlap_duration)
        return result.numpy() if isinstance(result, torch.Tensor) else result

    @_exclusive
    def generate_long(
        self,
        text: str,
        language_id: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        min_p: float = 0.05,
        top_p: float = 1.0,
        overlap_duration: float = 0.05,
        max_new_tokens: Optional[int] = None,
        progress_callback=None,
        show_progress: bool = False,
        seed: Optional[int] = None,
        n_cfm_timesteps: Optional[int] = None,
        watermark: bool = False,
    ) -> torch.Tensor:
        """
        Generate long-form speech with adaptive chunking strategy.

        Automatically chooses the best chunking strategy based on text length:
        - Short texts (< 50 words): Individual sentence processing (MLX optimal)
        - Long texts (>= 50 words): Grouped sentence processing (reduces overhead)

        Args:
            text: Input text to synthesize (any length)
            language_id: Language code (e.g., 'en', 'es', 'ja', 'zh')
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: Voice exaggeration/expressiveness level (0.0-1.0)
            cfg_weight: Classifier-free guidance weight (0.0-1.0)
            temperature: Sampling temperature for token generation
            repetition_penalty: Penalty for repeating tokens
            min_p: Minimum probability threshold for sampling
            top_p: Nucleus sampling threshold
            overlap_duration: Duration in seconds of crossfade between chunks
            max_new_tokens: Maximum tokens to generate per chunk
            progress_callback: Optional callback function for progress monitoring
            seed: Random seed for deterministic generation
            n_cfm_timesteps: S3Gen CFM steps (None: the vocoder's default)
            watermark: Apply the Perth watermark once, to the final waveform.
                Default False, as :meth:`generate`.

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
        import time as _time

        # Validate language_id
        if language_id and language_id.lower() not in SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )

        # Prepare initial conditioning
        if audio_prompt_path:
            if progress_callback:
                progress_callback(
                    stage="preparing_conditionals", audio_path=audio_prompt_path
                )
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert (
                self.conds is not None
            ), "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        self._set_exaggeration(exaggeration)

        # Get adaptive chunks based on text length
        lang = language_id.lower() if language_id else "en"
        chunks_to_generate, chunking_strategy = get_adaptive_chunks(text, lang=lang)

        if not chunks_to_generate:
            chunks_to_generate = [text]

        total_words = count_words(text)
        num_chunks = len(chunks_to_generate)
        lang_name = SUPPORTED_LANGUAGES.get(lang, language_id)

        if progress_callback:
            progress_callback(
                stage="text_split",
                total_chunks=num_chunks,
                chunk_previews=[
                    (i + 1, count_words(chunk), chunk[:50])
                    for i, chunk in enumerate(chunks_to_generate)
                ],
            )

        # Print generation plan
        print_generation_plan(
            total_words,
            chunks_to_generate,
            chunking_strategy,
            prefix=f"[MLX Multilingual - {lang_name}] ",
            is_long_form=True,
        )

        # Generate each chunk with status updates
        audio_chunks = []
        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            chunk_words = count_words(chunk_text)

            if progress_callback:
                progress_callback(
                    stage="chunk_start",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    text_preview=chunk_text[:50],
                    word_count=chunk_words,
                )

            print_chunk_generating(i, num_chunks, chunk_text)
            chunk_start = _time.time()

            # Increment seed for each chunk to avoid repetition
            chunk_seed = seed + i if seed is not None else None

            # Generate chunk
            chunk_audio = self._generate_single(
                chunk_text,
                language_id=language_id,
                cfg_weight=cfg_weight,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                show_progress=False,  # Suppress token progress, observability handles chunk progress
                seed=chunk_seed,
                n_cfm_timesteps=n_cfm_timesteps,
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = (
                chunk_audio.shape[-1] / self.sr
                if hasattr(chunk_audio, "shape")
                else len(chunk_audio) / self.sr
            )
            print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            # MEMORY OPTIMIZATION: Move to CPU immediately
            if isinstance(chunk_audio, torch.Tensor):
                chunk_audio = chunk_audio.detach().cpu()
            audio_chunks.append(chunk_audio)

            # Aggressive memory cleanup after each chunk
            clear_device_memory()

            if progress_callback:
                progress_callback(
                    stage="chunk_complete",
                    chunk_index=i,
                    chunk_number=i + 1,
                    total_chunks=num_chunks,
                    audio_shape=(
                        chunk_audio.shape
                        if hasattr(chunk_audio, "shape")
                        else (len(chunk_audio),)
                    ),
                )

        total_time = _time.time() - total_start

        # Crossfade and concatenate all chunks
        if progress_callback:
            progress_callback(
                stage="crossfading",
                total_chunks=len(audio_chunks),
                overlap_duration=overlap_duration,
            )

        print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, overlap_duration)

        # Final summary
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        total_audio_duration = len(result_np) / self.sr
        print_generation_complete(
            total_time, total_audio_duration, num_chunks, prefix="[MLX Multilingual] "
        )

        # Store generation time for external use (e.g., benchmarking)
        self.last_generation_time = total_time

        wav = self._finish_generate(
            (
                torch.from_numpy(result_np).unsqueeze(0)
                if isinstance(result_np, np.ndarray)
                else result.unsqueeze(0)
            ),
            watermark,
            total_time,
            num_chunks,
        )

        if progress_callback:
            progress_callback(
                stage="complete",
                total_chunks=len(audio_chunks),
                final_audio_shape=(1, len(result_np)),
            )

        return wav
