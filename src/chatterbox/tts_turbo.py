import os
import math
import logging
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import torch
import perth
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .models.s3gen.const import S3GEN_SIL
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond
from .models.t3.modules.t3_config import T3Config
from .models.utils import clear_device_memory

# Shared generation utilities
from .generation_utils import (
    SPACY_AVAILABLE,
    split_into_sentences,
    get_adaptive_chunks,
    crossfade_chunks,
    print_generation_plan,
    print_chunk_generating,
    print_chunk_completed,
    print_generation_complete,
    print_crossfading,
)

try:
    import pyloudnorm as ln
    PYLOUDNORM_AVAILABLE = True
except ImportError:
    PYLOUDNORM_AVAILABLE = False

try:
    from transformers import GPT2Tokenizer
    GPT2_AVAILABLE = True
except ImportError:
    GPT2_AVAILABLE = False

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


def punc_norm(text: str) -> str:
    """
    Quick cleanup func for punctuation from LLMs or
    containing chars not seen often in the dataset.
    
    Relaxed version for Turbo model.
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (" - ", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        (""", "\""),
        (""", "\""),
        ("'", "'"),
        ("'", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTurboTTS:
    ENC_COND_LEN = 15 * S3_SR  # 15 seconds at 16kHz
    DEC_COND_LEN = 10 * S3GEN_SR  # 10 seconds at 24kHz

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTurboTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        # Turbo specific hp
        hp = T3Config.turbo()
        t3 = T3(hp=hp)
        t3_state = load_file(ckpt_dir / "t3_turbo_v1.safetensors")
        t3.load_state_dict(t3_state, strict=False)
        t3.to(device).eval()

        # Meanflow S3Gen for single-step decoding
        s3gen = S3Gen(flow_type="causal", use_meanflow=True)
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        # GPT2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(str(ckpt_dir))

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTurboTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        local_path = snapshot_download(
            repo_id=REPO_ID,
            token=os.getenv("HF_TOKEN") or True,
            # Optional: Filter to download only what you need
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.pt", "*.model"]
        )

        return cls.from_local(local_path, device)

    def norm_loudness(self, wav, sr, target_lufs=-27):
        if not PYLOUDNORM_AVAILABLE:
            return wav
        try:
            meter = ln.Meter(sr)
            loudness = meter.integrated_loudness(wav)
            gain_db = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)
            if math.isinf(gain_linear) or math.isnan(gain_linear):
                return wav
            return wav * gain_linear
        except Exception:
            return wav

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5, norm_loudness=True):
        ## Load and norm reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)
        s3gen_ref_wav = s3gen_ref_wav.astype(np.float32)  # Ensure float32 for MPS compatibility

        assert len(s3gen_ref_wav) / S3GEN_SR > 5.0, "Audio prompt must be longer than 5 seconds!"

        if norm_loudness:
            s3gen_ref_wav = self.norm_loudness(s3gen_ref_wav, S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
        ref_16k_wav = ref_16k_wav.astype(np.float32)  # Ensure float32 after resample

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        top_p=0.95,
        top_k=1000,
        min_p=0.0,
        repetition_penalty=1.2,
        norm_loudness=True,
    ):
        """
        Generate speech from text using Chatterbox-Turbo.
        
        Args:
            text: Input text to synthesize (supports paralinguistic tags like [laugh], [cough])
            audio_prompt_path: Path to reference audio for voice cloning (required on first call)
            exaggeration: Emotion exaggeration factor (0.0 to 1.0) - IGNORED in Turbo
            cfg_weight: Classifier-free guidance weight - IGNORED in Turbo
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            min_p: Minimum probability threshold - IGNORED in Turbo
            repetition_penalty: Penalty for repeating tokens
            norm_loudness: Whether to normalize reference audio loudness
        
        Returns:
            Generated audio waveform as torch tensor (24kHz)
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and will be ignored.")

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        silence = torch.tensor([S3GEN_SIL] * 5, device=self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def _generate_single(
        self,
        text: str,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 1000,
        repetition_penalty: float = 1.2,
        apply_watermark: bool = True,
    ) -> torch.Tensor:
        """
        Generate speech for a single sentence/chunk (internal method).
        
        Args:
            text: Input text (single sentence/chunk)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Repetition penalty factor
            apply_watermark: Whether to apply watermark
        
        Returns:
            Generated audio waveform as torch tensor
        """
        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        text_tokens = text_tokens.input_ids.to(self.device)

        speech_tokens = self.t3.inference_turbo(
            t3_cond=self.conds.t3,
            text_tokens=text_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Remove OOV tokens and add silence to end
        speech_tokens = speech_tokens[speech_tokens < 6561]
        silence = torch.tensor([S3GEN_SIL] * 5, device=self.device)
        speech_tokens = torch.cat([speech_tokens, silence])

        wav, _ = self.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=self.conds.gen,
        )
        wav = wav.squeeze(0).detach().cpu().numpy()
        
        if apply_watermark:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        
        return torch.from_numpy(wav).unsqueeze(0)

    def generate_long(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        top_p=0.95,
        top_k=1000,
        repetition_penalty=1.2,
        overlap_duration=0.05,
        language: str = "en",
        norm_loudness=True,
        show_progress: bool = True,
    ):
        """
        Generate long-form speech with adaptive chunking strategy.
        
        Automatically chooses the best chunking strategy based on text length.
        
        Args:
            text: Input text to synthesize (any length)
            audio_prompt_path: Path to reference audio file for voice cloning
            exaggeration: IGNORED in Turbo (kept for API compatibility)
            cfg_weight: IGNORED in Turbo (kept for API compatibility)
            temperature: Sampling temperature for token generation
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            repetition_penalty: Penalty for repeating tokens
            overlap_duration: Duration in seconds of crossfade between chunks
            language: Language code for sentence tokenization (e.g., "en", "de", "fr")
            norm_loudness: Whether to normalize reference audio loudness
            show_progress: Whether to show progress information

        Returns:
            torch.Tensor: Generated audio waveform with shape (1, num_samples)
        """
        import time as _time
        
        # Prepare initial conditioning
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration, norm_loudness=norm_loudness)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        if cfg_weight > 0.0 or exaggeration > 0.0:
            logger.warning("CFG and exaggeration are not supported by Turbo version and will be ignored.")

        # Get adaptive chunks based on text length
        chunks_to_generate, chunking_strategy = get_adaptive_chunks(text, lang=language)
        
        if not chunks_to_generate:
            chunks_to_generate = [text]
        
        total_words = len(text.split())
        num_chunks = len(chunks_to_generate)

        # Print generation plan
        if show_progress:
            print_generation_plan(total_words, chunks_to_generate, chunking_strategy, is_long_form=True, prefix="[Turbo] ")

        # Single chunk - generate directly
        if num_chunks == 1:
            if show_progress:
                print_chunk_generating(0, 1, chunks_to_generate[0])
            gen_start = _time.time()
            
            result = self._generate_single(
                chunks_to_generate[0],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                apply_watermark=True,
            )
            
            gen_time = _time.time() - gen_start
            audio_duration = result.shape[-1] / self.sr
            if show_progress:
                print_chunk_completed(0, 1, gen_time, audio_duration)
                print_generation_complete(gen_time, audio_duration, 1)
            
            return result

        # Multiple chunks - generate each and crossfade
        audio_chunks = []
        total_start = _time.time()

        for i, chunk_text in enumerate(chunks_to_generate):
            if show_progress:
                print_chunk_generating(i, num_chunks, chunk_text)
            chunk_start = _time.time()

            # Generate without watermark for intermediate chunks
            chunk_audio = self._generate_single(
                chunk_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                apply_watermark=False,
            )

            chunk_time = _time.time() - chunk_start
            chunk_duration = chunk_audio.shape[-1] / self.sr
            if show_progress:
                print_chunk_completed(i, num_chunks, chunk_time, chunk_duration)

            # Move to CPU immediately for memory efficiency
            if isinstance(chunk_audio, torch.Tensor):
                chunk_audio = chunk_audio.detach().cpu()
            audio_chunks.append(chunk_audio)
            
            # Aggressive memory cleanup after each chunk
            clear_device_memory()

        total_time = _time.time() - total_start

        # Crossfade and concatenate all chunks
        if show_progress:
            print_crossfading(num_chunks)
        result = crossfade_chunks(audio_chunks, self.sr, overlap_duration)

        # Apply watermark to final concatenated audio
        result_np = result.numpy() if isinstance(result, torch.Tensor) else result
        watermarked_result = self.watermarker.apply_watermark(result_np, sample_rate=self.sr)

        # Final summary
        total_audio_duration = len(watermarked_result) / self.sr
        if show_progress:
            print_generation_complete(total_time, total_audio_duration, num_chunks)

        return torch.from_numpy(watermarked_result).unsqueeze(0)
