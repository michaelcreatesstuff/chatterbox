# Copyright (c) 2025 Resemble AI
# MIT License
import logging
from typing import Union, Optional, List, Any, Tuple

logger = logging.getLogger(__name__)

from tqdm import tqdm
import torch
import torch.nn.functional as F
import psutil
import os
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig
from transformers.generation.logits_process import TopPLogitsWarper, RepetitionPenaltyLogitsProcessor, MinPLogitsWarper
from transformers.cache_utils import DynamicCache

from .modules.learned_pos_emb import LearnedPositionEmbeddings

from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..utils import AttrDict


logger = logging.getLogger(__name__)

# Debug logging - set CHATTERBOX_DEBUG=1 to enable verbose memory logging
DEBUG_LOGGING = os.environ.get("CHATTERBOX_DEBUG", "0") == "1"

# Use static KV cache for better MPS memory management
USE_STATIC_CACHE = os.environ.get("CHATTERBOX_STATIC_CACHE", "1") == "1"


class MPSOptimizedCache:
    """
    Custom KV cache optimized for MPS (Apple Silicon).
    
    The standard StaticCache uses indexed assignment which is very slow on MPS (~0.7ms/update).
    This implementation uses narrow().copy_() which is 43x faster (~0.016ms/update).
    
    Like StaticCache, this pre-allocates all memory upfront to prevent MPS driver memory
    from growing unbounded during generation.
    """
    
    def __init__(
        self,
        config: LlamaConfig,
        max_cache_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        batch_size: int = 2,  # Default to 2 for CFG (conditional + unconditional)
    ):
        self.max_cache_len = max_cache_len
        self.num_layers = config.num_hidden_layers
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        
        # Pre-allocate cache tensors for all layers
        # Shape: (batch_size, num_kv_heads, max_cache_len, head_dim)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        for _ in range(self.num_layers):
            self.key_cache.append(
                torch.zeros(self.batch_size, self.num_kv_heads, max_cache_len, self.head_dim, 
                           device=device, dtype=dtype)
            )
            self.value_cache.append(
                torch.zeros(self.batch_size, self.num_kv_heads, max_cache_len, self.head_dim,
                           device=device, dtype=dtype)
            )
        
        # Track the current sequence length
        self._seq_length = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new key/value states using MPS-optimized operations.
        
        Args:
            key_states: (batch, num_kv_heads, seq_len, head_dim)
            value_states: (batch, num_kv_heads, seq_len, head_dim)
            layer_idx: Which layer's cache to update
            cache_kwargs: Optional dict with 'cache_position' tensor
            
        Returns:
            Tuple of (keys, values) containing all cached states up to current position
        """
        seq_len = key_states.shape[2]
        
        # Get cache position (where to write)
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            cache_position = cache_kwargs["cache_position"]
            start_pos = cache_position[0].item()
        else:
            start_pos = self._seq_length
        
        # Safety check: prevent cache overflow with clear error message
        if start_pos + seq_len > self.max_cache_len:
            raise ValueError(
                f"Cache overflow: Requesting index {start_pos + seq_len} "
                f"but max_cache_len is {self.max_cache_len}. "
                f"Consider increasing max_new_tokens buffer or reducing generation length."
            )
        
        # Use narrow().copy_() for fast MPS updates (43x faster than indexed assignment)
        # narrow(dim, start, length) creates a view, copy_() writes to it
        self.key_cache[layer_idx].narrow(2, start_pos, seq_len).copy_(key_states)
        self.value_cache[layer_idx].narrow(2, start_pos, seq_len).copy_(value_states)
        
        # Update sequence length tracking (only on first layer to avoid redundant updates)
        if layer_idx == 0:
            self._seq_length = start_pos + seq_len

        # Return the full cache tensors (not narrowed views) to match StaticCache behavior
        # The attention mechanism will handle masking to use only the relevant portions
        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx]
        )
    
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get the current sequence length in the cache."""
        return self._seq_length
    
    def get_max_length(self) -> int:
        """Get the maximum cache length."""
        return self.max_cache_len
    
    def reset(self):
        """Reset the cache for a new generation."""
        self._seq_length = 0
        # Optionally zero out the tensors (not strictly necessary since we track seq_length)
        # for k, v in zip(self.key_cache, self.value_cache):
        #     k.zero_()
        #     v.zero_()
    
    def __len__(self) -> int:
        """Return number of layers (for compatibility checks)."""
        return self.num_layers
    
    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cache for a specific layer. Returns full tensors to match StaticCache."""
        return (
            self.key_cache[layer_idx],
            self.value_cache[layer_idx]
        )
    
    def __iter__(self):
        """Iterate over layer caches."""
        for layer_idx in range(self.num_layers):
            yield self[layer_idx]
    
    # ---- Methods required by HuggingFace transformers ----
    
    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> Tuple[int, int]:
        """
        Return (kv_length, kv_offset) for attention mask generation.

        Returns the max cache length (not current length) to match StaticCache behavior.
        The actual KV states returned by update() will be narrowed to the current length.
        """
        return self.max_cache_len, 0
    
    @property
    def is_sliding(self) -> list[bool]:
        """Whether the layers of the cache use sliding window attention."""
        return [False] * self.num_layers
    
    @property
    def is_compileable(self) -> bool:
        """Whether this cache is compatible with torch.compile."""
        return False
    
    def get_max_cache_shape(self) -> Tuple[int, int, int, int]:
        """Return the shape of the pre-allocated cache tensors."""
        return (self.batch_size, self.num_kv_heads, self.max_cache_len, self.head_dim)


def get_memory_info():
    """Get comprehensive memory info matching Activity Monitor on Mac."""
    vm = psutil.virtual_memory()
    
    info = {
        'sys_used_gb': vm.used / 1024**3,
        'sys_available_gb': vm.available / 1024**3,
        'sys_percent': vm.percent,
    }
    
    # macOS specific: get wired and app memory via vm_stat
    try:
        import subprocess
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        page_size = 16384  # Default for Apple Silicon
        
        stats = {}
        for line in lines:
            if ':' in line:
                key, val = line.split(':')
                try:
                    stats[key.strip()] = int(val.strip().rstrip('.'))
                except:
                    pass
        
        if 'Pages wired down' in stats:
            info['wired_gb'] = (stats['Pages wired down'] * page_size) / 1024**3
        if 'Pages occupied by compressor' in stats:
            info['compressed_gb'] = (stats['Pages occupied by compressor'] * page_size) / 1024**3
        if 'Pages active' in stats:
            info['active_gb'] = (stats['Pages active'] * page_size) / 1024**3
    except:
        pass
    
    # MPS memory
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
            info['mps_allocated_mb'] = torch.mps.current_allocated_memory() / 1024**2
            if hasattr(torch.mps, 'driver_allocated_memory'):
                info['mps_driver_mb'] = torch.mps.driver_allocated_memory() / 1024**2
        except:
            pass
    
    return info


def log_memory(step, label=""):
    """Log comprehensive memory usage at a specific step."""
    if not DEBUG_LOGGING:
        return
    
    info = get_memory_info()
    
    parts = [f"[T3 MEM] Step {step:4d} {label}:"]
    parts.append(f"Sys={info['sys_used_gb']:.1f}GB ({info['sys_percent']:.0f}%)")
    
    if 'wired_gb' in info:
        parts.append(f"Wired={info['wired_gb']:.1f}GB")
    if 'compressed_gb' in info:
        parts.append(f"Compressed={info['compressed_gb']:.1f}GB")
    if 'active_gb' in info:
        parts.append(f"Active={info['active_gb']:.1f}GB")
    if 'mps_allocated_mb' in info:
        parts.append(f"MPS={info['mps_allocated_mb']:.0f}MB")
    if 'mps_driver_mb' in info:
        parts.append(f"MPSDriver={info['mps_driver_mb']:.0f}MB")
    
    print(" | ".join(parts))


def clear_device_memory():
    """
    Clear GPU memory for both CUDA and MPS devices.
    Follows the proper MPS cleanup sequence: del → gc.collect() → synchronize → empty_cache
    """
    import gc
    # Step 1: Run garbage collector to remove Python references
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # Step 2: Synchronize to ensure all operations are complete
        torch.mps.synchronize()
        # Step 3: Empty the cache to release memory back to system
        torch.mps.empty_cache()
        # Step 4: Final synchronize to ensure buffers are released
        torch.mps.synchronize()


def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"


class T3(nn.Module):
    """
    Token-To-Token (T3) TTS model using huggingface transformer models as backbones,
        * tokenization, including start / stop tokens are always added externally to this class
        * conditioning data like CLAP, emotion, etc are all in a separate file for more modularity
        * careful! this class assumes relative positional encoding -- with absolute PE, we would at
            least want to reset the position to 0 when speech tokens begin, and optionally use a
            different PE embedding space for speech.
    """

    def __init__(self, hp=None, device=None, use_alignment_analyzer=None):
        if hp is None:
            hp = T3Config.english_only()  # Default to English-only config for backward compatibility
        super().__init__()
        self.hp = hp
        
        # Import here to avoid circular import
        from .llama_configs import get_optimal_dtype_str
        
        # Create config with device-optimized dtype
        config_dict = LLAMA_CONFIGS[hp.llama_config_name].copy()
        config_dict['torch_dtype'] = get_optimal_dtype_str(device)
        self.cfg = LlamaConfig(**config_dict)
        
        # Determine if we need alignment analyzer (only for multilingual by default)
        # If use_alignment_analyzer is None, auto-detect based on whether model is multilingual
        if use_alignment_analyzer is None:
            use_alignment_analyzer = hp.is_multilingual if hasattr(hp, 'is_multilingual') else False
        self._use_alignment_analyzer = use_alignment_analyzer
        
        # Only use 'eager' attention if alignment analyzer is needed (requires output_attentions)
        # SDPA is much faster on MPS but doesn't support output_attentions
        if use_alignment_analyzer:
            self.cfg._attn_implementation = 'eager'
        else:
            # Use SDPA for better performance, especially on MPS
            self.cfg._attn_implementation = 'sdpa'
        
        logger.info(f"T3 config: attn_implementation={self.cfg._attn_implementation}, dtype={config_dict['torch_dtype']}, device={device}")
        
        self.tfmr = LlamaModel(self.cfg)
        self.dim = self.cfg.hidden_size
        self.deepspeed_patch_applied = False
        self._compiled_model = None  # For torch.compile cache

        # conditioning / embedding
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # custom position embedding
        if hp.input_pos_emb == "learned":
            max_text_seq_len = hp.max_text_tokens + 2
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, self.dim)

            max_mel_seq_len = hp.max_speech_tokens + 2 + 2
            self.speech_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, self.dim)

        # logit projection
        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=False)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def compile_for_device(self, device=None):
        """
        Apply torch.compile() optimization for the given device.
        This can provide 1.2-1.5x speedup on supported backends.
        
        Args:
            device: Target device. If None, uses self.device.
        """
        if self._compiled_model is not None:
            return  # Already compiled
            
        device = device or self.device
        device_str = str(device).lower()
        
        try:
            if 'mps' in device_str:
                # MPS doesn't fully support inductor yet, use aot_eager
                self.tfmr = torch.compile(self.tfmr, backend="aot_eager")
                self._compiled_model = self.tfmr
                logger.info("Applied torch.compile with aot_eager backend for MPS")
            elif 'cuda' in device_str:
                # CUDA supports the full inductor backend
                self.tfmr = torch.compile(self.tfmr, backend="inductor")
                self._compiled_model = self.tfmr
                logger.info("Applied torch.compile with inductor backend for CUDA")
            else:
                # CPU - use eager backend  
                self.tfmr = torch.compile(self.tfmr, backend="eager")
                self._compiled_model = self.tfmr
                logger.info("Applied torch.compile with eager backend for CPU")
        except Exception as e:
            logger.warning(f"torch.compile failed, continuing without compilation: {e}")
            self._compiled_model = None

    def prepare_conditioning(self, t3_cond: T3Cond):
        """
        Token cond data needs to be embedded, so that needs to be here instead of in `T3CondEnc`.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens) + \
                self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)  # (B, len_cond, dim)

    def prepare_input_embeds(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        cfg_weight: float = 0.0,
    ):
        # prepare input embeddings (skip backbone tranformer embeddings)
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, len_cond, dim)
        text_emb = self.text_emb(text_tokens)  # (B, len_text, dim)
        if cfg_weight > 0.0:
            text_emb[1].zero_()  # CFG uncond

        speech_emb = self.speech_emb(speech_tokens)  # (B, len_speech, dim)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        len_cond = cond_emb.size(1)

        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        # concat
        # embeds = torch.stack([
        #     torch.cat((ce, te, se))
        #     for ce, te, se in zip(cond_emb, text_emb, speech_emb)
        # ])  # (B, length, dim)

        # More memory efficient
        embeds = torch.cat([cond_emb, text_emb, speech_emb], dim=1)

        return embeds, len_cond

    def forward(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
        training=False,
    ):
        _ensure_BOT_EOT(text_tokens, self.hp)

        # prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # backbone tranformer forward
        tfmr_out = self.tfmr.forward(
            input_ids=None,
            # position_ids=position_ids, # TODO? ROPE should be fine?
            inputs_embeds=embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=(not training),
        )
        hidden_states = tfmr_out.hidden_states[-1]  # final tfmr layer output, (B, seq, dim)

        # post-processing: splice out text and speech parts of hidden states
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        # text_latents = torch.zeros(B, len_text, dim, dtype=dtype, device=device)
        # speech_latents = torch.zeros(B, len_speech, dim, dtype=dtype, device=device)

        # More memory efficient - direct slicing
        text_latents = hidden_states[:, len_cond:len_cond + len_text, :]
        speech_latents = hidden_states[:, len_cond + len_text:len_cond + len_text + len_speech, :]

        ttl, stl = text_token_lens, speech_token_lens
        for i in range(B):
            text_end = len_cond + ttl[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + stl[i].item()
            text_latents[i, :ttl[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :stl[i]] = hidden_states[i, speech_start:speech_end]

        # logit projection
        text_logits = self.text_head(text_latents)
        speech_logits = self.speech_head(speech_latents)

        return AttrDict(
            text_logits=text_logits,
            text_latents=text_latents,
            speech_logits=speech_logits,
            speech_latents=speech_latents,
            hidden_states=hidden_states,
        )

    def loss(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        speech_tokens: torch.LongTensor,
        speech_token_lens: torch.LongTensor,
    ):
        "training method"
        len_text = text_tokens.size(1)
        len_speech = speech_tokens.size(1)
        assert len_text == text_token_lens.max()
        assert len_speech == speech_token_lens.max()

        out = self.forward(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            text_token_lens=text_token_lens,
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            training=True,
        )  # (B, seq, vocab_size)

        # Calc CCE losses
        IGNORE_ID = -100
        device = out.text_logits.device
        mask_text = torch.arange(len_text, device=device)[None] >= text_token_lens[:, None]  # (B, len_text)
        mask_speech = torch.arange(len_speech, device=device)[None] >= speech_token_lens[:, None]  # (B, len_speech)
        masked_text = text_tokens.masked_fill(mask_text, IGNORE_ID)
        masked_speech = speech_tokens.masked_fill(mask_speech, IGNORE_ID)
        loss_text = F.cross_entropy(out.text_logits, masked_text, ignore_index=IGNORE_ID)
        loss_speech = F.cross_entropy(out.speech_logits, masked_speech, ignore_index=IGNORE_ID)

        return loss_text, loss_speech

    @torch.inference_mode()
    def inference(
        self,
        *,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        initial_speech_tokens: Optional[Tensor]=None,

        # misc conditioning
        prepend_prompt_speech_tokens: Optional[Tensor]=None,

        # HF generate args
        num_return_sequences=1,
        max_new_tokens=None,
        stop_on_eos=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        min_p=0.05,
        length_penalty=1.0,
        repetition_penalty=1.2,
        cfg_weight=0.5,
    ):
        """
        Args:
            text_tokens: a 1D (unbatched) or 2D (batched) tensor.
        """
        # Validate / sanitize inputs
        assert prepend_prompt_speech_tokens is None, "not implemented"
        _ensure_BOT_EOT(text_tokens, self.hp)
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)

        # Default initial speech to a single start-of-speech token
        if initial_speech_tokens is None:
            initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        # Prepare custom input embeds
        embeds, len_cond = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=initial_speech_tokens,
            cfg_weight=cfg_weight,
        )

        # In order to use the standard HF generate method, we need to extend some methods to inject our custom logic
        # Note the llama-specific logic. Other tfmr types can be added later.

        # Initialize patched_model if not already done (avoid reinitializing on every inference)
        if not hasattr(self, '_patched_model_initialized') or not self._patched_model_initialized:
            # Default to None for English models, only create for multilingual
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(
                    self.tfmr,
                    None,
                    text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)),
                    alignment_layer_idx=9, # TODO: hparam or something?
                    eos_idx=self.hp.stop_speech_token,
                )
                assert alignment_stream_analyzer.eos_idx == self.hp.stop_speech_token

            patched_model = T3HuggingfaceBackend(
                config=self.cfg,
                llama=self.tfmr,
                speech_enc=self.speech_emb,
                speech_head=self.speech_head,
                alignment_stream_analyzer=alignment_stream_analyzer,
            )
            self.patched_model = patched_model
            self._patched_model_initialized = True

        # # Run normal generate method, which calls our custom extended methods
        # return self.patched_model.generate(
        #     inputs=initial_speech_tokens,
        #     decoder_cond=embeds,
        #     bos_token_id=self.hp.start_speech_token,
        #     eos_token_id=(self.hp.stop_speech_token if stop_on_eos else -1),
        #     pad_token_id=self.hp.stop_speech_token,
        #     max_new_tokens=max_new_tokens or self.hp.max_speech_tokens,
        #     num_return_sequences=num_return_sequences,
        #     temperature=temperature,
        #     min_p=min_p,
        #     length_penalty=length_penalty,
        #     repetition_penalty=repetition_penalty,
        #     do_sample=do_sample,
        #     # cache_implementation=None if not self.compiled else "static",
        # )

        device = embeds.device

        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)  # shape: (B, 1, embed_dim)
        bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # batch_size=2 for CFG
        bos_embed = torch.cat([bos_embed, bos_embed])

        # Combine condition and BOS token for the initial input
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated token ids; start with the BOS token.
        # Pre-allocate tensor for generated_ids to avoid concatenation overhead
        generated_ids = torch.zeros(1, max_new_tokens + 1, dtype=torch.long, device=device)
        generated_ids[0, 0] = self.hp.start_speech_token
        num_generated = 1
        
        # Pre-allocate tensor for predicted tokens (instead of list to avoid memory accumulation)
        predicted_tokens = torch.zeros(1, max_new_tokens, dtype=torch.long, device=device)
        num_predicted = 0

        # Instantiate the logits processors.
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)
        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))

        # ---- Initial Forward Pass with StaticCache ----
        # Add memory cleanup before the memory-intensive operation
        import gc
        clear_device_memory()

        # Disable gradient checkpointing during inference (it adds overhead)
        # Only useful during training to trade compute for memory
        if hasattr(self.patched_model, 'gradient_checkpointing_disable'):
            self.patched_model.gradient_checkpointing_disable()

        # Only enable output_attentions if alignment analyzer is being used (multilingual models)
        # This is a major performance optimization - attention materialization is O(n²) memory
        needs_attentions = self.patched_model.alignment_stream_analyzer is not None
        
        # ---- MPS-OPTIMIZED KV CACHE ----
        # Pre-allocates all memory upfront, preventing MPS driver memory growth.
        # Uses narrow().copy_() which is 43x faster than indexed assignment on MPS.
        # DynamicCache grows incrementally, causing Metal to allocate new buffers each step,
        # which leads to unbounded driver memory growth on MPS.
        context_length = inputs_embeds.size(1)  # Initial context tokens
        max_cache_length = context_length + max_new_tokens + 10  # +10 buffer for safety
        
        # Create MPS-optimized cache with pre-allocated memory
        # This allocates all KV cache memory upfront instead of growing dynamically
        static_cache = MPSOptimizedCache(
            config=self.cfg,
            max_cache_len=max_cache_length,
            device=device,
            dtype=inputs_embeds.dtype,
        )
        logger.info(f"📦 Created MPSOptimizedCache: max_cache_len={max_cache_length} (context={context_length} + max_new={max_new_tokens})")
        log_memory(0, "after_static_cache_creation")
        
        try:
            output = self.patched_model(
                inputs_embeds=inputs_embeds,
                past_key_values=static_cache,  # Use pre-allocated cache
                use_cache=True,
                output_attentions=needs_attentions,  # Only when alignment analyzer is used
                output_hidden_states=True,  # Required by T3 backend, keep enabled
                return_dict=True,
            )
        except Exception as e:
            # Clean up cache on error
            if hasattr(static_cache, 'reset'):
                static_cache.reset()
            del static_cache
            raise e
        # Cache is updated in-place, so we just keep using the same object
        past = output.past_key_values  # This is the same cache object, now populated
        current_logits = output.logits  # Store logits for first iteration
        del output  # Free the initial output immediately

        # Keep gradient checkpointing enabled throughout generation for better memory management
        # (Previously disabled here, but keeping it on reduces memory usage)

        # Clean up memory after initial forward pass
        clear_device_memory()
        log_memory(0, "after_initial_forward")

        # ---- Generation Loop using kv_cache ----
        for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
            # Log memory every 50 steps
            if i % 50 == 0:
                log_memory(i, "loop_start")
            
            logits_step = current_logits[:, -1, :]                
            # CFG combine  → (1, V)
            cond   = logits_step[0:1, :]
            uncond = logits_step[1:2, :]
            cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
            logits = cond + cfg * (cond - uncond)
            
            # Free logits_step early
            del logits_step
            
            # Apply alignment stream analyzer integrity checks
            if self.patched_model.alignment_stream_analyzer is not None:
                if logits.dim() == 1:            # guard in case something upstream squeezed
                    logits = logits.unsqueeze(0) # (1, V)
                # Pass the last generated token for repetition tracking
                last_token = generated_ids[0, num_generated - 1].item() if num_generated > 0 else None
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=last_token)  # (1, V)

            # Apply repetition penalty using only the actual generated portion
            ids_for_proc = generated_ids[:1, :num_generated]
            logits = repetition_penalty_processor(ids_for_proc, logits)  # expects (B,V)
            
            # Apply temperature scaling.
            if temperature != 1.0:
                logits = logits / temperature
                
            # Apply min_p and top_p filtering
            logits = min_p_warper(ids_for_proc, logits)
            logits = top_p_warper(ids_for_proc, logits)

            # Convert logits to probabilities and sample the next token.
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (B, 1)
            
            # Free intermediate tensors
            del probs, logits

            # Store predicted token in pre-allocated tensor (avoids memory accumulation from list)
            predicted_tokens[0, num_predicted] = next_token.view(-1)
            num_predicted += 1
            
            # Use in-place assignment instead of concatenation (faster, less memory)
            generated_ids[0, num_generated] = next_token.view(-1)
            num_generated += 1

            # Check for EOS token.
            if next_token.view(-1) == self.hp.stop_speech_token:
                logger.info(f"✅ EOS token detected! Stopping generation at step {i+1}")
                break

            # Get embedding for the new token.
            next_token_embed = self.speech_emb(next_token)
            next_token_embed = next_token_embed + self.speech_pos_emb.get_fixed_embedding(i + 1)

            #  For CFG
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            # Forward pass with only the new token and the cached past.
            forward_output = self.patched_model(
                inputs_embeds=next_token_embed,
                past_key_values=past,
                output_attentions=needs_attentions,  # Only when alignment analyzer is used
                output_hidden_states=False,  # Not needed during generation, saves memory
                return_dict=True,
            )
            # Extract what we need and delete the output object immediately
            past = forward_output.past_key_values
            current_logits = forward_output.logits
            del forward_output
            del next_token_embed  # Free embedding tensor

            # With MPSOptimizedCache, memory is pre-allocated upfront, so aggressive per-step
            # cleanup is no longer needed. This improves performance.
            # Only sync periodically for memory logging when DEBUG is enabled
                
            # Memory logging every 50 steps (when DEBUG enabled)
            if i % 50 == 0:
                log_memory(i, "after_forward")

        log_memory(num_predicted, "generation_complete")
        
        # Trim predicted_tokens to actual length
        predicted_tokens = predicted_tokens[:, :num_predicted]
        
        # CRITICAL: Clean up cache and intermediate tensors to prevent memory leak
        # reset() clears the cache contents but keeps the allocated memory
        # We need to delete the cache object entirely to free the memory
        logger.info(f"🧹 Cleaning up MPSOptimizedCache...")
        if hasattr(past, 'reset'):
            past.reset()  # Clear cache contents first
        # Delete the cache to free memory (important for MPS)
        del past
        
        # Explicit cleanup of static_cache reference (same object as past)
        if 'static_cache' in dir():
            del static_cache
            
        del current_logits
        del embeds
        del inputs_embeds
        del bos_embed
        del bos_token
        del generated_ids
        
        # Reset alignment stream analyzer state if it exists
        if self.patched_model.alignment_stream_analyzer is not None:
            self.patched_model.alignment_stream_analyzer.reset()
        
        # Force memory cleanup
        clear_device_memory()
        log_memory(num_predicted, "after_cleanup")
        
        return predicted_tokens
