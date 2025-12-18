from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    def __init__(self, text_tokens_dict_size=704, kv_cache_dtype='float16'):
        self.start_text_token = 255
        self.stop_text_token = 0
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = 2048

        self.start_speech_token = 6561
        self.stop_speech_token = 6562
        self.speech_tokens_dict_size = 8194
        self.max_speech_tokens = 4096

        self.llama_config_name = "Llama_520M"
        self.input_pos_emb = "learned"
        self.speech_cond_prompt_len = 150

        self.encoder_type = "voice_encoder"
        self.speaker_embed_size = 256
        self.use_perceiver_resampler = True
        self.emotion_adv = True

        # KV cache dtype optimization: 'float16' (default, recommended) or None for full precision
        # Float16 KV cache provides 18-32% speed improvement with significant memory savings
        self.kv_cache_dtype = kv_cache_dtype

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]
    
    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls, kv_cache_dtype='float16'):
        """Create configuration for English-only TTS model.

        Args:
            kv_cache_dtype: KV cache dtype - 'float16' (default, recommended) or None for full precision.
                          Float16 provides 18-32% speed improvement with significant memory savings.
        """
        return cls(text_tokens_dict_size=704, kv_cache_dtype=kv_cache_dtype)

    @classmethod
    def multilingual(cls, kv_cache_dtype='float16'):
        """Create configuration for multilingual TTS model.

        Args:
            kv_cache_dtype: KV cache dtype - 'float16' (default, recommended) or None for full precision.
                          Float16 provides 18-32% speed improvement with significant memory savings.
        """
        return cls(text_tokens_dict_size=2454, kv_cache_dtype=kv_cache_dtype)

    @classmethod
    def turbo(cls, kv_cache_dtype='float16'):
        """Create configuration for Chatterbox-Turbo model.

        Turbo is a 350M parameter model optimized for low-latency voice agents.
        Uses GPT2-Medium backbone with distilled speech decoder.

        Key differences from standard Chatterbox:
        - GPT2-Medium backbone (350M vs 520M Llama)
        - Larger text vocabulary (GPT2 tokenizer: 50276 vs EnTokenizer: 704)
        - Smaller speech vocabulary (6563 vs 8194)
        - No perceiver resampler (simpler architecture)
        - No emotion adversarial training
        - Longer speech conditioning prompt (375 vs 150)
        - Native paralinguistic tag support ([cough], [laugh], etc.)

        Args:
            kv_cache_dtype: KV cache dtype - 'float16' (recommended) or None.
        """
        config = cls.__new__(cls)

        # GPT2 tokenizer has larger vocab
        config.text_tokens_dict_size = 50276  # GPT2 vocab (50257) + special tokens
        config.start_text_token = 50256  # GPT2 special token
        config.stop_text_token = 50257
        config.max_text_tokens = 2048

        # Turbo uses smaller speech vocab
        config.speech_tokens_dict_size = 6563  # Reduced from 8194
        config.start_speech_token = 6561
        config.stop_speech_token = 6562
        config.max_speech_tokens = 4096

        # GPT2-Medium backbone (350M parameters)
        config.llama_config_name = "GPT2_Medium"

        # Turbo doesn't use custom position embeddings (relies on GPT2's built-in learned PE)
        config.input_pos_emb = None

        # Longer speech conditioning prompt for better voice cloning
        config.speech_cond_prompt_len = 375  # Increased from 150

        # Turbo simplifications (no perceiver, no emotion adversarial)
        config.encoder_type = "voice_encoder"
        config.speaker_embed_size = 256
        config.use_perceiver_resampler = False  # Disabled for turbo
        config.emotion_adv = False  # Disabled for turbo

        # KV cache optimization
        config.kv_cache_dtype = kv_cache_dtype

        return config