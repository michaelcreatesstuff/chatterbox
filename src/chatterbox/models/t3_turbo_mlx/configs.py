# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
GPT-2 backbone configs and T3 hyperparameters of Chatterbox-Turbo / Nano.

The two config dicts are copies of upstream ``models/t3/llama_configs.py``
(``GPT2_MEDIUM_CONFIG`` for Turbo, ``GPT2_SMALL_CONFIG`` for Nano). They live
here because this fork's torch T3 has no GPT-2 branch.
"""

from dataclasses import dataclass

from ..t3.modules.t3_config import T3Config

GPT2_MEDIUM_CONFIG = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 8196,
    "n_embd": 1024,
    "hidden_size": 1024,
    "n_head": 16,
    "n_layer": 24,
    "n_positions": 8196,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
    "vocab_size": 50276,
}

GPT2_SMALL_CONFIG = {
    "activation_function": "gelu_new",
    "architectures": ["GPT2LMHeadModel"],
    "attn_pdrop": 0.1,
    "bos_token_id": 50256,
    "embd_pdrop": 0.1,
    "eos_token_id": 50256,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "model_type": "gpt2",
    "n_ctx": 8196,
    "n_embd": 768,
    "hidden_size": 768,
    "n_head": 12,
    "n_layer": 12,
    "n_positions": 8196,
    "n_special": 0,
    "predict_special_tokens": True,
    "resid_pdrop": 0.1,
    "summary_activation": None,
    "summary_first_dropout": 0.1,
    "summary_proj_to_labels": True,
    "summary_type": "cls_index",
    "summary_use_proj": True,
    "task_specific_params": {"text-generation": {"do_sample": True, "max_length": 50}},
    "vocab_size": 50276,
}

GPT2_CONFIGS = {
    "GPT2_medium": GPT2_MEDIUM_CONFIG,
    "GPT2_small": GPT2_SMALL_CONFIG,
}


@dataclass
class GPT2ConfigMLX:
    """The GPT-2 fields the MLX backbone needs (HF ``GPT2Config`` names)."""

    n_embd: int = 1024
    n_head: int = 16
    n_layer: int = 24
    n_positions: int = 8196
    layer_norm_epsilon: float = 1e-5
    n_inner: int = 0  # 0 = 4 * n_embd, as HF's ``n_inner=None``

    @classmethod
    def from_dict(cls, d: dict) -> "GPT2ConfigMLX":
        if d.get("activation_function", "gelu_new") != "gelu_new":
            raise NotImplementedError("Only gelu_new GPT-2 configs are supported")
        return cls(
            n_embd=d["n_embd"],
            n_head=d["n_head"],
            n_layer=d["n_layer"],
            n_positions=d["n_positions"],
            layer_norm_epsilon=d.get("layer_norm_epsilon", 1e-5),
            n_inner=d.get("n_inner") or 0,
        )

    @property
    def hidden_size(self) -> int:
        return self.n_embd

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def inner_dim(self) -> int:
        return self.n_inner or 4 * self.n_embd


class TurboT3Config(T3Config):
    """``T3Config`` as upstream ``ChatterboxTurboTTS.from_local`` sets it up.

    ``n_channels`` is read from :data:`GPT2_CONFIGS` (the base class looks the
    name up in the fork's Llama configs, which have no GPT-2 entries), or from
    an explicit ``gpt2_config`` dict (tests use tiny random configs).
    """

    def __init__(self, nano: bool = False, gpt2_config: dict = None):
        super().__init__(text_tokens_dict_size=50276)
        self.llama_config_name = "GPT2_small" if nano else "GPT2_medium"
        self.speech_tokens_dict_size = 6563
        self.input_pos_emb = None
        self.speech_cond_prompt_len = 375
        self.use_perceiver_resampler = False
        self.emotion_adv = False
        self.gpt2_config = dict(
            gpt2_config
            if gpt2_config is not None
            else GPT2_CONFIGS[self.llama_config_name]
        )

    @property
    def n_channels(self) -> int:
        return self.gpt2_config["hidden_size"]
