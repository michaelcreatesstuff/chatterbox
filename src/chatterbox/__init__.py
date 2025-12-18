try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox")


from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .tts_turbo import ChatterboxTurboTTS
from .models import (
    DEBUG_LOGGING,
    is_debug,
    set_mlx_cache_limit,
    set_mlx_memory_limit,
)
from .text_sanitizer import (
    sanitize_text_for_tts,
    has_unsupported_characters,
    get_unsupported_characters_summary,
    UnsupportedCharactersSummary,
)

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "ChatterboxMultilingualTTS",
    "ChatterboxTurboTTS",
    "SUPPORTED_LANGUAGES",
    "DEBUG_LOGGING",
    "is_debug",
    "set_mlx_cache_limit",
    "set_mlx_memory_limit",
    "sanitize_text_for_tts",
    "has_unsupported_characters",
    "get_unsupported_characters_summary",
    "UnsupportedCharactersSummary",
]