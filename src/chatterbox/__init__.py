try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-mlx")

# Check for lzma support (required by librosa dependency)
import sys

try:
    import lzma

    # Actually try to use it to detect if _lzma C extension is missing
    lzma.LZMADecompressor()
except (ImportError, AttributeError) as e:
    error_msg = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  chatterbox-mlx requires Python to be compiled with lzma support            ║
║                                                                              ║
║  Your Python installation is missing the '_lzma' module.                    ║
║  This typically happens when Python was compiled without liblzma headers.   ║
║                                                                              ║
║  To fix this (macOS with Homebrew + pyenv):                                 ║
║    brew install xz                                                           ║
║    pyenv uninstall {version}                                                 ║
║    pyenv install {version}                                                   ║
║    # Then recreate your virtual environment                                 ║
║                                                                              ║
║  See: https://github.com/michaelcreatesstuff/chatterbox#installation        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """.format(
        version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    print(error_msg, file=sys.stderr)
    raise ImportError(
        "Python lzma module not found. Install liblzma-dev and recompile Python. "
        "See error message above for details."
    ) from e

from .tts import ChatterboxTTS
from .vc import ChatterboxVC
from .mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
from .models import (
    DEBUG_LOGGING,
    is_debug,
    set_mlx_cache_limit,
    set_mlx_memory_limit,
)

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "ChatterboxMultilingualTTS",
    "SUPPORTED_LANGUAGES",
    "DEBUG_LOGGING",
    "is_debug",
    "set_mlx_cache_limit",
    "set_mlx_memory_limit",
]
