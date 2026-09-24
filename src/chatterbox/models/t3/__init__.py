from .modules.t3_config import T3Config as T3Config

__all__ = ["T3", "T3Config"]


# T3 imports transformers; resolve it lazily (PEP 562) so importing
# models.t3.modules.* (as the MLX path does) doesn't load transformers.
def __getattr__(name):
    if name == "T3":
        from .t3 import T3

        globals()["T3"] = T3
        return T3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
