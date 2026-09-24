from .const import S3GEN_SR as S3GEN_SR

__all__ = ["S3Token2Wav", "S3Gen", "S3GEN_SR"]


# S3Token2Wav depends on diffusers, which imports transformers when it is
# installed; resolve it lazily (PEP 562) so importing S3GEN_SR stays cheap.
def __getattr__(name):
    if name in ("S3Token2Wav", "S3Gen"):
        from .s3gen import S3Token2Wav

        # "S3Gen" is a legacy alias for backwards compatibility
        globals()["S3Token2Wav"] = S3Token2Wav
        globals()["S3Gen"] = S3Token2Wav
        return S3Token2Wav
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
