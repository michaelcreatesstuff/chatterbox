# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Perth watermark helpers shared by the MLX pipelines.

- :class:`PerthWatermarkMixin` gives a pipeline a lazily built
  ``perth.PerthImplicitWatermarker`` (``watermarker``) and ``_watermark(wav)``.
  Nothing is imported or built until the first watermarked call, so an engine
  that runs with ``watermark=False`` pays no load time or memory for it.
- :class:`StreamWatermark` watermarks a stream piece by piece. Each piece is
  watermarked with the end of the previous (unwatermarked) audio as left
  context (:data:`WATERMARK_CONTEXT_S`), so the joins stay clean.

``ChatterboxTurboMLX`` (Turbo / Nano), ``ChatterboxFlashMLX`` and
``ChatterboxMultilingualTTSMLX`` use them.
"""

from typing import Callable

import numpy as np
import torch

from .models.utils import TORCH_LOCK

# Unwatermarked audio fed back as left context to the next stream piece.
WATERMARK_CONTEXT_S = 0.25


def build_perth_watermarker():
    """A new ``perth.PerthImplicitWatermarker`` (imports Perth on first call)."""
    import perth

    return perth.PerthImplicitWatermarker()


class PerthWatermarkMixin:
    """Lazy Perth watermarker for a pipeline with a sample rate ``self.sr``.

    ``watermarker`` builds the watermarker on first read; assigning it (for
    example a stub in tests) replaces it. The instance stores it in
    ``_watermarker``.
    """

    _watermarker = None

    @property
    def watermarker(self):
        if self._watermarker is None:
            with TORCH_LOCK:
                if self._watermarker is None:
                    self._watermarker = build_perth_watermarker()
        return self._watermarker

    @watermarker.setter
    def watermarker(self, value) -> None:
        self._watermarker = value

    @property
    def watermarker_loaded(self) -> bool:
        """True once the watermarker is built (or assigned)."""
        return self._watermarker is not None

    def _watermark(self, wav: np.ndarray) -> np.ndarray:
        """Perth-watermark a 1-D waveform at ``self.sr``; returns 1-D float32."""
        # Perth runs torch ops; with autograd on it leaves tensors with a
        # grad_fn behind on every call.
        wm = self.watermarker
        with TORCH_LOCK, torch.inference_mode():
            out = wm.apply_watermark(
                np.asarray(wav, dtype=np.float32).reshape(-1), sample_rate=self.sr
            )
        return np.asarray(out, dtype=np.float32).reshape(-1)


class StreamWatermark:
    """Watermark consecutive stream pieces with left context.

    ``apply`` is a whole-waveform watermark function (``_watermark``). Each
    call watermarks ``[context, piece]`` and returns the piece's part, where
    ``context`` is the last ``context_s`` seconds of the unwatermarked audio
    before the piece. The output has the piece's length.
    """

    def __init__(
        self,
        apply: Callable[[np.ndarray], np.ndarray],
        sr: int,
        context_s: float = WATERMARK_CONTEXT_S,
    ):
        self.apply = apply
        self.ctx_len = max(0, int(context_s * sr))
        self.ctx = np.zeros((0,), dtype=np.float32)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if not len(y):
            return y
        x = np.concatenate([self.ctx, y])
        w = self.apply(x)[len(self.ctx) :]
        if len(w) < len(y):
            w = np.concatenate([w, y[len(w) :]])
        self.ctx = x[-self.ctx_len :] if self.ctx_len else self.ctx[:0]
        return w[: len(y)].astype(np.float32, copy=False)
