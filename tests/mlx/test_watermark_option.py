# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for the ``watermark`` option of the MLX engines and the shared Perth
helpers (:mod:`chatterbox.watermarking`).

* Stub tests (no weights): the watermarker is built lazily; ``watermark=True``
  calls it once per whole output, or once per stream piece with left
  context; the stream keeps its piece count and lengths; the defaults
  (multilingual and Flash off, Turbo / Nano on) never change.
* Cached-model tests: with the default arguments, multilingual (fp16 T3 +
  meanflow), Turbo, Nano and Flash give the output of 1.3.0. The speech
  tokens given to the vocoder must be byte-identical (SHA-256), the audio
  must have the same length, and its RMS per 100 ms frame must match within
  :data:`FRAME_RMS_ATOL`. The audio itself is not compared bit for bit: the
  MLX vocoder is not bit-reproducible, even in 1.3.0 (the same tokens and
  seed give differences up to ~1e-7, RMS ~2e-9, call to call). A watermark
  changes the frame RMS by far more than the tolerance. The references are
  in ``data/watermark_defaults_1.3.0.json``. They were made on one machine,
  so the tests skip when the checkpoints are not cached, or when the MLX /
  torch versions or the chip differ. Flash needs a reference voice: set
  ``CHATTERBOX_TEST_VOICE`` to a wav file.

Write the reference file again (from the library on ``PYTHONPATH``)::

    python tests/mlx/test_watermark_option.py --write-golden [CASE ...]
"""

import gc
import glob
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "data" / "watermark_defaults_1.3.0.json"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
SR = 24000
FRAME = 2400  # 100 ms
FRAME_RMS_ATOL = 1e-6

EN_TEXT = (
    "The quick brown fox jumps over the lazy dog near the river bank. "
    "Then it rests in the shade of an old oak tree for a while."
)
ES_TEXT = (
    "Soy Myna, una voz asistente que corre solo en tu Mac. No tengo internet "
    "ni soy de ninguna otra marca. ¿Qué necesitas que haga hoy?"
)


class _Watermarker:
    """Stub Perth: halves the audio and records every input."""

    def __init__(self):
        self.inputs = []

    def apply_watermark(self, wav, sample_rate):
        assert sample_rate == SR
        self.inputs.append(np.array(wav, dtype=np.float32))
        return np.asarray(wav, dtype=np.float32) * 0.5


# --------------------------------------------------------------------------- #
#  Shared helpers
# --------------------------------------------------------------------------- #


class TestHelpers(unittest.TestCase):
    def test_lazy_build_once(self):
        import chatterbox.watermarking as W

        class Host(W.PerthWatermarkMixin):
            sr = SR

        stub = _Watermarker()
        with mock.patch.object(W, "build_perth_watermarker", return_value=stub) as b:
            h = Host()
            self.assertFalse(h.watermarker_loaded)
            b.assert_not_called()
            x = np.linspace(-1, 1, 100, dtype=np.float32)
            np.testing.assert_allclose(h._watermark(x), 0.5 * x)
            h._watermark(x)
            self.assertEqual(b.call_count, 1)
            self.assertIs(h.watermarker, stub)
            self.assertTrue(h.watermarker_loaded)
        other = _Watermarker()
        h.watermarker = other
        self.assertIs(h.watermarker, other)
        self.assertIsNone(Host()._watermarker)  # instances do not share it

    def test_stream_context_and_lengths(self):
        from chatterbox.watermarking import WATERMARK_CONTEXT_S, StreamWatermark

        stub = _Watermarker()
        sw = StreamWatermark(lambda x: stub.apply_watermark(x, SR), SR)
        ctx = int(WATERMARK_CONTEXT_S * SR)
        rng = np.random.default_rng(0)
        pieces = [rng.standard_normal(n).astype(np.float32) for n in (3000, 9000, 200, 7000)]
        outs = [sw(p) for p in pieces]
        self.assertEqual([len(o) for o in outs], [len(p) for p in pieces])
        self.assertEqual(len(stub.inputs), len(pieces))
        whole = np.concatenate(pieces)
        end = 0
        for p, inp in zip(pieces, stub.inputs):
            # The input is the unwatermarked audio before the piece, then the piece.
            start = max(0, end - ctx)
            np.testing.assert_array_equal(inp, whole[start : end + len(p)])
            end += len(p)
        np.testing.assert_allclose(np.concatenate(outs), 0.5 * whole)
        self.assertEqual(len(sw(np.zeros(0, np.float32))), 0)
        self.assertEqual(len(stub.inputs), len(pieces))  # empty piece: no call

    def test_stream_matches_turbo_1_3_0(self):
        """StreamWatermark reproduces Turbo 1.3.0's inline per-piece code."""
        from chatterbox.watermarking import StreamWatermark

        def apply(x):  # not pointwise: depends on the whole input
            return (x * 0.9 + np.mean(x)).astype(np.float32)

        def old(pieces):  # tts_turbo_mlx.py 1.3.0, generate_stream.out()
            wm_ctx = np.zeros((0,), dtype=np.float32)
            wm_ctx_len = int(0.25 * SR)
            res = []
            for y in pieces:
                x = np.concatenate([wm_ctx, y])
                w = apply(x)[len(wm_ctx) :]
                if len(w) < len(y):
                    w = np.concatenate([w, y[len(w) :]])
                wm_ctx = x[-wm_ctx_len:]
                res.append(w[: len(y)].astype(np.float32, copy=False))
            return res

        rng = np.random.default_rng(1)
        pieces = [rng.standard_normal(n).astype(np.float32) for n in (24000, 100, 30000)]
        sw = StreamWatermark(apply, SR)
        for a, b in zip(old(pieces), [sw(p) for p in pieces]):
            np.testing.assert_array_equal(a, b)


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #


class TestDefaults(unittest.TestCase):
    def _default(self, fn):
        return inspect.signature(fn).parameters["watermark"].default

    def test_engine_defaults(self):
        from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX as MTL
        from chatterbox.tts_flash_mlx import ChatterboxFlashMLX as Flash
        from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX as Turbo

        for fn in (MTL.generate, MTL.generate_stream, MTL.generate_long, Flash.generate):
            self.assertIs(self._default(fn), False, fn.__qualname__)
        for fn in (Turbo.generate, Turbo.generate_stream):
            self.assertIs(self._default(fn), True, fn.__qualname__)

    def test_mtl_does_not_import_perth(self):
        src = (Path(__file__).parents[2] / "src/chatterbox/mtl_tts_mlx.py").read_text()
        self.assertNotIn("import perth", src)


# --------------------------------------------------------------------------- #
#  Multilingual (stubs from test_mtl_vocoder_stream)
# --------------------------------------------------------------------------- #


def _mtl_fixtures():
    sys.path.insert(0, str(HERE))
    import test_mtl_vocoder_stream as F

    return F


class TestMultilingual(unittest.TestCase):
    def setUp(self):
        self.F = _mtl_fixtures()
        M = self.F.M
        for p in (
            mock.patch.object(M, "split_into_sentences", self.F._split),
            mock.patch.object(M, "SPACY_AVAILABLE", True),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _tts(self, vocoder="meanflow"):
        tts = self.F._tts(vocoder)
        self.assertFalse(tts.watermarker_loaded)  # lazy: nothing built in __init__
        tts.watermarker = _Watermarker()
        return tts

    def test_off_by_default(self):
        tts = self._tts()
        text = self.F.THREE_SENTENCES
        a = self.F._np(tts.generate(text, "en", seed=1))
        self.assertEqual(tts.last_timings["watermark_s"], 0.0)
        self.assertEqual(tts.last_timings["n_chunks"], 3)
        b = np.concatenate(list(tts.generate_stream(text, "en", seed=1)))
        self.assertEqual(tts.last_timings["watermark_s"], 0.0)
        c = self.F._np(tts.generate_long(text, "en", seed=1))
        self.assertEqual(tts.watermarker.inputs, [])
        np.testing.assert_array_equal(
            a, self.F._np(tts.generate(text, "en", seed=1, watermark=False))
        )
        np.testing.assert_array_equal(
            b,
            np.concatenate(list(tts.generate_stream(text, "en", seed=1, watermark=False))),
        )
        np.testing.assert_array_equal(
            c, self.F._np(tts.generate_long(text, "en", seed=1, watermark=False))
        )
        self.assertEqual(tts.watermarker.inputs, [])

    def test_generate_once_per_output(self):
        for text in ("Just one short sentence here.", self.F.THREE_SENTENCES):
            for method in ("generate", "generate_long"):
                tts = self._tts()
                fn = getattr(tts, method)
                plain = self.F._np(fn(text, "en", seed=4))
                marked = fn(text, "en", seed=4, watermark=True)
                self.assertEqual(tuple(marked.shape), (1, len(plain)))
                self.assertEqual(len(tts.watermarker.inputs), 1, (method, text))
                self.assertEqual(marked.dtype, fn(text, "en", seed=4).dtype)
                # generate_long's crossfade is float64; Perth gets float32.
                np.testing.assert_allclose(tts.watermarker.inputs[0], plain, atol=1e-7)
                np.testing.assert_allclose(self.F._np(marked), 0.5 * plain, atol=1e-7)
                self.assertGreaterEqual(tts.last_timings["watermark_s"], 0.0)

    def test_stream_per_piece_with_context(self):
        from chatterbox.watermarking import WATERMARK_CONTEXT_S

        for vocoder, first in (("meanflow", 25), ("meanflow", 0), ("pt", 25)):
            tts = self._tts(vocoder)
            text = self.F.THREE_SENTENCES + " " + self.F.LONG_SENTENCE
            kw = dict(seed=7, first_chunk_tokens=first)
            plain = list(tts.generate_stream(text, "en", **kw))
            n_plain = tts.last_timings["n_pieces"]
            marked = list(tts.generate_stream(text, "en", watermark=True, **kw))
            t = tts.last_timings
            self.assertEqual(len(marked), len(plain))
            self.assertEqual(t["n_pieces"], n_plain)
            self.assertEqual([len(m) for m in marked], [len(p) for p in plain])
            self.assertEqual(len(tts.watermarker.inputs), len(marked))
            self.assertGreater(t["watermark_s"], 0.0)
            whole = np.concatenate(plain)
            ctx, end = int(WATERMARK_CONTEXT_S * SR), 0
            for p, m, inp in zip(plain, marked, tts.watermarker.inputs):
                self.assertEqual(m.dtype, np.float32)
                np.testing.assert_allclose(m, 0.5 * p, atol=1e-7)
                start = max(0, end - ctx)
                np.testing.assert_array_equal(inp, whole[start : end + len(p)])
                end += len(p)


# --------------------------------------------------------------------------- #
#  Flash (stubs as test_t3_flash_mlx.TestPipeline)
# --------------------------------------------------------------------------- #


class TestFlash(unittest.TestCase):
    def _tts(self):
        import torch

        sys.path.insert(0, str(HERE))
        from test_t3_flash_mlx import tiny_cond_mlx, tiny_mlx

        from chatterbox.tts_flash_mlx import ChatterboxFlashMLX

        class _Tok:
            def text_to_tokens(self, text):
                n = max(1, len(text.split()))
                return torch.arange(2, 2 + n, dtype=torch.int32)[None] % 40 + 2

        class _S3Gen:
            def inference(self, speech_tokens, ref_dict, n_cfm_timesteps=None):
                n = 480 * len(speech_tokens)
                return torch.linspace(-0.5, 0.5, n)[None], None

        t3 = tiny_mlx()
        tts = ChatterboxFlashMLX(t3, _S3Gen(), ve=None, tokenizer=_Tok(), s3gen_backend="mps")
        tts._t3_cond_emb = t3.prepare_conditioning(tiny_cond_mlx())
        tts._ref_dict = {}
        self.assertFalse(tts.watermarker_loaded)
        tts.watermarker = _Watermarker()
        return tts

    def test_off_by_default_and_once_when_on(self):
        tts = self._tts()
        text = "one two three four five six. " * 12
        plain = tts.generate(text, seed=5, max_batch_chunks=2)
        self.assertGreater(tts.last_timings["n_chunks"], 1)
        self.assertEqual(tts.last_timings["watermark_s"], 0.0)
        np.testing.assert_array_equal(plain, tts.generate(text, seed=5, max_batch_chunks=2, watermark=False))
        self.assertEqual(tts.watermarker.inputs, [])
        marked = tts.generate(text, seed=5, max_batch_chunks=2, watermark=True)
        self.assertEqual(len(tts.watermarker.inputs), 1)
        np.testing.assert_array_equal(tts.watermarker.inputs[0], plain)
        self.assertEqual(marked.dtype, np.float32)
        self.assertEqual(marked.shape, plain.shape)
        np.testing.assert_allclose(marked, 0.5 * plain, atol=1e-7)


# --------------------------------------------------------------------------- #
#  Cached models: default output is byte-identical to 1.3.0
# --------------------------------------------------------------------------- #


def _snapshot(repo: str, *files: str):
    for d in sorted(glob.glob(str(HUB / f"models--ResembleAI--{repo}/snapshots/*"))):
        if all((Path(d) / f).exists() for f in files):
            return Path(d)
    return None


def _env() -> dict:
    import mlx.core as mx
    import torch

    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        chip = platform.machine()
    return {"mlx": mx.__version__, "torch": torch.__version__, "chip": chip}


def _digest(wav, tokens=None) -> dict:
    a = wav.numpy() if hasattr(wav, "numpy") else np.asarray(wav)
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    pad = np.concatenate([a, np.zeros((-len(a)) % FRAME, np.float32)]).astype(np.float64)
    out = {
        "n": int(len(a)),
        "frame_rms": [float(x) for x in np.sqrt(np.mean(pad.reshape(-1, FRAME) ** 2, axis=1))],
    }
    if tokens is not None:
        h = hashlib.sha256()
        for t in tokens:
            h.update(np.ascontiguousarray(t, dtype=np.int64).tobytes() + b"|")
        out["tokens_sha256"] = h.hexdigest()
        out["n_tokens"] = int(sum(len(t) for t in tokens))
    return out


def _recording(obj, name):
    """Wrap ``obj.<name>`` to record its first argument (the speech tokens)."""
    seen, orig = [], getattr(obj, name)

    def wrapper(tokens, *a, **k):
        seen.append(np.asarray(tokens).reshape(-1).copy())
        return orig(tokens, *a, **k)

    setattr(obj, name, wrapper)
    return seen


def _same(got: dict, want: dict) -> list:
    """Differences between two case digests (empty: they match)."""
    errs = []
    if set(got) != set(want):
        return [f"keys {sorted(got)} != {sorted(want)}"]
    for k, w in want.items():
        g = got[k]
        if not isinstance(w, dict):
            if g != w:
                errs.append(f"{k}: {g!r} != {w!r}")
            continue
        for f in ("n", "tokens_sha256", "n_tokens"):
            if g.get(f) != w.get(f):
                errs.append(f"{k}.{f}: {g.get(f)!r} != {w.get(f)!r}")
        if g["n"] == w["n"]:
            d = float(np.max(np.abs(np.subtract(g["frame_rms"], w["frame_rms"])), initial=0.0))
            if d > FRAME_RMS_ATOL:
                errs.append(f"{k}.frame_rms: max diff {d:.3g} > {FRAME_RMS_ATOL}")
    return errs


def _free():
    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def _case_mtl():
    mtl = _snapshot("chatterbox", "t3_mtl23ls_v2.safetensors", "conds.pt", "s3gen.pt")
    turbo = _snapshot("chatterbox-turbo", "s3gen_meanflow.safetensors")
    if mtl is None or turbo is None:
        return None
    from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

    m = ChatterboxMultilingualTTSMLX.from_local(
        mtl, device="mps", vocoder="meanflow", t3_dtype="float16", meanflow_path=turbo
    )
    gen = dict(exaggeration=0.3, cfg_weight=0.5, temperature=0.8, seed=11)
    toks = _recording(m, "_vocode_mlx")
    wav = m.generate(ES_TEXT, "es", show_progress=False, **gen)
    out = {
        "mtl_generate": _digest(wav, toks),
        "mtl_stream": _digest(np.concatenate(list(m.generate_stream(ES_TEXT, "es", **gen)))),
    }
    del m
    _free()
    return out


def _case_turbo(nano: bool):
    name = "chatterbox-nano" if nano else "chatterbox-turbo"
    t3 = "t3_nano_v1.safetensors" if nano else "t3_turbo_v1.safetensors"
    snap = _snapshot(name, t3, "conds.pt", "s3gen_meanflow.safetensors")
    if snap is None:
        return None
    from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX

    m = ChatterboxTurboMLX.from_local(snap, nano=nano)
    key = "nano" if nano else "turbo"
    toks = _recording(m, "_vocode")
    wav = m.generate(EN_TEXT, seed=11)
    out = {
        f"{key}_generate": _digest(wav, toks),
        f"{key}_stream": _digest(np.concatenate(list(m.generate_stream(EN_TEXT, seed=11)))),
    }
    del m
    _free()
    return out


def _case_flash():
    snap = _snapshot("chatterbox-flash", "t3_flash.safetensors", "s3gen.safetensors")
    voice = os.environ.get("CHATTERBOX_TEST_VOICE")
    if snap is None or not voice or not Path(voice).exists():
        return None
    from chatterbox.tts_flash_mlx import ChatterboxFlashMLX

    m = ChatterboxFlashMLX.from_local(snap)
    m.prepare_conditionals(voice)
    toks = _recording(m, "_vocode")
    out = {
        "flash_generate": _digest(m.generate(EN_TEXT, seed=11), toks),
        "voice": Path(voice).name,
    }
    del m
    _free()
    return out


CASES = {
    "mtl": _case_mtl,
    "turbo": lambda: _case_turbo(False),
    "nano": lambda: _case_turbo(True),
    "flash": _case_flash,
}


@unittest.skipUnless(GOLDEN.exists(), "no reference hashes")
class TestDefaultsMatch130(unittest.TestCase):
    """Default arguments give the same bytes as chatterbox-mlx 1.3.0."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text())
        env = _env()
        if env != cls.golden["env"]:
            raise unittest.SkipTest(f"references made on {cls.golden['env']}, not {env}")

    def _check(self, case):
        want = self.golden["cases"].get(case)
        if want is None:
            self.skipTest(f"no reference for {case}")
        got = CASES[case]()
        if got is None:
            self.skipTest(f"{case}: checkpoint (or CHATTERBOX_TEST_VOICE) not available")
        if "voice" in want and got.get("voice") != want["voice"]:
            self.skipTest(f"{case}: reference made with voice {want['voice']}")
        self.assertEqual(_same(got, want), [])

    def test_multilingual(self):
        self._check("mtl")

    def test_turbo(self):
        self._check("turbo")

    def test_nano(self):
        self._check("nano")

    def test_flash(self):
        self._check("flash")


def _write_golden(names):
    data = {"env": _env(), "cases": {}}
    if GOLDEN.exists():
        old = json.loads(GOLDEN.read_text())
        if old.get("env") == data["env"]:
            data["cases"] = old["cases"]
    for name in names or list(CASES):
        got = CASES[name]()
        print(name, got, flush=True)
        if got is not None:
            data["cases"][name] = got
    GOLDEN.parent.mkdir(exist_ok=True)
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--write-golden":
        _write_golden(sys.argv[2:])
    else:
        unittest.main()
