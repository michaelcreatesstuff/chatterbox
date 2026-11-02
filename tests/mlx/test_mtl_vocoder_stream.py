# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for the multilingual MLX pipeline's vocoder option, T3 precision option
and ``generate_stream`` (``ChatterboxMultilingualTTSMLX``), plus the shared
stream module.

No model weights: T3, the tokenizer and both vocoders are stubs. The vocoder
stubs map every token to a fixed waveform value, so a token's audio does not
depend on the window it is vocoded in: a correct window join then reproduces
the whole-chunk vocode exactly.
"""

import inspect
import re
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

import chatterbox.mtl_tts_mlx as M
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.streaming_mlx import S3GEN_SIL

SOS, EOS = 6561, 6562
SR = 24000
LONG_SENTENCE = (
    "This is one long sentence with many words so that the token stream "
    "runs long enough for several vocoder windows to happen in a row."
)
THREE_SENTENCES = (
    "The first sentence of this reply has ten words in it. "
    "The second sentence of this reply also has ten words. "
    "And the third sentence of this reply has ten words too."
)


def _split(text, lang="en"):
    """spaCy-free sentence splitter (keeps the tests offline and fast)."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


# --------------------------------------------------------------------------- #
#  Stubs
# --------------------------------------------------------------------------- #


class _HP:
    start_text_token = 255
    stop_text_token = 0
    max_speech_tokens = 1000
    speech_cond_prompt_len = 150


class _T3Stub:
    """Tokens depend only on the seed and the text length; EOS last."""

    tokens_per_text_token = 8

    def __init__(self):
        self.hp = _HP()
        self.calls = []

    def _ids(self, text_tokens, max_new_tokens, seed):
        n = min(
            max_new_tokens - 1, self.tokens_per_text_token * int(text_tokens.shape[1])
        )
        rng = np.random.default_rng(0 if seed is None else seed)
        return np.concatenate([rng.integers(0, 6561, n), [EOS]]).astype(np.int64)

    def inference(self, *, t3_cond, text_tokens, max_new_tokens, seed=None, **kw):
        self.calls.append(dict(max_new_tokens=max_new_tokens, seed=seed, **kw))
        return mx.array(self._ids(text_tokens, max_new_tokens, seed)[None])


class _StreamingT3Stub(_T3Stub):
    """Also has the fast loop's ``generate_stream`` (yields id blocks)."""

    def __init__(self):
        super().__init__()
        self.stream_state = []  # per call: {"done": bool, "closed": bool}

    def generate_stream(
        self, *, t3_cond, text_tokens, max_new_tokens, seed=None, block_tokens=4, **kw
    ):
        state = {"done": False, "closed": False}
        self.stream_state.append(state)
        ids = self._ids(text_tokens, max_new_tokens, seed)
        try:
            for i in range(0, len(ids), block_tokens):
                yield ids[i : i + block_tokens]
            state["done"] = True
        finally:
            state["closed"] = True


class _Tokenizer:
    def text_to_tokens(self, text, language_id=None):
        n = max(1, len(text.split()))
        return torch.arange(1, n + 1, dtype=torch.long)[None]


def _token_value(tokens):
    return ((np.asarray(tokens) % 97) / 97.0 - 0.5).astype(np.float32)


class _MLXVocoderStub:
    """MLX S3Gen stand-in: token ``t`` -> 960 samples of ``_token_value(t)``."""

    def __init__(self, meanflow=True):
        self.meanflow = meanflow
        self.calls = []  # (tokens, finalize, n_cfm_timesteps, kind)
        self.cache_ok = []
        fade = (np.cos(np.linspace(np.pi, 0, 480)) + 1) / 2
        self.trim_fade = mx.array(
            np.concatenate([np.zeros(480), fade]).astype(np.float32)
        )

    def flow_inference(
        self, speech_tokens, ref_dict, finalize=False, n_cfm_timesteps=None
    ):
        toks = np.array(speech_tokens).reshape(-1)
        self.calls.append((toks.copy(), finalize, n_cfm_timesteps, "flow"))
        if not finalize:
            toks = toks[: max(0, len(toks) - 3)]
        frames = np.repeat(_token_value(toks), 2)
        return mx.array(np.tile(frames[None, None], (1, 80, 1)))

    def hift_inference(self, speech_feat, cache_source=None):
        v = np.array(speech_feat)[0, 0]
        wav = np.repeat(v, 480)[None]
        if cache_source is not None and cache_source.shape[2] > 0:
            c = np.array(cache_source)
            self.cache_ok.append(np.allclose(c, wav[None][:, :, : c.shape[2]]))
        return mx.array(wav), mx.array(wav[None])

    def inference(self, speech_tokens, ref_dict, finalize=True, n_cfm_timesteps=None):
        mel = self.flow_inference(speech_tokens, ref_dict, finalize, n_cfm_timesteps)
        self.calls[-1] = self.calls[-1][:3] + ("inference",)
        wav, src = self.hift_inference(mel)
        wav = np.array(wav)
        n = min(wav.shape[1], 960)
        wav[:, :n] *= np.array(self.trim_fade)[:n]
        return mx.array(wav), src


class _PTS3GenStub:
    """PyTorch S3Gen stand-in (same token -> audio map, non-final drops 3)."""

    def __init__(self):
        self.calls = []  # (n_tokens, finalize, kwargs)
        self.cond_calls = 0

    def inference(self, speech_tokens, ref_dict, finalize=False, **kw):
        toks = speech_tokens.cpu().numpy().reshape(-1)
        self.calls.append((len(toks), finalize, dict(kw)))
        if not finalize:
            toks = toks[: max(0, len(toks) - 3)]
        wav = np.repeat(_token_value(toks), 960)[None]
        return torch.from_numpy(wav), None


def _conds():
    t3 = T3Cond(
        speaker_emb=torch.zeros(1, 256),
        cond_prompt_speech_tokens=torch.zeros(1, 10, dtype=torch.long),
        emotion_adv=0.5 * torch.ones(1, 1, 1),
    )
    gen = {
        "prompt_token": torch.zeros(1, 10, dtype=torch.long),
        "prompt_token_len": torch.tensor([10]),
        "prompt_feat": torch.zeros(1, 20, 80),
        "prompt_feat_len": None,
        "embedding": torch.zeros(1, 192),
    }
    return M.Conditionals(t3, gen)


def _tts(vocoder="meanflow", t3=None):
    s3gen_mlx = None if vocoder == "pt" else _MLXVocoderStub(vocoder == "meanflow")
    return M.ChatterboxMultilingualTTSMLX(
        t3=t3 or _T3Stub(),
        s3gen=_PTS3GenStub(),
        ve=None,
        tokenizer=_Tokenizer(),
        device="cpu",
        conds=_conds(),
        vocoder=vocoder,
        s3gen_mlx=s3gen_mlx,
    )


def _np(wav):
    return wav.numpy().reshape(-1) if hasattr(wav, "numpy") else np.asarray(wav)


class _Base(unittest.TestCase):
    def setUp(self):
        p1 = mock.patch.object(M, "split_into_sentences", _split)
        p2 = mock.patch.object(M, "SPACY_AVAILABLE", True)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)


# --------------------------------------------------------------------------- #
#  Vocoder option and defaults
# --------------------------------------------------------------------------- #


class TestDefaults(_Base):
    def test_loader_defaults_are_1_2_0(self):
        cls = M.ChatterboxMultilingualTTSMLX
        for fn in (cls.from_local, cls.from_pretrained, cls.__init__):
            p = inspect.signature(fn).parameters
            self.assertEqual(p["vocoder"].default, "pt")
        for fn in (cls.from_local, cls.from_pretrained):
            p = inspect.signature(fn).parameters
            self.assertEqual(p["t3_dtype"].default, "fp32")
            self.assertIsNone(p["quantize_bits"].default)
            self.assertIsNone(p["meanflow_path"].default)
        # Positional order of the 1.2.0 arguments is unchanged.
        self.assertEqual(
            list(inspect.signature(cls.from_local).parameters)[:5],
            ["ckpt_dir", "t3_config", "use_default_speaker", "device", "vocoder"],
        )

    def test_default_instance_uses_pytorch_s3gen(self):
        tts = M.ChatterboxMultilingualTTSMLX(
            t3=_T3Stub(),
            s3gen=_PTS3GenStub(),
            ve=None,
            tokenizer=_Tokenizer(),
            device="cpu",
            conds=_conds(),
        )
        self.assertEqual(tts.vocoder, "pt")
        self.assertIsNone(tts.s3gen_mlx)
        wav = _np(tts.generate("Hello there my friend.", "en", seed=1))
        # 1.2.0 call: finalize=False and no n_cfm_timesteps keyword.
        self.assertEqual(tts.s3gen.calls, [(len(wav) // 960 + 3, False, {})])
        tts.generate("Hello there my friend.", "en", seed=1, n_cfm_timesteps=3)
        self.assertEqual(tts.s3gen.calls[-1][2], {"n_cfm_timesteps": 3})

    def test_vocoder_validation(self):
        kw = dict(
            t3=_T3Stub(), s3gen=None, ve=None, tokenizer=_Tokenizer(), device="cpu"
        )
        with self.assertRaises(ValueError):
            M.ChatterboxMultilingualTTSMLX(**kw, vocoder="hifigan")
        with self.assertRaises(ValueError):
            M.ChatterboxMultilingualTTSMLX(**kw, vocoder="meanflow")  # no s3gen_mlx
        with self.assertRaises(ValueError):  # meanflow flag mismatch
            M.ChatterboxMultilingualTTSMLX(
                **kw, vocoder="mlx", s3gen_mlx=_MLXVocoderStub(meanflow=True)
            )
        with self.assertRaises(ValueError):
            M.ChatterboxMultilingualTTSMLX.from_local("/nonexistent", vocoder="x")

    def test_meanflow_path_resolution(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            with self.assertRaises(FileNotFoundError):
                M.resolve_meanflow_path(None, d)
            with self.assertRaises(FileNotFoundError):
                M.resolve_meanflow_path(d, None)
            # from_local fails before it loads anything.
            with self.assertRaises(FileNotFoundError):
                M.ChatterboxMultilingualTTSMLX.from_local(d, vocoder="meanflow")
            f = d / M.MEANFLOW_FILENAME
            f.write_bytes(b"")
            self.assertEqual(M.resolve_meanflow_path(None, d), f)
            self.assertEqual(M.resolve_meanflow_path(d, None), f)
            self.assertEqual(M.resolve_meanflow_path(str(f), None), f)

    def test_meanflow_vocode_as_turbo(self):
        tts = _tts("meanflow")
        wav = _np(tts.generate("Hello there my friend.", "en", seed=4))
        toks, finalize, steps, kind = tts.s3gen_mlx.calls[-1]
        self.assertEqual(kind, "inference")
        self.assertTrue(finalize)
        self.assertEqual(steps, 2)
        np.testing.assert_array_equal(toks[-3:], [S3GEN_SIL] * 3)
        self.assertTrue((toks[:-3] < 6561).all())
        self.assertEqual(len(wav), 960 * len(toks))
        self.assertEqual(tts.s3gen.calls, [])  # PyTorch vocoder unused
        tts.generate("Hello there my friend.", "en", seed=4, n_cfm_timesteps=1)
        self.assertEqual(tts.s3gen_mlx.calls[-1][2], 1)

    def test_mlx_vocode_as_pytorch_path(self):
        tts = _tts("mlx")
        pt = _tts("pt")
        a = _np(tts.generate("Hello there my friend.", "en", seed=4))
        b = _np(pt.generate("Hello there my friend.", "en", seed=4))
        toks, finalize, steps, _ = tts.s3gen_mlx.calls[-1]
        self.assertFalse(finalize)
        self.assertEqual(steps, 5)
        self.assertNotIn(S3GEN_SIL, toks[-3:].tolist())
        # Same tokens, same end handling as the PyTorch path (the stubs
        # differ only by the MLX trim fade on the first 40 ms).
        self.assertEqual(len(a), len(b))
        np.testing.assert_allclose(a[960:], b[960:], atol=1e-6)

    def test_mlx_ref_dict_follows_conds(self):
        tts = _tts("meanflow")
        r1 = tts._mlx_ref_dict()
        self.assertIs(tts._mlx_ref_dict(), r1)
        tts.conds = _conds()  # what a voice cache does
        self.assertIsNot(tts._mlx_ref_dict(), r1)


# --------------------------------------------------------------------------- #
#  T3 precision option
# --------------------------------------------------------------------------- #


class _TinyT3(nn.Module):
    def __init__(self):
        super().__init__()
        self.tfmr = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64))
        self.speech_head = nn.Linear(64, 32)


class TestT3Precision(unittest.TestCase):
    def test_fp32_untouched(self):
        m = _TinyT3()
        w = m.tfmr.layers[0].weight
        self.assertIs(M.apply_t3_precision(m, "fp32", None), m)
        self.assertIs(m.tfmr.layers[0].weight, w)
        self.assertEqual(w.dtype, mx.float32)

    def test_fallback_cast_and_quantize(self):
        m = M.apply_t3_precision(_TinyT3(), "fp16")
        self.assertEqual(m.tfmr.layers[0].weight.dtype, mx.float16)
        m = M.apply_t3_precision(_TinyT3(), "fp16", quantize_bits=8)
        self.assertIsInstance(m.tfmr.layers[0], nn.QuantizedLinear)
        self.assertIsInstance(m.speech_head, nn.Linear)

    def test_uses_t3_finalize_when_present(self):
        class WithFinalize(_TinyT3):
            def finalize(self, dtype, quantize_bits=None):
                self.finalized = (dtype, quantize_bits)

        m = WithFinalize()
        M.apply_t3_precision(m, "fp16", quantize_bits=4)
        self.assertEqual(m.finalized, ("float16", 4))
        self.assertEqual(m.tfmr.layers[0].weight.dtype, mx.float32)

    def test_bad_values(self):
        with self.assertRaises(ValueError):
            M.apply_t3_precision(_TinyT3(), "bf8")
        with self.assertRaises(ValueError):
            M.apply_t3_precision(_TinyT3(), "fp16", quantize_bits=3)


# --------------------------------------------------------------------------- #
#  generate_stream
# --------------------------------------------------------------------------- #


class TestStream(_Base):
    def test_windows_match_generate_single_chunk(self):
        for t3 in (_T3Stub(), _StreamingT3Stub()):
            tts = _tts("meanflow", t3=t3)
            ref = _np(tts.generate(LONG_SENTENCE, "en", seed=3))
            tts.s3gen_mlx.calls.clear()
            pieces = list(
                tts.generate_stream(
                    LONG_SENTENCE,
                    "en",
                    seed=3,
                    first_chunk_tokens=25,
                    chunk_tokens=20,
                    context_tokens=10,
                )
            )
            t = tts.last_timings
            self.assertEqual(t["n_chunks"], 1)
            self.assertGreater(len(pieces), 2)
            self.assertGreater(t["n_windows"], 2)
            self.assertEqual(t["n_pieces"], len(pieces))
            self.assertIsNotNone(t["first_audio_s"])
            self.assertEqual(t["vocoder"], "meanflow")
            self.assertTrue(all(p.dtype == np.float32 and p.ndim == 1 for p in pieces))
            got = np.concatenate(pieces)
            self.assertEqual(len(got), len(ref))
            np.testing.assert_allclose(got, ref, atol=1e-6)
            finals = [c[1] for c in tts.s3gen_mlx.calls]
            self.assertTrue(finals[-1])
            self.assertFalse(any(finals[:-1]))
            self.assertTrue(tts.s3gen_mlx.cache_ok and all(tts.s3gen_mlx.cache_ok))
            self.assertTrue(all(c[2] == 2 for c in tts.s3gen_mlx.calls))

    def test_multi_chunk_matches_generate(self):
        for vocoder, first in (("meanflow", 25), ("meanflow", 0), ("pt", 25)):
            tts = _tts(vocoder)
            ref = _np(tts.generate(THREE_SENTENCES, "en", seed=7))
            pieces = list(
                tts.generate_stream(
                    THREE_SENTENCES, "en", seed=7, first_chunk_tokens=first
                )
            )
            got = np.concatenate(pieces)
            self.assertEqual(tts.last_timings["n_chunks"], 3)
            self.assertGreaterEqual(len(pieces), 3)
            self.assertEqual(len(got), len(ref), (vocoder, first))
            np.testing.assert_allclose(got, ref, atol=1e-5)

    def test_mlx_vocoder_streams_with_turbo_end_handling(self):
        tts = _tts("mlx")
        pieces = list(tts.generate_stream(LONG_SENTENCE, "en", seed=2))
        self.assertGreater(tts.last_timings["n_windows"], 1)
        toks, finalize, steps, _ = tts.s3gen_mlx.calls[-1]
        self.assertTrue(finalize)
        self.assertEqual(steps, 5)
        np.testing.assert_array_equal(toks[-3:], [S3GEN_SIL] * 3)
        # Same audio as one final-mode vocode of the tokens + 3 silence tokens.
        ids = tts.t3._ids(mx.zeros((2, len(LONG_SENTENCE.split()) + 2)), 10**6, 2)
        ids = np.concatenate([ids[ids < 6561], [S3GEN_SIL] * 3])
        ref = _np(tts._vocoder_host()._vocode(ids, 5, 2, finalize=True))
        np.testing.assert_allclose(np.concatenate(pieces), ref, atol=1e-6)

    def test_short_chunk_is_one_vocode(self):
        tts = _tts("meanflow")
        tts.t3.tokens_per_text_token = 2
        ref = _np(tts.generate("Hi there.", "en", seed=0))
        tts.s3gen_mlx.calls.clear()
        pieces = list(tts.generate_stream("Hi there.", "en", seed=0))
        self.assertEqual(len(pieces), 1)
        np.testing.assert_array_equal(pieces[0], ref)
        self.assertEqual(tts.last_timings["n_windows"], 1)
        self.assertEqual(len(tts.s3gen_mlx.calls), 1)

    def test_first_piece_before_t3_finishes(self):
        tts = _tts("meanflow", t3=_StreamingT3Stub())
        gen = tts.generate_stream(LONG_SENTENCE, "en", seed=0)
        first = next(gen)
        self.assertGreater(len(first), 0)
        self.assertFalse(tts.t3.stream_state[0]["done"])
        rest = list(gen)
        self.assertTrue(tts.t3.stream_state[0]["done"])
        self.assertGreater(len(rest), 0)

    def test_seed_determinism(self):
        tts = _tts("meanflow", t3=_StreamingT3Stub())

        def run(seed):
            return np.concatenate(list(tts.generate_stream(THREE_SENTENCES, "en", seed=seed)))

        a, b, c = run(5), run(5), run(6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(len(a) == len(c) and np.array_equal(a, c))

    def test_close_early_releases_lock(self):
        for vocoder in ("meanflow", "pt"):
            tts = _tts(vocoder, t3=_StreamingT3Stub())
            gen = tts.generate_stream(THREE_SENTENCES, "en", seed=0)
            next(gen)
            self.assertEqual(tts._stream_active, 1)
            # The same thread may not start another call on this instance ...
            with self.assertRaises(RuntimeError):
                tts.generate("again", "en")
            with self.assertRaises(RuntimeError):
                tts.generate_long("again", "en")
            with self.assertRaises(RuntimeError):
                tts.prepare_conditionals("unused.wav")
            with self.assertRaises(RuntimeError):
                next(tts.generate_stream("again", "en"))
            # ... and another thread waits for the lock.
            got = []
            th = threading.Thread(target=lambda: got.append(tts._lock.acquire(timeout=0.2)))
            th.start()
            th.join()
            self.assertEqual(got, [False])

            gen.close()
            self.assertEqual(tts._stream_active, 0)
            if vocoder == "meanflow":
                self.assertTrue(tts.t3.stream_state[0]["closed"])
                self.assertFalse(tts.t3.stream_state[0]["done"])
            got = []

            def other_thread():
                ok = tts._lock.acquire(timeout=1.0)
                got.append(ok)
                if ok:
                    tts._lock.release()

            th = threading.Thread(target=other_thread)
            th.start()
            th.join()
            self.assertEqual(got, [True])
            self.assertGreater(len(_np(tts.generate("Hello again my friend.", "en", seed=1))), 0)
            self.assertGreater(len(list(tts.generate_stream("Hello again my friend.", "en", seed=1))), 0)

    def test_no_watermark(self):
        tts = _tts("meanflow")
        with mock.patch.object(tts.watermarker, "apply_watermark") as wm:
            list(tts.generate_stream(THREE_SENTENCES, "en", seed=1))
            tts.generate(THREE_SENTENCES, "en", seed=1)
        wm.assert_not_called()

    def test_empty_and_bad_args(self):
        tts = _tts("meanflow")
        self.assertEqual(list(tts.generate_stream("  ", "en")), [])
        self.assertEqual(tts.last_timings, {"n_chunks": 0})
        with self.assertRaises(ValueError):
            list(tts.generate_stream("hi", "en", lookahead_tokens=2))
        with self.assertRaises(ValueError):
            list(tts.generate_stream("hi", "en", first_chunk_tokens=1))
        with self.assertRaises(ValueError):
            list(tts.generate_stream("hi", "xx"))
        self.assertEqual(tts._stream_active, 0)


class TestSharedModule(unittest.TestCase):
    def test_turbo_reexports_shared_names(self):
        import chatterbox.streaming_mlx as S
        import chatterbox.tts_turbo_mlx as T

        for name in ("_WindowVocoder", "_ChunkJoiner", "ref_dict_to_mlx", "S3GEN_SIL"):
            self.assertIs(getattr(T, name), getattr(S, name))


if __name__ == "__main__":
    unittest.main()
