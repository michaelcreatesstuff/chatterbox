# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for streaming on the MLX Turbo / Nano pipeline
(``T3TurboMLX.generate_stream`` and ``ChatterboxTurboMLX.generate_stream``).

Tiny random T3 weights only. The vocoder stub maps every token to a fixed
waveform value, so a token's audio does not depend on the window it is
vocoded in: a correct window join then reproduces the whole-chunk vocode
exactly.
"""

import unittest

import mlx.core as mx
import numpy as np
import torch

from chatterbox.generation_utils import crossfade_chunks
from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from chatterbox.models.t3_turbo_mlx import (
    GPT2_SMALL_CONFIG,
    T3TurboMLX,
    TurboT3Config,
)

TINY_CONFIG = dict(GPT2_SMALL_CONFIG)
TINY_CONFIG.update(n_embd=64, hidden_size=64, n_head=4, n_layer=2, n_positions=2048)
SOS, EOS = 6561, 6562
LONG_TEXT = "This sentence has exactly ten words in it for testing. " * 9


def tiny_t3(seed=0, eos_bias=-1e9):
    """Random tiny T3. ``eos_bias=-1e9`` never stops (rows run to their cap)."""
    mx.random.seed(seed)
    m = T3TurboMLX(hp=TurboT3Config(gpt2_config=TINY_CONFIG))
    m.attn_bucket = 32
    bias = np.array(m.speech_head.bias)
    bias[SOS] = -1e9
    bias[EOS] = eos_bias
    m.speech_head.bias = mx.array(bias)
    m.finalize("float32")
    return m


def tiny_cond(seed=0):
    g = torch.Generator().manual_seed(1000 + seed)
    spk = torch.randn(1, 256, generator=g)
    prompt = torch.randint(0, 6561, (1, 375), generator=g)
    return spk, prompt


def text_ids(n):
    return mx.array((np.arange(n)[None] * 37 % 50000 + 11).astype(np.int32))


# --------------------------------------------------------------------------- #
#  Stubs
# --------------------------------------------------------------------------- #


class _Tok:
    def __call__(self, text, return_tensors=None, padding=None, truncation=None):
        n = max(1, len(text.split()))
        return {"input_ids": (np.arange(n)[None] * 31 % 5000 + 7).astype(np.int64)}


def _token_value(tokens):
    return ((np.asarray(tokens) % 97) / 97.0 - 0.5).astype(np.float32)


class _S3GenStub:
    """Token ``t`` -> 960 samples of ``_token_value(t)`` (2 mel frames x 480)."""

    def __init__(self):
        self.calls = []  # (n_tokens, finalize)
        self.cache_ok = []
        fade = (np.cos(np.linspace(np.pi, 0, 480)) + 1) / 2
        self.trim_fade = mx.array(np.concatenate([np.zeros(480), fade]).astype(np.float32))

    def flow_inference(self, speech_tokens, ref_dict, finalize=False, n_cfm_timesteps=None):
        toks = np.array(speech_tokens).reshape(-1)
        self.calls.append((len(toks), finalize))
        if not finalize:
            toks = toks[: max(0, len(toks) - 3)]
        frames = np.repeat(_token_value(toks), 2)
        return mx.array(np.tile(frames[None, None], (1, 80, 1)))

    def hift_inference(self, speech_feat, cache_source=None):
        v = np.array(speech_feat)[0, 0]
        wav = np.repeat(v, 480)[None]
        src = mx.array(wav[None])
        if cache_source is not None and cache_source.shape[2] > 0:
            c = np.array(cache_source)
            self.cache_ok.append(np.allclose(c, wav[None][:, :, : c.shape[2]]))
        return mx.array(wav), src

    def inference(self, speech_tokens, ref_dict, finalize=True, n_cfm_timesteps=None):
        mel = self.flow_inference(speech_tokens, ref_dict, finalize, n_cfm_timesteps)
        wav, src = self.hift_inference(mel)
        wav = np.array(wav)
        wav[:, :960] *= np.array(self.trim_fade)
        return mx.array(wav), src


class _Watermarker:
    def __init__(self):
        self.calls = 0

    def apply_watermark(self, wav, sample_rate):
        self.calls += 1
        return wav * 0.5


def _conds():
    spk, prompt = tiny_cond()
    return {
        "t3": {
            "speaker_emb": spk,
            "clap_emb": None,
            "cond_prompt_speech_tokens": prompt,
            "cond_prompt_speech_emb": None,
            "emotion_adv": torch.ones(1, 1, 1),
        },
        "gen": {
            "prompt_token": torch.zeros(1, 10, dtype=torch.long),
            "prompt_token_len": torch.tensor([10]),
            "prompt_feat": torch.zeros(1, 20, 80),
            "prompt_feat_len": None,
            "embedding": torch.zeros(1, 192),
        },
    }


def _tts(eos_bias=-1e9):
    from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX

    tts = ChatterboxTurboMLX(
        tiny_t3(seed=4, eos_bias=eos_bias),
        _S3GenStub(),
        ve=None,
        tokenizer=_Tok(),
        conds=_conds(),
    )
    tts._watermarker = _Watermarker()
    return tts


# --------------------------------------------------------------------------- #
#  T3
# --------------------------------------------------------------------------- #


class TestT3Stream(unittest.TestCase):
    def setUp(self):
        spk, prompt = tiny_cond()
        self.cond = T3CondMLX(
            speaker_emb=mx.array(spk.numpy()),
            cond_prompt_speech_tokens=mx.array(prompt.numpy().astype(np.int32)),
        )

    def test_blocks_concatenate_to_generate(self):
        # eos_bias 6: rows stop at random points, which exercises the stop trim.
        for eos_bias in (-1e9, 6.0):
            m = tiny_t3(seed=1, eos_bias=eos_bias)
            for seed in range(4):
                ref = m.generate(self.cond, text_ids(7), seed=seed, max_gen_len=40)
                stats = {}
                blocks = list(
                    m.generate_stream(
                        self.cond,
                        text_ids(7),
                        seed=seed,
                        max_gen_len=40,
                        block_tokens=4,
                        stats=stats,
                    )
                )
                got = np.concatenate(blocks) if blocks else np.zeros(0, np.int64)
                np.testing.assert_array_equal(got, ref)
                self.assertTrue(all(len(b) for b in blocks))
                if eos_bias < 0:
                    self.assertEqual(len(ref), 41)
                    self.assertGreater(len(blocks), 5)
                    self.assertEqual(stats["n_steps"], 40)

    def test_first_block_before_decode_ends(self):
        m = tiny_t3(seed=1)
        gen = m.generate_stream(self.cond, text_ids(5), seed=0, max_gen_len=60, block_tokens=4, stats=(stats := {}))
        first = next(gen)
        self.assertGreaterEqual(len(first), 1)
        self.assertLess(stats["n_steps"], 60)
        gen.close()
        # An abandoned stream leaves the model usable.
        out = m.generate(self.cond, text_ids(5), seed=0, max_gen_len=60)
        self.assertEqual(len(out), 61)


# --------------------------------------------------------------------------- #
#  Pipeline helpers
# --------------------------------------------------------------------------- #


class TestChunkJoiner(unittest.TestCase):
    def test_matches_crossfade_chunks(self):
        from chatterbox.tts_turbo_mlx import _ChunkJoiner

        rng = np.random.default_rng(0)
        for n_chunks in (1, 2, 3, 5):
            chunks = [rng.standard_normal(rng.integers(3000, 9000)).astype(np.float32) for _ in range(n_chunks)]
            ref = crossfade_chunks([torch.from_numpy(c) for c in chunks], 24000, 0.05)
            ref = ref.numpy() if hasattr(ref, "numpy") else np.asarray(ref)
            joiner = _ChunkJoiner(n_chunks, 24000, 0.05)
            parts = []
            for ci, c in enumerate(chunks):
                joiner.begin_chunk()
                cuts = np.sort(rng.integers(0, len(c), 3))
                for j, p in enumerate(np.split(c, cuts)):
                    parts.append(joiner.push(p))
                    if ci == n_chunks - 1 and j == len(cuts):
                        parts.append(joiner.finish())
            got = np.concatenate(parts)
            self.assertEqual(len(got), len(ref))
            np.testing.assert_allclose(got, ref, atol=1e-5)


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #


class TestPipelineStream(unittest.TestCase):
    def test_single_chunk_windows_match_generate(self):
        tts = _tts()
        ref = tts.generate("hello there", seed=3, max_gen_len=150, watermark=False)
        tts.s3gen.calls.clear()
        pieces = list(
            tts.generate_stream(
                "hello there",
                seed=3,
                max_gen_len=150,
                watermark=False,
                first_chunk_tokens=25,
                chunk_tokens=20,
                context_tokens=10,
            )
        )
        t = tts.last_timings
        self.assertGreater(len(pieces), 2)
        self.assertGreater(t["n_windows"], 2)
        self.assertEqual(t["n_pieces"], len(pieces))
        self.assertIsNotNone(t["first_audio_s"])
        self.assertTrue(all(p.dtype == np.float32 and p.ndim == 1 for p in pieces))
        got = np.concatenate(pieces)
        self.assertEqual(len(got), len(ref))
        np.testing.assert_allclose(got, ref, atol=1e-6)
        # Windows: non-final S3Gen calls first, one final call last; every
        # HiFiGAN source cache lines up with the window it replaces.
        finals = [f for _, f in tts.s3gen.calls]
        self.assertEqual(finals[-1], True)
        self.assertFalse(any(finals[:-1]))
        self.assertTrue(tts.s3gen.cache_ok and all(tts.s3gen.cache_ok))
        # A window re-vocodes at most context + its new tokens + lookahead.
        self.assertLess(max(n for n, _ in tts.s3gen.calls[:-1]), 151)

    def test_short_chunk_is_one_vocode(self):
        tts = _tts()
        ref = tts.generate("hi", seed=0, max_gen_len=12, watermark=False)
        pieces = list(tts.generate_stream("hi", seed=0, max_gen_len=12, watermark=False))
        self.assertEqual(len(pieces), 1)
        np.testing.assert_array_equal(pieces[0], ref)
        self.assertEqual(tts.last_timings["n_windows"], 1)

    def test_multi_chunk_matches_generate(self):
        tts = _tts()
        kw = dict(seed=1, max_gen_len=60, watermark=False, max_batch_chunks=1)
        ref = tts.generate(LONG_TEXT, **kw)
        n_chunks = tts.last_timings["n_chunks"]
        self.assertGreater(n_chunks, 2)
        for first in (20, 0):
            pieces = list(tts.generate_stream(LONG_TEXT, first_chunk_tokens=first, **kw))
            got = np.concatenate(pieces)
            self.assertEqual(len(got), len(ref))
            np.testing.assert_allclose(got, ref, atol=1e-5)
            t = tts.last_timings
            self.assertEqual(t["n_chunks"], n_chunks)
            self.assertEqual(t["batch_sizes"], [1] * n_chunks)
            self.assertGreaterEqual(len(pieces), n_chunks)

    def test_rest_chunks_batched(self):
        tts = _tts()
        pieces = list(tts.generate_stream(LONG_TEXT, seed=1, max_gen_len=30, watermark=False, max_batch_chunks=2))
        t = tts.last_timings
        n = t["n_chunks"]
        self.assertEqual(t["batch_sizes"][0], 1)  # chunk 0 alone
        self.assertEqual(t["n_groups"], 1 + -(-(n - 1) // 2))
        self.assertTrue(all(b == 2 for b in t["batch_sizes"][1:]))
        self.assertGreater(sum(len(p) for p in pieces), 0)

    def test_first_piece_before_t3_finishes(self):
        tts = _tts()
        orig = tts.t3.generate_stream
        state = {"done": False}

        def spy(*a, **kw):
            yield from orig(*a, **kw)
            state["done"] = True

        tts.t3.generate_stream = spy
        gen = tts.generate_stream("hello there", seed=0, max_gen_len=200, watermark=False)
        first = next(gen)
        self.assertGreater(len(first), 0)
        self.assertFalse(state["done"])
        rest = list(gen)
        self.assertTrue(state["done"])
        self.assertGreater(len(rest), 0)

    def test_seed_determinism(self):
        tts = _tts()

        def run(seed):
            return np.concatenate(list(tts.generate_stream(LONG_TEXT, seed=seed, max_gen_len=40, watermark=False)))

        a, b, c = run(5), run(5), run(6)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(len(a) == len(c) and np.array_equal(a, c))

    def test_watermark_per_piece(self):
        tts = _tts()
        plain = list(tts.generate_stream("hello there", seed=2, max_gen_len=120, watermark=False))
        tts._watermarker.calls = 0
        marked = list(tts.generate_stream("hello there", seed=2, max_gen_len=120))
        self.assertEqual(len(marked), len(plain))
        self.assertEqual(tts._watermarker.calls, len(marked))
        for p, m in zip(plain, marked):
            self.assertEqual(len(p), len(m))
            np.testing.assert_allclose(m, 0.5 * p, atol=1e-6)

    def test_lock_guard_and_close(self):
        tts = _tts()
        gen = tts.generate_stream("hello there", seed=0, max_gen_len=120, watermark=False)
        next(gen)
        with self.assertRaises(RuntimeError):
            tts.generate("again", max_gen_len=5)
        with self.assertRaises(RuntimeError):
            tts.release_caches()
        with self.assertRaises(RuntimeError):
            next(tts.generate_stream("again", max_gen_len=5))
        gen.close()
        self.assertEqual(tts._stream_active, 0)
        self.assertGreater(len(tts.generate("again", max_gen_len=5, watermark=False)), 0)

    def test_empty_and_bad_args(self):
        tts = _tts()
        self.assertEqual(list(tts.generate_stream("  ")), [])
        self.assertEqual(tts.last_timings, {"n_chunks": 0})
        with self.assertRaises(ValueError):
            list(tts.generate_stream("hi", lookahead_tokens=2))
        with self.assertRaises(ValueError):
            list(tts.generate_stream("hi", first_chunk_tokens=1))


if __name__ == "__main__":
    unittest.main()
