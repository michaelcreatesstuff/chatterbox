# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for the fast T3MLX decode loop (``T3MLX.generate(fast=True)``).

- AlignmentStreamAnalyzerNP keeps the rules of AlignmentStreamAnalyzerMLX
  (same decisions on the same attention rows, and each rule on its own).
- LlamaModelMLX.forward_fast matches the manual-attention forward
  (hidden states and the aligned-head weights), across KV-cache growth.
- The fast loop and the 1.2.0 loop sample the same tokens with the same seed
  (tiny random Llama; fp32).
- generate() leaves the MLX cache limit as it found it (both loops).
- finalize() casts / quantizes and the fast loop runs on the result.

No real weights are loaded by default. ``TestRealWeights`` (opt-in,
CHATTERBOX_T3_REAL=1) compares both loops on the real multilingual T3.
"""

import os
import unittest

import mlx.core as mx
import numpy as np

from chatterbox.models.t3.llama_configs import LLAMA_CONFIGS, LLAMA_520M_CONFIG_DICT
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3_mlx.inference.alignment_stream_analyzer_mlx import (
    LLAMA_ALIGNED_HEADS,
    AlignmentStreamAnalyzerMLX,
    AlignmentStreamAnalyzerNP,
)
from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from chatterbox.models.t3_mlx.modules.llama_mlx import (
    LlamaConfigMLX,
    LlamaKVCache,
    LlamaModelMLX,
)
from chatterbox.models.t3_mlx.t3_mlx import T3MLX

# Enough layers for LLAMA_ALIGNED_HEADS (layer 13) and the real width (the
# perceiver is fixed at 1024); a narrow MLP keeps it small (~75M parameters).
TINY_NAME = "Llama_tiny_fast_test"
TINY_CONFIG = dict(LLAMA_520M_CONFIG_DICT)
TINY_CONFIG.update(intermediate_size=128, num_hidden_layers=14)
LLAMA_CONFIGS.setdefault(TINY_NAME, TINY_CONFIG)

EOS_IDX = 7


def _analyzer_pair(i, j):
    return (
        AlignmentStreamAnalyzerMLX(text_tokens_slice=(i, j), eos_idx=EOS_IDX),
        AlignmentStreamAnalyzerNP((i, j)),
    )


def _mlx_decision(analyzer, rows, next_token, vocab=16):
    """Run the MLX analyzer on ``rows`` and read back (suppress, force) from the logits."""
    # The MLX analyzer slices [j:, i:j] at frame 0 out of a (T, T) map;
    # embed the rows at the bottom of a square map.
    n, Lk = rows.shape
    full = np.zeros((Lk, Lk), dtype=np.float32)
    full[Lk - n :, :] = rows
    attn = [mx.array(full)[None, None]]  # one "layer", one "head"
    for b in range(len(analyzer.last_aligned_attns)):
        analyzer.update_attention(attn, buffer_idx=b, layer_idx=0, head_idx=0)
    logits = analyzer.step(mx.zeros((1, vocab)), next_token=next_token)
    logits = np.array(logits)[0]
    force = logits[EOS_IDX] == 32768.0 and np.all(np.delete(logits, EOS_IDX) == -32768.0)
    suppress = (not force) and logits[EOS_IDX] == -32768.0
    return bool(suppress), bool(force)


def _np_decision(analyzer, rows, next_token):
    suppress, force = analyzer.step(rows, next_token=next_token)
    return (suppress and not force), force


class TestAnalyzerNP(unittest.TestCase):
    """The numpy analyzer makes the same decisions as the MLX analyzer."""

    def _script(self, S, frames, rng, tokens):
        """Prefill rows (2 query rows from j) then single rows; a diagonal walk with noise."""
        i, j = 10, 10 + S
        rows_list = []
        for t in range(frames):
            n = 2 if t == 0 else 1
            Lk = j + 1 + t + (1 if t == 0 else 0)
            r = rng.random((n, Lk)).astype(np.float32) * 0.05
            pos = min(S - 1, t // 2)
            r[:, i + pos] += 0.6 if t < 2 * S + 4 else 0.02
            r /= r.sum(axis=1, keepdims=True)
            rows_list.append(r)
        return i, j, rows_list, tokens[:frames]

    def _compare(self, S, frames, seed, tokens):
        rng = np.random.default_rng(seed)
        i, j, rows_list, toks = self._script(S, frames, rng, tokens)
        ref, ours = _analyzer_pair(i, j)
        for t, rows in enumerate(rows_list):
            last = toks[t - 1] if t > 0 else None
            # rows for the MLX path must cover positions [j:]; frame 0 has 2 rows
            self.assertEqual(
                _mlx_decision(ref, rows, last), _np_decision(ours, rows, last), f"frame {t}"
            )
            self.assertEqual(ref.text_position, ours.text_position)
            self.assertEqual(ref.complete, ours.complete)
            self.assertEqual(ref.started, ours.started)
        np.testing.assert_allclose(np.array(ref.alignment), ours.alignment, rtol=1e-6)

    def test_matches_mlx_analyzer_random_walks(self):
        rng = np.random.default_rng(0)
        for seed in range(6):
            S = int(rng.integers(4, 30))
            tokens = list(rng.integers(0, 6, size=200))  # small alphabet: repetitions happen
            self._compare(S, frames=3 * S + 10, seed=seed, tokens=tokens)

    def test_matches_mlx_analyzer_distinct_tokens(self):
        for seed in range(4):
            self._compare(S=12, frames=50, seed=100 + seed, tokens=list(range(100, 200)))

    # Rules one by one (numpy analyzer)

    def _row(self, S, pos, peak=0.9, i=0, extra=1):
        r = np.full((1, i + S + extra), (1 - peak) / (S + extra), dtype=np.float32)
        r[0, i + pos] = peak
        return r

    def test_suppress_eos_before_completion(self):
        S = 20
        a = AlignmentStreamAnalyzerNP((0, S))
        prefill = np.concatenate([self._row(S, 0), self._row(S, 0)])
        self.assertEqual(a.step(prefill), (True, False))
        self.assertEqual(a.step(self._row(S, 1), next_token=1), (True, False))

    def test_short_text_never_suppresses(self):
        S = 5
        a = AlignmentStreamAnalyzerNP((0, S))
        self.assertEqual(a.step(np.concatenate([self._row(S, 0)] * 2)), (False, False))

    def test_long_tail_forces_eos(self):
        S = 10
        a = AlignmentStreamAnalyzerNP((0, S))
        a.step(np.concatenate([self._row(S, 0)] * 2))
        for p in range(1, S):
            a.step(self._row(S, p), next_token=100 + p)
        self.assertTrue(a.complete)
        forced = False
        for k in range(10):  # stay on the last text token
            _, forced = a.step(self._row(S, S - 1), next_token=200 + k)
            if forced:
                break
        self.assertTrue(forced)
        self.assertTrue(a.forced["long_tail"])

    def test_token_repetition_before_completion(self):
        S = 30
        a = AlignmentStreamAnalyzerNP((0, S))
        a.step(np.concatenate([self._row(S, 0)] * 2))
        toks = [5, 6, 5, 7, 5, 8, 5]  # 4x token 5 in the last 7
        force = False
        for k, t in enumerate(toks):
            _, force = a.step(self._row(S, min(k + 1, 3)), next_token=t)
        self.assertTrue(force)
        self.assertTrue(a.forced["token_repetition"])
        self.assertFalse(a.complete)

    def test_attention_collapse_after_completion(self):
        S = 10
        a = AlignmentStreamAnalyzerNP((0, S))
        a.step(np.concatenate([self._row(S, 0)] * 2))
        for p in range(1, S - 2):
            a.step(self._row(S, p), next_token=100 + p)
        self.assertTrue(a.complete)
        flat = np.full((1, S + 1), 0.05, dtype=np.float32)
        flat[0, S - 3] = 0.1  # peak below the threshold, position unchanged
        results = [a.step(flat, next_token=300 + k)[1] for k in range(3)]
        self.assertEqual(results, [False, False, True])
        self.assertTrue(a.forced["attention_collapse"])


def _tiny_t3(seed=0):
    mx.random.seed(seed)
    hp = T3Config.multilingual()
    hp.llama_config_name = TINY_NAME
    t3 = T3MLX(hp=hp)
    t3.eval()
    mx.eval(t3.parameters())
    return t3


def _tiny_inputs(t3, n_text=12, cfg=True, seed=1):
    rng = np.random.default_rng(seed)
    hp = t3.hp
    body = rng.integers(1, 200, size=n_text)
    row = np.concatenate([[hp.start_text_token], body, [hp.stop_text_token]])
    text = np.stack([row, row]) if cfg else row[None]
    cond = T3CondMLX(
        speaker_emb=mx.array(rng.standard_normal((1, 256)).astype(np.float32)),
        cond_prompt_speech_tokens=mx.array(
            rng.integers(0, 6000, size=(1, hp.speech_cond_prompt_len))
        ),
        emotion_adv=0.5,
    )
    return cond, mx.array(text.astype(np.int32))


GEN_KW = dict(
    temperature=0.8,
    top_p=1.0,
    min_p=0.05,
    repetition_penalty=2.0,
    cfg_weight=0.5,
    show_progress=False,
)


class TestForwardFast(unittest.TestCase):
    """forward_fast == the manual-attention forward (fp32)."""

    def test_hidden_and_aligned_weights_match(self):
        mx.random.seed(3)
        cfg = LlamaConfigMLX.from_dict(TINY_CONFIG)
        model = LlamaModelMLX(cfg)
        mx.eval(model.parameters())
        heads = {}
        for layer_idx, head_idx in LLAMA_ALIGNED_HEADS:
            heads.setdefault(layer_idx, []).append(head_idx)

        B, L0, D = 2, 9, cfg.hidden_size
        x0 = mx.random.normal((B, L0, D))
        rows_from = 6
        # step=4 forces two cache reallocations over the run
        cache = LlamaKVCache(cfg.num_hidden_layers, step=4)
        h_fast, w_fast = model.forward_fast(x0, cache, aligned_heads=heads, rows_from=rows_from)
        ref = model(x0, cache=None, output_attentions=True)
        np.testing.assert_allclose(
            np.array(h_fast[:, -1]), np.array(ref["last_hidden_state"][:, -1]), atol=1e-4
        )
        ref_rows = mx.stack(
            [ref["attentions"][li][0, h, rows_from:, :] for li, h in LLAMA_ALIGNED_HEADS]
        ).mean(axis=0)
        np.testing.assert_allclose(
            np.array(mx.stack(w_fast).mean(axis=0)), np.array(ref_rows), atol=1e-5
        )

        ref_cache = ref["cache"]
        for t in range(6):
            x = mx.random.normal((B, 1, D))
            h_fast, w_fast = model.forward_fast(x, cache, aligned_heads=heads, rows_from=0)
            ref = model(x, cache=ref_cache, output_attentions=True)
            ref_cache = ref["cache"]
            np.testing.assert_allclose(
                np.array(h_fast[:, -1]), np.array(ref["last_hidden_state"][:, -1]), atol=1e-4
            )
            ref_rows = mx.stack(
                [ref["attentions"][li][0, h, -1:, :] for li, h in LLAMA_ALIGNED_HEADS]
            ).mean(axis=0)
            np.testing.assert_allclose(
                np.array(mx.stack(w_fast).mean(axis=0)), np.array(ref_rows), atol=1e-5
            )
        self.assertEqual(cache.offset, L0 + 6)


class TestFastLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t3 = _tiny_t3()

    def _both(self, seed, analyzer, n_text=12, cfg=True, max_new=40):
        cond, text = _tiny_inputs(self.t3, n_text=n_text, cfg=cfg, seed=seed)
        kw = dict(GEN_KW, cfg_weight=0.5 if cfg else 0.0)
        out = []
        for fast in (False, True):
            mx.random.seed(seed)
            toks = self.t3.generate(
                t3_cond=cond,
                text_tokens=text,
                max_new_tokens=max_new,
                use_alignment_analyzer=analyzer,
                fast=fast,
                **kw,
            )
            out.append(np.array(toks)[0].tolist())
        return out

    def test_same_tokens_as_legacy_loop(self):
        for seed in range(4):
            for analyzer in (True, False):
                legacy, fast = self._both(seed, analyzer)
                self.assertEqual(legacy, fast, f"seed={seed} analyzer={analyzer}")

    def test_same_tokens_without_cfg(self):
        legacy, fast = self._both(7, True, cfg=False)
        self.assertEqual(legacy, fast)

    def test_generate_tokens_streams_same_ids(self):
        cond, text = _tiny_inputs(self.t3, seed=2)
        mx.random.seed(2)
        ids = list(
            self.t3.generate_tokens(
                t3_cond=cond, text_tokens=text, max_new_tokens=20, **GEN_KW
            )
        )
        mx.random.seed(2)
        whole = self.t3.generate(t3_cond=cond, text_tokens=text, max_new_tokens=20, **GEN_KW)
        self.assertEqual(ids, np.array(whole)[0].tolist())
        self.assertEqual(self.t3.last_generate_stats["steps"], len(ids))

    def test_cache_limit_unchanged(self):
        cond, text = _tiny_inputs(self.t3, seed=3)
        limit = 123 * 1024 * 1024
        old = mx.set_cache_limit(limit)
        try:
            for fast in (True, False):
                self.t3.generate(
                    t3_cond=cond, text_tokens=text, max_new_tokens=8, fast=fast, **GEN_KW
                )
                self.assertEqual(mx.set_cache_limit(limit), limit, f"fast={fast}")
        finally:
            mx.set_cache_limit(old)


class TestFinalize(unittest.TestCase):
    def test_float16_runs_and_keeps_dtype(self):
        t3 = _tiny_t3(seed=5).finalize("float16")
        self.assertEqual(t3.compute_dtype, mx.float16)
        self.assertEqual(t3.tfmr.layers[0].self_attn.q_proj.weight.dtype, mx.float16)
        cond, text = _tiny_inputs(t3, seed=4)
        toks = t3.generate(t3_cond=cond, text_tokens=text, max_new_tokens=10, **GEN_KW)
        self.assertEqual(toks.shape[0], 1)
        self.assertGreater(toks.shape[1], 0)

    def test_default_keeps_float32(self):
        t3 = _tiny_t3(seed=5).finalize()
        self.assertEqual(t3.compute_dtype, mx.float32)

    def test_bad_dtype_and_bits(self):
        t3 = _tiny_t3(seed=5)
        with self.assertRaises(ValueError):
            t3.finalize("float8")
        with self.assertRaises(ValueError):
            t3.finalize("float32", quantize_bits=3)


@unittest.skipUnless(
    os.environ.get("CHATTERBOX_T3_REAL") == "1",
    "real-weight check; set CHATTERBOX_T3_REAL=1 (loads ResembleAI/chatterbox)",
)
class TestRealWeights(unittest.TestCase):
    """Fast fp32 loop vs 1.2.0 loop on the real multilingual T3, same seed."""

    SENTENCES = [
        ("es", "Hola, ¿cómo estás hoy? Espero que todo vaya bien."),
        ("es", "Mañana vamos a la playa con mis amigos."),
        ("de", "Guten Morgen, wie geht es dir heute?"),
        ("de", "Ich habe das Buch gestern Abend fertig gelesen."),
        ("fr", "Bonjour, je m'appelle Myna et je parle français."),
        ("fr", "Il fait très beau aujourd'hui à Paris."),
        ("ru", "Привет, как у тебя дела сегодня?"),
        ("ru", "Я люблю читать книги по вечерам."),
        ("it", "Buongiorno, come stai oggi?"),
        ("pt", "Eu gosto muito de música brasileira."),
        ("es", "El tren sale a las ocho y media, no llegues tarde, por favor."),
        ("es", "Gracias por tu ayuda con el proyecto de ayer."),
        ("de", "Kannst du mir bitte sagen, wo der nächste Bahnhof ist?"),
        ("de", "Das Wetter ist heute schön, aber morgen soll es regnen."),
        ("fr", "Nous allons au cinéma ce soir, tu veux venir avec nous ?"),
        ("fr", "Merci beaucoup, c'était vraiment délicieux."),
        ("ru", "Сегодня мы идём в театр с друзьями."),
        ("ru", "Скажите, пожалуйста, сколько это стоит?"),
        ("ja", "今日はとても良い天気ですね。"),
        ("nl", "Ik ga morgen met de fiets naar mijn werk."),
    ]

    def test_eos_step_matches_legacy(self):
        from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

        tts = ChatterboxMultilingualTTSMLX.from_pretrained(device="cpu")
        results = []
        for seed, (lang, text) in enumerate(self.SENTENCES, start=1000):
            cond, toks = _real_inputs(tts, text, lang)
            row = []
            for fast in (False, True):
                mx.random.seed(seed)
                out = tts.t3.generate(
                    t3_cond=cond,
                    text_tokens=toks,
                    max_new_tokens=1000,
                    fast=fast,
                    **dict(GEN_KW, top_p=1.0),
                )
                row.append(np.array(out)[0].tolist())
            results.append((lang, row))
            print(
                f"[real] {lang} seed={seed} legacy={len(row[0])} fast={len(row[1])} "
                f"same={row[0] == row[1]}"
            )
        same = sum(r[0] == r[1] for _, r in results)
        # Exact equality is expected in most rows; fp32 SDPA vs manual attention can
        # flip a near-tie sample, after which the rows diverge (sampling noise).
        self.assertGreaterEqual(same, len(results) // 2)
        eos = tts.t3.hp.stop_speech_token
        for _, (a, b) in results:
            self.assertEqual(a[-1] == eos, b[-1] == eos)  # both end on EOS (or neither)


def _real_inputs(tts, text, lang):
    import torch.nn.functional as F

    from chatterbox.mtl_tts_mlx import punc_norm

    hp = tts.t3.hp
    t = tts.tokenizer.text_to_tokens(punc_norm(text), language_id=lang)
    import torch

    t = torch.cat([t, t], dim=0)
    t = F.pad(F.pad(t, (1, 0), value=hp.start_text_token), (0, 1), value=hp.stop_text_token)
    c = tts.conds.t3
    cond = T3CondMLX(
        speaker_emb=mx.array(c.speaker_emb.cpu().numpy()),
        cond_prompt_speech_tokens=mx.array(c.cond_prompt_speech_tokens.cpu().numpy()),
        emotion_adv=float(c.emotion_adv[0, 0, 0].item()),
    )
    return cond, mx.array(t.cpu().numpy())


if __name__ == "__main__":
    unittest.main()
