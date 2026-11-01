# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for the MLX Chatterbox-Turbo / Nano T3 (``chatterbox.models.t3_turbo_mlx``).

Parity tests run against a torch reference built from a tiny random
``transformers.GPT2Model`` plus the T3 embeddings / heads, decoded with a copy
of upstream ``T3.inference_turbo`` (HF logits processors). No real weights are
loaded; the real-config tests build models on the ``meta`` device or without
evaluating their parameters.
"""

import threading
import unittest

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX
from chatterbox.models.t3_turbo_mlx import (
    GPT2_MEDIUM_CONFIG,
    GPT2_SMALL_CONFIG,
    T3TurboMLX,
    TurboT3Config,
    trim_at_stop,
)
from chatterbox.models.t3_turbo_mlx.gpt2_mlx import decode_mask, prefill_mask
from chatterbox.models.t3_turbo_mlx.sampling_mlx import (
    gumbel_sample,
    onehot_mask,
    process_logits,
    sampling_params,
)

TINY_CONFIG = dict(GPT2_SMALL_CONFIG)
TINY_CONFIG.update(n_embd=64, hidden_size=64, n_head=4, n_layer=2, n_positions=1024)
SOS, EOS, V = 6561, 6562, 6563


class TorchTurboRef(nn.Module):
    """Upstream T3 with a GPT-2 config (``is_gpt``), Turbo hyperparameters."""

    def __init__(self, hp: TurboT3Config):
        super().__init__()
        self.hp = hp
        cfg = GPT2Config(**hp.gpt2_config)
        self.tfmr = GPT2Model(cfg)
        dim = cfg.hidden_size
        self.cond_enc = nn.Module()
        self.cond_enc.spkr_enc = nn.Linear(hp.speaker_embed_size, dim)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, dim)
        self.text_head = nn.Linear(dim, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(dim, hp.speech_tokens_dict_size, bias=True)

    def cond_emb(self, speaker_emb, prompt_tokens):
        spkr = self.cond_enc.spkr_enc(speaker_emb.view(-1, 256))[:, None]
        return torch.cat([spkr, self.speech_emb(prompt_tokens)], dim=1)

    def prefix(self, speaker_emb, prompt_tokens, text_tokens, speech_tokens):
        return torch.cat(
            [
                self.cond_emb(speaker_emb, prompt_tokens),
                self.text_emb(text_tokens),
                self.speech_emb(speech_tokens),
            ],
            dim=1,
        )

    @torch.inference_mode()
    def inference_turbo(
        self,
        speaker_emb,
        prompt_tokens,
        text_tokens,
        temperature=0.8,
        top_k=1000,
        top_p=0.95,
        repetition_penalty=1.2,
        max_gen_len=1000,
    ):
        """Copy of upstream ``T3.inference_turbo`` (tqdm removed)."""
        logits_processors = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            logits_processors.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            logits_processors.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            logits_processors.append(TopPLogitsWarper(top_p))
        if repetition_penalty != 1.0:
            logits_processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))

        speech_start_token = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
        embeds = self.prefix(speaker_emb, prompt_tokens, text_tokens, speech_start_token)

        generated_speech_tokens = []
        llm_outputs = self.tfmr(inputs_embeds=embeds, use_cache=True)
        hidden_states = llm_outputs[0]
        past_key_values = llm_outputs.past_key_values

        speech_hidden = hidden_states[:, -1:]
        speech_logits = self.speech_head(speech_hidden)
        processed_logits = logits_processors(speech_start_token, speech_logits[:, -1, :])
        probs = F.softmax(processed_logits, dim=-1)
        next_speech_token = torch.multinomial(probs, num_samples=1)

        generated_speech_tokens.append(next_speech_token)
        current_speech_token = next_speech_token

        for _ in range(max_gen_len):
            current_speech_embed = self.speech_emb(current_speech_token)
            llm_outputs = self.tfmr(
                inputs_embeds=current_speech_embed,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden_states = llm_outputs[0]
            past_key_values = llm_outputs.past_key_values
            speech_logits = self.speech_head(hidden_states)

            input_ids = torch.cat(generated_speech_tokens, dim=1)
            processed_logits = logits_processors(input_ids, speech_logits[:, -1, :])
            if torch.all(processed_logits == -float("inf")):
                break
            probs = F.softmax(processed_logits, dim=-1)
            next_speech_token = torch.multinomial(probs, num_samples=1)

            generated_speech_tokens.append(next_speech_token)
            current_speech_token = next_speech_token
            if torch.all(next_speech_token == self.hp.stop_speech_token):
                break

        all_tokens = torch.cat(generated_speech_tokens, dim=1)
        if all_tokens.size(1) > 0 and all_tokens[0, -1] == self.hp.stop_speech_token:
            all_tokens = all_tokens[:, :-1]
        return all_tokens


def tiny_pair(seed=0, stop_bias=None, dtype="float32", attn_bucket=32):
    """A random torch reference and the MLX model with the same weights."""
    torch.manual_seed(seed)
    hp = TurboT3Config(gpt2_config=TINY_CONFIG)
    ref = TorchTurboRef(hp).eval()
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if "ln_" in name and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn_like(p))
            elif "emb" in name or "wpe" in name or "wte" in name:
                p.copy_(torch.randn_like(p))
            elif "speech_head" in name:
                p.copy_(0.3 * torch.randn_like(p))
            else:
                p.copy_(0.1 * torch.randn_like(p))
        if stop_bias is not None:
            ref.speech_head.bias[EOS] = stop_bias
    m = T3TurboMLX(hp=hp)
    m.attn_bucket = attn_bucket  # several window traces in short tests
    m.load_torch_state_dict(ref.state_dict())
    m.finalize(dtype)
    return ref, m


def tiny_cond(seed=0):
    g = torch.Generator().manual_seed(1000 + seed)
    spk = torch.randn(1, 256, generator=g)
    prompt = torch.randint(0, 6561, (1, 375), generator=g)
    return spk, prompt


def cond_mlx(spk, prompt):
    return T3CondMLX(
        speaker_emb=mx.array(spk.numpy()),
        cond_prompt_speech_tokens=mx.array(prompt.numpy().astype(np.int32)),
    )


def text_ids(n, offset=0):
    return torch.arange(n, dtype=torch.long)[None] * 37 % 50000 + 11 + offset


class TestRealConfig(unittest.TestCase):
    """Key / shape mapping for the real Turbo and Nano configs (no weights)."""

    def _check(self, nano):
        from mlx.utils import tree_flatten

        from chatterbox.models.t3_turbo_mlx.gpt2_mlx import is_conv1d_weight

        cfg = GPT2_SMALL_CONFIG if nano else GPT2_MEDIUM_CONFIG
        with torch.device("meta"):
            tfmr = GPT2Model(GPT2Config(**cfg))
        torch_shapes = {"tfmr." + k: tuple(v.shape) for k, v in tfmr.state_dict().items()}
        D = cfg["n_embd"]
        torch_shapes.update(
            {
                "cond_enc.spkr_enc.weight": (D, 256),
                "cond_enc.spkr_enc.bias": (D,),
                "text_emb.weight": (50276, D),
                "speech_emb.weight": (6563, D),
                "speech_head.weight": (6563, D),
                "speech_head.bias": (6563,),
            }
        )
        m = T3TurboMLX(nano=nano)  # parameters stay unevaluated
        ours = {k: tuple(v.shape) for k, v in tree_flatten(m.parameters())}
        self.assertEqual(set(ours) - set(torch_shapes), set())
        self.assertEqual(set(torch_shapes) - set(ours), {"tfmr.wte.weight"})
        for k, shape in ours.items():
            expect = torch_shapes[k][::-1] if is_conv1d_weight(k) else torch_shapes[k]
            self.assertEqual(shape, expect, k)
        self.assertEqual(m.cfg.n_layer, cfg["n_layer"])
        self.assertEqual(m.hp.n_channels, D)
        self.assertEqual(m.hp.speech_cond_prompt_len, 375)

    def test_turbo_keys(self):
        self._check(nano=False)

    def test_nano_keys(self):
        self._check(nano=True)


class TestForwardParity(unittest.TestCase):
    def setUp(self):
        self.ref, self.m = tiny_pair()
        self.spk, self.prompt = tiny_cond()

    def _mlx_teacher_forced(self, text, toks):
        """Cond K/V + prefill + one decode step per token, alternating buffers."""
        m = self.m
        ck, cv = m.cond_kv(cond_mlx(self.spk, self.prompt))
        lc = ck[0].shape[2]
        row = m.build_row(mx.array(text.numpy().astype(np.int32)))
        lr = row.shape[1]
        cache = m.get_cache(1, lc + lr + len(toks) + 1)
        pad = mx.zeros((1,), mx.int32)
        h, cache.buffers = m.tfmr.prefill(
            row, mx.arange(lc, lc + lr)[None], prefill_mask(pad, lc, lr), ck, cv,
            cache.buffers,
        )
        logits = [m.speech_head(h[:, -1])]
        prev_k = prev_v = [mx.zeros((1, ck[0].shape[1], 1, ck[0].shape[3]))] * len(ck)
        for i, t in enumerate(toks):
            col = lc + lr + i
            attn_len = m.attn_len_for(col, cache.max_seq)
            ks, vs = cache.buffers[i % 2]
            h, prev_k, prev_v, ks, vs = m.tfmr.decode(
                m.speech_emb(mx.array([t], mx.int32))[:, None],
                mx.array([col], mx.int32), mx.array([col], mx.int32),
                mx.array([col - 1 if i else col], mx.int32), prev_k, prev_v,
                decode_mask(pad, lc, mx.array([col], mx.int32), attn_len), ks, vs, attn_len,
            )
            cache.buffers[i % 2] = (ks, vs)
            logits.append(m.speech_head(h[:, 0]))
        return np.array(mx.stack(logits, axis=1))

    def test_prefill_and_decode_logits(self):
        text = text_ids(9)
        toks = [17, 4000, 17, 6000, 5, 6561, 100, 2, 3, 3001, 42]  # crosses 2 merges
        with torch.no_grad():
            seq = torch.tensor([[SOS] + toks])
            emb = self.ref.prefix(self.spk, self.prompt, text, seq)
            hidden = self.ref.tfmr(inputs_embeds=emb).last_hidden_state
            lp = emb.shape[1] - len(toks)
            ref_logits = self.ref.speech_head(hidden[:, lp - 1 :]).numpy()
        ours = self._mlx_teacher_forced(text, toks)
        self.assertEqual(ours.shape, ref_logits.shape)
        np.testing.assert_allclose(ours, ref_logits, rtol=1e-4, atol=1e-4)

    def test_left_padded_prefill(self):
        """A left-padded row gives the same last-position logits as alone."""
        m = self.m
        ck, cv = m.cond_kv(cond_mlx(self.spk, self.prompt))
        lc = ck[0].shape[2]
        short, long_ = text_ids(4), text_ids(10, offset=3)
        r_s = m.build_row(mx.array(short.numpy().astype(np.int32)))
        r_l = m.build_row(mx.array(long_.numpy().astype(np.int32)))
        lr = r_l.shape[1]
        pad = lr - r_s.shape[1]
        rows = mx.concatenate(
            [mx.concatenate([mx.zeros((1, pad, m.dim)), r_s], axis=1), r_l], axis=0
        )
        pad_arr = mx.array([pad, 0], mx.int32)
        cache = m.get_cache(2, lc + lr + 8)
        pos = mx.maximum(mx.arange(lc, lc + lr)[None] - pad_arr[:, None], 0)
        h, _ = m.tfmr.prefill(
            rows, pos, prefill_mask(pad_arr, lc, lr), ck, cv, cache.buffers
        )
        got = np.array(m.speech_head(h[:, -1]))
        with torch.no_grad():
            for row, text in enumerate([short, long_]):
                emb = self.ref.prefix(self.spk, self.prompt, text, torch.tensor([[SOS]]))
                hid = self.ref.tfmr(inputs_embeds=emb).last_hidden_state
                want = self.ref.speech_head(hid[:, -1]).numpy()[0]
                np.testing.assert_allclose(got[row], want, rtol=1e-4, atol=1e-4)

    def test_cond_kv_cached_by_identity(self):
        cond = cond_mlx(self.spk, self.prompt)
        a = self.m.cond_kv(cond)
        self.assertIs(self.m.cond_kv(cond)[0], a[0])
        b = self.m.cond_kv(cond_mlx(self.spk, self.prompt))
        self.assertIsNot(b[0], a[0])
        np.testing.assert_allclose(np.array(b[0][0]), np.array(a[0][0]), atol=1e-6)
        self.assertEqual(a[0][0].shape, (1, 4, 376, 16))

    def test_gelu_new(self):
        from transformers.activations import NewGELUActivation

        import mlx.nn as mnn

        x = torch.linspace(-6, 6, 1001)
        np.testing.assert_allclose(
            np.array(mnn.gelu_approx(mx.array(x.numpy()))),
            NewGELUActivation()(x).numpy(),
            atol=1e-5,
        )


class TestProcessors(unittest.TestCase):
    """The on-device chain vs HF's processors on random logits."""

    CASES = [
        dict(temperature=0.8, top_k=1000, top_p=0.95, penalty=1.2),  # upstream
        dict(temperature=1.0, top_k=1000, top_p=0.95, penalty=1.2),
        dict(temperature=0.0, top_k=50, top_p=1.0, penalty=1.2),
        dict(temperature=0.5, top_k=0, top_p=0.8, penalty=1.0),
        dict(temperature=1.3, top_k=7000, top_p=0.5, penalty=2.0),
        dict(temperature=0.8, top_k=1, top_p=0.95, penalty=1.2),
    ]

    def _hf(self, logits, ids, temperature, top_k, top_p, penalty):
        procs = LogitsProcessorList()
        if temperature > 0 and temperature != 1.0:
            procs.append(TemperatureLogitsWarper(temperature))
        if top_k > 0:
            procs.append(TopKLogitsWarper(top_k))
        if top_p < 1.0:
            procs.append(TopPLogitsWarper(top_p))
        if penalty != 1.0:
            procs.append(RepetitionPenaltyLogitsProcessor(penalty))
        return procs(ids, logits.clone())

    def test_matches_hf(self):
        g = torch.Generator().manual_seed(0)
        for case in self.CASES:
            for first_step in (True, False):
                with self.subTest(**case, first_step=first_step):
                    logits = 3.0 * torch.randn(3, V, generator=g)
                    if first_step:
                        ids = torch.full((3, 1), SOS)
                    else:
                        ids = torch.randint(0, V, (3, 12), generator=g)
                        ids[:, -1] = logits.argmax(-1)  # penalise a kept token
                    want = self._hf(logits, ids, **case)
                    presence = np.zeros((3, V), dtype=bool)
                    for b in range(3):
                        presence[b, ids[b].numpy()] = True
                    params, flags = sampling_params(
                        case["temperature"], case["top_k"], case["top_p"], case["penalty"]
                    )
                    got = np.array(
                        process_logits(
                            mx.array(logits.numpy()), mx.array(presence), *params, **flags
                        )
                    )
                    w = want.numpy()
                    np.testing.assert_array_equal(np.isfinite(got), np.isfinite(w))
                    fin = np.isfinite(w)
                    np.testing.assert_allclose(got[fin], w[fin], rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(
                        np.array(mx.softmax(mx.array(got), axis=-1)),
                        torch.softmax(want, -1).numpy(),
                        atol=1e-6,
                    )

    def test_gumbel_sampling_distribution(self):
        logits = mx.array([[0.0, 1.0, 2.0, -mx.inf, 0.5]])
        probs = np.array(mx.softmax(logits, axis=-1))[0]
        keys = mx.random.key(3)[None]
        draws = []
        for _ in range(4000):
            tok, keys = gumbel_sample(logits, keys)
            draws.append(int(tok[0]))
        freq = np.bincount(draws, minlength=5) / len(draws)
        self.assertEqual(freq[3], 0.0)
        np.testing.assert_allclose(freq, probs, atol=0.03)

    def test_onehot(self):
        m = np.array(onehot_mask(mx.array([0, 5], mx.int32), 7))
        self.assertTrue(m[0, 0] and m[1, 5])
        self.assertEqual(int(m.sum()), 2)


def _ref_greedy(ref, spk, prompt, text, max_gen_len, **kw):
    kw.setdefault("top_k", 1)
    return ref.inference_turbo(spk, prompt, text, max_gen_len=max_gen_len, **kw)[0].numpy()


class TestGenerateParity(unittest.TestCase):
    def test_greedy_matches_reference(self):
        ref, m = tiny_pair()
        spk, prompt = tiny_cond()
        cond = cond_mlx(spk, prompt)
        for n_text, max_len in ((6, 40), (13, 23)):
            text = text_ids(n_text)
            want = _ref_greedy(ref, spk, prompt, text, max_len)
            got = m.generate(
                cond, mx.array(text.numpy().astype(np.int32)), top_k=1,
                max_gen_len=max_len, seed=0,
            )
            np.testing.assert_array_equal(got, want)
            self.assertEqual(len(got), max_len + 1)

    def test_greedy_variants_match_reference(self):
        """No repetition penalty / no temperature, uncompiled, window buckets."""
        ref, m = tiny_pair(seed=2)
        spk, prompt = tiny_cond(2)
        cond = cond_mlx(spk, prompt)
        text = text_ids(7)
        tt = mx.array(text.numpy().astype(np.int32))
        for kw in (
            dict(repetition_penalty=1.0),
            dict(temperature=0.0),
            dict(top_p=1.0, repetition_penalty=1.5),
        ):
            with self.subTest(**kw):
                want = _ref_greedy(ref, spk, prompt, text, 30, **kw)
                for compile_fn in (True, False):
                    got = m.generate(
                        cond, tt, top_k=1, max_gen_len=30, compile_fn=compile_fn, **kw
                    )
                    np.testing.assert_array_equal(got, want)
        for bucket in (0, 1, 7, 256):
            m.attn_bucket = bucket
            m.release_caches()
            got = m.generate(cond, tt, top_k=1, max_gen_len=30)
            np.testing.assert_array_equal(got, _ref_greedy(ref, spk, prompt, text, 30))

    def test_eos_stop(self):
        """Stops at the first EOS after the first token and strips it."""
        spk, prompt = tiny_cond(1)
        cond = cond_mlx(spk, prompt)
        text = text_ids(8)
        tt = mx.array(text.numpy().astype(np.int32))
        found = False
        for bias in np.linspace(0.0, 6.0, 25):
            ref, m = tiny_pair(seed=1, stop_bias=float(bias))
            want = _ref_greedy(ref, spk, prompt, text, 60)
            if 2 < len(want) < 40:
                found = True
                stats = {}
                got = m.generate(cond, tt, top_k=1, max_gen_len=60, stats=stats)
                np.testing.assert_array_equal(got, want)
                # Stopped two steps after EOS (pipeline depth), not at max_gen_len.
                self.assertLessEqual(stats["n_steps"], len(want) + 2)
                break
        self.assertTrue(found, "no stop bias gave an EOS mid-sequence")

    def test_trim_at_stop(self):
        np.testing.assert_array_equal(trim_at_stop(np.array([EOS, 3, 4, EOS, EOS]), EOS), [EOS, 3, 4])
        np.testing.assert_array_equal(trim_at_stop(np.array([1, 2]), EOS), [1, 2])
        np.testing.assert_array_equal(trim_at_stop(np.array([EOS]), EOS), [EOS])


class TestBatchDecode(unittest.TestCase):
    def setUp(self):
        self.ref, self.m = tiny_pair(seed=3)
        self.spk, self.prompt = tiny_cond(3)
        self.cond = cond_mlx(self.spk, self.prompt)
        self.texts = [
            mx.array(text_ids(n, offset=n).numpy().astype(np.int32)) for n in (5, 12, 8)
        ]

    def test_greedy_batch_matches_single(self):
        singles = [
            self.m.generate(self.cond, t, top_k=1, max_gen_len=25) for t in self.texts
        ]
        batch = self.m.generate_batch(self.cond, self.texts, top_k=1, max_gen_len=25)
        for s, b in zip(singles, batch):
            np.testing.assert_array_equal(b, s)
        # and the reference
        for t, b in zip(self.texts, batch):
            want = _ref_greedy(
                self.ref, self.spk, self.prompt, torch.from_numpy(np.array(t)).long(), 25
            )
            np.testing.assert_array_equal(b, want)

    def test_seeded_sampling_batch_matches_single(self):
        seeds = [11, 12, 13]
        singles = [
            self.m.generate(self.cond, t, max_gen_len=30, seed=s)
            for t, s in zip(self.texts, seeds)
        ]
        batch = self.m.generate_batch(self.cond, self.texts, max_gen_len=30, seeds=seeds)
        for s, b in zip(singles, batch):
            np.testing.assert_array_equal(b, s)
        again = self.m.generate(self.cond, self.texts[0], max_gen_len=30, seed=11)
        np.testing.assert_array_equal(again, singles[0])
        other = self.m.generate(self.cond, self.texts[0], max_gen_len=30, seed=99)
        self.assertFalse(np.array_equal(other, singles[0]))

    def test_different_eos_points(self):
        texts = [mx.array(text_ids(n, offset=7 * n).numpy().astype(np.int32)) for n in range(3, 11)]
        for bias in np.linspace(0.0, 6.0, 25):
            _, m = tiny_pair(seed=3, stop_bias=float(bias))
            singles = [m.generate(self.cond, t, top_k=1, max_gen_len=50) for t in texts]
            lens = {len(s) for s in singles}
            if len(lens) > 2 and min(lens) < 51:
                break
        else:
            self.fail("no stop bias gave different EOS points")
        batch = m.generate_batch(self.cond, texts, top_k=1, max_gen_len=50)
        for s, b in zip(singles, batch):
            np.testing.assert_array_equal(b, s)

    def test_per_row_caps(self):
        """A row that never emits EOS stops at its cap; other rows are unaffected."""
        _, m = tiny_pair(seed=3, stop_bias=-100.0)  # no row ever emits EOS
        singles = [m.generate(self.cond, t, top_k=1, max_gen_len=30) for t in self.texts]
        stats = {}
        batch = m.generate_batch(
            self.cond, self.texts, top_k=1, max_gen_len=30,
            max_gen_lens=[7, 30, 12], stats=stats,
        )
        self.assertEqual([len(b) for b in batch], [8, 31, 13])  # 1 + cap tokens
        for s_, b in zip(singles, batch):
            np.testing.assert_array_equal(b, s_[: len(b)])
        # Every row capped early -> the group ends at the largest cap.
        stats = {}
        batch = m.generate_batch(
            self.cond, self.texts, top_k=1, max_gen_len=30,
            max_gen_lens=[5, 9, 3], stats=stats,
        )
        self.assertEqual([len(b) for b in batch], [6, 10, 4])
        self.assertLessEqual(stats["n_steps"], 9)
        with self.assertRaises(ValueError):
            m.generate_batch(self.cond, self.texts, max_gen_lens=[1])

    def test_padding_slots_and_cache_reuse(self):
        a = self.m.generate_batch(self.cond, self.texts[:2], top_k=1, max_gen_len=20, n_slots=3)
        cache = self.m._cache
        self.assertEqual(cache.batch_size, 3)
        b = self.m.generate_batch(self.cond, self.texts, top_k=1, max_gen_len=20)
        self.assertIs(self.m._cache, cache)  # same batch size, fits: reused
        self.assertEqual(len(a), 2)
        for x, y in zip(a, b[:2]):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(self.m.generate_batch(self.cond, []), [])

    def test_precomputed_cond_and_half_precision(self):
        emb = self.m.prepare_conditioning(self.cond)
        self.assertEqual(emb.shape, (1, 376, self.m.dim))
        a = self.m.generate(self.cond, self.texts[0], top_k=1, max_gen_len=10)
        b = self.m.generate(emb, self.texts[0], top_k=1, max_gen_len=10)
        np.testing.assert_array_equal(a, b)
        for dtype, q in (("float16", None), ("bfloat16", None), ("float16", 8)):
            _, m = tiny_pair(seed=3)
            m.finalize(dtype, quantize_bits=q)
            out = m.generate(self.cond, self.texts[1], max_gen_len=12, seed=0)
            self.assertEqual(out.dtype, np.int64)
            self.assertTrue(((out >= 0) & (out < V)).all())

    def test_decode_writes_in_place(self):
        """Pipelined steps update the two cache buffers in place (no copies)."""
        cfg = dict(TINY_CONFIG, n_embd=256, hidden_size=256, n_head=4, n_layer=2)
        m = T3TurboMLX(hp=TurboT3Config(gpt2_config=cfg)).finalize("float32")
        texts = self.texts * 2  # B = 6
        m.generate_batch(self.cond, texts, max_gen_len=200, seeds=list(range(6)))
        one_buffer = m.cache_nbytes / 2
        mx.reset_peak_memory()
        before = mx.get_active_memory()
        m.generate_batch(self.cond, texts, max_gen_len=200, seeds=list(range(6)))
        self.assertLess(mx.get_peak_memory() - before, 0.5 * one_buffer)

    def test_release_caches(self):
        self.m.generate(self.cond, self.texts[0], top_k=1, max_gen_len=5)
        self.assertGreater(self.m.cache_nbytes, 0)
        self.m.release_caches()
        self.assertEqual(self.m.cache_nbytes, 0)
        self.assertEqual(self.m._fns, {})
        self.assertIsNone(self.m._cond)


# --------------------------------------------------------------------------- #
#  Pipeline (stubs: tokenizer, vocoder, watermarker; tiny T3)
# --------------------------------------------------------------------------- #


class _Tok:
    def __init__(self):
        self.texts = []

    def __call__(self, text, return_tensors=None, padding=None, truncation=None):
        self.texts.append(text)
        n = max(1, len(text.split()))
        return {"input_ids": (np.arange(n)[None] * 31 % 5000 + 7).astype(np.int64)}


class _S3GenMLX:
    def __init__(self):
        self.calls = []

    def inference(self, speech_tokens, ref_dict, finalize=True, n_cfm_timesteps=None):
        self.calls.append((np.array(speech_tokens).reshape(-1), n_cfm_timesteps))
        return mx.ones((1, 480 * speech_tokens.shape[-1])), None


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


class TestPipeline(unittest.TestCase):
    def _tts(self, **kw):
        from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX

        _, t3 = tiny_pair(seed=4)
        tts = ChatterboxTurboMLX(t3, _S3GenMLX(), ve=None, tokenizer=_Tok(), conds=_conds(), **kw)
        tts._watermarker = _Watermarker()
        return tts

    def test_generate_contract(self):
        from chatterbox.tts_turbo_mlx import S3GEN_SIL

        tts = self._tts()
        self.assertEqual(tts.sr, 24000)
        self.assertEqual(tts._ref_dict["prompt_token_len"], 10)
        wav = tts.generate("hello there", seed=0, max_gen_len=20)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.ndim, 1)
        t = tts.last_timings
        for key in ("t3_s", "s3gen_s", "watermark_s", "n_speech_tokens", "n_chunks", "n_steps"):
            self.assertIn(key, t)
        self.assertEqual(t["n_chunks"], 1)
        toks, n_cfm = tts.s3gen.calls[-1]
        self.assertEqual(n_cfm, 2)
        np.testing.assert_array_equal(toks[-3:], [S3GEN_SIL] * 3)
        self.assertTrue((toks[:-3] < 6561).all())
        self.assertEqual(len(toks) - 3, t["n_speech_tokens"])
        self.assertEqual(tts._watermarker.calls, 1)
        np.testing.assert_allclose(wav, 0.5)  # stub vocoder ones, stub watermark * 0.5
        self.assertEqual(tts.tokenizer.texts[-1], "Hello there.")  # punc_norm

        tts.generate("hello there", seed=0, max_gen_len=20, watermark=False)
        self.assertEqual(tts._watermarker.calls, 1)
        self.assertEqual(tts.last_timings["watermark_s"], 0.0)

    def test_empty_text_and_missing_conds(self):
        from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX

        tts = self._tts()
        for text in ("", "   \n"):
            wav = tts.generate(text)
            self.assertEqual(wav.shape, (0,))
            self.assertEqual(wav.dtype, np.float32)
            self.assertEqual(tts.last_timings, {"n_chunks": 0})
        self.assertEqual(tts.s3gen.calls, [])
        _, t3 = tiny_pair(seed=4)
        bare = ChatterboxTurboMLX(t3, _S3GenMLX(), ve=None, tokenizer=_Tok())
        with self.assertRaises(ValueError):
            bare.generate("Hello.")

    def test_long_text_batched_groups(self):
        tts = self._tts()
        sentence = "This sentence has exactly ten words in it for testing. "
        text = sentence * 9  # 90 words -> several chunks
        wav = tts.generate(text, seed=1, max_gen_len=12, max_batch_chunks=2, watermark=False)
        t = tts.last_timings
        self.assertGreater(t["n_chunks"], 1)
        self.assertEqual(len(tts.s3gen.calls), t["n_chunks"])
        self.assertEqual(t["n_groups"], -(-t["n_chunks"] // 2))
        self.assertTrue(all(b == t["batch_sizes"][0] for b in t["batch_sizes"]))
        self.assertGreater(len(wav), 0)
        # max_batch_chunks=1 runs every chunk alone; split_sentences=False one pass.
        tts.generate(text, seed=1, max_gen_len=12, max_batch_chunks=1, watermark=False)
        self.assertEqual(tts.last_timings["batch_sizes"], [1] * t["n_chunks"])
        tts.generate(text, seed=1, max_gen_len=12, split_sentences=False, watermark=False)
        self.assertEqual(tts.last_timings["n_chunks"], 1)

    def test_lock_and_threads(self):
        tts = self._tts()
        self.assertIsInstance(tts._lock, type(threading.RLock()))
        errors, results = [], []

        def run(seed):
            try:
                results.append(tts.generate("Hi there.", seed=seed, max_gen_len=8))
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)

    def test_release_caches(self):
        tts = self._tts()
        tts.generate("Hello.", max_gen_len=5, watermark=False)
        self.assertGreater(tts.t3.cache_nbytes, 0)
        tts.release_caches()
        self.assertEqual(tts.t3.cache_nbytes, 0)
        tts.generate("Hello.", max_gen_len=5, watermark=False)  # rebuilds

    def test_chunk_caps_passed(self):
        from chatterbox.tts_turbo_mlx import chunk_speech_cap

        self.assertEqual(chunk_speech_cap("one two three four"), int(1.5 * 75 + 25))
        self.assertEqual(chunk_speech_cap("word " * 200, max_gen_len=1000), 1000)
        tts = self._tts()
        seen = []
        orig = tts.t3.generate_batch

        def spy(*a, **kw):
            seen.append(kw.get("max_gen_lens"))
            return orig(*a, **kw)

        tts.t3.generate_batch = spy
        tts.generate("Hello there friend.", max_gen_len=20, watermark=False)
        self.assertEqual(seen[-1], [20])  # cap 200 clipped to max_gen_len
        tts.generate("Hello there friend.", split_sentences=False, watermark=False, max_gen_len=20)
        self.assertIsNone(seen[-1])  # single shot: upstream max_gen_len only

    def test_punc_norm_and_postprocess(self):
        from chatterbox.tts_turbo_mlx import S3GEN_SIL, postprocess_speech_tokens, punc_norm

        self.assertEqual(punc_norm("hello   world"), "Hello world.")
        self.assertEqual(punc_norm("Wait… what: “yes”"), 'Wait,  what, "yes".')  # upstream keeps the double space
        self.assertEqual(punc_norm("Ok!"), "Ok!")
        out = postprocess_speech_tokens(np.array([1, 6561, 6562, 7000, 5]))
        np.testing.assert_array_equal(out, [1, 5, S3GEN_SIL, S3GEN_SIL, S3GEN_SIL])


class TestConditioningHelpers(unittest.TestCase):
    def test_normalize_loudness_keeps_float32(self):
        # pyloudnorm returns a float64 gain; a float64 wav breaks the S3
        # tokenizer's float32 mel filters.
        from chatterbox.tts_turbo_mlx import normalize_loudness

        wav = (np.random.default_rng(0).standard_normal(24000 * 6) * 0.1).astype(np.float32)
        self.assertEqual(normalize_loudness(wav, 24000).dtype, np.float32)

    def test_ref_dict_to_mlx_accepts_grad_tensors(self):
        # conds.pt tensors can require grad; numpy() on them raises.
        from chatterbox.tts_turbo_mlx import ref_dict_to_mlx

        ref = {
            "prompt_token": torch.zeros(1, 5, dtype=torch.long),
            "prompt_token_len": torch.tensor([5]),
            "prompt_feat": torch.randn(1, 10, 80, requires_grad=True),
            "embedding": torch.randn(1, 80, requires_grad=True),
        }
        out = ref_dict_to_mlx(ref)
        self.assertEqual(out["prompt_feat"].shape, (1, 10, 80))
        self.assertEqual(out["prompt_token_len"], 5)


if __name__ == "__main__":
    unittest.main()
