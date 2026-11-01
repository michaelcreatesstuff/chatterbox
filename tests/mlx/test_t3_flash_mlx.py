# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Tests for the MLX Chatterbox-Flash T3 (``chatterbox.models.t3_flash_mlx``).

Parity tests run against the upstream torch reference (``chatterbox-flash``
0.1.0, torch SDPA engine) on a tiny random model. The reference is found via
``import chatterbox_flash`` or the source tree in ``$CHATTERBOX_FLASH_SRC``;
those tests are skipped when neither is available. No real weights are loaded.
"""

import os
import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from chatterbox.models.t3 import llama_configs
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3_flash_mlx import T3FlashMLX, trim_speech_tokens
from chatterbox.models.t3_flash_mlx.backbone_mlx import (
    batch_visibility_mask,
    fused_commit_mask,
    padded_prefix_mask,
)
from chatterbox.models.t3_flash_mlx.sampling_mlx import (
    make_step_fn,
    masked_quantile,
    omnivoice_unmask_schedule,
    time_shift_quantiles,
)
from chatterbox.models.t3_mlx.modules.cond_enc_mlx import T3CondMLX

TINY = "Llama_flash_tiny"
TINY_CONFIG = dict(llama_configs.LLAMA_520M_CONFIG_DICT)
TINY_CONFIG.update(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
)
llama_configs.LLAMA_CONFIGS[TINY] = TINY_CONFIG


def _load_upstream():
    """Import the reference package, patched for this repo / transformers 5."""
    try:
        import chatterbox_flash  # noqa: F401
    except ImportError:
        src = os.environ.get("CHATTERBOX_FLASH_SRC")
        if not src or not Path(src, "chatterbox_flash").is_dir():
            return None
        sys.path.insert(0, src)
    try:
        import chatterbox_flash.engines.torch_sdpa as ts
        import chatterbox_flash.model as fm
    except Exception:  # pragma: no cover - environment dependent
        return None

    # This repo's T3 has no ``is_gpt`` (upstream chatterbox does).
    fm.ChatterboxFlashT3.is_gpt = False

    # transformers 5: DynamicCache.crop(n) only accepts negative n.
    def _crop(cache, max_length):
        cur = cache.get_seq_length()
        if cur > max_length:
            cache.crop(max_length - cur)

    ts._crop_dynamic_cache = _crop
    return fm


FM = _load_upstream()
needs_upstream = unittest.skipIf(FM is None, "chatterbox_flash reference not found")


def tiny_hp() -> T3Config:
    hp = T3Config.english_only()
    hp.llama_config_name = TINY
    hp.use_perceiver_resampler = False
    hp.text_tokens_dict_size = 50
    hp.start_text_token = 1
    hp.stop_text_token = 0
    hp.speech_tokens_dict_size = 200
    hp.start_speech_token = 190
    hp.stop_speech_token = 191
    hp.max_text_tokens = 100
    hp.max_speech_tokens = 400
    return hp


def tiny_mlx(seed: int = 0, quantize_bits=None) -> T3FlashMLX:
    mx.random.seed(seed)
    model = T3FlashMLX(tiny_hp())
    # Sharpen the speech head so argmax / ranking aren't decided by noise.
    model.speech_head.weight = model.speech_head.weight * 8.0
    return model.finalize("float32", quantize_bits=quantize_bits)


def tiny_cond_mlx(seed: int = 1):
    rng = np.random.default_rng(seed)
    return T3CondMLX(
        speaker_emb=mx.array(rng.standard_normal((1, 256)).astype(np.float32)),
        cond_prompt_speech_tokens=mx.array(rng.integers(0, 150, (1, 6))),
        emotion_adv=0.5,
    )


TEXT = [1, 5, 9, 17, 23, 4, 0]


class _UpstreamPair:
    """A tiny upstream torch model and its MLX copy with identical weights."""

    _pair = None

    @classmethod
    def get(cls):
        if cls._pair is None:
            torch.manual_seed(0)
            ref = FM.ChatterboxFlashT3(tiny_hp()).eval()
            with torch.no_grad():
                ref.speech_head.weight.mul_(8.0)
            ours = T3FlashMLX(tiny_hp())
            ours.load_torch_state_dict(ref.state_dict())
            ours.finalize("float32")
            rng = np.random.default_rng(3)
            spk = rng.standard_normal((1, 256)).astype(np.float32)
            ptoks = rng.integers(0, 150, (1, 6))
            cond_t = T3Cond(
                speaker_emb=torch.from_numpy(spk),
                cond_prompt_speech_tokens=torch.from_numpy(ptoks),
                emotion_adv=0.5 * torch.ones(1, 1, 1),
            )
            cond_m = T3CondMLX(
                speaker_emb=mx.array(spk),
                cond_prompt_speech_tokens=mx.array(ptoks),
                emotion_adv=0.5,
            )
            cls._pair = (ref, ours, cond_t, cond_m)
        return cls._pair


def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().numpy()
    return np.array(x.astype(mx.float32))


# --------------------------------------------------------------------------- #
#  Pure helpers
# --------------------------------------------------------------------------- #


class TestQuantile(unittest.TestCase):
    def test_matches_torch_quantile(self):
        rng = np.random.default_rng(0)
        for trial in range(200):
            L = 16
            vals = rng.standard_normal(L).astype(np.float32)
            if trial % 5 == 0:  # ties
                vals = np.round(vals, 1)
            valid = rng.random(L) < rng.uniform(0.05, 1.0)
            if not valid.any():
                valid[rng.integers(L)] = True
            for q in [0.0, 1.0, 0.5, 0.99, 0.9, float(rng.random())]:
                ref = torch.quantile(torch.from_numpy(vals[valid]), q).item()
                got = masked_quantile(
                    mx.array(vals), mx.array(valid), mx.array(q, dtype=mx.float32)
                ).item()
                self.assertAlmostEqual(got, ref, places=6, msg=f"q={q}")


class TestSchedules(unittest.TestCase):
    @needs_upstream
    def test_omnivoice_schedule_matches_reference(self):
        for n in range(0, 17):
            for K in (1, 2, 5, 10, 16):
                for t in (0.0, 0.1, 0.5, 1.0, 2.0):
                    self.assertEqual(
                        omnivoice_unmask_schedule(n, K, t),
                        FM._omnivoice_unmask_schedule(n, K, t),
                    )

    def test_schedule_sums(self):
        for n in range(1, 17):
            self.assertEqual(sum(omnivoice_unmask_schedule(n, 10, 0.5)), n)

    def test_time_shift_quantiles(self):
        q = time_shift_quantiles(10, 0.1)
        self.assertAlmostEqual(q[0], 0.99)
        self.assertAlmostEqual(q[-1], 0.9)


@needs_upstream
class TestStepParity(unittest.TestCase):
    """Deterministic step (temperature=0, position_temperature=0) vs reference."""

    def _run(self, use_cfg, tau):
        rng = np.random.default_rng(7)
        V, bl, K, MASK, STOP = 60, 16, 10, 60, 59
        prior = rng.random(V).astype(np.float32)
        prior /= prior.sum()
        step = make_step_fn(
            use_cfg=use_cfg,
            sample=False,
            position_noise=False,
            use_quantile=tau > 0,
            mask_id=MASK,
            stop_id=STOP,
        )
        w = 1.0
        for trial in range(60):
            k = trial % K
            B = 2 if use_cfg else 1
            logits = (rng.standard_normal((B, bl, V)) * 3).astype(np.float32)
            blk = rng.integers(0, 50, bl)
            blk[rng.random(bl) < 0.7] = MASK
            if not (blk == MASK).any():
                blk[0] = MASK
            sched = omnivoice_unmask_schedule(bl, K, 0.5)

            lt = torch.from_numpy(logits)
            pc = pu = None
            blk_logits = lt[0]
            if use_cfg:
                pc = torch.softmax(lt[0], -1)
                pu = torch.softmax(lt[1], -1)
                from chatterbox_flash.cfg_guidance import (
                    apply_zero_text_cfg_from_logits,
                )

                blk_logits = apply_zero_text_cfg_from_logits(lt[0], lt[1], w)
            ref = FM._pmi_count_early_step_unmask(
                blk_logits,
                torch.from_numpy(blk),
                bl=bl,
                k=k,
                K=K,
                MASK=MASK,
                device=torch.device("cpu"),
                temperature=0.0,
                temperature_sampling="multinomial",
                omnivoice_unmask_schedule_k=sched[k],
                time_shift_tau=tau,
                prior_override_v=torch.from_numpy(prior),
                block_marginal_prior=None,
                pmi_cfg_probs_c_bl=pc,
                pmi_cfg_probs_u_bl=pu,
                cfg_scale_for_pmi=w,
                position_temperature=0.0,
            )
            new_blk, flags = step(
                mx.array(logits),
                mx.array(blk.astype(np.int32)),
                mx.array(prior),
                mx.random.key(0),
                mx.array(sched[k], dtype=mx.int32),
                mx.array(time_shift_quantiles(K, tau)[k], dtype=mx.float32),
                mx.array(k == K - 1),
                mx.array(w, dtype=mx.float32),
                mx.array(1.0, dtype=mx.float32),
            )
            np.testing.assert_array_equal(np.array(new_blk), ref["xt_step"].numpy())
            n_mask, has_eos = flags.tolist()
            self.assertEqual(n_mask, ref["n_mask"])
            self.assertEqual(bool(has_eos), bool((ref["xt_step"] == STOP).any()))

    def test_cfg_quantile(self):
        self._run(use_cfg=True, tau=0.1)

    def test_cfg_large_tau(self):
        self._run(use_cfg=True, tau=0.6)

    def test_no_cfg_no_quantile(self):
        self._run(use_cfg=False, tau=0.0)


# --------------------------------------------------------------------------- #
#  Model parity against the torch reference
# --------------------------------------------------------------------------- #


@needs_upstream
class TestForwardParity(unittest.TestCase):
    def setUp(self):
        self.ref, self.ours, self.cond_t, self.cond_m = _UpstreamPair.get()

    def _ref_prefixes(self):
        ref = self.ref
        tt = torch.tensor([TEXT])
        with torch.no_grad():
            ce = ref.prepare_conditioning(self.cond_t)
            # prepare_conditioning caches the prompt embedding on the cond
            self.cond_t.cond_prompt_speech_emb = None
            p_cond = ref._build_prefix_emb(ce, tt, "cpu", torch.float32)
            text_emb = ref.text_emb(tt) + ref.text_pos_emb(tt)
            sos = torch.full((1, 1), ref.hp.start_speech_token)
            p_null = torch.cat(
                [
                    torch.zeros_like(ce),
                    torch.zeros_like(text_emb),
                    ref._embed_speech_tokens(sos),
                ],
                dim=1,
            )
        return p_cond, p_null

    def test_prefix_embedding(self):
        p_cond, p_null = self._ref_prefixes()
        ce = self.ours.prepare_conditioning(self.cond_m)
        ours = self.ours.build_prefix(ce, mx.array([TEXT]), cfg=True)
        np.testing.assert_allclose(_np(ours[0:1]), _np(p_cond), atol=1e-5)
        np.testing.assert_allclose(_np(ours[1:2]), _np(p_null), atol=1e-5)

    def test_uncond_prior(self):
        ref_prior = self.ref._compute_uncond_block_prior(
            16,
            self.ref.hp.start_speech_token,
            self.ref.mask_token_id,
            torch.float32,
            torch.device("cpu"),
        )
        np.testing.assert_allclose(
            _np(self.ours._uncond_prior), _np(ref_prior), atol=1e-6, rtol=1e-4
        )

    def test_prefix_block_and_shifted_logits(self):
        from chatterbox_flash.engines.torch_sdpa import TorchSDPAEngine

        p_cond, p_null = self._ref_prefixes()
        lp = p_cond.shape[1]
        engine = TorchSDPAEngine(self.ref, 256, torch.float32, batch_size=2)
        rng = np.random.default_rng(5)
        blocks = [rng.integers(0, 201, 16), rng.integers(0, 201, 16)]

        ours = self.ours
        ce = ours.prepare_conditioning(self.cond_m)
        prefix = ours.build_prefix(ce, mx.array([TEXT]), cfg=True)
        cache = ours.get_cache(2, 256)

        with torch.no_grad():
            ref_shift = engine.prefix_forward([p_cond, p_null])
        h = ours.forward_prefix(prefix, cache)
        shift = h[:, lp - 1 : lp]
        np.testing.assert_allclose(_np(shift), _np(ref_shift), atol=1e-4)

        cs = lp
        for blk in blocks:
            tok_t = torch.from_numpy(blk)[None].expand(2, -1)
            with torch.no_grad():
                emb_t = self.ref._embed_speech_tokens(tok_t)
                # tentative step, then the commit of the same tokens
                ref_h = engine.block_forward(emb_t, cs)
                ref_h = engine.block_forward(emb_t, cs)
                ref_logits = self.ref.speech_head(
                    torch.cat([ref_shift, ref_h[:, :15]], dim=1)
                )
                engine.advance_cache(16)
            emb_m = ours.embed_speech_block(mx.array(blk.astype(np.int32)))[None]
            emb_m = mx.broadcast_to(emb_m, (2,) + emb_m.shape[1:])
            ours.forward_block(emb_m, cache, cs)
            h = ours.forward_block(emb_m, cache, cs)
            logits = ours.shifted_logits(shift, h)
            np.testing.assert_allclose(_np(h), _np(ref_h), atol=1e-4)
            np.testing.assert_allclose(_np(logits), _np(ref_logits), atol=1e-3)
            ref_shift = ref_h[:, 15:16]
            shift = h[:, 15:16]
            cs += 16


@needs_upstream
class TestGenerateParity(unittest.TestCase):
    def setUp(self):
        self.ref, self.ours, self.cond_t, self.cond_m = _UpstreamPair.get()

    def _ref_generate(self, N, **kw):
        self.cond_t.cond_prompt_speech_emb = None
        out = self.ref.generate(
            t3_cond=self.cond_t,
            text_tokens=torch.tensor([TEXT]),
            text_token_lens=torch.tensor([len(TEXT)]),
            total_speech_len=N,
            temperature=0.0,
            position_temperature=0.0,
            backend="torch",
            use_cuda_graph=False,
            **kw,
        )
        return out[0].numpy()

    def _check(self, N, **kw):
        ref = self._ref_generate(N, **kw)
        modes = [(False, True), (False, False), (True, False)]
        for compiled in (True, False):
            for fuse, pipe in modes:
                stats = {}
                ours = self.ours.generate(
                    self.cond_m,
                    mx.array([TEXT]),
                    N,
                    temperature=0.0,
                    position_temperature=0.0,
                    fuse_commit=fuse,
                    pipeline=pipe,
                    compiled=compiled,
                    stats=stats,
                    **kw,
                )
                np.testing.assert_array_equal(
                    ours, ref, err_msg=f"compiled={compiled} fuse={fuse} pipe={pipe}"
                )
        return ref, stats

    def test_cfg_default_schedule(self):
        ref, stats = self._check(70, cfg_scale=1.0, time_shift_tau=0.1)
        self.assertGreater(len(ref), 0)
        self.assertGreaterEqual(stats["n_blocks"], 1)

    def test_no_cfg(self):
        self._check(40, cfg_scale=0.0, time_shift_tau=0.1)

    def test_no_early_decoding(self):
        self._check(36, cfg_scale=1.0, time_shift_tau=0.0, num_steps=4)

    def test_aggressive_early_decoding(self):
        self._check(52, cfg_scale=2.0, time_shift_tau=0.8, num_steps=6)

    def test_eos_early_exit(self):
        """Boost the stop-token logit so EOS lands mid-sequence (with MASKs left).

        Boosts stay small: a saturated softmax (p == 1) gives exact PMI ties,
        whose top-k order is implementation-defined in both libraries.
        """
        w0 = self.ref.speech_head.weight.detach().clone()
        stop = self.ref.hp.stop_speech_token
        try:
            for seed, boost in ((0, 0.4), (1, 0.5), (2, 0.6)):
                with torch.no_grad():
                    self.ref.speech_head.weight.copy_(w0)
                    g = torch.Generator().manual_seed(seed)
                    self.ref.speech_head.weight[stop] += boost * torch.randn(
                        64, generator=g
                    )
                # the reference caches the prior regardless of weight changes
                self.ref._uncond_block_prior_cache = {}
                ours = T3FlashMLX(tiny_hp())
                ours.load_torch_state_dict(self.ref.state_dict())
                ours.finalize("float32")
                saved, self.ours = self.ours, ours
                try:
                    ref, _ = self._check(120, cfg_scale=1.0, time_shift_tau=0.1)
                finally:
                    self.ours = saved
                self.assertLess(len(ref), 120)
        finally:
            with torch.no_grad():
                self.ref.speech_head.weight.copy_(w0)
            self.ref._uncond_block_prior_cache = {}


# --------------------------------------------------------------------------- #
#  MLX-only behaviour
# --------------------------------------------------------------------------- #


class TestFusedCommit(unittest.TestCase):
    def test_fused_forward_matches_unfused(self):
        model = tiny_mlx()
        ce = model.prepare_conditioning(tiny_cond_mlx())
        prefix = model.build_prefix(ce, mx.array([TEXT]), cfg=True)
        lp = prefix.shape[1]
        rng = np.random.default_rng(2)
        blk = mx.array(rng.integers(0, 190, 16).astype(np.int32))
        for bl_next in (16, 5):
            nxt = mx.full((bl_next,), model.mask_token_id, dtype=mx.int32)
            e_blk = mx.broadcast_to(model.embed_speech_block(blk)[None], (2, 16, 64))
            e_nxt = mx.broadcast_to(
                model.embed_speech_block(nxt)[None], (2, bl_next, 64)
            )

            cache = model.tfmr.make_cache(2, 256, mx.float32)
            model.forward_prefix(prefix, cache)
            h_commit = model.forward_block(e_blk, cache, lp)
            h_next = model.forward_block(e_nxt, cache, lp + 16)

            cache2 = model.tfmr.make_cache(2, 256, mx.float32)
            model.forward_prefix(prefix, cache2)
            fused = model.forward_block(
                mx.concatenate([e_blk, e_nxt], axis=1),
                cache2,
                lp,
                mask=fused_commit_mask(lp, 16, bl_next),
            )
            np.testing.assert_allclose(_np(fused[:, :16]), _np(h_commit), atol=1e-5)
            np.testing.assert_allclose(_np(fused[:, 16:]), _np(h_next), atol=1e-5)
            end = lp + 16 + bl_next
            for a, b in zip(cache.keys, cache2.keys):
                np.testing.assert_allclose(
                    _np(a[:, :, :end]), _np(b[:, :, :end]), atol=1e-5
                )

    def test_fused_generate_matches_unfused(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        kw = dict(temperature=0.6, position_temperature=5.0, seed=11)
        a = model.generate(cond, mx.array([TEXT]), 90, fuse_commit=False, **kw)
        b = model.generate(cond, mx.array([TEXT]), 90, fuse_commit=True, **kw)
        c = model.generate(cond, mx.array([TEXT]), 90, pipeline=False, **kw)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, c)


class TestCompiledBlock(unittest.TestCase):
    """Fixed-shape compiled block forward vs. the eager forward."""

    def _prefixed(self, model, max_seq=512):
        ce = model.prepare_conditioning(tiny_cond_mlx())
        prefix = model.build_prefix(ce, mx.array([TEXT]), cfg=True)
        cache = model.tfmr.make_cache(2, max_seq, mx.float32)
        hidden = model.forward_prefix(prefix, cache)
        lp = prefix.shape[1]
        return cache, lp, hidden[:, lp - 1 : lp]

    def test_block_fn_matches_eager(self):
        model = tiny_mlx()
        rng = np.random.default_rng(5)
        blk = mx.array(rng.integers(0, 190, 16).astype(np.int32))
        for bucket in (256, 0):
            model.attn_bucket = bucket
            cache_e, lp, ctx = self._prefixed(model)
            cache_c, _, _ = self._prefixed(model)
            # two blocks: the second sees the first one's K/V in the cache
            for cs in (lp, lp + 16):
                e = mx.broadcast_to(model.embed_speech_block(blk)[None], (2, 16, 64))
                h = model.forward_block(e, cache_e, cs)
                ref_logits = model.shifted_logits(ctx, h)
                logits, ctx_c = model.run_block_fn(cache_c, blk, ctx, cs)
                np.testing.assert_allclose(
                    _np(logits), _np(ref_logits), atol=1e-4, rtol=1e-4
                )
                np.testing.assert_allclose(_np(ctx_c), _np(h[:, 15:16]), atol=1e-5)
                end = cs + 16
                for a, b in zip(
                    cache_e.keys + cache_e.values, cache_c.keys + cache_c.values
                ):
                    np.testing.assert_allclose(
                        _np(a[:, :, :end]), _np(b[:, :, :end]), atol=1e-5
                    )
                ctx = h[:, 15:16]

    def test_fused_block_fn_matches_eager(self):
        model = tiny_mlx()
        rng = np.random.default_rng(6)
        blk = mx.array(rng.integers(0, 190, 16).astype(np.int32))
        nxt = mx.full((5,), model.mask_token_id, dtype=mx.int32)
        cache_e, lp, ctx = self._prefixed(model)
        cache_c, _, _ = self._prefixed(model)
        e = mx.broadcast_to(model.embed_speech_block(blk)[None], (2, 16, 64))
        en = mx.broadcast_to(model.embed_speech_block(nxt)[None], (2, 5, 64))
        fused = model.forward_block(
            mx.concatenate([e, en], axis=1),
            cache_e,
            lp,
            mask=fused_commit_mask(lp, 16, 5),
        )
        ref = model.shifted_logits(fused[:, 15:16], fused[:, 16:])
        logits, ctx_c = model.run_block_fn(
            cache_c, mx.concatenate([blk, nxt]), ctx, lp, n_first=16
        )
        np.testing.assert_allclose(_np(logits), _np(ref), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(_np(ctx_c), _np(fused[:, 15:16]), atol=1e-5)

    def test_compiled_generate_matches_eager(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        kw = dict(temperature=0.6, position_temperature=5.0, seed=11)
        a = model.generate(cond, mx.array([TEXT]), 300, compiled=False, **kw)
        for bucket in (256, 64, 0):
            model.attn_bucket = bucket
            b = model.generate(cond, mx.array([TEXT]), 300, compiled=True, **kw)
            np.testing.assert_array_equal(a, b, err_msg=f"attn_bucket={bucket}")

    def test_traces_reused(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()

        def block_traces():
            return sorted(k for k in model._block_fns if k[0] != "copy")

        model.generate(cond, mx.array([TEXT]), 90, seed=0)
        # full blocks only (the 10-token tail runs eagerly), one window bucket
        self.assertEqual(len(block_traces()), 1)
        n = len(model._block_fns)
        for i in range(3):
            model.generate(cond, mx.array([TEXT]), 90, seed=i)
        self.assertEqual(len(model._block_fns), n)
        model.finalize("float32")
        self.assertEqual(len(model._block_fns), 0)

    def test_cache_updated_in_place(self):
        """The compiled forward must not copy the cache each step."""
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        model.generate(cond, mx.array([TEXT]), 64, seed=0)
        cache = model._cache
        cache.keys = [mx.zeros((2, 2, 8192, 16)) for _ in cache.keys]
        cache.values = [mx.zeros((2, 2, 8192, 16)) for _ in cache.values]
        cache.max_seq = 8192
        mx.eval(cache.keys, cache.values)
        for pipeline in (True, False):
            model.generate(cond, mx.array([TEXT]), 64, seed=0, pipeline=pipeline)
            mx.clear_cache()
            mx.reset_peak_memory()
            base = mx.get_active_memory()
            model.generate(cond, mx.array([TEXT]), 64, seed=1, pipeline=pipeline)
            self.assertIs(model._cache, cache)
            # A copy of a single layer's K (2 MB) would already break this.
            self.assertLess(
                mx.get_peak_memory() - base, cache.keys[0].nbytes, f"{pipeline=}"
            )


class TestGenerateMLX(unittest.TestCase):
    def test_seed_reproducible(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        a = model.generate(cond, mx.array([TEXT]), 64, seed=3)
        b = model.generate(cond, mx.array([TEXT]), 64, seed=3)
        np.testing.assert_array_equal(a, b)
        self.assertLessEqual(len(a), 64)

    def test_precomputed_cond_emb(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        ce = model.prepare_conditioning(cond)
        a = model.generate(cond, mx.array([TEXT]), 48, seed=4)
        b = model.generate(ce, mx.array([TEXT]), 48, seed=4)
        np.testing.assert_array_equal(a, b)

    def test_quantized_and_half_precision_run(self):
        for bits, dtype in [(8, "float16"), (4, "bfloat16")]:
            mx.random.seed(0)
            model = T3FlashMLX(tiny_hp()).finalize(dtype, quantize_bits=bits)
            out = model.generate(tiny_cond_mlx(), mx.array([TEXT]), 40, seed=1)
            self.assertLessEqual(len(out), 40)
            self.assertTrue(np.all((out >= 0) & (out <= model.mask_token_id)))

    def test_repeated_calls_reuse_cache(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        model.generate(cond, mx.array([TEXT]), 80, seed=0)
        cache = model._cache
        nbytes = model.cache_nbytes
        mx.clear_cache()
        base = mx.get_active_memory()
        for i in range(30):
            model.generate(cond, mx.array([TEXT]), 80 - (i % 7), seed=i)
            self.assertIs(model._cache, cache)
        mx.clear_cache()
        self.assertEqual(model.cache_nbytes, nbytes)
        self.assertLess(mx.get_active_memory() - base, 1 << 20)

    def test_cache_grows_in_buckets(self):
        model = tiny_mlx()
        cache = model.get_cache(2, 100)
        self.assertEqual(cache.max_seq, 512)
        self.assertIs(model.get_cache(2, 500), cache)
        grown = model.get_cache(2, 600)
        self.assertEqual(grown.max_seq, 1024)
        self.assertIs(model.get_cache(2, 300), grown)


class TestBatchDecode(unittest.TestCase):
    """``generate_batch``: chunks as rows of one batch vs. one by one."""

    TEXTS = [TEXT, [1, 7, 0], [1, 3, 8, 12, 30, 31, 32, 33, 34, 2, 0], [1, 44, 2, 9, 0]]
    DET = dict(temperature=0.0, position_temperature=0.0)

    def _check(self, model, texts, Ns, seeds=None, **kw):
        cond = tiny_cond_mlx()
        seeds = seeds or list(range(len(texts)))
        single, single_stats = [], []
        for t, N, s in zip(texts, Ns, seeds):
            st = {}
            single.append(
                model.generate(cond, mx.array([t]), N, seed=s, stats=st, **kw)
            )
            single_stats.append(st)
        stats = {}
        batch = model.generate_batch(
            cond,
            [mx.array([t]) for t in texts],
            Ns,
            seeds=seeds,
            stats=stats,
            **{k: v for k, v in kw.items() if k != "fuse_commit"},
        )
        self.assertEqual(len(batch), len(texts))
        for c, (a, b) in enumerate(zip(single, batch)):
            np.testing.assert_array_equal(a, b, err_msg=f"chunk {c}")
        # same blocks and early exits per chunk; one round per block index
        self.assertEqual(stats["n_blocks"], [s["n_blocks"] for s in single_stats])
        self.assertEqual(
            stats["steps_per_block"], [s["steps_per_block"] for s in single_stats]
        )
        self.assertEqual(stats["n_rounds"], max(stats["n_blocks"]))
        return single, stats

    def test_rope_per_row_offset(self):
        """``RoPE`` with a ``(B,)`` offset (llama3 freqs) == one row at a time."""
        model = tiny_mlx()
        rope = model.tfmr.layers[0].self_attn.rope
        q = mx.random.normal((3, 4, 5, 16))
        k = mx.random.normal((3, 2, 5, 16))
        off = mx.array([0, 37, -4], dtype=mx.int32)
        qb, kb = rope(q, k, offset=off)
        for b in range(3):
            qs, ks = rope(q[b : b + 1], k[b : b + 1], offset=int(off[b].item()))
            np.testing.assert_allclose(_np(qb[b : b + 1]), _np(qs), atol=1e-6)
            np.testing.assert_allclose(_np(kb[b : b + 1]), _np(ks), atol=1e-6)

    def test_masks(self):
        pad = mx.array([0, 2], dtype=mx.int32)
        m = np.array(padded_prefix_mask(pad, 4))[:, 0]
        np.testing.assert_array_equal(m[0], np.tril(np.ones((4, 4), bool)))
        want = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]])
        np.testing.assert_array_equal(m[1], want.astype(bool))
        vis = batch_visibility_mask(
            mx.array(4), pad, mx.array([3, 1], dtype=mx.int32), 8
        )
        self.assertEqual(vis.shape, (2, 1, 1, 8))
        np.testing.assert_array_equal(
            np.array(vis)[:, 0, 0].astype(int),
            [[1, 1, 1, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0, 0, 0]],
        )

    def test_deterministic_matches_single(self):
        """Different text lengths, budgets and short last blocks, CFG on / off."""
        model = tiny_mlx()
        Ns = [90, 64, 130, 40]
        for kw in (
            dict(self.DET),
            dict(self.DET, pipeline=False),
            dict(self.DET, cfg_scale=0.0),
            dict(self.DET, time_shift_tau=0.5),
        ):
            with self.subTest(**kw):
                single, _ = self._check(model, self.TEXTS, Ns, **kw)
                self.assertEqual(len({len(x) for x in single}), len(Ns))

    def test_uncompiled_matches(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        texts = [mx.array([t]) for t in self.TEXTS[:3]]
        a = model.generate_batch(cond, texts, [48, 64, 33], **self.DET)
        b = model.generate_batch(
            cond, texts, [48, 64, 33], compile_fn=False, **self.DET
        )
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)

    def test_different_eos_points(self):
        """Chunks stop at different EOS points; live chunks are unaffected."""
        mx.random.seed(0)
        model = T3FlashMLX(tiny_hp())
        w = model.speech_head.weight * 8.0
        stop = model.hp.stop_speech_token
        rng = np.random.default_rng(1)
        w[stop] = w[stop] + 0.8 * mx.array(rng.standard_normal(64).astype(np.float32))
        model.speech_head.weight = w
        model.finalize("float32")
        Ns = [160, 160, 130, 96]
        single, stats = self._check(model, self.TEXTS, Ns, **self.DET)
        lens = [len(s) for s in single]
        self.assertEqual(lens[2:], [130, 96])
        self.assertLess(lens[0], 160)
        self.assertLess(lens[1], lens[0])
        # EOS chunks drop out early: fewer blocks than the batch rounds
        self.assertEqual(stats["n_rounds"], 9)
        self.assertLess(stats["n_blocks"][1], stats["n_blocks"][0])
        self.assertLess(stats["n_blocks"][0], 9)

    def test_seeded_sampling_matches_single(self):
        """Per-chunk keys split as in ``generate``: same samples (whole blocks)."""
        model = tiny_mlx()
        self._check(model, self.TEXTS[:3], [96, 64, 128], seeds=[3, 4, 5])

    def test_padding_slots_and_empty(self):
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        texts = [mx.array([t]) for t in self.TEXTS[:2]]
        a = model.generate_batch(cond, texts, [48, 64], **self.DET)
        stats = {}
        b = model.generate_batch(
            cond, texts, [48, 64], n_slots=4, stats=stats, **self.DET
        )
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(model._batch_caches[0].batch_size, 8)
        self.assertEqual(len(stats["n_blocks"]), 2)
        out = model.generate_batch(cond, texts, [0, 0])
        self.assertEqual([len(o) for o in out], [0, 0])

    def test_cache_in_place_and_flat_memory(self):
        """Batched forwards update both caches in place; no growth over calls."""
        model = tiny_mlx()
        cond = tiny_cond_mlx()
        texts = [mx.array([t]) for t in self.TEXTS]
        model.generate_batch(cond, texts, [64, 64, 48, 64], seeds=[0, 1, 2, 3])
        caches = list(model._batch_caches)
        for c in caches:
            self.assertEqual(c.batch_size, 8)
            c.keys = [mx.zeros((8, 2, 8192, 16)) for _ in c.keys]
            c.values = [mx.zeros((8, 2, 8192, 16)) for _ in c.values]
            c.max_seq = 8192
            mx.eval(c.keys, c.values)
        nbytes = model.cache_nbytes
        model.generate_batch(cond, texts, [64, 64, 48, 64], seeds=[0, 1, 2, 3])
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        for i in range(5):
            model.generate_batch(cond, texts, [64, 60, 48, 64], seeds=[i] * 4)
            for a, b in zip(model._batch_caches, caches):
                self.assertIs(a, b)
        # A copy of a single layer's K (8 MB) would already break this; the
        # ~5 MB peak is two overlapping pipelined forwards' activations.
        self.assertLess(mx.get_peak_memory() - base, caches[0].keys[0].nbytes)
        mx.clear_cache()
        self.assertEqual(model.cache_nbytes, nbytes)
        self.assertLess(mx.get_active_memory() - base, 1 << 20)


class TestRealConfig(unittest.TestCase):
    """Llama-520M layout checks that don't load any weights."""

    def test_rope_matches_hf_inv_freq(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

        from chatterbox.models.t3_flash_mlx.backbone_mlx import FlashAttention
        from chatterbox.models.t3_mlx.modules.llama_mlx import LlamaConfigMLX

        cfg_dict = llama_configs.LLAMA_CONFIGS["Llama_520M"]
        hf = LlamaRotaryEmbedding(LlamaConfig(**cfg_dict))
        attn = FlashAttention(LlamaConfigMLX.from_dict(cfg_dict))
        np.testing.assert_allclose(
            1.0 / np.array(attn.rope._freqs), hf.inv_freq.numpy(), rtol=1e-6
        )

    def test_checkpoint_keys_match(self):
        """Compare parameter names / shapes with the cached checkpoint header."""
        import glob
        import json
        import struct

        from mlx.utils import tree_flatten

        hub = os.path.expanduser("~/.cache/huggingface/hub")
        files = glob.glob(
            f"{hub}/models--ResembleAI--chatterbox-flash/snapshots/*/t3_flash.safetensors"
        )
        if not files:
            self.skipTest("chatterbox-flash checkpoint not cached")
        with open(files[0], "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        ckpt = {k: tuple(v["shape"]) for k, v in header.items()}
        model = T3FlashMLX()
        params = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
        self.assertEqual(sorted(set(params) - set(ckpt)), [])
        self.assertEqual(
            sorted(set(ckpt) - set(params)),
            ["text_head.weight", "tfmr.embed_tokens.weight"],
        )
        for k, shape in params.items():
            self.assertEqual(shape, ckpt[k], k)


class TestTrim(unittest.TestCase):
    def test_trim(self):
        hp = tiny_hp()
        toks = np.array([3, 200, 5, 190, 7, 191, 8])
        np.testing.assert_array_equal(trim_speech_tokens(toks, hp), [3, 5, 7])


class TestPipeline(unittest.TestCase):
    """``ChatterboxFlashMLX.generate`` contract with a tiny T3 and stub vocoder."""

    def test_generate_contract(self):
        from chatterbox.tts_flash_mlx import (
            ChatterboxFlashMLX,
            speech_len_for_text_tokens,
        )

        class _Tok:
            def text_to_tokens(self, text):
                return torch.tensor([[5, 9, 17, 23]], dtype=torch.int32)

        class _S3Gen:
            def __init__(self):
                self.calls = []

            def inference(self, speech_tokens, ref_dict, n_cfm_timesteps=None):
                self.calls.append((speech_tokens.clone(), n_cfm_timesteps))
                return torch.zeros(1, 480 * len(speech_tokens)), None

        t3 = tiny_mlx()
        s3 = _S3Gen()
        tts = ChatterboxFlashMLX(t3, s3, ve=None, tokenizer=_Tok(), s3gen_backend="mps")
        self.assertEqual(tts.sr, 24000)
        tts._t3_cond_emb = t3.prepare_conditioning(tiny_cond_mlx())
        tts._ref_dict = {}
        wav = tts.generate("Hello there.", seed=0)
        self.assertEqual(wav.dtype, np.float32)
        self.assertEqual(wav.ndim, 1)
        t = tts.last_timings
        for key in (
            "t3_s",
            "s3gen_s",
            "n_speech_tokens",
            "n_forwards",
            "n_blocks",
            "avg_steps_per_block",
            "n_chunks",
            "batch_rounds",
        ):
            self.assertIn(key, t)
        toks, n_cfm = s3.calls[-1]
        self.assertEqual(n_cfm, 2)
        self.assertEqual(len(toks), t["n_speech_tokens"])
        self.assertTrue(bool((toks < t3.hp.start_speech_token).all()))
        self.assertEqual(speech_len_for_text_tokens(10), 300)
        self.assertEqual(speech_len_for_text_tokens(100), 600)

    def _tts(self):
        from chatterbox.tts_flash_mlx import ChatterboxFlashMLX

        class _Tok:
            def text_to_tokens(self, text):
                n = max(1, len(text.split()))
                return torch.arange(2, 2 + n, dtype=torch.int32)[None] % 40 + 2

        class _S3Gen:
            def __init__(self):
                self.calls = []

            def inference(self, speech_tokens, ref_dict, n_cfm_timesteps=None):
                self.calls.append(len(speech_tokens))
                return torch.ones(1, 480 * len(speech_tokens)), None

        t3 = tiny_mlx()
        tts = ChatterboxFlashMLX(
            t3, _S3Gen(), ve=None, tokenizer=_Tok(), s3gen_backend="mps"
        )
        tts._t3_cond_emb = t3.prepare_conditioning(tiny_cond_mlx())
        tts._ref_dict = {}
        return tts

    def test_single_shot_unchanged(self):
        """split_sentences=False is the upstream path (upstream budget, one pass)."""
        from chatterbox.tts_flash_mlx import speech_len_for_text_tokens

        tts = self._tts()
        text = "one two three. " * 20
        wav = tts.generate(text, seed=5, split_sentences=False)
        t = tts.last_timings
        self.assertEqual(t["n_chunks"], 1)
        enc = tts._encode_text(text)
        budget = speech_len_for_text_tokens(int(enc.shape[1]))
        self.assertEqual(t["speech_budget"], budget)
        toks = tts.t3.generate(tts._t3_cond_emb, enc, budget, seed=5)
        n = len(trim_speech_tokens(toks, tts.t3.hp))
        self.assertEqual(t["n_speech_tokens"], n)
        self.assertEqual(len(wav), 480 * n)

    def test_plan_chunks(self):
        from chatterbox.tts_flash_mlx import count_words, plan_chunks

        sentences = [" ".join(["word"] * n) + "." for n in (20, 20, 3, 20)]
        chunks = plan_chunks(" ".join(sentences), splitter=lambda *a: list(sentences))
        # the 3-word sentence joins its shorter neighbour
        self.assertEqual([count_words(c) for c in chunks], [20, 23, 20])
        tail = [" ".join(["word"] * n) + "." for n in (30, 4)]
        chunks = plan_chunks(" ".join(tail), splitter=lambda *a: list(tail))
        self.assertEqual([count_words(c) for c in chunks], [34])
        # within the cap: never split
        self.assertEqual(plan_chunks("a b c. d e."), ["a b c. d e."])

    def test_split_generate_sums_timings(self):
        import chatterbox.tts_flash_mlx as tfm

        tts = self._tts()
        sentences = [" ".join(["word"] * 20) + "." for _ in range(3)]
        orig = tfm.split_text_intelligently
        tfm.split_text_intelligently = lambda text, cap, lang: list(sentences)
        try:
            wav = tts.generate(" ".join(sentences), seed=0)
        finally:
            tfm.split_text_intelligently = orig
        t = tts.last_timings
        self.assertEqual(t["n_chunks"], 3)
        calls = tts.s3gen.calls
        self.assertEqual(len(calls), 3)
        self.assertEqual(t["n_speech_tokens"], sum(calls))
        self.assertEqual(t["speech_budget"], 3 * tfm.speech_budget_for_words(20))
        # 20 ms trimmed per chunk and 50 ms overlaps (crossfade_chunks)
        n = sum(480 * c - 480 for c in calls) - 2 * 1200
        self.assertEqual(len(wav), n)
        self.assertEqual(wav.dtype, np.float32)

    def test_batched_chunks(self):
        """max_batch_chunks=1 is the one-by-one path; >1 gives the same tokens."""
        import chatterbox.tts_flash_mlx as tfm

        sentences = [" ".join(["word"] * n) + "." for n in (20, 9, 30, 14, 25)]
        orig = tfm.split_text_intelligently
        tfm.split_text_intelligently = lambda text, cap, lang: list(sentences)
        runs = {}
        try:
            for kw in (dict(), dict(temperature=0.0, position_temperature=0.0)):
                for mb in (1, 4):
                    tts = self._tts()
                    toks = []
                    tts._vocode = lambda t, n, s: toks.append(t) or np.ones(
                        480 * len(t)
                    )
                    tts.generate(" ".join(sentences), seed=7, max_batch_chunks=mb, **kw)
                    runs[(bool(kw), mb)] = (toks, dict(tts.last_timings), tts)
        finally:
            tfm.split_text_intelligently = orig

        # max_batch_chunks=1 == T3FlashMLX.generate per chunk with seed + i
        toks, t, tts = runs[(False, 1)]
        self.assertEqual(t["n_chunks"], 5)
        self.assertEqual(t["batch_rounds"], t["n_blocks"])
        totals = []
        for i, s in enumerate(sentences):
            budget = tfm.speech_budget_for_words(tfm.count_words(s))
            enc, total = tts._encode_with_budget(s, budget, True, None)
            totals.append(total)
            ref = tts.t3.generate(tts._t3_cond_emb, enc, total, seed=7 + i)
            np.testing.assert_array_equal(toks[i], trim_speech_tokens(ref, tts.t3.hp))
        # the 30-word chunk is capped by the upstream budget: a short last block
        self.assertEqual(totals[2] % 16, 12)

        # batched: 5 chunks -> two groups of 3 (one dead slot), same tokens;
        # seeded sampling may differ only in a short last block (its noise
        # is drawn at full block shape)
        for det in (True, False):
            one, t1, _ = runs[(det, 1)]
            four, t4, tts4 = runs[(det, 4)]
            self.assertEqual(t4["n_chunks"], 5)
            self.assertEqual(tts4.t3._batch_caches[0].batch_size, 6)
            self.assertEqual(t4["n_blocks"], t1["n_blocks"])
            self.assertLess(t4["batch_rounds"], t1["batch_rounds"])
            self.assertLess(t4["n_forwards"], t1["n_forwards"])
            for a, b, total in zip(one, four, totals):
                if det or total % 16 == 0:
                    np.testing.assert_array_equal(a, b)
                else:
                    # equal raw tokens before the tail: the trimmed outputs
                    # share all but at most the tail's (total % 16) tokens
                    n = min(len(a), len(b)) - total % 16
                    np.testing.assert_array_equal(a[:n], b[:n])

    def test_budget_and_presets(self):
        from chatterbox.tts_flash_mlx import (
            EARLY_DECODING_PRESETS,
            speech_budget_for_words,
        )

        self.assertEqual(speech_budget_for_words(0), 64)
        self.assertEqual(speech_budget_for_words(11), 176)  # 162.5 -> 11 blocks
        self.assertEqual(speech_budget_for_words(35), 464)
        self.assertEqual(EARLY_DECODING_PRESETS["paper"], 0.1)
        tts = self._tts()
        with self.assertRaisesRegex(ValueError, "preset"):
            tts.generate("Hello there.", preset="nope")
        tts.generate("Hello there.", seed=0, preset="fast")
        self.assertIn("avg_steps_per_block", tts.last_timings)

    def test_meanflow_guard(self):
        from chatterbox.tts_flash_mlx import _require_meanflow

        class _Old:
            def __init__(self):
                pass

        class _New:
            def __init__(self, meanflow=False):
                pass

        class _Wrapped(_Old):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        with self.assertRaisesRegex(RuntimeError, "meanflow"):
            _require_meanflow(_Old, "Old")
        with self.assertRaisesRegex(RuntimeError, "meanflow"):
            _require_meanflow(_Wrapped, "Wrapped")
        _require_meanflow(_New, "New")


class TestGuards(unittest.TestCase):
    """Edge cases and concurrency of ``ChatterboxFlashMLX.generate``."""

    _tts = TestPipeline._tts

    def test_num_steps_zero_raises(self):
        tts = self._tts()
        with self.assertRaises(ValueError):
            tts.generate("one two three.", num_steps=0)
        with self.assertRaises(ValueError):
            tts.t3.generate(
                tts._t3_cond_emb, tts._encode_text("one two."), 32, num_steps=0
            )

    def test_empty_text_gives_empty_wav(self):
        tts = self._tts()
        for text in ("", "   \n"):
            wav = tts.generate(text, seed=0)
            self.assertEqual(wav.shape, (0,))
            self.assertEqual(tts.last_timings["n_chunks"], 0)

    def test_tiny_chunk_is_dropped_before_crossfade(self):
        """A 1-2 token chunk used to crash crossfade_chunks (negative offset)."""
        import chatterbox.tts_flash_mlx as tfm

        tts = self._tts()
        lengths = [1000, 24000, 24000, 24000]
        timing = dict(
            t3_s=0.0,
            s3gen_s=0.0,
            n_speech_tokens=1,
            speech_budget=1,
            n_forwards=1,
            n_blocks=1,
            batch_rounds=1,
            steps_per_block=[1],
        )

        def fake_group(texts, budgets, **kwargs):
            n = len(texts)
            return [np.ones(lengths.pop(0), np.float32) for _ in range(n)], [
                dict(timing) for _ in range(n)
            ]

        sentences = [" ".join(["word"] * 20) + "." for _ in range(4)]
        orig = tfm.split_text_intelligently
        tfm.split_text_intelligently = lambda text, cap, lang: list(sentences)
        tts._generate_group = fake_group
        try:
            wav = tts.generate(" ".join(sentences), seed=0)
        finally:
            tfm.split_text_intelligently = orig
        self.assertGreater(len(wav), 3 * 24000 - 3 * 2400)

    def test_concurrent_generate_is_serialised(self):
        """Two threads on one instance used to segfault on the shared caches."""
        import threading

        tts = self._tts()
        kwargs = dict(temperature=0.0, position_temperature=0.0, seed=0)
        texts = ["one two three four five.", "six seven eight."]
        ref = [tts.generate(t, **kwargs) for t in texts]
        results, errors = {}, []

        def worker(i):
            try:
                for r in range(3):
                    results[(i, r)] = tts.generate(texts[i], **kwargs)
            except Exception as e:  # pragma: no cover - reported below
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in (0, 1)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(errors, [])
        for (i, _), wav in results.items():
            np.testing.assert_array_equal(wav, ref[i])

    def test_release_caches_frees_and_rebuilds(self):
        tts = self._tts()
        kwargs = dict(temperature=0.0, position_temperature=0.0, seed=0)
        ref = tts.generate("one two three four five.", **kwargs)
        self.assertGreater(tts.t3.cache_nbytes, 0)
        tts.release_caches()
        self.assertEqual(tts.t3.cache_nbytes, 0)
        again = tts.generate("one two three four five.", **kwargs)
        np.testing.assert_array_equal(again, ref)

    def test_single_leftover_chunk_runs_unbatched(self):
        """3 chunks at max_batch_chunks=2 batch the first 2 and run the last alone."""
        import chatterbox.tts_flash_mlx as tfm

        tts = self._tts()
        sentences = [" ".join(["word"] * 20) + "." for _ in range(3)]
        groups, singles = [], []
        orig_group, orig_chunk = tts._generate_group, tts._generate_chunk
        tts._generate_group = lambda texts, *a, **k: (
            groups.append(len(texts)) or orig_group(texts, *a, **k)
        )
        tts._generate_chunk = lambda text, *a, **k: (
            singles.append(text) or orig_chunk(text, *a, **k)
        )
        orig = tfm.split_text_intelligently
        tfm.split_text_intelligently = lambda text, cap, lang: list(sentences)
        try:
            tts.generate(" ".join(sentences), seed=0, max_batch_chunks=2)
        finally:
            tfm.split_text_intelligently = orig
        self.assertEqual(groups, [2])
        self.assertEqual(len(singles), 1)
        self.assertEqual(tts.last_timings["n_chunks"], 3)


if __name__ == "__main__":
    unittest.main()
