"""Meanflow S3Gen tests (tiny random weights only, no checkpoints).

Covers:
- torch time mixer / decoder-with-r / 2-step meanflow solve vs upstream
  resemble-ai/chatterbox master (loaded via ``git show upstream/master:...``;
  skipped when that ref is unavailable),
- torch vs MLX parity of the same pieces,
- non-meanflow paths unaffected by the meanflow flag,
- S3Gen-level plumbing of ``meanflow`` / ``n_cfm_timesteps`` on both backends.
"""

import subprocess
import types
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
import torch

from chatterbox.models.s3gen.decoder import ConditionalDecoder
from chatterbox.models.s3gen.flow_matching import CausalConditionalCFM
from chatterbox.models.s3gen.utils.intmeanflow import get_intmeanflow_time_mixer
from chatterbox.models.s3gen_mlx.convert_weights import convert_s3gen_state_dict
from chatterbox.models.s3gen_mlx.decoder_mlx import ConditionalDecoderMLX
from chatterbox.models.s3gen_mlx.flow_matching_mlx import CausalConditionalCFMMLX

REPO = Path(__file__).resolve().parents[2]
TINY = dict(
    in_channels=320,
    out_channels=80,
    causal=True,
    channels=[64],
    dropout=0.0,
    attention_head_dim=16,
    n_blocks=1,
    num_mid_blocks=1,
    num_heads=2,
    act_fn="gelu",
)
T = 24


def _inputs(B=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 80, T, generator=g)
    mu = torch.randn(B, 80, T, generator=g)
    cond = torch.randn(B, 80, T, generator=g)
    spks = torch.randn(B, 80, generator=g)
    mask = torch.ones(B, 1, T)
    mask[..., -3:] = 0.0  # exercise masking
    return x, mask, mu, spks, cond


def _tiny_decoder(meanflow, seed=0, randomize_mixer=True):
    torch.manual_seed(seed)
    dec = ConditionalDecoder(meanflow=meanflow, **TINY).eval()
    if meanflow and randomize_mixer:
        # Off-diagonal mixer so the r branch actually matters.
        with torch.no_grad():
            dec.time_embed_mixer.weight.normal_(0, 0.05)
    return dec


def _load_upstream(relpath, modname):
    """Exec an upstream s3gen file with its relative imports bound to this repo."""
    try:
        src = subprocess.run(
            ["git", "show", f"upstream/master:src/chatterbox/models/s3gen/{relpath}"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("upstream/master not available")
    src = src.replace("from .", "from chatterbox.models.s3gen.")
    mod = types.ModuleType(modname)
    exec(compile(src, f"upstream/{relpath}", "exec"), mod.__dict__)
    return mod


def _to_mlx_decoder(dec):
    prefix = "flow.decoder.estimator."
    sd = {prefix + k: v for k, v in dec.state_dict().items()}
    conv = convert_s3gen_state_dict(sd)
    weights = [(k[len(prefix) :], mx.array(v)) for k, v in conv.items()]
    mdec = ConditionalDecoderMLX(meanflow=dec.meanflow, **TINY)
    mdec.load_weights(weights, strict=True)
    return mdec


def _mx(*ts):
    return [mx.array(t.numpy()) for t in ts]


# --------------------------------------------------------------------------
# torch vs upstream
# --------------------------------------------------------------------------


def test_time_mixer_matches_upstream_init():
    up = _load_upstream("utils/intmeanflow.py", "up_intmeanflow")
    a, b = up.get_intmeanflow_time_mixer(16), get_intmeanflow_time_mixer(16)
    assert b.bias is None
    assert torch.equal(a.weight, b.weight)
    assert torch.equal(b.weight, torch.cat([torch.eye(16), torch.zeros(16, 16)], 1))


def test_torch_decoder_with_r_matches_upstream():
    up = _load_upstream("decoder.py", "up_decoder")
    dec = _tiny_decoder(meanflow=True)
    ref = up.ConditionalDecoder(meanflow=True, **TINY).eval()
    ref.load_state_dict(dec.state_dict(), strict=True)
    x, mask, mu, spks, cond = _inputs()
    t, r = torch.tensor([0.25]), torch.tensor([0.75])
    with torch.inference_mode():
        out = dec(x, mask, mu, t, spks, cond, r=r)
        exp = ref(x, mask, mu, t, spks, cond, r=r)
        # r must matter with an off-diagonal mixer
        other = dec(x, mask, mu, t, spks, cond, r=torch.tensor([0.5]))
    torch.testing.assert_close(out, exp, rtol=0, atol=0)
    assert not torch.allclose(out, other)


def test_torch_meanflow_solve_matches_upstream():
    up_dec = _load_upstream("decoder.py", "up_decoder")
    up_fm = _load_upstream("flow_matching.py", "up_flow_matching")
    dec = _tiny_decoder(meanflow=True)
    ref_dec = up_dec.ConditionalDecoder(meanflow=True, **TINY).eval()
    ref_dec.load_state_dict(dec.state_dict(), strict=True)
    cfm = CausalConditionalCFM(spk_emb_dim=80, estimator=dec)
    ref_cfm = up_fm.CausalConditionalCFM(spk_emb_dim=80, estimator=ref_dec)

    _, mask, mu, spks, cond = _inputs()
    noise = torch.randn(1, 80, T, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        # noised_mels spans all frames, so both z's are exactly `noise`.
        out, _ = cfm(mu, mask, 2, spks=spks, cond=cond, noised_mels=noise, meanflow=True)
        exp, _ = ref_cfm(
            mu, mask, 2, spks=spks, cond=cond, noised_mels=noise, meanflow=True
        )
    torch.testing.assert_close(out, exp, rtol=0, atol=0)


def test_noised_mels_only_replaces_generated_frames():
    dec = _tiny_decoder(meanflow=True)
    cfm = CausalConditionalCFM(spk_emb_dim=80, estimator=dec)
    calls = []
    cfm.basic_euler = lambda z, **kw: calls.append(z.clone()) or z
    _, mask, mu, spks, cond = _inputs()
    noise = torch.full((1, 80, 10), 7.0)
    cfm(mu, mask, 2, spks=spks, cond=cond, noised_mels=noise, meanflow=True)
    z = calls[0]
    assert torch.equal(z[..., T - 10 :], noise)
    assert torch.equal(z[..., : T - 10], cfm._get_rand_noise(T, z.device, z.dtype)[..., : T - 10])


# --------------------------------------------------------------------------
# non-meanflow unchanged
# --------------------------------------------------------------------------


def test_torch_non_meanflow_unchanged_by_flag():
    plain = _tiny_decoder(meanflow=False)
    mf = _tiny_decoder(meanflow=True, randomize_mixer=False)
    assert plain.time_embed_mixer is None
    assert not any("time_embed_mixer" in k for k in plain.state_dict())
    assert set(mf.state_dict()) - set(plain.state_dict()) == {"time_embed_mixer.weight"}
    # Same RNG stream for all shared params (mixer is created after init).
    for k, v in plain.state_dict().items():
        assert torch.equal(v, mf.state_dict()[k]), k
    x, mask, mu, spks, cond = _inputs()
    t = torch.tensor([0.3])
    with torch.inference_mode():
        a = plain(x, mask, mu, t, spks, cond)
        b = plain(x, mask, mu, t, spks, cond, r=torch.tensor([0.9]))  # r ignored
        # diagonal-init mixer passes t through: meanflow == plain for any r
        c = mf(x, mask, mu, t, spks, cond, r=torch.tensor([0.9]))
    assert torch.equal(a, b)
    torch.testing.assert_close(a, c)


def test_torch_cfm_default_path_unchanged():
    dec = _tiny_decoder(meanflow=False)
    cfm = CausalConditionalCFM(spk_emb_dim=80, estimator=dec)
    _, mask, mu, spks, cond = _inputs()
    torch.manual_seed(1)
    with torch.inference_mode():
        a, _ = cfm(mu, mask, 5, spks=spks, cond=cond)
        b, _ = cfm(mu, mask, 5, spks=spks, cond=cond, noised_mels=None, meanflow=False)
        # explicit reference: cosine schedule + midpoint CFG solve
        z = cfm._get_rand_noise(T, mu.device, mu.dtype)
        ts = 1 - torch.cos(torch.linspace(0, 1, 6) * 0.5 * torch.pi)
        c = cfm.solve_midpoint(z, ts, mu, mask, spks, cond)
    assert torch.equal(a, b) and torch.equal(a, c)


# --------------------------------------------------------------------------
# torch vs MLX
# --------------------------------------------------------------------------


def test_mlx_time_mixer_init_matches_torch():
    mdec = ConditionalDecoderMLX(meanflow=True, **TINY)
    ref = get_intmeanflow_time_mixer(TINY["channels"][0] * 4).weight.detach().numpy()
    np.testing.assert_array_equal(np.array(mdec.time_embed_mixer.weight), ref)
    assert "bias" not in mdec.time_embed_mixer
    plain = ConditionalDecoderMLX(**TINY)
    assert "time_embed_mixer" not in plain


def test_convert_weights_maps_time_embed_mixer():
    dec = _tiny_decoder(meanflow=True)
    key = "flow.decoder.estimator.time_embed_mixer.weight"
    conv = convert_s3gen_state_dict({key: dec.time_embed_mixer.weight.detach()})
    assert list(conv) == [key]
    np.testing.assert_array_equal(conv[key], dec.time_embed_mixer.weight.detach().numpy())


def test_mlx_decoder_with_r_matches_torch():
    dec = _tiny_decoder(meanflow=True)
    mdec = _to_mlx_decoder(dec)
    x, mask, mu, spks, cond = _inputs()
    t, r = torch.tensor([0.25]), torch.tensor([0.75])
    with torch.inference_mode():
        exp = dec(x, mask, mu, t, spks, cond, r=r).numpy()
    out = mdec(*_mx(x, mask, mu, t, spks, cond), r=mx.array(r.numpy()))
    np.testing.assert_allclose(np.array(out), exp, rtol=1e-4, atol=1e-4)


def test_mlx_meanflow_solve_matches_torch():
    dec = _tiny_decoder(meanflow=True)
    mdec = _to_mlx_decoder(dec)
    cfm = CausalConditionalCFM(spk_emb_dim=80, estimator=dec)
    mcfm = CausalConditionalCFMMLX(spk_emb_dim=80, estimator=mdec)
    _, mask, mu, spks, cond = _inputs()
    noise = torch.randn(1, 80, T, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        exp, _ = cfm(mu, mask, 2, spks=spks, cond=cond, noised_mels=noise, meanflow=True)
    out, _ = mcfm(
        *_mx(mu, mask),
        2,
        spks=mx.array(spks.numpy()),
        cond=mx.array(cond.numpy()),
        noised_mels=mx.array(noise.numpy()),
        meanflow=True,
    )
    np.testing.assert_allclose(np.array(out), exp.numpy(), rtol=1e-4, atol=1e-4)


def test_mlx_non_meanflow_midpoint_matches_torch():
    """Default (CFG midpoint) path still agrees across backends."""
    dec = _tiny_decoder(meanflow=False)
    mdec = _to_mlx_decoder(dec)
    cfm = CausalConditionalCFM(spk_emb_dim=80, estimator=dec)
    mcfm = CausalConditionalCFMMLX(spk_emb_dim=80, estimator=mdec)
    _, mask, mu, spks, cond = _inputs()
    z = torch.randn(1, 80, T, generator=torch.Generator().manual_seed(4))
    ts = 1 - torch.cos(torch.linspace(0, 1, 6) * 0.5 * torch.pi)
    with torch.inference_mode():
        exp = cfm.solve_midpoint(z, ts, mu, mask, spks, cond)
    out = mcfm.solve_midpoint(*_mx(z, ts, mu, mask, spks, cond))
    np.testing.assert_allclose(np.array(out), exp.numpy(), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------
# S3Gen-level plumbing (stubbed, no model construction)
# --------------------------------------------------------------------------


def test_torch_s3gen_flow_inference_plumbing():
    from chatterbox.models.s3gen.s3gen import S3Token2Mel, S3Token2Wav

    seen = {}

    def fake_forward(self, speech_tokens, **kw):
        seen.update(kw)
        return torch.zeros(1)

    obj = S3Token2Wav.__new__(S3Token2Wav)
    torch.nn.Module.__init__(obj)
    obj.flow = types.SimpleNamespace(token_mel_ratio=2, pre_lookahead_len=3)
    type(obj).device = property(lambda self: torch.device("cpu"))
    orig = S3Token2Mel.forward
    S3Token2Mel.forward = fake_forward
    try:
        toks = torch.zeros(1, 7, dtype=torch.long)
        obj.meanflow = True
        obj.flow_inference(toks, ref_dict={}, finalize=True)
        assert seen["n_cfm_timesteps"] == 2
        assert seen["noised_mels"].shape == (1, 80, 14)
        obj.flow_inference(toks, ref_dict={}, finalize=False, n_cfm_timesteps=4)
        assert seen["n_cfm_timesteps"] == 4
        assert seen["noised_mels"].shape == (1, 80, 8)
        obj.meanflow = False
        obj.flow_inference(toks, ref_dict={}, finalize=True)
        assert seen["n_cfm_timesteps"] is None and seen["noised_mels"] is None
    finally:
        S3Token2Mel.forward = orig
        del type(obj).device


@pytest.mark.parametrize("meanflow,n,expected", [(True, None, 2), (True, 3, 3), (False, None, None), (False, 7, 7)])
def test_mlx_s3gen_plumbing(meanflow, n, expected):
    from chatterbox.models.s3gen_mlx.s3gen_mlx import S3Token2MelMLX

    seen = {}

    def fake_flow(**kw):
        seen.update(kw)
        return mx.zeros((1, 80, 2)), None

    obj = types.SimpleNamespace(meanflow=meanflow, flow=fake_flow)
    ref = dict(prompt_token=None, prompt_token_len=0, prompt_feat=None, embedding=None)
    S3Token2MelMLX.__call__(obj, mx.zeros((1, 3)), ref, True, n_cfm_timesteps=n)
    assert seen["n_timesteps"] == expected
    assert seen["meanflow"] is meanflow


def test_s3gen_signatures_parallel():
    import inspect

    from chatterbox.models.s3gen.s3gen import S3Token2Wav
    from chatterbox.models.s3gen_mlx.s3gen_mlx import S3Token2MelMLX, S3Token2WavMLX

    assert inspect.signature(S3Token2Wav.__init__).parameters["meanflow"].default is False
    assert inspect.signature(S3Token2MelMLX.__init__).parameters["meanflow"].default is False
    for fn in (S3Token2Wav.inference, S3Token2WavMLX.inference):
        assert inspect.signature(fn).parameters["n_cfm_timesteps"].default is None


@pytest.mark.parametrize("time1", [1, 5, 37])
def test_rel_shift_matches_torch(time1):
    """MLX rel_shift must equal the torch reinterpretation, not a transpose."""
    from chatterbox.models.s3gen.transformer.attention import (
        RelPositionMultiHeadedAttention,
    )
    from chatterbox.models.s3gen_mlx.transformer.attention_mlx import (
        RelPositionMultiHeadedAttentionMLX,
    )

    x = np.random.default_rng(0).standard_normal((2, 3, time1, 2 * time1 - 1))
    x = x.astype(np.float32)
    ref = RelPositionMultiHeadedAttention.rel_shift(None, torch.from_numpy(x)).numpy()
    out = np.array(RelPositionMultiHeadedAttentionMLX.rel_shift(None, mx.array(x)))
    np.testing.assert_array_equal(out, ref)
