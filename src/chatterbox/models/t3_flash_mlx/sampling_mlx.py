# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
Per-step sampling, PMI scoring and unmasking for Chatterbox-Flash (MLX).

Port of ``chatterbox_flash.model._pmi_count_early_step_unmask`` (with
``calibration.compute_pmi`` and ``cfg_guidance``) for a single sample. Every
data-dependent branch of the reference (variable-size masked subsets, ``topk``
of a data-dependent ``k``, ``torch.quantile`` over the masked subset) is
rewritten with fixed shapes so the whole step compiles with ``mx.compile`` and
runs on the GPU without host round trips:

* the masked subset is handled by sorting with ``+inf`` / ``-inf`` fill values;
* ``topk(n)`` becomes ``rank < n`` where ``rank`` comes from a double argsort;
* the quantile uses ``torch.quantile``'s linear interpolation and ``lerp``
  formula on the sorted masked values.

The step returns the new block tokens plus a 2-element ``flags`` array
``[n_mask, has_eos]`` that the caller reads once per step for early exit.
"""

from typing import Callable, Dict, List, Tuple

import mlx.core as mx

# calibration._LOG_EPS
LOG_EPS = 1e-8
# model._gumbel_sample
GUMBEL_EPS = 1e-10


def omnivoice_unmask_schedule(
    n_total_mask: int, num_steps: int, t_shift: float
) -> List[int]:
    """Per-step unmask counts under OmniVoice's time-shifted ``r_n`` curve.

    Exact copy of ``chatterbox_flash.model._omnivoice_unmask_schedule``.
    """
    K = max(1, int(num_steps))
    N = max(0, int(n_total_mask))
    if N == 0:
        return [0] * K
    if K == 1:
        return [N]
    tau = float(t_shift)
    if tau <= 0:
        base = N // K
        out = [base] * K
        out[-1] += N - sum(out)
        return out
    ts = [tau * (s / K) / (1.0 + (tau - 1.0) * (s / K)) for s in range(K + 1)]
    counts: List[int] = []
    cum = 0
    rem = N
    for s in range(K):
        if s == K - 1:
            num = rem
        else:
            target = int(round(N * ts[s + 1]))
            num = max(0, target - cum)
            num = min(num, rem)
        counts.append(num)
        cum += num
        rem -= num
    return counts


def time_shift_quantiles(num_steps: int, time_shift_tau: float) -> List[float]:
    """Early-decoding quantile levels ``q_k = max(0, 1 - tau * (k+1) / K)``."""
    K = int(num_steps)
    return [max(0.0, 1.0 - time_shift_tau * (k + 1) / K) for k in range(K)]


def masked_quantile(values: mx.array, valid: mx.array, q: mx.array) -> mx.array:
    """``torch.quantile(values[valid], q)`` (linear interpolation), fixed shape.

    Args:
        values: ``(L,)`` float32 values.
        valid: ``(L,)`` bool, at least one ``True``.
        q: float32 scalar in ``[0, 1]``.
    """
    L = values.shape[0]
    m = mx.sum(valid).astype(mx.float32)
    sorted_vals = mx.sort(mx.where(valid, values, mx.inf))
    rank = q * (m - 1.0)
    below = mx.floor(rank)
    above = mx.ceil(rank)
    weight = rank - below
    lo = mx.take(sorted_vals, mx.clip(below.astype(mx.int32), 0, L - 1))
    hi = mx.take(sorted_vals, mx.clip(above.astype(mx.int32), 0, L - 1))
    # torch.lerp: start + w * (end - start) if w < 0.5 else end - (end - start) * (1 - w)
    return mx.where(
        weight < 0.5, lo + weight * (hi - lo), hi - (hi - lo) * (1.0 - weight)
    )


def _log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def _pmi(probs: mx.array, sampled: mx.array, prior: mx.array) -> mx.array:
    """``log p_i(x_i) - log prior(x_i)`` with the reference's clamping."""
    p_i = mx.take_along_axis(probs, sampled[:, None], axis=-1)[:, 0]
    p_bar = mx.take(prior, sampled)
    return mx.log(mx.maximum(p_i, LOG_EPS)) - mx.log(mx.maximum(p_bar, LOG_EPS))


def make_step_fn(
    *,
    use_cfg: bool,
    sample: bool,
    position_noise: bool,
    use_quantile: bool,
    mask_id: int,
    stop_id: int,
) -> Callable:
    """Build one denoising step specialised on the structural flags.

    The returned function has signature::

        step(logits, blk, prior, key, n_sched, q, is_last, cfg_w, inv_temp)
            -> (new_blk, flags)

    with ``logits`` ``(B_fwd, bl, V)`` float32 (row 0 conditional, row 1
    unconditional when ``use_cfg``), ``blk`` ``(bl,)`` int32, ``prior`` ``(V,)``
    float32 and the remaining arguments 0-d arrays.
    """

    def step(logits, blk, prior, key, n_sched, q, is_last, cfg_w, inv_temp):
        k_sample, k_pos = mx.random.split(key)
        lc = logits[0]
        if use_cfg:
            lu = logits[1]
            lsc = _log_softmax(lc)
            lsu = _log_softmax(lu)
            guided = lsc + cfg_w * (lsc - lsu)
        else:
            guided = lc

        if sample:
            sampled = mx.random.categorical(guided * inv_temp, key=k_sample)
        else:
            sampled = mx.argmax(guided, axis=-1)
        sampled = sampled.astype(blk.dtype)

        # PMI against the precomputed unconditional block prior
        if use_cfg:
            pmi_c = _pmi(mx.softmax(lc, axis=-1), sampled, prior)
            pmi_u = _pmi(mx.softmax(lu, axis=-1), sampled, prior)
            pmi = (1.0 + cfg_w) * pmi_c - cfg_w * pmi_u
        else:
            pmi = _pmi(mx.softmax(lc, axis=-1), sampled, prior)

        is_mask = blk == mask_id
        if position_noise:
            # Reference: _gumbel_sample(zeros_like(pmi), T) == plain Gumbel noise
            # (0 / T vanishes), so T only switches the noise on.
            u = mx.random.uniform(shape=pmi.shape, key=k_pos)
            g = -mx.log(-mx.log(u + GUMBEL_EPS) + GUMBEL_EPS)
            pmi = mx.where(is_mask, pmi + g, pmi)

        # pmi_count_early: max(count schedule, quantile ceiling), capped at m_cur
        m_cur = mx.sum(is_mask).astype(mx.int32)
        n_sched_c = mx.minimum(mx.maximum(n_sched, 0), m_cur)
        if use_quantile:
            tau_k = masked_quantile(pmi, is_mask, q)
            n_quant = mx.sum(is_mask & (pmi >= tau_k)).astype(mx.int32)
            n_unmask = mx.minimum(mx.maximum(n_sched_c, n_quant), m_cur)
        else:
            n_unmask = n_sched_c

        # top-n_unmask masked positions by PMI (descending)
        score = mx.where(is_mask, pmi, -mx.inf)
        order = mx.argsort(-score)
        rank = mx.argsort(order)
        do_unmask = is_mask & (rank < n_unmask)
        do_unmask = mx.where(is_last, is_mask, do_unmask)

        new_blk = mx.where(do_unmask, sampled, blk)
        n_mask = mx.sum(new_blk == mask_id).astype(mx.int32)
        has_eos = mx.any(new_blk == stop_id).astype(mx.int32)
        return new_blk, mx.stack([n_mask, has_eos])

    return step


def make_batch_step_fn(n_chunks: int, rows_per_chunk: int, **flags) -> Callable:
    """Run :func:`make_step_fn`'s step independently for ``n_chunks`` chunks.

    The returned function has signature::

        step(logits, blk, prior, keys, n_sched, q, is_last, cfg_w, inv_temp,
             active) -> (new_blk, flags)

    with ``logits`` ``(n_chunks * rows_per_chunk, bl, V)`` (chunk ``c`` owns
    rows ``[c * rows_per_chunk, (c + 1) * rows_per_chunk)``, conditional row
    first), ``blk`` ``(n_chunks, bl)``, ``keys`` ``(n_chunks, 2)`` (one PRNG
    key per chunk), ``n_sched`` ``(n_chunks,)``, ``active`` ``(n_chunks,)``
    bool and the rest shared 0-d arrays. Each chunk runs the single-chunk step
    on its own rows with its own key, so sampling, PMI, the quantile and the
    top-k stay per chunk. An inactive chunk keeps its tokens. ``flags`` is
    ``(n_chunks, 2)`` of ``[n_mask, has_eos]``.
    """
    single = make_step_fn(**flags)
    R = int(rows_per_chunk)

    def step(logits, blk, prior, keys, n_sched, q, is_last, cfg_w, inv_temp, active):
        new_blks, all_flags = [], []
        for c in range(n_chunks):
            new_c, flags_c = single(
                logits[c * R : (c + 1) * R],
                blk[c],
                prior,
                keys[c],
                n_sched[c],
                q,
                is_last,
                cfg_w,
                inv_temp,
            )
            new_blks.append(mx.where(active[c], new_c, blk[c]))
            all_flags.append(flags_c)
        return mx.stack(new_blks), mx.stack(all_flags)

    return step


class StepFnCache:
    """Compiled step functions keyed by their structural flags."""

    def __init__(self, compile_fn: bool = True):
        self._compile = compile_fn
        self._fns: Dict[Tuple, Callable] = {}

    def get(self, **flags) -> Callable:
        key = tuple(sorted(flags.items()))
        fn = self._fns.get(key)
        if fn is None:
            fn = make_step_fn(**flags)
            if self._compile:
                fn = mx.compile(fn)
            self._fns[key] = fn
        return fn

    def get_batch(self, n_chunks: int, rows_per_chunk: int, **flags) -> Callable:
        """Compiled :func:`make_batch_step_fn` for ``n_chunks`` chunks."""
        key = ("batch", int(n_chunks), int(rows_per_chunk)) + tuple(
            sorted(flags.items())
        )
        fn = self._fns.get(key)
        if fn is None:
            fn = make_batch_step_fn(n_chunks, rows_per_chunk, **flags)
            if self._compile:
                fn = mx.compile(fn)
            self._fns[key] = fn
        return fn
