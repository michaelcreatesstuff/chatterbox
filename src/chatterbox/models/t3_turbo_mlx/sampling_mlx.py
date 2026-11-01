# Copyright (c) 2025 MichaelYangAI
# MIT License

"""
On-device sampling of Chatterbox-Turbo speech tokens (MLX, fixed shapes).

Port of the HF ``LogitsProcessorList`` that upstream ``T3.inference_turbo``
builds, in the same order:

1. ``TemperatureLogitsWarper`` (only if ``temperature > 0`` and ``!= 1``)
2. ``TopKLogitsWarper(top_k)`` (only if ``top_k > 0``)
3. ``TopPLogitsWarper(top_p)`` (only if ``top_p < 1``)
4. ``RepetitionPenaltyLogitsProcessor(penalty)`` (only if ``penalty != 1``)

then ``softmax`` + ``multinomial``. The repetition penalty runs last, so it
rescales the logits that survived top-k / top-p (filtered ones stay ``-inf``).
Its ``input_ids`` are a ``(B, V)`` boolean *presence* mask instead of a token
list: the penalty is applied once per distinct token, which is what HF's
gather / scatter does too.

Top-k / top-p use one ascending sort. Both remove a prefix of the sorted
values, so each becomes a value threshold: HF removes ``x < kth largest`` for
top-k, and for top-p everything below the smallest kept sorted value. The one
difference from HF is at exact ties on the top-p boundary, where HF's choice
depends on the sort order of equal values; here all tied values are kept.

Sampling is Gumbel-max with one PRNG key per row, so a row samples the same
tokens whether it decodes alone or in a batch (up to float rounding).
"""

from typing import Tuple

import mlx.core as mx


def process_logits(
    logits: mx.array,
    presence: mx.array,
    temperature: mx.array,
    top_p: mx.array,
    penalty: mx.array,
    *,
    use_temperature: bool,
    top_k: int,
    use_top_p: bool,
    use_penalty: bool,
) -> mx.array:
    """The HF processor chain on float32 ``(B, V)`` logits; filtered -> ``-inf``.

    ``temperature``, ``top_p`` and ``penalty`` are float32 0-d arrays (so a
    compiled trace serves every value); the flags and ``top_k`` are static.
    ``top_p`` holds ``1 - top_p`` computed on the host (see
    :func:`sampling_params`), as HF compares against the Python float.
    """
    V = logits.shape[-1]
    neg = mx.array(-mx.inf, mx.float32)
    x = logits
    if use_temperature:
        x = x / temperature
    asc = None
    if top_k > 0:
        k = min(int(top_k), V)
        asc = mx.sort(x, axis=-1)
        kth = asc[:, V - k : V - k + 1]
        x = mx.where(x < kth, neg, x)
        asc = mx.where(asc < kth, neg, asc)
    if use_top_p:
        if asc is None:
            asc = mx.sort(x, axis=-1)
        cum = mx.cumsum(mx.softmax(asc, axis=-1, precise=True), axis=-1)
        remove = cum <= top_p
        # min_tokens_to_keep = 1: the largest value always stays.
        remove = remove & (mx.arange(V) < V - 1)
        first_kept = remove.astype(mx.int32).sum(axis=-1, keepdims=True)
        cut = mx.take_along_axis(asc, first_kept, axis=-1)
        x = mx.where(x < cut, neg, x)
    if use_penalty:
        pen = mx.where(x < 0, x * penalty, x / penalty)
        x = mx.where(presence, pen, x)
    return x


def sampling_params(temperature: float, top_k: int, top_p: float, penalty: float):
    """``(arrays, flags)`` for :func:`sample_tokens` from the upstream arguments.

    The flags reproduce the ``if`` conditions under which upstream appends
    each processor.
    """
    arrays = (
        mx.array(float(temperature) if temperature > 0 else 1.0, mx.float32),
        mx.array(1.0 - float(top_p), mx.float32),
        mx.array(float(penalty), mx.float32),
    )
    flags = dict(
        use_temperature=bool(temperature > 0 and temperature != 1.0),
        top_k=int(top_k) if top_k > 0 else 0,
        use_top_p=bool(top_p < 1.0),
        use_penalty=bool(penalty != 1.0),
    )
    return arrays, flags


def gumbel_sample(x: mx.array, keys: mx.array) -> Tuple[mx.array, mx.array]:
    """One categorical draw per row of ``(B, V)`` logits; ``keys`` ``(B, 2)``.

    Returns ``(tokens (B,) int32, new_keys)``.
    """
    B, V = x.shape
    toks, new_keys = [], []
    for b in range(B):
        key, sub = mx.random.split(keys[b])
        g = mx.random.gumbel(shape=(V,), key=sub)
        toks.append(mx.argmax(x[b] + g))
        new_keys.append(key)
    return mx.stack(toks).astype(mx.int32), mx.stack(new_keys)


def onehot_mask(tokens: mx.array, V: int) -> mx.array:
    """``(B,)`` ids -> ``(B, V)`` boolean one-hot."""
    return mx.arange(V)[None, :] == tokens[:, None]


def sample_tokens(
    logits: mx.array,
    presence: mx.array,
    finished: mx.array,
    keys: mx.array,
    temperature: mx.array,
    top_p: mx.array,
    penalty: mx.array,
    *,
    stop_id: int,
    use_temperature: bool,
    top_k: int,
    use_top_p: bool,
    use_penalty: bool,
):
    """Process, sample and update the per-row decode state.

    A finished row keeps emitting ``stop_id``. Returns ``(tokens, presence,
    finished, keys)`` where ``presence`` now also holds the new tokens.
    """
    x = process_logits(
        logits,
        presence,
        temperature,
        top_p,
        penalty,
        use_temperature=use_temperature,
        top_k=top_k,
        use_top_p=use_top_p,
        use_penalty=use_penalty,
    )
    tok, keys = gumbel_sample(x, keys)
    stop = mx.array(stop_id, mx.int32)
    tok = mx.where(finished, stop, tok)
    finished = finished | (tok == stop)
    presence = presence | onehot_mask(tok, logits.shape[-1])
    return tok, presence, finished, keys
