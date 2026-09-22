# Short-chunk hallucination in `generate()` — analysis and proposed fix

Written from a downstream investigation (translate-chatterbox bulk
pipeline). Everything below is measured on this fork's shipped code
(`chatterbox-mlx` 1.0.4, installed from PyPI), single voice, English,
Apple Silicon / MPS.

## Symptom

Short scripts synthesized through `ChatterboxMultilingualTTSMLX.generate()`
come back with invented words spliced between correct sentences. ASR
transcripts of real output:

```
reference : No obstacle halts me. I charge forward.
heard     : No obstacle holds me, be said. I charge forward. Cue's efface of eye.

reference : My mindset is bright. Good energy flows to me.
heard     : My mindset is bright Good energy flows to me. Well may and ...
```

Long-form output is unaffected.

## Root cause

`generate()` splits its input into sentences and synthesizes each one as
an independent generation, with no minimum size:

- `src/chatterbox/mtl_tts_mlx.py:531` — `split_into_sentences(text, lang)`
- `:537-539` — `punc_norm` per sentence, drop empties
- `:610-630` — loop calling `_generate_single` per sentence

A 41-word script therefore becomes **9 separate generations of 2-4
words each**.

Each of those gets a token budget from
`src/chatterbox/generation_utils.py::estimate_max_tokens`, which floors
at 80 tokens for `word_count <= 9`. At `S3_TOKEN_RATE = 25`
(`src/chatterbox/models/s3tokenizer/s3tokenizer.py:23`) that is **3.2
seconds of audio budget**. "I charge forward." needs about 1 second. The
remaining ~2 seconds is headroom the model fills with speech-shaped
babble when EOS does not fire promptly.

The fingerprint is unambiguous: across 240 generations, **every failing
short chunk terminated at exactly 3.06s** — the 80-token ceiling minus
the lookahead trim — regardless of whether the text was 2, 3, or 4
words.

`generate_long()` does not have this problem because it groups
sentences to ~50 words via `get_adaptive_chunks`. Only the short-form
path is exposed.

## Measurements

240 generations per condition, ASR-scored with `whisper-small-mlx`
(temperature 0, `condition_on_previous_text=False`), counting alignment
insertions against the reference.

| condition | hallucination rate |
|---|---|
| 2-word chunk | 10.0% (8/80) |
| 3-word chunk | 33.8% (27/80) |
| 4-word chunk | 10.0% (8/80) |
| 9-word merged chunk | 1.2% (1/80) |

| weights | short-chunk rate | significance |
|---|---|---|
| v2 (`t3_mtl23ls_v2`) | 17.9% (43/240) | baseline |
| v3 (`t3_mtl23ls_v3`) | 11.7% (28/240) | z=1.93, **p=0.054 — not significant** |
| merged text (v2) | 1.2% (1/80) | p<0.001 vs v2, p=0.005 vs v3 |

Per item containing three short chunks: **44.7% → 3.7%** with merging.

Note the word-count buckets each use one distinct sentence, so length
and phrasing are confounded; the supported claim is short (2-4 words)
≫ longer (9 words), not a precise per-word curve.

## Proposed fixes, in priority order

### A. Merge short sentences in `generate()` — the actual fix

At `mtl_tts_mlx.py:539`, after the empty-sentence filter, group adjacent
sentences to a minimum word count before the generation loop.
`generation_utils.get_adaptive_chunks` already implements the grouping;
it is simply gated behind `ADAPTIVE_THRESHOLD_WORDS = 50` and only
reachable from `generate_long()`. Applying the same grouping
unconditionally takes the example script from 9 chunks to 2.

This also improves prosody — 2-word chunks are synthesized with no
cross-sentence context.

### B. Tighten the token floor

`estimate_max_tokens` floors at 80 tokens (3.2s) for `word_count <= 9`.
That is far more headroom than a 3-word utterance can justify. Scaling
the floor with word count, or capping absolute headroom, bounds the
damage of any EOS failure to a fraction of a second.

Note the working tree currently has an uncommitted rewrite of this
function using `max(word_count * 25, 100)` — a **100**-token floor
(4.0s), which is more generous than the 80 that shipped in 1.0.4.

### C. Gate EOS suppression on the sticky completion flag

`src/chatterbox/models/t3_mlx/inference/alignment_stream_analyzer_mlx.py:258`

```python
if cur_text_posn < S - 3 and S > 5:      # current
if not self.complete and cur_text_posn < S - 3 and S > 5:   # proposed
```

`self.complete` is sticky (`:186`) but suppression tests the
instantaneous `argmax` position (`:163`). Once attention goes diffuse at
end-of-utterance it can drift backwards, re-triggering hard EOS
suppression (`-32768`) on an utterance the analyzer already considers
finished. With `S = 13-20` for short chunks, `S > 5` never disengages.

Related: monotonic masking is disabled at `:151`, whereas the PyTorch
reference still applies it
(`models/t3/inference/alignment_stream_analyzer.py:107-108`). The mask
governs most of a short utterance, so removing it leaves `cur_text_posn`
free to bounce. The comment attributes "infinite EOS suppression" to the
mask; that symptom is better explained by the un-gated suppression in
`:258`, so restoring the mask together with the `not self.complete`
guard is worth testing as a pair.

### D. Correction to `DEBUG_EXTRA_TEXT.md`

That document investigates this exact symptom and concludes
"ROOT CAUSE: Encoder Upsampling", based on "90 tokens → 1.92s"
(~47 tokens/s). The actual rate is `S3_TOKEN_RATE = 25`, so 90 tokens is
**3.60s** against 3.48s measured, and 161 tokens is **6.44s** against
6.32s measured. S3Gen output matches its token count to within the
lookahead trim — there is no 81% over-generation, and the
`Upsample1D(stride=2)` in `upsample_encoder.py:236` is the by-design
25Hz→50Hz mel ratio. The investigation appears to have stopped on that
unit error, which is why no fix landed.

### E. Observability

The single highest-value log, in the loop at `mtl_tts_mlx.py:610-630`:
per chunk, emit index, `repr(sentence)`, word count, `S`, the
`max_new_tokens` actually used, speech tokens returned, **whether EOS
fired**, and chunk duration. The signature of this bug is
`tokens_generated == max_new_tokens` with no EOS.

Today `t3_mlx.py:527` logs EOS detection at `info` — its *absence* is
the signal, which is not greppable. `mtl_tts_mlx.py:801-804` already has
a "No EOS token found" branch but it is behind `CHATTERBOX_DEBUG` and
printed to stdout rather than logged.

## Downstream workaround (already shipped in translate-chatterbox)

`pipeline/coalesce.py` merges adjacent sentences to >=8 words and joins
them with commas before calling `generate()`, so the internal splitter
keeps each group intact. End-to-end on three previously-defective items:
mean WER 0.205 → 0.051, mean insertions 6.3 → 0.3, all high-WER flags
cleared.

This is a workaround, not a fix — the token headroom is still there, it
is just no longer being handed to 3-word utterances. Fix A belongs here,
where it also benefits the backend and CLI consumers.

Two gotchas worth knowing if the merge lands upstream: a terminator
regex of `[.!?]+$` does not match `today!"`, and CSV-derived scripts
carry quote characters mid-string where per-sentence edge-stripping
cannot reach them.
