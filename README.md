# Chatterbox MLX - Apple Silicon Optimized TTS

[![PyPI version](https://badge.fury.io/py/chatterbox-mlx.svg)](https://badge.fury.io/py/chatterbox-mlx)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An MLX-optimized fork of [Resemble AI's Chatterbox TTS](https://github.com/resemble-ai/chatterbox) for Apple Silicon, delivering up to 2.4x faster inference.**

---

## Installation

```bash
pip install chatterbox-mlx
```

### Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+ (tested primarily with 3.11.12. Also tested with 3.12.12)
- ~4GB disk space for model weights

**Important:** Python must be compiled with `lzma` support. If you're using pyenv:

```bash
# Install xz library first (provides liblzma)
brew install xz

# Then install Python (or reinstall if already installed)
pyenv install 3.11.12  # or your preferred version
```

If you see an error about `ModuleNotFoundError: No module named '_lzma'`, you need to install `xz` and reinstall Python.

---

## CLI Usage

Generate speech directly from the terminal:

```bash
    # Generate English speech (auto-generated filename):
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing."

    # Generate Spanish speech:
    chatterbox "La inteligencia artificial ha logrado avances notables en los últimos años." --lang es

    # Use the --voice flag to provide a reference audio file for voice cloning:
    chatterbox "Artificial intelligence has made remarkable strides in recent years, particularly in the field of natural language processing." --voice speaker.wav

    # Run multilingual benchmark (saves to benchmark_output/)
    chatterbox --benchmark --languages en es
```

### CLI Options

| Option            | Description                                  | Default           |
| ----------------- | -------------------------------------------- | ----------------- |
| `-o, --output`    | Output WAV file path                         | Auto-generated    |
| `-l, --lang`      | Language code (en, es, fr, de, ja, zh, etc.) | `en`              |
| `-v, --voice`     | Reference audio for voice cloning            | None              |
| `--exaggeration`  | Emotion intensity (0.0-1.0)                  | `0.5`             |
| `--cfg`           | Classifier-free guidance weight              | `0.5`             |
| `--backend`       | Backend: hybrid-mlx, mlx, pytorch            | `hybrid-mlx`      |
| `--benchmark`     | Run multilingual benchmark                   | False             |
| `--languages`     | Languages to benchmark                       | en es fr de ja zh |
| `--no-save-audio` | Don't save benchmark audio files             | False (saves)     |
| `-q, --quiet`     | Suppress progress messages                   | False             |

---

## Quick Start

```python
import torchaudio as ta
from chatterbox.tts_mlx import ChatterboxTTSMLX

# Load model (downloads weights automatically on first run). T3 runs on MLX;
# device sets where S3Gen runs: "mps" (default, Apple GPU) or "cpu".
model = ChatterboxTTSMLX.from_pretrained(device="mps")


# Generate speech
text = "Hello! This is Chatterbox running with MLX optimization on Apple Silicon."
wav = model.generate(text)
ta.save("output.wav", wav, model.sr)

# Voice cloning with reference audio
wav = model.generate(
    text,
    audio_prompt_path="reference_voice.wav",
    exaggeration=0.5,  # Emotion intensity (0.0-1.0)
    cfg_weight=0.5,    # Classifier-free guidance
)
```

### Long-Form Audio Generation

For texts longer than ~50 words, use chunked generation:

```python
long_text = """
Your long text here. It can span multiple paragraphs and sentences.
The generate_long method will automatically split it at sentence boundaries,
generate each chunk separately, and crossfade them together seamlessly.
"""

wav = model.generate_long(
    long_text,
    audio_prompt_path="reference_voice.wav",
    chunk_size_words=50,
    overlap_duration=0.1,
)
ta.save("long_output.wav", wav, model.sr)
```

### Chatterbox-Nano and Chatterbox-Turbo (fastest, English)

[Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo) (350M) and [Chatterbox-Nano](https://huggingface.co/ResembleAI/chatterbox-nano) (110M) use a GPT-2 T3 decoder and the meanflow S3Gen vocoder. Both models use one class. This port runs the T3 decode loop, the sampling and the vocoder in MLX.

```python
import soundfile as sf
from chatterbox.tts_turbo_mlx import ChatterboxTurboMLX

model = ChatterboxTurboMLX.from_pretrained(nano=True)  # nano=False loads Turbo
# Optional: clone a voice (longer than 5 s). Without it, the built-in voice is used.
model.prepare_conditionals("reference_voice.wav")

wav = model.generate("Hello! [chuckle] This is Chatterbox-Nano on Apple Silicon.", seed=0)
sf.write("nano.wav", wav, model.sr)
print(model.last_timings)  # t3_s, s3gen_s, watermark_s, n_speech_tokens, n_chunks, ...
```

`generate()` splits long text at sentence boundaries. It decodes up to 8 chunks together as one batch, then crossfades them. Each chunk has a token limit from its word count, so one chunk cannot run on. The output has a Perth watermark, the same as upstream. Set `watermark=False` to skip it.

Measured on an Apple M4 (32 GB), fp16. RTF is generation time divided by audio time, so lower is faster. WER comes from Whisper large-v3-turbo. WER ranges cover three LibriSpeech voices (two female, one male), the built-in voice (Nano and Turbo only), and two seeds.

| System | Medium text (13-15 s audio) | 292-word text (~1:35-2:00 audio) |
| --- | --- | --- |
| **Chatterbox-Nano, this port** | **RTF 0.12 (8.3x real time), WER 0.00** | **RTF 0.083 (12x), WER 0.005-0.007** |
| Chatterbox-Turbo, this port | RTF 0.25 (4.0x), WER 0.00-0.03 | RTF 0.12 (8.2x), WER 0.002-0.012 |
| Chatterbox-Flash, this port | RTF 0.37 (2.7x), WER 0.00-0.23 | RTF 0.31 (3.2x), WER 0.00-0.003 |
| Chatterbox-Nano, upstream PyTorch (MPS) | RTF 0.32 (3.2x), WER 0.00-0.03 | collapses (no sentence splitting) |
| Chatterbox-Turbo, mlx-audio fp16 | RTF 0.51 (2.0x) | RTF 0.53 (1.9x) |

For the 292-word text, the WER is the mean over the sentence chunks. Whisper adds repeated words to clips longer than about 30 s, so a WER of the full clip is not reliable.

Useful options:

- `quantize_bits` (load time): `8` makes Turbo about 30% faster on medium text. `4` makes Nano about 25% faster on medium text. Do not use `4` with Turbo: long text runs on and its WER goes up. The quantization WER checks used a synthetic reference only. Listen to the output before you use quantization.
- `max_batch_chunks` (default 8): set 1 to decode chunks one by one. Long text is then 1.5-2x slower.
- `release_caches()`: frees the KV caches. The next call rebuilds them.

Known limits:

- Use a reference clip of clear speech, 10-15 s long. A non-speech clip (a tone or music) makes the model babble at the start of each sentence chunk.
- `cfg_weight`, `exaggeration` and `min_p` have no effect, the same as upstream Turbo.

#### Streaming (Nano and Turbo)

Use `generate_stream()` to play audio before the full text is done. It yields 1-D float32 pieces at 24 kHz (`model.sr`). Play the pieces in order. Together they make one waveform.

```python
import numpy as np
import sounddevice as sd  # any audio output works

with sd.OutputStream(samplerate=model.sr, channels=1, dtype="float32") as out:
    for piece in model.generate_stream("Hello! This is Chatterbox-Nano, streaming.", seed=0):
        out.write(piece[:, None])
print(model.last_timings["first_audio_s"])  # time from the call to the first piece
```

How it works:

1. The text is split into the same sentence chunks as `generate()`.
2. The first chunk decodes alone. While T3 decodes it, S3Gen converts token windows to audio. The first window has `first_chunk_tokens` tokens (default 25, 1 s of audio). The next window has `chunk_tokens` tokens (default 50). Each window after that has double the tokens, up to 200.
3. Each window also converts `context_tokens` tokens (default 50) on its left and decodes `lookahead_tokens` tokens (default 5) past its end. Consecutive windows crossfade over 40 ms. This keeps the joins clean.
4. The other chunks decode in batch groups of up to `max_batch_chunks` chunks. Each chunk is yielded when its audio is ready.
5. The chunks crossfade the same way as in `generate()`.

Options:

- `first_chunk_tokens=0`: turn off token windows. The first chunk is then yielded whole.
- `watermark=True` (default): each piece gets its own Perth watermark. The watermarker sees the last 0.25 s of the previous piece as context.

Rules:

- Exhaust the generator or call `close()` on it. The generator holds the model lock until then, so other threads wait.
- Do not call `generate()`, `generate_stream()`, `prepare_conditionals()` or `release_caches()` on the same model inside the loop. These calls raise `RuntimeError`, because the open stream uses the same KV cache.
- Do not expect the same samples as `generate()` with the same seed. The window joins differ, and the first chunk decodes at batch size 1. A first chunk that ends before its first window gives the same audio as `generate()`.

### Chatterbox-Flash (English)

[Chatterbox-Flash](https://huggingface.co/ResembleAI/chatterbox-flash) replaces the autoregressive T3 decoder with a block-diffusion decoder. It unmasks a block of speech tokens in parallel per step. This port runs the full T3 loop and the meanflow S3Gen vocoder in MLX.

```python
import soundfile as sf
from chatterbox.tts_flash_mlx import ChatterboxFlashMLX

model = ChatterboxFlashMLX.from_pretrained()  # MLX vocoder by default
model.prepare_conditionals("reference_voice.wav")  # required: Flash has no built-in voice

wav = model.generate("Hello! This is Chatterbox-Flash running on Apple Silicon.", seed=0)
sf.write("flash.wav", wav, model.sr)
print(model.last_timings)  # t3_s, s3gen_s, n_speech_tokens, n_chunks, ...
```

`generate()` splits text over 35 words at sentence boundaries. It decodes up to 4 chunks together as one batch, then crossfades them. You do not need a separate long-form method.

Measured on an Apple M4 (32 GB), fp16. RTF is generation time divided by audio time, so lower is faster. The WER is the range over three LibriSpeech voices and two seeds, from Whisper large-v3-turbo.

| System | Medium text (15 s audio) | 292-word text (~1:50 audio) |
| --- | --- | --- |
| **Chatterbox-Flash, this port** | **RTF 0.37 (2.7x real time)** | **RTF 0.31 (3.2x), WER 0.00-0.003** |
| Chatterbox-Turbo, mlx-audio fp16 | RTF 0.51 (2.0x) | RTF 0.53 (1.9x) |
| Chatterbox-Flash, upstream MLX backend | RTF 0.99 (1.0x) | not run (RTF 1.25 on a 75-word text) |

Useful options:

- `block_size` (load time, default 24): diffusion block length. Set 16 for the paper configuration. Set 32 for about 20% more speed at a small WER cost.
- `preset` (`"paper"`, `"balanced"`, `"fast"`): early-decoding strength. `"paper"` is the default. Stronger presets use fewer steps per block, at some quality cost on short text.
- `max_batch_chunks` (default 4): set 1 to decode chunks one by one.
- `release_caches()`: frees the KV caches after a long document. The next call rebuilds them.

Known limits:

- Keep `cfg_scale=1.0`. Without guidance, long text loops and the pitch drifts.
- With some voices, Flash adds a few babbled words after the end of a short text (WER 0.23 on medium text with one of three test voices).
- 4-bit quantization breaks long text. Do not use it with Flash.
- By default, Flash output has no watermark, the same as upstream Flash. Set `watermark=True` in `generate()` to add a Perth watermark.

## 🙏 Acknowledgements

This project is built on top of the excellent **[Chatterbox TTS](https://github.com/resemble-ai/chatterbox)** by [Resemble AI](https://resemble.ai). I'm deeply grateful for their work in creating and open-sourcing a production-grade, multilingual text-to-speech system under the MIT license.

**This fork focuses specifically on MLX optimizations for Apple Silicon.** If you're looking for the original project with CUDA support and the full feature set, please visit the [official Resemble AI repository](https://github.com/resemble-ai/chatterbox).

---

## What's Different in This Fork?

This package provides **native MLX acceleration** for Apple Silicon Macs, achieving significant performance improvements:

| Text Length       | CPU Baseline | MLX Optimized | Speedup         |
| ----------------- | ------------ | ------------- | --------------- |
| Short (5 words)   | 8.91s        | 3.70s         | **2.4x faster** |
| Medium (31 words) | 57.51s       | 24.40s        | **2.4x faster** |
| Long (94 words)   | 137.92s      | 62.66s        | **2.2x faster** |

### Key Optimizations

- **MLX-Native T3 Model**: The 520M parameter Llama 3 backbone runs entirely on MLX
- **Float16 KV Cache**: Up to 5.8 GB memory savings with 32% faster generation
- **Hybrid Architecture**: Combines MLX speed with PyTorch quality controls
- **Long-Form Generation**: Intelligent chunking with crossfade for extended audio

---

## Benchmark Results

All benchmarks run on **Apple M4 (32GB RAM), macOS 15.4, Python 3.11, PyTorch 2.8.0**.

### English TTS Performance

| Device         | Text   | Words | Time    | RTF   |
| -------------- | ------ | ----- | ------- | ----- |
| **Hybrid-MLX** | short  | 5     | 4.08s   | 0.65x |
| **Hybrid-MLX** | medium | 31    | 25.24s  | 0.73x |
| **Hybrid-MLX** | long   | 94    | 62.66s  | 0.74x |
| Pure MLX       | short  | 5     | 3.70s   | 0.69x |
| Pure MLX       | medium | 31    | 24.40s  | 0.72x |
| Pure MLX       | long   | 94    | 68.82s  | 0.71x |
| CPU            | short  | 5     | 8.91s   | 0.27x |
| CPU            | medium | 31    | 57.51s  | 0.33x |
| CPU            | long   | 94    | 137.92s | 0.34x |

**Key findings:**

- **Hybrid-MLX** recommended for production (best quality/speed balance)
- **Pure MLX** fastest for short texts, but quality degrades on long texts
- **2.2-2.4x speedup** vs CPU baseline across all text lengths

### Multilingual Performance

| Device     | Language | Time   | RTF   |
| ---------- | -------- | ------ | ----- |
| Hybrid-MLX | English  | 12.25s | 0.71x |
| Hybrid-MLX | Spanish  | 14.74s | 0.76x |
| Pure MLX   | English  | 14.55s | 0.67x |
| Pure MLX   | Spanish  | 13.78s | 0.75x |
| MPS        | English  | 19.96s | 0.50x |
| MPS        | Spanish  | 21.06s | 0.51x |
| CPU        | English  | 25.64s | 0.32x |
| CPU        | Spanish  | 31.31s | 0.33x |

### Visual Comparison

```
                    GENERATION TIME COMPARISON

     Short (5 words)
     ├─ CPU        ████████████████████████████████████████ 8.91s
     ├─ Hybrid-MLX ██████████████████ 4.08s (2.2x faster)
     └─ Pure MLX   ████████████████ 3.70s (2.4x faster)

     Medium (31 words)
     ├─ CPU        ████████████████████████████████████████ 57.51s
     ├─ Hybrid-MLX █████████████████ 25.24s (2.3x faster)
     └─ Pure MLX   ████████████████ 24.40s (2.4x faster)

     Long (94 words)
     ├─ CPU        ████████████████████████████████████████ 137.92s
     ├─ Hybrid-MLX █████████████████ 62.66s (2.2x faster)  ✓ Best quality
     └─ Pure MLX   ██████████████████ 68.82s (2.0x faster)
```

### Backend Comparison

| Backend         | Description                    | RTF   | Memory | Recommendation       |
| --------------- | ------------------------------ | ----- | ------ | -------------------- |
| **Hybrid-MLX**  | T3 (MLX) + S3Gen (PyTorch/MPS) | 0.74x | ~16GB  | ✅ Production use    |
| **Pure MLX**    | Everything on MLX              | 0.71x | ~14GB  | Minimal dependencies |
| **PyTorch MPS** | Full PyTorch on MPS            | 0.51x | ~14GB  | Fallback             |
| **CPU**         | PyTorch on CPU                 | 0.34x | ~14GB  | Baseline             |

_RTF = Real-Time Factor (audio_duration / generation_time). Higher is better._

---

## Running Benchmarks

You can reproduce these benchmarks on your own hardware.

### English TTS Benchmark

```bash
# Full benchmark (all backends)
python benchmark_mps.py --runs 3 --validate

# Quick test with Hybrid-MLX only
python benchmark_mps.py --hybrid-mlx-only --runs 1

# CPU baseline only
python benchmark_mps.py --cpu-only --runs 1

# With voice cloning
python benchmark_mps.py --audio-prompt speaker.wav --runs 3

# Enable memory debugging
DEBUG_MEMORY=1 python benchmark_mps.py --hybrid-mlx-only
```

**Options:**
| Flag | Description |
|------|-------------|
| `--warmup N` | Warmup runs before timing (default: 1) |
| `--runs N` | Number of timed benchmark runs (default: 3) |
| `--devices` | Backends to test: `mps`, `cpu`, `hybrid-mlx`, `mlx`, `mlx-q4` |
| `--audio-prompt FILE` | Reference audio for voice cloning |
| `--output-dir DIR` | Output directory (default: `benchmark_output/`) |
| `--validate` | Enable Whisper transcription validation (computes WER) |
| `--mps-only` | Only benchmark PyTorch MPS |
| `--cpu-only` | Only benchmark CPU |
| `--hybrid-mlx-only` | Only benchmark Hybrid-MLX |
| `--mlx-only` | Only benchmark Pure MLX |
| `--debug-memory` | Enable detailed memory logging |

### Multilingual Benchmark

```bash
# Test specific languages
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es fr de ja zh \
    --runs 3

# Quick test with Hybrid-MLX
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es \
    --hybrid-mlx-only

# With validation
python benchmark_multilingual.py \
    --audio-prompt speaker.wav \
    --languages en es fr \
    --validate
```

**Supported Languages:**
`en` (English), `es` (Spanish), `fr` (French), `de` (German), `it` (Italian), `pt` (Portuguese), `ru` (Russian), `ja` (Japanese), `zh` (Chinese), `ko` (Korean), `ar` (Arabic), `hi` (Hindi), `tr` (Turkish), `pl` (Polish), `nl` (Dutch), `sv` (Swedish), `da` (Danish), `no` (Norwegian), `fi` (Finnish), `el` (Greek), `he` (Hebrew), `ms` (Malay), `sw` (Swahili)

### Benchmark Output

Results are saved to:

- `benchmark_output/benchmark_results.json` - English TTS results
- `benchmark_multilingual_output/multilingual_results.json` - Multilingual results
- Generated audio files: `{device}_{category}.wav`

---

## Architecture

Chatterbox is a two-stage TTS pipeline. This fork accelerates the most compute-intensive component (T3) with MLX:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CHATTERBOX MLX PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ VoiceEncoder │    │     T3       │    │       S3Gen          │   │
│  │  (PyTorch)   │───▶│    (MLX)     │───▶│   (PyTorch/MPS)      │   │
│  │   ~2M params │    │  520M params │    │     ~80M params      │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
│                            ▲                                        │
│                            │                                        │
│                    2.4x faster with MLX                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Supported Languages

All 23 languages from the original Chatterbox are supported:

Arabic • Danish • German • Greek • English • Spanish • Finnish • French • Hebrew • Hindi • Italian • Japanese • Korean • Malay • Dutch • Norwegian • Polish • Portuguese • Russian • Swedish • Swahili • Turkish • Chinese

```python
from chatterbox.mtl_tts_mlx import ChatterboxMultilingualTTSMLX

model = ChatterboxMultilingualTTSMLX.from_pretrained(device="mps")

# French
wav = model.generate("Bonjour, comment ça va?", language_id="fr")

# Japanese
wav = model.generate("こんにちは、元気ですか？", language_id="ja")
```

By default, multilingual output has no watermark, the same as upstream. Set `watermark=True` in `generate()`, `generate_long()` or `generate_stream()` to add a Perth watermark. The stream watermarks each piece and keeps the piece lengths. The watermarker loads on first use, so it costs nothing when it is off.

---

## Tips for Best Results

### General Use

- Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most cases
- Ensure reference audio matches target language to avoid accent transfer

### Expressive Speech

- Lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7) for dramatic delivery
- Higher exaggeration speeds up speech; lower cfg_weight compensates

### Memory Usage

Enable debug logging to monitor memory:

```bash
DEBUG_MEMORY=1 python your_script.py
```

---

## Differences from Original Chatterbox

| Feature             | Original (Resemble AI) | This Fork              |
| ------------------- | ---------------------- | ---------------------- |
| **Target Hardware** | NVIDIA CUDA            | Apple Silicon          |
| **ML Framework**    | PyTorch                | MLX + PyTorch hybrid   |
| **T3 Inference**    | PyTorch                | MLX (2.4x faster)      |
| **KV Cache**        | Float32                | Float16 (32% faster)   |
| **Long-form Audio** | Basic                  | Chunked with crossfade |

---

## Credits & Links

- **Original Project**: [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox)
- **Resemble AI**: [resemble.ai](https://resemble.ai) - For creating and open-sourcing this incredible TTS system
- **Demo**: [Hugging Face Space](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- **Evaluation**: [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

### Upstream Dependencies

- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [MLX](https://github.com/ml-explore/mlx)

---

## License

MIT License - Same as the original Chatterbox project.

---

## Citation

If you use this project, please cite the original Chatterbox:

```bibtex
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
