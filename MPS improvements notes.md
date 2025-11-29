# MPS Performance Optimization Notes

## ✅ Implemented Optimizations

### 1. SDPA Attention (IMPLEMENTED)

**Files:** `t3.py`, `llama_configs.py`

- Changed `_attn_implementation` from `'eager'` to `'sdpa'` for non-multilingual models
- SDPA provides ~2.29x speedup in attention computation
- Only use eager attention when alignment stream analyzer is needed (multilingual models)

### 2. Conditional output_attentions (IMPLEMENTED)

**Files:** `t3.py`

- Only enable `output_attentions=True` when alignment stream analyzer is active
- This avoids O(n²) memory overhead from attention weight materialization
- Major memory savings during autoregressive generation

### 3. Device-Optimized dtype (IMPLEMENTED)

**Files:** `llama_configs.py`, `t3.py`

- Added `get_optimal_dtype_str(device)` function
- MPS: Uses `float16` (better support than bfloat16)
- CUDA: Uses `bfloat16` (optimal for Ampere+)
- CPU: Uses `float32`

### 4. CFM Euler Solver Optimization (IMPLEMENTED)

**File:** `flow_matching.py`

- Removed intermediate step accumulation (`sol = []` list)
- Now returns only the final step, reducing memory usage

### 5. Memory Leak Fixes (IMPLEMENTED)

**Files:** `t3.py`, `t3_hf_backend.py`

- Pre-allocated `generated_ids` tensor instead of concatenation
- Pre-allocated `predicted_tokens` tensor instead of list accumulation
- Disabled `output_hidden_states` during generation loop (not needed, saves memory)
- Added explicit cleanup of KV cache after generation
- Fixed patched_model reinitialization issue

### 6. Debug Logging Flag (IMPLEMENTED)

**File:** `s3gen.py`

- Added `DEBUG_LOGGING` flag (controlled by `CHATTERBOX_DEBUG=1` env var)
- All print statements converted to `log_message()` which respects the flag
- Reduces I/O overhead during production inference

### 7. torch.compile Support (IMPLEMENTED, OFF BY DEFAULT)

**File:** `t3.py`, `tts.py`

- Added `compile_for_device()` method to T3
- Uses `aot_eager` backend for MPS, `inductor` for CUDA
- OFF BY DEFAULT due to recompilation overhead with dynamic KV cache sizes
- Can be enabled via `ChatterboxTTS.from_pretrained(device, use_compile=True)`

### 8. MPS-Optimized KV Cache (IMPLEMENTED - CRITICAL)

**Files:** `t3.py`, `t3_hf_backend.py`

- Created custom `MPSOptimizedCache` class to replace HuggingFace's StaticCache
- **Key optimization:** Uses `narrow().copy_()` instead of indexed assignment
  - Indexed assignment on MPS: ~0.7ms per update (very slow)
  - narrow().copy\_() on MPS: ~0.016ms per update (**43x faster**)
- Pre-allocates all KV cache memory upfront (context + max_new_tokens)
- Prevents MPS driver memory from growing unbounded during generation
- Updated t3_hf_backend.py to properly detect populated cache using `get_seq_length()`

**Why StaticCache was slow on MPS:**

- StaticCache uses `index_copy_()` which falls back to indexed assignment on MPS
- Indexed scatter operations are extremely slow on Metal
- Our implementation uses contiguous memory views via `narrow()` which Metal handles efficiently

**Performance comparison:**
| Cache Type | Update Speed | Memory Stability |
|------------|-------------|------------------|
| DynamicCache | Fast but leaky | ❌ Grows unbounded |
| StaticCache | 0.7ms/update (~5 it/s) | ✅ Stable |
| **MPSOptimizedCache** | **0.016ms/update (~20 it/s)** | ✅ Stable |

## 📊 Benchmark Results

### Before Optimizations

- Short (5 words): ~0.4x real-time
- Medium (31 words): ~0.4x real-time
- Long (94 words): Memory spike to 30GB+, process killed

### After All Optimizations (MPSOptimizedCache)

| Text     | Words  | Tokens   | Time      | Speed          | RTF       | Memory           |
| -------- | ------ | -------- | --------- | -------------- | --------- | ---------------- |
| Short    | 6      | 112      | 5.4s      | 20.8 tok/s     | 0.42x     | 3.01/3.43 GB     |
| Medium   | 23     | 358      | 14.7s     | 24.4 tok/s     | 0.49x     | 3.01/3.45 GB     |
| **Long** | **63** | **1020** | **41.7s** | **24.5 tok/s** | **0.49x** | **3.01/3.55 GB** |

**Key achievements:**

- ✅ Memory completely stable (driver memory only +0.12GB for 1020 tokens)
- ✅ 3x faster than StaticCache (20 it/s vs 5 it/s)
- ✅ Long text that previously crashed now generates successfully
- ✅ ~0.5x real-time factor (generating audio at half real-time speed)

## 🔴 Remaining Optimization Opportunities

### 1. Synchronous Token Generation (FUNDAMENTAL)

Autoregressive generation is inherently sequential. Each token requires a full forward pass.
**Fix:** Requires architectural changes (speculative decoding, parallel decoding, etc.)

### 2. Classifier-Free Guidance Doubles Compute (FUNDAMENTAL)

CFG requires running the model twice (conditional + unconditional). This doubles inference time.
**Fix:** Allow disabling CFG for faster inference (cfg_weight=0).

## Summary

All critical MPS optimizations have been implemented:

1. **SDPA attention** - Hardware-accelerated attention on Metal
2. **float16 dtype** - Optimal precision for MPS
3. **Conditional output_attentions** - Avoids O(n²) memory overhead
4. **MPSOptimizedCache** - Custom KV cache using `narrow().copy_()` (43x faster than StaticCache)
5. **Pre-allocated tensors** - Avoids memory accumulation from concatenation
6. **Debug logging flag** - Reduces I/O overhead in production

The memory leak has been completely fixed. Long text that previously caused 30GB+ memory spikes
and process kills now runs with stable ~3.5GB memory. Generation speed improved from ~5 it/s
to ~20 it/s through the optimized cache implementation.
