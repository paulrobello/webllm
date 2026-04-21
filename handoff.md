# Handoff: WebLLM Real Model Inference

> **Date:** 2026-04-21
> **Status:** Inference pipeline produces non-zero logits with working KV cache and
>            attention that depends on history, but predictions are systematically
>            weaker than the HuggingFace reference. There is still at least one
>            subtle bug somewhere in the attention or residual-stream scaling
>            path; the most likely bug site is outlined below.
> **Plan file:** `/Users/probello/.claude/plans/i-want-to-create-mutable-toucan.md`

---

## The residual bug (LAYER-BY-LAYER EVIDENCE)

For token `"▁hello"` (id=22172) at position 0, dim 624 is an outlier
dimension that develops rapidly at layer 2 in the reference model.
Side-by-side comparison of per-layer hidden-state magnitude:

| Layer | HF rms | Mine rms | HF max&#124;v&#124; (at dim) | Mine max&#124;v&#124; (at dim) |
|-------|--------|----------|--------------------------|-------------------------------|
| emb   | 0.015  | 0.015    | 0.05 (331)               | 0.05 (matches)                |
| 0     | 0.024  | 0.021    | 0.40 (1454)              | 0.11 (2014)                   |
| 1     | 0.029  | 0.026    | 0.23 (1447)              | 0.11 (2014)                   |
| **2** | **4.37** | **0.039** | **-157.74 (624)**     | **-0.75 (624)**               |
| 3     | 4.37   | 0.053    | -157.79 (624)            | -0.68 (624)                   |
| ...   | ...    | ...      | ...                      | ...                           |
| 20    | 4.19   | 0.53     | 151 (624-family)         | 2.5 (411)                     |
| 21    | 1.74   | 0.63     | 21 (various)             | 5.6 (1308)                    |

**The outlier DIMENSION is right (624).** **The SIGN is right (negative).**
**The MAGNITUDE is ~200× too small.** This ratio holds through every
subsequent layer — my pipeline reproduces the right qualitative behavior
but with dramatically less amplification.

Single-token probe against HF (id=22172 "▁hello"):
- HF top-5: ▁to(9.93), .(9.65), ,(9.60), :(9.44), !(9.12)
- Mine top-5: гля(9.74), indows(9.68), ▁rör(8.87), urale(8.75), alone(8.69)

Logit magnitudes are in the same ballpark (~9-10), but the
*directions* are completely different. Without the outliers at dim 624
and neighbors, the final hidden state projects onto different logit
directions, giving different (wrong) top-k tokens.

That's a concrete, reproducible smoking gun for a specific numerical
issue somewhere in layer 2's attention+FFN pathway (or shared across
all layers but only visible once outliers develop). Next session's
debugging should start by isolating whether layer 2's attention output
or FFN output is the undersized component — instrument each separately
via a per-layer debug tensor readback.

### Other evidence

| Probe                              | HuggingFace FP32 | Our pipeline |
|-----------------------------------|------------------|--------------|
| `"The capital of France is"` no BOS | top-1 **Paris** (12.50) | top-1 **▁a** (9.18); Paris not in top-5 |
| `"The capital of France is"` + BOS  | top-1 **Paris** (13.39) | top-1 **tes** (7.52); predictions radically wrong |
| Embedding row rms for BOS           | (not checked)    | 0.0022 (matches manual Q4_0 dequant of GGUF bytes) |
| K cache after prefill of "hello"    | —                | non-zero, reasonable magnitude (rms ~0.35) |
| K cache after prefill of BOS alone  | —                | 17× smaller (rms ~0.022) |
| KV history differential             | —                | HISTORY MATTERS — attention does use the cache |
| Norm weights                       | —                | GPU bytes match raw GGUF F32 bytes exactly |
| Tokenization                       | —                | Matches HF IDs exactly |
| Mask shader                        | —                | Fixed to use `row_in_head = (idx / ne0) % ne1` |

My final logits are systematically ~3 smaller than HF's (top logits ~9 vs ~12).
The delta between HF's top-1 (Paris, 12.50) and HF's #3 (▁a, 12.09) is only
0.4 — small enough that a ~0.5 logit error can flip the ranking. So the
remaining bug likely produces a small but *systematic* error that accumulates
through the 22 layers and scrambles the top-k ranking at the output head.

### Most plausible remaining culprits

1. **GQA broadcast via ggml_mul_mat.** TinyLlama has 32 query heads and 4 KV
   heads (ratio 8). The webgpu `ggml_webgpu_mul_mat` passes
   `(src1->ne[2] / src0->ne[2])` as a broadcast factor and relies on the
   shader to index the correct kv head per query head. If that shader path is
   buggy in the browser-Emscripten build (different from native Dawn), GQA
   attention produces subtly wrong outputs. Easiest test: run with a
   non-GQA model (nHeads == nKvHeads) and see if output quality improves.

2. **Non-contiguous V-view mul_mat.** `fullV` is a strided view of `kv.v`
   with `nb[0]=4`, `nb[1]=maxCtx*4` — the memory layout is **sparse** along
   the M dimension (headDim). If the webgpu mul_mat shader assumes the
   M-dimension stride equals K*type_size (contiguous), it reads wrong bytes
   for every position after the first headDim entry. llama.cpp runs the
   same view successfully on native Dawn; the bug may be
   Emscripten/browser-specific.

3. **RoPE dimension count mismatch.** GGUF has `llama.rope.dimension_count`
   which the loader doesn't read (uses headDim as default). For TinyLlama
   they happen to match (both 64), so this isn't the current bug, but worth
   flagging if another model behaves worse.

### What is definitively NOT the bug

- Tokenizer (SPM + ▁ prefix + byte fallback — all tested against HF).
- Embedding dequant (manual Python Q4_0 dequant matches our GPU readback
  exactly for every probed token).
- RMSNorm epsilon (1e-5 matches GGUF).
- RMSNorm gain application (verified present at all three sites).
- Norm weight values (match GGUF bytes exactly).
- KV cache population (probed, works).
- Attention's causal mask (custom shader fixed for multi-head and then
  replaced entirely by `ggml_soft_max_ext`; both gave same behavior).
- Forward-expand ordering of KV writes vs attention reads (verified
  via history-differential).

---

## Debug Tools (kept, documented, for the next run)

`src/inference/model-inference.ts` exposes four debug methods. From the
browser console when the smoke test is loaded (`window.inference` and
`window.tokenizer` are stashed):

```js
// Raw dequantized embedding row
await window.inference.debugReadEmbeddingRow(1);  // BOS

// K / V cache slice (probe prefill first)
window.inference.resetKVCache();
await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
await window.inference.debugReadKCache(/*layer*/ 0, /*nBytes*/ 64*4, /*offset*/ 0);
await window.inference.debugReadVCache(0, 64*4, 0);

// F32 norm weight first N floats
await window.inference.debugReadNormWeight("attn0", 8);  // "attn0" | "ffn0" | "output"

// Final hidden state after layers [0..L] for a single token (full transformer
// rebuild through layer L; no KV sharing with the main forward pass)
await window.inference.debugLayerOutput(/*tokenId*/ 1, /*layerIdx*/ 0);
```

`smoke-test/real-model.html` runs these automatically and prints the
results, plus a `[7a] KV history differential` that compares decode
logits for the same probe token after two different prefixes (the key
signal for "attention ignoring history"). Keep them — they caught four
of the bugs fixed so far and will catch the next one too.

---

## What Was Fixed (cumulative)

1. Embedding lookup used `opCpy` Q4_0→F32 (unsupported; replaced with
   `ggml_get_rows`).
2. Leaf input data (`posTensor`, `tokenIdsTensor`, mask) must be written
   with `backendTensorSet` *after* `backendAllocCtxTensors`.
3. SPM tokenizer: ▁ normalization (encode + decode), code-point
   iteration, byte-fallback via `<0xHH>` text.
4. KV writes were orphaned by `graph_build_forward_expand` (unreachable
   from logits) — now explicitly expanded per layer.
5. KV writes ordered BEFORE attention reads in the graph node list
   (same-graph create-up-front + per-layer expansion pattern).
6. RMSNorm gamma was never multiplied in — now applied at all three
   norm sites (attn, ffn, final).
7. Custom `GGML_OP_DIAG_MASK_INF` shader broken past head 0; later
   replaced entirely by `ggml_soft_max_ext` with an explicit causal
   mask tensor.
8. WASM build -O1 → -O3 (3.4MB → 1.77MB).
9. Sampling wired in via the existing `Sampler` class (temp / top-k /
   top-p / repetition penalty).
10. `ggml_soft_max_ext` + `op_get_rows` WASM bindings added.

---

## Files Modified (this session + last session)

### This repo (`/Users/probello/Repos/webllm/`)

| File | Change |
|------|--------|
| `src/inference/model-inference.ts` | All the structural fixes above; `maskTensor` creation + upload; graph created up-front; per-layer `build_forward_expand` for KV writes; `debugReadKCache` / `debugReadVCache` / `debugReadEmbeddingRow` / `debugReadNormWeight` / `debugLayerOutput` |
| `src/inference/ggml-wasm.ts` | `opGetRows(a, b)` + `opSoftMaxExt(x, mask, scale, maxBias)` bindings |
| `src/inference/tokenizer.ts` | SPM ▁ normalization + byte fallback via `<0xHH>` lookup |
| `src/wasm/webgpu-bridge.cpp` | `op_get_rows` + `op_soft_max_ext` wrappers |
| `src/wasm/CMakeLists.txt` | Exports; `-O3 -sASSERTIONS=0` |
| `smoke-test/real-model.html` | Diagnostic pipeline (norm probe, KV write probe, KV history differential, per-prompt top-k dumps, sampling, TinyLlama Zephyr chat template variants, single-BOS sanity). `window.{inference, tokenizer, parsedModel}` exposed for console probing. |
| `tests/tokenizer.test.ts` | SPM ▁ tests + byte-fallback test (café) |

### llama.cpp (`/Users/probello/Repos/llama.cpp/`, uncommitted)

| File | Change |
|------|--------|
| `ggml/src/ggml.c` | Iterative `ggml_visit_parents_graph` for deep transformer graphs |
| `ggml/src/ggml-webgpu/ggml-webgpu.cpp` | Custom `GGML_OP_DIAG_MASK_INF` shader (fixed row-in-head); `batch_compute_passes = false`; per-dispatch compute passes; non-aborting error handler under `__EMSCRIPTEN__`; buffer-conflict temp-buffer copies |

---

## Environment

```bash
source ~/emsdk/emsdk_env.sh
make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
cd smoke-test && python3 -m http.server 8031
# Cache-bust: ?v=$(date +%s). Reuse tab 2B4CAD087712E09145E6699CB897AFF3 on port 60925.
```

Ground-truth reference available via `llama-cli` (brew-installed) and HF
transformers:

```bash
# Chat mode (uses chat template internally)
llama-cli -m smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf

# Raw token-level reference
uv run --no-project --with transformers --with torch --with sentencepiece \
  python3 -c '<HF script — see handoff for template>'
```

---

## Next Session Starting Point

1. Run the diagnostic differential — `debugLayerOutput(450, L)` for
   L=0..22, compare its rms / values to HF's per-layer hidden states.
   The layer where they diverge is where the bug lives.
2. If layer 0 matches HF: the bug is in the multi-layer residual path
   (KV cache dependency ordering at later layers? CPU-like build order?).
3. If layer 0 already differs: bug is in a single layer. Most likely
   candidate is the GQA `mul_mat(fullK, qp)` or the non-contiguous V-view
   `mul_mat(fullV, attnW)` on the browser WebGPU backend. Isolate by
   running a non-GQA model (e.g. a Phi-2 or a llama-2-7b-base if vram
   allows).
4. Once it works, strip diagnostic scaffolding from `real-model.html`
   (keep the debug methods in `model-inference.ts` — they pay their rent).
