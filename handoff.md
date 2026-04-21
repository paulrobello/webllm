# Handoff: WebLLM Real Model Inference

> **Date:** 2026-04-21
> **Status:** ✅ **END-TO-END WORKING.** TinyLlama 1.1B Q4_0 produces coherent,
>            factually correct output in the browser.
>            `"The capital of France is"` → top-1 **"▁Paris"** (14.36). Full
>            greedy continuation: *"Paris.2. The capital of Spain is Madrid.3.
>            The capital of Italy is Rome.4. The capital of..."*
>            `"The quick brown"` → top-1 **"▁fo"** (18.22), "fox" at #2.
>            Current measured perf is ~58–59 tok/s decode, ~125 ms prefill.
> **Plan file:** `/Users/probello/.claude/plans/i-want-to-create-mutable-toucan.md`

---

## The final bug: V cache permute

After every other fix was in place, the remaining bug was a single wrong
permutation argument in the V cache write path.

### What was wrong

```ts
// v3 shape is [headDim=64, nKvHeads=4, nTokens]
const v3P = wasm.opPermute(v3, 2, 0, 1, 3);
```

In ggml, `ggml_permute(src, axis0, axis1, axis2, axis3)` means: "source
dim `i` becomes destination axis `axis_i`." So passing `(2, 0, 1, 3)`
with a source shape of `[headDim, nKvHeads, nTokens]` produces
destination shape `[nKvHeads, nTokens, headDim]` (= `[4, 1, 64]` for a
single-token prefill).

But the V cache is laid out as `[nTokens, headDim, nKvHeads]` — the
`vWriteView` expected `[1, 64, 4]`. The element counts matched, so the
subsequent `ggml_cpy` silently wrote to the wrong cache positions
(element-wise interpretation with mismatched logical shapes). Every
attention step downstream was reading V values from shuffled positions.

### The fix

```ts
const v3P = wasm.opPermute(v3, 1, 2, 0, 3);
```

`(1, 2, 0, 3)` maps source `(headDim=0, nKvHeads=1, nTokens=2)` → dest
`(nTokens=0, headDim=1, nKvHeads=2)`, which matches the V cache layout.

### Why this escaped all prior probes

Four independent probes incorrectly suggested the pipeline was
structurally fine:

- **`debugReadKCache` looked normal** — K uses a different, correct
  permute `(0, 2, 1, 3)`.
- **`debugReadVCache` read one entry** — it hit position 0, which
  coincidentally contained real data from the misplaced V write, so it
  looked populated.
- **KV history differential reported HISTORY MATTERS** — attention
  *was* reading different values depending on history; the values were
  just scrambled rather than right.
- **Single-layer RMS matched HF at layer 0** — the scrambling distorted
  directions but preserved overall magnitude well enough that bulk
  statistics looked OK.

The actual bug was only detectable by comparing per-dim hidden state
values against HF, which is what finally pinned it down.

---

## Evidence of the fix

| Prompt                              | HF FP32 top-1         | Our pipeline top-1     |
|-------------------------------------|-----------------------|------------------------|
| `"The capital of France is"` (+BOS) | **Paris** (13.39)     | **▁Paris** (14.36)     |
| `"The quick brown"` (+BOS)          | fox / similar          | **▁fo** (18.22), fox #2 |
| `"The capital of France is"` no BOS | Paris (12.50)         | Paris / matching top-5 |

End-to-end greedy output for `"The capital of France is"`:
```
Paris.2. The capital of Spain is Madrid.3. The capital of Italy is Rome.
4. The capital of [...]
```

Coherent, factually correct, grammatically sound. Current measured perf is
~58–59 tok/s decode, ~125 ms prefill on TinyLlama 1.1B Q4_0 through the
Emscripten WebGPU backend.

---

## 2026-04-21 decode profiling result

A follow-up profiling pass instrumented `ModelInference.forward()` phase by
phase and added `bun run eval/perf.ts --profile` to scrape per-step decode
traces from the smoke-test page.

Result for single-token decode steps:

- total: **16.75 ms**
- `ctxCreate`: **0.00 ms** (0.0%)
- `buildGraph`: **0.21 ms** (1.3%)
- `backendAlloc`: **0.05 ms** (0.3%)
- `uploadLeaves`: **0.02 ms** (0.1%)
- `graphCompute`: **6.92 ms** (41.3%)
- `downloadLogits`: **9.53 ms** (56.9%)
- `teardown`: **0.02 ms** (0.1%)

Conclusion: the planned "build decode graph in C" rewrite is **not worth
pursuing right now**. The targeted non-GPU overhead (`ctxCreate` +
`buildGraph` + `backendAlloc` + `teardown`) is only **1.7%** of the decode
step, far below the 30% gate from the plan. The dominant costs are GPU
compute and logits readback.

Phase A was therefore **skipped**. If decode perf work resumes, the highest
leverage directions are reducing readback cost or reducing GPU work, not
moving graph construction from TS into C.

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
8. **V cache permute used wrong ggml_permute arguments** — the silent
   shape mismatch in the subsequent cpy scrambled V values in the cache.
   Fixed `(2, 0, 1, 3)` → `(1, 2, 0, 3)`.
9. WASM build -O1 → -O3 (3.4MB → 1.77MB).
10. Sampling wired in via the existing `Sampler` class (temp / top-k /
    top-p / repetition penalty).
11. `ggml_soft_max_ext` + `op_get_rows` WASM bindings added.

---

## Debug Tools (kept, documented)

`src/inference/model-inference.ts` has instrumented debug helpers. From
the browser console on `smoke-test/real-model.html`:

```js
// Raw dequantized embedding row
await window.inference.debugReadEmbeddingRow(1);  // BOS

// K / V cache slices (probe prefill first)
window.inference.resetKVCache();
await window.inference.forward(new Int32Array([22172]), new Int32Array([0]));
await window.inference.debugReadKCache(0, 64*4, 0);
await window.inference.debugReadVCache(0, 64*4, 0);

// F32 norm weight first N floats
await window.inference.debugReadNormWeight("attn0", 8);  // "attn0" | "ffn0" | "output"

// Full transformer through layer L, with optional checkpoint
// inside that layer. This was the tool that found the V-permute bug:
//   const h = await inference.debugLayerOutput(22172, 0, "attn_v");
//   // Compare to HF's l0 V projection
await window.inference.debugLayerOutput(
  22172,  // tokenId
  0,      // layerIdx
  "layer_output"
    // or: "pre_attn" | "attn_normed" | "attn_q" | "attn_k" | "attn_v"
    //     | "attn_out" | "post_attn"
    //     | "ffn_normed" | "ffn_gate" | "ffn_up" | "ffn_hidden" | "ffn_out"
);
```

`smoke-test/real-model.html` runs a chain of diagnostics automatically
and stashes `window.{inference, tokenizer, parsedModel}` for console use.
Keep them around — every future numerical-correctness bug will need
something like this.

---

## Files Modified

### This repo (`/Users/probello/Repos/webllm/`)

| File | Change |
|------|--------|
| `src/inference/model-inference.ts` | All the fixes listed above + the full debug helpers |
| `src/inference/ggml-wasm.ts` | `opGetRows(a, b)` + `opSoftMaxExt(x, mask, scale, maxBias)` bindings |
| `src/inference/tokenizer.ts` | SPM ▁ normalization + byte fallback via `<0xHH>` lookup |
| `src/wasm/webgpu-bridge.cpp` | `op_get_rows` + `op_soft_max_ext` wrappers |
| `src/wasm/CMakeLists.txt` | Exports; `-O3 -sASSERTIONS=0` |
| `smoke-test/real-model.html` | Sampling, TinyLlama chat template (optional), diagnostics, window exposure |
| `tests/tokenizer.test.ts` | SPM ▁ tests + byte-fallback test |

### llama.cpp (`/Users/probello/Repos/llama.cpp/`, uncommitted)

| File | Change |
|------|--------|
| `ggml/src/ggml.c` | Iterative `ggml_visit_parents_graph` for deep transformer graphs |
| `ggml/src/ggml-webgpu/ggml-webgpu.cpp` | `batch_compute_passes = false`; per-dispatch compute passes; non-aborting error handler under `__EMSCRIPTEN__`; buffer-conflict temp-buffer copies |

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

---

## Next Session (if needed)

The pipeline is working. Highest-value follow-ups now are:

1. Investigate whether decode can avoid or shrink full-vocab logits readback;
   profiling says this is the biggest per-step cost.
2. Look for GPU-side reductions in decode compute (`graphCompute`) before any
   TS→C graph-build rewrite; the profiling gate ruled that rewrite out for now.
3. Strip inline diagnostic scaffolding from `real-model.html` for a clean
   chat-style demo. Keep the debug methods in `model-inference.ts`.
4. Wire up the existing `Generator` + `InferenceSession` classes for a
   proper streaming JS API.
5. The latent 3+ binding buffer-conflict edge case in
   `ggml_backend_webgpu_build_multi` remains untested — no llama op hits
   it today.
6. Test on a larger model (Phi-2, Llama-2-7B) now that the small model
   works.
