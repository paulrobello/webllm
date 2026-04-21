# Handoff: WebLLM Real Model Inference

> **Date:** 2026-04-21
> **Status:** Forward pass produces valid non-zero logits; tokenizer encodes/decodes
>            LLaMA SentencePiece correctly. Generated output still degenerates — evidence
>            points to attention not using the KV cache effectively during decode.
> **Plan file:** `/Users/probello/.claude/plans/i-want-to-create-mutable-toucan.md`

---

## Current Smoke Test Output (v=28)

```
encode("hello")                 → [22172]            (LLaMA "▁hello")
encode("Once upon a time")      → [9038, 2501, 263, 931]   (▁Once ▁upon ▁a ▁time)
decode(prompt tokens)           → "Once upon a time"
first generated token           → 2557 ("▁frame"), logit ≈ 7.13
generation throughput           → 41.2 tok/s decode, 128 ms prefill
output                          → "Once upon a timeframebufferedinburghlegensteinberg..."
runtime errors                  → none
visible smoke-test fails        → 0
```

## What's Fixed

### Session 1
1. **All-zero logits** — embedding lookup used `opCpy` from Q4_0 view to F32 view,
   which is not a supported type pair on the WebGPU backend (`ggml-webgpu.cpp:3527-3532`).
   Fix: replace with `ggml_get_rows(tokEmb, tokenIds)` which is supported for Q4_0→F32.
2. **Leaf input data silently discarded** — `ctxCreate` uses `no_alloc=true`, so
   `tensor->data` is null at `tensorSetData` time. Fix: upload leaf inputs
   (`posTensor`, `tokenIdsTensor`) via `backendTensorSet` *after*
   `backendAllocCtxTensors`.

### Session 2
3. **SPM tokenizer dropped whitespace** — `encode("hello")` → `[12199]` ("hello"),
   not `[22172]` ("▁hello"). Two sub-bugs:
   - No LLaMA-style SPM normalization. Fix: prepend `▁` and replace spaces
     with `▁` on encode; strip the leading `▁` and convert back to space on decode.
     Controlled by `TokenizerConfig.addPrefixSpace` (default `true`).
   - `encodeSpm` iterated raw UTF-8 bytes (Latin-1 JS chars), which could never
     match multi-byte vocab entries like `▁`. Fix: iterate Unicode code points
     instead, so `▁` is one symbol.
4. **Decode position off-by-one** in `smoke-test/real-model.html` —
   `pos = tokens.length + step + 1` skipped position `tokens.length`. Fix:
   `pos = tokens.length + step`.
5. **Removed debug `fprintf`** spam from
   `ggml/src/ggml-webgpu/ggml-webgpu.cpp::ggml_backend_webgpu_build_multi`
   (`CONFLICT [...]` and `DISPATCH [...]` prints). Kept the correctness logic.

---

## Remaining Issues

### 1. Decode doesn't appear to use the KV cache effectively (NEW — highest priority)

Symptom: with two very different prompt tokenizations, the first generated token
and the full 40-token continuation are identical (`[2557, 9040, 287, ...]`),
even though prefill logit values differ slightly. The continuation is also a
tight repetition loop (`"...chiadepressionnelittle bitumengelettapersteinberg"`
repeats verbatim).

Evidence the forward pass *is* using the prompt during prefill:
- Different prompts produce different top-1 logit values (`7.10` vs `7.13`).
- `prompt decoded` round-trips correctly.

Evidence the decode path is degenerate:
- Identical multi-token continuations from different prefixes is essentially
  impossible if attention + KV-cache are intact.
- The repetition is byte-for-byte, consistent with attention output being
  (near-)constant regardless of context.

Hypotheses to investigate, in order:
1. **KV cache writes are landing in the wrong slot during decode.**
   Check that the `pastLen * kNb1` offset in `model-inference.ts:231` actually
   refers to the right stride when the prefill wrote a batch of `nTokens` and the
   decode step writes `nTokens=1`. In particular, verify `kNb1` == stride between
   positions in the `[headDim, maxCtx, nKvHeads]` layout — if the K buffer is
   laid out with max-context as a different dimension, this offset is wrong.
2. **The `opCpy(cont(kRopeP), kWriteView)` writes are being silently skipped.**
   The types are F32→F32 so `supports_op` passes, but the *dst view* may have
   non-trivial misalignment or shape that the WebGPU cpy shader doesn't support.
   Add a debug readback of `kv.k[layer=0]` after a prefill forward and confirm
   the first `nTokens` slots actually contain the prompt's K values.
3. **`pastLen` isn't propagating into `opDiagMaskInf`.** The custom shader uses
   `params.n_past` (session-1 patch). Verify for pastLen > 0 that the mask uses
   `j > i + n_past` correctly — a miswired value could mask out all prior tokens
   and give the observed attention-ignores-history behavior.
4. **Temp-buffer copy race on decode.** The buffer-conflict resolution in
   `build_multi` copies source to a temp buffer before each dispatch. For some
   dispatch shapes it's possible the source data isn't yet visible on the GPU at
   copy time. Test by forcing all shaders to `read_write` bindings under
   `__EMSCRIPTEN__` (advisor option C from session 1) and re-running.

Fastest way to triage: add a debug readback of attention output after layer 0 for
two different prompts and confirm they diverge. If they don't, attention is the
bug. If they do, the bug is later (merge, FFN, or KV propagation between layers).

### 2. SPM byte-fallback is wrong for LLaMA

In `tokenizer.ts::encodeSpm` the byte fallback emits `0x0100 + byte_value`, but
LLaMA stores byte tokens by text (`<0xHH>`) at low, non-contiguous IDs. The
preprocessing added in session 2 means whitespace no longer hits the fallback,
so normal text works. But any input character not resolvable through the vocab
still drops silently. Fix by looking up `<0xHH>` byte-token text instead of a
fixed offset, and add a regression test for a Unicode input like `"café"`.

### 3. Build still `-O1 -sASSERTIONS=2`

Keep while debugging the decode issue. Switch to `-O3` once inference is clean.

### 4. Browser buffer-conflict temp-buffer: latent 3+ binding edge case

Nested loop in `ggml_backend_webgpu_build_multi` can miss a conflict between
bindings b and c if entry a was already rewritten to a temp buffer. No llama op
has 3+ tensor bindings, so it doesn't bite today. If we add flash-attn or
mul_mat_id later, revisit.

---

## Files Modified

### In this repo (`/Users/probello/Repos/webllm/`)

| File | Change |
|------|--------|
| `src/inference/model-inference.ts` | Embedding uses `opGetRows`; leaf input data uploaded via `backendTensorSet` after backend alloc |
| `src/inference/ggml-wasm.ts` | `opGetRows(a, b)` binding; pre-existing `opDiagMaskInf` + heap-based async readback |
| `src/inference/tokenizer.ts` | SPM ▁ normalization (encode + decode), code-point iteration, `addPrefixSpace` flag |
| `src/wasm/webgpu-bridge.cpp` | `op_get_rows` wrapper |
| `src/wasm/CMakeLists.txt` | `_op_get_rows` exported |
| `src/models/model-loader.ts` | `embeddingHeadLength` = `embeddingLength / headCount` default; `headCountKv` defaults to `headCount` |
| `smoke-test/real-model.html` | TinyLlama end-to-end test page; decode position off-by-one fixed |
| `tests/tokenizer.test.ts` | SPM tests use `addPrefixSpace: false`; two new ▁-normalization tests |

### In llama.cpp dependency (`/Users/probello/Repos/llama.cpp/`, uncommitted patches)

| File | Change |
|------|--------|
| `ggml/src/ggml.c` | Iterative `ggml_visit_parents_graph` (malloc stack, MAX=4096) for deep transformer graphs |
| `ggml/src/ggml-webgpu/ggml-webgpu.cpp` | `GGML_OP_DIAG_MASK_INF` WGSL + `supports_op`; `batch_compute_passes = false`; per-dispatch compute passes in `build_multi` else branch; non-aborting error handler under `__EMSCRIPTEN__`; buffer-conflict temp-buffer copies (debug `fprintf`s removed) |

---

## Environment

```bash
source ~/emsdk/emsdk_env.sh
make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.js src/wasm/build/webllm-wasm.wasm smoke-test/
cd smoke-test && python3 -m http.server 8031
# Navigate: http://localhost:8031/real-model.html?v=<N>
# Reuse tab 2B4CAD087712E09145E6699CB897AFF3 on agentchrome port 60925
```

---

## Next Session Starting Point

Investigate the decode-time attention / KV-cache bug (item 1 above). The cheapest
diagnostic is a two-prompt differential: call forward twice with different
prompts, read back `attn_out` at layer 0, and confirm they diverge. If they
don't, narrow into attention (K/V write offset, `opDiagMaskInf` params, or
temp-buffer race). If they do, the bug lives later in the pipeline.
