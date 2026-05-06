# Phase 3 / Option A-prime — Stage 2 (descriptor ABI extension)

**Date:** 2026-05-06
**Status:** CLOSED — outcome **B (next missing op kernel surfaces)**.
**Commits:**
- llama.cpp `webllm-browser-patches`: `<pending>` (descriptor ABI bump 18→19, `(buf_handle, offset)` per tensor) — patch stack +5 → +5 (one of the existing patches amended).
- webllm: `<pending>` (matmul.ts / rms-norm.ts / set-rows.ts dispatchers + golden tests).

## Goal (recap)

Stage 1.5 surfaced a Phase 2 ABI bug: the descriptor's per-tensor "handle"
slot was actually the within-buffer offset. Under Option A-prime with a
real model loaded, ~6 large JSEP buffers each contain 100+ tensors at
distinct offsets — `dataManager.get(handle)` rightly threw
"invalid handle 0" because every non-base tensor's "handle" decoded to
its byte offset.

Stage 2 splits the conflated slot into `(buf_handle, offset)` so
dispatchers can correctly bind any tensor regardless of its position
within its parent JSEP buffer.

## What changed

### llama.cpp (`ggml/src/ggml-jsep/ggml-jsep.cpp`)

- `GGML_JSEP_TENSOR_BLOCK_I32`: 18 → 19.
- `ggml_jsep_write_tensor_block` now emits:
  - `out[0]` = buffer handle (from `tensor->buffer->context->handle`)
  - `out[1]` = within-buffer byte offset (from `jsep_tensor_handle(t)`)
  - `out[2]` = type
  - `out[3..6]` / `out[7..10]` = ne low/high
  - `out[11..14]` / `out[15..18]` = nb low/high
- Top-of-file descriptor layout comment updated to match.

### webllm TS dispatchers

- `src/inference/jsep/ops/matmul.ts`:
  - `TENSOR_BLOCK_I32` → 19.
  - `JsepTensorMeta` replaces `handle` with `bufHandle` + `offset`.
  - `readDescriptor` reads new slot positions.
  - `dispatchMatmul` binds via `{buffer, offset, size: rec.size - offset}`
    using `dataManager.get(meta.bufHandle)`.
- `src/inference/jsep/ops/rms-norm.ts` + `set-rows.ts`: same offset-aware
  binding pattern.
- Golden tests `tests/jsep-matmul-golden.test.ts` +
  `tests/jsep-rms-norm-golden.test.ts`: `makeMeta` / `makeF32Meta` now
  emit `bufHandle = handle, offset = 0` (each test tensor lives at
  offset 0 in its own dataManager allocation).

The synthetic offload probe (`smoke-test/p2-v2-offload-probe.src.ts` +
`webllm_synthetic_offload_probe`) was left untouched — it builds tensors
on host_buft, which Stage 1.5's `supports_buft = jsep_buft only` already
makes inert at the routing layer (no descriptors built for those
tensors). The probe's banked Outcome E framing remains valid.

## Gate

`make wasm-build-jsep && make checkall` → green
(747 pass / 36 skip / 0 fail; fmt + lint + typecheck × 2 clean).

Browser smoke at `?v=A-prime-stage2`:

```
[1/8] Initializing JSEP WASM module...
[2/8] Acquiring WebGPU device...
[3/8] Installing JSEP callbacks (must precede webllm_load_model)...
[4/8] Initializing ggml-webgpu backend...
[5/8] Fetching GGUF from /models/tinyllama-1.1b-chat-q4_0.gguf...
     loaded 637.8 MiB
[6/8] Loading model + creating context...
     model loaded in 263 ms; vocab = 32000
     counters@load = {"alloc":6,"free":0,"write":134,"read":0,
                      "clear":1,"runOp":0,"sync":3}
[7/8] Decoding prompt (6 tokens)...
FAIL — matmul Q4_K kernel: deferred to Task 7
       (browser smoke covers via real weights)
       at buildMatmulShader (matmul.ts:316)
       at dispatchMatmul → jsepRunOp → wasm graph_compute
```

`sched_reserve` continues to pass cleanly (798 nodes / 379 splits /
5.00 ms) and the model now lives entirely on JSEP — `jsep_buf model
buffer size = 455.06 MiB`, KV cache 11 MiB across 22 layers (`dev = JSEP`
on every layer). The 181 MiB `CPU model buffer size` is the
`token_embd.weight` family that ggml's CPU repack preempts (see stderr
`done_getting_tensors: tensor 'token_embd.weight' (q4_K) (and 66 others)
cannot be used with preferred buffer type CPU_REPACK, using CPU
instead`) — host residency for those tensors is unrelated to the
descriptor ABI.

## Outcome B

The descriptor ABI extension worked: `dispatchMatmul` was reached for a
real decode dispatch with a non-zero, valid `bufHandle` (no more
"invalid handle 0" failure). The next missing kernel surfaced exactly as
the TODO Stage 5 op surface table predicted — **Q4_K matmul** — which
the Phase 2 matmul kernel deliberately deferred (`matmul.ts:316`
throws "matmul Q4_K kernel: deferred to Task 7").

TinyLlama-1.1B-Chat is mostly Q4_K-quantized weights (attn_q/k/v/output,
ffn_gate/up/down — 7 weights × 22 layers + token_embd + output =
~155 Q4_K tensors). With Q4_K the dominant matmul dtype, no real decode
can proceed without the kernel.

## Patch budget

- llama.cpp `webllm-browser-patches`: 5 patches (unchanged — Stage 2
  amends the existing Phase 2 skeleton commit `48acb658d` in place
  rather than landing a new patch, since the descriptor layout was
  always intended to be ABI-stable but is still pre-1.0).
- webllm: 4 commits (Stage 0 `b640d17`, Stage 1 `e60a39e`, Stage 1.5
  `ef5ccac`, Stage 2 `<pending>`).

## Stage 3 brief

Add Q4_K matmul kernel to `src/inference/jsep/ops/matmul.ts`. ~150 LOC
following the existing matmul kernel-selector pattern. Q4_K block
geometry per `ggml-common.h`:

- Super-block of 256 elements (8 sub-blocks × 32 elements).
- 144 bytes per super-block:
  - `d`: f16 super-block scale (2 B)
  - `dmin`: f16 super-block min (2 B)
  - `scales[12]`: 6-bit packed sub-block scales + mins (12 B)
  - `qs[128]`: 4-bit quantized values, 2 nibbles per byte (128 B)

Dequant for sub-block s, element i in [0, 32):
```
sc, m = unpack_6bit_scales(scales, s)
x[s*32 + i] = d * sc * q[s*32 + i] - dmin * m
```

Acceptance: spike at `?v=A-prime-stage3` decodes 5 tokens of "The
capital of France is" and prints a continuation. Per-token tok/s within
±20% of Outcome D's 81 tok/s baseline closes Phase 3.
