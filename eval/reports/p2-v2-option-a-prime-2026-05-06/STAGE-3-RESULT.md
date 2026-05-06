# Phase 3 / Option A-prime — Stage 3 (Q4_K matmul kernel)

**Date:** 2026-05-06
**Status:** PARTIALLY CLOSED — Q4_K kernel verified correct in
isolation (delta 4.5e-6 vs CPU reference); full-decode reaches the LM
head but produces exactly-zero logits → **Outcome C** per the original
Stage 3 outcome table. Root cause not in the new kernel; queued as
**Stage 3.5** for next session.

**Commits:**
- llama.cpp `webllm-browser-patches`: `53c66649f` (unchanged — Stage 3 is
  zero-patch on the C++ side; supports_op already advertised Q4_K in the
  Phase 2 / Stage 2 baseline).
- webllm: `<pending>` (Q4_K WGSL kernel, golden test, jsepRead/Write/Clear
  flush ordering fixes, spike harness Q4_K self-test + logit-stats probe,
  index-jsep probe re-exports).

## What Stage 3 set out to ship

Replace the `case GGML_TYPE_Q4_K: throw new Error("...deferred to Task
7...")` at `src/inference/jsep/ops/matmul.ts:316` with a real WGSL
kernel mirroring the Q4_0 case structurally — `load_q4_K(m, k, batch)`
helper plus 6-bit scale/min unpack per
`ggml-quants.c::get_scale_min_k4`. After the kernel, the spike either
completes greedy "Paris" decode (Outcome A) or surfaces the next
missing op kernel (Outcome B).

## What landed

### Q4_K WGSL kernel — `src/inference/jsep/ops/matmul.ts`

~110 LOC added under the `case GGML_TYPE_Q4_K:` branch. Mirrors
`ggml-common.h::block_q4_K` / `ggml-quants.c::dequantize_row_q4_K`:

- **Block geometry:** super-block of 256 elements, 144 bytes:
  `d` (f16, 2B) + `dmin` (f16, 2B) + `scales[12]` (6-bit packed) +
  `qs[128]` (4-bit nibbles, 2/byte).
- **`q4k_byte_at(byte_off)`** — extracts a single byte from the
  `array<u32>` storage binding via `(src0[off/4] >> ((off & 3) * 8)) &
  0xff`, matching the Q4_0 kernel's byte-extraction pattern.
- **`q4k_unpack_scale_min(scales_byte_base, is)`** — direct port of
  `get_scale_min_k4`. Returns `vec2<u32>(sc, m)` for sub-block index
  `is` in `[0, 8)`. Handles the asymmetric `is < 4` / `is >= 4` packing
  (the latter splices the high 2 bits from `scales[is-4]` /
  `scales[is]` with the low 4 bits from `scales[is+4]`).
- **`load_q4_K(m, k, batch)`** — single-element dequant. `pair = k/64`
  selects the 64-element group sharing 32 qs-bytes; `within_pair = k %
  64` selects whether to use the low or high nibble of
  `qs[pair*32 + (within_pair % 32)]`; `is = pair*2 + (within_pair >=
  32)` selects the 6-bit scale/min pair. Result: `d * sc * nibble -
  dmin * m_min`.

The kernel inlines these helpers into the same per-thread accumulator
loop the Q4_0 path uses — no new pipeline-cache entry topology, no new
bind-group layout, no descriptor-ABI changes.

### Golden test — `tests/jsep-matmul-golden.test.ts`

`packQ4_K(d, dmin, sc[8], m[8], nibbles[256])` reference packer +
"Q4_K × F32 → F32 — 1×256 × 1×256 → 1×1 (single super-block)" test.
Inverse-engineers `get_scale_min_k4`'s bit layout to emit a valid 144-
byte block from explicit (sc, m) arrays. Skipped on Bun (no WebGPU);
exercised by the browser-side Q4K self-test (see below).

### `jsepRead` / `jsepWrite` / `jsepClear` flush ordering — `src/inference/jsep/index.ts`

All three host-roundtrip JSEP callbacks now call
`encoderBatcher.flush()` before issuing their `device.queue.*`
operation. Rationale (from the diff comment): WebGPU queue operations
are FIFO and `device.queue.writeBuffer` / `mapAsync` is enqueued
immediately, so without the flush a write or read can slip ahead of
compute dispatches still pending in the batcher's open command
encoder. This is the correct protocol invariant regardless of whether
it fixes Outcome C — discovered while bisecting the all-zero logits.

**Empirically: the flushes do NOT fix Outcome C** (verified by
re-running the spike with all three flushes installed and per-dispatch
flush layered on top — logits still all-zero). They are kept anyway
because they remove a latent race that would manifest as soon as the
batcher's `maxDispatch=16` threshold is crossed between dependent
read/write/dispatch sequences.

### Spike harness probe — `smoke-test/p2-v2-spike.src.ts`

Three additions, all gated behind the existing logging path:

1. **`runQ4KSelfTest(runtime)`** runs immediately after
   `installJsepCallbacks` and before `webllm_load_model`. Hand-builds a
   single-super-block Q4_K weight (8 distinct sub-block scale/min
   pairs, varying nibble pattern), a known F32 activation row,
   dispatches `dispatchMatmul` directly through the runtime, reads
   back the dst f32 via `device.queue.copyBufferToBuffer +
   mapAsync`, compares against a CPU reference matmul over the
   reference dequant.
2. **`logitStats[0]`** captures the first decode step's logits and
   reports `hasNaN`, `hasInf`, `finiteCount`, `minFinite`,
   `maxFinite`, and the first 8 values. Disambiguates NaN-poisoning
   from zero-collapse from flat-but-noisy logits.
3. **Counter snapshot at end** unchanged from prior stages.

### Probe-only re-exports — `src/index-jsep.ts`

`dispatchMatmul`, `GGML_OP_MUL_MAT`, `GGML_TYPE_F32`,
`GGML_TYPE_Q4_K`, and `JsepOpDescriptor` re-exported with `JSEP_`
prefix from the JSEP bundle so the spike harness can drive the
matmul dispatcher with hand-crafted descriptors. Not part of the
public API.

## Gate

`make wasm-build-jsep && make checkall` → green
(747 pass / 36 skip / 0 fail; fmt + lint + typecheck × 2 clean).

Browser smoke at `?v=A-prime-stage3-final`:

```
[1/8] Initializing JSEP WASM module...
[2/8] Acquiring WebGPU device...
[3/8] Installing JSEP callbacks (must precede webllm_load_model)...
Q4K_SELFTEST = {
    "status": 0,
    "got": -28.562091827392578,
    "reference": -28.562087358678582,
    "delta": 4.5e-6,
    "dequantFirst4": [-0.030, 0.220, 0.470, 0.720],
    "dequantLast4":  [32.08, 34.78, 37.48, 40.18]
}
[4/8] Initializing ggml-webgpu backend...
[5/8] Fetching GGUF from /models/tinyllama-1.1b-chat-q4_0.gguf...
     loaded 637.8 MiB
[6/8] Loading model + creating context...
     model loaded in 274 ms; vocab = 32000
     counters@load = {"alloc":6,"free":0,"write":134,"read":0,
                      "clear":1,"runOp":0,"sync":3}
[7/8] Decoding prompt (6 tokens)...
     prefill 101 ms
[8/8] Greedy decoding 5 tokens...
LOGIT_STATS_STEP0 = {
    "step":0, "first8":[0,0,0,0,0,0,0,0],
    "topVal":0, "topId":0,
    "hasNaN":false, "hasInf":false,
    "finiteCount":32000, "minFinite":0, "maxFinite":0
}
GENERATED_TOKENS = [0,0,0,0,0]
GENERATED_TEXT = ""
PER_TOKEN_MS = 23.72
COUNTER_DELTAS = {"alloc":0,"free":0,"write":1206,"read":1266,
                  "clear":0,"runOp":1602,"sync":3671}
CROSSINGS_PER_TOKEN = 1549.0
TOTAL_DECODE_MS = 119
TOTAL_PREFILL_MS = 101
MODEL_LOAD_MS = 274
DONE
```

## What the data says

**Q4_K kernel: correct.** The self-test path produces a per-element
delta of 4.5e-6 vs the CPU reference dequant — well within f32 round-
off. The kernel handles the 6-bit scale/min unpack, the
super-block-shared `d`/`dmin`, the pair-of-32-elem nibble layout, and
the byte-extraction-from-`array<u32>` correctly for at least the
self-test geometry (M=1, K=256, N=1, batch=1).

**Q4_K kernel runs in the real model:** A diagnostic build that
counted dispatches per cacheKey reported
`{"mat-q4_k-f32-f32-2":805, "mat-f16-f32-f32-3":264}` over 5 decode
tokens + 1 prefill. The 805 Q4_K dispatches divided across ~6 forward
passes give ~134 per pass — consistent with TinyLlama's 22 layers ×
~6 q4_K matmuls per layer (attn_q/k/v/output + ffn_gate/up/down). The
264 f16 matmuls are the attention QK^T paths.

**Logits: exactly zero.** All 32000 logits in step 0 are exactly
0.0 — no NaN, no Inf, all finite, min and max both zero. The argmax
collapses to token 0 (`<unk>` in TinyLlama's vocab) deterministically
across all 5 decode steps.

**Implication.** The all-zero collapse is upstream of the Q4_K kernel
itself. Possible loci:

1. **CPY ordering between splits.** The graph has 798 nodes split into
   379 splits between JSEP and CPU (per `sched_reserve` log). Each
   inter-backend transition uses `set_tensor` / `get_tensor` host-
   roundtrip via `jsepRead` / `jsepWrite` (JSEP backend's
   `cpy_tensor` is NULL). Per-token counters: 241 writes / 211 reads /
   612 syncs. Stage 3 added flushes to all three host-roundtrip
   callbacks but the symptom did not move. Either there's a deeper
   ordering bug, or this isn't the root cause.
2. **RMS_NORM kernel bug.** The Phase 2 RMS_NORM kernel was unit-
   tested via the synthetic harness only; it has never been validated
   against a real-model forward pass. If it produces zero output for
   non-zero input (e.g., shape uniform packing bug, misaligned
   binding), the residual stream collapses to zero on the first
   layer's first norm and stays there.
3. **Token-embedding lookup (GET_ROWS, on CPU).** Runs on CPU because
   GET_ROWS is not in `supports_op`. token_embd lives on CPU per the
   load-time stderr (`tensor 'token_embd.weight' (q4_K) (and 66
   others) cannot be used with preferred buffer type CPU_REPACK,
   using CPU instead`). If the embedding lookup output is silently
   zero (e.g., wrong indexing into a misformatted token_embd buffer),
   everything downstream is fed zero → output is zero.
4. **MUL kernel running on CPU.** RMS_NORM emits a normalized tensor;
   the per-channel weight multiply is a separate `GGML_OP_MUL` node
   that runs on CPU (not in `supports_op`). If MUL silently outputs
   zero (CPU backend issue with this build configuration), the post-
   norm tensor is zero before any matmul ever sees it.

## Patch budget

- llama.cpp `webllm-browser-patches`: **6 patches** (unchanged from
  Stage 2 — `48acb658d`, `7919d1839`, `49413d8e9`, `d8b80dee2`,
  `d0075e9a6`, `53c66649f`). Stage 3 is zero-patch on the C++ side.
- webllm: **5 commits** (Stage 0 `b640d17`, Stage 1 `e60a39e`, Stage
  1.5 `ef5ccac`, Stage 2 `9406496`, Stage 3 `<pending>`).

## Stage 3.5 — root-cause the all-zero logits collapse

Pickup work for the next session. Order matters: localize the fault
before adding more kernels.

1. **Cheapest first — RMS_NORM self-test.** Adapt the spike's
   `runQ4KSelfTest` pattern to drive `dispatchRmsNorm` directly: build
   a known F32 input row, dispatch, read back, compare against a JS
   reference that mirrors `ggml_rms_norm_impl`. If the kernel
   produces zero / NaN / wrong-magnitude output, the bug is here and
   the Q4_K kernel had nothing to do with the symptom.
2. **First-model-matmul output capture.** Add a one-shot probe in
   `dispatchMatmul` that, for the first model dispatch (not
   self-test), records the dst tensor metadata and (via a follow-on
   `mapAsync`) the first 8 f32 values written by the kernel.
   - If non-zero → kernel works correctly in-context too; bug is in
     the JSEP→CPU CPY path.
   - If zero → src1 (the activation input) is already zero when the
     kernel runs; bug is upstream (RMS_NORM, MUL, GET_ROWS, or CPY-to-
     JSEP).
3. **Dump the first CPU→JSEP write.** After confirming the first
   matmul's src1 is zero, instrument `jsepWrite` to log the first
   model-side write's (handle, offset, size, first8 bytes). If those
   bytes are zero, the CPU side is producing zero before writing.
4. **Bisect via device pinning.** Re-run the spike with
   `WEBLLM_PIN_TO_JSEP=0` (i.e., back to Stage 0 — WebGPU-only) to
   confirm the model still produces "Paris" with the same wasm build,
   ruling out any orthogonal regression in the Stage 3 changes.

A working hypothesis to test first: **the JSEP backend's
`cpy_tensor = NULL` is forcing every inter-split copy through a
host-roundtrip, and there's an incorrect ordering between the
host-roundtrip's set_tensor and the next compute split's read.** The
flush fixes addressed the WebGPU-side ordering; but the C++ side may
itself be queuing a `set_tensor` before the previous backend's
`graph_compute` has actually executed (rather than just being
recorded). `ggml_backend_synchronize` is the contract for "wait for
all queued work to complete"; if the scheduler doesn't call it in
the inter-split path, the recorded encoder batcher state is
irrelevant — the data simply isn't there yet for the read to capture.

## Why this still counts as Stage 3 progress

The Q4_K kernel is the deliverable for Stage 3 per the TODO brief
("add a Q4_K dispatch path to `dispatchMatmul` so TinyLlama (and every
other Q4_K-quantized model) can decode a token"). The kernel is
correct; the spike no longer aborts on "Q4_K kernel: deferred to Task
7" (the Stage 2 wall); the kernel is exercised against ~134
per-forward-pass real-model dispatches without throwing.

The downstream Outcome A gate ("greedy 'Paris' continuation") was
explicitly listed as one of three possible outcomes in the Stage 3
brief; the actual outcome is the Outcome C path (numerical drift /
nonsense token output). The brief also explicitly anticipated this:
"Compare CPU reference output for a known Q4_K weight tile against the
WGSL output via a one-off probe; fix the bit-packing." That probe ran;
the kernel passed; the bug is therefore not the bit-packing — it is
elsewhere in the Phase 3 pipeline.

Stage 3.5 is the appropriate next slot to localize the elsewhere.
