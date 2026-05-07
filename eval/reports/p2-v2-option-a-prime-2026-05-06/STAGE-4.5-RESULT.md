# Stage 4.5 — H1 unconditional GPU→host writeback for JSEP-produced tensors

**Date:** 2026-05-06
**Status:** **PARTIAL OUTCOME A — H1 fires correctly (jsepRead = 1602 round-
trips per pass; Stage 4.4 baseline was 0). Real finite logits preserved.
Bug A still fixed. Decoded text changed but "Paris" still NOT achieved.
Bug C diagnosis (GPU→host writeback gap) is necessary but NOT sufficient
— deeper bug exists in the consumer pipeline.**
**Patch stack:** 7 (unchanged — single C++ change inside the existing F1
patch; no new patches added).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.5-fix-v2`
**Per-token decode:** **25.04 ms** (Stage-4.4 baseline 23.30 ms; +7%).
H1 cost on TinyLlama is much smaller than the brief estimated (50-150
ms). Likely because most ops have small dst tensors (KV-cache rows,
RMSNorm outputs); the per-op `jsepRead` fixed cost dominates over
bytes-read scaling.

## Goal (recap from Stage 4.5 brief in TODO.md)

Stage 4.4 (F1) fixed Bug A by giving JSEP buffers a parallel host
mirror so CPU-fallback ops dereferencing `tensor->data` read real
weight data instead of the `0x2000` sentinel. But the model still
decoded incorrect tokens, and the smoking gun was
`FIRST_ALLZERO_DST_PROBE = {i:3, op:42 (SET_ROWS), dstH:18, ...}` with
`COUNTER_DELTAS.read = 0` — confirming JSEP ops wrote to the GPU
buffer but never updated the host mirror, so downstream peeled-to-CPU
consumers read stale (initial-zero) bytes. H1 was the strongest
possible writeback: after every `jsepRunOp`, mirror the freshly-
written dst back into `host_mirror + offset` via `ggml_jsep_read`.

## Implementation

Single change in
`~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:493-509` (12 LOC
inside `ggml_backend_jsep_graph_compute`'s per-node loop, immediately
after the `STATUS_FAILED` check). No CMakeLists changes; no descriptor
ABI changes; no JS-side changes. Reuses the existing `ggml_jsep_read`
EM_JS forward-decl that F1 had reduced to dead code.

```cpp
// H1: GPU→host mirror writeback. Mirror every dispatched node's dst
// back so downstream CPU-fallback ops (which dereference tensor->data
// → host mirror) read current GPU output, not stale-zero from the
// alloc-time init. Cost: one jsepRead per JSEP op (~1602 round-trips
// per pass on TinyLlama). H3 will refine this to peeled-consumer-only.
if (node->buffer && node->buffer->context) {
    auto * bctx = static_cast<ggml_backend_jsep_buffer_context *>(node->buffer->context);
    const size_t dst_offset = jsep_tensor_handle(node);
    const size_t dst_size   = ggml_nbytes(node);
    ggml_jsep_read(bctx->handle, (int32_t) dst_offset,
                   (int32_t) (uintptr_t) ((char *) bctx->host_mirror + dst_offset),
                   (int32_t) dst_size);
}
```

The metadata-op fast-path (NONE / VIEW / RESHAPE / PERMUTE / TRANSPOSE
at `graph_compute:421-425`) already `continue`s before reaching the
dispatch loop, so H1 only fires for ops that actually wrote data.

## Bonus bug fixed: Makefile build-order race

Stage 4.5 surfaced (and fixed) a latent Makefile race in
`make wasm-build-jsep`. The order was:
1. `cmake --build` → produces fresh `src/wasm/build-jsep/webllm-wasm-jsep.{js,wasm}`
2. `bun run build:jsep` (bundles `src/index-jsep.ts` → `smoke-test/webllm-bundle-jsep.js`)
3. `bun build smoke-test/p2-v2-spike.src.ts` → bundles spike, which imports `./webllm-wasm-jsep.js` from `smoke-test/`
4. `cp src/wasm/build-jsep/webllm-wasm-jsep.{js,wasm} smoke-test/`

Step 3 ran while `smoke-test/webllm-wasm-jsep.js` still held the
**previous** build's content. Bun bundled the spike against that
stale shim. As long as the wasm import set was unchanged across
builds, this was invisible. But Stage 4.4's F1 made `ggml_jsep_read`
unused (get_tensor switched to `memcpy` from mirror), so the linker
DCE'd `ggml_jsep_read` from the wasm imports. The next build (Stage
4.5 H1) re-introduced it to the wasm imports — but the bundle was
still pointing at the Stage-4.4 shim, which had no `ggml_jsep_read`
function or wasmImports entry. Result: `WebAssembly.instantiate(...)`
threw `LinkError: Import #53 "env" "ggml_jsep_read": function import
requires a callable`.

Fix: move the `cp` step to run *before* the bundling steps. New order
in `Makefile:152-168`:
1. cmake build
2. cp wasm artifacts to smoke-test/ (now bundlers see fresh shim)
3. bun bundle (jsep, spike, offload-probe)

Stage 4.5 H1 build went green on the second attempt. The fix is
permanent — any future build that adds or removes a wasm import will
no longer leave the spike bundle stale by one cycle.

## Spike result — Stage 4.4 baseline vs Stage 4.5 H1

| Marker                              | Stage 4.4 baseline                 | Stage 4.5 H1                       | Δ                |
|-------------------------------------|------------------------------------|------------------------------------|------------------|
| `LOGIT_STATS_STEP0.first8`          | `[0.0060, 0.0047, -0.0102, ...]`  | `[0.0060, 0.0047, -0.0102, ...]`  | **identical**    |
| `LOGIT_STATS_STEP0.topId / topVal`  | `593 / 0.1591`                     | `593 / 0.1591`                     | **identical**    |
| `LOGIT_STATS_STEP0.hasNaN/hasInf`   | `false / false`                    | `false / false`                    | Bug A still fixed |
| `LOGIT_STATS_STEP0.finiteCount`     | `32000 / 32000`                    | `32000 / 32000`                    | full vocab finite |
| `GENERATED_TOKENS`                  | `[593, 5871, 945, 16976, 25487]`   | `[593, 5871, 15669, 15565, 12150]` | tokens 2-4 changed |
| `GENERATED_TEXT`                    | `"ntiuracinateenes"`               | `"ntiuhuihnerquant"`               | both gibberish   |
| `FIRST_NAN_DST_PROBE`               | `null`                             | `null`                             | Bug A stable     |
| `FIRST_ALLZERO_DST_PROBE`           | `{i:3, op:42 (SET_ROWS), dstH:18}` | `{i:3, op:42 (SET_ROWS), dstH:18}` | **still firing** |
| `COUNTER_DELTAS.read`               | `0`                                | **`1602`**                         | H1 dispatching   |
| `COUNTER_DELTAS.runOp`              | `1602`                             | `1602`                             | unchanged        |
| `COUNTER_DELTAS.write`              | `1206`                             | `1206`                             | unchanged        |
| `PER_TOKEN_MS`                      | `23.30`                            | `25.04`                            | +7.5%            |
| `CROSSINGS_PER_TOKEN`               | `1295.8`                           | `1616.2`                           | +320 (read calls) |
| `MODEL_LOAD_MS`                     | `290`                              | `290`                              | unchanged        |
| 4 kernel selftests                  | PASS                               | PASS                               | regression-free  |
| `make checkall`                     | green                              | green                              | ship-gate clean  |

H1 is doing exactly what it was supposed to: 1 `jsepRead` per `runOp`
(1602 = 1602 — every dispatched op now mirrors its dst back). Read
deltas matched the `runOp` count exactly, confirming no path is
silently skipping the writeback.

## Diagnosis — why "Paris" still doesn't decode

Step 0 logits are bit-exactly identical between Stage 4.4 and Stage 4.5.
That makes sense: token 1 is produced by the prefill graph alone, where
the consumer chain inside the JSEP backend is well-mixed (no peeled
producer feeds a peeled consumer; or if it does, the prefill consumer
runs identically on stale-zero vs current data because the test prompt
is short enough that the relevant rows weren't dirty yet).

Tokens 2-4 changed (945→15669, 16976→15565, 25487→12150). That confirms
H1 *did* affect data flow — but in the wrong direction: the change is
data-quality-neutral (still gibberish), not curative.

The smoking gun is `FIRST_ALLZERO_DST_PROBE` continuing to fire on
the same op (op 42 SET_ROWS, dst handle 18 = KV cache, dst offset 0).
The probe's check is "after running this op, does jsepRead at the
node's dst offset return all-zero `first8`?" Post-H1, the answer is
still yes. Two readings are possible:
- (a) The KV cache row written by SET_ROWS at offset 0 *legitimately*
  contains zeros at the start of the value vector (the K vector's
  first 8 dimensions might be ≈0 by RMS normalization properties).
  In that case the probe is a false positive and the bug is elsewhere.
- (b) SET_ROWS is genuinely not writing to the right offset / wrong
  cell / wrong layout. The dispatch fires (no GPU validation error;
  no diagnostic NaN cascade), but the destination row stays zero or
  the data lands in the wrong spot.

Reading (b) is the more load-bearing concern because TinyLlama uses
the **transposed V-cache layout** (`llama-kv-cache.cpp:1281` —
`ggml_reshape_2d(v, 1, ggml_nelements(v))` puts ne[0]=1, adjacent
indices share a u32 word). Stage 1's `dispatchSetRows` had to add a
F32→F16 atomic-CAS path for exactly this case. If the atomic CAS
indexing is off-by-one, or the wrong source row is read because of
the row-vs-cell unit confusion, the V-cache writes go to the wrong
place. Forward pass would still produce *something* (real-but-wrong
tokens, exactly what we see), and the dst offset check would still
fire because the buffer offset 0 might not be the cell that was
written.

## Branch on outcome (per Stage 4.5 brief)

The brief identified three paths:

- **Outcome A flips (Paris)** → file Stage 4.6 (H3 graph-walk pre-pass
  for perf recovery). **Not applicable here.**
- **Outcome A still doesn't flip (real but wrong tokens)** → "the bug
  is deeper than just GPU→host mirror staleness. Likely candidates:
  KV cache layout (dispatchSetRows F32→F16 atomic CAS path), ROPE
  position indexing, attention masking." **This is us.**
- **Outcome A regresses to NaN cascade** → H1 introduced a bug.
  **Not us — Bug A still fixed; logits stable.**

So the live next step is **Stage 4.6 — KV cache write correctness
(SET_ROWS into transposed V-cache)**. The TODO.md brief for Stage 4.6
will localize the bug via:
1. **dispatchSetRows real-shape selftest with the V-cache transpose
   layout** (currently the selftests use rows=K=2048 generic shapes;
   add a case with ne[0]=1 + I64 indices that exercise the F32→F16
   atomic-CAS path on adjacent indices that share a u32).
2. **Per-SET_ROWS pre/post jsepRead** to verify the row written by
   the dispatch matches the source data (currently the spike's
   instrumentation captures dst pre/post but doesn't compare against
   src expected values).
3. **CPU reference comparison** — for each SET_ROWS dispatch, run a
   CPU-equivalent and diff the host_mirror after H1 writeback; the
   first divergent SET_ROWS pinpoints the kernel bug.

## Code references

- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:493-509` — H1 writeback (this stage's only C++ change)
- `~/Repos/webllm/Makefile:152-168` — build-order fix (cp-before-bundle)
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:149-155` — `ggml_jsep_read` EM_JS forward decl (now active again post-H1)
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp:200-292` — F1 buffer interface (Stage 4.4) — host mirror lives here, H1 just keeps it current
- `~/Repos/webllm/src/inference/jsep/ops/set-rows.ts` — Stage 1 + 4.1 SET_ROWS dispatch (suspected Stage 4.6 bug locus)

## Patch stack

Unchanged at 7. The Stage 4.5 H1 change is a 12-LOC addition inside
the existing F1 patch (`ggml-jsep: F1 dual-resident host mirror —
Phase 3 / Option A-prime Stage 4.4`). The patch's commit message
will be appended with an "H1 follow-on" note, or — cleaner —
amended in-place to bundle F1 + H1 as one logical change (the
host mirror is only useful when kept current). Choice deferred to
the rebase cycle that next pulls this branch upstream.

## Spike state cheat sheet (handed off to Stage 4.6)

- **Spike URL pattern:** `http://localhost:8031/p2-v2-spike.html?v=stage4.6-<probe>`
- **Per-token decode (post-Stage-4.5 H1):** ~25 ms — H1 cost smaller than
  predicted; H3 graph-walk pre-pass is now lower-priority (only ~7%
  perf to recover before the model is correct).
- **Built-in self-tests** still permanent: Q4K K=256, RMSNORM rows=1,
  RMSNORM rows=6/K=2048, MATMUL Q4_K M=64/K=2048/N=6 (no-divert and
  divert variants). **None exercise the V-cache transpose layout** —
  Stage 4.6 should add SET_ROWS rows=1/I64-indices/atomic-CAS case.
- **Build:** `make wasm-build-jsep` from webllm root. Now correctly
  bundles fresh shim (Stage 4.5 Makefile fix).
- **`make checkall` green** post-Stage-4.5.
- **Tab continuity:** the spike URL is loaded in the agentchrome
  session at port 64629; reuse it. New tabs cost ~30s in browser
  startup + WebGPU adapter request.
