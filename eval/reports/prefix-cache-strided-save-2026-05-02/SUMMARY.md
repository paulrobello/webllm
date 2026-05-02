# Prefix-cache strided-save probe — qwen3-8b-iq3m

**Date:** 2026-05-02
**Verdict:** **NEGATIVE (§28 template)** — strided per-head reads cut snapshot payload by 3× (576 MB → 195 MB) but moved Pattern B tick-2 wall by ~0% on the interleaved probe. Per-call overhead at 576 strided reads exactly cancelled the bandwidth savings vs 72 full-tensor reads.

## Hypothesis

After Phase 1a + 1b (commits `71ea997` + `979593f`), the at-scale probe summary diagnosed the residual ~300 ms gap as bandwidth-bound: ~600 MB transferred per save+load against ~1-2 GB/s effective WebGPU↔CPU bandwidth. Strided per-head reads (only `[0, nTokens)` slots per slab instead of the full `maxCtx`-sized tensor) would cut readback payload by `maxCtx / nTokens` ≈ 3.1× at 1325/4096. Projected to take Pattern B tick-2 wall from ~2.72 s to ~2.0-2.2 s.

## Implementation

`serializeKVCache` was modified to issue `headCountKv` separate offset reads per tensor instead of one full-tensor read. The existing `beginDownloadFromTensor(tensor, byteLength, offset)` API already supports offset/size, so no C++ patch was needed. Total request count grew from `2 × layerCount = 72` to `2 × layerCount × headCountKv = 576` on Qwen3-8B. All requests were pipelined via `_backend_tensor_get_async_begin` before any await.

The output buffer layout was unchanged (`[layer0.K_compacted | layer0.V_compacted | ...]`), so the load side and on-disk format stayed identical.

## Measurement

Same matrix as `probe-prefix-cache-interleaved-2026-05-02` (4 NPCs × 2 ticks × 2 patterns, round-robin, ~1100-token per-NPC distinct personas, FA on).

| build                                  | Pattern B tick-2 wall (median) |
|----------------------------------------|--------------------------------|
| pre-strided (post Phase 1a+1b + #694)  | 2719 ms (commit `752421c`)     |
| **strided save (576 begin-calls)**     | **2736 ms (this probe)**       |

Difference is +17 ms — within run-to-run noise. Pattern A and tick-1 walls were also unchanged.

## Why the prediction was wrong

**The readback wasn't actually bandwidth-bound at the JS-visible level.** Phase 1a already pipelined all 72 begin-calls, letting the WebGPU command queue overlap GPU→CPU copies. The wall-time cost of pipelined readback was already small relative to other components.

A more accurate Pattern B tick-2 wall decomposition (qwen3-8b-iq3m at this scale):

| component                                   | est. ms |
|---------------------------------------------|---------|
| Decode (32 tokens at ~23 tok/s on IQ3 8B)   | ~1374   |
| Load (24 batched `backendTensorSet3` calls) | ~1100   |
| Tail prefill (~80 tokens × 12.5 ms)         | ~46     |
| Save (overlapped with subsequent JS work)   | small   |
| Misc bookkeeping                            | small   |
| **Total**                                   | ~2700   |

The dominant non-decode I/O is the **load** path (~1100 ms), not the save. Load is sync ASYNCIFY-suspending — round-trip count matters, not bandwidth. Strided writes via single-call `backendTensorSet` would *regress* (576 sync calls × ~1.8 ms = ~1037 ms vs current ~1100 ms; net wash even if zero bandwidth, and worse because each call still pays bandwidth).

To improve load further would require a new C++ batch primitive (e.g., `backend_tensor_set_strided` taking a list of (offset, size) pairs and writing them all in a single FFI call). That's patch territory on `webllm-browser-patches` and adds rebase drift surface against upstream. Per the workflow doctrine, scoring this lever:

- Maintenance burden: +1 patch on the stack (current 11 patches).
- Surface area: medium — new C++ primitive, JS wrapper, error paths.
- Risk to load-bearing invariants: low — additive primitive, doesn't touch existing paths.
- Reversibility: high — can revert without affecting current correctness.
- External-dependency exposure: high — patch must rebase cleanly past upstream `ggml-webgpu` changes.

**Decision: defer.** Without a measurement showing the C++ primitive would actually win on wall-time (rather than just reduce bandwidth that isn't visible at the JS level), adding a patch is speculative. Pattern B is already correct (post-#694) and ~6× faster than Pattern A in the workload that matters (interleaved). The remaining ~300 ms gap vs Pattern A in the *sequential* workload is a curiosity, not a blocker — the sequential workload prefers Pattern A's session-tracker fast-path anyway.

## Probe-first doctrine: what we learned

This probe followed the §27 free-win sweep pattern intent (cheap experiment before commit) and produced a §28 template outcome (lever closed harder than expected). The bandwidth-bound floor estimate from the at-scale SUMMARY's "v2 #1 trajectory" section was overly optimistic — it assumed JS-visible bandwidth would dominate after pipelining, when in reality the WebGPU runtime already overlaps reads at the command-queue level.

**Updated mental model:**

- **Save (post Phase 1a):** small wall-time component, hidden behind GPU command queue. Strided reads can't speed it up further at the JS level.
- **Load:** dominant per-call I/O cost. Bandwidth + sync ASYNCIFY round-trip count both matter. Phase 1b already attacked round-trips (72→24); further wins need a strided-write primitive.
- **Bandwidth-bound floor:** real, but at the *GPU/runtime* level. Whether it's wall-visible depends on whether the work overlaps with other JS+GPU activity.

## Caveats

- Single-run probe. Run-to-run noise on tick-2 wall is ~50-100 ms; +17 ms delta is within that band.
- Did not test at-scale (sequential) probe — Pattern B's mechanism is identical, so result should be the same.
- Did not measure JS heap allocation peak (576 × 339 KB = ~195 MB peak vs 72 × 8 MB = ~576 MB peak). Strided would have saved ~380 MB of transient heap; might matter on memory-constrained devices, but no measurement and no observed pressure on 16 GB+ baseline.
