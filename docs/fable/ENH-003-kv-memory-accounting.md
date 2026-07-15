# ENH-003 — KV-Cache + Scratch Accounting in MemoryPool, with Pressure Events

> **Status**: proposed · **Effort/risk**: Medium · **Depends on**: ARC-002 (weights-level MemoryPool wiring) must land first

## Goal

`MemoryPool` reflects the full doctrine budget — weights + KV cache + scratch — so `canLoad`
answers honestly on the 16 GB floor with Three.js resident, and consumers get the pressure events
the README promises when the envelope tightens.

## Current state

- After ARC-002 (AUDIT.md), `MemoryPool` charges model **weights** at registration and frees on
  unload. KV and scratch are uncharged.
- Doctrine (CLAUDE.md): 16 GB floor → ~10-11 GB WebGPU-visible; Three.js 0.5-1 GB; KV + browser
  overhead 1-2 GB; ~7-8 GB left for the model. Numbers are the calibration targets, not code.
- KV geometry is known at load time from GGUF hyperparameters: bytes ≈
  `2 (K+V) × layerCount × contextLength × kvHeadCount × headDim × 2 (f16 bytes)`. The
  authoritative field names live where the KV cache is actually allocated — find it:
  par-mem `find_code` "allocate KV cache tensors" (repo_id `webllm`) or grep `kv` in
  `src/inference/model-inference.ts` init path; SWA models (Gemma 4) may size some layers by
  window, not full context — mirror whatever the allocation code actually does, don't idealize.
- Scratch: the graph compute arena (`graphMem` sizing in `model-inference.ts`) and staging/readback
  buffers. These are the most double-count-prone — see step 3.

## Implementation steps

1. **KV charge**: in the same registration path ARC-002 instrumented
   (`engine.ts::_buildInferenceAndRegister`), after the inference engine is constructed, compute
   KV bytes from the *actual* allocation parameters (expose a `getKvCacheBytes()` accessor on
   `ModelInference` that reports what it allocated — preferred over re-deriving the formula in the
   engine, which would drift). Charge `memoryPool.allocate(`${modelId}:kv`, bytes)`; free the tag
   in `unloadModel`.
2. **Embedder/encoder paths**: encoders have no KV cache (CLAUDE.md: 32K contexts are "fine for
   embedders (no KV cache)") — the accessor returns 0 there; causal embedders follow their real
   allocation. Do not special-case in the engine; the accessor owns the truth.
3. **Scratch charge (estimate, clearly labeled)**: add `getScratchBytes()` to the inference
   engines returning the graph arena size they compute (`graphMem`) plus any persistent staging
   buffers they own. Charge as `${modelId}:scratch`. **Double-counting guard**: only count buffers
   the engine itself allocates and owns; do NOT add a blanket "WASM heap" figure (weights already
   account for the dominant heap use, and heap growth is observable but not attributable).
   Comment each accessor with what is and isn't counted.
4. **Pressure events**: Read `src/core/memory-pool.ts` for its existing event/callback surface
   (README claims "pressure-based eviction" — the mechanism may exist unwired). Emit
   `memory-pressure` through the engine's existing `eventHandlers` mechanism when
   `usedBytes / budget` crosses 0.9 on any allocate; include `{ used, budget, byTag }` in the
   payload. Do NOT implement auto-eviction — single-model-active doctrine means there is nothing
   safe to evict automatically; the event is the consumer's signal.
5. **Types/docs**: document the accounting granularity on `WebLLMConfig.memoryBudget` JSDoc
   ("estimated: weights + KV + engine-owned scratch; excludes browser/driver overhead").
6. **Tests**: unit-test the accessors against a synthetic hyperparameter set (assert the formula
   against a hand-computed value for a known config, e.g. a qwen3-1.7b-shaped hp block); pool
   test: allocate weights+kv+scratch, assert `canLoad` rejects a second model that would exceed
   budget; event test: handler fires crossing 0.9.

## Files to touch

- `src/core/engine.ts`, `src/core/memory-pool.ts`, `src/inference/model-inference.ts`
  (+ `encoder-inference.ts`, `causal-embedder-inference.ts` accessors), `src/core/types.ts` (JSDoc)
- `tests/` (extend memory-pool/model-manager suites)

## Verification

1. `make checkall`.
2. Browser sanity: load a canonical model on the smoke page; log `memoryPool` usage (a temporary
   console line or the engine's diag surface) and eyeball against expectation: an 8B Q4_K_M ≈ 5 GB
   weights + 4K-context KV in the hundreds-of-MB range. Remove the temporary line.
3. Pressure event: set a deliberately small `memoryBudget` in a scratch page config and confirm
   the event fires at load.

## Rollback

Accessors and charges are additive; revert to weights-only accounting by removing the `:kv`/`:scratch`
allocate/free pairs. No persisted state or public API removal involved.
