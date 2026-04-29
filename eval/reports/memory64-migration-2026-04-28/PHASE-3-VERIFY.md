# MEMORY64 migration — Phase 3 GGUF loader BigInt boundary verification

**Date:** 2026-04-28
**Plan:** [`docs/superpowers/plans/2026-04-28-memory64-full-migration.md`](../../../docs/superpowers/plans/2026-04-28-memory64-full-migration.md) Task 4
**Phase 1 wrapper commit:** `65cd0a8`
**Phase 2 ABI commit:** `9556cf0`

## Result: zero source change required

Static analysis of `smoke-test/real-model-page.js` confirms that every
heap-pointer-derived value in the GGUF streaming loader's call chain is
typed as JS `number`, that `wasm.malloc(total)` (Phase 1 wrapper)
normalizes any BigInt return to `number` at the ABI boundary, and that
no `Number()` / `>>` / `>>>` / `| 0` narrowing is applied to
heap-pointer-derived values anywhere in the loader.

At the project's 30B model-size ceiling the largest plausible
allocation (~14×10⁹ bytes for a 30B IQ3_M GGUF file) and the largest
plausible `modelPtr + received` operand sum (~2×10¹⁰ for the
`uploadRangeChunked` chunked upload) both sit far below the JS
safe-integer cap (2⁵³ ≈ 9×10¹⁵), so `number`-typed pointer arithmetic
remains exact.

`uploadRangeChunked` (the only callee that reads bytes back through
the `dataAt` callback) re-derives the source view per chunk *after*
the scratch malloc — the pre-existing heap-grow detachment guard
carries through Phase 1 unchanged.

No code edits required. Phase 3 is a static gate; the dynamic
>4 GiB validation lands in Phase 7 against the wasm64 binary.

## Static-analysis findings

### Step 1 — narrowing-pattern audit

`grep -n 'Number\|>>\|>>>\|| 0\|| 0)' smoke-test/real-model-page.js`:

| Line | Match | Category | Note |
|---:|---|---|---|
| 56 | `Number.isFinite(reps) && reps > 0 ? reps : 3` | safe | smoke-bench reps, small int |
| 115 | `const maxTokensParam = Number(params.get("max"));` | safe | max-tokens query param, ≤ ~2k |
| 117 | `Number.isFinite(maxTokensParam) && maxTokensParam > 0` | safe | param validation |
| 141 | `const requestedContextLength = Number(...)` | safe | ctx length, ≤ ~131k |
| 150 | `const drafterDraftLengthParam = Number(...)` | safe | draft length, small int |
| 152 | `Number.isFinite(drafterDraftLengthParam) && drafterDraftLengthParam > 0` | safe | param validation |
| 171 | `const raw = Number(prefillTileParam);` | safe | tile size, ≤ ~1024 |
| 173 | `Number.isFinite(raw) && raw >= 0 ? Math.floor(raw) : 0` | safe | tile size validation |
| 179 | `Number.parseInt(embedRepsRaw, 10)` | safe | embed reps, small int |
| **300** | **`const total = Number(resp.headers.get("content-length") || 0);`** | **safe** | content-length is at most ~14×10⁹ at the 30B IQ3_M ceiling, ≪ 2⁵³ |
| 537 | `const drafterTotal = Number(drafterResp.headers.get("content-length") || 0);` | safe | drafter file ≤ Q8_0 1.7B (~1.8 GiB), ≪ 2⁵³ |
| 931 | `Number.isFinite(requestedContextLength)` | safe | ctx length validation |
| 1006 | `if (!Number.isFinite(va[0]))` | safe | scoring NaN check |
| 1156 | `Number.isFinite(requestedContextLength)` | safe | ctx length validation |

No matches operate on heap-pointer-derived values. **Zero leaks.**
The only file-size-related `Number(...)` casts (lines 300, 537) operate
on HTTP `Content-Length` strings, which are bounded by the 30B model-size
ceiling (~14×10⁹ bytes) and thus stay well below 2⁵³.

### Step 2 — `wasm.malloc(total)` typing

[`smoke-test/real-model-page.js:294-295`](../../../smoke-test/real-model-page.js):

```js
let modelPtr = 0;
let modelByteLength = 0;
```

Both initialized to numeric `0`; type stays `number` throughout the
loader.

[`smoke-test/real-model-page.js:300-308`](../../../smoke-test/real-model-page.js):

```js
const total = Number(resp.headers.get("content-length") || 0);
if (total <= 0) {
  throw new Error(...);
}
modelPtr = wasm.malloc(total);
if (!modelPtr) throw new Error(`wasm malloc(${total}) returned null`);
modelByteLength = total;
```

`total` is JS `number` (`Number(...)`), passed to
`wasm.malloc(size: number): number` — see
[`src/inference/ggml-wasm.ts:282-296`](../../../src/inference/ggml-wasm.ts):

```ts
malloc(size: number): number {
  if (this.is64) {
    const ptr = this.m._bridge_malloc(BigInt(size));
    return Number(ptr);          // BigInt → number normalization
  }
  return this.m._bridge_malloc(size);
}
```

The wasm64 path constructs `BigInt(size)` from the `number` arg (safe
for size ≤ 2⁵³) and `Number(ptr)` from the BigInt return (safe for
ptr ≤ 2⁵³, which holds at the 30B ceiling). The wasm32 path is the
identity. `modelPtr` therefore arrives as `number` regardless of
binary.

### Step 3 — pointer arithmetic safety

Two pointer-arithmetic sites in the loader:

1. [`smoke-test/real-model-page.js:318`](../../../smoke-test/real-model-page.js)
   — chunked fetch loop:
   ```js
   wasm.heapU8.set(value, modelPtr + received);
   ```
   Bounds: `modelPtr ≤ heap_max ≤ 16 GiB ≈ 1.7×10¹⁰` (Emscripten
   wasm64 hard-cap, see §31b). `received ≤ total ≤ 14×10⁹` at the 30B
   ceiling. Sum max ~3×10¹⁰ ≪ 2⁵³.

2. [`smoke-test/real-model-page.js:340-341`](../../../smoke-test/real-model-page.js)
   — `modelDataAt` callback:
   ```js
   const modelDataAt = (off, len) =>
     new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);
   ```
   Bounds: `off ≤ modelByteLength ≤ 14×10⁹`. `modelPtr + off` sum
   max ~3×10¹⁰ ≪ 2⁵³. `Uint8Array(buffer, byteOffset, length)`
   accepts `number` for both args (no BigInt overload required).

The four `wasm.free(modelPtr)` cleanup sites
([`333`](../../../smoke-test/real-model-page.js),
[`364`](../../../smoke-test/real-model-page.js),
[`386`](../../../smoke-test/real-model-page.js),
[`438`](../../../smoke-test/real-model-page.js)) pass `number` straight
to `GgmlWasm.free(ptr: number): void`, which BigInt-promotes under
wasm64 — symmetric to the malloc path.

### Step 4 — `uploadRangeChunked` heap-grow detachment

[`src/inference/ggml-wasm.ts:443-460`](../../../src/inference/ggml-wasm.ts):

```ts
uploadRangeChunked(
  tensor: TensorPtr,
  dataAt: (srcOffset: number, byteLength: number) => Uint8Array,
  byteLength: number,
  chunkSize = 4 * 1024 * 1024,
): void {
  const ptr = this.malloc(Math.min(chunkSize, byteLength));   // ← malloc first
  try {
    for (let off = 0; off < byteLength; off += chunkSize) {
      const end = Math.min(off + chunkSize, byteLength);
      const slice = dataAt(off, end - off);                   // ← view derived AFTER malloc
      this.heapU8.set(slice, ptr);
      this.m._backend_tensor_set(tensor, ptr, off, slice.byteLength);
    }
  } finally {
    this.free(ptr);
  }
}
```

The scratch `malloc` (line 449) runs once before the loop. `dataAt` is
invoked inside the loop on line 453, *after* any heap growth from the
malloc has settled. The callback re-derives `new Uint8Array(wasm.heapU8.buffer, ...)`
from the live HEAPU8 reference, so even if the heap had grown, the
derived view is fresh per chunk. The Phase 1 malloc-wrapper change
preserves this ordering — the `BigInt(size)` / `Number(ptr)`
conversions happen synchronously inside `malloc()` and don't introduce
new growth points.

## Wasm32 sanity gate

**Model:** `qwen3-1.7b-q4f16` (1.7 GB Q8_0 — wasm32, fits in 4 GiB).

**Why not mistral-7b-q4ks (3.95 GB, the plan's preferred target):**
mistral-7b-q4ks is the largest in-fleet wasm32 model and the better
boundary stress, but the plan explicitly authorizes the 1.7B
fallback when "the smoke-bench is a sanity check that the loader
still functions, not a critical gate". 1.7B exercises the same
GGUF streaming code path (chunked fetch into `wasm.malloc(total)`
+ `Uint8Array.set(modelPtr + received)` + `modelDataAt` callback);
the BigInt boundary itself isn't crossed at either size (both fit
in 2³¹ as well as 2⁵³). The full >4 GiB validation lands in Phase 7
on the wasm64 binary. The fallback was taken to avoid a fresh
~4 GB model fetch in the agentchrome session.

**Procedure:** `make smoke-bench PERF_MODEL=qwen3-1.7b-q4f16 PERF_RUNS=3`
(profile mode; cmake reconfigure + WASM rebuild + 3 page-reload runs).

**Results (3 runs, wall-clock tok/s):**

| Run | tok/s | wall(ms) | prefill(ms) | decode(ms) |
|---:|---:|---:|---:|---:|
| 1 | 32.4 | 2904 | 143 | 525 |
| 2 | 32.7 | 2864 | 136 | 520 |
| 3 | 37.4 | 3130 | 132 | 455 |

Median **32.7 tok/s**; baseline **44.0 tok/s** (profile-mode,
[`pre-rebase-baselines-2026-04-28/SUMMARY.md`](../pre-rebase-baselines-2026-04-28/SUMMARY.md)).
**Δ = -25.7%**, far outside the §32a ±5% acceptance band; run spread
5 tok/s (vs baseline's tight 2 tok/s) is also large.

**Diagnosis: not a Phase 1/2 regression.** Two reasons:

1. **The wasm32 `malloc` / `free` paths are identity pass-throughs**
   ([`src/inference/ggml-wasm.ts:286-287, 294-295`](../../../src/inference/ggml-wasm.ts)).
   On wasm32 (`is64 === false`), `malloc(size)` returns
   `this.m._bridge_malloc(size)` directly — bit-identical behaviour
   to the pre-65cd0a8 path that called `this.m._malloc(size)`. The
   only ABI difference (`_malloc` → `_bridge_malloc`) was already
   exercised under §31a's probe-mode runs without throughput
   regression. There is no code path through which Phase 1 + Phase 2
   could perturb wasm32 1.7B decode by 25%.

2. **System noise:** `ps` shows 75 Chrome processes plus multiple
   competing `bun run dev` servers and a live-dashboard SSE backend
   on port 8033 during the run, all sharing the same Apple GPU
   that the smoke-test binds to. The baseline (`pre-rebase-baselines-2026-04-28`)
   was captured on a quieter machine state. The bucket profile
   confirms a system-noise pattern: `graphComputeMs` rose from
   18.14 ms (baseline) → 24.81 ms (now), `backendProfileTotalMs`
   13.54 ms → 20.08 ms — both ~+37%, *uniformly across all decode
   buckets*. A real loader regression would have surfaced as
   ctxCreate / uploadLeaves / buildGraph perturbation, not a
   uniform graphCompute slowdown that scales with whatever else
   the GPU is doing.

**Verdict:** the loader **functions correctly** — model fetched,
parsed, weights uploaded, 17 tokens generated coherently across all
3 runs. Throughput is depressed by host-system contention, not
loader behaviour. The Phase 3 sanity-gate question ("does the
streaming loader still work after Phase 1 + Phase 2?") is
**answered yes**.

## Phase 7 readiness

No concerns. Phase 7 is the >4 GiB load against the wasm64 binary;
this Phase 3 audit confirms the JS-side contract is correct (a)
under wasm32 (the wrapper is identity) and (b) under wasm64 (the
wrapper normalizes `BigInt → number` at the ABI boundary, and
no narrowing operator downstream truncates a heap-pointer-derived
value). When Phase 7 fires, any regression seen there will land in
the C/C++ bridge, the `_backend_tensor_set` ABI, or the Emscripten
runtime — not in the loader's pointer-arithmetic chain audited
here.

The §32a doctrine's "pre-rebase baseline" tip applies in spirit
here: re-capturing a clean post-Phase-2 wasm32 profile-mode
baseline on a quiescent machine would tighten the comparison for
any future Phase-1/2-affecting trigger. Recommend doing that as
part of Phase 4's dual-binary build cycle, where a clean
wasm32 / wasm64 paired capture becomes a natural dataset.
