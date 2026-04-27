# §D Encoder Perf Pass — Design

**Date:** 2026-04-27
**Status:** Design (pre-implementation; awaits writing-plans handoff)
**TODO entry:** §D in `TODO.md` "Active next steps" (deferred kernel-tuning targets)
**Pattern:** Mirrors the §17 / §18 / §19 / §20 measure-and-close template.

## Goal

Characterize current `engine.embed()` throughput and latency on the
arctic-embed encoder fleet (33M `arctic-embed-s` and 109M `arctic-embed-m`),
identify hotspots through profile-mode backend attribution, then ship the
levers that pass acceptance gates. Both single-text latency and batch
throughput are in scope. The cycle ends when all candidate levers have been
measured and either shipped or closed with documented rationale.

The encoder forward pass landed 2026-04-24 and works correctly
(`cosine('happy', 'joyful') ≈ 0.77`; 8/8 tasks / 93% accuracy on
`bench-full --profiles arctic-embed-s`). It has not been touched for
performance work. This is its first dedicated cycle.

## Non-goals

The following are intentionally out of scope for this cycle. Each is a
real candidate revisited only if Phase 1 data forces it or a follow-up
cycle picks it up.

- **Concatenated-graph batched compute.** Running K texts in one graph
  with cross-text attention masks. L3 alternative to sequential-loop
  implementation. Revisited only if measured headroom remains after
  Phase 4.
- **Encoder kernel work** (matmul tuning, attention shape rewrites,
  Q-quant variants on encoder weights). BERT uses standard `mul_mat`,
  not the GEMV path §A targeted; F16 weights, not Q-family. Same
  applicability constraints as §17 will likely apply.
- **Wave-2 bigger encoder model registration** (e.g., `bge-m3` ≈ 568M,
  `gte-large-en-v1.5` ≈ 434M). Mirrors §10 wave-2 pattern — register
  only after current fleet shows lever ceiling rising with size.
- **Quantized encoder weights.** Arctic-embed ships F16; no Q4 GGUF
  in pipeline. Quant would be its own cycle.
- **Multi-encoder concurrent `embed()`.** Two different encoder models
  in parallel. Out of scope; per-`EncoderInference` levers only.
- **Streaming `embed()` API.** Encoder is single-shot. N/A.
- **MEMORY64 for larger encoders.** 4 GiB cap not in play.

## Workload definitions

Two workloads are measured. The same lever may target one or both.

- **Single-text latency.** One `embed("text")` call. Metric: p50 wall
  time (ms) over N = 30 reps after a 5-rep warmup. Captured in two
  fixtures:
  - `short`: `"happy"` (≈3 tokens after `[CLS] ... [SEP]` framing).
  - `long`: ≈200-token English paragraph (specific text fixed in
    `eval/fixtures/embed-prompts.ts`, committed alongside the harness).
- **Batch throughput.** Either `embedBatch(texts)` (once API lands) or
  sequential `embed()` × K (baseline). Metric: texts/second over a
  fixed mixed batch (K = 64; 32 short + 32 long). p50 of 3 trials.

Both workloads run on both arctic-embed-s and arctic-embed-m, with both
non-profile and `--profile` mode. Non-profile numbers drive throughput
claims and gate decisions. Profile-mode numbers drive hotspot ranking.
Mixing the two is the failure mode that derailed the Apr-22 regression
investigation (see preamble of TODO §10).

## Architecture & touch points

- **Measurement (new file).** `eval/embed-perf.ts` — sibling to
  `eval/perf.ts`. CLI flags: `--model <id>` (default both), `--mode
  <single|batch|both>` (default `both`), `--reps <N>` (default 30 single
  / 3 batch), `--profile` (off by default), `--out <dir>` (default
  `eval/reports/embed-perf-<YYYY-MM-DD>/`). Drives the smoke-test page
  with appropriate URL params; collects backend attribution traces from
  the page in profile mode using the `eval/perf.ts::fetchDecodeTraces`
  pattern (renamed/generalized to fit encoder workloads). Writes one
  log per (model, fixture, mode) cell, plus `summary.md` with the
  baseline table.
- **Test fixtures (new file).** `eval/fixtures/embed-prompts.ts` exports
  `EMBED_PROMPTS = { short, long, batchMixed }`. Pinned text content;
  single source of truth for both `embed-perf.ts` and any unit tests.
- **Levers — `src/inference/encoder-inference.ts`** (additive, lever-
  by-lever; see Phase plan below for which lever lands in which phase).
- **Public API surface — `src/core/engine.ts`.** Adds
  `engine.embedBatch(modelId, texts)` in Phase 4. `engine.embed()`
  signature and semantics unchanged across the cycle.
- **Smoke page — `smoke-test/real-model-page.js`.** Step `[8/8]` keeps
  the existing single-text `embed()` call as the integration regression.
  No URL-param or default-on changes.
- **Dashboard.** Existing Embeddings section (cosine / median latency /
  throughput) ingests the harness output via the existing
  `live-server.ts` SSE/SQLite path. No new dashboard work.
- **Tests.**
  - `tests/encoder-cosine-parity.test.ts` (new): pre-/post-lever cosine
    on `('happy', 'joyful')` within ±0.005, run as part of
    `make checkall` against a captured pre-cycle baseline.
  - `tests/embed-batch.test.ts` (new, lands with Phase 4): asserts
    `embedBatch([t])` is byte-for-byte equal to `[embed(t)]`.
  - Existing `tests/wordpiece-golden.test.ts` and the encoder pool/
    normalize tests remain unchanged guards on correctness.

## Phase plan

Each phase is a separate commit on `feat/encoder-perf` with its own
ship/skip gate. The lever order after Phase 1 is set by data; the
expected sequence below is based on inspection only.

### Phase 1 — Harness + baseline

Land `eval/embed-perf.ts` and `eval/fixtures/embed-prompts.ts`. Capture
non-profile and `--profile` numbers on both arctic models × short/long
fixtures × single/batch modes. **No code changes outside `eval/`.**

**Output:** `eval/reports/embed-perf-2026-04-27/summary.md` with:
- Per-(model, fixture, mode) p50 wall time and per-text rate.
- Profile-mode breakdown: `graphComputeMs`, `downloadResultMs`,
  build+alloc+upload, teardown — all in ms and as fraction of step
  total.
- Profile-mode backend attribution: `backendMatmulMs`,
  `backendEncodeOverheadMs`, `backendAttentionMs`,
  `backendDispatchCount` per cell.

**Decision rule:** the rank of the four buckets above sets the lever
order for Phases 2–4. If `downloadResultMs` is ≥15% of step total on
single-text → L2 promotes ahead of L1. If
`build+alloc+upload+teardown` is ≥15% → L1 first (expected). If matmul
or encode dominates and the others are small → L4 promotes; L1/L2
demoted.

### Phase 2 — Lever 1 (expected: graph / ctx reuse)

Change: keep ggml ctx + graph buffer alive across `embed()` calls for
the same `EncoderInference` instance. `buildGraph` runs once per
distinct token count `N`; if `N` matches the cached graph's `N`, reuse;
otherwise rebuild (still cheaper than the rebuild-every-call status
quo when `N` clusters around fixture sizes). Per-call work shrinks to
leaf upload + graphCompute + readback + (no teardown).

**Implementation choice (decide at impl time based on `GgmlWasm`
surface):** either (i) split into a long-lived "weight ctx" and a
re-allocatable "graph ctx" — current `loadWeights` already creates a
single ctx that holds both; this lever splits them; or (ii) keep one
ctx and add a `wasm.ctxReset()` between graphs. Whichever lands first
behind a constructor flag `reuseGraph?: boolean` (default `true` once
G1 + G3 pass; off as escape hatch).

**Concurrency:** ctx reuse introduces shared state. Add a per-
`EncoderInference` `Promise`-chain mutex around `embed()` (mirrors
`Generator.generate`) so two parallel `embed()` calls serialize cleanly.
Cheap, no behavioural change for single-call callers.

**Re-measure:** single-text + sequential-batch on both fixtures × both
models, non-profile + profile. Gate: G1 + G3.

### Phase 3 — Lever 2 (expected: GPU pool / readback shrink)

Change: pool happens on the graph. Download `[E]` (E = embedding
length) instead of `[E, N]`. Two sub-cases:

- **CLS pooling** (`hp.poolingType === "cls"`): `opGetRows(finalHidden,
  [0])` → `[E]`. Trivial.
- **Mean pooling**: prefer `opSumRows` / `opMean` if exposed by
  `GgmlWasm`. If not, build via `[E, N] @ [N, 1]` matmul against an
  `oneOverN: TensorPtr` ones-vector × `1/N`. If neither is clean, fall
  back to current `[E, N]` readback + CPU pool, but hoist the L2-norm
  sqrt onto the graph (still some readback shrink). Worst case is a
  partial win, not a blocker — close L2 with the partial result and
  continue.

L2-normalize: a 1-D op of size `E` (≤ 768 for arctic-m). Either keep
on CPU after readback (negligible) or use `opNormL2` if available.

**Re-measure:** single-text only on both fixtures × both models,
non-profile + profile. Gate: G1 + G3.

### Phase 4 — Lever 3 (`embedBatch` public API)

Change: add `WebLLM.embedBatch(modelId, texts: string[]):
Promise<Float32Array[]>` to `src/core/engine.ts` and
`EncoderInference.embedBatch(tokenIdsList: Int32Array[])` to the
inference layer. Implementation: a sequential loop over texts that
reuses the warm ctx from Phase 2. Concatenated-graph batched compute
is *not* part of this phase — see Non-goals; revisit only if Phase 1
data + L1 leaves clear headroom.

**Tests:** `tests/embed-batch.test.ts` asserts byte-for-byte parity
with `[embed(t) for t in texts]` and exercises the empty-list and
single-element cases.

**Re-measure:** batch throughput only on K = 64 mixed fixture × both
models, non-profile + profile. Gate: G2 + G3.

### Phase 5 — Lever 4 / opportunistic

Whatever profile-mode attribution surfaces in Phase 1 that wasn't
covered by L1/L2/L3 and looks shippable. Possible: F16-cast point in
`buildGraph`, dispatch-count reductions in attention shape, redundant
`opCont`/permute hoists. Could be empty if Phase 1 says nothing else
moves measurably. Same per-lever G1 + G3 gate.

### Phase 6 — Closure writeup

Add a `### Completed on YYYY-MM-DD §<N>` entry to `TODO.md` mirroring
§17 / §18 / §20. Final perf table (pre vs post per shipped lever),
levers shipped vs deferred, gate decisions, raw logs in
`eval/reports/embed-perf-2026-04-27/`. Update `Resumption checklist`
to note §D is closed.

## Acceptance gates

Each gate is checked per lever. Only levers that pass ship.

- **G1 — Single-text latency.** A lever ships only if it drops
  `embed()` p50 wall time by ≥10% on at least one of {arctic-embed-s,
  arctic-embed-m} AND no model regresses >3%. Measured non-profile,
  3-trial median, both short + long fixture (gate evaluated on
  whichever fixture the lever targets).
- **G2 — Batch throughput.** Lever 3 (`embedBatch`, K = 64) ships only
  if its texts/sec beats sequential-`embed()` baseline by ≥1.5× on at
  least one model AND no model regresses. Measured non-profile,
  3-trial median, mixed short+long fixture.
- **G3 — Cosine parity + accuracy.** *Every* shipped lever must keep
  `cosine('happy','joyful')` within ±0.005 of pre-cycle baseline AND
  keep `bench-full --profiles arctic-embed-s` at 8/8 tasks / ≥93%
  overall. Correctness gate, applies independently of G1/G2. A lever
  that fails G3 reverts unconditionally.

A lever that hits G3 but misses G1/G2 reverts (no shipping a no-op
change).

## Risks & rollback

- **Ctx lifetime (L1).** Reusing the ggml ctx across calls means
  tensors built in `buildGraph` accumulate inside it. Resolutions:
  split weight ctx and graph ctx (preferred), or `ctxReset` between
  graphs, or ship behind a constructor flag if neither is clean. See
  Phase 2 implementation choice above.
- **GPU pool op availability (L2).** `opSumRows`/`opMean` may not be
  exposed via the WASM bridge. Fallbacks documented in Phase 3.
- **Concurrent `embed()` calls (L1 + L3).** Mutex per `EncoderInference`
  serializes them. Same `Promise`-chain pattern as `Generator.generate`.
- **`embedBatch` API contract.** Additive. `embed(t)` semantics
  unchanged. `embedBatch([t])` byte-for-byte equals `[embed(t)]`,
  pinned by test.
- **Smoke / dashboard fallout.** None expected — `[8/8]` calls
  `embed()` once; ctx reuse is transparent. G3 catches any cosine
  drift before merge.

**Rollback per phase.** Each phase is its own commit on
`feat/encoder-perf`. If a lever fails its gate, `git revert <sha>`;
subsequent phases either rebase onto the new tip or get re-evaluated.
Mirrors the per-piece-revertible commit pattern from §20
(`91d8e26 / 4138232 / 4bfa6f4 / faccb8e`).

## Stop conditions

Cycle closes (Phase 6) when any of:

- All four gates hit by end of Phase 4 → close with shipped wins, no
  Phase 5.
- A lever's measured impact is in the noise *and* nothing else profiles
  as a hotspot → close early; document what was tried.
- Two consecutive levers fail their gate → close, escalate to a
  follow-up brainstorming pass before continuing.

## Resumption pointers

When the next session starts:

1. `git log --oneline -10 feat/encoder-perf` — confirm current phase tip.
2. `cat eval/reports/embed-perf-2026-04-27/summary.md` — last captured
   numbers.
3. This design + the TODO closure entry (once Phase 6 lands) carry the
   full lever inventory and decision history.

## Implementation handoff

After user spec review, this design hands off to `superpowers:writing-
plans` for a phase-by-phase implementation plan keyed to the gates
above. Per global preference, the writing-plans choice is Subagent-
Driven (this session) — execute via `superpowers:subagent-driven-
development`.

## Phase 2 Implementation Choice (2026-04-27)

**Picked option:** Same-graph-cache

**Why:** The bridge (`src/wasm/webgpu-bridge.cpp`) maintains a *stack* of
ggml contexts (`g_ctx_stack`), not a single global pointer. `ctx_create`
pushes, `ctx_free` pops, and `current_ctx()` always returns the top.
This means the existing layout already cleanly separates the weight
ctx (pushed in `loadWeights`) from the per-call graph ctx (pushed
inside `embed()`); the only reason embed() incurs the rebuild cost
today is that it `ctx_free`s the graph ctx in its `finally` block.
`backendAllocCtxTensors` calls `ggml_backend_alloc_ctx_tensors(current_ctx(), …)`
— it allocates a GPU buffer for the tensors in the *top* ctx only, so
keeping the graph ctx alive across calls reuses the same graph buffer
and the same leaf/graph node pointers without touching the weight
ctx underneath. This option therefore strictly dominates options 1
and 2: no bridge change (option 1 was never required — split-ctx is
already the de facto layout), and we avoid option 2's per-call graph
rebuild and metadata leak. Largest expected speedup at the smallest
blast radius.

**Bridge changes:** None. The stack-based ctx API in
`webgpu-bridge.cpp` already supports a long-lived graph ctx; no new
`ctx_switch` / `ctx_reset` / `secondary_ctx` primitive is required.

**API surface added/changed in `EncoderInference`:**

- (private) `ensureGraphCache(N: number): void` — if no cached entry
  matches `N`, tear down any existing cache (pop graph ctx, free graph
  buf), then push a new graph ctx, run `buildGraph(N)`, `graphNew`,
  `graphBuildForwardExpand`, and `backendAllocCtxTensors`, caching
  `{ N, graphCtxIndex, graph, graphBuf, leaves, finalHidden }` on the
  instance.
- (private field) `private graphCache: { N, graph, graphBuf, leaves,
  finalHidden } | null = null;` — single-entry cache; replace on N
  change rather than keeping a multi-entry map (encoder workloads in
  scope here all reuse a single padded-N).
- `embed()` body shrinks to: `ensureGraphCache(N)` → upload leaves
  via `backendTensorSet3` → `graphCompute(graph)` → readback +
  pool/normalize. No `ctxCreate` / `graphNew` / `backendAllocCtxTensors`
  / `backendBufferFree` / `ctxFree` per call.
- `dispose()` extended to also free `graphCache.graphBuf` and pop the
  graph ctx (one extra `ctxFree`) when the cache is non-null, before
  freeing the weight buffer + popping the weight ctx. Order matters:
  pop top of stack first.

**Per-call invariant after L1 lands:**

- `ctxCreate` is called exactly **twice** per `loadWeights` lifecycle:
  once for the weight ctx (in `loadWeights`), once for the graph ctx
  (lazily, on first `embed()` or first `embed()` after an N change).
  It is called **zero** times in the steady-state `embed()` body.
- `backendAllocCtxTensors` is called exactly twice per lifecycle for
  the same reason; `backendBufferFree` is called twice (once per
  buffer) at `dispose()` (or once for the graph buf when N changes
  mid-lifecycle).
- The steady-state `embed()` body is: `backendTensorSet3` (3 leaves)
  → `await graphCompute(graph)` → `downloadFromTensor(finalHidden)`
  → `poolAndNormalize`. No allocation, no graph rebuild.
- The bridge ctx stack depth is exactly 2 in steady state (weight at
  index 0, graph at index 1).

**Open questions for the implementer:**

- The graph ctx `memSize` is currently sized as
  `hp.layerCount * 32768 + N * hp.embeddingLength * 24` — confirm this
  budget is still correct when `buildGraph` runs only once per N
  (not per call). It should be: nothing about the metadata footprint
  scales with call count, only with N + layer count. Verify by
  running with a large N (e.g. 256) and watching for `ggml_init`
  failures in the bridge's stderr after several embeds.
- `wasm.uploadRangeChunked` vs `backendTensorSet3` interaction with a
  long-lived ctx: the weight upload path in `loadWeights` already
  works against a ctx beneath a (transient) graph ctx in the current
  code, so this should be a no-op concern, but confirm in the
  N-change recreate path that re-pushing the graph ctx still leaves
  weight tensor pointers valid (they should — they live in the lower
  ctx).
- Whether the WebGPU backend's profile counters
  (`webgpu_last_graph_profile_*`) reset cleanly across repeat
  `graph_compute` calls on the same graph. If not, the §D5 perf
  measurements in `eval/embed-perf.ts` may need a one-time warmup
  call before sampling.
- Single-entry vs LRU-by-N cache: spec assumes single-entry (replace
  on N mismatch). If real workloads pad to one of a small set of Ns
  (e.g. 32/64/128/256), upgrade to a small `Map<N, CacheEntry>` —
  but defer until profiling shows N-thrash.

**Test strategy:**

- `tests/encoder-inference.test.ts`: add a `GgmlWasm` mock (or extend
  the existing one) that counts calls to `ctxCreate`, `ctxFree`,
  `backendAllocCtxTensors`, `backendBufferFree`, `graphNew`, and
  `graphBuildForwardExpand`. Drive 5 successive `embed()` calls with
  the same N and assert each counter is **1** (post-`loadWeights`,
  pre-`dispose`).
- Same test, switch N mid-stream (e.g. 3 calls at N=64, 2 at N=128):
  assert `ctxCreate` increments by exactly 1 on the N-switch boundary
  and `backendBufferFree` is called exactly once for the old graph
  buf at that boundary. Total counter at end: `ctxCreate` = 3
  (weight + 2 graph ctxes), `ctxFree` = 0 until `dispose`,
  `backendAllocCtxTensors` = 3, `backendBufferFree` = 1.
- `dispose()` after a non-null cache: assert `ctxFree` is called
  twice (graph then weight) and `backendBufferFree` is called twice
  (graph buf then weight buf), in that order. Order is load-bearing
  because the bridge stack pops top-down.
- Numerical regression: run the existing
  `tests/encoder-inference.test.ts` (or whichever covers
  end-to-end embed correctness) before and after the change with the
  same fixture; embeddings must match bit-for-bit (or within a tight
  tolerance if FP non-determinism intrudes — should not, since the
  graph and inputs are identical).
- Browser smoke: re-run the embed smoke route after L1 lands and
  confirm no console errors from a stale-buffer-pointer crash on
  the second `embed()` call. This is the highest-risk failure mode
  if the long-lived graph buf interacts badly with WebGPU's resource
  lifetime.

## Phase 2 Result + Phase 2.5 Diagnostic + §D Closure (2026-04-27)

### Phase 2 result: L1 measured + reverted

L1 same-graph-cache implemented at commit `5eb1f73`, measured at `f0d89f1`,
reverted at `3a6a366`. Single-text p50 wall ms vs Phase 1 baseline:

| Cell                          | Baseline | L1   |  Δ%   |
|-------------------------------|---------:|-----:|------:|
| arctic-embed-s short          |    34.00 | 34.20|  +0.6%|
| arctic-embed-s long           |    25.70 | 26.30|  +2.3%|
| arctic-embed-m short          |    52.00 | 53.40|  +2.7%|
| arctic-embed-m long           |    41.90 | 37.90|  -9.5%|
| arctic-embed-s batchMixed t/s |    33.5  | 33.6 |   flat|
| arctic-embed-m batchMixed t/s |    21.3  | 22.5 |  +5.6%|

Three slight regressions, one improvement. The arctic-embed-m long -9.5%
reading is bimodal trial noise (~34 ms cluster + ~38-39 ms cluster, 50/50
split) — not a real lever effect. G1 strict reading: no model dropped ≥10%
AND zero of three regressions exceed 3%, but the single sub-threshold
improvement is within run-to-run variance. Per gate rule "lever that hits G3
but misses G1/G2 reverts (no shipping a no-op change)" → **revert**. Cosine
preserved at 0.76 throughout (G3 part 1 passed in browser smoke).

### Phase 2.5 diagnostic: where the 30-50 ms actually lives

Temporary instrumentation added to `EncoderInference.embed()` recorded
sub-call timings to `globalThis.__embedSubtraces` over 30 reps of
arctic-embed-s short. Means:

| Bucket           | mean ms | % of step |
|------------------|--------:|----------:|
| ctxAndBuild      |    0.27 |      0.8% |
| upload (leaves)  |    0.01 |      0.0% |
| **graphCompute** |  **32.5** | **95.6%** |
| download         |    1.04 |      3.1% |
| pool/normalize   |    0.02 |      0.1% |

`graphCompute` is overwhelmingly dominant. Per-call ctx + graph rebuild is
<1 ms total — confirms why L1 was a no-op. Download is ~1 ms — L2 (GPU pool
/ readback shrink) would buy at most ~3% of step time even if perfectly
executed. ctx/graph/buffer rebuild + leaf upload + pool combined are
under 5% of step time.

A 33M F16 model is ~66 MB of weights. On Apple Silicon at ~200 GB/s memory
bandwidth, one full pass should take <1 ms of actual compute. So the 30 ms
in `graphCompute` is **not memory-bound or compute-bound** — it's
**dispatch / kernel-launch overhead**. The encoder graph has ~390 tensors
(12 layers × ~33 ops/layer + entry/exit) → ~390 dispatches × ~80 µs/dispatch
≈ 31 ms. The arithmetic matches.

### Lever portfolio re-ranked against Phase 2.5 data

| Lever | Targeted bucket | Headroom (% of step) | Verdict |
|-------|-----------------|---------------------:|---------|
| L1 ctx/graph reuse | ctxAndBuild      |  <1% | measured + reverted |
| L2 GPU-side pool   | download         |  ~3% | not worth shipping for ~1 ms |
| L3 embedBatch (sequential loop) | per-call overhead | <1% | no-op on dispatch count |
| **L4 concat-graph batched compute** | **graphCompute via dispatch amortization** | **up to 90%** | only viable lever |

L4 (concat-graph batched compute) was explicitly listed as a non-goal in
this spec because it's a substantial structural change: requires either a
block-diagonal F16 attention mask (up to ~85 MB at K=64 for batchMixed —
too large; would need K≤8) or a 4D padded batch dim refactor of the entire
`buildGraph`. Either route adds correctness risk (per-text positions,
per-text segment IDs, per-text pooling at output) and is multi-hour
implementer work with measurable revert probability if mask construction
or batch-dim refactor goes wrong.

### Closure decision

Cycle closes per the spec's stop rule: "a lever's measured impact is in
the noise AND nothing else profiles as a hotspot → close early; document
what was tried." L1's null result + Phase 2.5's dispatch-overhead
characterization rules out L2/L3-sequential without measurement. L4 is
out of scope.

**What ships from this cycle (kept on `main` after merge):**

- `eval/embed-perf.ts` harness CLI + the `EmbedPerfTrace` /
  `waitForEmbedPerfResult` pieces in `eval/browser-smoke.ts`.
- `eval/fixtures/embed-prompts.ts` pinned text fixtures.
- `smoke-test/real-model-page.js` `?embedPerf=…&embedReps=…&embedFixture=…`
  URL-param hooks (causal-LM and encoder branches).
- `Makefile` `embed-perf` and `embed-perf-baseline` targets.
- `tests/encoder-cosine-parity.test.ts` G3 baseline guard.
- `eval/reports/embed-perf-baseline-cosine.json` cosine pin (0.76, ±0.005).
- `eval/reports/embed-perf-2026-04-27-baseline/` baseline measurements.
- `eval/reports/embed-perf-2026-04-27-L1/` L1 measurements (negative result).

**What's reverted:** `feat(encoder): L1 same-graph-cache across embed()
calls` (`5eb1f73` reverted by `3a6a366`).

**Future-cycle resurrection paths:**

1. **Concat-graph batched compute (deferred L4).** If a real use-case
   emerges for batch encoder throughput (e.g., a RAG ingestion pipeline
   in the SDK consumer), open a new cycle that targets dispatch
   amortization specifically. Two implementation options to evaluate
   then: (a) flat concat + block-diagonal mask at K≤8 (4-8× speedup
   ceiling, mask construction adds ~5 KB at K=8 short / ~5 MB at K=8
   long, manageable); (b) padded 4D batch (cleaner, requires full
   `buildGraph` rewrite to 4D — bigger blast radius). The harness in
   this cycle is ready to measure the result against G2.

2. **Larger encoder model registration (deferred wave-2 from
   non-goals).** If `bge-m3` or `gte-large-en-v1.5` ever land in the
   fleet, single-text p50 may flip from dispatch-bound to
   compute/bandwidth-bound — at which point L1 (and possibly L2) regain
   relevance. Re-measure then.

3. **Backend-side dispatch coalescing.** If `ggml-webgpu` ever grows a
   command-buffer-coalescing optimization upstream, that automatically
   addresses the bucket Phase 2.5 surfaced here without any §D-side
   changes. Worth re-running this cycle's harness on a future llama.cpp
   rebase to spot it for free.
