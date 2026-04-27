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
