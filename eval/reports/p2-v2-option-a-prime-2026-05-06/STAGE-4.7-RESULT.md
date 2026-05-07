# Stage 4.7 D2-tight — synchronous post-dispatch SET_ROWS dst readback

**Date:** 2026-05-07
**Status:** **D2-tight landed; Reading R1 confirmed.** The first SET_ROWS
dispatch (i=3, K-cache layer 0, dstO=0, divert) silently fails to write
its dst — `dstImmediateFirst8U16 = [0,0,0,0,0,0,0,0]` captured *before*
any later op runs. Every other captured SET_ROWS (i=4, 14, 15, 26, 27,
39, 40, 51, 52) writes correctly, with `dstImmediate == dstPost` byte-
exact across all 8 captured F16 cells. Stage 4.8 brief: localize the
dispatcher's first-call corner case in the SET_ROWS divert path.
**Patch stack:** 7 (unchanged — D2-tight is spike instrumentation only).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.7-fix`

## Goal

Stage 4.6 D2-lite established that the FIRST SET_ROWS dispatch (i=3,
K-cache layer 0) reads back `[0,0,0,0,0,0,0,0]` at end-of-decode while
every other captured SET_ROWS reads back non-zero F16 cells. Two
readings:

- **R1** — i=3's dispatch silently failed; dst was never written.
- **R2** — i=3 wrote correctly; a later op overwrote cells 0..7 with
  zeros between i=3 and end-of-decode (~110 SET_ROWS + 5 decode
  steps later).

D2-lite captured dst via a `Promise.resolve().then(...)` microtask, so
the readback runs only after `mod.jsepRunOp` returns; by then later
runOps have already queued GPU work, and the read reflects end-of-
decode state, not immediate post-dispatch state. D2-tight closes that
gap by reading SYNCHRONOUSLY inside the wrapper, before `mod.jsepRunOp`
returns. Since `graph_compute` is in `JSPI_EXPORTS`
(`src/wasm/CMakeLists.txt:158`), JSPI awaits the returned `Promise<number>`
on the wasm side, suspending the wasm caller while the readback resolves.
No later op can run between i=3's dispatch and the readback — so a zero
read here is unambiguously "i=3 itself did not write".

## Implementation

`smoke-test/p2-v2-spike.src.ts` (~50 LOC of additions on top of
D2-lite):

1. New field `dstImmediateFirst8U16: number[] | null` on
   `SetRowsDiagEntry` (plus optional `errImmediate` and `immediateMs`
   for diagnostics).
2. `mod.jsepRunOp` rewritten as `async`. The `Promise<number>` return
   is cast to `any` to satisfy the synchronous TS interface; runtime
   behaviour is what matters — JSPI awaits the promise on the wasm
   side.
3. After `runOpOrig(...)` returns, when `setRowsDiagEntry` is set
   (i.e. this is one of the first 10 SET_ROWS dispatches) we:
   - call `runtime.encoderBatcher.flush()` to force the dispatch to
     submit;
   - allocate a 16-byte staging buffer;
   - `copyBufferToBuffer(dstRec.buffer, diag.dstO, staging, 0, 16)`;
   - `device.queue.submit([enc.finish()])`;
   - `await staging.mapAsync(GPUMapMode.READ, 0, 16)`;
   - copy out 8 `Uint16` cells, populate `diag.dstImmediateFirst8U16`,
     destroy the staging buffer.

Verification:
- `make typecheck` green (cast through `any` keeps the JsepModule
  interface honest while allowing the runtime async swap).
- Selftests still 6/6 PASS (Q4K, RMSNORM single, RMSNORM multi,
  MATMUL no-divert/divert, SETROWS V-cache no-divert/divert).
- Spike chat path unchanged: `GENERATED_TEXT = "ntiuhuihnerquant"`,
  PER_TOKEN_MS = 25.24 (vs 25.16 baseline — +0.3% within noise),
  COUNTER_DELTAS unchanged
  (`{alloc:0, free:0, write:1206, read:1602, clear:0, runOp:1602, sync:3671}`).
- `immediateMs` per dispatch ranges 78.5–96.8 ms (warmup-dominated;
  the readbacks land during prefill, not decode, so per-token decode
  cost is unaffected).

## Key data — all 10 captured SET_ROWS dispatches

| i  | seq | dstH | dstO    | divert | src1Idx (first 8)          | dstImmediateFirst8U16                          | dstPostFirst8U16 (end-of-decode)               |
|----|-----|------|---------|--------|----------------------------|------------------------------------------------|------------------------------------------------|
| 3  | 13  | 23   | 0       | true   | 0,1,2,3,4,5,0,0            | **0,0,0,0,0,0,0,0 ← anomaly**                  | 0,0,0,0,0,0,0,0                                 |
| 4  | 15  | 23   | 262144  | true   | 0,512,1024,1536,...        | 32820,33200,734,362,33433,210,0,0              | 32820,33200,734,362,33433,210,0,0               |
| 14 | 44  | 23   | 524288  | true   | 0,1,2,3,4,5,0,0            | 32820,32794,67,78,32952,162,32863,57           | 32820,32794,67,78,32952,162,32863,57            |
| 15 | 46  | 23   | 786432  | true   | 0,512,1024,1536,...        | 1206,36837,3289,1855,608,2495,0,0              | 1206,36837,3289,1855,608,2495,0,0               |
| 26 | 76  | 23   | 1048576 | true   | 0,1,2,3,4,5,0,0            | 1206,33112,33335,503,32795,33250,32864,206     | 1206,33112,33335,503,32795,33250,32864,206      |
| 27 | 78  | 23   | 1310720 | true   | 0,512,1024,1536,...        | 33168,35526,60,4291,36740,4202,0,0             | 33168,35526,60,4291,36740,4202,0,0              |
| 39 | 111 | 23   | 1572864 | true   | 0,1,2,3,4,5,0,0            | 1206,33112,33335,503,32795,33250,32864,206     | 1206,33112,33335,503,32795,33250,32864,206      |
| 40 | 113 | 23   | 1835008 | true   | 0,512,1024,1536,...        | 33284,5286,39662,37190,6570,1642,0,0           | 33284,5286,39662,37190,6570,1642,0,0            |
| 51 | 145 | 23   | 2097152 | true   | 0,1,2,3,4,5,0,0            | 32820,32794,67,78,32952,162,32863,57           | 32820,32794,67,78,32952,162,32863,57            |
| 52 | 147 | 23   | 2359296 | true   | 0,512,1024,1536,...        | 2372,38037,36850,3934,35498,39516,0,0          | 2372,38037,36850,3934,35498,39516,0,0           |

Two diagnostic conclusions:

1. **R1 confirmed for i=3.** The synchronous post-dispatch readback
   is all zeros — there is no opportunity for a later op to have
   overwritten anything because no later op has yet run. The
   dispatch itself silently failed to land the K-cache layer 0
   position-0..5 write at dstO=0.

2. **R2 ruled out — at least for i=3.** If R2 had been correct,
   `dstImmediate` would have shown the expected F16 values (the
   `f16(src0[0..7])` cells). It does not; it shows zeros byte-
   exact with the deferred read. So the data was never there to
   begin with.

3. **Every non-i=3 SET_ROWS is bit-exactly stable** between immediate
   and end-of-decode reads. So no later op overwrites positions
   0..7 of dstH=23 at any of those eight `dstO` values across the
   entire prefill+decode window. (This also weakens the candidate
   "rogue overwriter" framings for the broader bug surface — at
   least within the dst-cell-window we sampled.)

## Why i=3 specifically?

i=3 is the very first SET_ROWS dispatch in the prefill graph (seq=13
in the unified runLog ordering). The shape pattern alternates
K-cache layer L (`[256, 512]`, sequential indices) and V-cache layer L
(`[1, 131072]`, strided indices). All ten dispatches take the divert
path. Every layer-L≥0 K-cache + every layer-L≥0 V-cache succeeds.
**Only i=3 — the FIRST divert SET_ROWS in the entire program — silently
fails.**

That points at first-call state in the dispatcher / encoder /
data-manager / pipeline cache. Suspect surfaces (in roughly
suspect-likelihood order):

- **Encoder-batcher first flush.** D2-tight's wrapper itself calls
  `runtime.encoderBatcher.flush()` post-dispatch — if `runOpOrig`
  internally also expects the dispatch to be flushed during a later
  op's flush (rather than during this op's own work), the very first
  flush could behave differently than every subsequent one (e.g.
  flushing an empty queue, or a queue that hasn't been initialised
  yet).
- **Pipeline cache miss for SET_ROWS divert variant.** First call has
  to compile the WGSL pipeline. If the divert path issues the
  dispatch BEFORE the pipeline is ready (or against a not-yet-
  finalised bind-group layout), the dispatch becomes a no-op.
- **Bind-group / temp-dst first allocation lifecycle.** `divert=true`
  means the dispatcher allocates a temporary destination buffer,
  pre-copies dst contents in, runs the kernel, and post-copies the
  result back. The very first temp-dst allocation may race with
  pipeline compile or with the staging-buffer cache.
- **Pre-copy of zero-initialised dst.** If pre-copy is executed but
  the actual kernel dispatch or the post-copy is skipped on first
  call, the result is exactly what we observe: dst stays at its
  pre-copy value (zero, because dstH=23's KV-cache buffer is fresh-
  allocated).

## Stage 4.8 brief — localize the first-call corner case

Goal: identify which step of the SET_ROWS divert dispatcher silently
no-ops on the very first call.

Approach (instrument-then-narrow):

1. Add `console.log` instrumentation inside `dispatchSetRows`'s divert
   branch (`src/inference/jsep/ops/set-rows.ts:418-494`) — log on
   first call only:
   - pre-copy buffer contents (read 8 F16 cells from temp-dst BEFORE
     pre-copy);
   - post-pre-copy (read same 8 cells from temp-dst AFTER pre-copy);
   - post-dispatch (read same 8 cells from temp-dst AFTER kernel
     dispatch — requires same JSPI-await trick we used here);
   - post-copy-back (read 8 cells from real-dst AFTER copy-back).
2. Branch on which read is zero:
   - **Pre-copy already zero, post-copy-back also zero**: pre-copy is
     correct (KV-cache is fresh-zero), kernel didn't write to temp-
     dst. Suspect pipeline cache miss or bind-group layout issue.
     Inspect first-call pipeline state. Probe: compile the SET_ROWS
     pipeline at module init (eager-compile) and re-run; if first
     call now succeeds, pipeline-compile-race confirmed.
   - **Pre-copy correct, post-copy-back stale**: copy-back is
     skipped on first call. Inspect whether the encoder commits
     between the kernel dispatch and the copy-back, or whether
     copy-back's source (temp-dst) is alive at the right point.
   - **Post-dispatch reads correct values, post-copy-back zero**:
     copy-back encodes wrong source/offset/size on first call
     specifically.
3. Optional cheap probe before any of the above: run a single warm-
   up SET_ROWS divert dispatch into a throwaway buffer at engine init
   (before model load) — if i=3 then succeeds in the production
   graph, the bug is purely a first-call ordering issue and the
   warm-up shipping is itself the fix.

Implementation hint: D2-tight already proves the JSPI-await pattern
works inside `mod.jsepRunOp`. Stage 4.8 can move the same per-step
read pattern INSIDE `dispatchSetRows`, but `dispatchSetRows` runs
under wasm — it'd need a JS-side instrumentation hook. Easier:
extend D2-tight's wrapper to also read temp-dst from
`runtime.dataManager.handles` (the divert path's temp-dst handle is
visible in `dataManager.handles` between alloc and free). Or rebuild
the wasm with debug `console.log` calls inside the divert branch.

## Code references

- `smoke-test/p2-v2-spike.src.ts:1480-1487` — `SetRowsDiagEntry` type
  with `dstImmediateFirst8U16` + `errImmediate` + `immediateMs`
- `smoke-test/p2-v2-spike.src.ts:1496-1502` — `mod.jsepRunOp` is now
  `async (…) => …`, cast to `any` at the closing brace
- `smoke-test/p2-v2-spike.src.ts:1769-1808` — synchronous post-
  dispatch readback inside the wrapper (after `runOpOrig`, before
  `return status`)
- `src/inference/jsep/ops/set-rows.ts:418-494` — divert path under
  investigation
- `src/inference/jsep/index.ts:140-156` — encoder batcher + data
  manager construction
- `src/wasm/CMakeLists.txt:151-164` — `JSPI_EXPORTS` (graph_compute
  listed)

## Patch stack

Unchanged at 7. D2-tight is spike instrumentation only — no wasm
or kernel changes.
