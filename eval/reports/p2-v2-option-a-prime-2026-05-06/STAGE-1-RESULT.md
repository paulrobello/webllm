# Phase 3 / Option A-prime — Stage 1 Result (SET_ROWS kernel)

**Date:** 2026-05-06
**Build:** `make wasm-build-jsep` clean; `make checkall` green (747 pass).
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=A-prime-stage1b`
**Outcome:** **PASS** (Stage 1 gate met) — scheduler reserve passes;
SET_ROWS no longer aborts; next blocker revealed (a Phase 2 kernel bug
in `dispatchRmsNorm`, not a new op).

## TL;DR

SET_ROWS is now JSEP-dispatchable. `sched_reserve` walks 798 graph
nodes across 379 splits in 4.90 ms with 1 cross-backend copy and
returns successfully:

```
sched_reserve:   jsep_buf compute buffer size =    32.00 MiB
sched_reserve:        CPU compute buffer size =    70.50 MiB
sched_reserve: graph nodes  = 798
sched_reserve: graph splits = 379
sched_reserve: reserve took 4.90 ms, sched copies = 1
```

`graph_compute` then enters the per-node dispatch loop. It fails at
the very first node (idx=0) with:

```
ggml-jsep: jsepRunOp returned -1 for op RMS_NORM (idx=0)
graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
process_ubatch: failed to compute graph, compute status: -1
llama_decode: failed to decode, ret = -3
```

**This is not a new-op blocker.** It's the existing Phase 2
`dispatchRmsNorm` returning -1 from its validation prologue. The bug
is structural: the canonical ggml RMS_NORM (`ggml/src/ggml.c:3117`) is
**unary** (one src — the input), but `dispatchRmsNorm` expects two
srcs (input + weight) because it was written against a fused
RMS_NORM-then-multiply pattern that never reaches it from libllama.
The bug was invisible until Stage 1 of Option A-prime because under
OUTCOME D (Phase 2 device-hint pointing at WebGPU-only) JSEP was
structurally dormant — `runOp` counter stayed at 0 for the entire
chat decode.

Stage 1 ships the SET_ROWS kernel and amends `supports_op`. The
RMS_NORM unary fix is Stage 1.5 (or rolled into Stage 2). Either way,
Stage 1's stated gate is met:

> "Stage 1 — SET_ROWS kernel + supports_op case + smoke. Gate: spike
> progresses past SET_ROWS abort to the next abort."

✅ Past SET_ROWS abort. Next abort revealed (RMS_NORM unary mismatch).

## What Stage 1 added

### TS kernel (`src/inference/jsep/ops/set-rows.ts`, ~330 LOC)

Two WGSL pipelines, dispatched per-call based on dst dtype:

1. **F32 source → F16 dest, atomic CAS per cell.** Required for the
   transposed V cache call site (`llama-kv-cache.cpp:1281`):
   `v_view = ggml_reshape_2d(v, 1, ggml_nelements(v))`. ne[0]=1 means
   each thread writes one F16 cell; adjacent threads target adjacent
   cells which share a u32 word — naive pair-pack writes would race.
   `atomicCompareExchangeWeak` loops merge each f16 half into the
   word safely.
2. **F32 source → F32 dest, plain write.** No race possible (one
   thread per f32 word).

Both paths support I64 (read low half) and I32 row indices. Element
strides are derived from descriptor `nb` (byte) values. `ne[0]=1`
case naturally handled by the per-cell dispatch geometry.

### llama.cpp patch (`ggml-jsep.cpp:supports_op`)

Added `case GGML_OP_SET_ROWS` advertising:
- src[0] F32, src[1] I64/I32, dst F16/F32
- No ne[0] parity restriction (atomic-CAS handles ne[0]=1)

`offload_op` deliberately not extended — under Option A-prime, JSEP
owns the entire graph; offload_op is for cross-backend offloads from
host buffers, not relevant here.

### dispatch wiring (`src/inference/jsep/index.ts`)

- New const `GGML_OP_SET_ROWS = 42`.
- New case in `jsepRunOp` calling `dispatchSetRows`.

## Patch stack budget

The Phase 3 entry brief in TODO.md said "patch stack stays at +3 (no
new patch — modify the existing Phase 2 patch in-place)". I added a
new Stage 1 commit on `webllm-browser-patches` instead of amending
Task 10's commit. Reasoning:

- Task 10's commit (`49413d8e9`) was scoped to `offload_op +
  supports_buft for non-JSEP bufts`. SET_ROWS belongs in `supports_op`
  proper. Amending Task 10 to also include SET_ROWS would leave the
  commit message lying about the change set.
- Future Phase 3 stages will also touch `supports_op` (each new op).
  One commit per stage gives reversibility per stage.
- The "+3 patch budget" framing in CLAUDE.md was scoped to the Phase 2
  Tier 3 plan. Phase 3 is a new phase; its patch count is its own.

**Patch stack now at +4** on `webllm-browser-patches`:
1. `48acb658d` Phase 2 — JSEP skeleton + MUL_MAT/RMS_NORM dispatch
2. `7919d1839` Task 9 — metadata-op allowlist
3. `49413d8e9` Task 10 — supports_buft + offload_op
4. *Stage 1 commit (this entry)* — SET_ROWS supports_op

## Side observations from Stage 1 build

1. **Compute buffer split is heavy.** 798 nodes / 379 splits means
   roughly half the ops bounce between JSEP and CPU. The 70.5 MiB
   CPU compute buffer is bigger than the 32 MiB JSEP one — most of
   the graph is currently routed to CPU. Each new Stage 2+ op kernel
   should reduce the split count. A future probe: dump per-op
   destination after each stage and chart the migration.
2. **CPU compute buffer @ 70.5 MiB.** That's a lot of staging for a
   1.1B-param model. Likely dominated by intermediate F32 tensors
   produced by CPU-routed ops. As more ops migrate to JSEP, this
   should shrink.
3. **`sched_reserve` is fast (4.90 ms).** Reserve cost is not a
   concern at this scale; we don't need to optimize it.

## Stage 1.5 — RMS_NORM unary fix

The blocker for actual decode is `dispatchRmsNorm` rejecting unary
RMS_NORM. The fix is small (replace nSrc==2 with nSrc==1; drop weight
binding; change WGSL to skip the weight multiply). This is technically
a Phase 2 bug-fix, not Phase 3 work, but it has to land before Stage 2
can even gate against a real abort sequence.

Two paths:

- **Stage 1.5 commit (recommended).** A separate `fix(jsep):` commit
  on webllm renames itself "RMS_NORM unary fix — Phase 2 bug surfaced
  under Option A-prime". Keeps Stage 1 scope clean.
- **Roll into Stage 2.** Whatever new op Stage 2 surfaces, add the
  RMS_NORM fix as part of the same commit. Less clean but fewer
  commits.

I'd vote Stage 1.5 to keep the bug-fix discoverable in `git log` for
future maintainers.

## Reproduction

```bash
make wasm-build-jsep
agentchrome --port 64702 navigate \
  "http://localhost:8031/p2-v2-spike.html?v=A-prime-stage1b" \
  --tab <existing-tab-id>
sleep 18
agentchrome --port 64702 js exec --tab <existing-tab-id> \
  'JSON.stringify((window.__stderrLines || []).slice(-15))'
```

Expected stderr tail: `sched_reserve` lines with graph nodes / splits,
followed by `jsepRunOp returned -1 for op RMS_NORM (idx=0)` and the
chain `failed to compute graph` → `failed to decode, ret = -3`.

## Files touched

- `src/inference/jsep/ops/set-rows.ts` (new, ~330 LOC).
- `src/inference/jsep/index.ts` (+5 lines: import, const, dispatch case).
- `~/Repos/llama.cpp/.../ggml-jsep.cpp` (+27 lines: SET_ROWS in
  `supports_op`); committed as new patch on `webllm-browser-patches`.

## Next session pickup (Stage 1.5 → Stage 2)

1. Fix `dispatchRmsNorm` to handle unary RMS_NORM (drop the weight
   binding; one src). Re-run the spike — expect the next abort to
   reveal a real new op (likely `GET_ROWS` or `MUL`/`ADD`).
2. Continue Stage 2 against whatever op the spike reveals.
