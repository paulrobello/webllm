# Phase 3 / Option A-prime — Stage 0 Pre-Flight Probe

**Date:** 2026-05-06
**Build:** `make wasm-build-jsep` (CMakeLists `WEBLLM_PIN_TO_JSEP=1`
default for jsep build); `make checkall` green.
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=A-prime-stage0`
**Outcome:** **PASS** — first scheduler abort is on `SET_ROWS` exactly
as predicted in the TODO Phase 3 brief. Op ordering for Stage 1+
confirmed.

## TL;DR

Inverting the device-hint to pin libllama to **JSEP only** (vs Phase 2
Task 11 which pinned to WebGPU only) puts weights + KV cache in
`jsep_buf` as designed. Model load completes cleanly:

- 23/23 layers offloaded to GPU
- `jsep_buf model buffer size = 455.06 MiB`
- `llama_kv_cache: jsep_buf KV buffer size = 11.00 MiB` (all 22 layers
  `dev = JSEP`)
- `CPU model buffer size = 181.11 MiB` (token_embd q4_K spill — see
  below)
- Output buffer size = 0.12 MiB (CPU; logits readback)

`createContext` aborts inside `sched_reserve` on the very first KV
write op:

```
/Users/probello/Repos/llama.cpp/ggml/src/ggml-backend.cpp:898:
  pre-allocated tensor (cache_k_l0 (view)) in a buffer (jsep_buf)
  that cannot run the operation (SET_ROWS)
Aborted()
```

JSEP's `supports_op` currently advertises only MUL_MAT + RMS_NORM
(post-Task-9 metadata allowlist). The scheduler can't dispatch
SET_ROWS to JSEP, but the tensor is pre-allocated in `jsep_buf`, so
the safety check at `ggml-backend.cpp:898` aborts the reservation.

## What this validates (vs OUTCOME D / OUTCOME E)

- **Inversion is correct.** Build + runtime path now picks JSEP as the
  single GPU device. No WebGPU buffers participate. KV cache lifetime
  is fully owned by JSEP. (Confirmed by stderr `dev = JSEP` lines 0
  through 21 + `jsep_buf model buffer size` + `jsep_buf KV buffer
  size`.)
- **Scheduler exposes the next-needed kernel naturally.** Without the
  device-hint inversion the scheduler never even hit SET_ROWS during
  decode (Outcome D had `runOp = 0` because `offload_op` requires
  host-buf src tensors and webgpu_buf is not host). Now SET_ROWS is
  the first abort because cache_k/cache_v writes are the first ops
  the scheduler attempts to schedule on the JSEP-resident KV tensors.
- **Op-ordering hypothesis from TODO confirmed.** The Phase 3 entry
  brief predicted SET_ROWS as Stage 1, with GET_ROWS / ROPE / SOFT_MAX
  / MUL / ADD as likely follow-ons. Stage 0 confirms SET_ROWS as the
  unblock point. Subsequent abort sequence will fall out of Stage 1's
  measurement.

## Side observations

1. **`token_embd.weight` (q4_K) spilled to CPU buffer (181.11 MiB).**
   Stderr: `done_getting_tensors: tensor 'token_embd.weight' (q4_K)
   (and 66 others) cannot be used with preferred buffer type
   CPU_REPACK, using CPU instead`. This is libllama's standard
   token_embd handling — the embedding matrix lives on CPU because
   GET_ROWS is inherently a host-staged gather even when output goes
   to GPU. No action item; this is expected and matches Outcome D
   layout.

2. **`load_all_data: device JSEP does not support async, host buffers
   or events`.** Cosmetic — JSEP is currently a synchronous GPU
   surface (the JS callbacks return synchronously to the C++ side).
   No action item for Stage 1; revisit if/when async upload becomes
   load-bearing.

3. **No JSEP `runOp` traffic before abort.** Counter snapshot at load
   time gets `{alloc, free, write, ...}` populated by KV/weight uploads
   but `runOp = 0` — the scheduler aborts during *reserve*, not during
   *compute*. This is fine for Stage 0; Stage 1 measures the first
   `runOp` increment after the SET_ROWS kernel lands.

## Decision: Stage 1 = SET_ROWS

Per the TODO Phase 3 plan (`TODO.md:891-895`), SET_ROWS is sized at
~150 LOC. Implementation pattern:

- New file `src/inference/jsep/ops/set-rows.ts` (mirror
  `matmul.ts` / `rms-norm.ts` descriptor read + pipeline-cache +
  dispatch).
- WGSL kernel: F32 + F16 → F16 row gather/write (KV cache is F16).
- Add `case GGML_OP_SET_ROWS` in `src/inference/jsep/index.ts`
  `jsepRunOp` dispatch.
- Add `case GGML_OP_SET_ROWS` (+ matching dtype check) in
  `ggml-jsep.cpp:supports_op` in
  `~/Repos/llama.cpp/webllm-browser-patches`. **Amend the existing
  Phase 2 patch in-place** — patch stack stays at +3 (no new patch).

Stage 1 gate: spike progresses past the SET_ROWS abort to the next
abort. JSEP `runOp` delta ≥ 1 over the prefill+decode window. Document
the next-revealed abort op so Stage 2 ordering is concrete.

## Reproduction

```bash
make wasm-build-jsep  # produces webllm-wasm-jsep.{js,wasm} with
                      # WEBLLM_PIN_TO_JSEP=1 baked in
make smoke-serve      # port 8031 (already running per session)
# In existing agentchrome session/tab:
agentchrome --port 64702 navigate \
  "http://localhost:8031/p2-v2-spike.html?v=A-prime-stage0" \
  --tab <existing-tab-id>
# Wait ~10-15s for fetch + load + abort
agentchrome --port 64702 js exec --tab <existing-tab-id> \
  'JSON.stringify((window.__stderrLines || []).slice(-80))'
```

Expected: stderr ends with the SET_ROWS abort at
`ggml-backend.cpp:898`; page log shows model loaded but spike
fails before [7/8].

## Patch budget

Stage 0 changed:
- `src/wasm/CMakeLists.txt` — add `WEBLLM_PIN_TO_JSEP=1` define on
  the JSEP build target (~9 lines including comment).
- `src/wasm/webgpu-bridge.cpp` — invert device-hint selection under
  the new flag; logging unchanged in shape, just labels (~20 lines
  swapped).

**Zero llama.cpp patches added.** Patch stack remains at +3
(`48acb658d` Phase 2 + `7919d1839` Task 9 metadata-op allowlist +
`49413d8e9` Task 10 supports_buft + offload_op).

## Links

- TODO entry: `TODO.md` § "Phase 3 entry: Option A-prime"
  (commit `eac3e7d`).
- Phase 2 closure (Outcome D + E): `eval/reports/p2-v2-prototype-2026-05-05/SUMMARY.md`.
- Spike harness: `smoke-test/p2-v2-spike.{html,src.ts}`.
- Bridge change: `src/wasm/webgpu-bridge.cpp` `webllm_load_model`
  device-hint block.
