# Phase 3 / Option A-prime — Stage 4.13 result

**Date:** 2026-05-07

**One-line outcome:** **PARTIAL CLOSE / Probe 3 reframed.** All three brief-predicted sub-cases (CPU-A "V's MUL_MAT skipped"; CPU-B "V's MUL_MAT runs but reads zero src1"; CPU-C "V's MUL_MAT routes to JSEP, allocator-coalesces dst onto h26+0") are **REJECTED**. Tensor-name capture (`tensor->name` in `set_tensor`) revealed Stage 4.10/4.11/4.12 had the K and V slot labels **swapped**: `(h26, 0)` is the **K** slot read by slice 3's first SET_ROWS, **not** V. `(h26, 528384)` is the V slot — and it works. The bug is **Kcur-0 (K projection layer 0)** landing as zeros at `(h26, 0)`, **not** Vcur-0. New diagnosis: **CPU-D (cross-backend buffer-aliasing bug):** Kcur-0's view `data` pointer aliases attn_norm-0's CPU scratch (addr 99827008). Between i=1 (attn_norm-0 set_tensor, valid normed×gain) and i=2 (Kcur-0 set_tensor, ZEROS), the first 6144 bytes of that shared scratch get zeroed by some intervening op. Slice 3's K SET_ROWS reads `(h26, 0)` after i=2 → K cache layer 0 = zeros → broken attention → garbage decode. V layer 0 (Vcur-0) lands valid at `(h26, 528384)` and slice 3's V SET_ROWS works correctly. **K projection on JSEP** (slice 1 / 2 MUL_MAT to h26+4194304) DOES compute, but its result never reaches `(h26, 0)` via a CPY-back; the only writer to `(h26, 0)` for layer 0 K is the broken Kcur-0 view set_tensor reading from the wrong (cleared) source. Patch stack: **9 → 10** (Probe 3 instrumentation patch; expected to revert at Stage 4.14 fix landing).

## Context

Stage 4.12 reported "V's projection result lands at `(h26, 0)` as a 6144-byte all-zeros set_tensor write while K's lands at `(h26, 528384)` as valid f32" and queued Probe 3 to disambiguate three CPU-side sub-cases (A/B/C) for the V producer. Stage 4.13 added per-MUL_MAT src1/dst capture in `ggml_backend_cpu_graph_compute`, per-handle `host_mirror` base publication in `alloc_buffer`, and `data_addr` + `tensor->name` capture in `set_tensor`. The tensor-name capture is what dissolved the Stage 4.10/4.11/4.12 K-vs-V slot ambiguity.

## Method

Three additive diagnostic captures (`ggml-cpu.cpp` + `ggml-jsep.cpp`, web-only via `__EMSCRIPTEN__`, no behavior changes):

1. **`ggml-cpu.cpp` `graph_compute` per-node MUL_MAT capture.** Inside the existing per-node loop, when `op == GGML_OP_MUL_MAT (29)`, push into `globalThis.__cpuMulMatDiag` (cap 60): `{callIdx, nodeIdx, src0_addr, src1_addr, dst_addr, dst_ne, src1_type, src1_first4}`. `src1_first4` reads 4 F32 from `src1->data` (or 4 U16 if F16). Lets us answer: does any CPU MUL_MAT have V's [256, 6] shape? does any have zero src1? where does the dst land?
2. **`ggml-jsep.cpp` `set_tensor` enrichment.** Add `data_addr` (CPU heap source pointer) and `name` (`UTF8ToString(tensor->name, 64)`) to existing `__setTensorLog` entry. The name is the load-bearing addition.
3. **`ggml-jsep.cpp` `alloc_buffer` host-mirror publication.** Push `globalThis.__jsepHostMirrorBases[handle] = host_mirror` on every buffer alloc. Lets JS compute `data_addr - host_mirror[h]` to detect when a CPU `data` pointer aliases a JSEP host_mirror slot.

Spike URL: `http://localhost:8031/p2-v2-spike.html?v=stage4.13-probe3b`. Decode reproduced bit-exactly (`GENERATED_TEXT="ntiuhuihnerquant"`, `topId=593/0.159`, per-token 127 ms — same as Stage 4.12 baseline; instrumentation invisible during decode steady state).

## Key findings

### Finding 1 — K and V slot labels were swapped in Stages 4.10/4.11/4.12

Tensor names from `__setTensorLog` (first 5 writes):

```
i=0  JSEP#embd#0                          off=0       sz=49152  f4=valid
i=1  JSEP#attn_norm-0#0                   off=0       sz=49152  f4=valid (normed×gain)
i=2  JSEP#Kcur-0 (view)#0                 off=0       sz=6144   f4=[0, 0, 0, 0]   ★ BROKEN
i=3  JSEP#leaf_8#0                        off=524288  sz=48     f4=ROPE freqs
i=4  JSEP#Vcur-0 (reshaped) (reshaped)#0  off=528384  sz=6144   f4=valid           ★ WORKS
```

`(h26, 0)` is the **K cur layer 0** slot, **not** V. `(h26, 528384)` is the **V cur layer 0** slot. Slice 3's two SET_ROWS source from these two offsets:

```
slice 3 node 0:  op=42 SET_ROWS  dst=h25+0       src0=h26+0       src1=h26+524288    ← K cache write
slice 3 node 4:  op=42 SET_ROWS  dst=h25+262144  src0=h26+528384  src1=h26+1052672   ← V cache write
```

Both `src0` are `op0=NONE` leaves — cross-backend boundary leaves the scheduler expects to be populated by external producers (CPU `set_tensor` or JSEP earlier slices) before slice 3 runs. K's leaf gets zeros. V's leaf gets valid f32. The bug is on the **K side**, not V.

### Finding 2 — All [256,6] CPU MUL_MATs are V projections (Vcur-N), not K

`__cpuMulMatDiag` has 10 entries with shape [256, 6] and 10 with [2048, 6] in the prefill window (cap 60). Matching each [256, 6] MUL_MAT's `dst_addr` to a `set_tensor` entry by `data_addr` ties them all to **`Vcur-N`** tensor names — Vcur-0 (callIdx=2, dst=108215616 → set_tensor `Vcur-0` at h26+528384), Vcur-1+ (callIdx=10..173, dst=93529408 → `Vcur-1+` at h26+0). **No CPU MUL_MAT produces a Kcur-N tensor.** K projections must run on JSEP — confirmed by the JSEP graph log showing 28 MUL_MATs (op=29) writing to handle 26 across 30 slices.

| CPU shape | Count | Identity |
|---|---|---|
| [256, 6]   | 10 | Vcur-{0..9} prefill |
| [2048, 6]  | 10 | output projection (kqv_out) for layers 0-9 prefill — src1 from attention's `softmax(Q×K^T)V` |
| [32000, 1] | 2  | lm_head (decode steps with broken hidden state) |
| [256, 1]   | 19 | Vcur-N decode steps |
| [2048, 1]  | 19 | output projection decode steps |

### Finding 3 — [2048, 6] / [2048, 1] CPU MUL_MATs read **zero or garbage** src1

Every captured [2048, *] CPU MUL_MAT has `src1_addr=127073344` (decode) / scratches that hold either `[0, 0, 0, 0]` or denormal-style garbage like `[183435.84, 3.19e-33, -1.6e+23, 1.4e+34]`. Identity: these are output projections (`kqv_out`) whose src1 is the attention output `softmax(Q×K^T)V`. With **K cache layer 0 = zeros** (Finding 1), `Q×K^T = 0`, softmax gives uniform attention, multiplied by V gives garbage/zero — exactly matching the observed src1 patterns. **Confirms K-cache-layer-0-zeros causes the cascade through every subsequent layer's attention output.**

### Finding 4 — Kcur-0's view `data` pointer aliases attn_norm-0's CPU scratch

i=2 (Kcur-0 set_tensor) `data_addr = 99827008` — the **same address** as i=1 (attn_norm-0). Kcur-0 is named `(view)`; in ggml a view shares the underlying tensor's `data` pointer. Kcur-0's underlying tensor's `data` is 99827008 = attn_norm-0's CPU scratch buffer.

But the K projection MUL_MAT runs on **JSEP**, not CPU, and its output goes to `(h26, 4194304)` (slice 1 or 2's MUL_MAT dst). Address `49464320 + 4194304 = 53658624`, which is **not** 99827008. So the JSEP K-projection output never reaches the address the Kcur-0 view points at.

When Kcur-0's set_tensor (i=2) fires, it reads from 99827008[0..6144) and copies to (h26, 0). At i=1, 99827008[0..16) was `[0.000005, 0.000012, ...]`. By i=2, it's `[0, 0, 0, 0]`. Some op in [i=1, i=2) zeroed the first 16+ bytes of that scratch. The K MUL_MAT result (which lives at host_mirror[26]+4194304 = 53658624) is **untouched** — it's just not what Kcur-0's view points at.

### Finding 5 — Comparable Vcur-0 path works because dst aliasing matches

Vcur-0's MUL_MAT runs on CPU (callIdx=2, dst_addr=108215616, src1=valid normed×gain). Its set_tensor (i=4) reads from 108215616 — the actual MUL_MAT dst, not someone else's scratch. So V's CPY captures real V data. The asymmetry between K and V is which backend computes the projection (JSEP vs CPU) combined with how the `Kcur-0 (view)` / `Vcur-0 (reshaped)` view tensors had their data pointers assigned by the scheduler/allocator.

## Diagnosis vs the brief's predicted sub-cases

| Sub-case | Predicted | Observed | Verdict |
|---|---|---|---|
| **CPU-A**: cpuMulMats empty AND no V on CPU | "Vproj never runs on CPU" | All 10 [256,6] CPU MUL_MATs ARE V (Vcur-N) | **REJECTED** — V runs on CPU correctly |
| **CPU-B**: any CPU MUL_MAT has src1=[0,0,0,0] | "Vproj runs with zero input" | Vproj src1 always valid; only kqv_out src1 is zero | **REJECTED** for V; **DOWNSTREAM ARTIFACT** for kqv_out |
| **CPU-C**: dst aliases host_mirror[26] | "Vproj allocator-coalesced onto h26+0" | Vproj dst lives in CPU scratch outside host_mirror[26]; the BROKEN tensor is **K** (not V); K's view aliases attn_norm-0's scratch (CPU), not host_mirror[26] | **REJECTED** as framed; partially rhyming with the underlying mechanism (view-aliasing into a stale scratch) |
| **CPU-D (NEW)**: Kcur-0 view `data` points at the wrong scratch (attn_norm-0's) instead of K projection's actual output | — | K projection on JSEP writes to h26+4194304; Kcur-0 view `data=99827008` (attn_norm-0's scratch); set_tensor reads zeros from cleared scratch and CPYs to (h26, 0); slice 3 reads zeros for K cache | **CONFIRMED** |

## Current state

- webllm tip: `e7f0bce` (Stage 4.12 TODO closure commit; no webllm changes for Stage 4.13).
- llama.cpp tip: `3b0e40d6f` on `webllm-browser-patches`, **patch stack 10** (Stage 4.13 Probe 3 instrumentation; additive, no behavior changes).
- Spike: `make wasm-build-jsep` green; `make checkall` green (747 pass / 36 skip / 0 fail).
- Per-token decode 127.40 ms (Stage 4.12 baseline 127.40 ms; instrumentation invisible).
- 6 spike selftests still PASS.

Diagnostic globals retained in spike + JSEP harness:
- `__jsepGraphLog` (Stage 4.10) — 30 slices × per-node metadata
- `__h1invDiag` (Stage 4.9) — 8 captures of host_mirror[26+0..32]
- `__interSliceLog` (Stage 4.11) — 60 enter/exit host_mirror[26+{0,528384}] snapshots
- `__cpuGraphLog` (Stage 4.12) — 30 CPU graph_compute calls × 9-i32-stride per node
- `__setTensorLog` (Stage 4.12 + 4.13 enrichment) — 200 entries gated on h26, now with `name` + `data_addr`
- `__cpuMulMatDiag` (Stage 4.13) — 60 CPU MUL_MAT entries with shape, src/dst pointers, src1 first 4 F32
- `__jsepHostMirrorBases` (Stage 4.13) — `{handle: host_mirror_addr}` map for every JSEP-allocated buffer

## Branch on outcome

Stage 4.14 brief queues the structural fix. Two candidate paths, ranked by load-bearing risk:

1. **Path R (recommended) — Make Kcur-0 view follow the actual K projection result.** Needs investigation of where Kcur-0's view `data` pointer is assigned (libllama graph builder or ggml allocator). The K projection on JSEP writes to `(h26, 4194304)`; Kcur-0 should view that, not attn_norm-0's CPU scratch. Likely requires either:
   - (R1) Routing Kcur-0's view through ggml's tensor view APIs in a way the JSEP scheduler honors (so the view inherits the JSEP-resident parent's `data`).
   - (R2) Forcing the K projection MUL_MAT to write into the slot Kcur-0 view points at (i.e., into `(h26, 0)` instead of `(h26, 4194304)`).
2. **Path U — Force K projection to CPU like V.** If the JSEP scheduler can't be made to write K projection's output where Kcur-0 view expects, route the K projection to CPU (parallel to V's path), so its result naturally lands in CPU scratch 99827008 where Kcur-0 view points. Requires a JSEP `supports_op` change to refuse K's MUL_MAT shape (likely brittle — may also affect Q's MUL_MAT, which has different dst routing).
3. **Path Q — Investigate the i=1→i=2 zeroing op.** Even with the right view routing, the underlying scratch at 99827008 gets zeroed between i=1 and i=2 by some unidentified op. If that op is the actual K projection (computing in-place into 99827008 with output zero because the JSEP K projection took precedence), Path R or U fixes the symptom by ensuring the zero output gets replaced by valid data. If it's a separate "init dst" or "memset before compute" op, the fix needs to either remove that op or ensure subsequent compute writes valid data.

The Stage 4.14 brief inlined in `TODO.md` queues a tighter localization probe (capture every CPU op that writes to addr 99827008 between i=1 and i=2, identified by `tensor->name`) before committing to Path R / U / Q, in the spirit of the workflow's "Probe-first is the default" doctrine.

## Files touched (Stage 4.13)

- `~/Repos/llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp` (+54 LOC) — Probe 3 CPU MUL_MAT capture
- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` (+22 LOC) — set_tensor name/data_addr enrichment + alloc_buffer host_mirror publication

No webllm changes (instrumentation lives entirely in llama.cpp).

## Patch stack delta

llama.cpp `webllm-browser-patches`: **9 → 10** (one additive Probe 3 instrumentation patch).
webllm: **0 commits**.
