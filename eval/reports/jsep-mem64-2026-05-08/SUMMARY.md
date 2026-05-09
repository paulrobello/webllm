# JSEP + MEM64 re-enablement probe — negative-result closure

Status: closed 2026-05-08 (negative result; original Phase 3 closure stance preserved)
Spec (superseded): [`../../docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md`](../../../docs/superpowers/specs/2026-05-08-jsep-mem64-build-target-design.md) (commit `f964915`)
Plan (superseded): [`../../docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md`](../../../docs/superpowers/plans/2026-05-08-jsep-mem64-build-target.md) (commit `e8201a6`)
Original Phase 3 closure: [`../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md`](../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md)

## TL;DR

Attempted to re-enable the canonical-6 7B+ JSEP-deferred subset
(mistral-7b-q4ks, llama-3.1-8b-iq3m, qwen3-8b-iq3m) by adding a
`wasm-build-jsep-mem64` build target. Two architectural blockers
surfaced — both at layers we deliberately built into the JSEP
infrastructure during Phases 0-3 (2026-05-05 → 2026-05-08). Neither
admits a tractable fix without unwinding load-bearing JSEP work.
**Phase 3 stays closed at the testable subset (3 of 6 canonical-6
models); the deferred subset remains under Stage 4.36 path 3 —
mathematical interpolation acceptance.**

This is a §31-style cap probe with a negative result. The probe is
the artifact: future sessions should not re-attempt JSEP+MEM64
without first solving one of the two blockers below.

## What was tried

Three-phase ship-grade plan (per the superseded spec/plan):

- **Phase A — build matrix.** Refactor `src/wasm/CMakeLists.txt` to
  thread `WEBLLM_BACKEND=jsep` × `WEBLLM_BUILD_MEM64=ON`; add
  `make wasm-build-jsep-mem64` recipe; verify linkage via a smoke
  probe HTML.
- **Phase B — refs capture.** Extend `p2-v2-ref-probe` with 3
  deferred entries + `requiresMem64` per-model flag; capture refs
  into `canonical6-refs.json` against the existing
  `webllm-wasm-mem64.js` (non-JSEP) reference.
- **Phase C — parity sweep.** Mirror the harness extension in
  `p2-v2-spike`; run JSEP+mem64 spike on the 3 deferred models;
  closure report.

Phase A Task A1 (CMakeLists.txt refactor) shipped cleanly; the
non-JSEP `wasm-build-mem64` target rebuilt without regression. Phase
A Task A2 (Makefile target + first build of the new combination)
hit blocker 1.

## Blocker 1 — `static_assert` guarding JSEP against MEMORY64

`ggml/src/ggml-jsep/ggml-jsep.cpp:830-832` (our
`webllm-browser-patches` branch) carries:

```cpp
// JSEP target is wasm32-only — the (int)(intptr_t) casts below
// truncate pointers; a MEMORY64 build would silently lose the
// high 32 bits.
static_assert(sizeof(void *) == 4,
              "ggml-jsep graph_compute assumes wasm32 pointer width; "
              "MEMORY64 would truncate intptr_t in the EM_ASM_INT call");
```

The guard was authored deliberately during Phase 2 / Phase 3 when
the JSEP boundary was designed. The `EM_ASM_INT` body at
`ggml-jsep.cpp:835` passes `desc` and `node->op_params` as
`(int)(intptr_t)` — under wasm64 this truncates 64-bit pointers.

Fixing requires widening:
- 6+ `EM_ASM`/`EM_ASM_INT` sites (lines 337, 365, 424, 497, 623,
  706, 835, 880).
- The `ggml_jsep_run_op` C-side fallback signature (line 845).
- The `ggml_jsep_read` / `ggml_jsep_write` / `ggml_jsep_clear`
  C↔JS bridges (`int32_t hostPtr` args).
- `Module.jsepRunOp` ABI in `src/inference/jsep/index.ts` and the
  per-op consumers in `src/inference/jsep/ops/*.ts` to pack/unpack
  the new pointer format (BigInt or paired-i32 halves).

Estimated ~100-300 LOC across the JSEP boundary that closed today
(2026-05-08, Stage 4.36) after 26 sub-stages of stabilization.
Risk: regressing the testable-subset parity that is the canonical-6
JSEP closure ceiling.

## Blocker 2 — `host_mirror` duplicates GGUF inside the wasm heap

Even if Blocker 1 were fixed (lifting the binary cap to 16 GiB
under wasm64), the JSEP architecture itself duplicates every weight
inside the wasm heap via `ggml_backend_jsep_buffer_context::host_mirror`
(`ggml-jsep.cpp:251`):

```cpp
struct ggml_backend_jsep_buffer_context {
    ...
    void *  host_mirror;  // F1 (Stage 4.4) — host-side parallel copy so CPU-
                          // fallback ops can dereference tensor->data
    ...
};
```

`get_base` returns `host_mirror` (line 285); `tensor->data` lands
inside the mirror; CPU-fallback consumers in mixed graphs depend on
that contract. `aligned_alloc` + `free` pair the mirror with the
buffer lifetime (lines 271, 285). The mirror was added at Stage 4.4
specifically to fix correctness on graphs with CPU-fallback ops
(e.g. GET_ROWS, MUL on cpu_buf inputs).

Concretely on JSEP wasm32 with even an idealized streaming-loader
that removes 2× residency at GGUF load time:

| Model | host_mirror (GGUF) | KV mirror (nCtx=512) | Scratch | Total | Margin in 4 GiB |
|-------|--------------------|----------------------|---------|-------|-----------------|
| mistral-7b-q4ks | 4.14 GiB | ~67 MiB | ~150 MiB | ~4.36 GiB | -0.36 GiB |
| llama-3.1-8b-iq3m | 3.78 GiB | ~67 MiB | ~150 MiB | ~4.00 GiB | -0.00 GiB |
| qwen3-8b-iq3m | 3.90 GiB | ~75 MiB | ~150 MiB | ~4.13 GiB | -0.13 GiB |

(KV mirror sizing: `nLayer × nHeadKV × headDim × 2 × 2 × nCtx`
bytes for F16; mistral/llama 32 × 8 × 128 × 2 × 2 × 512 ≈ 67 MiB;
qwen3-8B 36 × 8 × 128 × 2 × 2 × 512 ≈ 75 MiB.)

**Streaming-loader covers 0 of 3 deferred models on JSEP wasm32**
because the 4 GiB cap is binding regardless of how the GGUF is
loaded. The mirror duplicates the full GGUF bytes inside the wasm
heap; only lifting to wasm64 (Blocker 1) helps.

(Note: the original Stage 4.36 deferred-subset writeup catalogued
streaming-loader as a separate re-enablement path — that catalogue
predates the present analysis and is now known to be insufficient
on the JSEP path. The `webllm-wasm-mem64.js` non-JSEP path is
unaffected; it has no `host_mirror` because the legacy graph-builder
path keeps weights GPU-only.)

## Why both blockers can't easily go away

Removing `host_mirror` would unwind Stage 4.4 (post-load mirror dual-
write) plus the H1 GPU→host writeback at `ggml_jsep_read`
(`ggml-jsep.cpp:866` in the post-Stage-4.16 `EM_ASYNC_JS` form).
That's the same problem the original Phase 2 `offload_op` path
attempted to solve and abandoned in favor of Option A-prime / full
JSEP residency — see
[`../p2-v2-prototype-2026-05-05/SUMMARY.md`](../p2-v2-prototype-2026-05-05/SUMMARY.md)
"Disposition (post-Task-12)".

Lifting Blocker 1 is necessary but not sufficient for the canonical-6
deferred subset under JSEP — `host_mirror` would still need to fit
inside whatever heap is available, but at 16 GiB cap with `host_mirror`
at 3-4 GiB plus KV/scratch, that path is structurally fine. So
Blocker 1's resolution actually unblocks the deferred subset
end-to-end. The reason we're not pulling it forward today: ~100-300
LOC across a freshly-stabilized boundary, no consumer ask for 7B+
JSEP, and the project's "measured gain over speculative" doctrine
requires evidence-driven motivation. Path 3 (acceptance via
mathematical interpolation, the original Stage 4.36 stance) is
still the right closure for now.

## What stays in place

- The Stage 4.36 testable-subset closure
  ([`../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md`](../p2-v2-option-a-prime-2026-05-06/STAGE-4.36-RESULT.md))
  is unchanged. 3-of-6 canonical-6 models pass the JSEP parity
  gate; the remaining 3 are deferred under path 3.
- `canonical6-refs.json` is unchanged (3 entries — the testable
  subset).
- Patch stack: 0 delta (no llama.cpp patches added). The static_assert
  at `ggml-jsep.cpp:830` is itself an existing project artifact.
- Build matrix: `make wasm-build`, `wasm-build-wasm32`,
  `wasm-build-mem64`, `wasm-build-jsep` are all unchanged; no new
  target. (Tasks A1 + A2 staged changes were reverted.)

## Re-evaluation triggers

Re-attempt JSEP+MEM64 only if **all three** of the following land:
1. A consumer ask specifically for 7B+ JSEP (the agent + Three.js
   primary use case is satisfied today by the testable subset's
   ≤1.7B ceiling and by the non-JSEP `webllm-wasm-mem64.js` path
   for >1.7B).
2. Capacity to absorb a JSEP-boundary refactor without a fresh
   spike-stabilization cycle.
3. A pre-rebase same-tip parity baseline of the testable subset
   (TinyLlama / Qwen3-0.6B / Qwen3-1.7B) captured *before* the
   widening, to gate regressions.

## What this probe banks for the project

Despite closing as negative, the probe produces three load-bearing
artifacts:
- **The blocker catalogue above** — the next session that considers
  this lever sees both walls upfront, costed.
- **A reproducer for the static_assert** — `make wasm-build-jsep-mem64`
  with the Task A2 Makefile recipe (currently reverted but
  trivially re-derivable from this report) hits the assert in
  ~30 seconds at the `ggml-jsep` compile step.
- **A precise numerical bound on the host_mirror constraint** — the
  table above documents the exact wasm32 cap arithmetic for
  future cap-probe cycles.

Per the project's TODO archival cadence: this report is the closure
artifact. The spec/plan stay in `docs/superpowers/{specs,plans}/`
with SUPERSEDED banners pointing here.
