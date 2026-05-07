# Stage 4.4 — F1 dual-resident weight buffers in ggml-jsep

**Date:** 2026-05-06
**Status:** **PARTIAL OUTCOME A — Bug A FIXED (NaN cascade resolved, real
finite logits, distinct non-zero generated tokens). "Paris" decode NOT
achieved. Follow-on Bug C surfaced: GPU→host writeback gap for tensors
that are JSEP-produced and CPU-fallback-consumed.**
**Patch stack:** 7 (was 6) — single commit on
`webllm-browser-patches` against `ggml-jsep.cpp`.
**Spike URL:** `http://localhost:8031/p2-v2-spike.html?v=stage4.4-fix`
**Per-token decode:** **23.22 ms** (within noise of Stage-4.3 baseline
23.92 ms). F1 dual-write at `set_tensor` only impacts model-load wall
time (one extra memcpy per of 134 weight uploads); steady-state decode
unchanged.

## Goal (recap from Stage 4.4 brief in TODO.md)

Stage 4.3 localized Bug A to
`ggml_backend_jsep_buffer_get_base` returning the sentinel `0x2000`,
so ggml-backend treated jsep-resident `tensor->data` as a valid host
pointer. CPU-fallback ops (`MUL`, `ADD`, `ROPE`, `SOFT_MAX`, the per-
channel RMSNorm gain `out[r,c] = normed[r,c] * attn_norm.weight[c]`,
…) dereferenced `data` directly, reading uninitialized wasm-heap RAM
for weight tensors. Garbage weights × valid input → garbage activations
→ ±Inf accumulators in the next MUL_MAT → NaN dst → cascade through
every downstream op.

F1 (dual-resident weights) allocates a parallel host-side mirror of
every JSEP buffer and changes `get_base` to return the mirror's base.
`tensor->data = host_mirror + offset` becomes a real, dereferenceable
host pointer, so CPU-fallback ops read real data. JSEP dispatch is
unaffected: the descriptor still encodes `(buf_handle, within-buffer
offset)`; the offset is invariant under choice of base.

## Implementation

Single-file change in
`~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` (~80 LOC of
diff, no other files touched). Patch stack +1 (now 7). Bullet-by-
bullet against the brief:

| Brief item                                            | Lands as                                                                                |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `host_mirror` field in buffer ctx                     | `void * host_mirror;` added to `ggml_backend_jsep_buffer_context`                       |
| `aligned_alloc` in alloc_buffer                       | rounds size up to `GGML_JSEP_BUFFER_ALIGN`, `aligned_alloc` + zero-init                 |
| `free` in free_buffer                                 | `free(ctx->host_mirror)` paired with `ggml_jsep_free`                                   |
| dual-write in `set_tensor`                            | `memcpy` to mirror BEFORE `ggml_jsep_write` to GPU                                      |
| read from mirror in `get_tensor`                      | `memcpy` from mirror; **drops the JS round-trip via `Module.jsepRead`**                 |
| memset both in `memset_tensor`                        | `memset(host_mirror+off, …)` then `ggml_jsep_clear`                                     |
| memset both in `clear`                                | `memset(host_mirror, …, ctx->size)` then `ggml_jsep_clear`                              |
| `get_base` returns `host_mirror`                      | **load-bearing change** — `tensor->data` now points into real host RAM                  |
| `jsep_tensor_handle` subtracts `host_mirror`          | recovers the same per-tensor offset value as before (offset invariant under base choice) |

The `GGML_JSEP_PTR_BASE = 0x2000` macro is left in place but no longer
referenced anywhere in the file — kept for now as a documentation
relic of the original sentinel-based design. Safe to delete in a
follow-up cleanup; not load-bearing.

The `jsep_tensor_handle` helper had to move from inline-at-top to
forward-declared-then-defined-after-the-struct, because it now reads
`t->buffer->context->host_mirror`.

Build: `make wasm-build-jsep` (≈10 s incremental against the cached
build-jsep tree). `make checkall` green (fmt / lint / typecheck / 747
tests pass / 36 skip / 0 fail).

## Bug A — fixed (vs Stage 4.3 baseline)

Side-by-side at the canonical Stage-4.3b instrumentation points,
TinyLlama Q4_K_M, prompt unchanged:

| Marker                              | Stage 4.3 baseline                  | Stage 4.4 (F1)                                          |
|-------------------------------------|-------------------------------------|---------------------------------------------------------|
| `Q4K_SELFTEST.status`               | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `RMSNORM_SELFTEST.status`           | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `RMSNORM_MULTIROW_NODIVERT.status`  | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `RMSNORM_MULTIROW_DIVERT.status`    | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `MATMUL_PROD_NODIVERT.status`       | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `MATMUL_PROD_DIVERT.status`         | 0 (PASS)                            | 0 (PASS) — unchanged                                    |
| `GPU_ERR_COUNT`                     | 0                                   | 0 — unchanged                                           |
| `FIRST_NAN_DST_PROBE`               | first NaN at i=1 MUL_MAT (cascade)  | **`null`** — no NaN at any op                           |
| `LOGIT_STATS_STEP0.first8`          | `[0,0,0,0,0,0,0,0]`                 | **`[0.0060, 0.0047, -0.0102, 0.0138, -0.0149, 0.0099, -0.0029, -0.0056]`** |
| `LOGIT_STATS_STEP0.topVal / topId`  | 0 / 0                               | **0.159 / 593**                                         |
| `LOGIT_STATS_STEP0.minFinite/maxFinite` | 0 / 0                           | **-0.0649 / 0.1591**                                    |
| `GENERATED_TOKENS`                  | `[0, 0, 0, 0, 0]`                   | **`[593, 5871, 945, 16976, 25487]`** — distinct, non-zero |
| `POSTPREFILL_BUF11@0` (first8)      | canonical NaN                       | **`[-1.067, 0.656, -0.110, -0.110, 0.082, 0.082, -0.684, -0.493]`** — finite |
| `POSTPREFILL_BUF11@4194304` (first8)| canonical NaN                       | **`[0.0055, 0.0006, 0.0023, 0.0056, 0.0012, 0.0024, -0.0003, 0.0041]`** — finite |
| `PER_TOKEN_MS`                      | 23.92                               | 23.22 (within noise)                                    |
| `MODEL_LOAD_MS`                     | (not captured)                      | 312                                                     |
| `COUNTER_DELTAS.write` per prefill  | 1206                                | 1206 (unchanged)                                        |
| `COUNTER_DELTAS.read` per prefill   | 1266                                | **0** (F1 drops `jsepRead` round-trips entirely)        |
| `COUNTER_DELTAS.runOp` per prefill  | 1602                                | 1602 (unchanged)                                        |

The NaN cascade is gone. The post-prefill activation buffer carries
real finite f32 values at every offset that Stage 4.3 captured as
canonical-NaN. The first-token logit distribution is real, finite,
and contains a clear top-1 token (id 593, conf 0.159 — modest but
peaked over the 32k vocab). The model now produces five **distinct
non-zero token ids** instead of `[0,0,0,0,0]`.

**Bug A is fixed.** The CPU-fallback per-channel RMSNorm gain (Stage
4.3's smoking-gun op between seq 2 and seq 3) now reads real
attention-norm weights, and the chain of downstream ops behaves.

## Bug C — partial: GPU→host writeback gap

The decoded text is **`"ntiuracinateenes"` (tokens 593, 5871, 945,
16976, 25487), not `"Paris"`**. The model is producing real but
incorrect tokens. Two diagnostic signals point at the cause:

1. `FIRST_ALLZERO_DST_PROBE = {i:3, seq:10, op:42, dstH:18, dstO:0,
   divert:true, first8:[0,0,0,0,0,0,0,0]}` — at op index 3 in the
   prefill graph, a JSEP `SET_ROWS` (`op=42` in the current ggml
   enum) writes to destination buffer handle 18 at offset 0 and
   the post-runOp readback (which now reads from the host mirror)
   returns all zeros.
2. Several offsets in `POSTPREFILL_BUF11` (the activations buffer,
   handle 19) are still all-zero post-prefill: offsets 524288,
   1052672, 2101248, 6295552. These are interleaved with offsets
   that DO have real f32 data (0, 528384, 4194304, 17829888,
   35655680).

The pattern is exactly the **cross-backend writeback gap** the
Stage 4.4 brief flagged in its "critical verification before
declaring success" footnote, but in the GPU→host direction:

> *"If a CPU op writes its output directly via `tensor->data` (not
> via `set_tensor`), the host_mirror gets updated but the GPU
> buffer stays stale; the next JSEP op reading that tensor sees
> old data."*

The actual failure mode is the **inverse**: a JSEP op writes its
output to the GPU buffer (via the WGSL kernel) and the host mirror
stays stale (no writeback). When a downstream CPU-fallback op
dereferences `tensor->data` (which now points into the host mirror,
post-F1), it reads the initial-zero contents from the mirror's
`memset(0)` at alloc time, never updated by the GPU.

`COUNTER_DELTAS.read = 0` post-F1 confirms this: the scheduler is
not inserting `get_tensor` calls to bridge JSEP→host, because
ggml-backend's view is "tensor->data is a valid host pointer, the
CPU op can read it directly." That assumption was false pre-F1 (it
read the `0x2000` sentinel and produced garbage); F1 makes it
*partially* true (real host RAM exists) but doesn't satisfy the
*currency* contract (no GPU→host writeback after JSEP ops).

This is a Stage 4.5 problem, not a Stage 4.4 regression.

## Patch stack diff

The new patch (call it **P7 — F1 dual-resident host mirror in
ggml-jsep**) is the seventh on `webllm-browser-patches`. It is
self-contained in `ggml/src/ggml-jsep/ggml-jsep.cpp` and orthogonal
to every prior patch in the stack:

| Patch | File                                              | Stage     |
|-------|---------------------------------------------------|-----------|
| P1    | `ggml/include/ggml-jsep.h` (header, registration) | Stage 1   |
| P2    | `ggml/src/CMakeLists.txt` (jsep build target)     | Stage 1   |
| P3    | `ggml/src/ggml-backend-reg.cpp` (registry hook)   | Stage 1   |
| P4    | `ggml/src/ggml-jsep/ggml-jsep.cpp` (skeleton)     | Stages 1.5–2 |
| P5    | `ggml/src/ggml-backend.cpp` (`offload_op` to JSEP)| Stage 2   |
| P6    | `ggml/src/ggml-jsep/ggml-jsep.cpp` (RMS_NORM divert tempDst lifecycle) | Stage 3.5 |
| **P7**| `ggml/src/ggml-jsep/ggml-jsep.cpp` (**F1 — host_mirror**) | **Stage 4.4** |

## Patch stack rebase risk register update

P7 touches the buffer interface (alloc / free / set / get / memset /
clear / get_base) and the `jsep_tensor_handle` helper. Rebase risk
against upstream `ggml-backend-impl.h`:

- **`ggml_backend_buffer_i` shape change** (low) — adding
  `set_tensor_2d` / `get_tensor_2d` slots upstream wouldn't conflict
  (we already initialize them to NULL).
- **`ggml_backend_buffer_init` signature change** (very low) —
  hasn't moved in 12 months.
- **A new `ggml_backend_buffer_i` callback that exposes
  GPU-buffer-aware operations** (low but has option value) — would
  give us a clean place to plumb the GPU→host writeback in Stage 4.5
  rather than instrumenting graph_compute.

## Files in this commit

- `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp`:
  - Forward-declare + out-of-line definition of `jsep_tensor_handle`
    (was inline at top; needs ctx access now).
  - `ggml_backend_jsep_buffer_context` gains `void * host_mirror`.
  - `alloc_buffer` allocates + zero-inits the mirror; bails (and
    `ggml_jsep_free`s the GPU buffer) on alloc failure.
  - `free_buffer` frees the mirror.
  - `set_tensor` / `memset_tensor` / `clear` apply the operation to
    BOTH the host mirror AND the GPU buffer.
  - `get_tensor` reads from the host mirror only (no jsepRead).
  - `get_base` returns `ctx->host_mirror`.
- (no llama.cpp-side files outside `ggml-jsep.cpp` touched)
- (no webllm-side files touched — the spike's diagnostic
  instrumentation already covers F1's signature; reusing
  `STAGE-4.3-RESULT.md`'s capture infrastructure)

## Next session pickup — Stage 4.5

Branch on the Bug C diagnosis above:

- **Option H1: per-runOp writeback.** In
  `ggml_backend_jsep_graph_compute`, after each `Module.jsepRunOp`,
  unconditionally `Module.jsepRead` the dst tensor's GPU range into
  the host mirror at the dst offset. **Cost:** ~1206 extra
  jsepRead calls per prefill (TinyLlama, ~700 ms at the spike's
  per-read cost). **Risk:** kills decode throughput unless we limit
  to dst tensors actually consumed by CPU ops downstream. **Win:**
  dead-simple, restores correctness invariant (host mirror is
  always current after any op).
- **Option H2: implement `cpy_tensor` callback.** When the scheduler
  needs to copy from a JSEP tensor to a CPU-buft tensor, our
  `cpy_tensor` issues a jsepRead to fetch the latest GPU data and
  memcpy into the destination. **Cost:** only fires at scheduler-
  decided cross-backend boundaries, not after every op. **Risk:**
  the scheduler might not always insert the explicit cpy_tensor —
  the current `read=0` counter suggests it doesn't see a need to,
  because the host mirror already exists.
- **Option H3: graph-walk pre-pass to mark tensors that need
  writeback.** Walk the cgraph in graph_compute, find tensors
  produced by JSEP ops that are consumed by peeled (CPU) ops or
  by the ultimate output, and only writeback those after each
  runOp. **Cost:** moderate; one extra graph walk per compute call.
  **Win:** minimal overhead in steady state.

Recommended starting point: **H1 as a correctness probe** (does the
model emit "Paris" if we writeback unconditionally?); if yes,
optimize via H3. If not, the bug is somewhere else (KV cache wiring,
ROPE, attention masking) and we re-diagnose with the corrected
host-mirror invariant in place.

## Exit criteria — met

- [x] F1 implemented as designed (all bullets in the brief's
      "Files to touch" section landed).
- [x] WASM rebuilds clean.
- [x] `make checkall` green.
- [x] All four kernel selftests still PASS.
- [x] `GPU_ERR_COUNT = 0`.
- [x] `LOGIT_STATS_STEP0.first8` non-zero finite values (was
      `[0,0,0,0,0,0,0,0]`).
- [x] `GENERATED_TOKENS` non-zero distinct token ids (was
      `[0, 0, 0, 0, 0]`).
- [x] `POSTPREFILL_BUF11` carries real f32 at most offsets (was
      canonical NaN at every offset).
- [x] No NaN cascade (`FIRST_NAN_DST_PROBE = null`, was first NaN
      at i=1).
- [x] Per-token decode within noise of Stage-4.3 baseline.
- [ ] **Decoded text = "Paris"** — NOT met. Decoded text =
      `"ntiuracinateenes"` — diagnosed as Bug C (GPU→host writeback
      gap), tracked in Stage 4.5 brief.

The brief's Step 3 calls this an "Outcome A flips" → file Stage 5
case; we're filing it instead as "Outcome A partially flips" →
Stage 4.5 because the decoded-text quality criterion is what makes
this load-bearing for the project's use case (agent dialogue), and
Stage 5 (perf characterization + accuracy bench) would be premature
on incoherent output. The patch stack still grew by 1; the F1 design
is validated; the next bug is well-localized.
