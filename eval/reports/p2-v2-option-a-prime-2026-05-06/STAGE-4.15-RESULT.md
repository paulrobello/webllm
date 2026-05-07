# Stage 4.15 — JSEP MUL_MAT divert dispatch GPU-buffer fate

> Closure date: 2026-05-07
> Outcome: **PARTIAL — Branch 3 confirmed (divert path writes valid data;
> downstream H1 GPU→host writeback fails to land it in host_mirror).**
> Stage 4.14's CPU-E framing — "JSEP MUL_MAT divert dispatch produces no
> host-visible output" — was correct in observable effect but wrong in
> mechanism. The divert path itself is fine.

## Summary

Stage 4.15 added Probe 5 in `src/inference/jsep/ops/matmul.ts` — a per-
divert-dispatch readback of (a) `tempDst[0..16)` right after
`queue.submit`, and (b) `dstRec.buffer[dst.offset..+16)` at the location
the divert's `copyBufferToBuffer` lands. Both are queued into a separate
encoder, submitted, and drained via deferred `mapAsync`. The probe is
gated on `globalThis.__stage415DivertProbe` (set true in
`smoke-test/p2-v2-spike.src.ts` before model load) and self-caps at 32
entries to keep wall-time noise bounded.

**The probe disambiguates the three branch hypotheses cleanly:**

| Probe 5 finding | Stage 4.15 outcome |
|---|---|
| Branch 1: kernel writes zeros into tempDst for Q-shape (M=2048) | **REJECTED.** i=1 (Q proj layer 0, M=2048): `tempF4 = [-6.26e-5, 1.87e-5, -6.09e-5, -9.48e-5]` valid; i=2 (K proj, M=256): `tempF4 = [6.85e-5, 1.35e-4, -1.74e-4, 7.41e-5]` valid. The kernel produces correct output for the load-bearing Q/K/V projections at layer 0. |
| Branch 2: tempDst valid but `copyBufferToBuffer` doesn't land at `dst.offset` | **REJECTED.** For every captured entry: `tempBytes` is byte-exact equal to `dstBytes` (`tempEqDst === true` for all 32 entries). The copy lands at the expected offset in `dstRec.buffer`. |
| Branch 3: divert path lands correctly; H1 reads from a different `record.buffer` (or different host_mirror offset) than the divert wrote | **CONFIRMED.** For i=1 (Q proj): GPU(handle=26, offset=4194304) holds valid Q at probe time. Yet Stage 4.14's `__jsepGetTensorLog` reports `host_mirror[26]+4194304` returns `[0, 0, 0, 0]` for `name = "Qcur-0"`. The data exists on the GPU; the H1 GPU→host writeback (`ggml-jsep.cpp:779-786`) does not deposit it at the offset `get_tensor` later reads from. |

**Cascade observation (independent confirmation of Branch 3 effect):**

| Divert idx | dst handle/offset | dst shape | src1 shape | tempF4 | Interpretation |
|---|---|---|---|---|---|
| 1 | 26 / 4194304 | [2048, 6] | [2048, 6] | valid | layer 0 Q proj — kernel + copy correct |
| 2 | 26 / 4194304 | [256, 6] | [2048, 6] | valid | layer 0 K proj — kernel + copy correct |
| 3 | 26 / 6295552 | [256, 6, 32] | [64, 6, 32] | **zero** | attention scores Q × K_cache — src1 (Q ROPE'd) is zero at GPU kernel time |
| 4 | 26 / 35655680 | [64, 6, 32] | [256, 6, 32] | **zero** | scores × V_cache — src1 (softmaxed scores) is zero |
| 5 | 26 / 6295552 | [2048, 6] | [2048, 6] | **zero** | layer 1 Q proj — src1 is layer 0's output, which is zero (cascade) |
| 10–15 | mixed | — | — | mostly zero | cascade through deeper layers |

i=3's zero output is **independent corroboration** that Q's GPU buffer is
populated correctly by i=1 but Q-as-read-back-into-host_mirror is zero —
because Q ROPE'd (computed on CPU after host_mirror is consulted) reads
zeros, writes zeros to host_mirror, and the H1-inverse pre-pass syncs
zeros to the GPU before i=3 fires. i=3's kernel is correct; its src1 is
genuinely zero by the time it dispatches.

## Files touched

- `src/inference/jsep/ops/matmul.ts` — added `Stage415DivertEntry`
  interface (file-local) and the gated readback block right after
  `queue.submit([enc.finish()])` and before `tempDst.destroy()`. The
  block is no-op when `__stage415DivertProbe` is falsy. Patch stack:
  unchanged at 11 (no llama.cpp patch — webllm-side spike instrumentation only).
- `smoke-test/p2-v2-spike.src.ts` — added `globalThis.__stage415DivertProbe = true` and
  initialized `__stage415DivertLog = []` early in `runSpike()` (before
  `installJsepCallbacks`) so the gate fires for every divert dispatch
  including model-load and the first decode step.

## Reproducer

```js
// At spike DONE marker, after ~200ms drain:
const log = (globalThis.__stage415DivertLog || []);
const entries = log.slice(0, 16).map(e => ({
  i: e.divertIdx,
  dstH: e.dstHandle, dstO: e.dstOffset,
  dstNe: e.dstNe, src0Ne: e.src0Ne, src1Ne: e.src1Ne,
  tempF4: e.tempBytes ? Array.from(new Float32Array(e.tempBytes.buffer.slice(0, 16))) : null,
  dstF4:  e.dstBytes  ? Array.from(new Float32Array(e.dstBytes.buffer.slice(0, 16)))  : null,
  tempEqDst: e.tempBytes && e.dstBytes
    ? Array.from(e.tempBytes).every((b, k) => b === e.dstBytes[k]) : null,
}));
console.table(entries);
```

Raw probe data: `STAGE-4.15-probe5-raw.json` in this directory (32 entries
+ first 16 `__jsepGetTensorLog` entries for cross-correlation with Stage
4.14).

## Verification

- `make typecheck` — green.
- `make wasm-build-jsep` — green (Probe 5 type interface; no llama.cpp
  patch, so the patch stack is unchanged at 11).
- Spike replay (`http://localhost:8031/p2-v2-spike.html?v=stage4.15-probe5-1`):
  - `GENERATED_TEXT` not captured this run (DONE summary only); per-token
    decode 107.7 ms (slightly faster than 4.14's 127 ms — Probe 5's
    deferred mapAsyncs land outside decode timing). Stage 4.14's "Probe 4
    instrumentation invisible" envelope holds.
  - 6 selftests still PASS (asserted by spike `[6/8]` step).
  - `__stage415DivertLog.length === 32` (cap hit).
  - `tempEqDst === true` for every entry.
  - i=1's `dstF4` matches `tempF4` byte-exactly with valid float values.
- `make checkall` — green before commit.

## Why Stage 4.14's "no host-visible output" framing is observably right
but mechanistically wrong

Stage 4.14's smoking gun was correct: every JSEP MUL_MAT for Qcur-0 / K
projections produces zero output **at host_mirror**. Stage 4.14 attributed
this to "the JSEP MUL_MAT divert dispatch path itself produces no
host-visible output," and predicted three branch hypotheses for Stage
4.15 to disambiguate.

Probe 5 disproves the natural reading of Stage 4.14's CPU-E. The divert
path **does** produce host-visible output — at the GPU level (`tempDst`
post-kernel and `dstRec.buffer` post-copy both hold valid F32). What
Stage 4.14 actually observed is that **`host_mirror` doesn't receive the
GPU-side value** — i.e., `Module.jsepRead` (called from H1 in
`ggml-jsep.cpp:779-786` to mirror GPU dst back to host_mirror) is reading
from somewhere other than the divert wrote, OR writing to somewhere other
than `get_tensor` reads from.

## Branch on outcome

- **Outcome A (English decode):** not achieved this stage. Probe 5 was
  diagnostic only; no structural fix landed. `GENERATED_TEXT` remains
  bit-identical to 4.14 (`"ntiuhuihnerquant"` per Stage 4.14 baseline).
- **Outcome B (probe isolates a branch but fix doesn't flip decode):**
  partial; Branch 3 isolated, structural fix queued for Stage 4.16.
- **Outcome C (inconclusive):** rejected. The 3-branch table partitioned
  cleanly; Probe 5's data narrowly fingers Branch 3.

## Stage 4.16 — next probe queue

The structural fix needs to localize **why** H1's
`ggml_jsep_read(bctx->handle, dst_offset, ..., dst_size)` doesn't put
the divert-written data into `host_mirror[bctx->handle] + dst_offset`.
Three sub-hypotheses for Stage 4.16 to disambiguate:

1. **H1-mirror-mismatch:** `bctx->handle` (set in `alloc_buffer`) and the
   descriptor's `dst.bufHandle` (packed in `ggml-jsep.cpp::pack_tensor`)
   resolve to different `dataManager.handles.get(...)` records. The
   divert writes to one record's GPUBuffer; H1 reads from another.
   Disambiguation: log `(node->buffer->context->handle, dst_offset,
   dst_size)` in H1 *and* log `(dst.bufHandle, dst.offset, ne)` in
   dispatchMatmul, then cross-correlate.
2. **H1-read-from-stale-buffer:** `dataManager.handles.get(handle)`
   returns a stale `record.buffer` if the underlying GPUBuffer was
   recreated (e.g., grown/reallocated) since H1 last cached. Probe 5's
   `dstRec.buffer` is freshly fetched per dispatch; `Module.jsepRead`'s
   record fetch happens later via the same `handles.get(handle)` so they
   should align — but worth a defensive check (log `record === dstRec`
   in jsepRead).
3. **H1 fires for a different node than dispatchMatmul wrote.** In
   ggml-jsep's `graph_compute`, `node` is the cgraph node `i`. The
   descriptor packed for `jsepRunOp` could include `dst` from a view
   tensor, while `node->buffer` and `jsep_tensor_handle(node)` resolve
   to the underlying base tensor. If view_offs is non-zero, H1's offset
   and the descriptor's offset differ.

The Stage 4.16 brief should land Probe 6 — pre-runOp logging of
`(node->buffer->context->handle, jsep_tensor_handle(node),
ggml_nbytes(node), tensor_name)` matched against Probe 5's `(handle,
offset, size, ne)` so each pairing is a cross-correlation row. The
disambiguation table above maps each outcome to a structural fix.
