# Stage 4.8 — SET_ROWS divert first-call silent failure: localized

**Date:** 2026-05-07
**Status:** CLOSED — root cause identified. Stage 4.7's "Reading R1" (i=3
dispatch silently fails) was a misframing: the dispatch is fine. The
**`src0` GPU buffer is stale** at i=3 dispatch time. The kernel reads
zeros (not the K-projection-after-ROPE values) and faithfully writes
zeros to the temp-dst, which copy-back propagates to real-dst.

This is the **host→GPU writeback gap** flagged but not addressed by the
Stage 4.4 brief. Stage 4.5 H1 closed the **GPU→host** direction; the
**host→GPU** direction remains open. Stage 4.9 will close it.

Patch stack: 7 (unchanged — no llama.cpp patch this stage).

## TL;DR

- **Step A (eager-warmup probe)**: did NOT fix it. i=3 still all-zero.
  Confirmed the bug is not generic first-call cold-start.
- **Step B (temp-dst windowed readback inside `dispatchSetRows`)**:
  captured pre-kernel, post-kernel, and post-copy-back snapshots of
  the FIRST production divert dispatch (i=3, K-cache layer 0,
  `dstO=0`).
- **Smoking-gun finding**: at i=3 dispatch time, `src0` (handle 26
  offset 0 — the K-projection-after-ROPE buffer) reads as **mostly
  zeros with sparse `-0.0` (0x80000000 byte pattern)**. The kernel
  faithfully reads f32, packs to f16 via `pack2x16float`, and writes
  via atomic CAS. f16(0.0)=0x0000 leaves cells unchanged; f16(-0.0)=
  0x8000 writes `0x8000` to the targeted half-word. Result: post-
  kernel temp-dst is mostly zero with sparse `0x8000` values that
  match where -0.0 sits in the GPU buffer.
- **Why i=4..52 work**: every other captured SET_ROWS reads from a
  different src0 offset (h26o528384 alternates with h26o0). At i=14,
  src0=h26o528384 — by then a JSEP runOp has written there, and Stage
  4.5 H1 ran `jsepRead` on its dst, which means the GPU buffer at that
  offset was actually written by a JSEP op AND mirrored. (For h26o0,
  the only writer is a CPU op chain — see below.)
- **Root cause**: between i=2 (K-projection MUL_MAT, dst h26o4194304)
  and i=3 (K-cache SET_ROWS, src0=h26o0), there is a chain of
  CPU-fallback ops (likely `CPY` from h26o4194304→h26o0 plus possibly
  ROPE) that update **host_mirror** at h26o0 with the rotated K data
  but **never write to the GPU buffer**. Stage 4.5 H1 only adds
  GPU→host writeback, so the GPU buffer at h26o0 retains its initial
  state (zeros, with a stray `0x80000000` byte pattern at one offset
  per row from somewhere — likely the host_mirror initialization or a
  prior allocator reuse).
- **Fix mandate (Stage 4.9)**: add **host→GPU writeback** so JSEP ops
  that read tensors written by CPU-fallback ops see fresh data.
  Symmetric counterpart to Stage 4.5 H1.

## Bootstrap state captured (post-Stage-4.7 D2-tight, pre-fix)

```
$ git log --oneline -1
5e8a228 docs(TODO): Stage 4.7 closed — queue Stage 4.8 first-call divert localization
$ ( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
e0fa38928   webllm-browser-patches   (patch stack 7)
$ make checkall
... 747 pass / 36 skip / 0 fail
```

## Step A — eager-warmup probe (NOT the fix)

Added a throwaway SET_ROWS divert dispatch at the end of
`installJsepCallbacks` (before returning the runtime), allocating
scratch buffers, constructing a synthetic descriptor with
`dst.bufHandle === srcs[2].bufHandle` to force the divert path, and
calling `dispatchSetRows()` directly (bypassing `mod.jsepRunOp` so the
spike's wrapper doesn't see it).

Tried two warm-up shapes:

1. Tiny shape (`NE0=8, NR=1, DST_ROWS=4`, ~256-byte buffers).
2. Production-matching shape (`NE0=256, NR=6, DST_ROWS=512`,
   `dstSize=262144 bytes` — same as i=3).

**Both: i=3 still showed `dstImmediateFirst8U16 = [0,0,0,0,0,0,0,0]`,
generated tokens unchanged (`[593, 5871, 15669, 15565, 12150]` ⇒
`"ntiuhuihnerquant"`).**

Conclusion: it is not a generic first-call cold-start. The bug is
specific to i=3's exact arguments — and as Step B shows, specifically
to the contents of i=3's `src0` GPU region.

## Step B — temp-dst windowed readback (the diagnostic that cracked it)

Added a **non-invasive hook** inside `dispatchSetRows` (gated on a
`globalThis.__stage48DivertHook` flag set only by the spike, and
fires on the first divert call). The hook attaches three side-buffer
copies into the divert command encoder, between the existing copies/
dispatch:

1. **Pre-kernel snapshot**: after the pre-copy-real-into-temp,
   before the compute pass — `tempDst[row*ne0*2 .. +16)` for
   `min(nr, 8)` rows.
2. **Post-kernel snapshot**: after the compute pass, before the
   copy-back — `tempDst[row*ne0*2 .. +16)`.
3. **Post-copy-back snapshot**: after the copy-back submit —
   `dst[dst.offset + row*ne0*2 .. +16)`.

Plus an `src0` capture taken at the same encoder timeline (between
pre-copy and compute pass): 16 bytes (4 F32) per row, for the same
8 rows. This shows what the kernel actually reads — mapAsync after
submit is FIFO-serialized so the side buffers reflect the exact GPU
state at each timeline point.

The hook fires once on the first **production** divert call (warmup
+ selftests have already run by the time the hook is set, just before
`bridge.decode()`).

### Findings (i=3, K-cache layer 0, dstO=0, dstSize=262144)

| Window | Row 0 [c0..c7] | Row 2 [c512..c519] | Row 5 [c1280..c1287] |
|---|---|---|---|
| pre-kernel (temp-dst from real-dst) | `[0,0,0,0,0,0,0,0]` | `[0,0,0,0,0,0,0,0]` | `[0,0,0,0,0,0,0,0]` |
| post-kernel (temp-dst after compute) | `[0,0,0,0,0,0,0,0]` | `[32768,0,0,0,0,0,0,0]` | `[0,0,0,32768,32768,0,32768,0]` |
| post-copy-back (real-dst) | `[0,0,0,0,0,0,0,0]` | `[32768,0,0,0,0,0,0,0]` | `[0,0,0,32768,32768,0,32768,0]` |
| src0 row N first 4 F32 (kernel input) | `[0, 0, 0, 0]` | `[-0.0, 0, 0, 0]` | (sparse 0/-0) |

`32768 = 0x8000` = f16 `-0.0`. **Row 0** of temp-dst stayed exactly
zero post-kernel — every thread read f32 `0.0` and wrote f16 `0x0000`,
which is a no-op via the atomic-CAS path (`old & mask | 0 == old & mask`).
**Rows 2..5** show sparse `0x8000` values exactly where the
corresponding src0 read returned f32 `-0.0` (raw byte pattern
`00 00 00 80` — sign bit set, mantissa 0).

Verified at the byte level for row 2 cell 0:

```
src0AtKernelTimeF32 row 2 raw bytes: 00 00 00 80  00 00 00 00  00 00 00 00  00 00 00 00
                                     ↑↑↑↑↑↑↑↑↑↑↑
                                     0x80000000 LE = f32(-0.0)
```

The kernel is correct. The dispatcher is correct. The divert path is
correct. **The GPU buffer at h26o0 simply does not hold the
K-projection-after-ROPE values that SET_ROWS expects.**

### Why h26o0 is stale at i=3

The runLog for the first 8 ops (op 25 = RMS_NORM, 29 = MUL_MAT,
42 = SET_ROWS) confirms the prefill graph topology:

| i | op | dst | srcs |
|---|---|---|---|
| 0 | RMS_NORM | h26o0 | h26o0 (in-place norm of input embeddings) |
| 1 | MUL_MAT (Q proj) | h26o4194304 | weights @ h26o0 (norm input) |
| 2 | MUL_MAT (K proj) | h26o4194304 | weights @ h26o0 (norm input) |
| **3** | **SET_ROWS (cache_k_l0)** | **h25o0** | **h26o0**, h26o524288 (idx), h25o0 |
| 4 | SET_ROWS (cache_v_l0) | h25o262144 | h26o528384, h26o1052672, h25o262144 |
| 5+ | MUL_MAT (attn) | ... | ... |

i=3's `src0` is at offset 0 of handle 26. But the K-projection MUL_MAT
output is at offset **4194304**, not 0. Between i=2 and i=3, the K
data needs to move from offset 4194304 → offset 0. There is no JSEP
op in the runLog for that move — meaning **the move is a CPU-fallback
op** (likely `CPY` plus possibly ROPE).

CPU-fallback ops post-Stage-4.4-F1 dereference `tensor->data =
host_mirror + offset` and read/write **host_mirror only**. They never
touch the GPU buffer. Stage 4.5 H1 reads GPU→host after every JSEP
runOp, but the inverse direction (host→GPU after every CPU op) is
absent. So:

- CPU `CPY`/`ROPE` chain: writes ROPE'd K into host_mirror at h26o0.
- GPU buffer at h26o0: unchanged from initial allocation (zeros,
  with sparse `0x80000000` byte patterns from somewhere in the
  allocator/host-mirror init — not load-bearing for the diagnosis).
- JSEP SET_ROWS at i=3: reads GPU buffer at h26o0 — gets stale zeros
  + sparse `-0.0`, kernel writes f16 of those to KV cache.

### Why i=14 and others work

i=14's src0 = h26o528384. The runLog shows JSEP MUL_MATs writing to
this region between i=4 and i=14 (e.g., i=5 dst=h26o6295552, but
others overlap into h26o5xxxxx range). When a JSEP op writes to that
GPU region, Stage 4.5 H1 ALSO copies it to host_mirror — and the GPU
buffer naturally holds fresh data. Subsequent JSEP reads see fresh
data.

But h26o0 sits in a region only written by CPU ops in the prefill
graph (post-norm input + post-CPY-of-K). JSEP ops only READ from
h26o0; they don't write. So h26o0's GPU side never gets refreshed.

## False trails ruled out

The investigation chased multiple hypotheses before landing on the
root cause. Documenting them so future readers don't repeat the
search:

1. **"Status -1 at i=3"** (initial Step B finding): turned out to be
   an artifact of an earlier Stage 4.8 sentinel-probe variant that
   added awaits BEFORE `runOpOrig`. The awaits suspended JSPI long
   enough that the C-stack `desc` array (`int32_t desc[200]`,
   declared OUTSIDE the for loop in `ggml-jsep.cpp:409`) was
   overwritten by the next iteration before `runOpOrig` re-read it.
   The wrapper's top-of-function read saw op=42; `runOpOrig` saw
   op=25; dispatchRmsNorm rejected the garbage descriptor and
   returned -1. **JSPI does NOT preserve the wasm stack across
   `EM_ASM_INT` Promise-await reentries when the EM_ASM body's
   awaits run microtasks that re-enter wasm-frame state**.
   Diagnostic-only — fixed by moving probe activity to AFTER
   `runOpOrig`.
2. **"Pipeline cache miss / first-dispatch race"**: ruled out by
   Step A (warmup didn't fix it).
3. **"Atomic CAS livelock"**: ruled out by observing that ALL rows
   show the same sparse-`0x8000` pattern when src0 has sparse
   `-0.0` — kernel is operating correctly, just on stale input.
4. **"i=3 silently no-ops" (Stage 4.7 R1 framing)**: incorrect.
   The dispatch runs; the kernel reads stale-but-real data; the
   writes happen but produce zero-or-`-0.0` cells indistinguishable
   from no-op for the 0x0000 cases.

## Files touched (all diagnostic, will revert in Stage 4.9)

- `src/inference/jsep/index.ts` — added shape-matched warmup at end
  of `installJsepCallbacks` (Step A).
- `src/inference/jsep/ops/set-rows.ts` — added entry/exit log
  pushed to `globalThis.__stage48SetRowsLog` and the divert-hook
  capture path (gated by `__stage48DivertHook`).
- `smoke-test/p2-v2-spike.src.ts` — `console.error` capture buffer,
  full-`nb` capture in `setRowsDiagEntry`, `__stage48Captures`
  result object with multi-row temp-dst + src0 readbacks.

`make checkall` green: 747 pass / 36 skip / 0 fail.

## Stage 4.9 brief — close the host→GPU writeback gap

**One-line goal**: when a JSEP op reads a tensor that was last
written by a CPU-fallback op, the GPU buffer must reflect the
host_mirror contents.

**One-line context**: Stage 4.4 F1 added the host_mirror; Stage 4.5
H1 added GPU→host writeback after each JSEP runOp. The symmetric
HOST→GPU sync after each CPU-fallback op is missing.

### Three implementation options

**H1-inverse (mirror Stage 4.5 H1)**: in `ggml_backend_jsep_graph_compute`,
detect each CPU-fallback op (i.e., a node where `device_supports_op`
returns false and the op didn't go through `jsepRunOp`) and after it
runs, copy its dst tensor's `host_mirror[offset..+size]` → GPU
buffer via `jsepWrite`. Symmetric to H1's `jsepRead` GPU→host. Cost:
one round-trip per CPU op (similar to H1's ~7% per-token overhead).

**H2 (graph-walk pre-pass)**: before each JSEP runOp dispatched via
`jsepRunOp`, walk the descriptor's source tensors and check if their
underlying buffers were last-written by a CPU op (track via a
"last-writer" tag per (handle, byte-range)). For each CPU-written
range that a JSEP op reads, sync host→GPU via `jsepWrite`. Cost:
only fires when needed; lower steady-state overhead.

**H3 (sync-on-demand inside dispatch ops)**: inside `dispatchMatmul`
/`dispatchRmsNorm`/`dispatchSetRows`, before reading source bind
groups, force a host→GPU sync for any source tensor flagged as
CPU-dirty. Most local; smallest blast radius.

H1-inverse is the simplest first-cut fix and parallels the existing
H1 pattern; H2/H3 are optimization paths if the per-op overhead
matters. Stage 4.9 should land H1-inverse first (cheap, mirrors
known-good Stage 4.5 H1), measure, then decide if H2/H3 is worth
the complexity.

### Exit criteria — Stage 4.9 closes when documented in `STAGE-4.9-RESULT.md`

1. The spike's diag for i=3 shows `src0AtKernelTimeF32` row 0..5
   with REAL F32 K data (not zeros / sparse -0.0).
2. Post-kernel temp-dst rows 0..5 show real f16 K data (not all
   zeros / sparse 0x8000).
3. Post-copy-back real-dst rows 0..5 match.
4. The spike's chat path decodes "Paris" (or some sensible
   English answer) — that's the load-bearing exit signal for the
   entire Phase 3 Stage 4.x sequence, finally cleared.
5. Patch stack: 7 → 8 (the H1-inverse change in `ggml-jsep.cpp`).
6. `make checkall` green.

After Stage 4.9 closes successfully, revert the Stage 4.8
diagnostic instrumentation (`__stage48SetRowsLog`, divert-hook
capture, console.error wrap, multi-row diag captures, shape-matched
warmup). Keep them only if Stage 4.9 needs them for follow-up
probes.
