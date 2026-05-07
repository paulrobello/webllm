# Stage 4.14 — Probe 4: localize the op zeroing addr 99811136 / refute Stage 4.13's CPU-D narrative

**Status:** CLOSED 2026-05-07. **Outcome C → refined diagnosis CPU-E.** Probe 4
falsifies Stage 4.13's "CPU op clobbers attn_norm-0's CPU scratch between i=1
and i=2" framing. The CPU graph between callIdx 1 and callIdx 2 contains zero
ops that write to the `(addr_e1, addr_e1 + 6144)` window. The cross-backend
bridge from JSEP-resident Q/K projection results to CPU-buft Kproj/Qproj
scratch slots fires correctly via `ggml_backend_jsep_buffer_get_tensor` — but
**every JSEP MUL_MAT for Q (every layer) and K layer 0 produces zero output at
host_mirror**. The real bug is in the JSEP MUL_MAT divert path's interaction
with H1, not in any CPU op.

## Headline

- `make checkall` green.
- All 6 spike kernel selftests PASS.
- `GENERATED_TEXT = "ntiuhuihnerquant"` bit-identical to Stage 4.13.
- `topId/topVal = 593/0.159` bit-identical to Stage 4.13.
- Per-token decode 127.92 ms (Stage 4.13 baseline 127.40 ms; Probe 4
  instrumentation invisible).
- `COUNTER_DELTAS = {alloc:0, free:0, write:4404, read:1602, clear:0,
  runOp:1602, sync:3671}` — `read == runOp`: H1 post-runOp writeback fires
  exactly once per JSEP op. So H1 is *not* the missing piece by call count.
- Patch stack: 10 → 11 (Probe 4 instrumentation in `ggml-cpu.cpp`
  post-compute capture + `ggml-jsep.cpp` `get_tensor` log; llama.cpp
  `ddeb2fb6e`). Stage 4.13's Probe 3 instrumentation retained (cumulative
  diagnostic).

## Probe 4: post-compute CPU dst readback + `get_tensor` log

Instrumentation added in this stage (both diagnostic-only — expected to revert
or refactor at Stage 4.15):

1. **`ggml/src/ggml-cpu/ggml-cpu.cpp`** — after `ggml_graph_compute(cgraph,
   &cplan)` returns in `ggml_backend_cpu_graph_compute`, walk every node and
   capture `(callIdx, nodeIdx, op, name, dst_addr, dst_type, src0_addr,
   src0_type, src0_name, dst_first4, src0_first4)` into
   `globalThis.__cpuOpDstLog` (cap 200). The src0 capture was added in a
   second pass after the first capture surfaced the `CPU#Qcur-N#0` /
   `CPU#Kcur-0#0` sched-allocated scratch names without showing what
   addresses ROPE was reading from.
2. **`ggml/src/ggml-jsep/ggml-jsep.cpp::ggml_backend_jsep_buffer_get_tensor`**
   — log every call with `(handle, offset, size, name, mirror_addr,
   dst_addr, f32_first4)` into `globalThis.__jsepGetTensorLog` (cap 64).
   This is the JSEP→CPU bridge endpoint that ggml-backend-sched calls when
   moving a JSEP-resident tensor into a CPU-buft scratch.

## Smoking-gun trace

**`__setTensorLog` (Stage 4.13 instrumentation, unchanged):**

| i | name                    | offset    | size  | data_addr | f32_first4         |
|---|-------------------------|-----------|-------|-----------|--------------------|
| 0 | `JSEP#input embedding`  | 0         | 49152 | …         | (initial embedding)|
| 1 | `JSEP#attn_norm-0#0`    | 0         | 49152 | 99811136  | `[5.4e-6, 1.2e-5, -1.4e-4, -1.1e-4]` ★ valid |
| 2 | `JSEP#Kcur-0 (view)#0`  | 0         | 6144  | 99811136  | `[0,0,0,0]` ★ "the bug" per Stage 4.13 |

**`__cpuOpDstLog` filtered to `dst_addr ∈ [99811136, 99817280)`
(addr_e1's 6144B window):** 65 entries. **Zero of them** correspond to a CPU
op writing the window between callIdx 1 and callIdx 2. The first 5:

| callIdx | nodeIdx | op (id/name)         | dst_addr  | dst_f4                     | src0_name      | src0_addr | src0_f4 |
|---------|---------|-----------------------|-----------|----------------------------|-----------------|-----------|---------|
| 1       | 0       | 7 / MUL (norm-0×gain) | 99811136  | `[5.4e-6, 1.2e-5, …]` ★ valid | `CPU#norm-0#0` | 99811136  | same    |
| 2       | 0       | 48 / ROPE (`Qcur-0`)  | 104022080 | `[0,0,0,0]`                | `CPU#Qcur-0#0` | 104022080 | `[0,0,0,0]` |
| 2       | 1       | 29 / MUL_MAT (`Vcur-0`) | 108216384 | `[-3.1e-6, -1.5e-6, …]` valid | `blk.0.attn_v.weight` (jsep) | 775404672 | (Q4_K bytes) |
| 2       | 2       | 36 / RESHAPE (`Vcur-0`) | 108216384 | same as nodeIdx 1 valid   | `Vcur-0`       | 108216384 | same    |
| 3       | 0       | 48 / ROPE (`Kcur-0`)  | 99811136  | `[0,0,0,0]`                | `CPU#Kcur-0#0` | 99811136  | `[0,0,0,0]` |

**Refutation of Stage 4.13's CPU-D narrative.** Stage 4.13 framed the bug as
"a CPU op zeroes addr 99811136 between i=1 and i=2." Probe 4 shows that
between callIdx 1 (CPU MUL writes valid attn_norm-0 to 99811136) and callIdx 3
(CPU ROPE reads `[0,0,0,0]` from 99811136), **no CPU op writes to that
address.** The CPU graph at callIdx 2 produces `Qcur-0` and `Vcur-0` at
*different* addresses (104022080 / 108216384). The 99811136 slot is recycled
by ggml's scratch allocator: at callIdx 1 it holds attn_norm-0, then at
callIdx 3 it's reassigned as the destination for `CPU#Kcur-0#0` (the
sched-allocated cross-backend bridge slot for ROPE's input).

So `Kcur-0` at addr 99811136 is **a fresh allocation that never received its
expected payload**, not a clobbered live value.

**`__jsepGetTensorLog` — the JSEP→CPU bridge:**

| i  | handle | offset    | size  | name                | mirror_addr | f32_first4 |
|----|--------|-----------|-------|---------------------|-------------|------------|
| 1  | 26     | 4194304   | 49152 | `Qcur-0`            | 53659904    | `[0,0,0,0]` ★ broken |
| 2  | 26     | 4194304   | 6144  | `Kcur-0`            | 53659904    | `[0,0,0,0]` ★ broken (aliases Q's slot) |
| 3  | 26     | 6295552   | 196608| `kq-0`              | 55761152    | `[0,0,0,0]` (downstream of broken Q/K) |
| 4  | 26     | 35655680  | 49152 | `kqv-0 (permuted)`  | 85121280    | `[0,0,0,0]` (downstream) |
| 5  | 26     | 6295552   | 49152 | `attn_out-0`        | 55761152    | `[0,0,0,0]` (downstream) |
| 6  | 26     | 2101248   | 49152 | `norm-0`            | 51566848    | valid `[-1.3e-3, 1.9e-3, …]` (RMS_NORM result; CPU set_tensor wrote it) |
| 10 | 26     | 6295552   | 49152 | `Qcur-1`            | 55761152    | `[0,0,0,0]` ★ broken |
| 11 | 26     | 528384    | 6144  | `Kcur-1`            | 49993984    | `[-3.1e-6, -1.5e-6, 4.0e-6, 4.7e-6]` "valid"? — see below |

**ggml-backend-sched correctly invokes the bridge.** The scheduler issues
exactly one `get_tensor` per cross-backend Q/K projection, with the right
handle and offset. JSEP's `get_tensor` (post-Stage-4.4 F1) reads from
`host_mirror[h] + offset` — so the `f32_first4` in this log is exactly what
host_mirror contained at bridge time. Conclusion: the bridge mechanism is
not broken; **the GPU→host_mirror sync (H1) failed for these specific
offsets** even though `read == runOp` in counters confirms H1 fires for
every runOp.

**The "valid" Kcur-1 read is stale V data, not a real K projection.** At
entry 11, the bridge reads `[-3.09e-6, -1.52e-6, 4.02e-6, 4.66e-6]` from
host_mirror[26]+528384. This is exactly Vcur-0's output at callIdx 2
(CPU MUL_MAT writes `[-3.09e-6, -1.52e-6, 4.02e-6, 4.66e-6]` to addr
108216384). Vcur-0's value lands in `host_mirror[26]+528384` because
the V cache slot for layer 0 in K cache occupies that offset, and the
scheduler `set_tensor`d V's CPU MUL_MAT result there before the K cache
SET_ROWS.

So the read at offset 528384 returns stale V data leftover from a
`set_tensor` write — not a fresh K projection. **Every JSEP MUL_MAT
for Q/K, every layer, fails to populate host_mirror at the bridge offset
sched expects to read.**

## Why Probe 4's findings refine the diagnosis to CPU-E

| Hypothesis (Stage 4.13 CPU-A/B/C, then refined CPU-D) | Status |
|---|---|
| A CPU op writes zeros to `addr_e1[0..6144)` between i=1 and i=2 | **REJECTED** — `__cpuOpDstLog` filtered to that window shows zero CPU ops there during the bridge interval. |
| Kcur-0's view `data` is mis-aliased onto attn_norm-0's CPU scratch | **REJECTED** — Kcur-0's bridge slot is a fresh sched allocation (`CPU#Kcur-0#0` at addr 99811136) created AFTER attn_norm-0's lifetime expired. The slot reuse is allocator-correct. |
| K projection is on JSEP and CPU ROPE reads its result from host_mirror | **PARTIALLY CONFIRMED** — sched does call JSEP's `get_tensor` to bridge; that mechanism works. But the host_mirror at the read offset contains zeros, so the bridge faithfully reads zeros. |

**New diagnosis (CPU-E — JSEP MUL_MAT divert path produces no host-visible
output).** All Q-projection MUL_MATs and the layer-0 K-projection MUL_MAT run
on JSEP under the divert pattern (dst aliases src1 = activations buffer h26;
divert allocates a temp GPUBuffer, dispatches kernel into temp,
`copyBufferToBuffer` from temp to dstRec.buffer at dst.offset). H1
(`module.jsepRead` post-runOp) calls `dataManager.readAsync(handle, offset,
host_mirror+offset, size)` which submits a `copyBufferToBuffer(record.buffer,
offset, staging, …)` then `mapAsync`. WebGPU FIFO ordering should mean H1's
readback sees the divert's result. Counters show H1 fires; data shows H1
reads zeros. **Therefore, either the divert kernel's tempDst contains zeros
(bind-group / kernel bug specific to Q-MUL_MAT shapes), or the
copyBufferToBuffer from tempDst to dstRec.buffer doesn't land at the
expected offset.** Stage 4.15 must localize between those two.

## Files changed

- `ggml/src/ggml-cpu/ggml-cpu.cpp` (post-compute per-node capture, ~50 LOC).
- `ggml/src/ggml-jsep/ggml-jsep.cpp` (`get_tensor` log, ~25 LOC).

Both are diagnostic-only and expected to revert/refactor in Stage 4.15.

## Stage 4.15 brief — localize divert dispatch's GPU-buffer fate

**One-line goal:** Determine whether the JSEP MUL_MAT divert path's tempDst
receives correct kernel output, and whether the copyBufferToBuffer to
dstRec.buffer lands at the expected offset, by adding per-divert-dispatch
mapAsync readbacks of (a) tempDst right after queue.submit, (b)
dstRec.buffer at `[dst.offset, dst.offset + 16)` right after queue.submit.

**One-line context:** Stage 4.14 ruled out CPU-side clobbering and confirmed
the JSEP→CPU bridge mechanism works; the gap is in the divert path's GPU
buffer write. Counter parity (`read == runOp`) shows H1 fires; the bytes
H1 reads back are zeros for every Q-projection and Kcur-0.

### Paste-and-go bootstrap

```bash
cd /Users/probello/Repos/webllm
git log --oneline -1
( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &

PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next(t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t["url"]))' 2>/dev/null)
[ -z "$TAB" ] && TAB=$(agentchrome --port "$PORT" tabs create "http://localhost:8031/p2-v2-spike.html?v=stage4.15-bootstrap" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')

agentchrome --port "$PORT" navigate "http://localhost:8031/p2-v2-spike.html?v=stage4.15-replay" --tab "$TAB"
until agentchrome --port "$PORT" js exec --tab "$TAB" 'document.getElementById("log").textContent.slice(-50)' 2>&1 | grep -qE "DONE|FAIL"; do sleep 3; done
```

### Implementation — Probe 5 (gated on a debug flag in dispatchMatmul)

In `src/inference/jsep/ops/matmul.ts` divert path, just before
`return 0`, capture two readbacks:

```ts
if ((globalThis as any).__stage415DivertProbe) {
    const dbg = (globalThis as any).__stage415DivertLog ??= [];
    if (dbg.length < 32) {
        // tempDst readback
        const tempStaging = ctx.device.createBuffer({
            size: Math.min(16, dstBytesNeeded),
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const dstStaging = ctx.device.createBuffer({
            size: Math.min(16, dstBytesNeeded),
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const probeEnc = ctx.device.createCommandEncoder();
        probeEnc.copyBufferToBuffer(tempDst, 0, tempStaging, 0, 16);
        probeEnc.copyBufferToBuffer(dstRec.buffer, dst.offset, dstStaging, 0, 16);
        ctx.device.queue.submit([probeEnc.finish()]);
        await tempStaging.mapAsync(GPUMapMode.READ);
        await dstStaging.mapAsync(GPUMapMode.READ);
        const tempBytes = new Uint8Array(tempStaging.getMappedRange().slice(0));
        const dstBytes  = new Uint8Array(dstStaging.getMappedRange().slice(0));
        tempStaging.unmap(); dstStaging.unmap();
        tempStaging.destroy(); dstStaging.destroy();
        dbg.push({
            divertIdx: dbg.length,
            dstHandle: dst.bufHandle, dstOffset: dst.offset,
            dstNe: [...dst.ne], src0Ne: [...src0.ne],
            tempFirst4F32: new Float32Array(tempBytes.buffer.slice(0, 16)),
            dstFirst4F32:  new Float32Array(dstBytes.buffer.slice(0, 16)),
        });
    }
}
```

Note: `dispatchMatmul` is currently sync; to allow the await, either gate the
probe path on an async helper called from a separate harness branch, or
expose a probe-only async wrapper. Easier: leave the dispatch sync, queue
both copies as part of the existing divert encoder, and read them in a
follow-on async drain triggered from the spike harness via a global flag —
exactly the pattern Stage 4.7 D2-tight used for SET_ROWS.

### Branch on Probe 5 outcome

| Probe 5 finding | Diagnosis | Structural fix |
|---|---|---|
| `tempFirst4F32 = [0,0,0,0]` (kernel writes zeros to temp) | Q-MUL_MAT kernel + bind-group bug for Q-shape (M=2048) | Audit kernel constants / dispatch dims for the Q shape; possibly the kernel is wedged on a workgroup-size mismatch for the larger M. |
| `tempFirst4F32` valid, `dstFirst4F32 = [0,0,0,0]` | copyBufferToBuffer doesn't land at `dst.offset` | Dst buffer/offset wiring bug; cross-check `dstRec.buffer` resolution from `dst.bufHandle` post-Stage-2 split. |
| Both valid | The bug is downstream of dispatchMatmul — H1's jsepRead/readAsync reads from a different buffer than the divert wrote | Inspect `record.buffer` lookup vs the dst buffer the divert wrote into. |

## Patch stack delta

10 → 11 (Probe 4 instrumentation + get_tensor log). Stage 4.13's set_tensor
name/data_addr capture and CPU MUL_MAT diag retained; cumulative diagnostic.
Stage 4.15 will revert most of the diagnostic instrumentation as part of the
structural fix commit, OR retain the kernel/divert self-test as a permanent
regression check (decision belongs in Stage 4.15's closure).
