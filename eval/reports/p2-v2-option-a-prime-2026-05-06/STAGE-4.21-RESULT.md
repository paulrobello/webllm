# Stage 4.21 — Probe 9c: GPU-side post-upload mapAsync readback hash

**Date:** 2026-05-07
**Probe:** FNV-1a-32 hash of the bytes that actually live on the JSEP
`GPUBuffer` for layer-0 `blk.0.attn_q.weight` and `blk.0.attn_k.weight`
post-`loadModel`, computed via `copyBufferToBuffer` → `mapAsync` →
`getMappedRange`. Compared against `entry.fnv1a_pre` from Stage 4.20
(set_tensor's pre-upload host-side hash).
**Outcome:** **F-1 (GPU bytes match `fnv1a_pre` — the entire upload chain
through the GPU is bit-clean; kernel re-investigation under production
conditions is the next probe target).**

## TL;DR

| Tensor                  | set_tensor pre-upload | GPU readback (`copyBufferToBuffer` + `mapAsync`) | size      | match |
| ----------------------- | --------------------- | ------------------------------------------------ | --------- | ----- |
| `blk.0.attn_q.weight`   | `0xf2f7188c`          | `0xf2f7188c`                                     | 2,359,296 | ✅    |
| `blk.0.attn_k.weight`   | `0x9399f36a`          | `0x9399f36a`                                     |   294,912 | ✅    |

Both Q4_0 layer-0 weight tensors live byte-identical on the GPU. The
chain `GGUF parse → ggml allocator → set_tensor → Module.jsepWrite →
device.queue.writeBuffer → JSEP GPUBuffer` is end-to-end bit-clean. The
Stage 4.18/4.19 production `Qcur-0` Δ = 5.24e-4 / `Kcur-0` Δ = 3.38e-4
**does not** originate in the host→GPU upload path.

`GENERATED_TEXT = "inonic boso-"` (gibberish, unchanged from Stage 4.20)
— the production bug remains; it is now localized strictly inside the
dispatch / kernel-execution boundary at production conditions.

## What was implemented

### Spike harness extension

`smoke-test/p2-v2-spike.src.ts` — immediately after the existing Stage
4.20 verdict block (`probe9b OUTCOME: F`):

- Walk `globalThis.__weightHashLog` (the entries pushed by the C++
  Stage 4.20 probe via `EM_ASM`, each carrying `{name, bufHandle, offset,
  size, fnv1a_pre}`).
- For each entry: `runtime.dataManager.get(entry.bufHandle)` →
  `{buffer: GPUBuffer, size}`. Verify `entry.offset + entry.size <=
  rec.size` to catch handle-mismapping early.
- Allocate a staging buffer with `MAP_READ | COPY_DST`, encode
  `copyBufferToBuffer(rec.buffer, entry.offset, staging, 0, entry.size)`,
  `device.queue.submit([encoder.finish()])`, `await staging.mapAsync(
  GPUMapMode.READ, 0, entry.size)`.
- Compute FNV-1a-32 over the mapped `Uint8Array` (same loop as the
  Stage 4.20 JS-side ref helper: `h = 2166136261`, `prime = 16777619`).
- `staging.unmap(); staging.destroy()`.
- Compare `(h >>> 0) === (entry.fnv1a_pre >>> 0)`; log
  `[probe9c] OUTCOME: F-1` (all match) or `F-2` (any differ).

The probe shares the existing `runtime.dataManager.get()` /
`runtime.device.queue.submit()` pattern used by `dumpBuf11Pre` in the
same file (Stage 4.2-and-later activations dumps), so no new helper is
needed.

### What did not change

- **No C++ changes.** Stage 4.20's pre-upload probe + `__weightHashLog`
  push is sufficient input; this stage only reads back from the GPU.
- **No JSEP buffer-usage flag changes.** The data manager already
  allocates every JSEP buffer with
  `STORAGE | COPY_SRC | COPY_DST` (`src/inference/jsep/gpu-data-manager.ts`
  lines 78-83 / 92-97). The 4×128 MiB weight buffers (handles 36-39 in
  the post-load `LIVE_BUFFERS` log) inherit `COPY_SRC` already; no flag
  widening required.
- **No WASM rebuild.** The probe is pure JS; only
  `bun build smoke-test/p2-v2-spike.src.ts` was needed.

## How the probe was run

```bash
# Spike build (no WASM rebuild needed).
bun build smoke-test/p2-v2-spike.src.ts \
  --outfile smoke-test/p2-v2-spike.js --target browser
# Bundled 12 modules in 6ms — p2-v2-spike.js 0.29 MB.

# Cache-busted navigation in the reused agentchrome session.
agentchrome --port "$PORT" navigate --tab "$SPIKE_TAB" \
  http://localhost:8031/p2-v2-spike.html?v=stage4.21-probe9c
agentchrome --port "$PORT" page wait --tab "$SPIKE_TAB" \
  --text OUTCOME --timeout 240000
```

## Captured output

Saved at `STAGE-4.21-spike-output.txt` in this directory; relevant lines:

```
[probe9b] weight-hash probe armed
model loaded in 271 ms; vocab = 32000
[probe9b] blk.0.attn_q.weight: pre=0xf2f7188c ref=0xf2f7188c size_pre=2359296 size_ref=2359296 match=true
[probe9b] blk.0.attn_k.weight: pre=0x9399f36a ref=0x9399f36a size_pre=294912  size_ref=294912  match=true
[probe9b] OUTCOME: F (hashes match — upload preserves bytes)
[probe9c] blk.0.attn_q.weight: pre=0xf2f7188c gpu=0xf2f7188c size=2359296 match=true
[probe9c] blk.0.attn_k.weight: pre=0x9399f36a gpu=0x9399f36a size=294912  match=true
[probe9c] OUTCOME: F-1 (GPU bytes match — upload chain bit-clean; kernel re-investigation)
LIVE_BUFFERS = [{"h":36,"size":134217728,"bucket":9},{"h":37,"size":134217728,"bucket":9},
                {"h":38,"size":134217728,"bucket":9},{"h":39,"size":134217728,"bucket":9},
                {"h":40,"size":16777216,"bucket":7},{"h":41,"size":67108864,"bucket":8}]
GENERATED_TOKENS = [297,8927,13601,29877,29899]
GENERATED_TEXT = "inonic boso-"
```

Console output check: only benign llama.cpp informational lines on
stderr (`llama_model_loader: loaded meta data...`, `adapter_info: ...`)
— no real failures.

`make checkall`: ✅ fmt + lint + typecheck + test (747 pass, 36 skip,
0 fail across 783 tests).

## What this rules out

The combination of Stage 4.20's `F` (set_tensor sees clean GGUF bytes)
and Stage 4.21's `F-1` (GPU buffers contain those same bytes) closes
the entire weight-upload chain under inspection:

```
GGUF file bytes
  │  (Stage 4.20 — JS-side ref hash via GgufParser)
  ▼
buf bytes parsed out of fetched ArrayBuffer
  │  (Stage 4.20 — C++ pre-upload hash inside set_tensor)
  ▼
set_tensor `data` argument bytes  ← matches GGUF
  │  (Stage 4.21 — GPU readback hash)
  ▼
JSEP GPUBuffer at (handle, offset, size)  ← matches set_tensor input
```

Every link in the chain is bit-identical. The 5.24e-4 production delta
must originate **inside the dispatch / kernel-execution boundary at
production conditions** in a way the Stage 4.18 standalone synthetic
sweep did not reproduce.

## Patch stack delta

**llama.cpp `webllm-browser-patches`:** unchanged at
`ef89f9314` (13 patches, including Stage 4.20 P10).

**webllm working tree:** Stage 4.21 adds a single ~70-LOC TS block in
`smoke-test/p2-v2-spike.src.ts` (Probe 9c) plus this report.

## Next probe — Stage 4.22

Per the Stage 4.21 brief's branch-on-outcome table (Outcome F-1):

> Dump the actual kernel `src0` + `src1` bytes from the GPU at
> production-dispatch time (one-shot `copyBufferToBuffer` interception
> inside `dispatchMatmul`), feed them into the standalone Stage-4.18
> synthetic harness, and check whether the synthetic kernel reproduces
> the 5.24e-4 delta with those exact inputs.

Two diagnostic paths emerge:

- **Yes** (synthetic harness on the captured production `src0`/`src1`
  reproduces 5.24e-4): the Stage 4.18 sweep missed an output-tile
  boundary case at M=2048 (TinyLlama Q-projection M dim). Re-run the
  sweep at every (M, K, N) actually used by `Qcur-0` with denser
  sampling around tile boundaries.
- **No** (synthetic produces the expected ≤1e-7 delta on the same
  bytes): the bug lives between the dispatch site and shader execution.
  Likely culprits: pipeline-cache key collision, bind-group binding
  offset mismatch, workgroup-count off-by-one, or src0/src1 swap at the
  descriptor level.
