# Stage 4.20 — Probe 9b: JSEP weight-upload byte-hash

**Date:** 2026-05-07
**Probe:** FNV-1a-32 hash on the bytes `set_tensor` sees for layer-0
`blk.0.attn_q.weight` and `blk.0.attn_k.weight`, compared against an
independent JS-side hash of the same tensors parsed out of the GGUF
buffer.
**Outcome:** **F (hashes match — upload preserves bytes from GGUF
through `set_tensor`).**

## TL;DR

| Tensor                  | C++ pre-upload (set_tensor) | JS-side ref (GGUF parse) | size       | match |
| ----------------------- | --------------------------- | ------------------------ | ---------- | ----- |
| `blk.0.attn_q.weight`   | `0xf2f7188c`                | `0xf2f7188c`             | 2,359,296  | ✅    |
| `blk.0.attn_k.weight`   | `0x9399f36a`                | `0x9399f36a`             |   294,912  | ✅    |

Both Q4_0 layer-0 weight tensors arrive at `set_tensor` byte-identical
to the bytes the GGUF file contains. The Stage 4.18/4.19 production
`Qcur-0` Δ = 5.24e-4 / `Kcur-0` Δ = 3.38e-4 cannot originate from
corruption between the GGUF parser and `set_tensor`.

## What was implemented

### 1. C++ pre-upload hash

`~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp` (Stage 4.20 P10
patch, +49 LOC):

- `static int g_weight_hash_probe = 0` file-scope flag.
- `extern "C" EMSCRIPTEN_KEEPALIVE void ggml_jsep_set_weight_hash_probe(int)`
  toggle export.
- `static inline uint32_t weight_hash_fnv1a(const void *, size_t)` —
  standard FNV-1a-32 (`h = 2166136261`, `prime = 16777619`).
- Inside `ggml_backend_jsep_buffer_set_tensor`, after the existing
  `host_mirror` memcpy + `ggml_jsep_write` calls and before the Stage
  4.12 setTensorLog gate, add a name-gated hash + `EM_ASM` log to
  `globalThis.__weightHashLog` when the probe is armed and the tensor
  name matches `blk.0.attn_q.weight` or `blk.0.attn_k.weight` (exact
  `std::strcmp`).

### 2. CMakeLists.txt export gating

`src/wasm/CMakeLists.txt`: append `_ggml_jsep_set_weight_hash_probe`
to `EXPORTED_FUNCTIONS` only when `WEBLLM_BACKEND_JSEP` is on. The
non-JSEP build doesn't link `ggml-jsep`; listing the symbol there
would yield a link-time "undefined symbol".

### 3. Spike harness wiring

`smoke-test/p2-v2-spike.src.ts`:

- Import `GgufParser` from `../src/models/gguf-parser.js`.
- Initialize `globalThis.__weightHashLog = []` and call
  `mod._ggml_jsep_set_weight_hash_probe(1)` BEFORE `bridge.loadModel(buf)`.
  Guard the call with `typeof setWeightHashProbe === "function"` so the
  spike still loads under non-JSEP builds.
- After `bridge.loadModel(buf)` returns, parse `buf` independently via
  `GgufParser.parse(buf)`, find the two target tensors, compute their
  byte size from `dimensions × ggmlTypeSize(type)` (Q4_0 = 18/32 bytes
  per element), slice `buf[dataOffset + tensor.offset, ..., +tSize]`,
  compute FNV-1a-32, compare against the C++ pre-upload entry.
- Emit a single-line verdict per tensor and a final `OUTCOME: E/F`
  classification line. Save the verdict array to
  `globalThis.__weightHashVerdict` for post-hoc inspection.

The JS-side parser computes the byte size locally because
`GgufTensorInfo` carries `name, nDimensions, dimensions, type, offset`
but not `size` (the parser only computes a fleet-wide `totalDataSize`).

## Smoking gun

```
[probe9b] weight-hash probe armed
[probe9b] blk.0.attn_q.weight: pre=0xf2f7188c ref=0xf2f7188c size_pre=2359296 size_ref=2359296 match=true
[probe9b] blk.0.attn_k.weight: pre=0x9399f36a ref=0x9399f36a size_pre=294912 size_ref=294912 match=true
[probe9b] OUTCOME: F (hashes match — upload preserves bytes)
```

Both hashes match to the bit, both sizes match, and the JS-side parse
agrees with the C++ side on the bytes-per-element calculation
(`2359296 = 4194304 × 18 / 32` for [2048,2048] Q4_0;
`294912 = 524288 × 18 / 32` for [256,2048] Q4_0).

Per-token decode within Stage 4.19 noise; `make checkall` green
(747 pass / 36 skip / 0 fail). Probe runs only at model-load time
(roughly two extra ~MiB FNV passes); decode-loop perf unaffected.

## What this rules out

- **Outcome E (corruption between GGUF parser and `set_tensor`).**
  The bytes that `set_tensor` hands off to `ggml_jsep_write` are the
  bytes the GGUF file contains. No mutation in libllama's GGUF parser,
  ggml allocator, or any caller chain in between.

The Stage 4.18 brief's three remaining sub-hypotheses:
- (U-A) attn_norm-0 input differs — already refuted by Stage 4.19
  (attn_norm-0 bit-identical).
- (U-B) Q4_0 weight bytes upload corruption — **partially refuted by
  Probe 9b** (set_tensor's view matches the GGUF). The remaining
  fragment (host→GPU buffer copy via `Module.jsepWrite` →
  `device.queue.writeBuffer`) is not yet measured.
- (U-C) WGSL pipeline cache key collision — not addressed.

## What this leaves open — Stage 4.21 split

Stage 4.20's brief flagged two F-class follow-ons in case (1)+(2)
classified to F:

- **F-1 (kernel re-investigate):** all hashes match through the GPU
  readback step → kernel must be doing something different at
  production conditions than at the synthetic Stage 4.18 sweep
  conditions. Plausible loci: output-tile boundary cases at M=2048,
  pipeline cache key collision, bind-group binding offset mismatch,
  workgroup count.
- **F-2 (host→GPU corruption):** pre-upload hash matches JS-parse but
  GPU readback hash differs ⇒ `Module.jsepWrite` →
  `device.queue.writeBuffer` corrupts. Common culprits: alignment
  padding, ArrayBuffer detach, view stride, size truncation.

Stage 4.21 brief queues the GPU-side post-upload readback hash as
the disambiguator. Implementation: in JS post-`loadModel`, walk
`__weightHashLog`; for each entry, locate the target `GPUBuffer` via
`dataManager.get(bufHandle).buffer`; stage a `mapAsync(GPUMapMode.READ)`
covering `[offset, offset+size)`; compute FNV-1a-32; compare to
`fnv1a_pre`. Match ⇒ Outcome F-1; differ ⇒ Outcome F-2.

## Patch stack

12 → 13 (added: Stage 4.20 P10 — `g_weight_hash_probe` flag +
`ggml_jsep_set_weight_hash_probe` toggle + FNV-1a hash hook in
`ggml_backend_jsep_buffer_set_tensor`).

## Files touched

| File                                                                         | Role                                          |
| ---------------------------------------------------------------------------- | --------------------------------------------- |
| `~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp`                         | Probe state, toggle, hash, EM_ASM log         |
| `src/wasm/CMakeLists.txt`                                                    | JSEP-only export gating                       |
| `smoke-test/p2-v2-spike.src.ts`                                              | Probe arm, JS-side ref hash, verdict          |

## Raw artifacts

- [`STAGE-4.20-spike-output.json`](STAGE-4.20-spike-output.json) —
  full agentchrome `page text` output (169 lines / 67 KiB), includes
  the verdict lines plus the unchanged Stage 4.18/4.19 selftests +
  checkpoints.

## Stage 4.21 brief — paste-and-go

**One-line goal:** Disambiguate F-1 vs F-2 by hashing the bytes that
land on the GPU buffer (after `Module.jsepWrite` → `queue.writeBuffer`)
and comparing to the C++ pre-upload hash.

**One-line context:** Stage 4.20 confirmed bytes are intact through
`set_tensor`; the next link in the chain — host→GPU `writeBuffer` —
remains unmeasured.

### Paste-and-go bootstrap

```bash
cd /Users/probello/Repos/webllm
git log --oneline -3
#   → <Stage 4.20 TODO closure>
#   → <Stage 4.20 reports closure>
#   → <Stage 4.20 feat commit>

( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → <Stage 4.20 P10 patch tip>   webllm-browser-patches   (patch stack 13)

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &

PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')

agentchrome --port "$PORT" navigate --tab "$SPIKE_TAB" "http://localhost:8031/p2-v2-spike.html?v=stage4.21-replay"
agentchrome --port "$PORT" page wait --tab "$SPIKE_TAB" --text OUTCOME --timeout 240000
agentchrome --port "$PORT" page text --tab "$SPIKE_TAB" | grep -E "probe9b|OUTCOME"
#   → expect Stage 4.20 baseline:
#     blk.0.attn_q.weight  pre=0xf2f7188c  ref=0xf2f7188c  match=true
#     blk.0.attn_k.weight  pre=0x9399f36a  ref=0x9399f36a  match=true
#     OUTCOME: F (hashes match — upload preserves bytes)
```

### Probe 9c implementation sketch

1. **JS-side dataManager lookup.** After `loadModel`, walk
   `globalThis.__weightHashLog`. For each entry:
   - `const bufRec = mod.__jsep.dataManager.get(entry.bufHandle)` →
     `{buffer: GPUBuffer, size: number}`.
   - Stage a copy: allocate `GPUBuffer({size: entry.size,
     usage: COPY_DST | MAP_READ})`, encode
     `copyBufferToBuffer(bufRec.buffer, entry.offset, staging, 0, entry.size)`,
     `device.queue.submit`, `await staging.mapAsync(GPUMapMode.READ)`,
     `getMappedRange()` → `Uint8Array`.
   - Compute FNV-1a-32 over the staging buffer.
   - `staging.unmap(); staging.destroy()`.
2. **Verdict.** Compare GPU hash to `entry.fnv1a_pre`. Match ⇒ F-1
   (kernel re-investigate); differ ⇒ F-2 (host→GPU corrupts).
3. **Cache the staging buffer per size class** (Q-shape and K-shape
   are the only two — 2.25 MiB and 0.28 MiB) to keep the probe cheap.
4. Edge case: under Option A-prime weights live in `jsep_buf` with
   `bufHandle ∈ {36, 37, 38, 39}` (the 4×128 MiB weight buckets seen
   in `LIVE_BUFFERS`). The dataManager `bufHandle` field in the log
   should already point to one of these. If `bufRec.buffer.usage`
   doesn't include `COPY_SRC`, the JSEP buffer-allocation code
   (`src/inference/jsep/data-manager.ts` or equivalent) needs to
   widen the usage flags. Check first; expected to already be set
   given the Stage 4.5/4.15 divert paths use copyBufferToBuffer.

### Files to touch

| File                                                                         | Role                                            |
| ---------------------------------------------------------------------------- | ----------------------------------------------- |
| `smoke-test/p2-v2-spike.src.ts`                                              | Add post-load GPU readback hash + verdict       |
| `src/inference/jsep/data-manager.ts` (only if usage flags need widening)     | Add `COPY_SRC` to weight buffer usage           |

### Exit criteria

- `STAGE-4.21-RESULT.md` documents Outcome F-1 (match) or F-2 (differ).
- `make checkall` green.
- TODO.md collapses Stage 4.20 brief to a one-line pointer; pastes
  Stage 4.21 closure paragraph + Stage 4.22 brief in its place.
