# Stage 4.29 — Probe 16: CPU-side `set_tensor` weight-hash probe

**Date:** 2026-05-07
**Probe:** Add a name-gated FNV-1a-32 hook inside
`ggml_backend_cpu_buffer_set_tensor` (`ggml-backend.cpp:2237`)
mirroring Stage 4.20's JSEP probe pattern. Test whether
`blk.0.ffn_norm.weight` (F32, 8 KiB) and `blk.0.ffn_down.weight`
(Q6_K, 9.4 MiB) — the two layer-0 weights Stage 4.28 confirmed
bypass the JSEP `set_tensor` hook — land on the default CPU buft
and, if so, whether the bytes the CPU op reads match the GGUF file.
**Outcome:** **P-16-silent** — CPU hook armed and exported correctly
(no "export missing" failure, `[probe16] CPU weight-hash probe
armed` line emitted), but fired **0/7** during model load. Neither
the JSEP buft (Stage 4.28: 5/7 fire) nor the default CPU buft (this
stage: 0/7) owns `ffn_norm.weight` and `ffn_down.weight`. The
remaining buft is most likely the GGUF mmap-direct path, in which
case there is no upload step for these tensors at all — `tensor->data`
points straight into the mmap'd file bytes and Stage 4.28's
`GgufParser` reference hash is, by transitivity, also the bytes the
CPU op reads.

## TL;DR

Unified `[STAGE-4.29]` block (JSEP + CPU set_tensor logs merged
across the same 7-name allowlist Stage 4.28 used):

| Tensor                       | size      | type | ref          | jsep_pre     | jsep_gpu     | cpu_pre        | match |
| ---------------------------- | --------: | ---: | ------------ | ------------ | ------------ | -------------- | ----- |
| `blk.0.attn_q.weight`        | 2,359,296 | Q4_0 | `0xf2f7188c` | `0xf2f7188c` | `0xf2f7188c` | `<missing>`    | ✅    |
| `blk.0.attn_k.weight`        |   294,912 | Q4_0 | `0x9399f36a` | `0x9399f36a` | `0x9399f36a` | `<missing>`    | ✅    |
| `blk.0.attn_output.weight`   | 2,359,296 | Q4_0 | `0xaae061b5` | `0xaae061b5` | `0xaae061b5` | `<missing>`    | ✅    |
| `blk.0.ffn_norm.weight`      |     8,192 | F32  | `0xcba312e0` | `<missing>`  | `<missing>`  | **`<missing>`**| ❌    |
| `blk.0.ffn_gate.weight`      | 6,488,064 | Q4_0 | `0xafdfc33a` | `0xafdfc33a` | `0xafdfc33a` | `<missing>`    | ✅    |
| `blk.0.ffn_up.weight`        | 6,488,064 | Q4_0 | `0x76f44e42` | `0x76f44e42` | `0x76f44e42` | `<missing>`    | ✅    |
| `blk.0.ffn_down.weight`      | 9,461,760 | Q6_K | `0xd1429247` | `<missing>`  | `<missing>`  | **`<missing>`**| ❌    |

`[STAGE-4.29] CPU hook fired on 0/7 weights (0 byte-clean, 0 dirty)`
`[STAGE-4.29] OUTCOME: P-16-silent (CPU hook never fired — bypass weights are not on CPU buft; escalate to untargeted set_tensor logging or Shape B)`

`GENERATED_TEXT = "inonic boso-"` (gibberish, unchanged from Stages
4.27 / 4.28 — confirms the new probe code path doesn't perturb
production decode behavior).

## What this rules out

**The default CPU buft (`ggml_backend_cpu_buffer_i` /
`ggml_backend_cpu_buffer_from_ptr_i` in `ggml-backend.cpp`) is not
where `ffn_norm.weight` and `ffn_down.weight` live.** The hook's
`g_cpu_weight_hash_probe` flag was set to 1 immediately before
`loadModel`, the export `_ggml_cpu_set_weight_hash_probe` resolved
(otherwise the spike would have logged `[probe16] CPU weight-hash
export missing — old build?`), and the function compiles into
the JSEP build (the `__EMSCRIPTEN__`-guarded EM_ASM block is reached
under that build because the WASM driver is Emscripten). Yet zero
calls landed on the allowlist for any of the 7 layer-0 weights
during model load. The default-CPU-buft `set_tensor` callback in
`ggml-backend.cpp` never fires for these tensors.

Combined with Stage 4.28's JSEP-set_tensor bypass result, the byte-
integrity question for `ffn_norm.weight` and `ffn_down.weight`
remains *uncaptured by upload-time hashing*. The next probe must
either (a) instrument every remaining buft's `set_tensor` to find
which one owns these tensors, or (b) read the bytes through a
different pathway entirely (cb_eval weight-tap, or direct
`tensor->data` peek post-load).

## What this surfaces

**The bypass-weights almost certainly land on the GGUF mmap-direct
buft.** The JSEP build's libllama is configured with `GGML_CPU=OFF`
(see `Makefile:139`), which strips the full ggml-cpu/ subdirectory
out of the link. The minimal CPU buffer interfaces in
`ggml-backend.cpp` survive (we just rebuilt and the symbol exported
cleanly), but with no full CPU backend registered, libllama's weight
allocator has limited choices:

1. **JSEP buft** for tensors the JSEP backend's `supports_op` claims
   it can dispatch (the 5 Q4_0 matmul weights — confirmed in Stage
   4.28's 5/7 fire).
2. **Some other buft for everything else.** With GGML_CPU=OFF, the
   most plausible candidate is the mmap-direct host buft that
   libllama installs when it loads a GGUF via memory mapping. That
   buft is special: its `set_tensor` callback is typically a no-op
   or absent because `tensor->data` already points at the mmap'd
   file region. There's nothing to upload — the bytes ARE the
   GGUF bytes.

If that is what's happening, then **the gain-vector mis-load
hypothesis (suspect 2) is effectively dead by transitivity:**
`tensor->data` for `ffn_norm.weight` and `ffn_down.weight` points
at the same bytes Stage 4.28's `GgufParser` reference hash already
verified — `0xcba312e0` and `0xd1429247` respectively, byte-identical
to the on-disk GGUF tensor data. The CPU op that reads
`ffn_norm.weight` reads exactly the GGUF bytes; no upload step
exists to corrupt them.

This isn't a closure (we haven't directly *measured* the bytes
the CPU op reads), but it strongly suggests the cascade source
must lie elsewhere. Stage 4.27 already showed `attn_norm-0` (the
*other* layer-0 RMSNorm output, also CPU-resident) is bit-identical
between JSEP and reference paths — meaning the CPU RMSNorm kernel
works correctly when its input is clean and its gain bytes are
the GGUF bytes. The mmap-direct hypothesis is consistent with
that result.

## Reframing the cascade source after Stage 4.29

The remaining cascade-source candidates after Stage 4.29:

1. ❌ **Suspect 1** (`attn_output.weight` byte-integrity gap) —
   closed Stage 4.28 (5/5 JSEP-resident weights bit-clean).
2. ⚠️ **Suspect 2** (`ffn_norm.weight` gain-vector mis-load) —
   not directly measured yet but strongly indirect-evidence dead.
   Stage 4.27's `attn_norm-0` bit-identical result implies the
   CPU RMSNorm kernel + its gain-vector access is fine when the
   input is clean. Stage 4.29's mmap-direct framing implies the
   bytes themselves are by-construction-clean. **Direct
   confirmation requires Stage 4.30's first probe.**
3. ✅ **Suspect 3** (first8-window blindness on `kqv_out-0`)
   — unchanged. Promotes to highest priority for Stage 4.30.
4. **(New)** Cascade source upstream of `attn_out-0`. Stage 4.27
   confirmed `Qcur-0` Δ=5.24e-4 → `Vcur-0` Δ=0 (V on CPU) →
   `attn_out-0` Δ=4.77e-3 is the first JSEP node to cross the
   1e-3 threshold. The 4.77e-3 at `attn_out-0` is an OUTPUT
   delta — its INPUTS (Qcur, Kcur, V) and the kernel arithmetic
   are the actual disagreement source. Stage 4.24/4.25/4.26
   already closed Q4_K matmul, accumulation order, and libllama
   precision as the cause for `Qcur-0`. So the 5.24e-4 at Qcur-0
   must come from yet another op or input. Stage 4.30 should
   probe whether `inp_embd` / `attn_norm-0` (Qcur's upstream
   input) is bit-identical between JSEP and reference at the
   bytes the JSEP MUL_MAT actually reads.

## What was implemented

### C++ CPU-side hook (`~/Repos/llama.cpp/ggml/src/ggml-backend.cpp`)

Mirrors Stage 4.20's JSEP probe pattern (`ggml-jsep.cpp:115-348`)
into the default-CPU-buft `set_tensor` callback:

```cpp
// Stage 4.29 Probe 16 — CPU-side set_tensor weight-hash probe.
static int g_cpu_weight_hash_probe = 0;

#ifdef __EMSCRIPTEN__
extern "C" EMSCRIPTEN_KEEPALIVE void ggml_cpu_set_weight_hash_probe(int enable) {
    g_cpu_weight_hash_probe = enable ? 1 : 0;
}
#endif

static inline uint32_t ggml_cpu_weight_hash_fnv1a(const void * data, size_t size) { /* FNV-1a-32 */ }

static void ggml_backend_cpu_buffer_set_tensor(...) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);
#ifdef __EMSCRIPTEN__
    if (g_cpu_weight_hash_probe && tensor->name[0] != '\0') {
        // 7-name allowlist matching Stage 4.20/4.28
        if (match) {
            const uint32_t h = ggml_cpu_weight_hash_fnv1a(data, size);
            EM_ASM({
                if (typeof globalThis.__cpuWeightHashLog === "undefined") {
                    globalThis.__cpuWeightHashLog = [];
                }
                globalThis.__cpuWeightHashLog.push({
                    name: UTF8ToString($0, 64), offset: $1, size: $2, fnv1a_pre: ($3 >>> 0),
                });
            }, ...);
        }
    }
#endif
    GGML_UNUSED(buffer);
}
```

The hook is added inside the same `ggml-backend.cpp:2237` function
that backs **both** `ggml_backend_cpu_buffer_i` (line 2272) and
`ggml_backend_cpu_buffer_from_ptr_i` (line 2286), so a single hook
covers the default CPU buft and the from-pointer CPU buft variants.
The `<emscripten/emscripten.h>` include is added at top of file
(matches the JSEP file's pattern at `ggml-jsep.cpp:80-82`).

### CMakeLists export (`src/wasm/CMakeLists.txt`)

`_ggml_cpu_set_weight_hash_probe` added to `EXPORTED_FUNCTIONS` —
unconditionally, since `ggml-backend.cpp` ships in every build (the
JSEP-only `_ggml_jsep_set_weight_hash_probe` export is the
exception, not the rule). Confirmed: WASM build linked clean, no
"undefined symbol" diagnostic.

### JS spike harness (`smoke-test/p2-v2-spike.src.ts`)

- **Pre-loadModel arming:** `__cpuWeightHashLog = []` +
  `_ggml_cpu_set_weight_hash_probe?.(1)` paired with the existing
  Stage 4.20 JSEP arming block. Logs `[probe16] CPU weight-hash
  probe armed` on success or `... export missing — old build?` on
  failure.
- **Post-loadModel disarming:** `_ggml_cpu_set_weight_hash_probe?.(0)`
  paired with the existing JSEP disarm.
- **Stage 4.29 unification block** appended after Stage 4.28's
  outcome line: walks the 7-name allowlist, looks up `ref` (from
  `__weightHashRef`), `jsep_pre` (from `__weightHashLog`),
  `jsep_gpu` (from `__weightHashGpuVerdict`/`gpuMap`), and
  `cpu_pre` (from `__cpuWeightHashLog`). Per-weight match := every
  captured hash agrees with ref AND at least one hash was captured.
  Outcome line synthesizes P-16-{clean,gain,ffn-down,silent,other}.

### What did not change

- No new patches to llama.cpp beyond the CPU `set_tensor` hook
  (still 1-stack against `ef89f9314`; Stage 4.28 already amended P10
  in place; this stage adds patch growth in `ggml-backend.cpp`,
  which previously had no webllm-side modification).
- No webllm runtime/inference changes outside the spike harness.
- Stage 4.28's JSEP `set_tensor` allowlist + Stage 4.20's JSEP probe
  unchanged.

## Captured output

Saved at [`STAGE-4.29-spike-output.txt`](STAGE-4.29-spike-output.txt).
The relevant lines:

```
     [probe9b] weight-hash probe armed
     [probe16] CPU weight-hash probe armed
     model loaded in 439 ms; vocab = 32000
     [probe9b] blk.0.attn_q.weight: pre=0xf2f7188c ref=0xf2f7188c size_pre=2359296 size_ref=2359296 match=true
     [probe9b] blk.0.attn_k.weight: pre=0x9399f36a ref=0x9399f36a size_pre=294912 size_ref=294912 match=true
     [probe9b] blk.0.attn_output.weight: pre=0xaae061b5 ref=0xaae061b5 size_pre=2359296 size_ref=2359296 match=true
     [probe9b] blk.0.ffn_norm.weight: pre=<missing> ref=0xcba312e0 size_pre=-1 size_ref=8192 match=false
     [probe9b] blk.0.ffn_gate.weight: pre=0xafdfc33a ref=0xafdfc33a size_pre=6488064 size_ref=6488064 match=true
     [probe9b] blk.0.ffn_up.weight: pre=0x76f44e42 ref=0x76f44e42 size_pre=6488064 size_ref=6488064 match=true
     [probe9b] blk.0.ffn_down.weight: pre=<missing> ref=0xd1429247 size_pre=-1 size_ref=9461760 match=false
     [STAGE-4.29] blk.0.attn_q.weight ref_hash=0xf2f7188c jsep_pre_hash=0xf2f7188c jsep_gpu_hash=0xf2f7188c cpu_pre_hash=<missing> match=true
     [STAGE-4.29] blk.0.attn_k.weight ref_hash=0x9399f36a jsep_pre_hash=0x9399f36a jsep_gpu_hash=0x9399f36a cpu_pre_hash=<missing> match=true
     [STAGE-4.29] blk.0.attn_output.weight ref_hash=0xaae061b5 jsep_pre_hash=0xaae061b5 jsep_gpu_hash=0xaae061b5 cpu_pre_hash=<missing> match=true
     [STAGE-4.29] blk.0.ffn_norm.weight ref_hash=0xcba312e0 jsep_pre_hash=<missing> jsep_gpu_hash=<missing> cpu_pre_hash=<missing> match=false
     [STAGE-4.29] blk.0.ffn_gate.weight ref_hash=0xafdfc33a jsep_pre_hash=0xafdfc33a jsep_gpu_hash=0xafdfc33a cpu_pre_hash=<missing> match=true
     [STAGE-4.29] blk.0.ffn_up.weight ref_hash=0x76f44e42 jsep_pre_hash=0x76f44e42 jsep_gpu_hash=0x76f44e42 cpu_pre_hash=<missing> match=true
     [STAGE-4.29] blk.0.ffn_down.weight ref_hash=0xd1429247 jsep_pre_hash=<missing> jsep_gpu_hash=<missing> cpu_pre_hash=<missing> match=false
     [STAGE-4.29] CPU hook fired on 0/7 weights (0 byte-clean, 0 dirty)
     [STAGE-4.29] OUTCOME: P-16-silent (CPU hook never fired — bypass weights are not on CPU buft; escalate to untargeted set_tensor logging or Shape B)
```

## Patch stack delta

**llama.cpp `webllm-browser-patches`:** new commit on top of
`1d1d64f76` adds the CPU-side `set_tensor` hook in
`ggml-backend.cpp`. Patch stack grows by one (existing 13 →
14 patches), since this is a brand-new modification to a file the
prior stack didn't touch.

**webllm working tree:**
- `src/wasm/CMakeLists.txt`: +6/-1 (the `_ggml_cpu_set_weight_hash_probe`
  export and an explanatory comment).
- `smoke-test/p2-v2-spike.src.ts`: +134/-0 (CPU probe arming +
  unification block + 4-way match logic + P-16 outcome
  synthesis).

## Stage 4.30 — queued probe

**Goal.** Either close suspect 2 directly (by reading the bytes
the CPU op reads for `ffn_norm.weight` post-load) or pivot to
suspect 3 (first8-window blindness on `kqv_out-0`).

**Why two paths.** The mmap-direct framing makes suspect 2 unlikely
on physical grounds, but the only way to actually CLOSE it is to
read those bytes via a different pathway than `set_tensor`. The
cheapest such pathway is a one-shot `tensor->data` peek
post-`loadModel`: walk the model's layer-0 `ffn_norm.weight` ggml
tensor, FNV-1a-32 hash `nbytes` bytes from `tensor->data`, compare
to the GGUF ref hash. If the bytes match, suspect 2 is dead by
direct measurement. If they don't, the gain-vector mis-load is
confirmed and we have a different bug than mmap-direct (something
copies into a separate buffer with corruption).

### Stage 4.30 paste-and-go bootstrap

```bash
cd /Users/probello/Repos/webllm
git log --oneline -5
#   → <Stage 4.29 TODO closure commit>     docs(TODO): Stage 4.29 closed — queue Stage 4.30 ...
#   → <Stage 4.29 reports commit>          docs(reports): Stage 4.29 closure — Outcome P-16-silent
#   → <Stage 4.29 spike commit>            feat(spike): Stage 4.29 Probe 16 — CPU-side set_tensor hook
#   → dd8c104                              docs(TODO): Stage 4.28 closed — queue Stage 4.29 CPU-side weight-hash probe
#   → fee50e9                              docs(reports): Stage 4.28 closure — Outcome P-15-jsep-bypass

( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → <Stage 4.29 commit on top of 1d1d64f76>   webllm-browser-patches   (patch stack 14)

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
```

### Probe 17 implementation sketch (Shape A: post-load tensor->data peek)

1. **C++ side:** add a small export
   `webllm_get_tensor_data_hash(const char * tensor_name, size_t * out_size)`
   that walks `model->tensors_by_name`, finds the named tensor,
   FNV-1a-32-hashes `ggml_nbytes(t)` bytes from `t->data`, returns
   the hash and writes the size to `*out_size`. Lives in
   `src/wasm/webgpu-bridge.cpp` (alongside the other `webllm_*`
   exports). Add `_webllm_get_tensor_data_hash` to
   `EXPORTED_FUNCTIONS`.
2. **JS spike side:** post-`loadModel`, call the export for each of
   the 7 layer-0 weights:
   ```js
   const sizeOut = mod._malloc(8);
   const hash = mod._webllm_get_tensor_data_hash(stringToNewUTF8(name), sizeOut);
   const size = mod.HEAPU32[sizeOut >> 2];
   mod._free(sizeOut);
   ```
3. **Verdict:** compare each hash against
   `__weightHashRef[name].fnv1a`. Output a unified
   `[STAGE-4.30] <name> ref=0x... data_peek=0x... size_data=N
   size_ref=M match=<bool>` line.
4. **Outcomes:**
   - **All 7 match** ⇒ P-17-clean. Suspect 2 is dead by direct
     measurement. Pivot to suspect 3.
   - **`ffn_norm.weight` mismatches** ⇒ P-17-gain. Gain-vector
     mis-load CONFIRMED (a copy somewhere is corrupting the bytes
     between mmap and CPU-op-read). Trace the upload byte trajectory.
   - **`ffn_down.weight` mismatches** ⇒ P-17-ffn-down. Q6_K
     upload path bug. Deep-dive Q6_K upload.
   - **One of the 5 JSEP-resident weights mismatches** ⇒ P-17-jsep-
     deep. JSEP-resident weights' `tensor->data` (which is the
     host_mirror per Stage 4.4 F1) is out of sync with the JSEP
     buffer's GPU-side bytes. Surprising; would re-open suspect 1.

### Files to read first

- [`STAGE-4.20-RESULT.md`](STAGE-4.20-RESULT.md) — JSEP `set_tensor`
  hook pattern; Stage 4.30's `webllm_get_tensor_data_hash` follows
  the same FNV-1a-32 + EM_ASM-push idiom but at a different point.
- [`STAGE-4.28-RESULT.md`](STAGE-4.28-RESULT.md) — refMap source
  (`__weightHashRef` populated by spike's GgufParser block).
- [`STAGE-4.29-RESULT.md`](STAGE-4.29-RESULT.md) — this report;
  P-16-silent classification + mmap-direct framing.
- `~/Repos/llama.cpp/ggml/src/ggml-backend.cpp:2237` — current
  hook site. Stage 4.30's tensor-data peek is *complementary* to
  this hook, not a replacement.
- `src/wasm/webgpu-bridge.cpp` — likely target file for the
  `webllm_get_tensor_data_hash` export.

### Risk register

1. **`tensor->data` may be a sentinel pointer for JSEP-resident
   weights.** Stage 4.4's F1 host_mirror fix means JSEP-resident
   tensors have `tensor->data` pointing into the host mirror (a
   separate allocation that's kept in sync with the JSEP GPU buffer
   via dual-write in `ggml_backend_jsep_buffer_set_tensor`). For
   the 5 JSEP-resident weights, the post-load peek should match
   ref — but if it doesn't, the host_mirror is out of sync, which
   would be a separate bug worth knowing about.
2. **Q6_K weight may be quant-converted in flight.** If libllama's
   model loader converts Q6_K to a different runtime format on
   load (e.g., for the Q6_K → Q4_0 row-based dequant path), the
   post-load `tensor->data` bytes won't match the GGUF Q6_K bytes.
   Diagnosis: check `tensor->type` post-load and compare against
   the GGUF metadata's tensor type. Same-type ⇒ direct mmap;
   different-type ⇒ in-flight conversion (and Stage 4.30's hash
   comparison would need a post-conversion reference, e.g. by
   dequantizing the GGUF Q6_K and re-hashing in the JS side at
   the runtime type).
3. **Even if all 7 hashes match,** the cascade source isn't
   localized — only suspect 2 closes. Stage 4.31 would then need
   to widen the kqv_out-0 first8-window blindness check or
   probe the 5.24e-4 disagreement at Qcur-0's INPUTS (`attn_norm-0`
   output bytes the JSEP MUL_MAT actually reads).

### Exit criteria — Stage 4.30 closes when documented in `STAGE-4.30-RESULT.md`

- All 7 layer-0 weight FNV-1a-32 hashes captured via `tensor->data`
  peek (in addition to the JSEP set_tensor + GPU readback + GgufParser
  reference hashes already captured).
- The `ffn_norm.weight` byte-integrity question answered by direct
  measurement:
  - **CLEAN** (P-17-clean): suspect 2 dead. Stage 4.31 pivots to
    suspect 3 / first8-window blindness on `kqv_out-0`.
  - **DIRTY** (P-17-gain): gain-vector mis-load CONFIRMED via
    direct measurement. Stage 4.31 traces the upload byte
    trajectory.
- Stage 4.31 paste-and-go brief queued for the chosen branch.

### Branch on Probe 17 outcome

- **All 7 match** (Outcome P-17-clean): suspect 2 dies by direct
  measurement. Stage 4.31 widens `node_dump_cb` to capture
  full-tensor stats on `kqv_out-0` (suspect 3) OR probes the
  upstream cascade source by hashing `attn_norm-0` output bytes
  the JSEP MUL_MAT reads (the actual Q-projection `src1`).
- **`ffn_norm.weight` mismatches** (Outcome P-17-gain): the
  gain-vector mis-load CONFIRMED. Stage 4.31 traces the buggy
  upload byte trajectory back to where mmap → CPU-op-read picks
  up corruption.
- **`ffn_down.weight` mismatches** (Outcome P-17-ffn-down): Q6_K
  upload path is the suspect. Stage 4.31 deep-dives the Q6_K
  upload + dispatch path.
- **A JSEP-resident weight mismatches** (Outcome P-17-jsep-deep):
  Stage 4.4's host_mirror is out of sync with the JSEP GPU
  buffer. Re-opens suspect 1 via a different mechanism.
