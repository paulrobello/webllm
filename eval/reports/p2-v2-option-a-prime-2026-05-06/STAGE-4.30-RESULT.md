# Stage 4.30 — Probe 17: post-load `tensor->data` byte-hash peek

**Date:** 2026-05-07
**Probe:** Add a `webllm_get_tensor_data_hash(model_handle, name,
*out_size)` export in `src/wasm/webgpu-bridge.cpp` that resolves a
tensor by name via `llama_internal_get_tensor_map(model)` and
FNV-1a-32-hashes `ggml_nbytes(t)` bytes from `t->data`. Spike harness
walks the same 7-name layer-0 allowlist Stage 4.28/4.29 used and
compares against the JS-side `GgufParser` reference hashes already
in `__weightHashRef`. Closes the byte-integrity question for
`blk.0.ffn_norm.weight` (F32 gain) and `blk.0.ffn_down.weight` (Q6_K)
via direct measurement on the only remaining buft after Stage 4.29
ruled out both JSEP and default-CPU `set_tensor` (the GGUF mmap-
direct host buft).
**Outcome:** **P-17-clean** — all **7/7** layer-0 weights byte-exact
at `tensor->data` post-`loadModel`. Suspect 2 (`ffn_norm.weight`
gain-vector mis-load) is **DEAD by direct measurement.** The cascade
source must lie upstream of the layer-0 weight bytes themselves.

## TL;DR

Unified `[STAGE-4.30]` block (post-load `tensor->data` peek vs JS-side
`GgufParser` reference, same 7-name allowlist as Stages 4.28/4.29):

| Tensor                       |     size  | type | ref          | data_peek    | match |
| ---------------------------- | --------: | ---: | ------------ | ------------ | ----- |
| `blk.0.attn_q.weight`        | 2,359,296 | Q4_0 | `0xf2f7188c` | `0xf2f7188c` | ✅    |
| `blk.0.attn_k.weight`        |   294,912 | Q4_0 | `0x9399f36a` | `0x9399f36a` | ✅    |
| `blk.0.attn_output.weight`   | 2,359,296 | Q4_0 | `0xaae061b5` | `0xaae061b5` | ✅    |
| `blk.0.ffn_norm.weight`      |     8,192 | F32  | `0xcba312e0` | `0xcba312e0` | ✅    |
| `blk.0.ffn_gate.weight`      | 6,488,064 | Q4_0 | `0xafdfc33a` | `0xafdfc33a` | ✅    |
| `blk.0.ffn_up.weight`        | 6,488,064 | Q4_0 | `0x76f44e42` | `0x76f44e42` | ✅    |
| `blk.0.ffn_down.weight`      | 9,461,760 | Q6_K | `0xd1429247` | `0xd1429247` | ✅    |

`[STAGE-4.30] tensor->data peek fired on 7/7 weights (7 byte-clean, 0 dirty)`
`[STAGE-4.30] OUTCOME: P-17-clean (all 7 layer-0 weights byte-exact at tensor->data; suspect 2 gain-vector mis-load DEAD by direct measurement; pivot Stage 4.31 to suspect 3 first8-window blindness on kqv_out-0 OR upstream cascade source at Qcur-0 inputs)`

`size_data == size_ref` for every weight, so the buft is not
quant-converting `ffn_down.weight` in flight (Stage 4.30 risk
register #2 also closed): `tensor->type` is preserved as Q6_K
end-to-end and the bytes seen by the kernel are the GGUF Q6_K
bytes byte-for-byte.

`GENERATED_TEXT` unchanged from Stages 4.27/4.28/4.29 (still
gibberish — confirms the new probe code path doesn't perturb
production decode behavior).

## What this rules out

**Suspect 2 — `ffn_norm.weight` gain-vector mis-load — is DEAD by
direct measurement.** Stage 4.29 made it indirect-evidence dead via
the mmap-direct framing; Stage 4.30 closes it directly. The 8 KiB
F32 gain vector that the layer-0 RMSNorm op reads is bit-identical
to the GGUF tensor data at every byte. The +38× amplification at
`ffn_norm-0` (Stage 4.27's load-bearing signal) is therefore not
caused by the gain bytes themselves — it's a faithful amplification
of an already-corrupted INPUT to the RMSNorm op (or of the JSEP
RMSNorm kernel itself, but Stage 4.27 already showed `attn_norm-0`
— the *other* layer-0 RMSNorm — is bit-identical).

**Suspect 1 (closed Stage 4.28) re-confirmed.** All 5 JSEP-resident
weight `tensor->data` hashes match GGUF refs, which means Stage 4.4
F1's host_mirror is in sync with the JSEP GPU buffer at the time of
the post-load peek. Risk-register item #1 (host_mirror out of sync)
did not fire — no P-17-jsep-deep escalation.

**Q6_K in-flight quant conversion (risk register #2) ruled out.**
`size_data == 9,461,760 == size_ref` for `ffn_down.weight`. If the
loader had converted Q6_K to another runtime format, the byte count
would have changed (e.g., Q6_K → f16 would yield 11,534,336 bytes
for the same element count). The buft is preserving Q6_K bytes
end-to-end.

## What this surfaces

**The cascade source is upstream of the layer-0 weight bytes.** With
all three weight-byte channels confirmed clean (JSEP `set_tensor`
host-side hash, JSEP GPU readback hash, post-load `tensor->data`
peek), the 5.24e-4 production Qcur-0 delta cannot be a weight-load
issue at *any* layer-0 weight. The remaining structural suspects
after Stage 4.30:

1. ❌ **Suspect 1** (`attn_output.weight` byte-integrity gap) —
   closed Stage 4.28; re-confirmed Stage 4.30 (post-load peek).
2. ❌ **Suspect 2** (`ffn_norm.weight` gain-vector mis-load) —
   **CLOSED by Stage 4.30 direct measurement.**
3. ✅ **Suspect 3** (first8-window blindness on `kqv_out-0`)
   — promoted to highest priority for Stage 4.31. Stage 4.27's
   first-divergent-op cascade (`attn_out-0` Δ=4.77e-3) lands on
   the post-attention output. The first8-window readback in the
   `node_dump_cb` instrumentation may be missing a region where
   the actual disagreement lives.
4. **Cascade source upstream of `attn_out-0`.** Stage 4.27's
   `Qcur-0` Δ=5.24e-4 is the first JSEP node to disagree with
   reference. Q-projection's INPUTS — `inp_embd` (post-`get_rows`,
   named `embd` per Stage 4.19) and `attn_norm-0` (RMSNorm output
   feeding the Q matmul as `src1`) — are the next probe targets.
   Stage 4.27 already showed `attn_norm-0` Δ=0 between JSEP and
   reference at the *first8 indices captured by node_dump_cb*, but
   that does not rule out a divergence at indices 8+ that the
   JSEP MUL_MAT reads when it executes Qcur. The Stage 4.31 brief
   below queues both options; the picked one depends on whether
   Stage 4.31's instrumentation finds first8-blindness or upstream
   input-bytes-as-read disagreement.

## What was implemented

### C++ export (`src/wasm/webgpu-bridge.cpp`)

```cpp
// Forward decl outside extern "C" — has C++ linkage.
const std::vector<std::pair<std::string, ggml_tensor *>> &
llama_internal_get_tensor_map(const struct llama_model * model);

extern "C" {
// ...

uint32_t webllm_get_tensor_data_hash(void* model_handle, const char* name,
                                     uint32_t* out_size_ptr) {
    if (out_size_ptr) *out_size_ptr = 0;
    if (!model_handle || !name) return 0;
    const llama_model* model = static_cast<const llama_model*>(model_handle);
    const auto& tensors = llama_internal_get_tensor_map(model);
    for (const auto& kv : tensors) {
        if (kv.first != name) continue;
        const ggml_tensor* t = kv.second;
        if (!t || !t->data) return 0;
        const size_t nbytes = ggml_nbytes(t);
        const uint8_t* bytes = static_cast<const uint8_t*>(t->data);
        uint32_t h = 2166136261u;
        for (size_t i = 0; i < nbytes; ++i) {
            h ^= bytes[i];
            h *= 16777619u;
        }
        if (out_size_ptr) *out_size_ptr = (uint32_t) nbytes;
        return h;
    }
    return 0;
}
}
```

The forward decl avoids pulling in the internal `src/llama-model.h`
header (which would drag in libllama's transitive C++ surface). The
internal symbol is C++-mangled but resolved at link time — same
mechanism as the `llama_internal_*` family used by upstream tools
(`llama-imatrix`, `llama-perplexity`, etc.).

### CMakeLists export (`src/wasm/CMakeLists.txt`)

```diff
   "_webllm_q4k_q8k_matmul,"
+  # Stage 4.30 Probe 17 — post-load tensor->data byte-hash peek.
+  # Lives in webgpu-bridge.cpp (this file's translation unit), so
+  # ships in every build — no WEBLLM_BACKEND_JSEP gating needed.
+  "_webllm_get_tensor_data_hash,"
   # Stage 4.29 Probe 16 — CPU-side set_tensor weight-hash probe.
   "_ggml_cpu_set_weight_hash_probe"
```

Unconditional export, since `webgpu-bridge.cpp` is in every build.

### JS spike harness (`smoke-test/p2-v2-spike.src.ts`)

New top-level `try` block immediately after the Stage 4.29 verdict
block (and before `bridge.createContext(...)`). For each of the
7-name allowlist:

1. `_malloc(lengthBytesUTF8(name)+1)` for the name + `_malloc(4)`
   for the out_size pointer.
2. `stringToUTF8(name, namePtr, nameLen)`.
3. Re-derive `Uint32Array(mod.HEAPU8.buffer)` after each malloc to
   survive heap-grow detachment, then zero the out_size slot.
4. `_webllm_get_tensor_data_hash(model, namePtr, outSizePtr)` →
   returns 32-bit FNV-1a hash.
5. Re-read `Uint32Array[outSizePtr >>> 2]` for the size.
6. `_free` both pointers.
7. Compare to `__weightHashRef[name].fnv1a` and `.size`. Append a
   `[STAGE-4.30] <name> ref=0x... data_peek=0x... size_data=N
   size_ref=M match=<bool>` line.

Synthesizes `P-17-{clean,gain,ffn-down,jsep-deep,other}` outcome.
Stashes verdict on `globalThis.__stage430Lines` /
`__stage430Outcome` for downstream introspection.

### What did not change

- No new patches to llama.cpp (Stage 4.30 is webllm-only — the
  export lives in `src/wasm/webgpu-bridge.cpp`, calls an existing
  llama.cpp internal symbol). Patch stack stays at 14
  (Stage 4.29's CPU-side hook is still HEAD).
- No webllm runtime/inference changes outside the spike harness.
- Stage 4.20/4.28 JSEP probe + Stage 4.29 CPU probe still active;
  Stage 4.30 adds a third independent measurement channel without
  removing the prior two.

## Captured output

Saved at [`STAGE-4.30-spike-output.txt`](STAGE-4.30-spike-output.txt).
The relevant Stage 4.30 lines:

```
     [STAGE-4.30] blk.0.attn_q.weight ref=0xf2f7188c data_peek=0xf2f7188c size_data=2359296 size_ref=2359296 match=true
     [STAGE-4.30] blk.0.attn_k.weight ref=0x9399f36a data_peek=0x9399f36a size_data=294912 size_ref=294912 match=true
     [STAGE-4.30] blk.0.attn_output.weight ref=0xaae061b5 data_peek=0xaae061b5 size_data=2359296 size_ref=2359296 match=true
     [STAGE-4.30] blk.0.ffn_norm.weight ref=0xcba312e0 data_peek=0xcba312e0 size_data=8192 size_ref=8192 match=true
     [STAGE-4.30] blk.0.ffn_gate.weight ref=0xafdfc33a data_peek=0xafdfc33a size_data=6488064 size_ref=6488064 match=true
     [STAGE-4.30] blk.0.ffn_up.weight ref=0x76f44e42 data_peek=0x76f44e42 size_data=6488064 size_ref=6488064 match=true
     [STAGE-4.30] blk.0.ffn_down.weight ref=0xd1429247 data_peek=0xd1429247 size_data=9461760 size_ref=9461760 match=true
     [STAGE-4.30] tensor->data peek fired on 7/7 weights (7 byte-clean, 0 dirty)
     [STAGE-4.30] OUTCOME: P-17-clean (all 7 layer-0 weights byte-exact at tensor->data; suspect 2 gain-vector mis-load DEAD by direct measurement; pivot Stage 4.31 to suspect 3 first8-window blindness on kqv_out-0 OR upstream cascade source at Qcur-0 inputs)
```

Stale earlier outcome lines from probes 9b / Stage 4.28 (e.g.
`[probe9b] OUTCOME: E (hash mismatch — upload corruption)` and
`[STAGE-4.28] OUTCOME: P-15-gain ...`) are artifacts of those
probes treating bypass-buft `<missing>` JSEP set_tensor entries as
mismatches. They are SUPERSEDED by Stage 4.30's direct-measurement
result: every layer-0 weight is byte-clean at the bytes the kernel
actually reads.

## Patch stack delta

**llama.cpp `webllm-browser-patches`:** unchanged — still 14
patches with `ebc7c3d82` (Stage 4.29 CPU-side hook) at HEAD. Stage
4.30 is webllm-only.

**webllm working tree:**
- `src/wasm/webgpu-bridge.cpp`: +47/-1 (forward decl for
  `llama_internal_get_tensor_map` + new `webllm_get_tensor_data_hash`
  export + comment block).
- `src/wasm/CMakeLists.txt`: +5/-1 (the
  `_webllm_get_tensor_data_hash` export entry + explanatory comment).
- `smoke-test/p2-v2-spike.src.ts`: +148/-0 (Stage 4.30 try-block
  with malloc/stringToUTF8/heap-rederive plumbing + 5-outcome
  synthesis).

## Stage 4.31 — queued probe

**Goal.** With suspect 2 closed, the next probe targets the
remaining structural suspects in priority order: suspect 3
(first8-window blindness on `kqv_out-0`) OR the upstream cascade
source at Qcur-0's INPUTS (`embd` / `attn_norm-0` bytes the JSEP
MUL_MAT actually reads at execution time). The cheaper of the two
is suspect 3 — widening `node_dump_cb`'s window from first8 to
full-tensor stats (mean / max / min / non-zero count) on
`kqv_out-0` only requires a code change in the existing callback
site; no new export. Run that first; on negative result (no
divergence at full-tensor scope), pivot to the upstream input-as-
read probe.

### Stage 4.31 paste-and-go bootstrap

```bash
cd /Users/probello/Repos/webllm
git log --oneline -5
#   → <Stage 4.30 TODO closure commit>     docs(TODO): Stage 4.30 closed — queue Stage 4.31 ...
#   → <Stage 4.30 reports commit>          docs(reports): Stage 4.30 closure — Outcome P-17-clean
#   → <Stage 4.30 spike commit>            feat(spike): Stage 4.30 Probe 17 — post-load tensor->data peek
#   → 59b28a2                              docs(TODO): Stage 4.29 closed — queue Stage 4.30 post-load tensor->data peek
#   → d7ae7e5                              docs(reports): Stage 4.29 closure — Outcome P-16-silent

( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → ebc7c3d82   webllm-browser-patches   (patch stack 14 — unchanged from Stage 4.29/4.30)

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
```

### Probe 18 implementation sketch (Shape A: widen `node_dump_cb` to full-tensor stats on `kqv_out-0`)

1. **C++ side (`src/wasm/webgpu-bridge.cpp:node_dump_cb`):** when
   the dumped tensor's name matches `kqv_out-0`, expand the
   readback from the existing first-8-elements window to the full
   tensor. Compute mean / abs-max / abs-min / NaN-count / Inf-count
   over all elements and push them as a single object to
   `globalThis.__stage431Stats[name]` via `EM_ASM`. Keep the
   existing first-8 readback for backwards compatibility with
   Stage 4.27's checkpoint diff.
2. **JS spike side:** after the existing Stage 4.27 checkpoint
   diff block, walk `__stage431Stats["kqv_out-0"]` and compare
   against a JS-side reference (re-run the reference path from
   Stage 4.27's helper or use the `__stage417Checkpoints` cache
   if it's keyed by name). Emit `[STAGE-4.31] kqv_out-0
   reference={mean: X, abs_max: Y, ...} jsep={mean: X', ...}
   delta_mean=... delta_abs_max=...` and synthesize a
   `P-18-{first8-blind, full-clean, full-dirty}` outcome.
3. **Run** at `?v=stage4.31-probe18&ingest=off`.

### Probe 18b implementation sketch (Shape B: hash `attn_norm-0` output bytes the JSEP MUL_MAT reads)

(Run only on Outcome P-18-full-clean — i.e. suspect 3 doesn't
fire even at full-tensor scope.)

1. **C++ side (`~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-jsep.cpp`):**
   in the JSEP `compute_op` for MUL_MAT, when `dst->name == "Qcur-0"`,
   FNV-1a-32-hash `src1->data` bytes (= the `attn_norm-0` output
   feeding Q-projection) and push to a global log via EM_ASM.
2. **JS spike side:** independently compute the JS-side
   `attn_norm-0` output reference via the `__stage417Checkpoints`
   cache or by re-running the reference RMSNorm on the captured
   input. Compare hashes.
3. **Outcomes:**
   - **Match** ⇒ Qcur's INPUTS as the kernel reads them are clean;
     the 5.24e-4 disagreement at Qcur-0 is in the kernel itself.
     Re-open Stages 4.24/4.25/4.26 with a tighter scope.
   - **Mismatch** ⇒ the upstream bug is in the bytes the JSEP
     MUL_MAT actually receives. Trace the path from RMSNorm
     output to MUL_MAT src1.

### Files to read first

- [`STAGE-4.30-RESULT.md`](STAGE-4.30-RESULT.md) — this report;
  P-17-clean classification + suspect-2 closure + remaining-suspect
  prioritization for Stage 4.31.
- [`STAGE-4.27-RESULT.md`](STAGE-4.27-RESULT.md) — Stage 4.17
  cascade re-capture (`attn_out-0` Δ=4.77e-3 first-divergent op);
  reference for Stage 4.31's full-tensor scope expansion.
- [`STAGE-4.17-RESULT.md`](STAGE-4.17-RESULT.md) — original 96-
  checkpoint cascade build-up that named `kqv_out-0` as a
  candidate first8-blind site.
- `src/wasm/webgpu-bridge.cpp:node_dump_cb` (current callback
  site for tensor stats).
- `~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-jsep.cpp:compute_op`
  (Shape B target if Shape A is clean).

### Risk register

1. **`kqv_out-0` may not have first8-blindness — full-tensor
   scope might also be clean.** That's the negative-result branch
   (Outcome P-18-full-clean). Stage 4.31b (Shape B) is the
   queued follow-on for that case; we already know what to do.
2. **`__stage417Checkpoints` may not cache `kqv_out-0`** — Stage
   4.27's checkpoint dictionary is keyed by node name, but the
   capture set is bounded by the cb_eval allowlist. If `kqv_out-0`
   isn't in the allowlist, Stage 4.31 must extend it (cheap; one
   string in `node_dump_cb`).
3. **Reference path divergence drift.** Stages 4.17 / 4.27 used a
   non-JSEP reference build to compute the truth values. Patch
   stack growth from 13 → 14 (Stage 4.29 added the CPU-side hook,
   no kernel change) does NOT affect the reference path, so
   delta values from Stage 4.27 should still be load-bearing for
   Stage 4.31. Sanity-check by re-running Stage 4.27's
   first-divergent-op cascade and confirming `attn_out-0` still
   reads Δ=4.77e-3 before trusting the Stage 4.31 deltas.
4. **The 5.24e-4 at Qcur-0 may be coming from rotary embedding
   inputs we haven't probed.** Q-projection's downstream path
   includes RoPE before reaching `kqv_out-0`. If `kqv_out-0`
   is full-clean and `attn_norm-0` (Q-proj's src1) is clean, the
   next candidate is the RoPE position embeddings (`inp_pos`).
   Stage 4.31c (Shape C) would extend the input-bytes-as-read
   probe to RoPE inputs.

### Exit criteria — Stage 4.31 closes when documented in `STAGE-4.31-RESULT.md`

- `kqv_out-0` full-tensor stats captured for both JSEP and
  reference paths; verdict on whether first8-blindness was
  hiding a real divergence.
- One of:
  - **P-18-first8-blind**: full-tensor delta exceeds 1e-3 while
    first8 was clean. Suspect 3 CONFIRMED. Stage 4.32 traces
    the kernel-output divergence at the disagreement window.
  - **P-18-full-clean**: full-tensor delta within noise (≤1e-5).
    Suspect 3 dies. Run Probe 18b (Shape B). Stage 4.32 picks
    based on Shape B's outcome.
  - **P-18-full-dirty**: full-tensor delta also large but
    spread across the tensor (not localized). Suspect 3 partial.
    Re-frame as kernel-arithmetic disagreement at scale; rerun
    Stage 4.26's libllama-vs-WGSL precision probe at `kqv_out-0`
    inputs.
- Stage 4.32 paste-and-go brief queued for the chosen branch.

### Branch on Probe 18 outcome

- **P-18-first8-blind** (suspect 3 hot): Stage 4.32 captures the
  full-tensor diff at idx-by-idx scale to localize where the
  kernel-output disagreement starts; re-uses Stage 4.27's
  first-divergent-op walk on a sliding window.
- **P-18-full-clean** (suspect 3 dies): Stage 4.32 = Probe 18b
  (Shape B) — hash `attn_norm-0` output bytes the JSEP MUL_MAT
  reads at Qcur execution time.
- **P-18-full-dirty** (kernel arithmetic at scale): Stage 4.32
  re-runs Stage 4.26's WGSL-vs-libllama matmul precision probe
  but at `kqv_out-0` inputs (post-RoPE Q × K^T) instead of
  Q-projection (Q4_K × f32).
