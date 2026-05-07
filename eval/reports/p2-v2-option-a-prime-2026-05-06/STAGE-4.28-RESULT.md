# Stage 4.28 — Probe 15: extended weight byte-integrity probe

**Date:** 2026-05-07
**Probe:** Extended Stage 4.20/4.21's pre-upload + GPU-readback FNV-1a-32
hash pattern from 2 layer-0 weights (`attn_q.weight`, `attn_k.weight`)
to 7 layer-0 weights, adding `attn_output.weight`, `ffn_norm.weight`,
`ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`. Each weight is
hashed inside `ggml_backend_jsep_buffer_set_tensor` (pre-upload, host
side), against a JS-side `GgufParser` reference parsed independently
from the GGUF buffer, **and** against a post-`device.queue.writeBuffer`
GPU readback via `copyBufferToBuffer` + `mapAsync`.
**Outcome:** **P-15-jsep-bypass** (5 weights byte-clean end-to-end through
JSEP; 2 weights — `ffn_norm.weight` and `ffn_down.weight` — are not
JSEP-buffer-resident at all and bypass the JSEP `set_tensor` hook
entirely. The original P-15-gain framing was wrong because the
gain-vector mis-load hypothesis is silent on the JSEP path; it has to
be tested on the CPU buft path.)

## TL;DR

| Tensor                        | size      | type | ref (GgufParser) | set_tensor pre-upload | GPU readback | match |
| ----------------------------- | --------: | ---: | ---------------- | --------------------- | ------------ | ----- |
| `blk.0.attn_q.weight`         | 2,359,296 |  Q4_0 | `0xf2f7188c`     | `0xf2f7188c`          | `0xf2f7188c` | ✅    |
| `blk.0.attn_k.weight`         |   294,912 |  Q4_0 | `0x9399f36a`     | `0x9399f36a`          | `0x9399f36a` | ✅    |
| `blk.0.attn_output.weight`    | 2,359,296 |  Q4_0 | `0xaae061b5`     | `0xaae061b5`          | `0xaae061b5` | ✅    |
| `blk.0.ffn_norm.weight`       |     8,192 |  F32  | `0xcba312e0`     | **(no JSEP fire)**    | **(n/a)**    | ⚠️ untestable |
| `blk.0.ffn_gate.weight`       | 6,488,064 |  Q4_0 | `0xafdfc33a`     | `0xafdfc33a`          | `0xafdfc33a` | ✅    |
| `blk.0.ffn_up.weight`         | 6,488,064 |  Q4_0 | `0x76f44e42`     | `0x76f44e42`          | `0x76f44e42` | ✅    |
| `blk.0.ffn_down.weight`       | 9,461,760 |  Q6_K | `0xd1429247`     | **(no JSEP fire)**    | **(n/a)**    | ⚠️ untestable |

(Type column derived from the implied bytes-per-element ratio:
`8192 / 2048 = 4.0` ⇒ F32; `9461760 / 11534336 ≈ 0.820 = 210/256` ⇒ Q6_K;
`2359296 / 4194304 = 0.5625 = 18/32` ⇒ Q4_0; etc.)

`GENERATED_TEXT = "inonic boso-"` (gibberish, unchanged from Stage 4.27).

## What this rules out

**Suspect 1 from Stage 4.27 (`attn_output.weight` byte-integrity gap)
is dead.** The output-projection weight is byte-identical end-to-end
through the GGUF parser → `set_tensor` → `Module.jsepWrite` →
`device.queue.writeBuffer` chain on its JSEP-resident layer-0 instance.
The 4.77e-3 Δ at `attn_out-0` cannot come from a corrupted
`attn_output.weight` upload.

**Three other weights also clear:** `ffn_gate.weight`, `ffn_up.weight`,
and (re-confirmed) `attn_q.weight` + `attn_k.weight`. So all five JSEP-
resident layer-0 weights are byte-clean.

## What this surfaces

**Suspect 2 (`ffn_norm.weight` gain-vector mis-load) cannot be tested
by the JSEP `set_tensor` hook.** The C++ probe is named-gated to fire
inside `ggml_backend_jsep_buffer_set_tensor`, which only runs for
tensors whose `buffer_type` is JSEP. The probe captured zero hits for
`blk.0.ffn_norm.weight` — meaning libllama allocated this tensor to a
different backend's buffer (almost certainly the CPU/host buft, which
makes sense given Stage 4.27's smoking-gun table shows `ffn_norm-0`
running on `backend=CPU`).

**`blk.0.ffn_down.weight` also bypasses JSEP `set_tensor`.** Its size
implies Q6_K (210/256 bytes per element), which is consistent with
Stage 4.22's surprise finding that "TinyLlama-1.1b-chat-q4_0.gguf
actually contains Q4_K projections + Q6_K embeddings." The Q6_K type
likely routes to a different backend at scheduler-allocation time —
or, more interestingly, to a CPU op even though the tensor lives in
the FFN matmul chain.

The remaining two structural suspects from Stage 4.27 therefore split
into:

1. **Suspect 2-a (CPU-buft `ffn_norm.weight` byte-integrity).** The
   CPU buffer's bytes for the gain vector are still untested. If those
   bytes diverge from the GGUF reference, the +38× amplification at
   `ffn_norm-0` is mathematically natural. **Requires a CPU-side
   set_tensor hook (or the brief's risk-register fallback: a no-op
   `ggml_view + ggml_dup` weight-tap via `cb_eval`).**
2. **Suspect 2-b (RMSNorm kernel arithmetic precision on CPU).** If
   the gain vector is byte-clean on CPU buft, the +38× amplification
   then has to come from the CPU RMSNorm kernel doing something
   different from the reference path. Stage 4.27 already showed
   `attn_norm-0` (the *other* RMSNorm output, on CPU) is bit-identical,
   so the kernel works correctly when its input is clean. The
   discriminator is whether the *input* to `ffn_norm-0` (= residual +
   `attn_out-0`) lands on the CPU op with bit-clean bytes or with
   the 4.77e-3 Δ already baked in.
3. **Suspect 3 (first8-window blindness on `kqv_out-0`)** — unchanged.

## What was implemented

### C++ allowlist extension (`~/Repos/llama.cpp/ggml/src/ggml-jsep/ggml-jsep.cpp`)

Stage 4.20's two-name allowlist (`blk.0.attn_q.weight`,
`blk.0.attn_k.weight`) extended to seven names with five additional
`std::strcmp` clauses:

```cpp
const bool match = (std::strcmp(tensor->name, "blk.0.attn_q.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.attn_k.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.attn_output.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.ffn_norm.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.ffn_gate.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.ffn_up.weight") == 0 ||
                    std::strcmp(tensor->name, "blk.0.ffn_down.weight") == 0);
```

No new exports, no new emjs callbacks, no new buffer-usage flag changes
— pure additive change to the existing allowlist branch.

### JS spike harness (`smoke-test/p2-v2-spike.src.ts`)

- `targetNames` extended from 2 to 7 to match the C++ list.
- `elemBytes()` extended from 5 to 6 cases (added Q6_K = 210/256 bytes
  per element to support `blk.0.ffn_down.weight`).
- New Stage 4.28 verdict block emits per-weight
  `[STAGE-4.28] <name> ref_hash=<H1> gpu_readback_hash=<H2> match=<bool>`
  lines and a synthesizing `[STAGE-4.28] OUTCOME: P-15-*` line:
  - `P-15-clean` — all 7 byte-exact (would close suspects 1 + 2; pivot
    to suspect 3).
  - `P-15-gain` — `ffn_norm.weight` is the first JSEP mismatch.
  - `P-15-output-proj` — `attn_output.weight` is the first JSEP
    mismatch.
  - `P-15-ffn` — one of `ffn_{gate,up,down}` is the first JSEP
    mismatch.

### What did not change

- No new patches to llama.cpp beyond the allowlist extension (still
  patch stack 13).
- No CMakeLists changes (the existing
  `_ggml_jsep_set_weight_hash_probe` export covers the toggle; the
  allowlist is internal C++ logic).
- No webllm runtime/inference changes outside the spike harness.

## Captured output

Saved at [`STAGE-4.28-spike-output.txt`](STAGE-4.28-spike-output.txt).
The relevant lines:

```
     [probe9b] weight-hash probe armed
     model loaded in 461 ms; vocab = 32000
     [probe9b] blk.0.attn_q.weight: pre=0xf2f7188c ref=0xf2f7188c size_pre=2359296 size_ref=2359296 match=true
     [probe9b] blk.0.attn_k.weight: pre=0x9399f36a ref=0x9399f36a size_pre=294912 size_ref=294912 match=true
     [probe9b] blk.0.attn_output.weight: pre=0xaae061b5 ref=0xaae061b5 size_pre=2359296 size_ref=2359296 match=true
     [probe9b] blk.0.ffn_norm.weight: pre=<missing> ref=0xcba312e0 size_pre=-1 size_ref=8192 match=false
     [probe9b] blk.0.ffn_gate.weight: pre=0xafdfc33a ref=0xafdfc33a size_pre=6488064 size_ref=6488064 match=true
     [probe9b] blk.0.ffn_up.weight: pre=0x76f44e42 ref=0x76f44e42 size_pre=6488064 size_ref=6488064 match=true
     [probe9b] blk.0.ffn_down.weight: pre=<missing> ref=0xd1429247 size_pre=-1 size_ref=9461760 match=false
     [probe9b] OUTCOME: E (hash mismatch — upload corruption)   ← misleading; "missing" not "corrupt"
     [probe9c] blk.0.attn_q.weight: pre=0xf2f7188c gpu=0xf2f7188c size=2359296 match=true
     [probe9c] blk.0.attn_k.weight: pre=0x9399f36a gpu=0x9399f36a size=294912 match=true
     [probe9c] blk.0.attn_output.weight: pre=0xaae061b5 gpu=0xaae061b5 size=2359296 match=true
     [probe9c] blk.0.ffn_gate.weight: pre=0xafdfc33a gpu=0xafdfc33a size=6488064 match=true
     [probe9c] blk.0.ffn_up.weight: pre=0x76f44e42 gpu=0x76f44e42 size=6488064 match=true
     [probe9c] OUTCOME: F-1 (GPU bytes match — upload chain bit-clean; kernel re-investigation)
     [STAGE-4.28] blk.0.attn_q.weight     ref_hash=0xf2f7188c gpu_readback_hash=0xf2f7188c match=true
     [STAGE-4.28] blk.0.attn_k.weight     ref_hash=0x9399f36a gpu_readback_hash=0x9399f36a match=true
     [STAGE-4.28] blk.0.attn_output.weight ref_hash=0xaae061b5 gpu_readback_hash=0xaae061b5 match=true
     [STAGE-4.28] blk.0.ffn_norm.weight   ref_hash=0xcba312e0 gpu_readback_hash=<missing>   match=false
     [STAGE-4.28] blk.0.ffn_gate.weight   ref_hash=0xafdfc33a gpu_readback_hash=0xafdfc33a match=true
     [STAGE-4.28] blk.0.ffn_up.weight     ref_hash=0x76f44e42 gpu_readback_hash=0x76f44e42 match=true
     [STAGE-4.28] blk.0.ffn_down.weight   ref_hash=0xd1429247 gpu_readback_hash=<missing>   match=false
     [STAGE-4.28] OUTCOME: P-15-gain (ffn_norm.weight gain-vector mis-load CONFIRMED; trace upload path)
```

Note the OUTCOME line in-spike says "P-15-gain" — that label is wrong
in retrospect. "Mis-load CONFIRMED" was the in-spike interpretation of
"hash missing on the JSEP side"; the correct interpretation is "JSEP
never sees this tensor, so the JSEP-side hash machinery is silent."
The reframed outcome is **P-15-jsep-bypass** (this report's title) —
the gain-vector hypothesis remains *open* and needs a CPU-buft probe
to test.

`make checkall` ✅ green (747 pass / 36 skip / 0 fail across 783 tests).

Per-token decode 1287 ms — elevated vs Stage 4.21's baseline because
the spike still carries Probe 14's CPU matmul + Probe 13's Kahan arming
in the post-Probe-9c path. None of those touch the production decode
loop.

## Patch stack delta

**llama.cpp `webllm-browser-patches`:** `ef89f9314` → unchanged in
behavior (the allowlist is a one-line additive change to the same
patch's body, not a new patch). Stack still at 13 patches; this could
be considered a P10 update rather than a P11, depending on how
the rebase tooling counts it.

**webllm working tree:** Stage 4.28 adds ~70 LOC to
`smoke-test/p2-v2-spike.src.ts` (extended `targetNames` + `elemBytes`
+ Stage-4.28 verdict synthesis block) plus this report.

## Stage 4.29 — queued probe

**Goal.** Test suspect 2-a: are the bytes that the CPU op for
`ffn_norm-0` reads from `blk.0.ffn_norm.weight` byte-identical to the
GGUF file's bytes for that tensor?

**Why this matters.** Stage 4.28 closed suspect 1 (attn_output upload)
and proved 5 of 7 layer-0 weights are byte-clean on the JSEP path. The
remaining two weights — `ffn_norm.weight` (most likely the cascade
amplifier) and `ffn_down.weight` (Q6_K, an FFN matmul weight) — live
on a different buffer type. They need a different probe.

**Two viable probe shapes** (pick one in Stage 4.29):

### Shape A — CPU-side `set_tensor` hook (preferred)

Locate the CPU backend's `set_tensor` callback (likely
`ggml_backend_cpu_buffer_set_tensor` in
`~/Repos/llama.cpp/ggml/src/ggml-backend.cpp` or the CPU buft impl)
and add a name-gated FNV-1a hook mirroring the JSEP probe. Push to a
new `globalThis.__cpuWeightHashLog`. The JS spike then walks both logs
and synthesizes a unified P-15-{clean,gain,...} verdict that covers all
7 weights uniformly.

**Pros:**
- Zero scheduler changes — the hook just observes whatever ggml's
  allocator chose to do.
- Same FNV-1a-32 plumbing as Stage 4.20/4.28 — minimal new code.

**Cons:**
- Requires touching libllama at a different file (not the JSEP file
  Stage 4.20 already patched).
- The CPU backend might not even be linked in the JSEP-only build —
  check first.

### Shape B — `cb_eval` weight-tap via no-op view

Wrap each suspect weight in a no-op `ggml_view` + `ggml_dup` schedule
at first `llama_decode` entry so the cb_eval allowlist sees it as a
compute output. The brief's risk-register #1 fallback flagged this as
the heavier option.

**Pros:**
- Hooks the weight at the moment it's actually consumed by an op
  (slightly stronger guarantee than upload-time hashing).

**Cons:**
- Schedule-mutation risk: the view/dup might trigger different
  scheduling decisions than the production path.
- Heavier patch surface.

**Recommendation:** Shape A. It mirrors the pattern Stages 4.20/4.21
already validated, and the scheduling-mutation risk in Shape B is a
real hazard given how much of this investigation has hinged on
incidental scheduler-routing details.

### Stage 4.29 paste-and-go bootstrap

```bash
# 1. Confirm working tree (after Stage 4.28 commits land).
cd /Users/probello/Repos/webllm
git log --oneline -5
#   → <Stage 4.28 TODO closure commit>     docs(TODO): Stage 4.28 closed — queue Stage 4.29 CPU-side weight-hash probe
#   → <Stage 4.28 reports commit>          docs(reports): Stage 4.28 closure — Outcome P-15-jsep-bypass
#   → <Stage 4.28 spike commit>            feat(spike): Stage 4.28 Probe 15 — extend weight allowlist 2 → 7
#   → 8500c9b                              docs(TODO): Stage 4.27 closed — queue Stage 4.28 weight byte-integrity probe
#   → f5438ae                              docs(reports): Stage 4.27 closure — Outcome A (cascade structurally identical to Stage 4.17)

( cd ~/Repos/llama.cpp && git rev-parse --short HEAD && git rev-parse --abbrev-ref HEAD )
#   → <Stage 4.28 commit on top of ef89f9314>   webllm-browser-patches   (patch stack 13 — Stage 4.28 amends P10 in place rather than adding a new patch)

lsof -nP -iTCP:8031 -sTCP:LISTEN | head -2 || make smoke-serve &
PORT=$(agentchrome connect --status | python3 -c 'import json,sys;print(json.load(sys.stdin)["port"])')
SPIKE_TAB=$(agentchrome --port "$PORT" tabs list | python3 -c 'import json,sys;print(next((t["id"] for t in json.load(sys.stdin) if "p2-v2-spike.html" in t.get("url","")), ""))' 2>/dev/null)
[ -n "$SPIKE_TAB" ] || SPIKE_TAB=$(agentchrome --port "$PORT" tabs create --background "http://localhost:8031/p2-v2-spike.html" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
```

### Probe 16 implementation sketch (Shape A — CPU-side hook)

1. Find the CPU backend's `set_tensor` impl. Likely candidates:
   `~/Repos/llama.cpp/ggml/src/ggml-backend.cpp::ggml_backend_cpu_buffer_set_tensor`
   or `~/Repos/llama.cpp/ggml/src/ggml-cpu/ggml-cpu-buffer.cpp` (location
   varies by upstream tip; search for a function that calls
   `memcpy((char*)buffer->data + offset, data, size)` without any GPU
   bridging).
2. Add a parallel `g_cpu_weight_hash_probe` flag + toggle export
   (`ggml_cpu_set_weight_hash_probe`) mirroring the JSEP probe.
3. Inside the CPU set_tensor, gate on the same 7-name allowlist; on
   match, FNV-1a hash the bytes and push to
   `globalThis.__cpuWeightHashLog`.
4. JS spike: `setCpuWeightHashProbe(1)` before `loadModel`, walk the
   merged log post-load, emit unified `[STAGE-4.29]` lines per weight.
5. Re-run; verify `ffn_norm.weight` and `ffn_down.weight` now appear in
   the combined log (proving they live on CPU buft, not "lost in
   space").

### Risk register

1. **CPU backend may not be the actual buft for these tensors.** If
   the JSEP-only build links a stripped-down CPU buft path, the hook
   may not fire. Diagnosis: log every set_tensor call (untargeted)
   for the first 50 invocations and see which buft sees `ffn_norm.weight`
   and `ffn_down.weight`. The brief's Shape B fallback (cb_eval weight
   tap) is the next escalation if the CPU hook is also silent.
2. **Even if both weights hash byte-clean,** the cascade source isn't
   localized. Stage 4.30 then has to widen the kqv_out-0 first8-window
   blindness check (suspect 3) — extend `node_dump_cb` to capture
   max-abs over the full tensor or sample positions 1-5 explicitly.
3. **Q6_K-specific path.** `ffn_down.weight` is Q6_K — TinyLlama is
   reportedly Q4_K projections + Q6_K embeddings (Stage 4.22 surprise
   finding). The Q6_K dequant + matmul might run on CPU because
   ggml-webgpu doesn't implement a Q6_K kernel. If that's the
   structural cause of `ffn_norm-0`'s amplification (Q6_K → CPU →
   slightly different arithmetic than the reference path's Q6_K →
   different-CPU-impl), it would be a much bigger hypothesis than a
   weight mis-load. Stage 4.30 may need to re-test this against a
   pure-Q4_0 model to disambiguate.

### Exit criteria — Stage 4.29 closes when documented in `STAGE-4.29-RESULT.md`

- All 7 layer-0 weight FNV-1a hashes captured (5 via JSEP set_tensor +
  2 via CPU set_tensor) and compared against JS-side `GgufParser`
  reference.
- The `ffn_norm.weight` byte-integrity question answered:
  - **CLEAN**: gain-vector mis-load (suspect 2) is dead. Pivot Stage
    4.30 to suspect 3 (first8-window blindness on `kqv_out-0`).
  - **DIRTY**: gain-vector mis-load CONFIRMED. Stage 4.30 traces the
    buggy upload byte trajectory.
- Stage 4.30 paste-and-go brief queued for the chosen branch.
