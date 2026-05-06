# P2-v2 Phase 2 Follow-on Spike — Task 8 Results

**Date:** 2026-05-06
**webllm SHA at run:** `4872307` (predecessor — this commit adds the spike)
**llama.cpp `webllm-browser-patches` SHA:** `48acb658d` (unchanged from Phase 2 closure)
**Browser + GPU:** Chrome (agentchrome session 64702), Apple Metal-3
**Model:** `tinyllama-1.1b-chat-q4_0.gguf` (637.8 MiB)
**Prompt:** 6 token IDs `[1, 450, 7483, 310, 3444, 338]` ("The capital of France is")

## Disposition: BLOCKED on scheduler buft routing (NOT dormant)

The spike harness `smoke-test/p2-v2-spike.html` (Tasks 8-1 + 8-2 runs)
loaded the JSEP-built WASM, installed JSEP callbacks, and called
`webllm_load_model`. Model load **succeeded** with weights placed in
the JSEP buffer:

```
load_tensors:     jsep_buf model buffer size =   455.06 MiB
     model loaded in 336 ms; vocab = 32000
```

Both backends were registered:

```
llama_context: backend_ptrs.size() = 2
```

KV cache assigned to JSEP for all 22 layers:

```
llama_kv_cache: layer  0..21: dev = JSEP
llama_kv_cache:   jsep_buf KV buffer size =   11.00 MiB
```

But `llama_new_context_with_model` aborted during graph reservation:

```
sched_reserve: reserving full memory module
sched_reserve: worst-case: n_tokens = 512, n_seqs = 1, n_outputs = 1
graph_reserve: reserving a graph for ubatch with n_tokens = 1, ...
/Users/probello/Repos/llama.cpp/ggml/src/ggml-backend.cpp:898:
  pre-allocated tensor (blk.0.attn_q.weight) in a buffer (jsep_buf)
  that cannot run the operation (NONE)
Aborted()
```

`createContext` aborted before any `llama_decode`, so no per-token
counters were captured. The 5-token greedy decode was never reached.

## Root cause

`ggml-jsep`'s `device_supports_op` (CLAUDE.md "per-binding 128 MiB
cap doctrine" + Phase 2 dtype matrix) returns true only for:

- `GGML_OP_MUL_MAT` with src1=F32, src0 ∈ {F32, F16, Q4_0, Q4_K}, dst=F32
- `GGML_OP_RMS_NORM` with src0=F32, dst=F32

Everything else returns false. But `llama-context.cpp::sched_reserve`
(via ggml-backend.cpp:898) walks every leaf tensor allocated in
`jsep_buf` and asserts that the **buffer's backend** supports the
consumer op. For tinyllama's first-layer ggml graph, `blk.0.attn_q.weight`
is read by an op that JSEP doesn't claim — likely a transpose / view /
permute or the matmul-with-non-F32-dst path — and the assertion fires
*before* the scheduler ever calls `graph_compute`.

This is **architecturally distinct from Task 7's "dormant" finding.**
Task 7 reported the JSEP backend as registered-but-never-routed-to.
The spike proves the JSEP backend is *fully engaged* for buffer
allocation (model + KV both land in `jsep_buf`) but the scheduler's
**reserve-time op-coverage check** rejects the partition before any
op can run. The Phase 2 dispatch table is too narrow for libllama's
graph topology — even tinyllama, the smallest model in tree, exercises
ops outside MUL_MAT + RMS_NORM that touch JSEP-resident weights.

## Counter snapshot

Counters at the abort moment were never logged because the abort fires
synchronously inside `webllm_create_context`, before the `counters@load`
print statement runs. Inferred from the model-load console (which
*completes* successfully):

| Callback | Count at abort | Source |
|---|---|---|
| `jsepAlloc` | ~25-50 (model + KV buffer-type allocations) | inferred from `jsep_buf model buffer size = 455.06 MiB` and per-layer KV |
| `jsepWrite` | ~200+ (one per weight tensor upload) | inferred from `create_tensor: loading tensor blk.{0..21}.*` console flood |
| `jsepRunOp` | **0** | abort precedes any op dispatch |
| `jsepRead` | 0 | no readback before abort |
| `jsepClear` | 0 | no clear before abort |
| `jsepSync` | 0 | no flush before abort |

To capture exact numbers, the spike harness would need a `try/catch`
around `bridge.createContext` that snapshots `module.__jsep.counters`
in the catch block. Adding that is a Task 8.1 follow-up if the diagnostic
warrants it; the qualitative finding ("alloc/write fired, runOp never did")
is sufficient to drive the Phase 3 redesign.

## Gate metrics — N/A (decode never ran)

| Metric | Legacy ref | Measured | Band |
|---|---|---|---|
| Per-token wall (5-token median) | 5.6 ms | **N/A** — decode never reached | BLOCKED |
| EM_ASM crossings/token (decode-only) | n/a | **N/A** | BLOCKED |
| Greedy 5/5 token equality | "I'm not" | **N/A** | BLOCKED |

## Files

- `smoke-test/p2-v2-spike.src.ts` (this commit) — spike harness source
- `smoke-test/p2-v2-spike.html` (this commit) — spike harness page
- `smoke-test/p2-v2-spike.js` (build artifact) — bundled spike
- `eval/reports/p2-v2-prototype-2026-05-05/spike-console-task8.log` — full console capture from Task 8-2 run

## Implications for next session

The Task 7 closure's framing ("dormant — backend never invoked")
**was incorrect** — the backend *is* invoked at allocation time, but
the scheduler's reserve-time op-coverage check rejects the partition
before any compute runs.

The fix is **not** the originally-planned decode-path swap (the spike
harness already drives `webllm_decode` directly and hits the same
abort). The fix is one of:

- **Option A — broaden `supports_op`.** Add stub-passthrough handling
  for the ops libllama touches on JSEP-resident leaves: GET_ROWS,
  TRANSPOSE, VIEW, PERMUTE, RESHAPE, plus the matmul dtype permutations
  the current narrow matrix excludes. Any op the scheduler reserves
  but the dispatch table can't run could fall back via a `jsepRunOp`
  return code that triggers a CPU copy. Scope: ggml-jsep.cpp +
  CommandEncoderBatcher + Phase 2 dispatch table widening.
- **Option B — narrow weight residency.** Force model weights to
  ggml-webgpu (or CPU), only let JSEP own *intermediate* tensors,
  and use `offload_op` to pull MUL_MAT + RMS_NORM operands into
  jsep_buf transiently. Scope: tensor-buffer-type assignment in
  llama-context init.
- **Option C — abandon multi-backend.** Replace ggml-webgpu entirely
  with JSEP and port every op libllama needs (the natural endpoint
  of Phase 3 anyway). Scope: ~30+ kernels.

Option A is the smallest delta to unblock Phase 2 metrics. Option B
requires understanding llama.cpp's buft routing contracts. Option C is
the long-term endpoint.

## Reproducibility

```bash
make wasm-build-jsep
# (smoke server on :8031 already running)
agentchrome --port 64702 navigate --tab <ID> "http://localhost:8031/p2-v2-spike.html?v=task8-N"
# Wait until #log contains "DONE" or "FAIL".
agentchrome --port 64702 console read --tab <ID> --limit 5000
```
