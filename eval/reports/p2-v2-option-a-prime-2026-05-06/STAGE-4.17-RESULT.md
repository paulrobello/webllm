# Stage 4.17 — Probe 7 (per-node first8 reference-diff): kernel-precision divergence

**Status:** PARTIAL CLOSED 2026-05-07. **Outcome B** confirmed: JSEP's
matmul produces consistent ~1e-4 — 5e-4 first8 deltas vs the non-JSEP
reference at production shapes; compounded across 22 layers, the
divergence amplifies to ±6 magnitude at `result_output` (logits) and
flips the top-1 token from id 3681 (" Paris") to id 297 ("in").
Localization is sharp enough to queue Stage 4.18 against the matmul
kernel + RMSNorm cascade rather than the cross-backend writeback layer.

**Patch stack:** 12 (unchanged from Stage 4.16).
**webllm:** +2 commits pending — cb_eval bridge hook + ref-probe harness.

## Headline

`p2-v2-spike.html` (JSEP Option A-prime) and `p2-v2-ref-probe.html`
(production non-JSEP `webllm-wasm.js`) ran the same TinyLlama Q4_0 GGUF,
same prompt token IDs `[1,450,7483,310,3444,338]`, same greedy
5-decode. Both armed `webllm_enable_node_dump(200)` against the new
`cb_eval` callback in `src/wasm/webgpu-bridge.cpp` (zero llama.cpp
patches — uses `llama_context_params::cb_eval` which already exists in
the public API). Both captured 96 `[CHECKPOINT]` lines spanning
{`Qcur-0`, `Kcur-0`, `Vcur-0`, `kq-0`, `kq_soft_max-0`, `kqv_out-0`,
`attn_out-0`, `ffn_norm-0`, `ffn_out-0`, `result_norm`,
`result_output`} × (1 prefill + 5 decode forward passes).

| Side | Top-1 id (step 0) | Top-1 logit | Generated text |
|---|---|---|---|
| **Reference (non-JSEP)** | 3681 (" Paris") | 13.04 | (correct) |
| **JSEP** | 297 ("in") | 10.46 | "inonic boso-" |

Per-token wall: 155.6 ms (ref) vs 474 ms (JSEP, post-Stage-4.16 H1
await). Reference uses production ggml-webgpu kernels; JSEP uses the
Stage-3 Q4_0/Q4_K WGSL kernels in `src/inference/jsep/ops/matmul.ts`.

## Smoking-gun table (prefill forward pass; full data in `STAGE-4.17-jsep-checkpoints.txt` / `STAGE-4.17-ref-checkpoints.txt`)

| idx | name | shape | max-abs-Δ first8 | comment |
|----:|------|-------|-----------------:|---------|
| 0–2 | `Qcur-0` | [2048,6,1,1] reshapes | **5.24e-4** | Q4_0 matmul output dim 2048 — small but real divergence at the very first compute |
| 3–4 | `Vcur-0` | [256,6] | **0.00** | bit-identical (anomaly — investigate Stage 4.18) |
| 5–7 | `Kcur-0` | [256,6] | **3.38e-4** | same kernel as V; non-zero diff |
| 8 | `kq-0` | [256,6,32] | 1.19e-2 | Q×K^T attention compute — first8 captures position-0 column with causal mask, so this delta is restricted to the un-masked diagonal element |
| 9 | `kq_soft_max-0` | [256,6,32] | **0.00** | first8 is `[1.0, 0, 0, …]` (softmax of a single un-masked entry); identical by construction |
| 10 | `kqv_out-0` | [2048,6] | **0.00** | softmax × V; first8 = V[position 0] which is bit-identical from idx 3-4 |
| **11** | **`attn_out-0`** | **[2048,6]** | **4.77e-3** | **first non-zero divergence after a string of zeros**; this is `residual + out_proj × kqv_out_post_permute`. Out-proj matmul shape is [2048,2048] × [2048,6] (same as Q-proj at idx 0). Kernel-precision delta dominates the residual sum. |
| 12 | `ffn_norm-0` | [2048,6] | 1.83e-1 | RMSNorm of attn_out — small absolute attn_out delta becomes large relative ffn_norm delta because RMSNorm scales by 1/√(mean²+ε); the position-0 first8 values are particularly small in magnitude |
| 13 | `ffn_out-0` | [2048,6] | 4.22e-2 | FFN output for layer 0 |
| **14** | **`result_norm`** | **[2048,1]** | **5.83** | post-22-layer RMSNorm — cumulative drift of layer-1 through layer-21 (unmonitored by allowlist) lands here. Catastrophic divergence. |
| **15** | **`result_output`** | **[32000,1]** | **6.61** | logits — top-1 flips |

## Findings

1. **Outcome B (kernel-correctness) confirmed.** The very first compute
   node where JSEP's data flow lands (`Qcur-0` at idx 0) already shows
   a 5.24e-4 first8 delta from the reference. This is below the 1e-3
   "structural" threshold but it's not zero — and it compounds. The
   matmul kernel under Q4_0 produces slightly different numerical
   outputs than ggml-webgpu's WGSL Q4_0 kernel.

2. **`attn_out-0` (idx 11) is the first cross-the-1e-3-threshold
   node.** Up through `kqv_out-0` (idx 10), all checkpoints either
   carry tiny noise (<1e-3) or are bit-identical. The output
   projection matmul (post-attention, before residual) is the kernel
   call that lands the first structural delta. Same shape as Q-proj
   (Q4_0 [2048,2048] × F32 [2048,6]), so the implication is consistent
   per-shape kernel imprecision.

3. **`Vcur-0` diff = 0.00 is anomalous.** Same shape, same Q4_0
   weights, same input as Kcur-0 (which differs by 3.4e-4). One
   plausible explanation: V-projection runs on CPU under JSEP's
   Option A-prime scheduler split (suggested by Stage 4.13's
   retracted-but-prescient observation that "V's MUL_MAT runs on
   CPU"). If V is the only Q-projection that survives to ground-
   truth precision, its identity is by accident of fallback path,
   not a property of correctness. Stage 4.18 should verify this.

4. **No NaN, no Inf, no all-zero pathology.** Stage 4.16's
   `EM_ASYNC_JS` fix landed correctly. The remaining bug is purely
   numerical: small per-kernel imprecision compounds across 22
   layers. By `result_norm` (post-layer-21), the absolute magnitude
   of the diff is +5.8, and `result_output` (logits) carries +6.6,
   which suffices to flip the argmax token entirely.

5. **kq_soft_max and kqv_out being identical at first8** is a
   first8-window artifact (causal mask masks all but position 0 in
   the first row, so the softmax is `[1, 0, …]` and kqv_out[pos=0]
   is just V[pos=0] which is itself bit-identical). Diff *would* be
   non-zero at later positions; first8 doesn't expose it. Stage 4.18
   should sample multiple positions or capture full-tensor stats
   (mean / max-abs) rather than first8 only.

## Branch decision

Per the Stage 4.17 brief's three-outcome table:

- **Outcome A (English decode):** ❌ — output is "inonic boso-",
  topId=297 ("in"); not a fluent continuation.
- **Outcome B (kernel-correctness localization):** ✅ — confirmed.
  Proceed to Stage 4.18 with kernel imprecision as the working
  hypothesis.
- **Outcome C (cross-backend ordering):** ❌ — refuted. Stage
  4.16's H1 await fix already addressed the ordering layer; the
  current divergence is purely numerical.

## Next: Stage 4.18 brief queued in TODO.md

Per-shape matmul self-test against a numpy / CPU reference dequant
at production shapes, isolating the precision-loss path. Three
sub-probes ranked by cheapest-first:
- (8a) per-shape Q4_0 matmul self-test sweep
- (8b) "is V really on CPU?" trace
- (8c) RMSNorm-of-near-zero-magnitude self-test

## Implementation summary (this stage)

- `src/wasm/webgpu-bridge.cpp` — added `node_dump_cb` (cb_eval
  function), `webllm_enable_node_dump(int)` JS export. ~50 LOC.
  Hooked into `webllm_create_context` via
  `cparams.cb_eval = node_dump_cb`. Allowlist of 11 layer-0 +
  final tensor names baked in.
- `src/wasm/CMakeLists.txt` — added `_webllm_enable_node_dump` to
  EXPORTED_FUNCTIONS.
- `smoke-test/p2-v2-spike.src.ts` — armed dump (`mod._webllm_enable_node_dump(200)`)
  before prefill; appended captured `[CHECKPOINT]` lines to log
  surface; exposed on `window.__stage417Checkpoints`.
- `smoke-test/p2-v2-ref-probe.src.ts` (NEW, ~150 LOC) +
  `p2-v2-ref-probe.html` — minimal non-JSEP reference: loads
  `webllm-wasm.js`, runs identical prefill + greedy 5-decode, captures
  matching checkpoint set on `window.__refCheckpoints`.
- Build: `make wasm-build-wasm32` + `make wasm-build-jsep` (both
  produce binaries with the new export). Wasm32 artifacts copied
  to smoke-test/ manually (Makefile asymmetry — only jsep build
  copies; wasm32 build does not).

## Artifacts

- `STAGE-4.17-jsep-checkpoints.txt` — 96 lines, JSEP spike capture.
- `STAGE-4.17-ref-checkpoints.txt` — 96 lines, non-JSEP reference capture.
- `STAGE-4.17-diff.py` — diff script. Run as
  `python3 STAGE-4.17-diff.py jsep-checkpoints.txt ref-checkpoints.txt`.

## Reproduction

```bash
cd /Users/probello/Repos/webllm
make wasm-build-wasm32
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/   # asymmetry
make wasm-build-jsep
make smoke-serve  # if not running on 8031

# In two browser tabs (reuse existing agentchrome session):
#   http://localhost:8031/p2-v2-ref-probe.html?v=stage4.17-replay
#   http://localhost:8031/p2-v2-spike.html?v=stage4.17-replay
# Wait for "DONE" in both, then:
agentchrome js exec --tab REF "JSON.stringify(window.__refCheckpoints)" > ref.txt
agentchrome js exec --tab JSEP "JSON.stringify(window.__stage417Checkpoints)" > jsep.txt
python3 eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.17-diff.py jsep.txt ref.txt
```
