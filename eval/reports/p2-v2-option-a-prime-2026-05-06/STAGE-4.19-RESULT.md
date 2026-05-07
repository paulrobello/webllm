# Stage 4.19 — Probe 9a: extend cb_eval allowlist with `attn_norm-0`, `inp_embd`, `l_out-0`

**Status:** CLOSED 2026-05-07. **Outcome: Branch 2** of the Stage 4.18 brief —
`attn_norm-0` (Q-projection's src1) is **bit-identical** between the JSEP
spike and the wasm32 reference (max_abs_delta = 0.000000), while `Qcur-0`
(Q-projection's MUL_MAT output, the very next captured node) differs by
**5.24e-4** on the very same prefill step. Combined with Stage 4.18's
exoneration of the kernel itself (≤1.68e-6 abs Δ vs f32-loop reference at
the (2048, 2048, 6) Q-projection shape), this localizes the source of the
production-prefill delta to **Q-projection's src0 — the Q4_0 weight bytes
as uploaded into the JSEP buffer**. Stage 4.20 queues Probe 9b: a
byte-hash check of pre-upload GGUF parse output vs post-upload JSEP buffer
contents for layer-0 wq/wk weights.

**Patch stack:** 12 (unchanged).
**webllm:** +2 commits pending —
- `feat(jsep)`: extend `NODE_DUMP_ALLOWLIST` with upstream-of-Q-projection nodes
- `docs(reports)`: this file + Stage 4.19 artifacts

## Headline

| Probe | Finding |
|---|---|
| **9a** (extend allowlist with `attn_norm-0` / `inp_embd` / `l_out-0`) | `attn_norm-0` runs on CPU on **both** sides and is bit-identical (Δ=0.000000); `Qcur-0` immediately after Q-projection differs by **5.24e-4**. Branch-2 outcome confirmed. `inp_embd` produced no checkpoint (it is a leaf input tensor, not a compute-op output — see "Note on `inp_embd`" below). |

## Smoking-gun diff — first 10 prefill checkpoints

From `STAGE-4.19-diff-output.txt`:

```
 idx name              ne                   jsep_be    max_abs_delta  jsep_first  ref_first
   0 attn_norm-0       [2048, 6, 1, 1]      CPU             0.000000    0.001405    0.001405
   1 Qcur-0            [2048, 6, 1, 1]      jsep_buf        0.000524   -0.016189   -0.016286
   2 Qcur-0            [64, 32, 6, 1]       jsep_buf        0.000524   -0.016189   -0.016286
   3 Qcur-0            [64, 32, 6, 1]       CPU             0.000524   -0.016189   -0.016286
   4 Vcur-0            [256, 6, 1, 1]       CPU             0.000000   -0.000799   -0.000799
   5 Vcur-0            [64, 4, 6, 1]        CPU             0.000000   -0.000799   -0.000799
   6 Kcur-0            [256, 6, 1, 1]       jsep_buf        0.000338    0.017701    0.017835
   7 Kcur-0            [64, 4, 6, 1]        jsep_buf        0.000338    0.017701    0.017835
   8 Kcur-0            [64, 4, 6, 1]        CPU             0.000338    0.017701    0.017835
   9 kq-0              [256, 6, 32, 1]      jsep_buf        0.011940   -0.005891   -0.005892
```

**Read it as:**

- **idx=0 attn_norm-0 Δ=0.000000.** RMSNorm runs on CPU on both sides.
  Same code path, same input embedding bytes ⇒ same output bytes. Confirms
  Stage 4.18 Probe 8b's "ffn_norm-0 routes to CPU" finding generalizes:
  `attn_norm-0` also routes to CPU under Option A-prime.

- **idx=1 Qcur-0 Δ=5.24e-4 on jsep_buf.** Q-projection MUL_MAT is the
  first JSEP-side op in the prefill. With Q-proj's src1 (= `attn_norm-0`)
  bit-identical and Stage 4.18 having ruled out the kernel at this exact
  shape, the only remaining input that could cause a 5.24e-4 delta is
  **src0 = `wq.weight` for layer 0**.

- **idx=4-5 Vcur-0 Δ=0.000000 on CPU.** Confirms Stage 4.18 Probe 8b's
  V-on-CPU routing under Option A-prime — V-projection runs on CPU and
  matches bit-for-bit with the wasm32 reference.

- **idx=6-8 Kcur-0 Δ=3.38e-4 on jsep_buf.** K-projection MUL_MAT (same
  weight-source mechanism as Q-projection but a different weight tensor
  `wk.weight`) shows the **same scale of error** (3.4e-4 vs Q-proj's
  5.2e-4), strongly suggesting both projections suffer from the same
  upload-side defect rather than a Q-specific issue.

- **idx=9 kq-0 Δ=1.19e-2 on jsep_buf.** Q@K^T amplifies the upstream
  Δ_Q + Δ_K by the dot-product accumulation length (here K=64 per head).
  Not an additional bug — error compounds as expected.

After idx=15 (`l_out-0`) the per-token deltas explode (≥0.04 by `result_norm`,
≥6 by `result_output`), as expected once the layer-0 cascade leaks into
the residual stream. Those rows are not separate bugs.

## Branch decision — confirmed

The Stage 4.18 brief named three branches. Probe 9a triggered **Branch 2**
exactly:

- **Branch 1 (attn_norm-0 differs):** REFUTED. Δ=0.000000.
- **Branch 2 (attn_norm-0 bit-identical, Qcur-0 still 5e-4):** CONFIRMED.
  Q-projection's weight upload is the suspect. Probe 9b queued.
- **Branch 3 (everything bit-identical):** REFUTED. Qcur-0 Δ=5.24e-4.

## Note on `inp_embd` — leaf input, no cb_eval

The Stage 4.18 brief asked Probe 9a to add `inp_embd` to the allowlist. It
was added (`src/wasm/webgpu-bridge.cpp::NODE_DUMP_ALLOWLIST`) but produced
**no captured checkpoint** in either spike or ref-probe runs. Reason:
`inp_embd` (llama-graph.cpp:1718-1720) is a leaf tensor created via
`ggml_new_tensor_2d` + `ggml_set_input` — it has no producing op, so
`cb_eval` (which fires per-op during graph compute) never fires for it.

The post-`ggml_get_rows` + `ggml_build_forward_select` output is named
`"embd"` (llama-graph.cpp:1778) and IS a compute-op output. If a future
probe needs to drill upstream of `attn_norm-0`, that's the right name to
add. **Branch 2 outcome makes this drill unnecessary** — Probe 9b inspects
the upload path directly.

The `inp_embd` entry in the allowlist is harmless (no-op) and the comment
above it documents the leaf-input gotcha for future readers.

## Backend tally for the 14 allowlisted names (this run)

```
attn_norm-0    n=6   jsep_be=CPU
attn_out-0     n=6   jsep_be=jsep_buf
ffn_norm-0     n=6   jsep_be=CPU
ffn_out-0      n=6   jsep_be=CPU
Kcur-0         n=18  jsep_be=CPU,jsep_buf
kq-0           n=6   jsep_be=jsep_buf
kq_soft_max-0  n=6   jsep_be=CPU
kqv_out-0      n=6   jsep_be=CPU
l_out-0        n=6   jsep_be=CPU
Qcur-0         n=18  jsep_be=CPU,jsep_buf
result_norm    n=6   jsep_be=CPU
result_output  n=6   jsep_be=CPU
Vcur-0         n=12  jsep_be=CPU
inp_embd       n=0   (leaf — see note above)
```

The CPU-routed nodes match Probe 8b's Option A-prime split. Newly added
`l_out-0` runs on CPU (residual-stream output of layer 0) — the
`l_out-0` Δ at idx=15 (≈0.04) is the cumulative drift of the layer-0
attention + FFN cascade leaking back into the residual stream, not a
new divergence point.

## Artifacts

- `STAGE-4.19-jsep-checkpoints.txt` — 108 CHECKPOINT lines from the JSEP spike.
- `STAGE-4.19-ref-checkpoints.txt` — 108 CHECKPOINT lines from the wasm32 ref-probe.
- `STAGE-4.19-diff-output.txt` — `STAGE-4.18-diff.py` output across both files.
- `src/wasm/webgpu-bridge.cpp::NODE_DUMP_ALLOWLIST` — extended with
  `inp_embd`, `attn_norm-0`, `l_out-0`. The `inp_embd` entry is retained
  as a documented no-op (see comment in source).

## Reproduction

```bash
# State precondition: webllm @ 4980ee3 + the allowlist diff in this commit;
# llama.cpp @ fc376580e on webllm-browser-patches; smoke server up on 8031.
make wasm-build-jsep                                    # cp baked in
make wasm-build-wasm32 \
  && cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/ \
  && bun build smoke-test/p2-v2-ref-probe.src.ts \
       --outfile smoke-test/p2-v2-ref-probe.js --target browser

# In agentchrome (reuse session if available):
agentchrome tabs create --background "http://localhost:8031/p2-v2-spike.html?v=stage4.19a"
agentchrome tabs create --background "http://localhost:8031/p2-v2-ref-probe.html?v=stage4.19a"
agentchrome page wait --tab <SPIKE> --text DONE --timeout 240000
agentchrome page wait --tab <REF>   --text DONE --timeout 240000

# Extract + diff:
agentchrome page text --tab <SPIKE> > /tmp/spike.json
agentchrome page text --tab <REF>   > /tmp/ref.json
SPIKE_FILE=$(jq -r .output_file /tmp/spike.json)
REF_FILE=$(jq -r .output_file /tmp/ref.json)
grep -oE '\[CHECKPOINT idx=[0-9]+ name=[^ ]+ type=[0-9]+ backend=[^ ]+ ne=\[[^]]+\] contig=[01] first8=\[[^]]+\]\]' \
  "$SPIKE_FILE" > eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-jsep-checkpoints.txt
grep -oE '\[CHECKPOINT idx=[0-9]+ name=[^ ]+ type=[0-9]+ backend=[^ ]+ ne=\[[^]]+\] contig=[01] first8=\[[^]]+\]\]' \
  "$REF_FILE"   > eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.19-ref-checkpoints.txt
python3 eval/reports/p2-v2-option-a-prime-2026-05-06/STAGE-4.18-diff.py \
  STAGE-4.19-jsep-checkpoints.txt STAGE-4.19-ref-checkpoints.txt
```

## Stage 4.20 brief seeds

The brief that replaces this one in `TODO.md` queues **Probe 9b** —
weight-upload byte-hash check. Sketch:

1. Add a JS-side flag `globalThis.__weightHashProbe` that, when truthy,
   makes `ggml_backend_jsep_buffer_set_tensor` (or wherever the JSEP
   backend ingests bytes for a tensor allocation) compute a fast 32-bit
   hash (FNV-1a or XXHash) of the source bytes BEFORE the host-mirror
   copy AND BEFORE the `Module.jsepWrite` call, and EM_ASMs both keyed
   by `(tensor_name, offset, size)` into a JS log.
2. Mirror in JS: after weight upload completes, walk the GGUF parse
   result for the production layer-0 wq/wk weight tensors and compute
   the same hash from the parsed bytes.
3. Compare. If pre-upload hash == post-upload hash but the production
   matmul still produces 5.24e-4 Δ, the upload is fine and Stage 4.20
   becomes Outcome F (re-open kernel investigation with production-
   weight inputs). If pre-upload hash != post-upload hash, Stage 4.20
   becomes Outcome E and a one-line set_tensor / alignment / stride fix
   should follow.
