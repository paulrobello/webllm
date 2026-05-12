# Stage 4.3 long-context parity probe — incremental capture

**Date:** 2026-05-12 EOS-7
**Tip:** `2c32f80` (incremental per-layer capture infrastructure committed).
**Outcome:** **PARTIAL** — incremental capture API verified correct
on TinyLlama (end-of-stack cosine 0.9855, greedy argmax matches HF).
Gemma 4 E2B at N=560 remains **blocked by the 128 MiB per-binding
cap**: even with all four cap-aware shavings landed (single-layer
tap pin via opView2d+opCont, sliced lm_head input, sliced
finalHidden readback, embedding-tap pin skipped), the Gemma 4
forward graph OOMs at node ~821 (≈layer 30 working set) — about
1.5 MB short of fitting.

## Gate (per `TODO.md:1325-1328`)

> Generate 1000-token output on a fixed long prompt; measure
> argmax-divergence vs HF reference or a known-good `llama-cli` run.
> Gate: no quality cliff at the 512-token SWA boundary.

Strict numerical interpretation (per-layer cosine ≥ 0.95 vs HF at
N > swa_window): **not met on Gemma 4 E2B** due to the per-binding
cap blocker.

Functional interpretation ("no quality cliff at 512-token boundary"):
**positive evidence already on record** from the Stage 4.1 long-
context closure (`eval/reports/gemma-4-stage4.1-longctx-closure-
2026-05-12/SUMMARY.md`): N=2238 chat-path produces fact-correct
retrieval drawn from prompt position ~280, well past the SWA window
at the late decode positions, with zero console errors.

## What we set out to do

After the Stage 4.1 chat-path non-crash closure (N=2238 prefill +
33 tokens decoded, fact-correct answer), the remaining gap to a
"numerical parity" closure was a per-layer cosine comparison against
the HuggingFace `transformers` reference at long context. The user
chose **path 2 of the 2026-05-12 decision**: "Incremental per-layer
capture (35 separate forwards)" — wire the unused `captureTaps`
scaffolding so each forward pins only one layer's tap, bypassing
the 128 MiB cap that blocked the full-tap mode.

## What landed (commit `2c32f80`)

`forwardWithLayerTaps` gained three new options:

| Option | What it does |
|---|---|
| `captureLayer?: number` | Pin only the named layer's `cur` for readback. Other layers stay intermediates (allocator reuses their slots). |
| `lastTokenLogitsOnly?: boolean` | Slice finalHidden's last column before lm_head matmul. Logits go from `N*V*F32` to `1*V*F32` (575 MB → 1 MB at N=560/V=262144). finalHidden readback uses the slice. |
| (gated on `lastTokenLogitsOnly`) | Skip the embedding-tap pin — per-layer SWA parity doesn't need the embedding readback. |

`smoke-test/parity-capture-page.js` gained `?mode=incremental`: loop
over layers 0..L-1, capture each, aggregate into a single
canonical `webllm.json`. Tolerates per-layer OOM (logs skipped
count, continues). One trailing `skipLayerTaps + lastTokenLogitsOnly`
pass fetches `finalNormHidden + top-K logits`.

The captured layer's tap is sliced via `opView2d + opCont` to a
fresh `[E, 1]` tensor — pinning just 6 KB instead of the full
N*E*F32 (3.4 MB at N=560), so the allocator can lifetime-pack the
parent `cur` once its next consumer completes.

## Verification — TinyLlama at N=6

Sanity check that the new API path is correct on a model that
*doesn't* hit the cap. Run:

```bash
URL="http://localhost:8031/parity-capture.html?model=tinyllama-1.1b-chat-q4_0&mode=incremental&inputIds=1,450,7483,310,3444,338"
# inputIds = HF-tokenized "The capital of France is" with BOS.
```

Result (full report:
[`eval/reports/parity-tinyllama-incremental-2026-05-12/REPORT.md`](../parity-tinyllama-incremental-2026-05-12/REPORT.md)):

| Metric | Value |
|---|---|
| Layers captured | 22 / 22 |
| First-block cosine (≥ 0.99 target) | 0.9987 ✅ |
| End-of-stack cosine (≥ 0.95 target) | 0.9855 ✅ |
| Final-norm cosine | 0.9855 |
| Top-1 argmax match HF | yes (id 3681 both sides) |
| Top-16 overlap | 15/16 |

**Diagnosis:** PASS. The incremental capture API produces parity
data indistinguishable from the existing full-mode capture
(`parity-tinyllama-2026-05-11/`). The infrastructure is correct
and reusable for any non-PLE/non-shared-KV model.

## Gemma 4 E2B at N=560 — blocker analysis

The 1129-token HF reference from Stage 4.1's debris
(`gemma-4-stage4-swa-mask-2026-05-12/hf-ref.json`) was first
attempted; the WebLLM single-layer capture OOM'd at node ~370
needing 40 MB / 39 MB available. The 40 MB allocation is
`inpPerLayer` (PLE per-layer projection materialization at
`pleDim × layerCount × N × F32 = 256 × 35 × 1129 × 4`). We pivoted
to N=560 (still crosses the 512 SWA window) using the existing
`hf-ref-560.json` reference.

At N=560 with all four shavings live:

```
ggml_tallocr_alloc: not enough space in the buffer
  to allocate node_821 (needed 3440640, available 1998848)
```

The dominant fixed costs in the graph allocator's lifetime-packed
view:

| Pinned/retained | Bytes at N=560 |
|---|---|
| `inpPerLayer` (PLE materialization, consumed per-block) | 20 MB |
| Causal mask + SWA mask (F16, padded to 32) | 1.3 MB |
| Per-block intermediates (peak: qk + attnW for global layers @ nHeads=16, headDim=256) | ~20 MB |
| Shared-KV `kRopePerLayer` + `v3PerLayer` for layers 0-14 (referenced by 15-34) | ~60 MB |
| Captured tap (1 layer, sliced to last token via opCont) | 6 KB |
| `finalHidden` (sliced to last token via opView2d before lm_head) | 6 KB |
| `logits` (1 × V × F32 after lm_head input slice) | 1 MB |
| **Total live peak (approx)** | **~105 MB** |

105 MB plus per-block scratch and op overhead lands just above
the 128 MiB cap — failing about 1.5 MB short at the deepest layer
processed (node 821 ≈ layer 30 of full stack).

## What was tried (all on `2c32f80` codebase)

| Lever | Implemented? | Effect |
|---|---|---|
| Single-layer tap pin (drop 34 of 35 per-layer pins) | ✅ | saved ~116 MB at N=560 (vs full mode); cleared the dominant pressure but did not get under the cap |
| Slice `lmHeadInput` last-token (N×V → 1×V) | ✅ | saved 575 MB at N=560 — large factor but not load-bearing at this N (the original full-N logits was never the failure node) |
| Slice `finalHidden` last-token readback | ✅ | saved 3.4 MB |
| Skip embedding-tap pin under `lastTokenLogitsOnly` | ✅ | saved 0 (allocator was already lifetime-packing the embedding output; the pin wasn't load-bearing) |
| Slice captured `cur` tap via opCont to last-token-only | ✅ | saved ~3.4 MB (pin footprint 3.4 MB → 6 KB) |
| Early-loop termination after `captureLayer` | ❌ disabled | broke graph allocator's buffer sizing (sized to ~21 MB, less than `inpPerLayer` alone); reverted |
| Backend patch (split `wgpu::Buffer` per layer scratch) | — not attempted | this is **path 4** of the original decision; user chose path 2 |

## Why Gemma 4 hits the cap and TinyLlama doesn't

Three Gemma 4-specific costs dominate the cap budget and are not
present on TinyLlama / qwen3 / mistral / llama-3:

1. **`inpPerLayer` PLE materialization (20 MB at N=560).** Pinned
   for the entire forward pass — every block slices into it. This
   tensor doesn't exist on architectures without per-layer
   embeddings.
2. **Shared-KV K/V retention (~60 MB at N=560).** Layers 15-34's
   attention reads from K/V computed in layers 0-14. The allocator
   must keep those K/V slots live across the whole pass. Models
   without shared-KV release K/V slots immediately after each
   layer's attention completes.
3. **head_dim 256 on every layer.** Global layers use 16 heads ×
   head_dim 256, producing larger Q/K/V/attention tensors than
   typical 1-2B models (TinyLlama: 4 heads × head_dim 64). The
   attention matrix at N=560/global is 16 * 560 * 560 * 4 = 20 MB
   per layer (live during the softmax → V matmul).

## Recommendations

For the Stage 4.3 strict gate (per-layer cosine ≥ 0.95 vs HF at
N > swa_window) on Gemma 4 E2B specifically:

1. **Path 4 from the 2026-05-12 decision (backend patch).** Modify
   `~/Repos/llama.cpp/ggml/src/ggml-webgpu` so per-layer scratch
   gets its own `wgpu::Buffer` (multiple bindings instead of one
   large packed buffer). Cap doctrine applies per-binding, not
   per-graph, so this releases ~60 MB of shared-KV K/V back to
   their own buffers. New patch on `webllm-browser-patches`.
   Estimated risk: medium (touches the load-bearing allocator).
2. **Bump SWA window-aware probe to a smaller PLE model.** Gemma 4
   E2B has `n_embd=1536` + 35 layers — a hypothetical smaller PLE
   model (1B / 22 layers) would have proportionally smaller
   `inpPerLayer` and might fit. No such model is in the registered
   fleet today.
3. **Accept Stage 4.1 closure as the campaign-level evidence.**
   The 2238-token chat path produces fact-correct retrieval drawn
   from prompt position ~280 — across the SWA window for late
   decode positions, the SWA mask was active and the answer is
   right. That is positive *functional* evidence that the SWA
   wiring works at >512-token contexts, even without per-layer
   cosines.

For Stage 4.4 (eval re-gate), recommendation **3** is the natural
fit: the eval suite tests model outputs, not internal residuals.
If the eval baseline holds at ≥68% the Gemma 4 SWA path is
qualitatively validated.

## Status after this probe

| Stage | Gate | Status |
|---|---|---|
| 4.0 mask-feasibility probe | shader admits banded mask | ✅ CLOSED 2026-05-11 |
| 4.1 per-layer mask construction | unit + smoke | ✅ CLOSED 2026-05-11 |
| 4.1 final gate (chat.html non-crash at long N) | 2238-token reply coherent | ✅ CLOSED 2026-05-12 |
| 4.2 Gemma 2 SWA derivation | per-layer pattern from `swa_period` | ✅ CLOSED 2026-05-12 |
| 4.3 long-context parity probe | per-layer cosine ≥ 0.95 vs HF | **PARTIAL** — TinyLlama path verified; Gemma 4 blocked by per-binding cap |
| 4.4 eval re-gate | 36-prompt eval ≥ 68% | queued |

The campaign Q2 (Stage 4) goal of "real sliding-window attention
on Gemma 4 SWA layers" remains validated by:
- 4.0 + 4.1 + 4.1-final-gate (implementation correctness +
  reachability)
- 4.2 (Gemma 2/3 derivation correctness + unit tests)
- the existing 2238-token fact-correct retrieval (functional SWA
  evidence)

Stage 4.3's *strict numerical* gate on Gemma 4 specifically is
deferred behind path 4 (backend per-binding patch) or rolled into
Stage 4.4's eval re-gate.

## Artifacts in this run dir

- `hf-ref.json` — HuggingFace reference at N=1129 (1.5 MB)
- `hf-ref-560.json` — HuggingFace reference at N=560 (1.5 MB)
- `webllm.json` — incremental capture attempt (all 35 layers failed; empty placeholders)
- `SUMMARY.md` — this report

Cross-references:
- Incremental API sanity check (PASS):
  [`eval/reports/parity-tinyllama-incremental-2026-05-12/REPORT.md`](../parity-tinyllama-incremental-2026-05-12/REPORT.md)
- Stage 4.1 long-context functional closure:
  [`eval/reports/gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md`](../gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md)
- Per-binding cap doctrine: `CLAUDE.md` "Per-binding 128 MiB cap doctrine"
- Code: commit `2c32f80` (`feat(parity-capture): incremental per-layer capture mode`)
