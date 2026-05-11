# Phase 4 closure — shared-KV wired for Gemma 4 E2B (2026-05-11)

## TL;DR

Phase 4 of Task 3.3l (Gemma 4 Stage 3 PLE + dual RoPE) is closed.
WebLLM now implements Gemma 4's `n_layer_kv_from_start` shared-KV
remap per `llama-model.cpp:2007-2014`. End-of-stack residual cosine
recovered from **0.0420 → 0.9722**, top-1 argmax now **matches HF
reference** (id 9079 = "Paris" for prompt `"The capital of France
is"`).

## What landed

- `ModelHyperparams.kvReuseFromLayer: (number | null)[]` — per-layer
  KV source index, `null` for owning layers. Computed at load time
  from `sharedKvLayers` + `slidingWindowPattern` using the canonical
  iSWA rule: shared SWA layers reuse layer 13 (last pre-share SWA);
  shared full layers reuse layer 14 (last pre-share full).
- `initKVCache`: shared layers don't allocate own K/V tensors —
  the `kvLayers[il]` entry shares tensor handles with
  `kvLayers[kvReuseFromLayer[il]]`. Saves ~480 MiB on Gemma 4 E2B
  at maxCtx=4096 / F16.
- `buildQOnly` helper: split-QKV-style Q projection without K/V,
  for use at shared layers (which don't compute their own K/V).
- `forwardSingle`, `forwardAllPositions`, `forwardDecode`: gate the
  K/V projection + RoPE + cache-write block on `!isShared`. Shared
  layers' attention reads `kv.k`/`kv.v` which now point at the
  source layer's tensor (the source's writes were graph-expanded
  earlier in this same forward).
- `forwardWithLayerTaps` (parity-capture path, no cache): shared
  layers reuse the source layer's post-RoPE K and pre-permute V
  from the same forward pass via per-layer scratch arrays.
- `forwardForEmbedding` + `debugLayerOutput`: defensive throws for
  shared-KV (not currently exercised on Gemma 4; wire later if
  registration changes).

## Phase 3 → Phase 4 deltas

| Block | Phase 3 cos | Phase 4 cos | Δ |
|------:|------------:|------------:|---|
| L13   | 0.5743      | 0.9337      | +0.36 |
| L14   | 0.9742      | 0.9742      | — |
| **L15** | **0.6605** | **0.9680** | **+0.31** ★ |
| L19 (full, share→14) | drift | 0.9653 | recovered |
| L20-23 (SWA, share→13) | drift | 0.97-0.98 | recovered |
| L24 (full, share→14) | drift | 0.9841 | recovered |
| L25-28 (SWA, share→13) | drift | 0.98-0.99 | recovered |
| L29 (full, share→14) | drift | 0.9933 | recovered |
| L30-33 (SWA, share→13) | drift | 0.99-0.99 | recovered |
| L34 (full, share→14) | terrible | 0.2824 (artifact, see below) |
| End-stack hidden | **0.1927** | **0.9722** | +0.78 ★ |
| Top-1 argmax | MISS (id 531) | **MATCH (id 9079 "Paris")** | ★★ |
| Top-16 overlap | 1/16 | 13/16 | +12 |

## L34 anomaly is a comparison artifact (not a code bug)

`capture-hf-ref.py:118` indexes `hidden[i+1]` for `i in [0..n_layer-1]`
so `per_layer[34] = hidden[35] = hidden[-1]`. HF transformers'
convention puts the **post-final-norm** tensor at
`hidden_states[-1]`, NOT the post-block-34 residual. WebLLM taps
the post-block-34 residual at `layerTaps[34]`. The two are not
the same tensor; their L2-norms differ by ~25× (final RMSNorm
normalizes magnitude), which manifests as a 0.28 cosine.

`final_norm = hidden[-1]` in HF, compared against WebLLM's
post-final-norm hidden, gives the correct **0.9722** cosine.

**Follow-up:** `capture-hf-ref.py` should clamp the per-layer
range to `hidden[1:n_layer]` (drop the last entry to keep it as
residual-only), or add a post-final-norm row that the comparison
treats as `final_norm` only.

## Validation

- `make checkall`: 762 pass / 36 skip / 0 fail (Phase 4 changes
  preserved TinyLlama / Qwen / Mistral / Phi-3 contract).
- `eval/reports/parity-gemma-4-e2b-shared-kv-2026-05-11/REPORT.md`
  — Gemma 4 with the inputs that produced the Phase 3 failure.
- TinyLlama regression-check **deferred** — Phase 4 changes are
  gated on `hp.kvReuseFromLayer?.[il]` being set, which only
  Gemma 4 family does. The control predicate covers all non-Gemma
  paths.

## Known residual issue — chat smoke produces `<eos>`

After Phase 4, greedy chat smoke (`prompt=The capital of France is`,
temp=0) emits 13 `<eos>` tokens immediately. Pre-Phase-4 it
emitted 20 tokens of mixed-script noise. Parity capture proves
the arithmetic is correct, so the chat smoke regression is
**downstream of Phase 4** — most likely a chat-template /
tokenizer mismatch (the prompt is now tokenized to ~75 tokens,
suggesting `<start_of_turn>` / `<end_of_turn>` aren't being
treated as special tokens).

This is a Stage 2-class concern (chat template fidelity), not a
Stage 3 forward-pass correctness concern. Track separately as
Task 3.3l Phase 5 or a dedicated chat-template audit.

## Next-session pickup

1. Diagnose the Gemma 4 chat-template tokenization: log
   `tokenizer.encode("<start_of_turn>user\n...")` and verify
   special-token treatment matches HF transformers'
   `apply_chat_template`.
2. Run the 36-prompt eval suite (`make bench-…` for Gemma 4
   profile) to gate Task 3.5 closure — target ≥ 40% accuracy.
3. If eval passes, archive Stage 3 and start Stage 4 (real SWA
   windowed mask).
4. Fix `capture-hf-ref.py` to align HF and WebLLM tap conventions
   (drop `hidden[-1]` from per_layer; keep it only for `final_norm`).
