# Parity Report: unsloth/gemma-4-E2B-it

- **HF capture**: `hf-ref.json` (2026-05-11T22:29:16Z, dtype=float32)
- **WebLLM capture**: `webllm.json` (2026-05-11T22:29:44.924Z)
- **Prompt**: `<|turn>user
Hi<turn|>
<|turn>model
`
- **n_layer / n_embd**: 35 / 1536
- **Thresholds**: cosine ≥ 0.95 end-of-stack, ≥ 0.99 first-block

## Per-layer residual (last token)

| Block | Cosine | L2 | Note |
|-------|--------|------|------|
| embed | 0.9972 | 3.0925 | |
| 0 | 0.9920 | 5.3326 | |
| 1 | 0.9874 | 4.4418 | |
| 2 | 0.9275 | 14.0669 | ⚠ below threshold (Δ vs prev: -0.0599) |
| 3 | 0.9543 | 17.6856 | |
| 4 | 0.9911 | 11.7993 | |
| 5 | 0.9885 | 11.5308 | |
| 6 | 0.9847 | 11.7107 | |
| 7 | 0.9638 | 17.7270 | |
| 8 | 0.9693 | 15.9536 | |
| 9 | 0.9309 | 23.3143 | ⚠ below threshold |
| 10 | 0.8630 | 28.7747 | ⚠ below threshold (Δ vs prev: -0.0678) |
| 11 | 0.8510 | 31.5647 | ⚠ below threshold |
| 12 | 0.8136 | 30.2731 | ⚠ below threshold |
| 13 | 0.8210 | 24.6356 | ⚠ below threshold |
| 14 | 0.9557 | 19.9334 | |
| 15 | 0.9534 | 22.4814 | |
| 16 | 0.9536 | 24.3590 | |
| 17 | 0.9425 | 25.9639 | ⚠ below threshold |
| 18 | 0.9411 | 25.6256 | ⚠ below threshold |
| 19 | 0.9154 | 30.7914 | ⚠ below threshold |
| 20 | 0.9316 | 25.7198 | ⚠ below threshold |
| 21 | 0.9346 | 25.7067 | ⚠ below threshold |
| 22 | 0.9477 | 22.7027 | ⚠ below threshold |
| 23 | 0.9510 | 20.1737 | |
| 24 | 0.9651 | 19.5992 | |
| 25 | 0.9597 | 20.5433 | |
| 26 | 0.9616 | 20.5624 | |
| 27 | 0.9636 | 21.0740 | |
| 28 | 0.9630 | 23.0394 | |
| 29 | 0.9627 | 22.1150 | |
| 30 | 0.9532 | 24.4146 | |
| 31 | 0.9422 | 26.7124 | ⚠ below threshold |
| 32 | 0.9273 | 30.6170 | ⚠ below threshold |
| 33 | 0.9275 | 25.7536 | ⚠ below threshold |
| 34 | 0.3455 | 299.4391 | ⚠ below threshold (Δ vs prev: -0.5819) |

## Final norm + logits (last token)

- Final-norm hidden cosine: **0.9055** (L2 131.6523)
- Top-16 logits overlap (set): **10/16**
- Greedy argmax match: **no**
  - hf-ref top-1: id 10979 (val 14.935763359069824)
  - webllm top-1: id 9259 (val 15.674729347229004)

## Diagnosis

- **Embedding output matches** (cos 0.9972 ≥ 0.99) → bug is INSIDE the per-block forward path, not in `opGetRows + opScale`.
- First layer with cosine below threshold: **block 2**
- First sudden Δ ≤ -0.05 between consecutive layers at: **block 2**

Inspect the op sequence inside that block (or the inputs from the prior block) for the bug.
