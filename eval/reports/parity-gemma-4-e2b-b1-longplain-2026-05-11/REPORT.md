# Parity Report: unsloth/gemma-4-E2B-it

- **HF capture**: `hf-ref.json` (2026-05-11T22:29:12Z, dtype=float32)
- **WebLLM capture**: `webllm.json` (2026-05-11T22:29:31.239Z)
- **Prompt**: `France is a country located primarily in Western Europe, with several overseas regions and territories. Its capital and largest city is Paris, located in the north on the Seine river. The official language is French, and the country uses the euro as its currency. France borders Belgium, Luxembourg, Germany, Switzerland, Monaco, Italy, Andorra, and Spain. Famous landmarks include the Eiffel Tower, the Louvre Museum, and the Palace of Versailles. The capital of France is`
- **n_layer / n_embd**: 35 / 1536
- **Thresholds**: cosine ≥ 0.95 end-of-stack, ≥ 0.99 first-block

## Per-layer residual (last token)

| Block | Cosine | L2 | Note |
|-------|--------|------|------|
| embed | 0.9953 | 3.8486 | |
| 0 | 0.9770 | 9.2126 | ⚠ below threshold |
| 1 | 0.9756 | 6.5387 | |
| 2 | 0.6784 | 23.7855 | ⚠ below threshold (Δ vs prev: -0.2972) |
| 3 | 0.8377 | 33.3885 | ⚠ below threshold |
| 4 | 0.9392 | 31.7137 | ⚠ below threshold |
| 5 | 0.9363 | 29.4352 | ⚠ below threshold |
| 6 | 0.8929 | 34.2676 | ⚠ below threshold |
| 7 | 0.8586 | 37.5683 | ⚠ below threshold |
| 8 | 0.8560 | 38.0042 | ⚠ below threshold |
| 9 | 0.6720 | 52.6902 | ⚠ below threshold (Δ vs prev: -0.1840) |
| 10 | 0.5235 | 55.0631 | ⚠ below threshold (Δ vs prev: -0.1485) |
| 11 | 0.2396 | 58.8625 | ⚠ below threshold (Δ vs prev: -0.2839) |
| 12 | 0.6230 | 45.9350 | ⚠ below threshold |
| 13 | 0.6409 | 34.8926 | ⚠ below threshold |
| 14 | 0.7816 | 44.3104 | ⚠ below threshold |
| 15 | 0.7385 | 50.6305 | ⚠ below threshold |
| 16 | 0.8060 | 45.9934 | ⚠ below threshold |
| 17 | 0.8225 | 40.2328 | ⚠ below threshold |
| 18 | 0.8641 | 36.9023 | ⚠ below threshold |
| 19 | 0.7590 | 46.3950 | ⚠ below threshold (Δ vs prev: -0.1051) |
| 20 | 0.7661 | 44.2998 | ⚠ below threshold |
| 21 | 0.7897 | 42.6853 | ⚠ below threshold |
| 22 | 0.7994 | 39.2477 | ⚠ below threshold |
| 23 | 0.6278 | 46.4940 | ⚠ below threshold (Δ vs prev: -0.1716) |
| 24 | 0.7746 | 47.1616 | ⚠ below threshold |
| 25 | 0.7583 | 47.1889 | ⚠ below threshold |
| 26 | 0.8043 | 44.8749 | ⚠ below threshold |
| 27 | 0.8436 | 46.2940 | ⚠ below threshold |
| 28 | 0.8864 | 48.0647 | ⚠ below threshold |
| 29 | 0.8949 | 44.5833 | ⚠ below threshold |
| 30 | 0.9038 | 41.1449 | ⚠ below threshold |
| 31 | 0.9134 | 35.8710 | ⚠ below threshold |
| 32 | 0.9131 | 33.6353 | ⚠ below threshold |
| 33 | 0.9158 | 27.0125 | ⚠ below threshold |
| 34 | 0.0509 | 166.4371 | ⚠ below threshold (Δ vs prev: -0.8649) |

## Final norm + logits (last token)

- Final-norm hidden cosine: **0.3894** (L2 176.2927)
- Top-16 logits overlap (set): **7/16**
- Greedy argmax match: **no**
  - hf-ref top-1: id 9079 (val 7.056691646575928)
  - webllm top-1: id 236761 (val 42.71852111816406)

## Diagnosis

- **Embedding output matches** (cos 0.9953 ≥ 0.99) → bug is INSIDE the per-block forward path, not in `opGetRows + opScale`.
- First layer with cosine below threshold: **block 0**
- First sudden Δ ≤ -0.05 between consecutive layers at: **block 2**

Inspect the op sequence inside that block (or the inputs from the prior block) for the bug.
