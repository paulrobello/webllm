# Parity Report: unsloth/gemma-4-E2B-it

- **HF capture**: `hf-ref.json` (2026-05-11T22:26:01Z, dtype=float32)
- **WebLLM capture**: `webllm.json` (2026-05-11T22:36:40.646Z)
- **Prompt**: `PLACEHOLDER — see input_token_ids; the chat-formatted text isn't reconstructed because special-token round-tripping isn't lossless across SPM variants.`
- **n_layer / n_embd**: 35 / 1536
- **Thresholds**: cosine ≥ 0.95 end-of-stack, ≥ 0.99 first-block

## Per-layer residual (last token)

| Block | Cosine | L2 | Note |
|-------|--------|------|------|
| embed | 0.9972 | 3.0925 | |
| 0 | 0.9998 | 0.7795 | |
| 1 | 0.9997 | 0.6906 | |
| 2 | 0.9996 | 1.1425 | |
| 3 | 0.9997 | 1.4311 | |
| 4 | 0.9995 | 2.8852 | |
| 5 | 0.9987 | 3.9151 | |
| 6 | 0.9987 | 3.4552 | |
| 7 | 0.9964 | 5.7543 | |
| 8 | 0.9961 | 6.0925 | |
| 9 | 0.9936 | 8.3389 | |
| 10 | 0.9892 | 8.5126 | |
| 11 | 0.9820 | 10.4988 | |
| 12 | 0.9839 | 8.4861 | |
| 13 | 0.9789 | 8.0846 | |
| 14 | 0.9944 | 7.0984 | |
| 15 | 0.9941 | 8.3109 | |
| 16 | 0.9933 | 9.9761 | |
| 17 | 0.9913 | 11.3613 | |
| 18 | 0.9925 | 10.4751 | |
| 19 | 0.9855 | 12.9096 | |
| 20 | 0.9868 | 10.4223 | |
| 21 | 0.9909 | 8.8779 | |
| 22 | 0.9920 | 8.3321 | |
| 23 | 0.9914 | 7.3731 | |
| 24 | 0.9939 | 7.5087 | |
| 25 | 0.9935 | 7.5798 | |
| 26 | 0.9958 | 6.5934 | |
| 27 | 0.9975 | 5.9436 | |
| 28 | 0.9985 | 5.5077 | |
| 29 | 0.9979 | 6.2915 | |
| 30 | 0.9981 | 6.1215 | |
| 31 | 0.9983 | 5.4569 | |
| 32 | 0.9982 | 5.3880 | |
| 33 | 0.9982 | 4.6379 | |
| 34 | 0.3722 | 239.6758 | ⚠ below threshold (Δ vs prev: -0.6260) |

## Final norm + logits (last token)

- Final-norm hidden cosine: **0.9951** (L2 39.1298)
- Top-16 logits overlap (set): **14/16**
- Greedy argmax match: **yes**
  - hf-ref top-1: id 3672 (val 9.133856773376465)
  - webllm top-1: id 3672 (val 9.66129207611084)

## Diagnosis

- **Embedding output matches** (cos 0.9972 ≥ 0.99) → bug is INSIDE the per-block forward path, not in `opGetRows + opScale`.
- First layer with cosine below threshold: **block 34**
- First sudden Δ ≤ -0.05 between consecutive layers at: **block 34**

Inspect the op sequence inside that block (or the inputs from the prior block) for the bug.
