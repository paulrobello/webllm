# Parity Report: TinyLlama/TinyLlama-1.1B-Chat-v1.0

- **HF capture**: `hf-ref.json` (2026-05-11T19:37:39Z, dtype=float32)
- **WebLLM capture**: `webllm.json` (2026-05-12T23:24:11.196Z)
- **Prompt**: `The capital of France is`
- **n_layer / n_embd**: 22 / 2048
- **Thresholds**: cosine ≥ 0.95 end-of-stack, ≥ 0.99 first-block

## Per-layer residual (last token)

| Block | Cosine | L2 | Note |
|-------|--------|------|------|
| 0 | 0.9987 | 0.0361 | |
| 1 | 0.9982 | 0.0524 | |
| 2 | 0.9938 | 0.1059 | |
| 3 | 0.9906 | 0.1841 | |
| 4 | 0.9852 | 0.2918 | |
| 5 | 0.9826 | 0.4123 | |
| 6 | 0.9813 | 0.4809 | |
| 7 | 0.9815 | 0.5743 | |
| 8 | 0.9804 | 0.6432 | |
| 9 | 0.9818 | 0.7179 | |
| 10 | 0.9806 | 0.8726 | |
| 11 | 0.9808 | 0.9613 | |
| 12 | 0.9829 | 1.1010 | |
| 13 | 0.9846 | 1.2342 | |
| 14 | 0.9843 | 1.3591 | |
| 15 | 0.9857 | 1.5327 | |
| 16 | 0.9862 | 1.7604 | |
| 17 | 0.9875 | 2.1177 | |
| 18 | 0.9887 | 2.6279 | |
| 19 | 0.9889 | 3.2402 | |
| 20 | 0.9872 | 4.1663 | |
| 21 | 0.9736 | 52.6213 | |

## Final norm + logits (last token)

- Final-norm hidden cosine: **0.9855** (L2 14.5161)
- Top-16 logits overlap (set): **15/16**
- Greedy argmax match: **yes**
  - hf-ref top-1: id 3681 (val 13.388450622558594)
  - webllm top-1: id 3681 (val 12.918659210205078)

## Diagnosis

**PASS** — every layer above threshold; no sudden drops.
End-of-stack cosine 0.9855 → OK
