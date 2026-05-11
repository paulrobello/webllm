# Parity Report: unsloth/gemma-4-E2B-it

- **HF capture**: `hf-ref.json` (2026-05-11T22:26:01Z, dtype=float32)
- **WebLLM capture**: `webllm.json` (2026-05-11T22:26:25.719Z)
- **Prompt**: `PLACEHOLDER — see input_token_ids; the chat-formatted text isn't reconstructed because special-token round-tripping isn't lossless across SPM variants.`
- **n_layer / n_embd**: 35 / 1536
- **Thresholds**: cosine ≥ 0.95 end-of-stack, ≥ 0.99 first-block

## Per-layer residual (last token)

| Block | Cosine | L2 | Note |
|-------|--------|------|------|
| embed | 0.9972 | 3.0925 | |
| 0 | 0.9900 | 5.9222 | ⚠ below threshold |
| 1 | 0.9467 | 8.8504 | ⚠ below threshold |
| 2 | 0.6520 | 28.9107 | ⚠ below threshold (Δ vs prev: -0.2947) |
| 3 | 0.7249 | 41.9863 | ⚠ below threshold |
| 4 | 0.9010 | 38.7227 | ⚠ below threshold |
| 5 | 0.8808 | 36.8673 | ⚠ below threshold |
| 6 | 0.8292 | 39.5054 | ⚠ below threshold (Δ vs prev: -0.0516) |
| 7 | 0.7143 | 50.2823 | ⚠ below threshold (Δ vs prev: -0.1149) |
| 8 | 0.8000 | 42.6568 | ⚠ below threshold |
| 9 | 0.4888 | 68.7125 | ⚠ below threshold (Δ vs prev: -0.3113) |
| 10 | 0.4749 | 55.2458 | ⚠ below threshold |
| 11 | 0.1538 | 63.2132 | ⚠ below threshold (Δ vs prev: -0.3211) |
| 12 | 0.3528 | 54.9032 | ⚠ below threshold |
| 13 | 0.3125 | 46.9889 | ⚠ below threshold |
| 14 | 0.7880 | 44.1576 | ⚠ below threshold |
| 15 | 0.6997 | 57.5862 | ⚠ below threshold (Δ vs prev: -0.0884) |
| 16 | 0.7451 | 58.0317 | ⚠ below threshold |
| 17 | 0.7592 | 55.8130 | ⚠ below threshold |
| 18 | 0.8209 | 48.8945 | ⚠ below threshold |
| 19 | 0.6132 | 63.1785 | ⚠ below threshold (Δ vs prev: -0.2077) |
| 20 | 0.5630 | 59.5038 | ⚠ below threshold (Δ vs prev: -0.0502) |
| 21 | 0.6700 | 53.1951 | ⚠ below threshold |
| 22 | 0.6929 | 50.3480 | ⚠ below threshold |
| 23 | 0.5140 | 57.8835 | ⚠ below threshold (Δ vs prev: -0.1789) |
| 24 | 0.6286 | 61.4625 | ⚠ below threshold |
| 25 | 0.6096 | 60.2995 | ⚠ below threshold |
| 26 | 0.6909 | 55.4950 | ⚠ below threshold |
| 27 | 0.7881 | 53.1556 | ⚠ below threshold |
| 28 | 0.8761 | 48.7080 | ⚠ below threshold |
| 29 | 0.8827 | 46.1752 | ⚠ below threshold |
| 30 | 0.8923 | 44.8302 | ⚠ below threshold |
| 31 | 0.9008 | 40.1513 | ⚠ below threshold |
| 32 | 0.9020 | 37.9396 | ⚠ below threshold |
| 33 | 0.9102 | 30.3168 | ⚠ below threshold |
| 34 | 0.0753 | 249.6266 | ⚠ below threshold (Δ vs prev: -0.8348) |

## Final norm + logits (last token)

- Final-norm hidden cosine: **0.5824** (L2 210.8846)
- Top-16 logits overlap (set): **0/16**
- Greedy argmax match: **no**
  - hf-ref top-1: id 3672 (val 9.133856773376465)
  - webllm top-1: id 1 (val 18.197433471679688)

## Diagnosis

- **Embedding output matches** (cos 0.9972 ≥ 0.99) → bug is INSIDE the per-block forward path, not in `opGetRows + opScale`.
- First layer with cosine below threshold: **block 0**
- First sudden Δ ≤ -0.05 between consecutive layers at: **block 2**

Inspect the op sequence inside that block (or the inputs from the prior block) for the bug.
