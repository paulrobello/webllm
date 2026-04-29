# Bucket C Phase 0 — Stage 2 reference vectors

**Date:** 2026-04-29
**Spec:** `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md` (`5ea108b`)
**Plan:** `docs/superpowers/plans/2026-04-29-embedding-bucket-c-phase-0-probe.md`
**Stage 1:** `STAGE-1-METADATA.md` (this directory)

## Inputs

5 fixed strings in `inputs.json` (verbatim copy from
`eval/reports/encoder-parity-2026-04-28/inputs.json`).

## Capture configuration

- **Model:** `Qwen/Qwen3-Embedding-0.6B`
- **sentence-transformers version:** 3.4.1 (recorded in `qwen3-embedding-0.6b-ref.json`'s `captured_with` field)
- **Pooling:** last-token (`qwen3.pooling_type = 3`)
- **Instruction prefix (query mode):**

  ```
  Instruct: Given a web search query, retrieve relevant passages that answer the query
  Query:
  ```

  Real LF (U+000A) between the two lines; no space after `Query:`. Documents pass raw.

- **Output dim:** 1024 (= `hidden_size`; no projection head)
- **Normalization:** L2 via sentence-transformers `normalize_embeddings=True`. Max magnitude deviation across all 10 vectors: **5.46e-08** — eight orders of magnitude inside the 1e-3 spec gate.

## Document mode (5 vectors)

| Row | Input (truncated) | Magnitude | First 3 dims |
|----:|-------------------|----------:|--------------|
|   0 | Hello world. | 1.000000 | [+0.00895, +0.00117, -0.00877] |
|   1 | The quick brown fox jumps … | 1.000000 | [+0.02414, -0.04546, -0.00333] |
|   2 | Embedding models map text … | 1.000000 | [-0.05574, -0.06162, -0.00848] |
|   3 | Café — naïve façade résumé piñata coöperate. 你好世界. … | 1.000000 | [+0.04532, -0.06311, -0.01294] |
|   4 | . | 1.000000 | [-0.03384, -0.03029, -0.01492] |

## Query mode (5 vectors)

| Row | Input (truncated, prefix elided) | Magnitude | First 3 dims |
|----:|----------------------------------|----------:|--------------|
|   0 | Hello world. | 1.000000 | [-0.00312, +0.01200, -0.00531] |
|   1 | The quick brown fox jumps … | 1.000000 | [+0.03934, -0.03995, -0.00702] |
|   2 | Embedding models map text … | 1.000000 | [-0.02735, -0.03843, -0.00326] |
|   3 | Café — naïve façade résumé piñata coöperate. 你好世界. … | 1.000000 | [+0.05705, -0.04128, -0.01140] |
|   4 | . | 1.000000 | [+0.00307, -0.01717, -0.01292] |

## Pairwise cosine (informational)

Mode prefix shifts the embedding subspace materially. Same-row doc-vs-query
cosine well below 1.0 confirms the prefix is being applied (a silent drop
would land it ≥ 0.99).

**Doc-vs-doc 5×5 cosine matrix:**

```
+1.000 +0.507 +0.405 +0.533 +0.628
+0.507 +1.000 +0.309 +0.378 +0.517
+0.405 +0.309 +1.000 +0.262 +0.450
+0.533 +0.378 +0.262 +1.000 +0.377
+0.628 +0.517 +0.450 +0.377 +1.000
```

**Query-vs-query 5×5 cosine matrix:**

```
+1.000 +0.281 +0.166 +0.461 +0.302
+0.281 +1.000 +0.115 +0.328 +0.290
+0.166 +0.115 +1.000 +0.167 +0.233
+0.461 +0.328 +0.167 +1.000 +0.225
+0.302 +0.290 +0.233 +0.225 +1.000
```

**Doc-vs-query (same row):**

| Row | Cosine |
|----:|-------:|
|   0 | 0.7200 |
|   1 | 0.8427 |
|   2 | 0.9087 |
|   3 | 0.9674 |
|   4 | 0.5409 |

Off-diagonal ranges: doc-vs-doc 0.262–0.628; query-vs-query 0.115–0.461. Healthy
distinguishability — no model collapse, no degenerate vectors. Row 4's strong
separation (0.5409) reflects the unicode-heavy input being the most prefix-
sensitive case; row 3's tightness (0.9674) reflects a short, semantically
dense fixture.

## Probe conclusion

**Recommendation: proceed to Phase 1.**

All six spec-listed risks are resolved or explicitly accepted:

1. ✅ **GGUF mirror keys present** — Stage 1 metadata diff against HF config
   showed three benign divergences (L2-normalize is consumer-side; dual EOS/EOT;
   `vocab_size` derive-from-tokens-table). None are blockers.
2. ✅ **Pooling type confirmed last-token** — `qwen3.pooling_type = 3`,
   matches HF README's `last_token_pool` example.
3. ✅ **Instruction-prefix convention pinned** — exact runtime byte sequence
   captured verbatim and reproduced in Stage 2 reference vectors.
4. ✅ **Hidden-state tap-point recommendation landed** — #2 post-output-norm
   (`model-inference.ts:912-916`, before `lm_head`), aligned with
   `~/Repos/llama.cpp/src/models/qwen3.cpp:98` `res->t_embd = cur`.
5. ✅ **Reference vectors L2-normalized** — magnitude assertion passed at
   1e-3 tolerance with margin 8 orders tighter (5.46e-08).
6. ✅ **Projection-head presence + dim characterized** — absent; output dim
   = `hidden_size` = 1024.

**Risks deferred (per spec):**
- a. **WebGPU graph-build cost in embed mode** — Phase 3 surfaces; Phase 4 measures.
  No probe-side gate.
- b. **Causal-mask semantics under last-token pooling** — Phase 2 implementation
  must validate against the parity gate using the reference vectors landed in
  Stage 2. Theoretically correct (last token sees all prior tokens, which is
  exactly what we want pooled).
- c. **4B/8B variant feasibility** — Out of scope per current cycle; gated on
  Phase 3 success at 0.6B.

**Open questions for the Phase 1 plan:**
- Architecture-truth-source path for chunked / batched embedding inference
  (the probe characterized single-text encoding; bulk-encode dispatch shape
  remains a Phase 1+ concern).
- Whether the embed-mode toggle should be exposed as `engine.embed(modelId, text)`
  (consistent with bucket A/B) or as a new entry point. Probably consistent.
- Tokenizer routing — Qwen3-Embedding's BPE vs the existing Qwen3 chat path's
  tokenizer; verify the existing tokenizer pipeline handles the embedder's
  vocab cleanly.

**Phase 1 entry posture: ready.** All Stage 2 artifacts are in place to gate
Phase 3 parity. Phase 1+ specs to be written when the next cycle queues
implementation.
