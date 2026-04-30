# Bucket D Phi-3.5-mini extension — DEMOTED (2026-04-30)

## Outcome

**Phi-3.5-mini-q4km (Phi-3.5 Mini Instruct, Q4_K_M GGUF) is demoted from
bucket D self-embedding** — `embeddingCapable: false`. The model passes
the row-by-row parity gate (10/10 at cos ≥ 0.91 vs PyTorch f16 reference)
but fails the semantic distinguishability gate: paraphrase cosines are
not separated from unrelated cosines under either pooling mode in
WASM Q4_K_M. `engine.embed(modelId, text)` would return semantically
random vectors despite matching the reference numerically.

This is a **§28 negative result** (cycle template per `CLAUDE.md`).
Retire the resurrection path; do not flip `embeddingCapable: true`
on this row without re-running the harness against a different quant
tier (Q5_K_M / Q6_K / f16).

The cycle did ship infrastructure that **stays in main**:

1. `embeddingPooling: "last-token" | "mean"` per-model registration
   field (BenchmarkModel + ModelLoadOptions + ModelEntry), threaded
   through `engine.embed` → `ModelInference.embed` (commits 7959873,
   1dcf0ed, 3efd6a1, 18346a6 — last reverted by 247da5c).
2. 16+16 cross-domain pair distinguishability harness in
   `eval/causal-embedder-parity.ts` (commit 4de154d).
3. Mean-margin gate `mean(P) − mean(U) ≥ 0.05` replacing the
   underpowered strict `min(P) > max(U)` (commit 031c2b9). Strict-min
   stays in the output as informational.

## Path of investigation

### Phase 0 — naive ship attempt (4-pair check, 2026-04-30 morning)

Passed 10/10 parity rows at cos ≥ 0.91 (`run.txt`) but the original
2+2 distinguishability check failed:

| pair                 | cosine   | verdict    |
| -------------------- | -------- | ---------- |
| paraphrase 0 (cat)   | 0.993844 | PASS       |
| paraphrase 1 (types) | 0.904927 | LOW        |
| unrelated 0          | 0.996105 | very HIGH  |
| unrelated 1          | 0.991011 | very HIGH  |

`min(P)=0.905 < max(U)=0.996` — strict FAIL. Working hypothesis:
last-token anisotropy.

### Phase 1 — PyTorch f16 probe (`probe-mean-pool.py`)

Computed both pooling modes against the same 2+2 fixtures in
PyTorch f16 (no quantization). Both modes **PASSED** the strict
criterion:

| mode       | mean(P) | mean(U) | strict |
| ---------- | ------- | ------- | ------ |
| last-token | 0.992   | 0.974   | PASS (margin 0.017) |
| mean-pool  | 0.972   | 0.949   | PASS (margin 0.024) |

Conclusion: the WASM 4-pair failure is **quant noise on one fixture**,
not anisotropy. Mean-pool helps slightly; both modes work in f16.

### Phase 2 — option B: mean-pool plumbing + 16+16 harness expansion

Implemented per-model `embeddingPooling` field; expanded
distinguishability harness to 16 paraphrase pairs (16 domains:
technology, sports, food, weather, finance, history, biology, music,
travel, education, medicine, law, art, geography, politics,
household) + 16 cross-domain unrelated pairs.

Re-captured `phi-3.5-mini-ref-mean.json` in PyTorch f16 with
mean-pool. Re-ran WASM parity against both refs.

| run                                | parity 0.91 | parity 0.80 | mean(P) | mean(U) | mean margin | strict |
| ---------------------------------- | ----------- | ----------- | ------- | ------- | ----------- | ------ |
| phi-3.5-mini-q4km **last-token**   | 10/10 PASS  | 10/10 PASS  | 0.987   | 0.993   | **−0.006**  | FAIL   |
| phi-3.5-mini-q4km **mean-pool**    | 7/10 PASS   | 10/10 PASS  | 0.944   | 0.971   | **−0.027**  | FAIL   |

**Both pooling modes fail the relaxed criterion** with negative
margin — paraphrases score *lower* than unrelated text on average.
Mean-pool's parity is also *worse* than last-token because Q4_K_M
quant noise compounds across all N hidden-state positions instead
of just position N−1.

### Phase 3 — qwen3-8b-iq3m re-validation (the bucket D flagship)

Ran the same 16+16 harness against qwen3-8b-iq3m (already shipped
`embeddingCapable: true`) to determine whether the strict gate's
fail was a harness defect or a real signal:

| run                                | parity 0.85 | mean(P) | mean(U) | mean margin | strict |
| ---------------------------------- | ----------- | ------- | ------- | ----------- | ------ |
| qwen3-8b-iq3m last-token           | 10/10 PASS  | 0.916   | 0.838   | **+0.084**  | FAIL   |

**qwen3-8b-iq3m fails strict-min on this set** (the photosynthesis
paraphrase scores 0.846 while a "photosynthesis... vs ...renewable
energy subsidies" cross-domain pair scores 0.740 — both touch
natural systems / biology, so the strict criterion can't separate
them). But mean-margin is solidly positive (+0.084), which is what
"useful but imperfect retrieval embedding" looks like.

Captured run: `qwen3-revalidation.txt`.

### Phase 4 — close out

Updated harness to gate on mean-margin ≥ 0.05; demoted phi-3.5;
phi-3.5 row now carries an inline note explaining why
`embeddingCapable` is absent.

## Lessons

1. **4 pairs (2+2) is statistically meaningless.** Both lucky and
   unlucky calls flip on one fixture. The 16+16 expansion exposes
   real signal vs noise.
2. **Strict-min is too tight** for any quantized embedder. Even
   the bucket D flagship fails it; vocabulary-overlap edge cases
   don't reflect retrieval quality.
3. **Mean-margin separates signal from noise cleanly.**
   - +0.084 (qwen3-8b-iq3m): genuinely discriminating, ships.
   - −0.006 to −0.027 (phi-3.5-mini-q4km): random / inverted, demote.
4. **Parity gate alone is insufficient.** A model can pass row-by-row
   cosine vs reference and still produce indiscriminate sentence
   embeddings. Distinguishability is its own gate.
5. **Mean-pool ≠ free anisotropy fix in quantized builds.** Q4_K_M
   noise compounds across all N positions, so mean-pool's parity is
   strictly worse. f16 gains do not transfer.
6. **The bucket D failure mode is per-model, not per-architecture.**
   Phi-3.5 fails; Qwen3-8B passes. Future bucket D candidates need
   to clear the 16+16 mean-margin gate, not just the row-level
   parity gate.

## Artifacts

- `run.txt` — original phase-0 pre-demotion run (10/10 parity PASS, 4-pair distinguishability FAIL)
- `qwen3-revalidation.txt` — qwen3-8b-iq3m re-validated under the new mean-margin gate (PASS +0.084)
- `eval/reports/bucket-d-phi3-probe-2026-04-30/probe-mean-pool.py` — f16 anisotropy probe
- `eval/reports/bucket-d-phi3-probe-2026-04-30/phi-3.5-mini-ref.json` — last-token f16 ref (pinned for audit)
- `eval/reports/bucket-d-phi3-probe-2026-04-30/phi-3.5-mini-ref-mean.json` — mean-pool f16 ref (pinned for audit)

## Key commits

| sha       | type     | summary                                                      |
| --------- | -------- | ------------------------------------------------------------ |
| 7959873   | feat     | add `embeddingPooling` field (last-token / mean)             |
| 1dcf0ed   | feat     | thread `embeddingPooling` through engine load path           |
| 3efd6a1   | feat     | wire URL param + smoke + parity harness                      |
| 18346a6   | feat     | set phi-3.5-mini-q4km to mean-pool *(later reverted)*        |
| 4de154d   | test     | expand distinguishability harness to 16+16 pairs             |
| dd9bb39   | test     | pin phi-3.5-mini mean-pool parity refs                       |
| 031c2b9   | test     | switch distinguishability gate from strict-min to mean-margin|
| 247da5c   | fix      | demote phi-3.5-mini-q4km from bucket D                       |
