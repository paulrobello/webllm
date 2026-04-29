# Embedding bucket C — Phase 0 probe design

**Date:** 2026-04-29
**Status:** approved (brainstorming complete; awaiting plan)
**Scope:** Phase 0 probe only. Phase 1-5 implementation specs are downstream
artifacts gated on this probe's exit recommendation.

## Goal

Land the artifacts that gate Phase 1-5 implementation of Qwen3-Embedding-0.6B
causal-LM-derived embedder support, *without* writing any production code.

Two staged deliverables (Stage 1 → user checkpoint → Stage 2) following the
project's `cap-probe-bump-first-doctrine` pattern: characterize metadata
before committing reference-vector capture to a fixture mode.

## Background

- **Bucket A (closed 2026-04-28):** BGE-{small,large}-en-v1.5 — register-and-run.
- **Bucket B (closed 2026-04-28):** jina-embeddings-v2-base-en (BERT+ALiBi+GeGLU)
  and nomic-embed-text-v1.5 (BERT+NEOX-RoPE+SwiGLU+fused-QKV) — added
  `EncoderInference` arch generalization, parity gate via
  sentence-transformers reference vectors at cosine ≥0.999.
- **Bucket C (this probe):** causal-LM-derived embedders. Qwen3-Embedding tops
  MTEB at 0.6B-8B as of 2026; landing this lever unlocks `gte-Qwen2-*`,
  `e5-mistral-*`, and any future causal-LM embedder using last-token pooling.

The bucket-C wrinkle vs A/B: causal-LM forward path runs through
`ModelInference` (KV-cache, sampling stack), not `EncoderInference`. Embed
mode requires bypassing the sampler and tapping hidden state directly.

## Deliverables

### Stage 1 artifact

`eval/reports/bucket-c-probe-2026-04-29/STAGE-1-METADATA.md`:

- **Embed-surface analysis.** 2-3 candidate tap-points in
  `ModelInference.forward()` for hidden-state extraction, each documented
  with: insertion graph node, tensor shape (`[E, N]`), residual-stream state,
  output-norm applied y/n, dispatch implications. Cross-referenced with
  `~/Repos/llama.cpp/src/models/qwen3.cpp` (or nearest equivalent) per the
  `llama-cpp-as-arch-truth-source.md` vault doctrine. Recommendation column.

- **Qwen3-Embedding-0.6B metadata table.** Two-column comparison:
  - **Intended (HF config):** read from `Qwen/Qwen3-Embedding-0.6B`
    `config.json` + `tokenizer_config.json` + README.
  - **As-shipped (GGUF):** parsed via `GgufParser.parse()` against the
    selected GGUF mirror.

  Rows: pooling type, normalization, projection-head presence + dim, RoPE
  config (mode/base/dim), vocab/special-token IDs (incl. `eos_token_id`,
  load-bearing for last-token pooling), `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`/`num_kv_heads`, FFN type, instruction-prefix
  conventions for query/document modes.

  Divergences flagged in red — bucket-B-style mirror-gap detection.

- **Stage 2 plan refinement.** Based on metadata, finalize fixture-mode
  count (default 2 modes; expand if Qwen3-Embedding documents more pooling
  modes than expected) and pin the exact instruction-prefix string the
  documentation prescribes for query mode.

### Stage 2 artifact

`eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md` plus
`eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`:

- 5 bucket-B fixtures captured under **document mode** (raw input).
- 5 bucket-B fixtures captured under **query mode** (instruction-prefixed
  per Stage-1-pinned string).
- Total: 10 reference vectors of dim `hidden_size`.
- L2-normalized via sentence-transformers `normalize_embeddings=True`
  (matches bucket-B harness convention).
- Per-row magnitude + dim verification before serialization.
- Probe-conclusion section recommending: proceed to Phase 1 immediately,
  pause for additional probing, or surface a scope concern.

## Stage 1 execution shape

**Inputs.** `src/inference/model-inference.ts`,
`src/inference/encoder-inference.ts`, `src/core/types.ts`,
`src/loader/gguf-parser.ts`, HF config for `Qwen/Qwen3-Embedding-0.6B`,
selected GGUF mirror.

**Steps.**

1. **Embed-surface analysis (read-only).** Walk `ModelInference.forward()`
   and identify candidate tap-points. Document each as a row in the
   tap-point table. Cross-reference llama.cpp qwen3 source as architecture
   truth source. Land 2-3 options with a recommendation column.

2. **GGUF mirror selection.** Confirm one mirror exists with f16 or Q8_0
   weights; document the URL convention (`ggufUrl` + file picker pattern
   matching `eval/models.ts`). Halt and surface if no clean mirror is
   available — Stage-1 exit condition.

3. **Download GGUF + parse.** Use existing tooling. Dump metadata via
   `GgufParser.parse()` and emit raw key-value pairs into the report
   filtered to embedding-relevant keys.

4. **Metadata table assembly.** Two-column comparison flagging any
   divergence in red.

5. **Stage 2 plan refinement.** Finalize fixture-mode count and pin the
   exact instruction-prefix string.

**Out of scope for Stage 1.**
- Any webllm-side code change (no `src/`, `eval/`, or `smoke-test/` writes
  outside the new `eval/reports/bucket-c-probe-2026-04-29/` directory).
- Reference-vector capture (Stage 2).
- Phase 1 implementation strategy decisions beyond surfacing options.

**Stage-1 exit criteria.**
- Report committed.
- Metadata table populated for all rows.
- Embed-surface analysis lists ≥2 tap-point options with a recommendation.
- No open question blocks Stage 2.

## Stage 2 execution shape

**Prerequisite.** Stage 1 report committed; instruction-prefix string
pinned; pooling-mode count finalized.

**Inputs.** `eval/encoder-parity.ts` and
`eval/reports/encoder-parity-2026-04-28/{capture-refs.py,inputs.json}`
(drop-in reusable). Qwen3-Embedding-0.6B HF model fetched fresh (~600 MB).
Stage-1 artifacts.

**Steps.**

1. **Adapt `capture-refs.py` for Qwen3-Embedding.** New file
   `eval/reports/bucket-c-probe-2026-04-29/capture-refs.py` (copy + edit;
   no edits to bucket-B's). Two-mode loop:
   - **Document mode.** 5 fixtures passed raw to
     `model.encode(texts, normalize_embeddings=True)`.
   - **Query mode.** 5 fixtures wrapped with the Stage-1-pinned
     instruction-prefix string, passed to `model.encode`.

2. **Magnitude + dim sanity check.** Pre-serialization, assert
   `|v|_2 == 1.0` (within fp tolerance) and `dim == expected_output_dim`
   for all 10 vectors, where `expected_output_dim` is taken from Stage 1's
   metadata table (projection-head output dim if present, else
   `hidden_size`). Halt with diagnostic if either fails.

3. **Serialize to `qwen3-embedding-0.6b-ref.json`.** Schema:
   ```json
   {
     "model": "Qwen/Qwen3-Embedding-0.6B",
     "captured_with": "sentence-transformers <version>",
     "pooling": "<from stage 1>",
     "instruction_prefix": "<exact string from stage 1>",
     "fixtures": [
       { "row": 0, "input": "...", "mode": "document", "vec": [..] },
       { "row": 0, "input": "...", "mode": "query", "vec": [..] },
       ...
     ]
   }
   ```
   Single file; both modes interleaved by fixture row for easy diffing
   during Phase 3.

4. **STAGE-2-REFERENCE-VECTORS.md.** Tables for both modes (5 rows each)
   with input excerpt + magnitude + first 3 dims as a fingerprint.
   Probe-conclusion section recommending Phase 1 entry.

5. **Optional informational sub-step.** Pairwise cosine across the 10
   vectors (5×5 doc-vs-doc, 5×5 doc-vs-query). Reveals whether mode prefix
   has a stronger effect than fixture content. Pure documentation; no gate.

**Out of scope for Stage 2.**
- webllm-side embed parity gate execution (Phase 3 work).
- 4B/8B variant ref capture (gated on 0.6B Phase 3 success).
- Any code touching `src/` or `eval/` outside the new probe directory.
- Adding Qwen3-Embedding-0.6B to `eval/models.ts` registry (Phase 1 work).

**Stage-2 exit criteria.**
- 10 reference vectors committed and verified at unit magnitude.
- Report committed.
- Probe-conclusion section recommends a clear Phase 1 entry posture.

## Risk register

Risks the probe explicitly addresses:

| # | Risk | Stage that addresses it | Failure mode if skipped |
|---|---|---|---|
| 1 | GGUF mirror omits keys we depend on (bucket-B nomic precedent) | Stage 1 step 4 (HF-vs-GGUF column diff) | Phase 1 implementation crashes on weight load with a misleading error |
| 2 | Pooling type assumption wrong (last-token vs attention-pooled vs CLS-style) | Stage 1 step 3 (raw key-value dump) | Cosine fails Signature-A across all fixtures, hard to localize |
| 3 | Instruction-prefix convention guessed wrong | Stage 1 step 5 + Stage 2 step 1 | Cosine fails Signature-C uniformly across query-mode fixtures |
| 4 | Hidden-state tap-point at wrong residual-stream depth | Stage 1 step 1 (≥2 options + reco) | Phase 3 discovers cosine ~0.95 across all fixtures, has to rebuild Phase 1 |
| 5 | Reference vectors not L2-normalized → mismatch with `EncoderInference.poolAndNormalize`'s normalized output | Stage 2 step 2 (magnitude assertion) | Cosine fails uniformly at ~0 or systematically off — false signal of a model bug |
| 6 | Projection head present but not characterized → wrong output dim | Stage 1 step 4 (projection-head row in metadata table) | Phase 1 ships an embed surface with mismatched dim, fails dim-shape assertions |

Risks deliberately *not* addressed by this probe (deferred):

- **a. WebGPU graph build cost for causal-LM forward in embed mode.**
  Phase 3 surfaces; Phase 4 measures. Probe reports the predicted cost
  but doesn't gate on it.
- **b. Causal mask semantics under last-token pooling.** Theoretically
  correct (last token sees all prior tokens — exactly what we want
  pooled) but worth validating in Phase 2 against the parity gate.
  Probe flags as known-attention-point.
- **c. 4B/8B variant feasibility.** Out of scope per TODO; Phase 5 work.

## Abort conditions

**Stage-1 abort conditions** (probe halts and reports up):
- No GGUF mirror with f16/Q8_0 weights exists for Qwen3-Embedding-0.6B
  at probe time.
- HF config reveals an architecture genuinely outside the `qwen` family
  (e.g., custom architecture key).
- GGUF metadata reveals a pooling type webllm has no precedent for AND
  the architecture-truth-source path can't disambiguate.

**Stage-2 abort conditions:**
- Reference vectors fail unit-magnitude assertion → diagnose before
  committing the JSON.
- sentence-transformers refuses to load the HF weights → fall back to
  direct HF transformers loading, document the change.

**Probe success signal.** Stage 2 report's probe-conclusion section can
recommend "proceed to Phase 1" with all 6 risks above either resolved or
explicitly accepted-for-later.

## Workflow integration

**Spec doc.** This file. Force-add per `docs/superpowers/` gitignore
convention. Commit message `docs(spec): bucket-c phase-0 probe design`.

**Plan doc.** `docs/superpowers/plans/2026-04-29-embedding-bucket-c-phase-0-probe.md`,
written via the `superpowers:writing-plans` skill after spec approval.
Force-add. Two-stage plan structure mirroring the deliverables sections.
Commit `docs(plan): bucket-c phase-0 probe plan`.

**Execution skill.** `superpowers:subagent-driven-development` per the
global preference (CLAUDE.md: "always choose Subagent-Driven without
asking"). Plan steps execute as discrete subagent dispatches with
verification checkpoints.

**Commit cadence per project doctrine.**
1. `docs(spec): bucket-c phase-0 probe design` — after spec self-review +
   user approval.
2. `docs(plan): bucket-c phase-0 probe plan` — after writing-plans.
3. `docs(probe): bucket-c stage 1 — metadata` — after Stage 1 execution.
4. `docs(probe): bucket-c stage 2 — reference vectors` — after Stage 2
   execution.
5. `docs(TODO): close bucket-c phase-0 probe` — TODO.md item 5 update
   with closure stub linking the probe report.

Each commit is its own — never bundled — so any single one is revertable
without nuking adjacent reasoning.

**Cross-links to land in vault** (post-probe, optional cadence work, not
part of the probe itself):
- Save a probe-pattern note to `~/ClaudeVault/Patterns/` if Stage 1
  reveals a generalizable lesson (e.g., "causal-LM-derived embedder
  probe must dump GGUF metadata before fixture capture").
- Cross-link from existing notes
  (`encoder-parity-gate-via-sentence-transformers.md`,
  `encoder-architecture-probe-saved-spec-rewrite.md`,
  `encoder-cosine-degradation-signatures.md`) to the new probe report so
  future bucket-C-shaped probes find the prior art.

**TODO.md surface update.** TODO item 5 currently reads "queued
2026-04-29 as next-session focus"; after probe lands, that block updates
to either:
- "Phase 0 probe closed YYYY-MM-DD; Phase 1-5 plan queued" — happy path.
- "Phase 0 probe closed YYYY-MM-DD with surfaced concern: <one-liner>;
  Phase 1 deferred pending <decision>" — if Stage 1/2 reveals a scope
  concern.

## Out of scope for any of this work

- Phase 1-5 spec/plan/implementation — downstream, gated on probe-conclusion.
- 4B/8B variants and other causal-LM embedders (`gte-Qwen2-*`,
  `e5-mistral-*`).
- §D concat-graph batched encoder work (still external-trigger candidate).

## References

- TODO.md item 5 (queued 2026-04-29; this spec executes it).
- `eval/reports/encoder-parity-2026-04-28/SUMMARY.md` (bucket-B closure;
  template for fixture-set + capture-refs harness).
- `~/ClaudeVault/Patterns/encoder-parity-gate-via-sentence-transformers.md`.
- `~/ClaudeVault/Patterns/encoder-architecture-probe-saved-spec-rewrite.md`.
- `~/ClaudeVault/Patterns/llama-cpp-as-arch-truth-source.md`.
- `~/ClaudeVault/Knowledge/encoder-cosine-degradation-signatures.md`
  (Phase 3 diagnostic ladder if parity fails).
- `~/ClaudeVault/Patterns/cap-probe-bump-first-doctrine.md` (bump-first
  doctrine; analog applied here as characterize-first-capture-second).
- `~/ClaudeVault/Patterns/probe-first-methodology-validates-architecture-pivots.md`
  (general probe-first doctrine).
