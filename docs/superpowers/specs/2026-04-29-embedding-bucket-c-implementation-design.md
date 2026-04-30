# Embedding bucket C — Qwen3-Embedding-0.6B implementation design

**Date:** 2026-04-29
**Status:** approved (brainstorming complete; awaiting plan)
**Phase 0 probe:** CLOSED 2026-04-29; see
`eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`.
**Predecessor specs:** bucket B at
`docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`;
Phi-3 closure at
`docs/superpowers/specs/2026-04-29-phi3-causal-lm-support-design.md`
(closest causal-LM precedent).

## 0. Goal & non-goals

**Goal.** Land Qwen3-Embedding-0.6B as the first causal-LM-derived
embedder in webllm. End-to-end: registration → forward path → engine
dispatch → parity-gate-validated against the Stage 2 reference vectors
(≥0.999 per-row cosine) → bench/dashboard integration → closure report.

**In scope.**

- New architecture enum entry `qwen3-embedding`.
- New file `src/inference/causal-embedder-inference.ts` housing the
  `CausalLMEmbedder` class (sibling to `EncoderInference`).
- `engine.embed(modelId, text)` widened to dispatch causal-embedder
  architectures alongside encoder architectures.
- `eval/models.ts` registration of `qwen3-embedding-0.6b-q0f16` and
  addition to `eval/smoke-profiles.ts`.
- Parity gate harness (`eval/causal-embedder-parity.ts`) consuming
  `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`.
- Tokenizer routing — reuse existing Qwen3 BPE pipeline as-is. Probe
  surfaced no divergence; Phase 1 verifies cleanly.
- `eval/embed-perf.ts` extension for causal-embedder bench coverage;
  dashboard Embeddings section gets a 7th row.
- Closure report at
  `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`.

**Non-goals (Phase 7+ / future cycles).**

- Qwen3-Embedding-4B / 8B variants. Gated on 0.6B passing parity + bench;
  separate spec when queued.
- Other causal-LM-derived embedders: `gte-Qwen2-*`, `e5-mistral-*`,
  `nomic-embed-code`. Each is a separate registration pass with its own
  arch enum entry; this spec does not cover them.
- Chunked / batched embed dispatch (single-text encoding only this cycle).
- F16 readback path for embed (today's f32-readback is the supported path).
- Matryoshka dimension truncation.
- Continuous-CI parity gate — bucket B precedent is one-time validation,
  not checked-in.

**Success criteria.**

- 10/10 reference fixtures (5 doc + 5 query) pass per-row cosine ≥0.999.
- Smoke-bench p50/p90 single-text wall-time captured for downstream
  observability (no specific gate; embed-perf isn't a tok/s workload).
- Browser smoke (`make smoke-serve` + `real-model.html`) successfully
  encodes a query and a document through the same `engine.embed()` API
  used by the parity harness.
- All `make checkall` (fmt + lint + typecheck + test) passes.

## 1. Locked design decisions (from brainstorming)

| Decision | Choice | Reason |
|---|---|---|
| API dispatch architecture | (b) New `CausalLMEmbedder` class wrapping `ModelInference`'s patterns; sibling to `EncoderInference`. | Symmetry with bucket B; clean separation from chat path; CLAUDE.md "load-bearing invariants" doctrine — `ModelInference` not modified. |
| Architecture enum representation | (b) Add `qwen3-embedding` to `ModelArchitecture` union; arch-enum-driven dispatch. | More explicit, more grep-friendly, harder to misroute. Each future causal-LM embedder pays a code-change tax (acceptable). |
| Parity gate strictness | (a) Per-row cosine ≥0.999. | Matches bucket B precedent; webllm f32 readback historically lands 1e-5 to 1e-6 numerical drift; well inside the gate. |
| Phasing structure | (1) Phi-3 6-phase mirror. | Most recent comparable precedent; produces revertable commits at natural unit boundaries. |

## 2. Components & files affected

### New files

- **`src/inference/causal-embedder-inference.ts`** (~250-400 LOC).
  - Class `CausalLMEmbedder` mirroring `EncoderInference`'s shape.
  - Constructor takes `wasm: GgmlWasm`, `hyperparams: ModelHyperparams`,
    raw weights from the loader. Throws if `architecture` is not in the
    new `CAUSAL_EMBEDDER_ARCHITECTURES` set.
  - `loadWeights(rawWeights, ...)` — reuses the existing Qwen3
    weight-loading path (no fused-projection; identical to chat Qwen3).
    Skips loading `weights.output` / `weights.outputBias` since the
    `lm_head` matmul is bypassed.
  - `embed(ids: Int32Array): Float32Array` — runs forward through 28
    transformer layers, taps hidden state at post-output-norm (tap-point
    #2 from probe), pools last-token, L2-normalizes, returns
    `Float32Array(1024)`.
  - Internal helper `forwardEmbed(ids)` — mirrors `forwardSingle` from
    `model-inference.ts` but: (a) builds the graph with no KV cache
    (single-pass over all tokens); (b) returns a tensor pointer to `cur`
    *before* `lm_head`; (c) does not invoke the sampling stack.
- **`eval/causal-embedder-parity.ts`** (~100-150 LOC).
  - Drives `engine.embed("qwen3-embedding-0.6b-q0f16", text)` for the
    5 fixtures × 2 modes (10 calls).
  - Loads `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`.
  - Per-row cosine computation (f32 dot product; both sides
    L2-normalized so cosine == dot).
  - Pass/fail report with magnitude check + cosine column. Halts
    non-zero exit on first failure with diagnostic output indexed by
    Signature A/B/C from `encoder-cosine-degradation-signatures`.
- **`tests/causal-embedder-inference.test.ts`** (~80-150 LOC).
  - Constructor reject for non-causal-embedder architecture.
  - `loadWeights` succeeds against a registered Qwen3-Embedding fixture
    (or skips if GGUF absent — bucket B precedent).
  - `embed()` returns dim-1024 Float32Array with magnitude 1.0 ± 1e-3.
- **`eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`**
  (closure report; produced in Phase 6).

### Modified files

- **`src/core/types.ts`** — add `qwen3-embedding` to `ModelArchitecture`
  union; define `CAUSAL_EMBEDDER_ARCHITECTURES` constant +
  `isCausalEmbedderArchitecture()` helper. The hyperparams pooling field
  carries `"last-token"` for this arch.
- **`src/core/engine.ts`** — add
  `causalEmbedderEngines: Map<string, CausalLMEmbedder>` map. Widen
  `embed(modelId, text)` to dispatch through the architecture-keyed
  ladder: encoder → causal-embedder → throw. Mirror `loadModel` plumbing
  to instantiate `CausalLMEmbedder` for matching architectures.
- **`src/models/model-loader.ts`** — architecture mapping:
  `qwen3-embedding` recognized in `extractHyperparams`. Read
  `qwen3.pooling_type` and surface as the typed pooling enum. The
  derivation rule from GGUF metadata is pinned in Phase 1 based on the
  probe's findings (likely: `general.architecture = qwen3` AND
  `qwen3.pooling_type` present and ≠ NONE → embedder; otherwise chat).
  Tokenizer load path: confirm Qwen3 BPE works for the embedder's
  vocab.
- **`eval/models.ts`** — register `qwen3-embedding-0.6b-q0f16`.
  Capabilities: `{ embedding: true, toolCalling: false,
  structuredOutput: false, vision: false }`. Family: `Qwen3-Embedding`.
  Mirror URL: pinned in Phase 1.
- **`eval/smoke-profiles.ts`** — add embedder smoke profile mirroring
  arctic-embed / bge profile entries.
- **`eval/embed-perf.ts`** — add `qwen3-embedding-0.6b-q0f16` to
  `ENCODER_MODELS` (rename to `EMBEDDER_MODELS` if consistent with the
  broader fleet — Phase 5 decision).
- **`smoke-test/real-model.html`** + **`smoke-test/real-model-smoke.js`**
  — engine routing for causal embedders mirrors the encoder routing
  path. Window-globals exposure if the parity harness drives the smoke
  page.
- **`TODO.md`** — item 5 closure stub updated post-validation.

### Files explicitly NOT touched

- `src/inference/model-inference.ts` — preserved untouched. The
  `CausalLMEmbedder` is a sibling, not a mode flag on the existing
  class. Per the API decision (b) and CLAUDE.md "load-bearing
  invariants" doctrine.
- `src/inference/encoder-inference.ts` — preserved; the new class is
  independent.
- `src/inference/sampler.ts`, `src/inference/generation.ts` — unused
  for embedders; not modified.

## 3. Phase-by-phase execution shape

Six phases. Each ends with a defined commit (or, for validation phases,
a pass/fail gate with no commit).

### Phase 1 — Types + arch enum + registration

Single commit: `feat(types): add qwen3-embedding architecture + register Qwen3-Embedding-0.6B`.

- Add `qwen3-embedding` to `ModelArchitecture` union in
  `src/core/types.ts`.
- Define
  `CAUSAL_EMBEDDER_ARCHITECTURES = ["qwen3-embedding"] as const` and
  `isCausalEmbedderArchitecture()` helper.
- Add a `pooling: PoolingType` field to hyperparams (or extend the
  encoder's existing pooling enum) so causal embedders carry
  `"last-token"`.
- Update `model-loader.ts:extractHyperparams` to recognize the embedder.
  Pin the derivation rule explicitly in the Phase 1 plan based on the
  probe's metadata findings.
- Register `qwen3-embedding-0.6b-q0f16` in `eval/models.ts` with the
  Qwen mirror URL. Add embedder smoke profile to
  `eval/smoke-profiles.ts`.

**Exit:** `make checkall` passes; the new model appears in the eval
CLI's model list; no inference yet — calling `engine.embed()` would
still throw because Phase 2 hasn't shipped the class.

### Phase 2 — `CausalLMEmbedder` class

Single commit: `feat(inference): CausalLMEmbedder for Qwen3-Embedding`
(loader + forward + unit test as one logical unit, matching Phi-3's
Phase 2 cadence).

- Create `src/inference/causal-embedder-inference.ts`. Class skeleton,
  constructor, `loadWeights` (reuses Qwen3 weight-loading path),
  `embed(ids)`.
- Forward graph: build the standard Qwen3 forward through 28 layers,
  with the `lm_head` matmul *omitted*. Tap hidden state at
  post-output-norm. Apply last-token pooling: select column at
  `ids.length - 1` from the `[E, N]` hidden tensor. L2-normalize.
- Add `tests/causal-embedder-inference.test.ts` covering: arch-mismatch
  reject; `loadWeights` success on fixture; `embed()` returns
  dim-1024 Float32Array with unit magnitude.

**Exit:** unit tests pass; `make checkall` clean. Engine can't dispatch
yet (Phase 3 work).

### Phase 3 — Engine routing + tokenizer wiring

Single commit:
`feat(engine): widen embed() dispatch + smoke routing for causal embedders`.

- `engine.ts:loadModel` instantiates `CausalLMEmbedder` for
  `isCausalEmbedderArchitecture(arch)` and stores in
  `causalEmbedderEngines`.
- `engine.embed(modelId, text)` widened with three-way dispatch:
  encoder → causal-embedder → throw. Tokenizer routing reuses the
  existing Qwen3 BPE path; confirm the embedder GGUF's tokenizer loads
  cleanly without bert-style cls/sep fallback.
- Update `smoke-test/real-model.html` + `smoke-test/real-model-smoke.js`
  to route causal embedders correctly. Window-globals exposure for
  harness drive-through.

**Exit:** `engine.embed("qwen3-embedding-0.6b-q0f16", "hello")` returns
a Float32Array of length 1024, magnitude ≈ 1.0. No parity check yet.

### Phase 4 — Parity gate

Pass/fail validation phase. Two commits:
1. `feat(eval): causal-embedder parity harness` — harness ships.
2. `docs(probe): bucket-c parity validation 10/10 PASS` — validation
   report with cosine table.

- Create `eval/causal-embedder-parity.ts`. Loads
  `qwen3-embedding-0.6b-ref.json`. Drives `engine.embed()` for 5 fixtures
  × 2 modes through the smoke page. Computes per-row cosine. Reports
  pass/fail.
- Run the harness end-to-end against the live smoke page
  (`make smoke-serve` + agentchrome). Capture results in
  `eval/reports/qwen3-embedding-validation-2026-04-29/PARITY.md`.

**Exit gate:** 10/10 rows ≥0.999 cosine. **If <10/10 pass, halt and
diagnose** via the encoder-cosine-degradation-signatures ladder
(Signature A → positional encoding; Signature B → activation/norm;
Signature C → instruction-prefix application). Do not proceed to Phase
5 until the gate clears.

### Phase 5 — Bench + dashboard

Single commit: `feat(eval): embed-perf coverage for Qwen3-Embedding`.

- Extend `eval/embed-perf.ts` to cover `qwen3-embedding-0.6b-q0f16`.
- Run `embed-perf` and capture p50/p90 single-text-short timings.
- Verify dashboard Embeddings section picks up the new model row
  automatically (bucket B precedent: dashboard reads live DB; no
  per-model dashboard code change).

**Exit:** embed-perf report committed at
`eval/reports/embed-perf-2026-04-29-qwen3/`; dashboard shows the row.

### Phase 6 — Closure report

Two commits:
1. `docs(probe): bucket-c implementation closure report` —
   `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`.
2. `docs(TODO): close bucket-c implementation` — TODO.md item 5 update.

- Closure report captures: probe → impl → parity (10/10 pass) → bench
  (p50/p90) → recommendation for 4B/8B variants and other causal-LM
  embedders.
- TODO.md item 5 closure stub: replace current "Phase 0 closed" stub
  with full closure stub including bench numbers + Phase 7+ posture.

**Exit:** SUMMARY.md committed; TODO updated; spec marked complete.

## 4. Parity gate detail

Phase 4's gate is the load-bearing contract. Specifying explicitly so
the implementation has zero ambiguity.

### Reference vectors

`eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`.
10 records: 5 rows × 2 modes (interleaved), each
`{row, input, mode, vec}` with `vec` a length-1024 f32 list.

### Per-row gate

For each of the 10 records:

1. Load the fixture's `input` and `mode`.
2. If `mode === "query"`, prepend the instruction prefix exactly as
   serialized in the JSON's `instruction_prefix` field. The harness
   reads the prefix from the JSON, **not** hard-coded — drift-resistance.
3. Call `engine.embed(modelId, prefixed_or_raw_text)`. Returns
   Float32Array(1024).
4. Compute cosine = `dot(webllm_vec, ref_vec)` (both already
   L2-normalized; cosine == dot).
5. Assert cosine ≥ **0.999**. Magnitude assert webllm_vec L2-norm ≈
   1.0 ± 1e-3.

### Reporting

Phase 4's `PARITY.md` writes a table:

```
| Row | Mode  | Input (truncated) | Cosine   | Pass |
|----:|-------|-------------------|---------:|:----:|
|   0 | doc   | Hello world.      | 0.9999X  |  Y   |
|   0 | query | Hello world.      | 0.9999X  |  Y   |
| ... | ...   | ...               | ...      | ...  |
```

Plus a one-line summary: `Result: 10/10 PASS at ≥0.999 cosine.`

### Failure handling — diagnostic ladder

If any row fails, halt the harness with diagnostic indexed by signature:

1. **All 10 rows fail uniformly low (<0.5).** Signature C — usually
   instruction-prefix not applied OR tokenizer mismatch. Check:
   `repr(prefix_used)` matches the JSON's `instruction_prefix`. If yes,
   suspect tokenizer divergence — diff token IDs from webllm's BPE vs
   the reference (sentence-transformers tokenizer).
2. **Doc rows pass, query rows fail.** Instruction prefix specifically
   broken — the LF or `Query:` byte sequence wasn't emitted as expected.
   Check `repr()` of the actual byte sequence passed to the model.
3. **All 10 rows land 0.95-0.99.** Signature B — activation/norm
   mismatch. Check: pre vs post output_norm tap (did Phase 2 take
   tap-point #2 = post, not #1 = pre?). Check: RoPE freq_base = 1000000
   (Qwen3-specific, not 10000).
4. **Length-monotonic degradation (short fixtures pass, long fail).**
   Signature A — RoPE mode wrong (should be NEOX) or position-IDs
   off-by-one in the tap.
5. **Magnitude failure (`|v|_2 ≠ 1.0`).** L2-normalization missing in
   `embed()`. Check: `EncoderInference.poolAndNormalize` is reused or
   the equivalent is implemented in `CausalLMEmbedder.embed()`.
6. **Mode mismatch (only one mode ever passes).** Tap-point applied
   differently per mode somehow — shouldn't be possible since the
   prefix is just text the tokenizer turns into more tokens; flag for
   investigation.

The diagnostic ladder is documented inline in the harness file as a
comment block so future regressions land at the right rung without
re-deriving the signature map.

### No CI gate

Per bucket B precedent, the parity harness is a one-time validation,
not a continuous-CI fixture. The reference vectors are heavy (228 KB)
and the harness depends on a live smoke server + GGUF download, neither
of which fit cleanly in `bun test`. Re-run manually if any forward-graph
change touches the load-bearing path.

## 5. Risk register

Risks the spec explicitly addresses:

| # | Risk | Mitigated in | Failure mode if missed |
|---|---|---|---|
| 1 | `qwen3-embedding` arch derivation rule unclear (GGUF says `qwen3`, not `qwen3-embedding`) | Phase 1 | Loader can't distinguish embedder from chat at registration time → wrong dispatch in `engine.embed()` |
| 2 | Tap-point #2 (post-output-norm) not actually what `qwen3.cpp` does | Phase 2 + Phase 4 parity | Cosine fails Signature B uniformly across all rows; rebuilds Phase 1-2 |
| 3 | Last-token pooling picks wrong token (off-by-one on `ids.length-1`, padding-token, or BOS) | Phase 2 + Phase 4 | Cosine fails per-row inconsistently |
| 4 | Instruction-prefix tokenizer round-trip drift (LF byte encodes differently than reference assumes) | Phase 3 + Phase 4 | Doc rows pass, query rows fail uniformly |
| 5 | Engine `embed()` dispatch ladder accidentally still throws `EncoderRequiredError` for causal embedders | Phase 3 | `engine.embed()` throws at runtime — caught by Phase 3 exit assertion |
| 6 | F16 vs F32 numerical drift through 28 layers exceeds 0.999 cosine tolerance | Phase 4 (gate decides) | Gate fails; adjudicate (relax to 0.998 with documented reason, or root-cause and fix) |
| 7 | `make checkall` lint/typecheck regression from new arch enum / new file | Phase 1 + Phase 2 | Caught at phase exit; blocks commit |
| 8 | Smoke page routing for causal embedders breaks chat models (cross-talk through shared `engine.ts`) | Phase 3 | Smoke chat tests fail; rollback Phase 3 commit |

Risks deliberately deferred:

- **a. KV-cache allocation when forward path doesn't need it.**
  `ModelInference` allocates KV-cache buffers eagerly; the embedder
  forward path won't use them. Phase 2's `CausalLMEmbedder` should *not*
  allocate KV cache. If it accidentally does, it's a memory waste but
  not a correctness bug — fix in Phase 2 if surfaced; not a load-bearing
  concern.
- **b. WebGPU graph-build cost amortization.** `forwardEmbed` rebuilds
  the graph each call. Phase 5 bench will measure; if cost is dominant,
  Phase 7+ work could cache. Out of scope this cycle.
- **c. Multi-text batching.** `engine.embed(modelId, text)` is
  single-text; bench will compare to bucket B p50/p90. Bulk encode is
  open question per probe report; Phase 7+ work.
- **d. F16 readback path.** webllm has only f32 readback today; F16
  path is a future cycle.
- **e. 4B and 8B variants.** Phase 6 closure report recommends, but
  registration is a separate spec.

### Phase-exit abort conditions

- **Phase 1:** if GGUF metadata can't reliably distinguish embedder
  from chat-Qwen3, halt and brainstorm a tighter derivation rule before
  Phase 2 ships.
- **Phase 4:** if 10/10 doesn't clear ≥0.999 after one full ladder pass,
  halt and surface to user before relaxing the gate.
- **Phase 5:** if `embed-perf` p50 is >10× bucket B's BGE-large
  baseline (which would suggest forward-graph rebuild is dominating),
  report it but don't gate on it — that's Phase 7+ optimization
  territory.

### Probe-already-resolved risks (spec inheritance from Phase 0)

- Pooling type confirmed: last-token.
- Output dim confirmed: 1024 (no projection head).
- Architecture confirmed: qwen3 family.
- Instruction-prefix runtime bytes pinned and serialized in JSON.
- Reference vectors validated at unit magnitude with margin 4 orders
  inside the gate.

## 6. Workflow integration

### Spec doc

This file. Force-add per `docs/superpowers/` gitignore convention.
Commit `docs(spec): bucket-c implementation design`.

### Plan doc

`docs/superpowers/plans/2026-04-29-embedding-bucket-c-implementation.md`.
Written via `superpowers:writing-plans` after spec approval. Force-add.
Six phases mirroring Section 3, each with bite-sized tasks (~2-5 minutes
per step) per the project's writing-plans cadence. Commit
`docs(plan): bucket-c implementation plan`.

### Execution skill

`superpowers:subagent-driven-development` per the global CLAUDE.md
preference. Per-phase implementer dispatch + two-stage review (spec
compliance + code quality). No mid-phase user checkpoints required by
the plan itself — global rule: "plan approval IS execution approval."

### Commit cadence per project doctrine

Each commit is its own — never bundled — so any single one is
revertable.

| # | Commit message | Phase |
|---|---|---|
| 1 | `docs(spec): bucket-c implementation design` | (this brainstorm) |
| 2 | `docs(plan): bucket-c implementation plan` | (writing-plans) |
| 3 | `feat(types): add qwen3-embedding architecture + register Qwen3-Embedding-0.6B` | Phase 1 |
| 4 | `feat(inference): CausalLMEmbedder for Qwen3-Embedding` | Phase 2 |
| 5 | `feat(engine): widen embed() dispatch + smoke routing for causal embedders` | Phase 3 |
| 6 | `feat(eval): causal-embedder parity harness` | Phase 4 (harness commit; gate run is no-commit) |
| 7 | `docs(probe): bucket-c parity validation 10/10 PASS` | Phase 4 (validation report) |
| 8 | `feat(eval): embed-perf coverage for Qwen3-Embedding` | Phase 5 |
| 9 | `docs(probe): bucket-c implementation closure report` | Phase 6 |
| 10 | `docs(TODO): close bucket-c implementation` | Phase 6 (TODO update) |

### Cross-links to land in vault

Post-implementation, optional cadence work, not part of this spec:

- If Phase 4 surfaces a generalizable lesson beyond what
  `causal-lm-embedder-fstring-prefix-rendering.md` (saved earlier this
  session) already covers, save a follow-up note. Likely candidate:
  "CausalLMEmbedder forward-graph reuses ModelInference weight-loading
  but bypasses lm_head + KV cache — sibling-class pattern over
  mode-flag pattern" if Phase 2 lands cleanly.

### TODO.md surface update

Item 5 currently shows "Phase 0 closed; Phase 1+ queued." Phase 6
closure updates it to one of:

- Happy path: "Bucket C **CLOSED <date>** — see
  `eval/reports/qwen3-embedding-validation-2026-04-29/SUMMARY.md`. 10/10
  parity at ≥0.999. p50 short-text embed: <ms>. 4B/8B variants and
  other causal-LM embedders queued as separate registration cycles."
- With surfaced concerns documented inline if Phase 4 or 5 surface
  notable findings.

## 7. Out of scope (recap)

- Anything requiring a forward-graph cache or batched dispatch — Phase 7+
  if it ever lands.
- The 4B/8B variants — separate spec when queued.
- Other causal-LM embedders (`gte-Qwen2-*`, `e5-mistral-*`,
  `nomic-embed-code`) — separate registration passes when queued, each
  register-and-run if the (b) arch-enum-per-family decision generalizes.
- Continuous-CI parity gate.
- F16 readback for embed.
- Matryoshka dimension truncation.

## 8. References

- TODO.md item 5 (Phase 0 closed; this spec executes Phase 1+).
- Spec predecessor: bucket B
  `docs/superpowers/specs/2026-04-28-encoder-non-bert-arch-design.md`.
- Spec predecessor: Phi-3
  `docs/superpowers/specs/2026-04-29-phi3-causal-lm-support-design.md`.
- Probe spec:
  `docs/superpowers/specs/2026-04-29-embedding-bucket-c-phase-0-probe-design.md`.
- Probe outcome:
  `eval/reports/bucket-c-probe-2026-04-29/STAGE-2-REFERENCE-VECTORS.md`.
- Reference vectors:
  `eval/reports/bucket-c-probe-2026-04-29/qwen3-embedding-0.6b-ref.json`.
- Vault: `~/ClaudeVault/Patterns/causal-lm-embedder-fstring-prefix-rendering.md`
  (saved this session).
- Vault: `~/ClaudeVault/Patterns/encoder-parity-gate-via-sentence-transformers.md`.
- Vault: `~/ClaudeVault/Patterns/llama-cpp-as-arch-truth-source.md`.
- Vault: `~/ClaudeVault/Knowledge/encoder-cosine-degradation-signatures.md`
  (Phase 4 diagnostic ladder).
