# Prefill-tile heuristic — replace dual-registry with hyperparam-derived default

**Date:** 2026-04-28
**Branch:** `main` (single-commit refactor; no side branch)
**Predecessors:** §22 (`feat/prefill-tiling-22`, ship-gated default-off plumbing),
§23 (`0c50e03`, default-on flip via `recommendedPrefillTile` registry field).
**TODO ref:** candidate-list item #4 (Heuristic-based prefill-tile default in
`ModelInference`).

## Summary

Replace the dual-source-of-truth pattern from §23 — `recommendedPrefillTile?: number`
field on `BenchmarkModel` plus a manually-mirrored `RECOMMENDED_PREFILL_TILE` map
in `smoke-test/real-model-page.js` — with a heuristic computed inside the
`ModelInference` constructor from `hp.layerCount` and `hp.embeddingLength`.
Adding a new model to the registry no longer requires editing a mirror map.

## Goal

Eliminate the dual-registry maintenance burden so a new model entry "just works"
without coordinated edits across `eval/models.ts` and
`smoke-test/real-model-page.js`. Preserve the §22+§23 behavioural contract on
every currently-registered model (5 at tile=128, 14 at tile=0).

## Non-goals

- Defining a new `prefillTileSize` value besides `0` and `128`. The §23
  resurrection path (a) "tile=64 fallback" is deferred until a future model
  actually trips the existing tile=128 budget; the heuristic is the natural
  encoding point at that time.
- Bytes-per-layer working-set proxy. Rejected during brainstorm — calibration
  is brittle and the §22 abort message's "available" value depends on
  preceding allocations, not a stable budget.
- Per-fleet GGUF-loading test guard. Rejected during brainstorm — couples
  test cost to model file I/O without adding meaningful safety over the
  boundary tests on synthetic hyperparams.

## Heuristic

A 2-axis rule mapping directly to the §22 abort signature ("32 layers × seq=512
of F32 intermediates exceeds graph allocator budget at `ggml-alloc.c:82`"):

```ts
export function computeDefaultPrefillTileSize(hp: ModelHyperparams): number {
  return hp.layerCount >= 32 && hp.embeddingLength >= 4096 ? 128 : 0;
}
```

Each gate has a physical justification we can cite in a comment above the
helper: layer count drives the number of F32 intermediate tensors per prefill
graph; embedding width drives each one's per-position size. Either gate alone
keeps the per-tile working set below the allocator's available budget.

## Override surface (unchanged from §23)

Three already-existing surfaces continue to win over the heuristic, in priority:

1. `new ModelInference(wasm, hp, { prefillTileSize: N })` — explicit ctor opt.
2. `?prefillTile=N` URL param on the smoke page (passes through to the ctor).
3. `--prefill-tile <n>` flag on `eval/perf.ts` (passes through to the ctor).

Force-disable path: pass `0` explicitly through any of the above.
Force-enable path: pass `128` (or any positive integer) explicitly.

## Files modified

| File | Change |
|------|--------|
| `src/inference/model-inference.ts` | Add exported `computeDefaultPrefillTileSize` helper. Change ctor default from `?? 0` to `?? computeDefaultPrefillTileSize(hp)`. |
| `eval/models.ts` | Delete `recommendedPrefillTile?: number` field from `BenchmarkModel` interface. Delete the field from 5 entries (mistral-7b q4ks/q3km/iq4xs, llama-3.1-8b-iq3m, qwen3-8b-iq3m). Remove the docstring block describing the dual-registry mirror requirement. |
| `smoke-test/real-model-page.js` | Delete `RECOMMENDED_PREFILL_TILE` map and the `else if (RECOMMENDED_PREFILL_TILE[modelId] !== undefined)` fallback branch. Smoke now passes `prefillTileSize` to the ctor only when `?prefillTile=` is set explicitly; otherwise the ctor's heuristic decides. The tile pill (shown iff `prefillTileSize > 0`) keeps current logic — still correct because the ctor sets the field. |
| `eval/perf.ts` | Drop the `model.recommendedPrefillTile` fallback in the `--prefill-tile` resolution. Pass `--prefill-tile` only when explicitly given. |
| `tests/eval-models.test.ts` | Delete the `describe("recommendedPrefillTile auto-default", ...)` block (2 tests). |
| `tests/prefill-tiling-config.test.ts` | Add 3 boundary tests on synthetic hyperparams (see Test plan). |

Net: ~−31 lines.

## Test plan

Add a `describe("prefillTileSize heuristic default", …)` block in
`tests/prefill-tiling-config.test.ts`, alongside the existing ctor-opt
validation tests:

1. **Both gates pass → 128.** `{ layerCount: 32, embeddingLength: 4096 }` →
   `inf.prefillTileSize === 128`.
2. **Either gate fails → 0.** Three sub-cases:
   - `{ layerCount: 31, embeddingLength: 4096 }` (below layer threshold).
   - `{ layerCount: 32, embeddingLength: 2048 }` (below embedding threshold).
   - `{ layerCount: 16, embeddingLength: 2048 }` (both below).
3. **Explicit ctor opt overrides heuristic.** `{ layerCount: 32,
   embeddingLength: 4096 }` plus `{ prefillTileSize: 0 }` → `inf.prefillTileSize
   === 0`.

Use the existing `STUB_WASM` and `STUB_HP` fixtures from the same test file.
Tests run under Bun; no browser, no GGUF parsing, no `navigator.gpu`.

Expected `make checkall` count after refactor: **428 pass / 11 skip / 0 fail**
(was 427; net is +1: −2 deleted in `eval-models.test.ts`, +3 added in
`prefill-tiling-config.test.ts`).

## Validation

### Phase A — static (no browser)

Required step **before any code edit**:

A.1 **Pre-edit hyperparam probe.** The brainstorm worked from informed-guess
`layerCount`/`embeddingLength` values. Validate against actual GGUF metadata
for all 19 registered (loadable) models before committing the constants.
Implementation options:

- One-shot script that loads each registered GGUF via the existing parser,
  prints `{id, layerCount, embeddingLength, current_recommended_tile,
  heuristic_tile}`, and asserts every row has `current === heuristic`.
- Or a temporary test that loads each model's metadata and runs the assertion
  inline; deleted after validation.

If any model misclassifies, halt and resolve before any code edit:
either tighten the threshold (e.g. raise the layer gate to 36 if a sub-7B
model ships with 32 layers + 4096 embedding) or special-case that model
inside the heuristic with a comment (`// model X needs Y because Z`). Do not
proceed to code edits with a known-wrong heuristic.

A.2 `make checkall` after edits. Expected 428 / 11 / 0.

### Phase B — browser smoke regression

After A.2 passes, rebuild the bundle (TS-only change; no WASM rebuild needed)
and run four manual checks via agentchrome on the existing smoke session:

B.1 **7B+ auto-default still works.** Navigate to `?model=mistral-7b-instruct-v0.3-q4ks`
with no `?prefillTile=` param. Confirm the `tile: 128` pill is visible and
prefill completes (no `ggml-alloc.c:82` abort). Decode coherent.

B.2 **Sub-7B fast path preserved.** Navigate to `?model=tinyllama-1.1b-chat-q4_0`
with no `?prefillTile=` param. Confirm no tile pill, prefill completes, TTFT
in normal range (no §22 +81% regression).

B.3 **Force-disable still works.** Navigate to
`?model=qwen3-8b-iq3m&prefillTile=0`. Confirm no tile pill and that prefill
aborts with the §22 `ggml-alloc.c:82` signature (override path healthy).

B.4 **Force-enable still works.** Navigate to
`?model=tinyllama-1.1b-chat-q4_0&prefillTile=128`. Confirm `tile: 128` pill
visible. Prefill should still complete (single-graph fits anyway); the test
is whether the override threads through, not whether it helps.

If any B step fails, `git reset --hard HEAD~1` and reopen the brainstorm.

## Commit

Single commit on `main`:
`refactor(prefill-tile): replace dual-registry pattern with hyperparam heuristic`.

Body cites §22 (abort cause), §23 (dual-registry pattern this replaces),
and §29's resumption-checklist note that this was candidate-list item #4.
The same commit also updates TODO.md with a §30 closure entry: candidate-list
item #4 flipped from "open / nice-to-have" to CLOSED, plus a one-line note
on the resumption checklist's "all closed" §-list.

## Risk and rollback

| Risk | Likelihood | Mitigation |
|------|-----------:|------------|
| Heuristic misclassifies a registered model | Low (Phase A.1 catches before edit) | Halt, re-tune, re-validate before edits |
| Future model breaks heuristic | Medium-low | Three override surfaces unchanged; doc comment on heuristic points at §22 |
| Sub-7B regression (accidental tile=128) | None on registered fleet | Phase B.2 verifies |
| 7B+ regression (auto-default lost) | None on registered fleet | Phase B.1 verifies |

**Rollback:** `git revert <sha>`. Zero schema migration, zero binary artifact
change, no WASM rebuild needed.

## Out of scope, recorded

- Bytes-per-layer working-set proxy (rejected approach 3).
- Per-model registry override field (rejected b2/b3 — would re-create the
  dual-source problem).
- Per-fleet GGUF-loading test (rejected c2 — coupling cost > safety gain).
- §22 future-resurrection path (a) "tile=64 fallback" — defer until a model
  trips it; encoded in the heuristic at that time.
- §23 future-resurrection path (c) "bundle the smoke map into the bundle" —
  moot once the map is gone.

## Brainstorm trail

Three sub-decisions, each picked with three approaches presented:

- **(a) threshold form** → Approach 2 (2-axis rule). Picked over single-axis
  product (1) and bytes-per-layer proxy (3) because it maps directly to the
  §22 abort signature and each axis is independently tunable.
- **(b) override surface** → b1 (ctor-only). Picked over b2 (keep both as
  belt-and-suspenders, strictly worse) and b3 (drop smoke map only, keep eval
  field; introduces same-model-different-default footgun).
- **(c) test surface** → c1 (boundary tests on synthetic hyperparams in
  `prefill-tiling-config.test.ts`). Picked over c2 (per-model fleet guard,
  overkill) and c3 (delete only, no replacement, no regression guard).
