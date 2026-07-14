# Audit Remediation Report

> **Project**: `@paulrobello/webllm` — browser-side LLM inference over WebGPU
> **Audit Date**: 2026-07-14 (AUDIT.md at commit `eccf6e6`)
> **Remediation Date**: 2026-07-14
> **Severity Filter Applied**: `all`, scoped per user to **"safe tier + backend/export calls"** — all security, the low-risk architecture/CI/gate-wiring/quality fixes, and the full documentation accuracy pass; the four large/decision-gated refactors deferred (see Requires Manual Intervention).
> **Branch**: `fix/audit-remediation` (base `eccf6e6` → HEAD `01a9a5d`, 5 commits, 70 files changed: 12 added, 5 deleted, 53 modified)

---

## Execution Summary

| Phase | Status | Agent (model) | Targeted | Resolved | Partial | Manual/Deferred |
|-------|--------|---------------|:--------:|:--------:|:-------:|:---------------:|
| 1 — Critical Security | ✅ | fix-security (opus) | 5 | 5 | 0 | 0 |
| 2a — Architecture core | ✅ | fix-architecture (opus) | 5 | 4 | 0 | DOC-015 (false positive) |
| 2b — Bundle hygiene + smoke gate | ✅ | fix-architecture (opus) | 2 | 2 | 0 | 0 |
| ARC-015 prelude | ⏭️ Reverted | fix-architecture (sonnet) | 1 | 0 | 0 | 1 (43 errors > guardrail) |
| 3a — Security (remaining) | ✅ | fix-security (sonnet) | 1 | 1 | 0 | 0 |
| 3b — Architecture (remaining) | ✅ | fix-architecture (opus) | 2 | 2 | 0 | 0 |
| 3c — Code Quality | ✅ | fix-code-quality (sonnet) | 4 | 3 | 1 (QA-012 .cpp) | QA-012 .cpp (needs WASM rebuild) |
| 3d — Documentation | ✅ | fix-documentation (opus) | 12 | 12 | 0 | 0 |
| 4 — Verification | ✅ | orchestrator (+ regression fix) | — | — | — | — |

**Overall**: **29 issues fully resolved**, 1 partial (QA-012 TS half done, `.cpp` half deferred for a WASM rebuild), 1 false positive (DOC-015), 1 reverted-by-design (ARC-015), and **~19 deferred** (the four large/decision-gated refactors + their dependents). Ship gate green throughout: `make checkall` = **797 pass / 23 skip / 0 fail** (39,338 assertions), up from the audit baseline of 782/36 (+2 ARC-002 tests, +13 IndexedDB tests un-skipped).

---

## Resolved Issues ✅

### Security
- **[SEC-001]** Path-traversal file write — `log_receiver.py` — **deleted** (verified unreferenced; closed probe sink).
- **[SEC-002]** Dev-server default bind — `eval/smoke-serve.ts`, `eval/live-server.ts` — `DEFAULT_HOST` → `127.0.0.1`; `--host 0.0.0.0` override preserved.
- **[SEC-003]** Dashboard CORS + `taskLists` — `eval/live-server.ts` — CORS scoped to localhost dev origins (echo validated `Origin` + `vary: origin`); `taskLists` capped at 20 lists / 500 tasks each with oldest-eviction.
- **[SEC-004]** Markdown XSS — `smoke-test/chat-render.js` — vendored DOMPurify 3.4.12; `renderMarkdown()` sanitizes marked output; **fail-safe** (escapes when DOMPurify hasn't loaded, so HTML is never served unsanitized).
- **[SEC-005]** Path filter — `eval/live-server.ts`, `eval/smoke-serve.ts` — `..`-denylist replaced with realpath-containment.
- **[SEC-006]** Error disclosure — both servers — raw `err.message` replaced with generic client messages; full detail kept server-side via `console.error`; hand-authored validation feedback preserved.
- **[SEC-007]** `Math.random` IDs — no action (audit-confirmed non-security correlation IDs).

### Architecture
- **[ARC-002]** Inert orchestration — `src/core/engine.ts`, `src/core/types.ts`, `src/core/memory-pool.ts` — **MemoryPool wired** (allocate on the shared load path across all 4 entry points, free on unload); `memoryBudget` made **optional** (8 GiB default, cites the 16 GB-floor doctrine); +2 budget-enforcement tests. Inert modules demoted from the public barrel in ARC-003.
- **[ARC-003]** Export surface — `src/index.ts`, `src/internal.ts` (new), `package.json` — deep internals moved to `./internal` subpath (mirrors `./persistence`); root barrel reduced to `WebLLM` + types + errors + sampling profiles + persistence; all broken consumers (smoke harness, build scripts) repointed. Breaking change is acceptable (project not published).
- **[ARC-005]** CI — `.github/workflows/ci.yml` (new) — `bun install --frozen-lockfile` + `make checkall` on push/PR; action refs pinned to exact verified tags (`actions/checkout@v7.0.0`, `oven-sh/setup-bun@v2.2.0`); no publish job.
- **[ARC-007]** JSEP boundary — `src/inference/jsep/pipeline-cache.ts` + 5 importers — decision: **keep JSEP**; renamed `PipelineCache` → `JsepPipelineCache` (collision resolved); `@experimental` TSDoc on `Backend`/`index-jsep`/`llama-bridge`/`llama-tokenizer`; `build-package.ts` strips prototype `.d.ts` from `dist/`.
- **[ARC-009]** Generated bundles — `.gitignore` + 4 `smoke-test/p*-*.js` — `git rm --cached` (kept on disk; `.src.ts` sources + build rules verified); uniform ignore policy.
- **[ARC-010]** Smoke harness under gates — `biome.json`, `package.json`, 20 hand-written `smoke-test/*.js` — blanket exclusion replaced with explicit generated/vendored ignore list; `smoke-test` added to `fmt`/`lint` scope; canonical stop-token files (`real-model-smoke.js`, `real-model-page.js`) now lint-clean; 8 trivial lint fixes.
- **[ARC-013]** Pin gate tools — `package.json`, `biome.json`, `bun.lock` — `typescript@7.0.2` + `@biomejs/biome@2.5.3` pinned exact; biome `$schema` updated to match.
- **[ARC-017]** Engine rename — `src/core/engine.ts`, `src/core/webllm-proxy.ts` — `resetConversation` → `resetModelSession` + `@deprecated` forwarding alias; mirrored on proxy. Worker RPC dispatch (`webllm-worker-host.ts:83`, `engine[msg.name]`) resolves both names, so the alias **is** the old→new mapping — the multi-surface trap is covered without a string rewrite. 8 in-repo value callers migrated.

### Code Quality
- **[QA-004]** IndexedDB coverage — `tests/pipeline-cache.test.ts`, `tests/persistence-indexeddb-store.test.ts`, `scripts/check-skip-count.ts` (new), `Makefile`, `package.json` — 13 IndexedDB tests un-skipped via `fake-indexeddb` (skip count 36 → 23); static skip-count ratchet (ceiling 13, sabotage-verified) wired into `make checkall`.
- **[QA-007]** WASM ABI typing — `src/inference/llama-bridge.ts` — `type WasmPtr = number | bigint`; ~30 per-line `noExplicitAny` suppressions removed; the 4 JSPI-wrapped exports typed `Promise<WasmPtr>` (the other pointer exports synchronous `WasmPtr`). Surfaces exactly the documented regression class (a `Promise` is assignable to `any` but not to `number | bigint`).
- **[QA-011]** Stale lint justifications — `generation.ts`, `gguf-parser.ts`, `model-loader.ts` — 3 expired "Phase 2" `noStaticOnlyClass` rationales replaced with the truthful one (namespace grouping; module conversion is ENH territory).
- **[QA-012]** (partial) Retired-probe sweep — `src/inference/jsep/ops/matmul.ts` — deleted the Stage-4.34/4.25 probe globals + arming branches (both CLOSED in `TODO_ARCHIVE.md`); correctly **retained** `__stage415DivertProbe` (a documented permanent regression check). `.cpp` half deferred — see Manual Intervention.

### Documentation
- **[DOC-001]** LICENSE (MIT, © 2026 Paul Robello) — added; README badge resolves.
- **[DOC-002]** `docs/MODEL_SUPPORT.md` — full accuracy pass: model count 14→30 (recounted from `eval/models.ts`), Mistral registered+canonical, `iq3m` supported, encoder/embedding path rewritten as shipped three-tier `engine.embed()` dispatch, Gemma 4 E2B added; hardcoded table → pointer + family table; emoji callouts → `> **Note:**`.
- **[DOC-003]** README Quick Start — compiles now (manual device step + `device` key removed; `memoryBudget` shown optional); architecture prose updated to ARC-002's outcome.
- **[DOC-004]** `docs/BENCHMARKS.md` — 5 dimensions / 56 tasks recounted; missing `cosine_similarity` scoring row + semantic-reasoning/embedding sections added; file ref fixed; Phi-3.5 contradiction resolved; 44-vs-56 CLI distinction noted.
- **[DOC-005]** CHANGELOG.md (Keep a Changelog) + `docs/RELEASING.md` — added; `schemaVersion` noted as compat surface; no tag/version bump.
- **[DOC-006]** `docs/reference/environment.md` — every `WEBLLM_*` var + Make override derived from grep (corrected audit's `WEBLLM_BUILD_MEM` → `WEBLLM_BUILD_MEM64`); greedy-temp comparability warning flagged.
- **[DOC-007]** CONTRIBUTING.md — prerequisites, gate, browser workflow, commit conventions, pointers.
- **[DOC-009]** `docs/LLAMA_CPP_PATCHES.md` — "four patches" → readback patches (3,4,5,10); three files named; rebase history promoted to proper H3.
- **[DOC-010]** `eval/reports/README.md` — convention index added (force-tracked; `eval/reports/` is otherwise gitignored); 5 loose 2026-04-24 files moved to `archive/`.
- **[DOC-011]** README Prerequisites section — consumers (Chrome-shaped constraints stated honestly) + contributors.
- **[DOC-012]** TODO.md header — relabeled `Baseline pinned` / `Last updated`.
- **[DOC-014]** README API table — normalized against the post-ARC-003 root barrel; internals → one `./internal` prose line; persistence surface added.

---

## Requires Manual Intervention 🔧

These are the **deferred** items — large refactors, decision gates, or work needing browser/WASM verification the automated run can't do. None block the green gate or the security/accuracy wins above.

### Highest-value, needs browser/GPU verification (cannot be auto-verified)
- **[ARC-001] / [QA-001]** Consolidate the six forward-pass graph builders in `src/inference/model-inference.ts` (4,383 lines). The audit's **top hazard class** — a fix landing in 4 of 5 sites is a silent parity bug only a browser `make smoke-bench` run can catch; Bun tests can't exercise the GPU path. Deferred until browser parity verification is in the loop. Playbook: 3 staged commits (mechanical hoists → single `buildForwardGraph` → optional diagnostics split), each browser-parity-gated.
- **[ARC-004] / [QA-010]** `ModelRecord` consolidation in `src/core/engine.ts` (seven parallel per-model Maps → one aggregate) + compile-time proxy surface.
- **[QA-002]** `Generator.generate` steering-state extraction — needs browser smoke on a Qwen3 thinking-mode model.
- **[QA-005]** Split `loadAndTest` (2,168-line CC-202 function) — now unblocked by ARC-010; needs full browser workflow verification.

### Decision / scope follow-ups
- **[ARC-015]** `exactOptionalPropertyTypes` — **reverted by design** (probe found 43 errors: 30 `src/` + 13 `tests`, exceeding the pre-committed ~40 guardrail). It is one mechanical pattern rooted in ~12–15 type definitions; a deliberate pass is ~20 edits grouped by root type (widen `field?: T` → `field?: T | undefined` where undefined is meaningful, else omit the key). The ARC-015 agent left a clean path-to-completion; safe to pick up anytime.
- **[ARC-006]** `InferencePipeline` shared interface (before a 4th engine kind lands).
- **[ARC-008] / [QA-009]** Live-server route-table refactor (Phase 1 security edits land first — done).
- **[ARC-011] / [QA-008]** Scorer-registry parity test (text-level Bun/browser name+body comparison).
- **[ARC-012]** JSPI/ABI build-time invariant check (`scripts/check-jspi-exports.ts` + `make checkall`) — would have caught both shipped historical regressions.
- **[ARC-014]** Extract worker bootstrap from the public barrel — browser-verify worker mode.
- **[ARC-016]** Normalize test layout (3 nested files → flat).
- **[QA-003]** JSEP uniform-buffer leak — **unblocked** by ARC-007's "keep" decision; cache uniforms by shape tuple. (Project memory records a JSEP+MEMORY64 negative closure on the wasm-integration paths; this is the pure-TS backend.)
- **[QA-013]** Document the JSEP `-1`/sentinel error convention — unblocked by ARC-007.
- **[QA-012] (.cpp half)** `src/wasm/webgpu-bridge.cpp` `g_probe_log` / `CHECKPOINT-IDX-DUMP` sweep — requires a coordinated `make wasm-build` (emsdk + patched `~/Repos/llama.cpp` + CMake + `exports.def`); a source edit without a rebuilt artifact ships a lie. TS half done.
- **[DOC-008]** JSDoc on the six public `engine.ts` entry points + `IndexedDBConversationStore` — after ARC-004.
- **[DOC-013]** Style-guide conformance sweep (emoji → `> **Note:**`, Mermaid `classDef`).
- **[DOC-015]** Duplicate `wasm-build-debug` Makefile target — **false positive**. The two `wasm-build-debug:` lines are Make's target-specific-variable idiom (`WEBLLM_ASSERTIONS=1` var + dependency list), not a duplicate recipe; `make -n wasm-build-debug` confirms one recipe with no override warning. No action needed.

### Deferred-but-unblocked quick wins (good next-up candidates)
QA-003, QA-005, QA-013, ARC-011, ARC-015, ARC-012 were all blocked on work that has now landed; each is independently shippable.

---

## Verification Results

- **Build / typecheck**: ✅ Pass — `tsc --noEmit` (src) + `tsc --noEmit -p tsconfig.test.json` (tests) both clean.
- **Lint**: ✅ Pass — `biome check src tests smoke-test` clean (163 files; smoke-test now included post-ARC-010).
- **Format**: ✅ Pass — `biome format` no fixes.
- **Tests**: ✅ Pass — **797 pass / 23 skip / 0 fail**, 39,338 assertions, ~6s (820 tests across 87 files). Baseline was 782/36; +2 ARC-002 budget tests, +13 QA-004 un-skipped IndexedDB tests.
- **Skip ratchet**: ✅ Pass — `check-skip-count` OK (13 `skipIf(` occurrences, ceiling 13).
- **Pre-commit hooks**: ✅ all 5 commits passed (gitleaks, detect-private-key, large-files, yaml/json checks, make fmt/lint/typecheck).
- **One regression caught + fixed in Phase 4**: SEC-004's fail-safe `renderMarkdown` returned escaped text when `DOMPurify` was absent, which the Bun unit test (no DOMPurify global) exercised. Fixed at the **test** (added a `DOMPurify` passthrough stub) rather than weakening the security fail-safe — production keeps "never serve unsanitized HTML."

### Not verified by this run (require manual browser/WASM execution)
Per scope, no agent ran browser, server, or WASM commands. These should be run before merging GPU/JSEP-adjacent work:
- **SEC-004** browser XSS check (paste `<img src=x onerror=...>`, confirm inert) — `docs/CHAT_PAGE.md` workflow.
- **ARC-010** smoke page end-to-end (reformatted harness still runs).
- All deferred GPU/JSEP/browser items above.

---

## Files Changed

**70 files** across 5 commits on `fix/audit-remediation` (`eccf6e6` → `01a9a5d`): 12 added, 5 deleted, 53 modified (+4,935 / −15,289; the deletions are dominated by the 4 de-indexed Emscripten bundles ~13K lines + `log_receiver.py`).

| Commit | Phase | Summary |
|--------|-------|---------|
| `6fbaa30` | baseline | import AUDIT.md + remediation plan |
| `e4b9959` | 1 | critical security (SEC-001/002/003/005/006) |
| `20da4ab` | 2a | JSEP boundary, wire MemoryPool, trim exports, pin gate tools (ARC-007/002/003/013) |
| `9268a50` | 2b | de-index bundles + smoke harness under gates (ARC-009/010) |
| `01a9a5d` | 3 | SEC-004, ARC-005/017, QA-004/007/011/012, DOC-001…014 + caller sweep + test fix |

Key new files: `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md`, `src/internal.ts`, `.github/workflows/ci.yml`, `scripts/check-skip-count.ts`, `docs/reference/environment.md`, `docs/RELEASING.md`, `smoke-test/vendor/purify.min.js`, `eval/reports/README.md`.

---

## Next Steps

1. **Review the branch diff** (`git diff eccf6e6..01a9a5d`) — nothing is merged; `fix/audit-remediation` is isolated with per-phase rollback commits.
2. **Spot-check the breaking changes** (ARC-003 export trim, ARC-007 rename) against any consumer you care about — the project is unpublished, so this is low-risk, but the public surface did narrow.
3. **Run the browser verifications** listed above (SEC-004 XSS, ARC-010 smoke page) before merging — they're the only things the automated gate can't cover.
4. **Pick up the deferred quick wins** (QA-003/QA-005/QA-013, ARC-011/012/015) — all now unblocked and independently shippable; re-running `/audit` after will reflect the new state.
5. **Tackle ARC-001 last** — it's the highest-risk refactor and the only one that genuinely needs the browser parity harness in the loop; the playbook's 3-stage plan bisects cleanly.
6. When ready, merge `fix/audit-remediation` to `main` (explicit confirmation, per policy — pushes/merges are user-gated). The `AUDIT.md` / `AUDIT-REMEDIATION-PLAN.md` / `AUDIT-REMEDIATION.md` artifacts can be deleted at that point (or kept for the decision record).
