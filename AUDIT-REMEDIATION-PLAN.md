# Audit Remediation Playbook

> **Source**: `AUDIT.md` (2026-07-14, commit `eccf6e6`).
> **Consumer**: `/fix-audit` agents (any model). Every entry is written to be executable
> without re-deriving the analysis. Entries are ordered to match the `## Remediation Plan`
> phases in AUDIT.md — execute top to bottom; Phase 3 sections may run in parallel by domain.
>
> **Global rules for every fix agent:**
> - Read the current file state before editing — a prior phase may have changed it (see the File Conflict Map in AUDIT.md).
> - Line numbers below were captured at `eccf6e6`; treat them as anchors, not gospel. Re-locate with Read/Grep or par-mem `find_symbol` (repo_id `webllm`) before editing.
> - The ship gate is `make checkall` (fmt + lint + typecheck + test). Run it after every issue unless the entry says otherwise. Never report an issue fixed with a red gate.
> - GPU-path changes cannot be verified by Bun tests. Where an entry requires browser verification, follow the "Browser regression workflow" in `CLAUDE.md` (`make smoke-serve` on 8031, reuse the agentchrome session/tab, cache-busted URL, check `#log` AND console). Use `?ingest=off` for throwaway runs.
> - Do not auto-generate or replace any secrets/tokens. Security fixes must preserve existing configuration semantics (flags/env overrides keep working).
> - Merged issues (QA-001≡ARC-001, QA-006≡ARC-009, QA-008≡ARC-011, QA-010≡ARC-004) are executed once at the entry marked as canonical; the duplicate stubs say so.

---

## Phase 1 — Critical Security (sequential, in this order)

### [SEC-001] Delete `log_receiver.py` (path-traversal file write)
- **Files**: `log_receiver.py` (repo root, ~15 lines)
- **Steps**:
  1. Confirm nothing references it: `grep -rn "log_receiver" Makefile eval/ smoke-test/ docs/ scripts/ package.json`. Expect zero hits (it was a hand-run probe sink for `eval/reports/p2-v2-option-a-prime-2026-05-06/`).
  2. `git rm log_receiver.py`.
  3. If (and only if) step 1 finds a live reference, do NOT delete; instead rewrite the handler: bind `("127.0.0.1", 8032)`, set `name = os.path.basename(self.path.lstrip("/")) or "log.txt"`, realpath-containment check against the target dir, and guard `content-length` (missing → 411; > 10 MB → 413).
- **Method**: The probe this served is closed (its SUMMARY.md exists under `eval/reports/`). Deletion beats hardening for dead tooling. Pitfall: don't "fix" the file with sanitization and leave the `0.0.0.0` bind — the inherited `SimpleHTTPRequestHandler` GET handler also serves the whole repo; only deletion or the full step-3 rewrite is acceptable.
- **Verify**: `test ! -f log_receiver.py && echo GONE`; `make checkall` (should be unaffected — the file is not in any TS/test path).

### [SEC-002] Default both dev servers to `127.0.0.1`
- **Files**: `eval/smoke-serve.ts:6`, `eval/live-server.ts:28`
- **Steps**:
  1. In each file find `DEFAULT_HOST = "0.0.0.0"` and change the value to `"127.0.0.1"`.
  2. Confirm the existing `--host` CLI flag / host override still parses (grep `--host` in both files); do not remove it — LAN use stays available as explicit opt-in.
  3. Update the two port-comment lines in `CLAUDE.md`? **No** — CLAUDE.md documents ports, not hosts; no doc change needed.
- **Method**: One-line constant change per file; the servers expose unauthenticated write endpoints (`/save-parity-fixture`, `/ingest`) so loopback must be the default. Pitfall: do not touch `parseServerArgs`/routing logic here — QA-009/ARC-008 rewrites that later and this must not conflict.
- **Verify**: `make checkall`; then `bun run eval/smoke-serve.ts --port 8031 &` and confirm `curl -s http://127.0.0.1:8031/ >/dev/null && echo OK` while `curl -s --max-time 2 http://$(ipconfig getifaddr en0):8031/` fails; kill the server.

### [SEC-003] Scope dashboard CORS; cap `taskLists`
- **Files**: `eval/live-server.ts:94-100` (CORS headers), `:427` (`taskLists` Map decl), `:542` (`taskLists.set`)
- **Steps**:
  1. Replace the wildcard CORS: compute `const ALLOWED_ORIGINS = new Set(["http://localhost:8031", "http://127.0.0.1:8031", "http://localhost:8033", "http://127.0.0.1:8033"]);` and in the header helper set `access-control-allow-origin` to the request's `Origin` header only when it is in the set (echo the origin, add `vary: origin`); otherwise omit the header entirely. Requests with no `Origin` (curl, same-origin) are unaffected.
  2. Cap task lists: before `taskLists.set(id, body.tasks)`, add `if (Array.isArray(body.tasks) && body.tasks.length > 500) return <400 response>;` and after the set, evict oldest insertion when `taskLists.size > 20` (`const first = taskLists.keys().next().value; if (taskLists.size > 20 && first !== undefined) taskLists.delete(first);` — Map preserves insertion order).
- **Method**: The dashboard is browsed from 8031/8033 pages only; echoing a validated origin keeps the SSE/fetch flows working while closing the any-website read/write hole. Pitfall: SSE responses must get the same treatment — make the change in the shared header helper (all responses flow through it at :94-100), not per-route. If smoke pages are ever served from another port, the set is the single place to extend.
- **Verify**: `make checkall`; start `make dashboard-serve`, then `curl -s -H "Origin: http://evil.example" -D- http://127.0.0.1:8033/health | grep -i access-control` shows no allow-origin, and with `Origin: http://localhost:8031` it echoes that origin. Stop the server.

### [SEC-005] Realpath-containment path filter in both static servers
- **Files**: `eval/live-server.ts:393`, `eval/smoke-serve.ts:124`
- **Steps**:
  1. In each file, find the sanitizer `rel.replace(/\.\./g, "").replace(/^\/+/, "")` feeding a `join(root, …)`.
  2. Replace with containment: keep the decode/normalize the file already does, then:
     ```ts
     import { resolve, sep } from "node:path";
     const rootAbs = resolve(STATIC_ROOT);
     const candidate = resolve(rootAbs, rel.replace(/^\/+/, ""));
     if (candidate !== rootAbs && !candidate.startsWith(rootAbs + sep)) return new Response("Not found", { status: 404 });
     ```
     then serve `candidate` instead of the joined path.
  3. Keep the existing directory-listing block in `smoke-serve.ts:131-133` intact.
- **Method**: Containment-after-resolve is immune to encoding/`..` variants the string-scrub denylist has to enumerate. Pitfall: `resolve` on an absolute-looking `rel` ignores the root — strip leading slashes first (kept from the original). Pitfall 2: don't break the cache-busting query strings — operate on `url.pathname` only, as the current code does.
- **Verify**: `make checkall`; start `make smoke-serve` and check `curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8031/%2e%2e/%2e%2e/package.json"` and `.../../../package.json` both return 404 while `http://127.0.0.1:8031/` returns 200. Stop the server.

### [SEC-006] Generic client error messages in dev servers
- **Files**: `eval/smoke-serve.ts:102-105,142-145`, `eval/live-server.ts:151,501,658`
- **Steps**:
  1. At each cited site, replace response bodies containing `err.message`/`String(err)` with a fixed string (`"invalid request"` for 400s, `"internal error"` for 500s).
  2. Add/keep a `console.error("<route>:", err)` beside each so the detail still reaches the server log.
- **Method**: Pure information-disclosure hygiene; behavior of success paths must not change. Pitfall: `live-server.ts` ingest validation intentionally returns *which field* failed — that is validation feedback for local harnesses, not an exception leak; keep field-name messages that are hand-authored strings, only replace ones that interpolate caught exception text.
- **Verify**: `make checkall`; grep each file for `err.message` / `${err}` in Response bodies — remaining hits must be hand-authored validation strings only.

---

## Phase 2 — Architecture (sequential, in this exact order)

### [ARC-007] Package the JSEP/prototype boundary as experimental; fix the `PipelineCache` name collision
- **Files**: `src/inference/jsep/pipeline-cache.ts`, `src/index-jsep.ts`, `src/core/types.ts` (Backend type docs), `scripts/build-package.ts`, `src/inference/llama-bridge.ts`, `src/inference/llama-tokenizer.ts`
- **Decision (pre-made — do not re-litigate)**: JSEP is **kept**, not deleted. TODO.md tracks it as active Phase-3 R&D; the recorded JSEP+MEMORY64 *negative closure* applies to the wasm-integration paths, not the pure-TS backend experiment. If the user has since decided to retire JSEP, stop and ask — deletion is a scope change.
- **Steps**:
  1. Rename the class in `src/inference/jsep/pipeline-cache.ts` from `PipelineCache` to `JsepPipelineCache`. Enumerate importers first: par-mem `get_symbol_context` on `PipelineCache` (scope_path `src/inference/jsep`) or `grep -rn "from.*jsep/pipeline-cache" src/ tests/`. Update every import site and any golden test referencing the name.
  2. Add `@experimental` TSDoc to: the `Backend` type's `"jsep"` arm in `src/core/types.ts`, the top of `src/index-jsep.ts`, and the module headers of `src/inference/llama-bridge.ts` and `src/inference/llama-tokenizer.ts` (the Tier-3 spike pair).
  3. In `scripts/build-package.ts`, exclude prototype declarations from the published types: after the `tsc --emitDeclarationOnly` step, delete `dist/**/inference/jsep/**/*.d.ts`, `dist/**/index-jsep.d.ts`, `dist/**/inference/llama-bridge.d.ts`, `dist/**/inference/llama-tokenizer.d.ts` (a small `rm` loop in the script; log what was removed).
  4. Do NOT move files into `src/experimental/` — a directory move would invalidate the line anchors every later phase relies on. The TSDoc + declaration-exclusion achieves the boundary; note the optional future move in a code comment only if one already exists.
- **Method**: Marks the boundary without churning imports repo-wide mid-remediation. Pitfall for the rename: search type-level references and string literals too (`"PipelineCache"` may appear in test descriptions — those can stay, but imports/type annotations must all move).
- **Verify**: `make checkall`; then `grep -rn "class PipelineCache" src/` returns only `src/core/pipeline-cache.ts`.

### [ARC-002] Wire MemoryPool; demote the inert orchestration modules
- **Files**: `src/core/memory-pool.ts`, `src/core/engine.ts` (`_buildInferenceAndRegister`, `unloadModel` :319-343, constructor state :217-228), `src/core/types.ts:17` (`memoryBudget`), `src/core/scheduler.ts`, `src/core/game-loop.ts`, `src/inference/stream-router.ts`, `src/core/pipeline-cache.ts`, `src/models/kv-cache.ts`, `src/index.ts`, `README.md:113-166`
- **Decision (pre-made)**: **Wire MemoryPool** (it backs the 16 GB-floor doctrine); **demote** `Scheduler`, `GameLoop`, `StreamRouter`, and core `PipelineCache` from the public barrel (keep the source files — removal is ARC-003's export trim + README's honesty pass, not deletion). `KVCache` stays as-is (its reset is real bookkeeping) but its README billing gets corrected.
- **Steps**:
  1. Wire MemoryPool bookkeeping in `src/core/engine.ts`:
     a. Locate `_buildInferenceAndRegister` (par-mem `find_symbol` name `_buildInferenceAndRegister`). Where the model is registered with `ModelManager`, call `this.memoryPool.allocate(<modelId or a derived tag>, <weightBytes>)` using the model buffer's byteLength (in scope at the load call sites — thread it as a parameter if `_buildInferenceAndRegister` doesn't already receive it).
     b. In `unloadModel` (:319-343), add the matching `this.memoryPool.free(...)` next to the other per-model teardown.
     c. Read `src/core/memory-pool.ts` first and match its actual API names (`allocate`/`free`/`release` — use what exists; do not redesign the pool).
  2. Make `memoryBudget` optional in `src/core/types.ts:17`: `memoryBudget?: MemoryBudget` (keep the type). In the engine constructor, default it (read `memory-pool.ts`/`model-manager.ts` for the consumed fields; default the total to 8 GiB per the 8B-ceiling doctrine, with a comment citing CLAUDE.md's hardware baseline).
  3. Confirm `ModelManager.canLoad` now sees non-zero `usedBytes` after a load (unit-testable: construct the pool, allocate, assert `canLoad` flips when the budget is exceeded — add/extend a test in `tests/` beside the existing memory-pool/model-manager tests if present; `ls tests/ | grep -i -E "memory|manager"` to find them).
  4. README correction is deferred to DOC-003/DOC-014 (Phase 3d) — but leave a one-line `TODO(DOC-003)` marker ONLY if the README claim would otherwise be shipped false; otherwise touch nothing in README here.
- **Method**: The minimal wiring that makes `memoryBudget` truthful: weights are the dominant allocation; KV-cache accounting can follow later (an ENH covers it). Pitfall: `loadModelFromUrl`, both `loadModelFromBuffer` overloads, and `adoptPreloadedModel` all funnel through `_buildInferenceAndRegister` — verify with par-mem `get_symbol_context` that the allocate call sits on the shared path, not one entry point. Pitfall 2: making a required field optional is backward-compatible; going the other way is not — don't "clean up" by requiring new fields.
- **Verify**: `make checkall`; the new/extended pool test proves `usedBytes > 0` after allocate and `canLoad` enforcement. Browser check not required (bookkeeping only), but `make smoke-serve` + one smoke page load is a cheap regression sniff if worker init changed.

### [ARC-003] Trim the public export surface; add an `./internal` subpath
- **Files**: `src/index.ts`, `package.json` (`exports` map), `src/internal.ts` (new)
- **Steps**:
  1. Inventory current exports: Read `src/index.ts` fully (it is ~240 lines).
  2. Create `src/internal.ts` re-exporting the deep internals currently in the barrel: `ModelInference`, `GgmlWasm`, `GgufParser`, `InferenceSession`, `Sampler`, `Generator`, `LightweightModel`, `detectChatTemplate`, `encodeChatPrompt`, plus the demoted ARC-002 modules (`Scheduler`, `GameLoop`, `StreamRouter`, core `PipelineCache`). Header comment: "Unstable internal surface for the smoke harness and power users — no semver guarantees."
  3. Reduce `src/index.ts` value exports to: `WebLLM`, the error classes/codes, sampling profiles, persistence-related re-exports it already carries, and **all type-only exports** (types are cheap and breaking to remove — keep every `export type`).
  4. In `package.json` `exports`, add `"./internal": { "import": "./dist/internal.js", "types": "./dist/internal.d.ts" }` mirroring the existing `./persistence` subpath entry exactly (same key shape/order).
  5. Fix the consumers you just broke: `grep -rn "from \"@paulrobello/webllm\"\|from \"../src\"\|from \"../../src\"" eval/ smoke-test/ tests/ scripts/` — wherever a removed symbol was imported from the root barrel, point it at `src/internal.ts` (in-repo relative imports) . The smoke-test bundle entry (`grep -rn "index.ts" scripts/ Makefile smoke-test/*.src.ts` to find it) is the main consumer of `detectChatTemplate`/`encodeChatPrompt`.
  6. par-mem sweep for stragglers: `get_symbol_context` on each moved symbol (repo_id `webllm`) lists remaining importers; also grep string literals and dynamic imports per the multi-site rule.
- **Method**: The `./persistence` subpath is the established pattern — copy it. Pitfall: `src/index.ts:160-238` contains the worker re-entry bootstrap — do NOT move or delete it in this issue (ARC-014 handles it); trim exports around it. Pitfall 2: `eval/` shims (`eval/scorer.ts`, `eval/types.ts`) re-export deliberately — update their import paths, keep their public names.
- **Verify**: `make checkall`; `bun run scripts/build-package.ts` succeeds (or fails only on the pre-existing missing-WASM guard, which is expected without artifacts); browser smoke page loads (bundle rebuild: check the Makefile's bundle target, then CLAUDE.md browser workflow) since the harness imports moved symbols.

### [ARC-001 / QA-001] Consolidate the forward-pass graph builders (canonical entry for both IDs)
- **Files**: `src/inference/model-inference.ts` (4,383 lines; `forwardSingle` :1479, `forwardForEmbedding` :1954, `forwardWithLayerTaps` :2207, `forwardAllPositions` :2858, `forwardDecode` :3237, `debugLayerOutput` :3945; duplicated `padTo` at :1523, :1981, :2258, :2885, :3266)
- **Steps** (three commits, each independently green — do not squash the stages):
  1. **Stage A — mechanical hoists (low risk).**
     a. Read the five `padTo` lambdas; confirm they are byte-identical. Hoist one copy to a private method `#padTo` (or module-scope function) and replace all five uses.
     b. The `graphMem = hp.layerCount * 32768 + totalLen * hp.embeddingLength * 256` sizing expression appears once per variant — hoist to a private `computeGraphMem(totalLen)` method. Watch for variant-specific constants (the "+32 … PLE projection nodes" comment at :1557 and :3288 indicates the sizing already absorbed corrections — if ANY variant's expression differs, stop and diff them; encode the difference as a parameter, never pick one silently).
     c. Similarly hoist the causal/SWA mask-construction block if the copies are identical modulo `totalLen`/window params (compare :~1550-1620 region across variants first; `writeCausalMaskF16` already exists — the duplication is in the code *around* it).
  2. **Stage B — single graph builder.** Define a private `buildForwardGraph(opts)` where `opts = { mode: "prefill" | "decode" | "all-positions" | "embedding", taps?: LayerTapSpec, output: "logits" | "argmax" | "hidden", positions, tokenIds, ... }`. Port `forwardSingle` onto it first (smallest delta), run Stage-B verification, then port `forwardDecode`, `forwardAllPositions`, `forwardForEmbedding`, `forwardWithLayerTaps` one at a time — each port is its own checkpoint. Each public `forward*` method keeps its exact signature and becomes input marshalling + `buildForwardGraph` + output readback.
  3. **Stage C (optional, skip if context is tight)** — move the `debug*` methods (~450 lines) to `src/inference/model-inference-diagnostics.ts` operating through a narrow interface. If skipped, file nothing: ENH plans cover it.
- **Method**: This is the audit's highest-risk refactor; the staging exists so any parity failure bisects to one variant port. par-mem queries before starting: `get_symbol_context` on each `forward*` method (external callers: `engine.ts`, `encoder-inference.ts`?, `causal-embedder-inference.ts`?, tests) and `get_impact` on `ModelInference` (expect Critical rating — that is why signatures must not change). Known constraint from CLAUDE.md regression lessons: `graphCompute()` is async-capable and must stay awaited; no `stackAlloc` across `await`. Do not "improve" numerical ordering, op fusion, or buffer reuse while porting — byte-identical graph construction is the goal; optimizations are ENH territory.
- **Verify** (after EVERY stage/port):
  1. `make checkall` (includes `tests/forward-verify-equivalence` and friends).
  2. Browser parity: CLAUDE.md browser workflow — `make smoke-serve`, run the smoke page on a small canonical model, `#log` steps all pass, zero relevant console errors.
  3. After the final port: `make smoke-bench PERF_MODEL=<a canonical model id from eval/models.ts> PERF_RUNS=3` and compare tok/s to the pre-rebase baselines pinned at `eval/reports/pre-rebase-baselines-*/SUMMARY.md` (within noise; this is a perf-neutral refactor).

### [ARC-004] Engine `ModelRecord` consolidation + compile-time proxy surface
- **Files**: `src/core/engine.ts` (state maps :217-228, `unloadModel` :319-343, proxy cast :278), `src/core/webllm-proxy.ts`, `tests/webllm-proxy-surface.test.ts`
- **Steps**:
  1. Define (in `engine.ts` or `src/core/types.ts`) `interface ModelRecord { wasm: …; inference?: ModelInference; encoder?: EncoderInference; causalEmbedder?: CausalLMEmbedder; session?: InferenceSession; chatChain?: …; eventHandlers?: … }` — derive exact field types by Reading the seven map declarations at :217-228 (wasmModules, inferenceEngines, encoderEngines, causalEmbedderEngines, sessions, modelChatChains, eventHandlers).
  2. Replace the seven maps with one `#models = new Map<string, ModelRecord>()`. Mechanical sweep: par-mem `get_symbol_context` on each old map field name, or grep each name within `engine.ts` (they are private — all uses are in-file). Every `X.get(id)` becomes `this.#models.get(id)?.field`; every `X.set(id, v)` upserts the record.
  3. Rewrite `unloadModel` to delete the single record (plus the still-external `ModelManager`/`ConversationPool` teardown it already does — those stay separate).
  4. Proxy typing: in `webllm-proxy.ts`, define `type WebLLMSurface = Pick<WebLLM, "chat" | "chatCompletion" | /* every public method the proxy mirrors — enumerate from the existing proxy class body */>;` declare `WebLLMProxy implements WebLLMSurface`, and change `engine.ts:278` to return `WebLLMProxy.init(config) as unknown as Promise<WebLLM>` unchanged for now BUT with the proxy class now failing compilation if a mirrored method drifts. (Removing the cast entirely requires the proxy to implement all of WebLLM — out of scope.)
  5. Leave `chatCompletionWithConversation` (:842-1195) intact — the `ConversationTurnRunner` split is deliberately deferred to an ENH plan; this issue is state consolidation only.
- **Method**: Single-record consolidation makes "did we clean everything up on unload" a one-line answer and future per-model features one-field additions. Pitfall: the three engine-kind maps are type-partitioned for a reason (ARC-006) — the record's optional fields preserve that partition without a union unpack; do not merge the *types*, only the *containers*. Pitfall 2: `tests/webllm-proxy-surface.test.ts` asserts surface parity at runtime — keep it; the `implements` clause complements, not replaces, it.
- **Verify**: `make checkall`; then browser workflow with `?worker=1`-style worker mode if the smoke page supports it (check `smoke-test/real-model-page.js` for the worker toggle) — worker mode exercises the proxy.

### [ARC-009 / QA-006] Uniform generated-bundle policy (canonical entry for both IDs)
- **Files**: `.gitignore`, `smoke-test/p0-spike.js`, `smoke-test/p1-fixture-regen.js`, `smoke-test/p1-tokenizer-parity.js`, `smoke-test/p2-v2-ref-probe.js`
- **Steps**:
  1. Confirm nothing serves them: `grep -rln "p0-spike.js\|p1-fixture-regen.js\|p1-tokenizer-parity.js\|p2-v2-ref-probe.js" smoke-test/*.html eval/ Makefile docs/ eval/reports/ --include="*.html" --include="*.ts" --include="*.md"`. HTML pages under `eval/reports/` that reference them are *archived reports* — if a report HTML script-loads one, keep that bundle and record which (expected: none; the reports pin conclusions as markdown).
  2. Verify each has a committed source + build rule: `ls smoke-test/*.src.ts` and grep the Makefile for the corresponding `bun build` targets. All four sources must exist; if one lacks a source, keep that bundle and note it.
  3. Add to `.gitignore` (near the existing `smoke-test/p2-v2-spike.js` entries): a comment line and the four filenames (explicit paths, not a glob — `p*-*.js` would also match hand-written files like `p2-v2-spike.src.ts` siblings; explicit is safer).
  4. `git rm --cached smoke-test/p0-spike.js smoke-test/p1-fixture-regen.js smoke-test/p1-tokenizer-parity.js smoke-test/p2-v2-ref-probe.js` (remove from index, keep on disk so local pages still work).
- **Method**: Matches the policy the newer bundles already follow. Pitfall: use `git rm --cached`, not `git rm` — deleting from disk breaks any locally-opened probe page and forces a rebuild to inspect history.
- **Verify**: `git status` shows the four as deleted-from-index + ignored; `make checkall` unaffected; `git check-ignore smoke-test/p0-spike.js` prints the path.

### [ARC-010] Bring the smoke-harness core under the quality gates
- **Files**: `biome.json` (the `"!!**/smoke-test"` exclusion), `smoke-test/real-model-smoke.js`, `smoke-test/real-model-page.js`, hand-written `.d.ts` shims in `smoke-test/`
- **Steps**:
  1. Read `biome.json`; change the blanket `smoke-test` exclusion to an explicit list that keeps excluding ONLY generated bundles and vendored libs: the four ARC-009 bundles (until deleted), `webllm-bundle*.js`, `webllm-wasm*.js`, any `vendor/` files (enumerate with `ls smoke-test/`), and `p2-v2-spike.js`/`p2-v2-offload-probe.js`. Hand-written files (`real-model-smoke.js`, `real-model-page.js`, `chat-render.js`, `chat-restore.js`, `dashboard.js`, `scorer-registrations.js`, etc. — everything with no `.src.ts` sibling and no "generated" header) become lintable.
  2. `make fmt` then `make lint` — fix what biome flags in the newly-included files. Expect mostly mechanical issues; formatting churn on these files is acceptable and isolated (they were never formatted).
  3. Do NOT convert the harness to TypeScript in this issue and do NOT split `loadAndTest` (that is QA-005). This issue is: gates on, files clean under the gates.
- **Method**: Smallest change that ends the "canonical stop-token logic lives in the unlinted zone" condition. Pitfall: `git diff --stat` after `make fmt` — if the formatter touched generated files you failed step 1; restore them and tighten the exclusion list. Per the surgical-format rule, commit only smoke-test files + biome.json in this change.
- **Verify**: `make checkall`; then the browser workflow end-to-end (the smoke page itself was reformatted — it must still run: all `#log` steps pass, no console errors).

---

## Phase 3a — Security (parallel-safe)

### [SEC-004] Sanitize markdown rendering in the chat page
- **Files**: `smoke-test/chat-render.js:34-40,72`; new vendored file `smoke-test/vendor/purify.min.js` (or alongside existing vendored libs — `ls smoke-test/` / check `make vendor-refresh` layout first and match it); the chat HTML page that loads `chat-render.js` (find with `grep -l "chat-render" smoke-test/*.html`)
- **Steps**:
  1. Locate how `marked` is vendored/loaded (grep `marked` in `smoke-test/*.html` and the Makefile's `vendor-refresh` target). Vendor DOMPurify the same way (pin a version; add it to the `vendor-refresh` target so the pin is reproducible).
  2. Load DOMPurify in the chat page before `chat-render.js` (same script-tag pattern as marked).
  3. In `renderMarkdown()` (:34-40) wrap the output: `body.innerHTML = DOMPurify.sanitize(marked(text));` — same at the :72 call site if it assigns separately.
- **Method**: marked ≥v5 passes raw HTML through by design; DOMPurify is the standard companion and a self-contained vendored file (no CDN — the smoke pages must work offline). Pitfall: don't switch to `textContent` — the page renders real markdown (code blocks, lists) that must keep working; sanitize, don't strip.
- **Verify**: `make checkall` (chat-render.js is now lint-covered post-ARC-010); browser: open the chat page per `docs/CHAT_PAGE.md`, paste `<img src=x onerror="document.title='XSS'">` into a message, confirm it renders inert (title unchanged) while normal markdown (bold, code fence) still renders.

### [SEC-007] `Math.random()` IDs — no action
- Reviewed and closed by the audit: correlation IDs, not secrets (`src/core/engine.ts:293`, `src/characters/character.ts:106`, `eval/live-server.ts:541`). **Do nothing.** Listed so /fix-audit does not "helpfully" swap in `crypto.randomUUID()` and churn test fixtures.

---

## Phase 3b — Architecture (parallel-safe within this section)

### [ARC-005] Minimal CI workflow
- **Files**: `.github/workflows/ci.yml` (new)
- **Steps**:
  1. Create the workflow: name `ci`; on `push` to `main` + `pull_request`; single job on `ubuntu-latest`; steps: `actions/checkout`, `oven-sh/setup-bun` (pin exact tags per the git-ci guide — verify each `uses:` ref resolves before committing: `git ls-remote https://github.com/oven-sh/setup-bun <tag>`), `bun install --frozen-lockfile`, `make checkall`.
  2. Do NOT add a publish job (publishing is user-gated; note it as a follow-up comment in the workflow).
  3. WASM/browser benches stay local — nothing in the workflow may invoke `make wasm-build`, `smoke-*`, or anything needing Chrome/emsdk.
- **Method**: `make checkall` already isolates the Bun-only gate; `build-package.ts` is deliberately NOT in the gate (needs WASM artifacts). Pitfall: pin action refs to exact tags (e.g. `actions/checkout@v4.x.y` style with a resolved tag), per the user's global pattern-matching rule.
- **Verify**: Local lint of the YAML (`bun x yaml-lint` or careful read); `make checkall` green locally is the same command CI runs. Note in the fix report that first-run verification happens on next push (do not push — user-gated).

### [ARC-006] `InferencePipeline` shared interface
- **Files**: `src/core/types.ts` (or a new `src/core/inference-pipeline.ts`), `src/core/engine.ts`, `src/inference/model-inference.ts`, `src/inference/encoder-inference.ts`, `src/inference/causal-embedder-inference.ts`
- **Steps**:
  1. Enumerate the union's methods actually called by the engine: grep `engine.ts` for the union type name(s) (`ModelInference | EncoderInference | CausalLMEmbedder`) and list every member access on those values.
  2. Define `interface InferencePipeline { readonly kind: "chat" | "encoder" | "causal-embedder"; dispose(): void | Promise<void>; embed?(…): …; }` — exact member list = the step-1 intersection; the `kind` discriminant replaces `isEncoderArchitecture`-style string guards at dispatch sites.
  3. Add `implements InferencePipeline` to the three classes (add a `kind` field to each; do not otherwise change them).
  4. Where mechanical, narrow engine signatures from the union to `InferencePipeline`; where a call site genuinely needs the concrete class (e.g. chat-only APIs), keep the concrete type — do not force casts.
- **Method**: Post-ARC-004 the containers are unified; this unifies the *type* seam so a fourth engine kind is additive. Pitfall: `embed()` signatures may differ across the three (tap-point vs. dedicated) — check before declaring the optional member; if they differ, model `embed` as the loosest common signature or leave it off the interface and keep that one dispatch site concrete.
- **Verify**: `make checkall`.

### [ARC-008 / QA-009-part] Live-server route table
- **Files**: `eval/live-server.ts:433-763` (`fetch`, CC 53)
- **Steps**:
  1. **Precondition**: Phase 1 (SEC-002/003/005/006) has landed — re-Read the file; line anchors have shifted.
  2. Build `const routes = new Map<string, (req, url) => Response | Promise<Response>>()` keyed `"GET /runs"`, `"POST /ingest"`, etc. Extract each existing `if` block body into a named handler function above, byte-for-byte (no logic changes).
  3. `fetch` becomes: CORS/preflight helper → exact-match route lookup → SSE arm (its own conditional — streaming responses don't fit the table) → static-file fallback → 404.
  4. Add the `taskLists` eviction from SEC-003 unchanged if this refactor moves that code — preserve it exactly.
- **Method**: Pure mechanical extraction; behavior-identical by construction. Pitfall: some branches match prefixes (`/runs/:id`?) — Read first; prefix routes stay as a small ordered array checked after the exact-match Map. Pitfall 2: keep response header composition going through the (now origin-scoped) shared CORS helper.
- **Verify**: `make checkall`; `make dashboard-serve`, then `curl` each endpoint: `/health`, `/runs`, `/evals`, `/system-profiles`, a POST `/ingest` with a minimal valid body (crib one from `eval/reports/` JSON), and confirm the dashboard page renders. `make import-reports` is a good end-to-end exercise (idempotent by design). Stop the server.

### [ARC-011 / QA-008] Scorer-registry parity test (canonical entry for both IDs)
- **Files**: new `tests/scorer-registry-parity.test.ts`; read-only: `eval/tasks/scorer-registrations.ts`, `smoke-test/scorer-registrations.js`, `tests/custom-scorers.test.ts:61-84`
- **Steps**:
  1. Read both registration files. Both call a registration function (`registerCustomScorer(name, fn)`) — the browser file imports it from `./webllm-bundle.js`, which does not exist under Bun, so **do not import the browser file**; parse it as text.
  2. Test: read both files with `Bun.file(...).text()`; extract registered names with a regex on the registration calls (`/registerCustomScorer\(\s*["']([^"']+)["']/g`); assert the two name-sets are equal and have size 13 (or the current count from the Bun side — derive, don't hardcode, so adding a 14th scorer on both sides passes).
  3. Stronger drift check: for each name, extract the function body text between registration calls; normalize whitespace; assert equal. If the two sides genuinely differ in idiom (imports vs. globals), fall back to asserting name-set equality only and add a comment explaining why body-hash was skipped.
- **Method**: Text-level parity is the only approach that works in Bun without a browser bundle. Pitfall: keep the existing `custom-scorers.test.ts` untouched — this is additive.
- **Verify**: `bun test tests/scorer-registry-parity.test.ts`; then `make checkall`.

### [ARC-012] JSPI/ABI build-time invariant check
- **Files**: new `scripts/check-jspi-exports.ts`; `Makefile` (checkall or a new `check-jspi` target); read-only: `src/wasm/CMakeLists.txt` (JSPI_EXPORTS list, ~lines 202-217), `src/inference/ggml-wasm.ts`
- **Steps**:
  1. Script logic: (a) parse `src/wasm/CMakeLists.txt` for the `JSPI_EXPORTS` list (Read the file to learn the exact CMake syntax first — it's a set/list of export names); (b) assert every `webllm-wasm*` executable target in the file links with `-sJSPI_EXPORTS` (regression lesson: the MEM64 target was once missed); (c) for each export name, find its binding call in `src/inference/ggml-wasm.ts` (`ccall`/`cwrap`/`Module._name` — grep the file for the name) and assert the call expression is `await`ed or `.then`-chained (regex on the surrounding line(s): `await` preceding, or assignment into a variable that is awaited — keep the heuristic simple and err toward failing with a clear message listing the suspect line).
  2. Exit non-zero with a per-name report on violation; print `ok: N exports checked` on success.
  3. Makefile: add `check-jspi: ; bun run scripts/check-jspi-exports.ts` and append it to the `checkall` dependency list.
- **Method**: A grep-level assertion would have caught both shipped regressions (missing per-target flag; unawaited promising-wrapped export). Pitfall: the inverse rule also matters but is NOT mechanically checkable (an export on the list whose binding consumes synchronously — like the removed `backend_alloc_ctx_tensors`); have the script print the checked list so humans can audit it, and keep the CLAUDE.md lesson as the governing doc.
- **Verify**: `make check-jspi` passes at current HEAD; sabotage-test it once locally (temporarily remove an `await` in a scratch copy — script must fail); `make checkall` green.

### [ARC-013] Pin gate-tool devDependencies
- **Files**: `package.json`, `biome.json` (`$schema`), `bun.lock`
- **Steps**: 1. In `package.json`, remove `^` from `typescript` and `@biomejs/biome` (pin the exact versions currently resolved — read them from `bun.lock`). Other devDeps may keep ranges. 2. Update `biome.json` `$schema` URL to match the pinned biome version. 3. `bun install` to refresh the lockfile.
- **Method**: The two gate tools define what "green" means; pinning makes `checkall` reproducible. Pitfall: pin to the *resolved* versions (no behavior change), not the latest.
- **Verify**: `make checkall` (identical results pre/post); `git diff package.json bun.lock biome.json` shows only the pins.

### [ARC-014] Extract the worker bootstrap from the public barrel
- **Files**: `src/index.ts:160-238`, new `src/worker-entry.ts`, `src/core/webllm-proxy.ts` (the `new Worker(...)` site)
- **Steps**:
  1. Read `src/index.ts:160-238` and find where the Worker is constructed with the barrel as its module URL (grep `new Worker` in `src/` — likely `webllm-proxy.ts`).
  2. Move the bootstrap block verbatim into `src/worker-entry.ts` (keep the `typeof`-guard).
  3. Point the `new Worker(new URL(...))` at `worker-entry.ts`.
  4. Check `scripts/build-package.ts` and the smoke-test bundle build for entry-point lists that must now include `worker-entry` (grep `index.ts` in `scripts/` and `Makefile`).
- **Method**: Keeps the barrel side-effect-free for bundlers. Pitfall: the worker URL resolution is build-sensitive — this is the one Low-priority item with real breakage potential; verify in-browser, not just via tsc.
- **Verify**: `make checkall`; rebuild the smoke bundle and run the browser workflow in worker mode (the proxy path) — model load must reach tokenizer-ready with no console errors.

### [ARC-015] Enable `exactOptionalPropertyTypes`
- **Files**: `tsconfig.json:22`, fallout across `src/`/`tests/`
- **Steps**: 1. Flip to `true`. 2. `make typecheck` — fix each error by choosing per-site: add `| undefined` to the property type when assigning `undefined` is intentional, or stop assigning `undefined` (use conditional spread/`delete`). 3. If fallout exceeds ~40 errors, revert and report the count instead of half-fixing (the audit rated this Low; a large diff is worse than the flag).
- **Method**: Codebase is otherwise fully strict. Pitfall: don't weaken types to `?: T | undefined` wholesale — only where undefined-assignment is semantically real.
- **Verify**: `make checkall`.

### [ARC-016] Normalize test layout
- **Files**: the three nested files under `tests/inference/` and `tests/models/` (enumerate: `find tests -mindepth 2 -name "*.test.ts"`)
- **Steps**: 1. `git mv` each nested test up to `tests/` (flat is the 84-file majority convention). 2. Fix their relative imports (`../../src/...` → `../src/...`). 3. Check `package.json`/`Makefile`/`tsconfig.test.json` for glob patterns referencing the subdirectories.
- **Method**: Follow the majority convention; don't invent a new tree. Pitfall: name collisions on move — prefix with the subsystem if a flat name already exists.
- **Verify**: `make checkall` — same test count as before the move (818 total / 782 pass at `eccf6e6`; compare totals, not exact pass count if earlier phases added tests).

### [ARC-017] Rename `resetConversation` → `resetModelSession` with deprecation alias
- **Files**: `src/core/engine.ts:1274`, `src/core/webllm-proxy.ts` (mirror), callers
- **Steps**:
  1. Enumerate callers first: par-mem `get_symbol_context` on `resetConversation` + grep for the string literal (worker message types may carry the method name as a string — check `worker-bridge.ts` message discriminants).
  2. Rename the method; add `/** @deprecated Use resetModelSession */ resetConversation(modelId: string) { return this.resetModelSession(modelId); }`.
  3. Mirror both methods on the proxy (post-ARC-004, the `WebLLMSurface` type makes omission a compile error).
  4. Update in-repo callers (eval/smoke-test) to the new name.
- **Method**: Alias preserves the public contract (ARC-003 kept semver discipline). Pitfall: the worker RPC path dispatches on method-name strings — a rename without the string-level sweep breaks worker mode silently; that is exactly the multi-surface trap rule 10 warns about.
- **Verify**: `make checkall`; proxy-surface test still green.

---

## Phase 3c — Code Quality (parallel-safe within this section, except where gated)

### [QA-001] — merged into **ARC-001** (Phase 2). No separate work. Confirm the Phase-2 entry completed; mark this ID with the same outcome.

### [QA-002] `Generator.generate` steering-state extraction
- **Files**: `src/inference/generation.ts:184-~715` (and `:170` class decl); tests: `tests/generation.test.ts`
- **Steps**:
  1. Read the whole method. Inventory the steering locals (`thinkDepth`, `thinkClosed`, `waitingForVisibleAnswer`, `hasVisibleAnswerText`, and the other ~6 mutable flags/counters).
  2. Extract `class SteeringState` (same file, above `Generator`): fields = the inventory; one method `onToken(tokenId, decodedText): { maskTokens: number[]; suppress: boolean; stop: boolean }` containing the existing branch logic verbatim; plus small named helpers where the branches naturally group (`enterThink`, `closeThink`).
  3. Extract `makeResult(finishReason)` covering the three duplicated result literals (:229, :249, :280) — diff the literals first; encode any field differences as parameters.
  4. Extract `selectDecodeStep(...)` for the greedy/topk/full × GPU/CPU dispatch.
  5. `generate` keeps its exact public signature and yield/return contract.
- **Method**: The 20 tests in `tests/generation.test.ts` are the net — run them after each extraction, not just at the end. CRITICAL pitfall (CLAUDE.md regression lesson): Qwen3 has TWO end-of-turn tokens (151645, 151643) that must stay masked during the post-`</think>` window and both treated as stop tokens — the extraction must move this logic, never simplify it; if a branch looks redundant, it probably encodes that lesson. Do not change any observable token stream.
- **Verify**: `bun test tests/generation.test.ts` after each step; `make checkall`; browser smoke on a qwen3 model specifically (thinking-mode path), confirming the visible answer appears and generation stops (no run-on).

### [QA-003] JSEP uniform-buffer leak + probe-global sweep *(gated: ARC-007 decided "keep")*
- **Files**: `src/inference/jsep/ops/matmul.ts:703-712` (+ probe globals ~:640-660, error returns :699-701), `src/inference/jsep/ops/rms-norm.ts:166`, `src/inference/jsep/ops/set-rows.ts:395`
- **Steps**:
  1. In each op, locate the per-dispatch `ctx.device.createBuffer(...)` for `shapeBuffer` (the FIXME(phase 3) comments mark them).
  2. Implement a shared uniform cache in the JSEP context: a `Map<string, GPUBuffer>` keyed by the serialized shape tuple (`${M},${K},${N},${batch},${strides…}`), owned by `ctx` (find the context type — grep `interface.*Ctx\|type.*Ctx` in `src/inference/jsep/`), populated on miss via `device.createBuffer` + `queue.writeBuffer`, destroyed in the context's dispose path. Shape counts are small (model-static), so no eviction needed — but add a size log if entries exceed 256.
  3. Sweep the probe globals in `matmul.ts`: delete `__stage434Probe21Arm`, `__stage425KahanArm`, the `console.log` at :652, and their arming branches (grep the names repo-wide first — a smoke page may set them; if `smoke-test/`/`eval/` references exist, delete those setters too; the stage-4.x probes are closed per TODO archive).
  4. Leave the `return -1` error convention in place (QA-013 documents it; changing conventions here would touch every caller).
- **Steps require**: the JSEP golden tests (`tests/` jsep suites) — note they are WebGPU-gated and skip under Bun; the real check is step-by-step below.
- **Method**: The FIXME's own proposal (cache/ring-buffer) — implement the simpler cache. Pitfall: `queue.writeBuffer` per dispatch on a cached buffer is only valid if the shape values for a given key never change — they can't, the key IS the values; write once on creation.
- **Verify**: `make checkall`; browser: run the JSEP smoke path (grep `smoke-test/*.html` and Makefile for the jsep/spike page; `?backend=jsep` — check `real-model-page.js` for the toggle) and confirm a multi-hundred-token generation completes; optionally observe `performance.memory`/adapter info stability across two runs in the same tab.

### [QA-004] Un-skip the IndexedDB test suites; add a skip-count ratchet
- **Files**: `tests/pipeline-cache.test.ts:7-10`, plus sibling skipped suites (find all: `grep -rln "skipIf(!indexedDBAvailable)\|skipIf(typeof indexedDB" tests/`), `package.json` (devDep), possibly a shared `tests/setup-indexeddb.ts`
- **Steps**:
  1. `bun add -d fake-indexeddb` (devDependency only — the runtime-zero-deps invariant applies to `dependencies`, untouched).
  2. In each IndexedDB-gated test file: add `import "fake-indexeddb/auto";` as the FIRST import, then remove the `skipIf` wrapper (revert to plain `test(...)`). Keep WebGPU-gated skips (jsep golden tests) — those genuinely need a GPU; this issue is IndexedDB only.
  3. Run each converted suite individually; fix environmental assumptions (e.g. persistent DB state across tests — `fake-indexeddb` is per-process; add `indexedDB.deleteDatabase(...)` in `beforeEach` if tests interfere).
  4. Skip ratchet: new `tests/skip-ratchet.test.ts` — but a test cannot easily count sibling skips at runtime in Bun. Instead do it as a gate step: a tiny script `scripts/check-skip-count.ts` that runs `bun test 2>&1`, parses the `N skip` summary, and fails if N exceeds a pinned constant (set it to the post-fix count; comment explaining the ratchet: lower it when skips are removed, never raise without justification). Wire into `Makefile` `checkall` after `test`. (If parsing proves brittle, grep-count `skipIf(` occurrences in `tests/` instead — static ratchet, same effect.)
- **Method**: `fake-indexeddb/auto` is the standard polyfill path and works under Bun. Pitfall: import order matters — the polyfill must land before any module that captures `globalThis.indexedDB` at import time (the store module may). Pitfall 2: don't delete the `indexedDBAvailable` helper if other files use it for feature-detection logic beyond skipping.
- **Verify**: `bun test tests/pipeline-cache.test.ts` shows the 5 tests RUNNING and passing (not skipped); `make checkall` — skip count dropped from 36 to (36 − converted count); ratchet script fails if you temporarily re-add a skip (sabotage-check once).

### [QA-005] Split `loadAndTest` by step boundaries *(gated: after ARC-010)*
- **Files**: `smoke-test/real-model-page.js:368-~2536`
- **Steps**:
  1. **Precondition**: ARC-010 landed (file is lint-covered and freshly formatted — re-Read it; anchors moved).
  2. Map the seams: grep the function for the step markers (`[1/8]` … `[8/8]`) and the mode branches (chat / bench / perf-trace).
  3. Extract one top-level `async function` per step (`stepAcquireAdapter`, `stepLoadModel`, … — name from what each logs) passing a single shared `ctx` object holding the current locals each step needs (build the ctx field list by reading each extracted block's free variables).
  4. Extract the three mode drivers similarly. `loadAndTest` becomes the sequential composition.
  5. Behavior-identical: same log lines, same ordering, same error paths. No logic edits.
- **Method**: Mechanical extraction using the numbering already present. Pitfall: hoisted `let`s mutated across steps must live on `ctx`, not stay as closure locals, or extraction reorders TDZ/mutation semantics. Pitfall 2: this file IS the ship gate for GPU work — the verification below is not optional.
- **Verify**: `make checkall`; full browser workflow: smoke page green end-to-end on one small + one canonical model, `#log` identical step sequence, zero console errors; run one `make smoke-bench PERF_MODEL=<canonical> PERF_RUNS=1` to confirm bench mode still drives (use `?ingest=off` note — smoke-bench doesn't thread it; run with dashboard down or accept the ingest).

### [QA-006] — merged into **ARC-009** (Phase 2). No separate work.

### [QA-007] `WasmPtr` typing for the llama-bridge export surface
- **Files**: `src/inference/llama-bridge.ts:122-218`
- **Steps**:
  1. Read the export-binding block. Add `type WasmPtr = number | bigint;` near the top.
  2. Replace each `any` parameter/return that represents a pointer/size with `WasmPtr` (non-pointer `any`s, if any, get their real primitive type). Delete the per-line `biome-ignore lint/suspicious/noExplicitAny` comments as each `any` disappears.
  3. Fix fallout at call sites within the file: arithmetic on `WasmPtr` needs the existing ABI-branch helpers (the file centralizes translation at its cwrap boundary per the audit — reuse those helpers; do not sprinkle `as number`).
- **Method**: The union surfaces exactly the bug class the regression lessons document (a `Promise` is assignable to `any` but not to `number | bigint` — tsc now catches unawaited bindings). Pitfall: if some binding legitimately returns `Promise<WasmPtr>` (JSPI-wrapped), type it as that, don't await-in-place without checking the JSPI_EXPORTS rules in CLAUDE.md.
- **Verify**: `make checkall`; `grep -c "noExplicitAny" src/inference/llama-bridge.ts` drops to ~0.

### [QA-008] — merged into **ARC-011** (Phase 3b). No separate work.

### [QA-009] — remaining scope merged into **ARC-008** (route table) and **SEC-003** (taskLists cap). Confirm both landed; no separate work.

### [QA-010] — merged into **ARC-004** (Phase 2). No separate work.

### [QA-011] Static-only namespace classes: convert or re-justify
- **Files**: `src/inference/generation.ts:170` (`Generator`), `src/models/gguf-parser.ts:18` (`GgufParser`), `src/models/model-loader.ts:146` (`ModelLoader`)
- **Steps**:
  1. For each: par-mem `get_symbol_context` (+ grep for `Generator.`/`GgufParser.`/`ModelLoader.` and bare-name imports, including `src/internal.ts` post-ARC-003 and smoke-test bundles) to size the caller set.
  2. **Do the cheap honest fix, not the module conversion**: these classes are exported (post-ARC-003 via `./internal`), so converting to plain modules is a breaking rename across the harness bundle. Instead update each `biome-ignore` justification comment to the truthful rationale ("namespace-style grouping; exported API — module conversion tracked as enhancement") — the audit's complaint is the *stale* "Phase 2" justification, and ENH plans cover real conversion.
  3. If (and only if) step 1 shows a trivially small in-repo caller set for one of them (< 5 sites, none in smoke bundles), converting that one to module functions is acceptable — keep the class as a deprecated thin wrapper if it's exported.
- **Method**: Honors §3 surgical-change defaults inside a quality fix. Pitfall: `Generator` is mid-QA-002 refactor territory — coordinate: run QA-002 first if the same agent owns both.
- **Verify**: `make checkall`.

### [QA-012] Retired-probe scaffolding sweep
- **Files**: `src/wasm/webgpu-bridge.cpp:440-490,881` (`g_probe_log`, `CHECKPOINT-IDX-DUMP`), `src/inference/jsep/ops/matmul.ts` (stage-4.x globals — done in QA-003 step 3 if that ran; check first)
- **Steps**:
  1. Confirm the probes are retired: grep TODO.md/TODO_ARCHIVE.md for the probe names/stages; the stage-4.x and checkpoint-idx probes belong to closed blocks. If a probe's block is still open in TODO.md, leave it and note which.
  2. Delete the `g_probe_log` machinery and `CHECKPOINT-IDX-DUMP` blocks from `webgpu-bridge.cpp` (both the declarations and the dump call sites — grep the identifier across `src/wasm/`).
  3. **WASM rebuild caveat**: `.cpp` edits require `make wasm-build` (needs emsdk + patched llama.cpp per CLAUDE.md). If the build environment is unavailable, make NO change to the .cpp file and report it skipped — a source edit without a rebuilt artifact ships a lie.
- **Method**: Probe-first doctrine keeps probes cheap to add; the archival cadence should sweep them. Pitfall: some `#ifdef`-style guards may compile the probes out already — if the code is dead under current build flags, deletion is still right but verify flags in `src/wasm/CMakeLists.txt` first.
- **Verify**: If rebuilt: `make wasm-build` succeeds + browser smoke green. Always: `make checkall`.

### [QA-013] Document the JSEP error convention *(gated: ARC-007 decided "keep")*
- **Files**: `src/inference/jsep/ops/matmul.ts` (module header), or a `src/inference/jsep/README.md`
- **Steps**: 1. Add a short block to the jsep module boundary (the ops' shared context file or a directory README): ops return `-1`/sentinel + `console.error` (C-ABI style) instead of throwing `WebLLMError`, because dispatch sites mirror the ggml call convention; conversion to typed errors happens at the backend boundary (name the actual boundary file — grep for what calls `dispatchMatmul`). 2. No behavior change.
- **Method**: Documents the seam the audit flagged rather than churning a prototype's call sites. Pitfall: keep it honest — if nothing converts errors at the boundary, say "callers must check sentinels" rather than describing an aspirational layer.
- **Verify**: `make checkall` (comment/doc only).

---

## Phase 3d — Documentation (parallel-safe within this section; DOC-003/DOC-014 wait for Phase 2)

### [DOC-001] Add the MIT LICENSE file
- **Files**: `LICENSE` (new, repo root)
- **Steps**: 1. Write the standard MIT license text, `Copyright (c) 2026 Paul Robello`. 2. Confirm `README.md:3` badge link (`LICENSE`) now resolves. 3. No `package.json` change needed (npm auto-includes root LICENSE; `"license": "MIT"` already set).
- **Verify**: `test -f LICENSE`; par-mem `find_broken_doc_links` (repo_id `webllm`) returns zero after reindex, or simply confirm the README link target exists.

### [DOC-002] `docs/MODEL_SUPPORT.md` full accuracy pass
- **Files**: `docs/MODEL_SUPPORT.md`; sources of truth (read-only): `eval/models.ts`, `src/core/types.ts`, `src/inference/encoder-inference.ts`, `src/inference/causal-embedder-inference.ts`, `README.md` (embed() docs), `eval/reports/bucket-d-parity-2026-04-29/SUMMARY.md`, the Gemma-4 validation report (find: `ls eval/reports/ | grep -i gemma`)
- **Steps**:
  1. Read `eval/models.ts` end to end; extract the registered model list (id, family, quant, params) — this is the ONLY source for model claims.
  2. Fix the five falsified claims from the audit: (a) model count 14 → current count (30 at audit time — recount, don't copy); (b) mistral row: registered + canonical fleet member (cite the two registered ids); (c) i-quants: `iq3m` is a supported `QuantFormat` with two canonical IQ3_M models; (d) rewrite the "Encoder Forward Pass" section from future-work to shipped behavior — three-tier `engine.embed()` dispatch (encoder → causal-embedder → chat-model tap), gated per-model by `embeddingCapable: true`, parity reports cited; (e) Gemma: add Gemma 4 E2B (PLE + dual-RoPE + shared-KV + SWA) with its closure date.
  3. Replace the hardcoded Registered Models table with a short instruction to enumerate live: "run `make bench-eval-models` (or read `eval/models.ts`)" plus only the *architecture-family* table (families change rarely; per-model rows rot).
  4. While in the file: swap emoji callouts to plain `> **Note:**` / `> **Warning:**` per `docs/DOCUMENTATION_STYLE_GUIDE.md` (DOC-013 overlap — do it here, once).
- **Method**: Derive every claim from code read THIS session. Pitfall for a smaller model: do not paraphrase the audit's numbers into the doc — recount from `eval/models.ts` at fix time; the count may have moved.
- **Verify**: For each rewritten claim, cite (in the fix report, not the doc) the source line that backs it. `make checkall` (docs don't gate, but the habit stands).

### [DOC-003] Fix the README Quick Start *(after ARC-002/ARC-003)*
- **Files**: `README.md` (Quick Start, lines ~48-72; architecture section 113-166 if ARC-002 changed module truth)
- **Steps**:
  1. Read the current `WebLLMConfig` in `src/core/types.ts` and the static `WebLLM.loadModelFromBuffer` signature in `src/core/engine.ts` (post-ARC-002 `memoryBudget` may now be optional — reflect reality).
  2. Rewrite the example: drop the manual `requestAdapter/requestDevice` step and the `device` config key; note in one sentence that the engine acquires its own WebGPU device. Keep the example minimal and aligned with the (accurate) persistence example's style further down.
  3. Update the architecture/orchestration prose (README:113-166) to match ARC-002's outcome: MemoryPool now truthfully budget-enforcing (weights-level); Scheduler/StreamRouter/GameLoop no longer billed as live orchestration (describe as internal/experimental or remove rows — match what ARC-003 exported).
  4. Type-sanity the snippet: paste it into a scratch `snippet.ts` in the scratchpad importing from `src/index.ts`, run `bun x tsc --noEmit` against it (or `bun build` it), then delete the scratch file.
- **Method**: The compile check in step 4 is the point — the audit's finding was precisely that the snippet fails typecheck. Pitfall: two initialization styles exist (`WebLLM.init` vs static loaders) — keep Quick Start on ONE style and note the other, don't blend.
- **Verify**: The step-4 compile passes; README links intact.

### [DOC-004] `docs/BENCHMARKS.md` dimensions/scoring/catalog refresh
- **Files**: `docs/BENCHMARKS.md`; sources (read-only): `src/evaluation/types.ts` (ScoringMethod), `eval/tasks/` (task sets per dimension), `eval/cli.ts` (`allTasks`), `eval/models.ts`
- **Steps**:
  1. Enumerate dimensions + task counts from `eval/tasks/` (audit found: tool-calling 12, reasoning 12, instruction-following 12, semantic-reasoning 12, embedding 8 — recount at fix time).
  2. Add sections for semantic-reasoning and embedding mirroring the existing dimension sections' structure (task table + scoring approach).
  3. Scoring table: enumerate the `ScoringMethod` union arms from `src/evaluation/types.ts`; add the missing cosine-similarity row; fix the file reference (`eval/types.ts` → `src/evaluation/types.ts`, noting the re-export shim).
  4. Model Catalog: correct counts/architectures from `eval/models.ts` (include mistral); resolve the Phi-3.5 contradiction by checking `eval/models.ts` for a registered phi entry — if registered, fix the "deferred" note; if not, fix the Recommended table.
  5. Add one explanatory note where the "44 tasks" CLI claim appears: `eval/cli.ts`'s `allTasks` intentionally omits the semantic-reasoning set (state the actual reason if a comment in `cli.ts` gives one; otherwise mark it "historical — full set is 56").
- **Method**: Same derive-from-source rule as DOC-002. Pitfall: dashboard chart labels must keep matching dimension names — copy exact dimension strings from the task definitions.
- **Verify**: Recounted numbers cited to source files in the fix report.

### [DOC-005] CHANGELOG + release process
- **Files**: `CHANGELOG.md` (new), `README.md` (short Releasing subsection) or `docs/RELEASING.md`
- **Steps**:
  1. Create `CHANGELOG.md` in Keep-a-Changelog format with `## [Unreleased]` and one `## [0.1.0] - 2026-07-14` section summarizing current state at a coarse grain (initial public shape: WebGPU/WASM dual backend, eval framework, persistence with `schemaVersion` 1, worker mode). Mine `git log --oneline` and TODO_ARCHIVE.md headings for the 5-10 headline entries — do NOT attempt a full retroactive history.
  2. Releasing doc: bump version in `package.json` → update CHANGELOG → tag `v<semver>` → CI publishes (note: CI publish job does not exist yet; publishing is via CI/CD only per repo policy — never local `npm publish`).
  3. Note the conversation-persistence `schemaVersion` as a tracked compatibility surface: schema bumps get a CHANGELOG entry.
- **Method**: Keep it honest — a thin start beats fake history. Pitfall: do not tag anything or bump the version; that's a user-gated release action.
- **Verify**: Files exist; links from README resolve.

### [DOC-006] Environment-variable reference
- **Files**: new `docs/reference/environment.md` (create `docs/reference/`), plus a pointer line in `docs/BENCHMARKS.md` and README's benchmarks section
- **Steps**:
  1. For each variable, grep its consumer and default: `grep -rn "WEBLLM_" eval/ src/ Makefile | grep -v reports/`. Audit's list: `WEBLLM_LIVE_BENCH_URL`, `WEBLLM_ASSERTIONS`, `WEBLLM_BENCH_EVAL_TEMPERATURE`, `WEBLLM_WASM_VARIANT`, `WEBLLM_BENCH_SESSION_ID`, `WEBLLM_STALL_TIMEOUT_MS`, `WEBLLM_HARD_TIMEOUT_MS`, `WEBLLM_SMOKE_RUNS_DIR`, `WEBLLM_BUILD_MEM`, `WEBLLM_BACKEND`; plus Make vars `SMOKE_PORT`, `DASHBOARD_PORT`, `PERF_MODEL`, `PERF_RUNS` (and any new ones the grep finds — the grep is authoritative).
  2. Table columns: Variable | Consumed by (file) | Default | Effect | Notes (e.g. comparability warning on `WEBLLM_BENCH_EVAL_TEMPERATURE` — greedy-temp doctrine).
  3. Follow `docs/DOCUMENTATION_STYLE_GUIDE.md` (TOC, Related Documentation section — every existing doc has them).
- **Verify**: Every table row's file:line cited in the fix report; grep finds no `WEBLLM_*` variable absent from the table.

### [DOC-007] CONTRIBUTING.md
- **Files**: `CONTRIBUTING.md` (new, root)
- **Steps**: Write ~60-100 lines: prerequisites (Bun; Chrome with WebGPU for browser regressions; emsdk + patched `~/Repos/llama.cpp` branch `webllm-browser-patches` only for WASM work); setup (`make install`); ship gate (`make checkall`); browser regression workflow (link CLAUDE.md section or summarize the 5 steps); commit conventions (`feat(...)`, `fix(...)`, `docs(spec):` etc. with the separate-commits doctrine); pointers to `docs/MODEL_SUPPORT.md` (adding models), `docs/LLAMA_CPP_PATCHES.md` (WASM/patch work), `docs/BENCHMARKS.md`; note that `docs/superpowers/` is gitignored and force-added by convention.
- **Method**: Aggregate what CLAUDE.md already says, rewritten for humans. Pitfall: don't duplicate long CLAUDE.md prose — link/point where possible so there's one source of truth.
- **Verify**: Links resolve; referenced Make targets exist (`make help`).

### [DOC-008] JSDoc for the public entry points *(after ARC-004)*
- **Files**: `src/core/engine.ts` (`init`, `exportConversation`, `importConversation`, `createCharacter`, `removeCharacter`, `shutdown` — re-locate post-ARC-004 with par-mem `find_symbol`; class banner + module header), `src/persistence/indexeddb-store.ts` (`IndexedDBConversationStore` class + `open()`)
- **Steps**: 1. For each method: one-sentence summary, `@param`/`@returns`, one `@example` where a README example exists to crib from, `@throws` for the typed errors it raises (grep the body for `throw new`). 2. Class banner for `WebLLM` (what it is, worker-vs-inline note, link to README). 3. `IndexedDBConversationStore`: model docs on the adjacent `ConversationStoreEntry` doc style.
- **Method**: Match the existing JSDoc voice in the same file (`chatCompletion` is the model). Pitfall: don't restate parameter types in prose; the signature carries them.
- **Verify**: `make checkall`; hover-check via `bun x tsc --noEmit` cleanliness (malformed JSDoc that breaks the parser would fail typecheck).

### [DOC-009] `docs/LLAMA_CPP_PATCHES.md` reference fixes
- **Files**: `docs/LLAMA_CPP_PATCHES.md`
- **Steps**: 1. Count the current patch inventory entries in the doc itself. 2. Fix the Troubleshooting sentence: "the four patches" → cite the actual readback-related patch numbers (audit suggests 3-5 and 10 — verify against the inventory list in the doc); name the three files explicitly where "the three files above" appears (copy the filenames from the code comment two sections up). 3. Restructure "Earlier rebase" H4s under a proper `### Rebase history` H3 (heading-level change only; keep content and anchor text).
- **Method**: Reference repair, not a rewrite — this doc is otherwise exemplary. Pitfall: heading changes can break in-repo anchors — `grep -rn "LLAMA_CPP_PATCHES.md#" docs/ README.md CLAUDE.md` first and update any fragment links.
- **Verify**: Anchor grep clean; doc renders with correct hierarchy (`bun x markdownlint` if available, else visual read).

### [DOC-010] `eval/reports/` index + archive sweep
- **Files**: new `eval/reports/README.md`; move 5 loose 2026-04-24 root files into `eval/reports/archive/`
- **Steps**: 1. `ls eval/reports/ | grep -v /` to enumerate loose root files; confirm the five are the 2026-04-24 timestamped JSON/HTML. 2. `git mv` them into `eval/reports/archive/` (create if absent; check whether an `archive/` already exists and match its layout). 3. Before moving, grep for inbound references: `grep -rn "<filename>" docs/ TODO.md TODO_ARCHIVE.md eval/ smoke-test/ --include="*.md" --include="*.ts"` — update any hits (the dashboard's `make import-reports` walks `eval/reports/` recursively, so imports still find them — confirm by reading the walker in `eval/` for path filters). 4. Write README.md: the `<area>-<date>/SUMMARY.md` convention, what a SUMMARY.md must contain (status, headline metric, links — crib the description from CLAUDE.md's TODO-archival section), the archive policy, and how `make import-reports` consumes the tree.
- **Method**: Pitfall: `import-reports` idempotency keys on runId/evalId, so moved files re-import harmlessly — but verify the walker doesn't skip `archive/` (read `eval/import-reports.ts` or the Make target's script).
- **Verify**: `make import-reports` runs clean (dashboard up) or is confirmed path-agnostic by code read; reference grep clean.

### [DOC-011] README prerequisites section
- **Files**: `README.md`
- **Steps**: Add a `## Prerequisites` section before Quick Start with two short lists: **Consumers** — WebGPU-capable browser; state the Chrome-shaped constraints honestly (developed/regression-tested on Chrome; JSPI + 128 MiB per-binding storage cap assumptions per CLAUDE.md); **Contributors** — Bun, Chrome for the browser workflow, emsdk + patched `~/Repos/llama.cpp` (branch `webllm-browser-patches`) only for `make wasm-build`.
- **Method**: Consolidate the scattered claims (Quick Start implication, CLAUDE.md, LLAMA_CPP_PATCHES prerequisites) into one place; link to LLAMA_CPP_PATCHES.md for WASM details. Pitfall: don't overclaim browser support (no Firefox/Safari testing evidence exists).
- **Verify**: Section renders; no duplicate prerequisites remain in Quick Start.

### [DOC-012] TODO.md header date
- **Files**: `TODO.md:3`
- **Steps**: Change the label to `> **Baseline pinned**: 2026-04-27 · **Last updated**: <today>` (keeps the historical pin the stale date actually meant while making updates self-describing).
- **Verify**: Visual; nothing else on the line changed.

### [DOC-013] Style-guide conformance sweep
- **Files**: `docs/MODEL_SUPPORT.md` (done in DOC-002 step 4 — verify, don't redo), Mermaid blocks in `README.md` and `docs/*.md`
- **Steps**: 1. `grep -rn "📝\|⚠️" docs/ README.md` — convert remaining emoji callouts to plain `> **Note:**` / `> **Warning:**`. 2. For each Mermaid block using repeated per-node `style N fill:...` lines, convert to `classDef` + `class` assignments with identical colors (use the dark-mode palette from the user's global prefs if colors are being touched anyway — but prefer zero color changes; this is a syntax normalization). Test each converted diagram renders (mermaid.live or the Artifact renderer) before committing.
- **Method**: Cosmetic; batch it last. Pitfall: Mermaid `classDef` syntax errors kill the whole diagram render — verify every edited diagram.
- **Verify**: Emoji grep clean; diagrams render.

### [DOC-014] README API table normalization *(after ARC-002/ARC-003)*
- **Files**: `README.md` (API Overview table)
- **Steps**: 1. Read the post-ARC-003 `src/index.ts` — the table documents exactly the public (root) export list now. 2. Remove the two method rows (`engine.removeCharacter(id)`, `engine.shutdown()`) from the class table — methods belong in the class's JSDoc (DOC-008), not the table. 3. Add rows for the persistence surface (`IndexedDBConversationStore` via `./persistence`, `exportConversation`/`importConversation` noted in prose under the table). 4. Internals now under `./internal` get one prose line ("unstable internals live in `@paulrobello/webllm/internal`"), not table rows.
- **Verify**: Every table row corresponds to a live root export (`grep "export" src/index.ts`); no removed export is still billed.

### [DOC-015] Remove the duplicate `wasm-build-debug` Makefile target
- **Files**: `Makefile`
- **Steps**: 1. `grep -n "^wasm-build-debug:" Makefile` — two hits. 2. Diff the two recipe bodies. If identical: delete the first. If different: keep the one that matches current CMake flags (compare against `wasm-build`'s recipe and `src/wasm/CMakeLists.txt` variables) and delete the other; make currently uses the LAST definition, so keeping the last preserves behavior.
- **Method**: Pitfall: keep-the-last preserves today's behavior; if you keep the first instead, you have silently changed what the target does.
- **Verify**: `make -n wasm-build-debug` prints one recipe; no `warning: overriding commands` emitted; `make help` still lists it once.

---

## Phase 4 — Final verification (after all phases)

1. `make checkall` — full gate green.
2. Browser regression workflow (CLAUDE.md): smoke page end-to-end on one canonical model; `#log` all steps pass; no relevant console errors.
3. `git status` — every changed file traces to an issue ID above; no stray formatter churn (restore unrelated reformatted files per the surgical-format rule).
4. Skip-count ratchet (QA-004) reflects the new baseline.
5. Do NOT commit or push anything unless the user has asked — report the working-tree state and per-issue outcomes instead.
