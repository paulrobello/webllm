# ENH-001 — Automated Browser Regression Lane (Playwright)

> **Status**: proposed · **Effort/risk**: Medium · **Depends on**: nothing (but should land *before* ARC-001/QA-005 refactors to serve as their safety net)

## Goal

A one-command, scriptable browser gate — `make test-browser` — that loads the existing smoke page
in Playwright-driven Chrome with the smallest registered model, asserts the full `#log` step
sequence succeeds, and fails on relevant runtime console errors. This automates the manual
agentchrome workflow in CLAUDE.md ("Browser regression workflow") without replacing agentchrome
for interactive debugging.

## Current state

- The ship gate `make checkall` is Bun-only; GPU paths are verified manually via agentchrome
  (CLAUDE.md workflow: `make smoke-serve` on 8031 → navigate cache-busted URL → check `#log` AND
  console). 36 tests skip permanently under Bun (AUDIT.md QA-004).
- `eval/browser-smoke.ts` already automates the flow through the agentchrome CLI — read it first;
  it encodes the success/failure conditions (which `#log` markers, which console errors are
  benign) that the Playwright test must replicate.
- **JSPI constraint** (load-bearing): the WASM build uses JSPI globally (CLAUDE.md regression
  lessons; pivot `b4d4b48`). `WebAssembly.promising` requires a Chrome feature flag in ad-hoc
  launches. Find the exact flags the current automation uses: `grep -rn "enable-features\|js-flags\|jspi\|JSPI" eval/browser-smoke.ts eval/perf.ts Makefile smoke-test/` — mirror them in the
  Playwright launch args.
- Smoke runs auto-post to the dashboard unless `?ingest=off` — the test must pass `ingest=off`.

## Implementation steps

1. **Discover the harness contract** (read-only): Read `eval/browser-smoke.ts` and
   `smoke-test/real-model-page.js` enough to record: (a) the page URL + query params used
   (model id param name, `?v=` cache-bust, `?ingest=off`); (b) the terminal success marker in
   `#log` (the `[8/8]`-style final step text — copy it exactly); (c) the benign console patterns
   (at minimum `adapter_info:` per CLAUDE.md — collect any others browser-smoke.ts filters).
2. **Pick the model**: read `eval/models.ts`; choose the smallest registered chat model by file
   size / params. Confirm its weights exist locally (the smoke server serves them — find the
   catalog/asset dir smoke-serve exposes; if the file is absent, the test must fail with a clear
   "model not present — run <the documented fetch command>" message, not a timeout).
3. **Install**: `bun add -d @playwright/test` then `bun x playwright install chromium`.
4. **Config** `playwright.config.ts` (repo root): one project, `use.headless: true`,
   `launchOptions.args` = the JSPI/WebGPU flags from step 1 discovery plus
   `--enable-unsafe-webgpu` if the discovered flags include it; `webServer` block runs
   `bun run eval/smoke-serve.ts --port 8034` (a fresh port — do NOT reuse 8031, a dev server may
   be up; note 8034 in `~/.claude/used_ports.md` if it becomes permanent) with
   `url: "http://127.0.0.1:8034"` readiness check. Test timeout ≥ 180 s (model load is slow cold).
5. **Test** `tests-browser/smoke.spec.ts` (new dir — keeps Bun's `tests/` glob untouched):
   - Navigate to `http://127.0.0.1:8034/real-model.html?model=<id>&ingest=off&v=${Date.now()}`
     (exact page filename from step 1).
   - Subscribe `page.on("console")` and `page.on("pageerror")`; buffer everything.
   - `await expect(page.locator("#log")).toContainText("<final success marker>", { timeout: 180_000 })`.
   - Assert the buffered console has no `error`-level entries after filtering the benign list.
6. **Make target**: `test-browser: ; bun x playwright test` with a comment "requires local GPU +
   model weights — not part of checkall". Do NOT add to `checkall` or CI (ARC-005's workflow must
   not run this).
7. **Docs**: one paragraph in CLAUDE.md's browser-workflow section: agentchrome = interactive
   debugging; `make test-browser` = scripted gate. Add the port to the Ports section.

## Files to touch

- `playwright.config.ts` (new), `tests-browser/smoke.spec.ts` (new)
- `package.json` (devDep), `Makefile` (new target), `CLAUDE.md` (workflow note)
- Read-only: `eval/browser-smoke.ts`, `smoke-test/real-model-page.js`, `eval/models.ts`

## Verification

1. `make test-browser` passes locally (model present, GPU available).
2. Sabotage check: temporarily point the test at a nonexistent model id — it must fail fast with
   the step-2 message, not hang. Revert.
3. `make checkall` unchanged (Playwright specs must not be picked up by `bun test` — confirm the
   `tests-browser/` dir is outside every Bun test glob; check `package.json`/`Makefile` test
   invocations).
4. Confirm no dashboard rows were created (`ingest=off` honored): with `make dashboard-serve` up,
   run the lane, then check the dashboard run count is unchanged.

## Rollback

Delete `tests-browser/`, `playwright.config.ts`, the Make target, and the devDep
(`bun remove @playwright/test`). No shipped code is touched, so rollback is total.
