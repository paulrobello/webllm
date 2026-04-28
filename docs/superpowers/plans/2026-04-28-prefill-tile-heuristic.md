# Prefill-tile heuristic refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the §23 dual-source-of-truth pattern (`recommendedPrefillTile` field on `BenchmarkModel` + mirrored `RECOMMENDED_PREFILL_TILE` map in the smoke page) with a single hyperparam-derived heuristic computed inside the `ModelInference` ctor.

**Architecture:** New exported helper `computeDefaultPrefillTileSize(hp)` in `src/inference/model-inference.ts`. Rule: `layerCount >= 32 AND embeddingLength >= 4096` → tile=128, else 0. Ctor default flips from `?? 0` to `?? computeDefaultPrefillTileSize(hp)`. The eval registry field, the smoke mirror map, the perf.ts fallback, and the registry-shape tests are deleted. Smoke page passes `prefillTileSize` to ctor only when `?prefillTile=` URL param is set; tile pill is rendered post-ctor based on `inference.prefillTileSize`.

**Tech Stack:** TypeScript / Bun (test runner) for the helper + tests; vanilla JS for the smoke page; no WASM rebuild needed (TS-only change).

**Spec reference:** `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md` (commit `ae68bbe`). Read the spec first; this plan does not duplicate its rationale.

---

## File map

- **Create (temporary, deleted in Phase 4):** `eval/probe-prefill-tile-heuristic.ts` — one-shot validation script.
- **Modify:** `src/inference/model-inference.ts` — add helper + ctor wiring.
- **Modify:** `tests/prefill-tiling-config.test.ts` — adjust `STUB_HP` to sub-7B shape; add 3 boundary tests.
- **Modify:** `eval/models.ts` — delete `recommendedPrefillTile` field from interface and 5 entries; remove the field's jsdoc block.
- **Modify:** `smoke-test/real-model-page.js` — delete `RECOMMENDED_PREFILL_TILE` map + fallback branch; pass `prefillTileSize` to ctor only when explicitly set; render tile pill post-ctor.
- **Modify:** `eval/perf.ts` — drop the `model.recommendedPrefillTile` fallback (lines ~178-188).
- **Modify:** `tests/eval-models.test.ts` — delete the `describe("recommendedPrefillTile auto-default", …)` block (lines 17-45).
- **Modify:** `TODO.md` — add §30 closure entry.

---

## Phase 0 — Pre-edit hyperparam probe (HALT-CONDITION)

**Purpose:** Validate the heuristic against actual GGUF metadata for every registered, downloaded model before any code edit. Spec §A.1 makes this mandatory; the brainstorm's hyperparam values were informed-guess.

### Task 0.1 — Write probe script

**Files:**
- Create: `eval/probe-prefill-tile-heuristic.ts`

- [ ] **Step 1: Write the script**

```ts
// eval/probe-prefill-tile-heuristic.ts
//
// One-shot probe used during the §30 prefill-tile heuristic refactor to
// validate that `layerCount >= 32 AND embeddingLength >= 4096 → 128 (else 0)`
// classifies every registered, downloaded model identically to the current
// `recommendedPrefillTile` field. Halts the refactor if any mismatch.
//
// Run: `bun run eval/probe-prefill-tile-heuristic.ts`
// Exit: 0 on full match, 1 on any mismatch or no models found.
//
// Delete this file at the end of Phase 4.

import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { ModelLoader } from "../src/models/model-loader.js";
import { BENCHMARK_MODELS } from "./models.js";

const SMOKE_DIR = "smoke-test/models";

function heuristic(layerCount: number, embeddingLength: number): number {
	return layerCount >= 32 && embeddingLength >= 4096 ? 128 : 0;
}

let mismatches = 0;
let probed = 0;
const rows: string[] = [];
rows.push(
	`status     id                                            arch     L      E    current  heuristic`,
);
rows.push(
	`---------  --------------------------------------------- -------  ----  ----  -------  ---------`,
);

for (const m of BENCHMARK_MODELS) {
	const path = join(SMOKE_DIR, `${m.id}.gguf`);
	const altPath = join(SMOKE_DIR, `${m.id}.GGUF`);
	const file = existsSync(path) ? path : existsSync(altPath) ? altPath : null;
	if (!file) {
		rows.push(`SKIP       ${m.id.padEnd(45)}  (not downloaded)`);
		continue;
	}
	let hp;
	try {
		const buf = readFileSync(file);
		hp = ModelLoader.parseModel(new Uint8Array(buf)).hyperparams;
	} catch (e) {
		rows.push(`PARSE-FAIL ${m.id.padEnd(45)}  ${(e as Error).message}`);
		mismatches++;
		continue;
	}
	probed++;
	const current = m.recommendedPrefillTile ?? 0;
	const h = heuristic(hp.layerCount, hp.embeddingLength);
	const ok = h === current;
	if (!ok) mismatches++;
	rows.push(
		`${ok ? "OK       " : "MISMATCH "}  ${m.id.padEnd(45)} ${String(hp.architecture).padEnd(7)} ${String(hp.layerCount).padStart(4)}  ${String(hp.embeddingLength).padStart(4)}  ${String(current).padStart(7)}  ${String(h).padStart(9)}`,
	);
}

console.log(rows.join("\n"));
console.log("");
console.log(
	`Probed ${probed} model(s); ${mismatches} mismatch(es)${
		mismatches === 0 ? "" : " — HALT before edit."
	}.`,
);
process.exit(mismatches > 0 ? 1 : 0);
```

- [ ] **Step 2: Verify the file compiles in isolation**

Run: `bun build eval/probe-prefill-tile-heuristic.ts --target=bun --outfile=/tmp/probe-build-check.js && rm /tmp/probe-build-check.js`
Expected: success (no output beyond the bundler's progress line); imports resolve.

- [ ] **Step 3: Commit-checkpoint NOT yet — leave file untracked**

The probe file is intentionally not committed; it is deleted in Phase 4. Skip `git add`.

### Task 0.2 — Run probe and gate the rest of the plan

- [ ] **Step 1: Run the probe**

Run: `bun run eval/probe-prefill-tile-heuristic.ts`

Expected — every downloaded model row prints `OK` and the final line reads `Probed N model(s); 0 mismatch(es).` Exit code 0.

Specifically, the 5 currently-tagged 7B+ models must show `current=128 heuristic=128`:
- `mistral-7b-instruct-v0.3-q4ks`
- `mistral-7b-instruct-v0.3-q3km`
- `mistral-7b-instruct-v0.3-iq4xs`
- `llama-3.1-8b-instruct-iq3m`
- `qwen3-8b-iq3m`

Every other downloaded model must show `current=0 heuristic=0`.

- [ ] **Step 2: HALT-CONDITION**

If exit code is non-zero (any `MISMATCH` row), STOP. Do not proceed to Phase 1. Report the mismatching rows to the user and re-open the brainstorm to choose between (a) tightening the threshold, (b) special-casing the offending model with a comment in the heuristic, or (c) some other resolution. Do not edit any source file with a known-wrong heuristic.

If exit code is 0, proceed to Phase 1.

---

## Phase 1 — Helper + ctor wiring + boundary tests

### Task 1.1 — Adjust `STUB_HP` so the existing "defaults to 0" test still passes after the heuristic flip

**Files:**
- Modify: `tests/prefill-tiling-config.test.ts:7-19`

The existing `STUB_HP` uses `layerCount: 32, embeddingLength: 4096` — exactly the values that would trigger the new heuristic to return 128. The existing `defaults to 0` test (lines 22-25) passes today because the ctor literally returns 0; after the refactor it would FAIL because the heuristic returns 128. Update `STUB_HP` to a sub-7B-shaped baseline so `defaults to 0` continues to mean "heuristic returns 0 because gates fail."

- [ ] **Step 1: Edit `STUB_HP` to sub-7B (TinyLlama-shaped) values**

Replace lines 7-19 with:

```ts
// Sub-7B shape (TinyLlama-1.1B): layerCount=22, embeddingLength=2048.
// Both gates of the §30 heuristic fail, so `defaults to 0` still
// reflects "no override AND heuristic returns 0." Boundary tests
// below override these fields explicitly to exercise the gates.
const STUB_HP = {
	architecture: "llama",
	layerCount: 22,
	embeddingLength: 2048,
	headCount: 32,
	headCountKv: 32,
	embeddingHeadLength: 64,
	feedForwardLength: 5632,
	vocabularySize: 32000,
	contextLength: 2048,
	ropeFreqBase: 10000,
	ropeFreqScale: 1.0,
} as unknown as ModelHyperparams;
```

- [ ] **Step 2: Run existing tests; they should still pass (heuristic helper not added yet, so ctor still returns 0 by default)**

Run: `bun test tests/prefill-tiling-config.test.ts`
Expected: `5 pass / 0 fail`. Same as baseline.

- [ ] **Step 3: Do NOT commit yet**

This change is bundled with Phase 1's other edits.

### Task 1.2 — Add 3 boundary tests (failing — helper does not yet exist)

**Files:**
- Modify: `tests/prefill-tiling-config.test.ts` — append a new `describe` block after the existing one.

- [ ] **Step 1: Append the new `describe` block**

Add this after line 53 (after the closing `});` of the existing `describe("prefillTileSize ctor option", ...)` block):

```ts

describe("prefillTileSize heuristic default", () => {
	test("layerCount>=32 AND embeddingLength>=4096 → 128", () => {
		const hp = {
			...STUB_HP,
			layerCount: 32,
			embeddingLength: 4096,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp);
		expect(inf.prefillTileSize).toBe(128);
	});

	test("either gate fails → 0", () => {
		const cases: Array<Partial<ModelHyperparams>> = [
			{ layerCount: 31, embeddingLength: 4096 }, // layer below
			{ layerCount: 32, embeddingLength: 2048 }, // emb below
			{ layerCount: 16, embeddingLength: 2048 }, // both below
		];
		for (const overrides of cases) {
			const hp = { ...STUB_HP, ...overrides } as ModelHyperparams;
			const inf = new ModelInference(STUB_WASM, hp);
			expect(inf.prefillTileSize).toBe(0);
		}
	});

	test("explicit prefillTileSize: 0 overrides heuristic-128", () => {
		const hp = {
			...STUB_HP,
			layerCount: 32,
			embeddingLength: 4096,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { prefillTileSize: 0 });
		expect(inf.prefillTileSize).toBe(0);
	});
});
```

- [ ] **Step 2: Run tests — first one MUST fail (heuristic not implemented yet)**

Run: `bun test tests/prefill-tiling-config.test.ts`

Expected:
- `prefillTileSize ctor option` block: 5 pass.
- `prefillTileSize heuristic default` block:
  - `layerCount>=32 AND embeddingLength>=4096 → 128`: **FAIL** (`expected 128, received 0`).
  - `either gate fails → 0`: PASS (current ctor returns 0 by default; this is incidentally satisfied).
  - `explicit prefillTileSize: 0 overrides heuristic-128`: PASS (ctor opt 0 wins; today works incidentally because there's no heuristic).
- Summary: `7 pass / 1 fail`.

This is the failing-test step before the implementation. Do not move on until the first heuristic test fails as expected.

### Task 1.3 — Implement helper + ctor wiring

**Files:**
- Modify: `src/inference/model-inference.ts:139-155`

- [ ] **Step 1: Add the exported helper above the `ModelInference` class**

Find the opening of the `ModelInference` class (search for `export class ModelInference`). Immediately above it, insert:

```ts
/**
 * Compute the default `prefillTileSize` for a model based on hyperparameters.
 *
 * Rule: `layerCount >= 32 AND embeddingLength >= 4096` → 128, else 0.
 *
 * Maps directly to the §22 abort signature observed in
 * `eval/reports/prefill-tiling-2026-04-27/00-phase0-diagnostic.txt`:
 * "32 layers × seq=512 of F32 intermediates" exceeds the host-side ggml graph
 * allocator budget at `ggml-alloc.c:82`. Either gate alone keeps the per-tile
 * working set below the budget on every currently-registered model.
 *
 * Override surface (ctor opt / `?prefillTile=` / `--prefill-tile`) wins
 * unconditionally — including the explicit-zero force-disable path.
 *
 * Spec: `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md`.
 */
export function computeDefaultPrefillTileSize(
	hp: ModelHyperparams,
): number {
	return hp.layerCount >= 32 && hp.embeddingLength >= 4096 ? 128 : 0;
}
```

- [ ] **Step 2: Update the ctor to fall back to the heuristic**

In `src/inference/model-inference.ts`, replace line 149:

```ts
		this.prefillTileSize = opts.prefillTileSize ?? 0;
```

with:

```ts
		this.prefillTileSize =
			opts.prefillTileSize ?? computeDefaultPrefillTileSize(hyperparams);
```

The negative-value guard (lines 150-154) stays.

- [ ] **Step 3: Run the heuristic tests — all should pass now**

Run: `bun test tests/prefill-tiling-config.test.ts`
Expected: `8 pass / 0 fail`.

### Task 1.4 — Run full ship gate

- [ ] **Step 1: Run `make checkall`**

Run: `make checkall`
Expected: tail shows `430 pass / 11 skip / 0 fail` (was 427 baseline; +3 boundary tests added; nothing deleted yet).

If any unrelated test fails, investigate before proceeding. Do not commit until the gate is green.

- [ ] **Step 2: Do NOT commit yet**

Phase 2 deletes the dual-registry surface; both phases land in a single commit per the spec.

---

## Phase 2 — Delete dual-registry surface

### Task 2.1 — Drop `eval/perf.ts` registry fallback

**Files:**
- Modify: `eval/perf.ts:178-188` and the use site at `:199-201`.

- [ ] **Step 1: Replace the fallback block with a direct pass-through**

Replace lines 178-188 with:

```ts
	// §30 prefill-tile heuristic: when --prefill-tile is omitted we leave
	// `prefillTile` undefined here so the smoke page also leaves it
	// unspecified — `ModelInference` ctor's hyperparam-derived default
	// (computeDefaultPrefillTileSize) decides. Explicit `--prefill-tile <n>`
	// (including 0) still wins.
	const effectivePrefillTile = opts.prefillTile;
```

The `extraParams` block at lines 199-201 already conditionally adds `prefillTile` only when `effectivePrefillTile !== undefined`; that logic is preserved without change.

- [ ] **Step 2: Verify `eval/perf.ts` typechecks**

Run: `bun run --bun tsc --noEmit -p .` *(or whatever the project's typecheck target is — see `make typecheck` in the Makefile)*
Expected: no errors.

If unsure, `make typecheck` is the canonical entry point per `CLAUDE.md` standard targets.

### Task 2.2 — Drop `smoke-test/real-model-page.js` mirror map and move tile pill render to post-ctor

**Files:**
- Modify: `smoke-test/real-model-page.js:150-172` (mirror map + fallback branch).
- Modify: `smoke-test/real-model-page.js:216-221` (early tile pill creation).
- Modify: `smoke-test/real-model-page.js:409-417` (post-ctor — append tile pill from constructed inference).

- [ ] **Step 1: Replace the mirror map + fallback branch with explicit-only resolution**

Replace lines 150-172 with:

```js
	// §22 prefill-tiling gate: `?prefillTile=N` forces a specific tile size.
	// When the URL param is absent, we leave the ctor opt undefined and let
	// `ModelInference`'s hyperparam-derived default decide (see §30 +
	// `computeDefaultPrefillTileSize` in src/inference/model-inference.ts).
	// Pass `?prefillTile=0` to force-disable on a 7B+ model.
	const prefillTileParam = params.get("prefillTile");
	let prefillTileOverride; // undefined → ctor heuristic decides
	if (prefillTileParam !== null) {
		const raw = Number(prefillTileParam);
		prefillTileOverride =
			Number.isFinite(raw) && raw >= 0 ? Math.floor(raw) : 0;
	}
```

- [ ] **Step 2: Delete the early tile-pill block (lines 216-221 in the old file)**

Remove the block:

```js
		if (prefillTileSize > 0) {
			const tilePill = document.createElement("span");
			tilePill.className = "mode-pill on";
			tilePill.textContent = `tile: ${prefillTileSize}`;
			modeBar.appendChild(tilePill);
		}
```

The pill will now be appended after construction, when we know the resolved value.

- [ ] **Step 3: Update the ctor call site**

Find the line:

```js
				inference = new ModelInference(wasm, parsed.hyperparams, {
					flashAttn: flashAttnEnabled,
					prefillTileSize,
				});
```

Replace `prefillTileSize` with `prefillTileSize: prefillTileOverride`. Result:

```js
				inference = new ModelInference(wasm, parsed.hyperparams, {
					flashAttn: flashAttnEnabled,
					prefillTileSize: prefillTileOverride,
				});
```

`prefillTileOverride` may be `undefined`; the ctor's `?? computeDefaultPrefillTileSize(hp)` then fires.

- [ ] **Step 4: Append tile pill post-ctor (only for `ModelInference`, not `EncoderInference`)**

Inside the same `else` branch where `ModelInference` is constructed (after `inference = new ModelInference(...)` and the trace setup, but still inside the same `else` block), add:

```js
				const resolvedTile = inference.prefillTileSize;
				if (resolvedTile > 0) {
					const tilePill = document.createElement("span");
					tilePill.className = "mode-pill on";
					tilePill.textContent = `tile: ${resolvedTile}`;
					modeBar.appendChild(tilePill);
				}
```

Only `ModelInference` exposes `prefillTileSize`; `EncoderInference` does not, so guarding inside the `else` branch is correct.

- [ ] **Step 5: Verify the smoke page still parses (syntax check)**

Run: `node --check smoke-test/real-model-page.js`
Expected: no output (success). Syntax errors are reported on stderr.

### Task 2.3 — Drop `eval/models.ts` field, jsdoc, and 5 entry uses

**Files:**
- Modify: `eval/models.ts:55-65` (jsdoc + field).
- Modify: `eval/models.ts:447, 472, 498, 525, 552` (5 entry-level uses).

- [ ] **Step 1: Delete the field and its jsdoc**

In `eval/models.ts`, delete lines 55-65 in their entirety:

```ts
	/**
	 * §22 default-on auto-tile: when set, the harness threads
	 * `prefillTileSize: <n>` through to `ModelInference` whenever no
	 * explicit override is provided. 7B+ entries set this to 128 to
	 * sidestep the host-side ggml graph-allocator abort at long
	 * prefills (`ggml-alloc.c:82`); sub-7B entries leave it unset so
	 * the single-graph fast path (and TinyLlama's measured TTFT) are
	 * preserved. Mirror the smoke page's `RECOMMENDED_PREFILL_TILE`
	 * map when changing this value.
	 */
	recommendedPrefillTile?: number;
```

- [ ] **Step 2: Delete the field uses on the 5 entries**

For each of the lines below, delete the `recommendedPrefillTile: 128,` line (and its preceding newline if any). The exact line numbers may shift slightly as you go; search for `recommendedPrefillTile: 128,` and remove all 5 occurrences.

Affected entries (verify by ID after deletion):
- `mistral-7b-instruct-v0.3-q4ks`
- `mistral-7b-instruct-v0.3-q3km`
- `mistral-7b-instruct-v0.3-iq4xs`
- `llama-3.1-8b-instruct-iq3m`
- `qwen3-8b-iq3m`

- [ ] **Step 3: Verify no `recommendedPrefillTile` references remain in `eval/models.ts`**

Run: `grep -c "recommendedPrefillTile" eval/models.ts`
Expected: `0`.

- [ ] **Step 4: Verify typecheck passes**

Run: `make typecheck`
Expected: no errors. (If `BenchmarkModel` is used elsewhere with `recommendedPrefillTile`, the compiler will surface it now.)

### Task 2.4 — Delete `tests/eval-models.test.ts` registry-shape block

**Files:**
- Modify: `tests/eval-models.test.ts:17-45`

- [ ] **Step 1: Delete the entire `describe("recommendedPrefillTile auto-default", …)` block (lines 17-45)**

After deletion, `tests/eval-models.test.ts` should contain only the `arctic-embed registration` describe block. Run:

```bash
grep -c "recommendedPrefillTile" tests/eval-models.test.ts
```

Expected: `0`.

### Task 2.5 — Run full ship gate

- [ ] **Step 1: Run `make checkall`**

Run: `make checkall`
Expected: tail shows `428 pass / 11 skip / 0 fail` (430 from Phase 1, minus the 2 deleted registry-shape tests).

- [ ] **Step 2: Do NOT commit yet**

Phase 3 verifies in the browser; Phase 4 produces the single final commit.

---

## Phase 3 — Browser smoke regression

**Setup:** Use the existing agentchrome session on port 61142 and the smoke server on port 8031. If either is missing, `agentchrome connect --status` and `make smoke-restart` per `CLAUDE.md`'s "Smoke serve" workflow.

### Task 3.1 — Rebuild bundle and confirm smoke server fresh

- [ ] **Step 1: Rebuild the bundle**

Run: `bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser`
Expected: stdout ends with `webllm-bundle.js  189.4X KB  (entry point)` (current baseline is 189416 bytes; the helper adds ~9 lines, expect a few hundred bytes growth).

No WASM rebuild needed — the change is TS-only.

- [ ] **Step 2: Confirm smoke server is up**

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8031/`
Expected: `200`.

If 000 / no response: `make smoke-restart`, then re-run the curl.

- [ ] **Step 3: Confirm agentchrome session is reachable**

Run: `agentchrome connect --status 2>&1 | tail -1`
Expected: a JSON blob with `"reachable":true`.

If not reachable: `agentchrome connect --launch --headless` then re-check status.

### Task 3.2 — B.1: 7B+ auto-default works (Mistral-7B)

- [ ] **Step 1: Find an existing real-model tab**

Run: `agentchrome --port 61142 tabs list | grep -o '"id":"[^"]*"[^}]*real-model' | head -1`

If none, navigate the active tab to the smoke page first via `agentchrome --port 61142 navigate http://localhost:8031/real-model.html`.

- [ ] **Step 2: Navigate the smoke tab to Mistral-7B with no `?prefillTile=`**

Run (replace `<TAB_ID>` with the tab id from Step 1):

```bash
agentchrome --port 61142 --tab <TAB_ID> navigate \
  "http://localhost:8031/real-model.html?model=mistral-7b-instruct-v0.3-q4ks&ctx=4096&v=$(date +%s)&max=24&prompt=Software+engineering"
```

- [ ] **Step 3: Wait for completion**

Run: `agentchrome --port 61142 --tab <TAB_ID> page wait --selector "#log" --text "[8/8]"`

If the wait times out, the smoke session likely failed; check `page text` for the abort.

- [ ] **Step 4: Confirm the `tile: 128` pill is present and prefill completed**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> js exec '(() => { const pill = [...document.querySelectorAll("#mode-bar .mode-pill")].find(e => e.textContent.startsWith("tile:")); const log = document.getElementById("log").textContent; return JSON.stringify({ pill: pill ? pill.textContent : null, log_has_88: log.includes("[8/8]"), abort: /ggml-alloc\.c:82/i.test(log) }); })()'
```

Expected JSON: `{"pill":"tile: 128","log_has_88":true,"abort":false}`.

If `pill === null` or `pill !== "tile: 128"`: B.1 FAIL — Phase 3 halts; revert with `git diff` review and rebuild.

If `abort === true`: ggml-alloc abort fired → heuristic not threading through. Halt.

### Task 3.3 — B.2: Sub-7B fast path preserved (TinyLlama)

- [ ] **Step 1: Navigate the same tab to TinyLlama with no `?prefillTile=`**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> navigate \
  "http://localhost:8031/real-model.html?model=tinyllama-1.1b-chat-q4_0&ctx=2048&v=$(date +%s)&max=24"
```

- [ ] **Step 2: Wait for completion**

Run: `agentchrome --port 61142 --tab <TAB_ID> page wait --selector "#log" --text "[8/8]"`

- [ ] **Step 3: Confirm NO tile pill and prefill completed**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> js exec '(() => { const pill = [...document.querySelectorAll("#mode-bar .mode-pill")].find(e => e.textContent.startsWith("tile:")); const log = document.getElementById("log").textContent; return JSON.stringify({ pill: pill ? pill.textContent : null, log_has_88: log.includes("[8/8]") }); })()'
```

Expected JSON: `{"pill":null,"log_has_88":true}`.

If `pill !== null`: B.2 FAIL — heuristic is wrongly tile-on for TinyLlama (regression — would hit §22's +81% TTFT). Halt.

### Task 3.4 — B.3: Force-disable still works (Qwen3-8B with `?prefillTile=0`)

- [ ] **Step 1: Navigate to Qwen3-8B with explicit force-disable**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> navigate \
  "http://localhost:8031/real-model.html?model=qwen3-8b-iq3m&ctx=4096&v=$(date +%s)&max=24&prefillTile=0&prompt=Software+engineering+is+the+systematic+application+of+engineering+principles+to+the+design"
```

This intentionally reproduces the §22 abort to prove the override path works.

- [ ] **Step 2: Wait for either completion or abort**

Run: `agentchrome --port 61142 --tab <TAB_ID> page wait --selector "#log" --text-or-regex "(ggml-alloc\.c:82|\[8/8\])" --timeout 90`

(If your `agentchrome page wait` doesn't support `--text-or-regex`, use two separate sequential waits or fall back to a `js exec` polling pattern.)

- [ ] **Step 3: Confirm NO tile pill and prefill aborted with §22 signature**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> js exec '(() => { const pill = [...document.querySelectorAll("#mode-bar .mode-pill")].find(e => e.textContent.startsWith("tile:")); const log = document.getElementById("log").textContent; return JSON.stringify({ pill: pill ? pill.textContent : null, abort: /ggml-alloc\.c:82|node_510 needed/i.test(log) }); })()'
```

Expected JSON: `{"pill":null,"abort":true}`.

If `pill !== null` (override didn't disable) or `abort === false` (something else changed): B.3 FAIL. Halt.

### Task 3.5 — B.4: Force-enable still works (TinyLlama with `?prefillTile=128`)

- [ ] **Step 1: Navigate to TinyLlama with explicit force-enable**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> navigate \
  "http://localhost:8031/real-model.html?model=tinyllama-1.1b-chat-q4_0&ctx=2048&v=$(date +%s)&max=24&prefillTile=128"
```

- [ ] **Step 2: Wait for completion**

Run: `agentchrome --port 61142 --tab <TAB_ID> page wait --selector "#log" --text "[8/8]"`

- [ ] **Step 3: Confirm tile pill present**

Run:

```bash
agentchrome --port 61142 --tab <TAB_ID> js exec '(() => { const pill = [...document.querySelectorAll("#mode-bar .mode-pill")].find(e => e.textContent.startsWith("tile:")); return pill ? pill.textContent : null; })()'
```

Expected: `"tile: 128"`.

If null: B.4 FAIL — explicit positive override didn't thread through. Halt.

### Task 3.6 — Phase 3 gate

- [ ] **Step 1: Confirm B.1 / B.2 / B.3 / B.4 all reported expected results**

If any failed: `git checkout -- src/inference/model-inference.ts smoke-test/real-model-page.js eval/models.ts eval/perf.ts tests/`, restore the prior bundle (`bun build ...` again on `main`'s pre-edit state), and re-open the brainstorm.

If all four passed: proceed to Phase 4.

---

## Phase 4 — TODO §30 closure + delete probe + commit

### Task 4.1 — Delete the probe script

**Files:**
- Delete: `eval/probe-prefill-tile-heuristic.ts`

- [ ] **Step 1: Remove the probe file**

Run: `rm eval/probe-prefill-tile-heuristic.ts`

The probe is intentionally not committed; deleting cleans the working tree.

- [ ] **Step 2: Confirm no other file references it**

Run: `grep -rn "probe-prefill-tile-heuristic" . --include="*.ts" --include="*.js" --include="*.md" 2>/dev/null`
Expected: only this plan file matches (in the prose).

### Task 4.2 — Add §30 closure to TODO.md

**Files:**
- Modify: `TODO.md` — three locations.

- [ ] **Step 1: Append a §30 entry alongside the §29 entry near line 1500**

Search for the `§29.` line in `TODO.md` (`grep -n "^§29\." TODO.md`). Immediately after the §29 paragraph, append:

```text
§30. ~~Heuristic-based prefill-tile default in `ModelInference`.~~ **CLOSED 2026-04-28 — refactor landed on `main`.** Replaced §23's dual-source-of-truth pattern (`recommendedPrefillTile` field on `BenchmarkModel` + mirrored `RECOMMENDED_PREFILL_TILE` map in `smoke-test/real-model-page.js`) with `computeDefaultPrefillTileSize(hp)` exported from `src/inference/model-inference.ts`. Rule: `layerCount >= 32 AND embeddingLength >= 4096` → 128, else 0. Maps directly to the §22 abort signature ("32 layers × seq=512 of F32 intermediates"). Pre-edit Phase 0 probe validated all 19 registered models classify identically to the prior registry. Tile pill in the smoke page now renders post-ctor from `inference.prefillTileSize` so the auto-default is visible without page-side duplication. Override surfaces unchanged: `{ prefillTileSize: N }` ctor opt, `?prefillTile=N` URL, `--prefill-tile <n>` CLI all win, including the explicit-zero force-disable path. Net change: ~−31 LOC, 427 → 428 tests. Spec: `docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md`. Plan: `docs/superpowers/plans/2026-04-28-prefill-tile-heuristic.md`.
```

- [ ] **Step 2: Update the resumption-checklist "all closed" marker**

Search for the `all closed` line near `§17/§18/...`. Append `/§30` to the list (mirrors the §29 update from commit `cf6dd4a`).

The line currently reads (find via `grep -n "§29" TODO.md` — the marker is in the resumption checklist around line 1486):

```text
below; A/B/C/F/§4-decode/§C-v1/§4-prefill/§C-v2-A/§D/§22/§24/§26/§27/§28/§29
```

Update to:

```text
below; A/B/C/F/§4-decode/§C-v1/§4-prefill/§C-v2-A/§D/§22/§24/§26/§27/§28/§29/§30
```

Also update the dashboard-state-check passage (`grep -n "§17/§18/§19/§20/§21/§22/§23/§24/§26/§27/§28/§29" TODO.md`) the same way — append `/§30`.

- [ ] **Step 3: Flip candidate-list item #4 to closed**

Search for the candidate-list block (`grep -n "Heuristic-based prefill-tile default" TODO.md`). The current text reads:

```text
4. **Heuristic-based prefill-tile default in `ModelInference`.**
   §23 lever (b) — fold the per-model registry into a
   ctor-side derivation from `hyperparams.layerCount ×
   embeddingLength`. Nice-to-have when the registered 7B+
   fleet grows past hand-curation; defer until that pressure
   actually exists.
```

Replace with:

```text
4. ~~**Heuristic-based prefill-tile default in `ModelInference`.**~~
   **CLOSED 2026-04-28 — §30.** §23 lever (b) landed: registry
   field `recommendedPrefillTile` and smoke mirror map deleted;
   ctor now derives from `hyperparams.layerCount ×
   embeddingLength` via `computeDefaultPrefillTileSize`. All 19
   registered models classify identically to the prior registry
   (Phase 0 probe). Override surfaces unchanged.
```

### Task 4.3 — Single commit on `main`

- [ ] **Step 1: Stage all source edits**

```bash
git add src/inference/model-inference.ts \
        tests/prefill-tiling-config.test.ts \
        tests/eval-models.test.ts \
        eval/models.ts \
        eval/perf.ts \
        smoke-test/real-model-page.js \
        TODO.md
```

(The probe file should already be deleted from disk and was never `git add`-ed, so it does not appear in `git status`.)

- [ ] **Step 2: Confirm the staged diff matches expectations**

Run: `git diff --cached --stat`
Expected: 7 files changed, ~+50 / ~−80 lines (net ~−30 lines per the spec).

- [ ] **Step 3: Run `make checkall` one more time on the staged tree**

Run: `make checkall`
Expected: `428 pass / 11 skip / 0 fail`.

- [ ] **Step 4: Commit**

```bash
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
refactor(prefill-tile): replace dual-registry pattern with hyperparam heuristic

§30 closure for candidate-list item #4 ("Heuristic-based prefill-tile
default in ModelInference") opened by §23.

Replaces §23's dual-source-of-truth pattern (eval/models.ts
`recommendedPrefillTile` field on BenchmarkModel + mirrored
RECOMMENDED_PREFILL_TILE map in smoke-test/real-model-page.js) with
`computeDefaultPrefillTileSize(hp)` exported from
src/inference/model-inference.ts.

Rule: layerCount >= 32 AND embeddingLength >= 4096 → 128 else 0.
Maps directly to the §22 abort signature ("32 layers × seq=512 of
F32 intermediates exceeds graph allocator budget at ggml-alloc.c:82").

Pre-edit Phase 0 probe (eval/probe-prefill-tile-heuristic.ts; deleted
post-validation) confirmed every downloaded registered model
classifies identically to the prior `recommendedPrefillTile` field.

Tile pill in the smoke page now renders post-ctor from
inference.prefillTileSize so the auto-default is visible without
page-side duplication of the heuristic.

Override surfaces unchanged: { prefillTileSize: N } ctor opt,
?prefillTile=N URL param, --prefill-tile <n> CLI flag all win,
including the explicit-zero force-disable path.

Test count: 427 → 428 (deleted 2 registry-shape tests in
tests/eval-models.test.ts; added 3 heuristic boundary tests in
tests/prefill-tiling-config.test.ts).

Browser smoke regression (B.1-B.4 from spec) verified:
- B.1: mistral-7b auto-defaults to tile=128, prefill completes.
- B.2: tinyllama auto-defaults to tile=0, prefill completes,
  no §22 +81% TTFT regression.
- B.3: qwen3-8b with ?prefillTile=0 forces tile off, ggml-alloc
  abort surfaces (override path healthy).
- B.4: tinyllama with ?prefillTile=128 forces tile on, pill
  visible (override path healthy).

Spec: docs/superpowers/specs/2026-04-28-prefill-tile-heuristic-design.md
Plan: docs/superpowers/plans/2026-04-28-prefill-tile-heuristic.md
checkall: 428 pass / 11 skip / 0 fail.
EOF
)"
```

- [ ] **Step 5: Confirm post-commit state**

Run: `git log --oneline -3 && git status`
Expected:
- Top of log shows `<sha> refactor(prefill-tile): replace dual-registry pattern with hyperparam heuristic`.
- `git status`: `nothing to commit, working tree clean` (probe file already removed; bundle artifact gitignored).

### Task 4.4 — Final verification

- [ ] **Step 1: Confirm no probe file leaked**

Run: `ls eval/probe-prefill-tile-heuristic.ts 2>&1`
Expected: `ls: ... No such file or directory`.

- [ ] **Step 2: Confirm no `recommendedPrefillTile` references remain anywhere**

Run: `grep -rn "recommendedPrefillTile\|RECOMMENDED_PREFILL_TILE" . --include="*.ts" --include="*.js" --include="*.md" 2>/dev/null | grep -v "^./TODO.md\|^./docs/superpowers/"`

Expected: empty (`TODO.md` and the spec/plan still reference the field by name as historical context — those are filtered out).

If any non-filtered hit remains, that's a leftover reference; fix and amend the commit before reporting done.

- [ ] **Step 3: Report task #438 completion to caller**

End-of-plan report should be one or two sentences per `CLAUDE.md`'s end-of-turn convention: "§30 landed at `<sha>`; checkall 428/11/0; smoke regression clean."

---

## Self-review notes (resolved before plan was committed)

- Spec coverage: every spec section has at least one task. §A.1 → Phase 0; helper + ctor wiring → Task 1.3; tests → Task 1.2; eval/models.ts deletion → Task 2.3; smoke page → Task 2.2 (incl. post-ctor pill move not in spec but required for UX correctness — added here); perf.ts → Task 2.1; eval-models.test.ts deletion → Task 2.4; B.1-B.4 → Tasks 3.2-3.5; commit shape → Task 4.3.
- The smoke-page post-ctor pill render (Task 2.2 Step 4) goes beyond the spec, which only said "delete the map and the conditional." The spec implicitly preserved the pill; UX would silently regress otherwise. Captured here so the spec self-review wasn't load-bearing.
- Task 1.1 (STUB_HP retune) addresses a hidden interaction: the existing `defaults to 0` test would break under the new heuristic if STUB_HP's layerCount/embeddingLength stayed at 32/4096. This is not in the spec because the spec didn't enumerate test-fixture coupling.
- Type consistency: `computeDefaultPrefillTileSize` is the same name in helper, ctor wiring, doc comment, commit message, and TODO entry.
- Placeholder scan: no TBD/TODO/etc. remain.
