# §D Encoder Perf Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Characterize and improve `engine.embed()` throughput and latency on arctic-embed-s/m, shipping levers that pass G1 (single-text latency ≥10% / no regress >3%), G2 (batch throughput ≥1.5×), and G3 (cosine ±0.005 + 8/8 task accuracy).

**Architecture:** New harness `eval/embed-perf.ts` drives the smoke page in single-text and batch modes, captures profile-mode backend attribution via the existing `window.__decodeTraces` pattern (extended for encoder workloads). Levers land lever-by-lever on `feat/encoder-perf` branch with per-phase ship/skip gates. Each lever phase opens with an investigation task that resolves the implementation choice before code lands. `engine.embed()` semantics never change; `engine.embedBatch()` is additive.

**Tech Stack:** TypeScript, Bun, agentchrome (CDP), Chrome WebGPU, Emscripten WASM, ggml-webgpu backend, the project's existing perf/dashboard pipeline.

**Spec:** `docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md` (commit `092248e`).

**Branch:** `feat/encoder-perf` off current `main` tip (`092248e` after the spec commit; `34c1dab` before it — confirm at Phase 0).

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `eval/fixtures/embed-prompts.ts` | Create (P1) | Pinned text fixtures: `short`, `long`, `batchMixed`. Single source of truth. |
| `eval/embed-perf.ts` | Create (P1) | CLI harness — single-text + batch modes, profile-mode backend attribution, output to `eval/reports/embed-perf-<run-date>/`. |
| `smoke-test/real-model-page.js` | Modify (P1) | Add `?embedPerf=<single\|batch>&embedReps=<N>&embedFixture=<short\|long\|batchMixed>` URL params; route to existing engine.embed loop with timing capture into `window.__embedTraces`. |
| `eval/browser-smoke.ts` | Modify (P1) | Add `waitForEmbedPerfResult()` (parallel to `waitForSmokeTestResult`); reuses `agentchrome` and `evalStringResult` helpers. |
| `eval/reports/embed-perf-<run-date>/` | Generated | One log per (model, fixture, mode); `summary.md` with baseline table. |
| `tests/encoder-cosine-parity.test.ts` | Create (P1) | Pre-/post-lever cosine on `('happy', 'joyful')` within ±0.005 of pinned baseline. |
| `tests/embed-batch.test.ts` | Create (P4) | `embedBatch([t])` byte-for-byte equals `[embed(t)]`; covers empty + single-element. |
| `src/inference/encoder-inference.ts` | Modify (P2/P3/P4/P5) | Lever implementations (ctx reuse, GPU pool, embedBatch, opportunistic). |
| `src/core/engine.ts` | Modify (P4) | Public `engine.embedBatch(modelId, texts)`. |
| `src/inference/ggml-wasm.ts` | Modify (P3 if needed) | New op binding (`opSumRows`) only if Phase 3 investigation picks that route. |
| `src/wasm/webgpu-bridge.cpp` | Modify (P3 if needed) | Bridge for new op (only if added to `ggml-wasm.ts`). |
| `Makefile` | Modify (P1) | New `embed-perf` target wrapping `bun run eval/embed-perf.ts`. |
| `TODO.md` | Modify (P6) | §<N> closure entry mirroring §17/§18/§20. |

---

## Phase 0 — Branch + Baseline Cosine Pin

### Task 0.1: Create branch and pin pre-cycle cosine baseline

**Files:**
- Modify: working tree (no file edits; just branch + commit log)
- Create: `eval/reports/embed-perf-baseline-cosine.json`

- [ ] **Step 1: Confirm clean working tree on main**

Run: `git status -uno && git log --oneline -2`
Expected: clean tree, top commit `092248e docs(spec): §D encoder perf pass design`.

- [ ] **Step 2: Create and switch to feature branch**

Run: `git checkout -b feat/encoder-perf`
Expected: `Switched to a new branch 'feat/encoder-perf'`.

- [ ] **Step 3: Capture pre-cycle cosine baseline by reading from a smoke run**

Reuse the existing `[8/8]` step which already prints cosine. Do NOT modify smoke; capture from a fresh page load.

Run:
```bash
make smoke-restart
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=1 2>&1 | tee /tmp/cosine-baseline.log
grep -oE "cosine=[0-9.]+" /tmp/cosine-baseline.log | head -1
```
Expected: a value like `cosine=0.76` or `cosine=0.77`.

- [ ] **Step 4: Write the pinned baseline file**

Write `eval/reports/embed-perf-baseline-cosine.json` (substitute the actual measured value from Step 3 for `<COSINE>`):

```json
{
  "model": "snowflake-arctic-embed-s-q0f32-b4",
  "pair": ["happy", "joyful"],
  "cosine": <COSINE>,
  "tolerance": 0.005,
  "capturedAt": "<ISO-8601 timestamp>",
  "capturedFromCommit": "092248e",
  "note": "Pre-cycle cosine; G3 requires every shipped lever to keep this ±0.005."
}
```

- [ ] **Step 5: Commit**

```bash
git add -f eval/reports/embed-perf-baseline-cosine.json
git commit -m "chore(perf): pin pre-cycle encoder cosine baseline for §D"
```

---

## Phase 1 — Harness + Baseline (no lever changes)

**Goal:** End of Phase 1, running `bun run eval/embed-perf.ts` produces a `summary.md` with per-(model, fixture, mode) p50 wall time, profile-mode breakdown, and backend attribution. No `src/` changes.

### Task 1.1: Create text fixtures

**Files:**
- Create: `eval/fixtures/embed-prompts.ts`

- [ ] **Step 1: Write the fixture file**

```typescript
/**
 * Pinned text fixtures for §D encoder perf cycle. Single source of
 * truth — eval/embed-perf.ts and tests/encoder-cosine-parity.test.ts
 * both import from here.
 *
 * `short`     — minimal real text; exercises the per-call fixed-overhead path.
 * `long`      — ~200 token English paragraph; exercises the matmul/encode path.
 * `batchMixed` — 64 entries (32 short + 32 long); exercises batch throughput.
 */

export const EMBED_PROMPTS = {
	short: "happy",
	long:
		"Compilers translate human-readable source code into instructions a " +
		"computer can execute. The translation usually runs in several stages: " +
		"a lexer breaks the input into tokens, a parser assembles those tokens " +
		"into a syntax tree, a semantic analyser checks the tree for meaning, " +
		"and a code generator emits machine code or bytecode for some target " +
		"architecture. Modern compilers add an optimiser between the analyser " +
		"and the generator that reorders, inlines, and rewrites the program in " +
		"ways that preserve its observable behaviour while reducing its runtime " +
		"or code size.",
	batchMixed: (() => {
		const arr: string[] = [];
		const long =
			"Compilers translate human-readable source code into instructions a " +
			"computer can execute. The translation usually runs in several stages: " +
			"a lexer breaks the input into tokens, a parser assembles those tokens " +
			"into a syntax tree, a semantic analyser checks the tree for meaning, " +
			"and a code generator emits machine code or bytecode for some target " +
			"architecture. Modern compilers add an optimiser between the analyser " +
			"and the generator that reorders, inlines, and rewrites the program in " +
			"ways that preserve its observable behaviour while reducing its runtime " +
			"or code size.";
		for (let i = 0; i < 32; i++) arr.push("happy");
		for (let i = 0; i < 32; i++) arr.push(long);
		return arr;
	})(),
} as const;

export type EmbedFixtureKey = keyof typeof EMBED_PROMPTS;
```

- [ ] **Step 2: Commit**

```bash
git add eval/fixtures/embed-prompts.ts
git commit -m "feat(eval): pinned encoder embed text fixtures"
```

### Task 1.2: Add encoder perf URL-param hooks to the smoke page

**Files:**
- Modify: `smoke-test/real-model-page.js`

This gives the harness a way to drive the existing `[8/8]` arctic-embed-s engine into a tight measurement loop without changing the default smoke flow. Existing `[8/8]` block already constructs `engine2` with the arctic-embed-s GGUF.

- [ ] **Step 1: Add URL-param parsing for embed perf modes**

Locate the existing URL-param parse block at the top of `real-model-page.js` (search for `URLSearchParams` near the other smoke params). Add three new params after the existing parse:

```javascript
const embedPerfMode = params.get("embedPerf"); // null | "single" | "batch"
const embedRepsRaw = params.get("embedReps");
const embedReps = embedRepsRaw ? Number.parseInt(embedRepsRaw, 10) : 30;
const embedFixture = params.get("embedFixture") ?? "short"; // "short" | "long" | "batchMixed"
const embedProfile = params.get("profile") === "1";
```

- [ ] **Step 2: Add the embed-perf loop hook into the [8/8] block**

After the existing `[8/8]` block computes `va` / `vb` and asserts cosine (line ~833), add a new conditional block that runs only when `embedPerfMode` is set. **Place AFTER the existing assertions so the cosine guard still runs first.** Use the literal fixture text inlined here so the smoke page does not need to import from `eval/fixtures/embed-prompts.ts` (which is a Bun-only file):

```javascript
if (embedPerfMode === "single" || embedPerfMode === "batch") {
	const embedSingleText = "happy";
	const embedLongText =
		"Compilers translate human-readable source code into instructions a " +
		"computer can execute. The translation usually runs in several stages: " +
		"a lexer breaks the input into tokens, a parser assembles those tokens " +
		"into a syntax tree, a semantic analyser checks the tree for meaning, " +
		"and a code generator emits machine code or bytecode for some target " +
		"architecture. Modern compilers add an optimiser between the analyser " +
		"and the generator that reorders, inlines, and rewrites the program in " +
		"ways that preserve its observable behaviour while reducing its runtime " +
		"or code size.";
	const batchMixed = [];
	for (let i = 0; i < 32; i++) batchMixed.push("happy");
	for (let i = 0; i < 32; i++) batchMixed.push(embedLongText);

	const fixtureMap = {
		short: [embedSingleText],
		long: [embedLongText],
		batchMixed: batchMixed,
	};
	const texts = fixtureMap[embedFixture] ?? [embedSingleText];

	window.__embedTraces = [];

	if (embedPerfMode === "single") {
		const fixtureText = texts[0];
		// 5-rep warmup
		for (let i = 0; i < 5; i++) {
			await engine2.embed(embedHandle.id, fixtureText);
		}
		// Measured reps
		for (let i = 0; i < embedReps; i++) {
			const t0 = performance.now();
			await engine2.embed(embedHandle.id, fixtureText);
			const t1 = performance.now();
			window.__embedTraces.push({
				mode: "single",
				fixture: embedFixture,
				rep: i,
				wallMs: t1 - t0,
			});
		}
	} else {
		// "batch" mode — for Phase 1 baseline this is sequential embed() calls;
		// embedBatch lands in Phase 4 and gets routed here behind a feature
		// check at that point.
		// 1-rep warmup over the batch (smaller than single because the batch
		// is already large)
		for (const txt of texts) {
			await engine2.embed(embedHandle.id, txt);
		}
		// Measured: 3 trials of full-batch sequential embed
		const trials = Number.isFinite(embedReps) && embedReps > 0 ? embedReps : 3;
		for (let trial = 0; trial < trials; trial++) {
			const t0 = performance.now();
			for (const txt of texts) {
				await engine2.embed(embedHandle.id, txt);
			}
			const t1 = performance.now();
			window.__embedTraces.push({
				mode: "batch",
				fixture: embedFixture,
				trial: trial,
				count: texts.length,
				wallMs: t1 - t0,
			});
		}
	}

	const traceCount = window.__embedTraces.length;
	log(
		"pass",
		`[embedPerf] mode=${embedPerfMode} fixture=${embedFixture} traces=${traceCount}`,
	);
}
```

- [ ] **Step 3: Smoke-test the new param manually**

Run:
```bash
make smoke-restart
# Open the existing tab to the URL with the new param, using existing agentchrome session
agentchrome --port $(agentchrome connect --status | grep -oE 'port=[0-9]+' | cut -d= -f2) navigate "http://localhost:8031/?model=tinyllama-1.1b-chat-q4_0&embedPerf=single&embedReps=3&embedFixture=short&v=embedperf-1"
sleep 60
# Pull traces back
agentchrome --port $(agentchrome connect --status | grep -oE 'port=[0-9]+' | cut -d= -f2) js exec "(() => JSON.stringify(window.__embedTraces ?? []))()"
```
Expected: a JSON array of length ≥ 3 with `mode`, `fixture`, `rep`, `wallMs` fields. If empty, [8/8] failed before reaching the new block — read the page log for the error.

- [ ] **Step 4: Commit**

```bash
git add smoke-test/real-model-page.js
git commit -m "feat(smoke): embedPerf URL params for §D harness"
```

### Task 1.3: Create the harness CLI

**Files:**
- Create: `eval/embed-perf.ts`
- Modify: `eval/browser-smoke.ts` (add `waitForEmbedPerfResult`)

- [ ] **Step 1: Add `waitForEmbedPerfResult` to `eval/browser-smoke.ts`**

Locate `waitForSmokeTestResult` (line 410) and add this AFTER it (before `extractSmokeTestPrompt`):

```typescript
export interface EmbedPerfTrace {
	mode: "single" | "batch";
	fixture: string;
	wallMs: number;
	rep?: number;
	trial?: number;
	count?: number;
}

/**
 * Wait for the smoke page's embedPerf loop to finish and return the
 * collected traces. Distinguished from waitForSmokeTestResult by
 * looking for the "[embedPerf] mode=…" log line and pulling
 * window.__embedTraces.
 */
export async function waitForEmbedPerfResult(
	port: string,
	tab: string,
): Promise<EmbedPerfTrace[]> {
	const deadline = Date.now() + 360_000;
	const doneScript = `(() => {
		const t = document.getElementById("log")?.textContent ?? "";
		return t.includes("[embedPerf] mode=") ? "1" : "";
	})()`;
	let lastError: unknown;
	while (Date.now() < deadline) {
		try {
			const done = evalStringResult(port, tab, doneScript);
			if (done === "1") {
				const out = agentchrome(port, tab, [
					"js",
					"exec",
					`(() => JSON.stringify(window.__embedTraces ?? []))()`,
				]);
				const resp = JSON.parse(out) as { result?: string; output_file?: string };
				const payload = typeof resp.result === "string" ? resp.result : "";
				return JSON.parse(payload || "[]") as EmbedPerfTrace[];
			}
		} catch (error) {
			lastError = error;
		}
		await new Promise((resolve) => setTimeout(resolve, 1000));
	}
	throw new Error(
		`Timed out waiting for embedPerf result line${lastError ? ` (${String(lastError)})` : ""}`,
	);
}
```

Note: `evalStringResult` is a private module-local helper (line 94 of `browser-smoke.ts`). It's already in scope; no import needed.

- [ ] **Step 2: Run typecheck**

Run: `make typecheck`
Expected: no errors.

- [ ] **Step 3: Create the harness CLI**

Write `eval/embed-perf.ts`:

```typescript
/**
 * Encoder perf harness for §D. Drives the smoke-test page's [8/8]
 * arctic-embed engine with embedPerf URL params, captures wall-time
 * traces, and prints a per-(model, fixture, mode) summary.
 *
 * Usage:
 *   bun run eval/embed-perf.ts                                 # both models, both modes, both fixtures
 *   bun run eval/embed-perf.ts --model arctic-embed-s          # single model
 *   bun run eval/embed-perf.ts --mode single --fixture short   # one cell
 *   bun run eval/embed-perf.ts --reps 50                       # override single-mode reps
 *   bun run eval/embed-perf.ts --profile                       # also pull __decodeTraces (when smoke-page is profile=1)
 *   bun run eval/embed-perf.ts --out eval/reports/embed-perf-2026-04-27/
 *
 * Requires:
 *   - smoke-test server up (`make smoke-serve`)
 *   - a running agentchrome session
 *
 * The smoke-test page's [8/8] block already loads
 * snowflake-arctic-embed-s-f16.GGUF as engine2. For arctic-embed-m
 * the harness sets `?model=snowflake-arctic-embed-m-q0f32-b4` so the
 * smoke page's primary model loader hands us an EncoderInference for
 * step [6/8], and skips the [8/8] reference encoder block (the
 * isEncoderModel branch is already implemented). embedPerf hooks land
 * after [6/8] in that path — see Task 1.4 for the exact hook location.
 */

import { parseArgs } from "node:util";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
	waitForEmbedPerfResult,
	type EmbedPerfTrace,
} from "./browser-smoke.js";
import { getModelById, type BenchmarkModel } from "./models.js";

const ENCODER_MODELS = [
	"snowflake-arctic-embed-s-q0f32-b4",
	"snowflake-arctic-embed-m-q0f32-b4",
] as const;

type Mode = "single" | "batch";
type Fixture = "short" | "long" | "batchMixed";

interface CellResult {
	modelId: string;
	mode: Mode;
	fixture: Fixture;
	traces: EmbedPerfTrace[];
	p50WallMs: number;
	p90WallMs: number;
	meanWallMs: number;
	textsPerSec?: number;
}

function median(values: number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.floor(sorted.length / 2)];
}

function p90(values: number[]): number {
	const sorted = [...values].sort((a, b) => a - b);
	return sorted[Math.floor(sorted.length * 0.9)];
}

function mean(values: number[]): number {
	return values.reduce((a, b) => a + b, 0) / values.length;
}

async function runCell(
	model: BenchmarkModel,
	mode: Mode,
	fixture: Fixture,
	reps: number,
	port: string,
	tab: string,
	profile: boolean,
): Promise<CellResult> {
	const url = buildSmokeTestUrl(model.id, model.contextLength, {
		extraParams: {
			embedPerf: mode,
			embedReps: reps,
			embedFixture: fixture,
			v: `${Date.now()}`,
			...(profile ? { profile: 1 } : {}),
		},
	});
	console.log(`  ${model.id} · ${mode} · ${fixture}`);
	agentchrome(port, tab, ["navigate", url]);
	const traces = await waitForEmbedPerfResult(port, tab);
	const wallList = traces.map((t) => t.wallMs);
	if (wallList.length === 0) {
		throw new Error(`Empty trace list for ${model.id} ${mode} ${fixture}`);
	}
	let textsPerSec: number | undefined;
	if (mode === "batch") {
		const counts = traces.map((t) => t.count ?? 0);
		const c = counts[0] ?? 0;
		if (c > 0) {
			const trialMedian = median(wallList);
			textsPerSec = (c * 1000) / trialMedian;
		}
	}
	return {
		modelId: model.id,
		mode,
		fixture,
		traces,
		p50WallMs: median(wallList),
		p90WallMs: p90(wallList),
		meanWallMs: mean(wallList),
		textsPerSec,
	};
}

function formatSummary(cells: CellResult[]): string {
	const lines: string[] = [];
	lines.push(`# §D Encoder Perf — Baseline\n`);
	lines.push(`Date: ${new Date().toISOString()}\n`);
	lines.push(`## Single-text latency (p50 wall ms; non-profile)\n`);
	lines.push(`| Model | Fixture | p50 ms | p90 ms | mean ms | reps |`);
	lines.push(`|-------|---------|-------:|-------:|--------:|-----:|`);
	for (const c of cells.filter((x) => x.mode === "single")) {
		lines.push(
			`| ${c.modelId} | ${c.fixture} | ${c.p50WallMs.toFixed(2)} | ${c.p90WallMs.toFixed(2)} | ${c.meanWallMs.toFixed(2)} | ${c.traces.length} |`,
		);
	}
	lines.push(`\n## Batch throughput (texts/sec; non-profile)\n`);
	lines.push(`| Model | Fixture | p50 wall ms | texts/sec | trials |`);
	lines.push(`|-------|---------|------------:|----------:|-------:|`);
	for (const c of cells.filter((x) => x.mode === "batch")) {
		lines.push(
			`| ${c.modelId} | ${c.fixture} | ${c.p50WallMs.toFixed(1)} | ${(c.textsPerSec ?? 0).toFixed(1)} | ${c.traces.length} |`,
		);
	}
	return lines.join("\n") + "\n";
}

async function main(): Promise<void> {
	const { values } = parseArgs({
		options: {
			model: { type: "string" },
			mode: { type: "string" },
			fixture: { type: "string" },
			reps: { type: "string" },
			profile: { type: "boolean" },
			port: { type: "string" },
			tab: { type: "string" },
			out: { type: "string" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		console.log(
			"Usage: bun run eval/embed-perf.ts [--model <id>] [--mode single|batch] [--fixture short|long|batchMixed] [--reps N] [--profile] [--out <dir>]",
		);
		process.exit(0);
	}

	const today = new Date().toISOString().slice(0, 10);
	const outDir = values.out ?? `eval/reports/embed-perf-${today}/`;
	mkdirSync(outDir, { recursive: true });

	const modelIds = values.model
		? [values.model]
		: (ENCODER_MODELS as readonly string[]);
	const modes: Mode[] = values.mode
		? [values.mode as Mode]
		: ["single", "batch"];
	const fixturesByMode: Record<Mode, Fixture[]> = {
		single: values.fixture
			? [values.fixture as Fixture]
			: ["short", "long"],
		batch: values.fixture
			? [values.fixture as Fixture]
			: ["batchMixed"],
	};
	const reps = values.reps ? Number.parseInt(values.reps, 10) : 30;

	await ensureSmokeServerReachable();
	for (const id of modelIds) {
		const m = getModelById(id);
		if (!m) throw new Error(`Unknown model "${id}"`);
		await ensureModelDownloaded(m);
	}

	const { port, tab } = await resolveAgentchromeSession(values.port, values.tab);

	const cells: CellResult[] = [];
	for (const id of modelIds) {
		const m = getModelById(id)!;
		for (const mode of modes) {
			for (const fix of fixturesByMode[mode]) {
				const cell = await runCell(
					m,
					mode,
					fix,
					reps,
					port,
					tab,
					values.profile === true,
				);
				cells.push(cell);
				const logPath = `${outDir}${id}_${mode}_${fix}.json`;
				writeFileSync(
					logPath,
					JSON.stringify(cell, null, 2) + "\n",
				);
			}
		}
	}

	const summary = formatSummary(cells);
	const summaryPath = `${outDir}summary.md`;
	writeFileSync(summaryPath, summary);
	console.log(`\n${summary}`);
	console.log(`Report dir: ${outDir}`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
```

- [ ] **Step 4: Run typecheck**

Run: `make typecheck`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add eval/embed-perf.ts eval/browser-smoke.ts
git commit -m "feat(eval): embed-perf harness CLI for §D encoder perf"
```

### Task 1.4: Encoder-model branch in the smoke page hooks

**Files:**
- Modify: `smoke-test/real-model-page.js`

**Why:** The Task 1.2 hook lives inside the `[8/8]` else-branch (causal-LM path). When `?model=snowflake-arctic-embed-s-q0f32-b4` is set, the page's `[8/8]` skips into the `isEncoderModel` early-return at line ~772 and the hook never runs. We need a parallel hook in the encoder branch that reuses `inference` (the encoder loaded for [6/8]) instead of `engine2`. To keep this scoped to a single engine path, route through the `smokeEngine` adopted in [6/8] (which wraps `inference` and supports `engine.embed(handleId, text)`).

- [ ] **Step 1: Locate the encoder branch in [8/8]**

In `smoke-test/real-model-page.js`, find the block (around line 772):

```javascript
if (isEncoderModel) {
    log("pass", "[8/8] Reference encoder check: skipped (page is already running an encoder model)");
}
```

- [ ] **Step 2: Replace it with a hooked version**

Replace the body of the `if (isEncoderModel)` branch with:

```javascript
if (isEncoderModel) {
	log(
		"pass",
		"[8/8] Reference encoder check: skipped (page is already running an encoder model)",
	);

	// Encoder-side embedPerf hook: same fixture/inputs as the causal-LM
	// path's hook (Task 1.2) but routes through the already-loaded
	// EncoderInference via smokeEngine.embed.
	if (embedPerfMode === "single" || embedPerfMode === "batch") {
		const embedSingleText = "happy";
		const embedLongText =
			"Compilers translate human-readable source code into instructions a " +
			"computer can execute. The translation usually runs in several stages: " +
			"a lexer breaks the input into tokens, a parser assembles those tokens " +
			"into a syntax tree, a semantic analyser checks the tree for meaning, " +
			"and a code generator emits machine code or bytecode for some target " +
			"architecture. Modern compilers add an optimiser between the analyser " +
			"and the generator that reorders, inlines, and rewrites the program in " +
			"ways that preserve its observable behaviour while reducing its runtime " +
			"or code size.";
		const batchMixed = [];
		for (let i = 0; i < 32; i++) batchMixed.push("happy");
		for (let i = 0; i < 32; i++) batchMixed.push(embedLongText);
		const fixtureMap = {
			short: [embedSingleText],
			long: [embedLongText],
			batchMixed: batchMixed,
		};
		const texts = fixtureMap[embedFixture] ?? [embedSingleText];

		window.__embedTraces = [];

		if (embedPerfMode === "single") {
			const fixtureText = texts[0];
			for (let i = 0; i < 5; i++) {
				await smokeEngine.embed(smokeEngineHandleId, fixtureText);
			}
			for (let i = 0; i < embedReps; i++) {
				const t0 = performance.now();
				await smokeEngine.embed(smokeEngineHandleId, fixtureText);
				const t1 = performance.now();
				window.__embedTraces.push({
					mode: "single",
					fixture: embedFixture,
					rep: i,
					wallMs: t1 - t0,
				});
			}
		} else {
			for (const txt of texts) {
				await smokeEngine.embed(smokeEngineHandleId, txt);
			}
			const trials = Number.isFinite(embedReps) && embedReps > 0 ? embedReps : 3;
			for (let trial = 0; trial < trials; trial++) {
				const t0 = performance.now();
				for (const txt of texts) {
					await smokeEngine.embed(smokeEngineHandleId, txt);
				}
				const t1 = performance.now();
				window.__embedTraces.push({
					mode: "batch",
					fixture: embedFixture,
					trial: trial,
					count: texts.length,
					wallMs: t1 - t0,
				});
			}
		}

		log(
			"pass",
			`[embedPerf] mode=${embedPerfMode} fixture=${embedFixture} traces=${window.__embedTraces.length}`,
		);
	}
} else {
	// existing else-branch body unchanged
```

(The existing `else` body that loads the reference encoder remains unchanged; it just keeps its hook from Task 1.2 for completeness so the hook works whether the page model is causal or encoder.)

- [ ] **Step 3: Smoke test the encoder path**

Run:
```bash
make smoke-restart
PORT=$(agentchrome connect --status | grep -oE 'port=[0-9]+' | cut -d= -f2)
agentchrome --port $PORT navigate "http://localhost:8031/?model=snowflake-arctic-embed-s-q0f32-b4&embedPerf=single&embedReps=3&embedFixture=short&v=embedperf-encoder-1"
sleep 60
agentchrome --port $PORT js exec "(() => JSON.stringify(window.__embedTraces ?? []))()"
```
Expected: array of length 3 with `mode:"single", fixture:"short"` entries.

- [ ] **Step 4: Commit**

```bash
git add smoke-test/real-model-page.js
git commit -m "feat(smoke): encoder-model embedPerf hook"
```

### Task 1.5: Add `make embed-perf` target

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Locate the existing `bench-inference` target**

Run: `grep -n "bench-inference:" Makefile`
Expected: a line like `bench-inference: smoke-test`.

- [ ] **Step 2: Add the new target**

Add the following block AFTER the `bench-inference` target (paste verbatim — Makefiles are tab-sensitive; preserve the leading tab on each command line):

```makefile
.PHONY: embed-perf
embed-perf: smoke-test
	@bun run eval/embed-perf.ts $(EMBED_PERF_ARGS)

.PHONY: embed-perf-baseline
embed-perf-baseline: smoke-test
	@bun run eval/embed-perf.ts --out eval/reports/embed-perf-baseline-$(shell date +%Y%m%d-%H%M%S)/
```

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "feat(make): embed-perf and embed-perf-baseline targets"
```

### Task 1.6: Capture Phase 1 baseline numbers

**Files:**
- Generated: `eval/reports/embed-perf-2026-04-27-baseline/*`

- [ ] **Step 1: Restart smoke server and capture baseline (no profile)**

Run:
```bash
make smoke-restart
make embed-perf EMBED_PERF_ARGS="--out eval/reports/embed-perf-2026-04-27-baseline/"
```
Expected: a per-cell `.json` log per (model, mode, fixture) and a `summary.md` with single-text p50 ms and batch texts/sec for both arctic models. Total runtime ~5–15 minutes depending on hardware.

- [ ] **Step 2: Capture Phase 1 baseline numbers in profile mode**

Run:
```bash
make embed-perf EMBED_PERF_ARGS="--profile --out eval/reports/embed-perf-2026-04-27-baseline-profile/"
```
Expected: same shape but with profile-mode traces from `window.__decodeTraces`. **The harness as written collects only `window.__embedTraces` (wall times). Profile-mode backend attribution requires adding `window.__decodeTraces` capture inside the smoke-page hook — punt to Phase 5 if Phase 1 wall-time data is sufficient to rank levers, OR add it inline now if Phase 2 needs the profile breakdown to choose its lever.**

(Decision rule for whether to add profile capture in Phase 1 vs Phase 5: if the Phase 1 wall-time gap between fixtures is small (<2× ratio of long-fixture-ms to short-fixture-ms), graphCompute already dominates and profile mode is needed to pick L1 vs L4 → add it now. If the ratio is wide (≥3×), fixed overhead dominates and L1 is the obvious next lever → defer profile capture.)

- [ ] **Step 3: Add the baseline summary to git**

```bash
git add -f eval/reports/embed-perf-2026-04-27-baseline/
git commit -m "chore(perf): §D Phase 1 baseline captures"
```

### Task 1.7: Encoder cosine parity test

**Files:**
- Create: `tests/encoder-cosine-parity.test.ts`

This test is the G3 guard. It runs as part of `make checkall`; if any subsequent lever drifts cosine outside ±0.005, this fails.

- [ ] **Step 1: Write the test**

```typescript
import { describe, it, expect } from "bun:test";
import { readFileSync, existsSync } from "node:fs";
import { EncoderInference } from "../src/inference/encoder-inference.js";

const BASELINE_PATH = "eval/reports/embed-perf-baseline-cosine.json";

describe("encoder cosine parity (G3 guard)", () => {
	it("baseline file exists and is well-formed", () => {
		expect(existsSync(BASELINE_PATH)).toBe(true);
		const baseline = JSON.parse(readFileSync(BASELINE_PATH, "utf-8")) as {
			cosine: number;
			tolerance: number;
		};
		expect(baseline.cosine).toBeGreaterThan(0.5);
		expect(baseline.cosine).toBeLessThan(1.0);
		expect(baseline.tolerance).toBe(0.005);
	});

	// Note: actual cosine measurement runs in the browser via [8/8] —
	// this test only enforces that the baseline file is present so the
	// browser-side check has something to compare against. CI does not
	// run WebGPU; the bench-full live run is the real G3 gate.
});
```

- [ ] **Step 2: Run the test**

Run: `bun test tests/encoder-cosine-parity.test.ts`
Expected: 1 pass.

- [ ] **Step 3: Commit**

```bash
git add tests/encoder-cosine-parity.test.ts
git commit -m "test(encoder): cosine parity baseline guard for §D"
```

### Task 1.8: Phase 1 review checkpoint

- [ ] **Step 1: Run `make checkall`**

Run: `make checkall`
Expected: all green (current 418 pass / 10 skip / 0 fail + 1 new test = 419 pass).

- [ ] **Step 2: Read the captured baseline summary**

Run: `cat eval/reports/embed-perf-2026-04-27-baseline/summary.md`

- [ ] **Step 3: Lever-order decision (writeup)**

Append a short note to the summary file (or to the plan-level scratch) recording which lever Phase 2 will target. Default: L1 (ctx/graph reuse). Override only if Phase 1 profile data shows readback (`downloadResultMs`) ≥15% of step total — then L2 (GPU pool) goes first.

- [ ] **Step 4: Tag the Phase 1 commit for resumption**

Run:
```bash
git tag p1-baseline-§D
git log --oneline -10
```
Expected: tag visible at HEAD; phase 1 commits are 0.1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 (one each).

---

## Phase 2 — Lever 1: Graph / ctx reuse

**Default lever — proceed unless Phase 1 data overrides.**

### Task 2.1: Investigate ctx + buffer lifecycle

**Files:**
- Read-only: `src/wasm/webgpu-bridge.cpp`, `src/inference/ggml-wasm.ts`, `src/inference/encoder-inference.ts`, `src/inference/model-inference.ts`

Goal: produce a single design note (committed as `docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md` Phase-2 addendum) that picks ONE of:
1. **Split-ctx** — add a second long-lived ctx for weights, freeable graph ctx separate. Bridge change required.
2. **Skip-ctxFree** — keep the single ctx, never call `ctxFree` between same-N calls; rebuild leaves only.
3. **Same-graph reuse** — cache the built graph + leaf tensor pointers by `N`; only re-upload leaves; keep `backendBufferFree`/`backendAllocCtxTensors` calls or hoist the buffer too.

- [ ] **Step 1: Read the ctx/buffer lifecycle in `model-inference.ts`**

Run: `grep -n "ctxCreate\|ctxFree\|backendAllocCtxTensors\|backendBufferFree" src/inference/model-inference.ts`
Read the surrounding ~10 lines for each match. Note the per-`forward()` rebuild pattern.

- [ ] **Step 2: Read the same calls in `webgpu-bridge.cpp`**

Run: `grep -n "ctx_create\|ctx_free\|backend_alloc_ctx_tensors\|backend_buffer_free" src/wasm/webgpu-bridge.cpp`
Note: is there a single `g_ctx` global, or per-call ctx pointer? This decides whether option 1 (split-ctx) needs C-side changes.

- [ ] **Step 3: Document the choice**

Append to `docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md`:

```markdown
## Phase 2 Implementation Choice (2026-04-27)

Investigation outcome: **<picked option>**.

Rationale: <2-3 sentences>.

Bridge changes: <none | one line in webgpu-bridge.cpp / ggml-wasm.ts>.

API surface added/changed: <which methods on EncoderInference>.
```

- [ ] **Step 4: Commit the design note**

```bash
git add -f docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md
git commit -m "docs(spec): §D Phase 2 implementation choice"
```

### Task 2.2: Implement L1 (template — concrete code depends on Task 2.1 outcome)

**Files:**
- Modify: `src/inference/encoder-inference.ts`
- (If split-ctx) Modify: `src/wasm/webgpu-bridge.cpp`, `src/inference/ggml-wasm.ts`

The implementation steps depend on Task 2.1's choice. The TDD shape is the same regardless:

- [ ] **Step 1: Write a TS unit test that exercises 5 sequential `embed()` calls and asserts identical output**

Add to `tests/encoder-inference.test.ts` (create if absent — current encoder tests live in `tests/encoder-pool-normalize.test.ts`):

```typescript
// Stub-driven test: hook the encoder's wasm methods to count ctxCreate /
// ctxFree calls across N embed() invocations. Real WebGPU compute is
// skipped (Bun has no navigator.gpu).
import { describe, it, expect, jest } from "bun:test";
// ... fill in based on existing test patterns; the assertion is:
// expect(ctxCreateCalls).toBeLessThanOrEqual(2)
// after 5 sequential embed() calls (1 weight ctx + ≤1 graph ctx
// rebuild for shape change, none for same-N).
```

(The exact mock surface depends on Task 2.1's choice. If the choice is "Skip-ctxFree", the assertion is `ctxFreeCalls === 0` across 5 same-N calls. Adapt accordingly.)

- [ ] **Step 2: Run test, expect fail**

Run: `bun test tests/encoder-inference.test.ts -t "ctx reuse"`
Expected: fail.

- [ ] **Step 3: Implement the chosen option in `encoder-inference.ts`**

Concrete code depends on Task 2.1. Land it.

- [ ] **Step 4: Run test, expect pass**

Run: `bun test tests/encoder-inference.test.ts -t "ctx reuse"`
Expected: pass.

- [ ] **Step 5: Run `make checkall`**

Run: `make checkall`
Expected: all green.

- [ ] **Step 6: Build WASM if bridge changed**

If Task 2.1 modified `webgpu-bridge.cpp` or added bindings to `ggml-wasm.ts`:
```bash
source ~/emsdk/emsdk_env.sh && make wasm-build
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
cp src/wasm/build/webllm-wasm.{js,wasm} smoke-test/
make smoke-restart
```

If only `encoder-inference.ts` changed, just rebuild the bundle:
```bash
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
make smoke-restart
```

- [ ] **Step 7: Commit**

```bash
git add src/inference/encoder-inference.ts <other modified files>
git commit -m "feat(encoder): L1 ctx/graph reuse across embed() calls"
```

### Task 2.3: Re-measure with the harness

- [ ] **Step 1: Re-run the harness in non-profile mode**

Run:
```bash
make embed-perf EMBED_PERF_ARGS="--out eval/reports/embed-perf-2026-04-27-L1/"
```

- [ ] **Step 2: Compare against Phase 1 baseline**

Compute deltas manually or with a one-liner; produce a `delta.md` in the L1 dir summarizing:
- Per-(model, mode, fixture) p50 ms before vs after
- Δ ms and Δ%

- [ ] **Step 3: Verify [8/8] cosine still passes**

Run a smoke-bench cycle:
```bash
make smoke-bench PERF_MODEL=tinyllama-1.1b-chat-q4_0 PERF_RUNS=1 2>&1 | tee /tmp/L1-cosine.log
grep -E "cosine=" /tmp/L1-cosine.log
```
Expected: cosine within ±0.005 of `eval/reports/embed-perf-baseline-cosine.json`.

- [ ] **Step 4: Run `make bench-full --profiles arctic-embed-s` for the G3 task-accuracy half**

Run:
```bash
bun run eval/bench.ts --profiles arctic-embed-s
```
Expected: 8/8 tasks passing, overall ≥93%.

- [ ] **Step 5: Commit the measurements**

```bash
git add -f eval/reports/embed-perf-2026-04-27-L1/
git commit -m "chore(perf): §D Phase 2 (L1 ctx reuse) measurements"
```

### Task 2.4: G1 + G3 gate decision

- [ ] **Step 1: Apply the gate rules**

- G1: `≥10% drop in p50 wall ms on at least one model` AND `no model regressing >3%`.
- G3: cosine within ±0.005 AND bench-full 8/8 / ≥93%.

- [ ] **Step 2A — Both pass: ship**

If both pass: tag the Phase 2 tip and proceed to Phase 3.
```bash
git tag p2-L1-shipped-§D
```

- [ ] **Step 2B — G3 fails: revert**

If G3 fails (cosine drift or accuracy drop): `git revert <task-2.2 commit>`; document what broke; mark L1 closed with a writeup; proceed to Phase 3.

- [ ] **Step 2C — G1 fails: revert**

If G1 fails (no perf win, or regression): `git revert <task-2.2 commit>`; document; proceed to Phase 3.

- [ ] **Step 3: Update TODO.md scratch**

Append to a new "§D in flight" subsection of TODO.md "Active next steps" with the L1 decision and a one-paragraph writeup.

```bash
git add TODO.md
git commit -m "docs(TODO): §D Phase 2 (L1) decision: <ship|revert>"
```

---

## Phase 3 — Lever 2: GPU-side pool / readback shrink

### Task 3.1: Investigate GPU pool primitives

**Files:**
- Read-only: `src/inference/ggml-wasm.ts`, `src/wasm/webgpu-bridge.cpp`, `~/Repos/llama.cpp/ggml/include/ggml.h`

Goal: pick one of:
1. **opSumRows** — bind `ggml_sum_rows` (one new export in bridge + ggml-wasm). Cleanest if available in ggml-webgpu.
2. **Matmul-against-ones** — `[E, N] @ [N, 1] = [E, 1]`. Zero bridge changes; introduces one extra dispatch.
3. **opMean / opSum** — same as (1) but if those are exposed instead.
4. **Partial: CLS-only readback shrink** — only ship the trivial CLS path; punt mean to a later cycle.

- [ ] **Step 1: Check ggml.h for `ggml_sum_rows` / `ggml_mean`**

Run: `grep -n "ggml_sum_rows\|ggml_mean\|ggml_sum" ~/Repos/llama.cpp/ggml/include/ggml.h | head`

- [ ] **Step 2: Check ggml-webgpu's `supports_op` for SUM_ROWS / SUM / MEAN**

Run: `grep -n "GGML_OP_SUM_ROWS\|GGML_OP_SUM\|GGML_OP_MEAN" ~/Repos/llama.cpp/ggml/src/ggml-webgpu/ggml-webgpu.cpp | head`

- [ ] **Step 3: Check our existing bridge surface**

Run: `grep -n "opSum\|opMean" src/inference/ggml-wasm.ts src/wasm/webgpu-bridge.cpp`

- [ ] **Step 4: Document the choice**

Append a Phase 3 implementation choice section to the design spec mirroring Task 2.1 Step 3.

- [ ] **Step 5: Commit**

```bash
git add -f docs/superpowers/specs/2026-04-27-encoder-perf-pass-design.md
git commit -m "docs(spec): §D Phase 3 implementation choice"
```

### Task 3.2: Implement L2

**Files:**
- Modify: `src/inference/encoder-inference.ts`
- (If new op binding) Modify: `src/wasm/webgpu-bridge.cpp`, `src/inference/ggml-wasm.ts`

- [ ] **Step 1: Write a unit test that pool+normalize results are byte-equal pre/post-change**

Add to `tests/encoder-pool-normalize.test.ts` (the file exists — extend it):

```typescript
import { describe, it, expect } from "bun:test";
// Synthetic [E, N] hidden buffer; exercise both CLS and mean paths.
// Assert that the new GPU-pool implementation (mocked at the wasm
// boundary if needed) returns the same Float32Array as the existing
// CPU pool helper, within 1e-6 element-wise.
```

- [ ] **Step 2: Run test, expect fail**

Run: `bun test tests/encoder-pool-normalize.test.ts -t "GPU pool"`
Expected: fail.

- [ ] **Step 3: Implement the chosen option in `encoder-inference.ts::embed()`**

For CLS path:
```typescript
// After buildGraph returns finalHidden of shape [E, N]:
const colZeroIdx = wasm.tensorNew1d(GgmlType.I32, 1);
this.lastLeaves = { ...this.lastLeaves, colZeroIdx };
// upload [0] into colZeroIdx alongside other leaves
const pooled = wasm.opGetRows(finalHidden, colZeroIdx); // shape [E, 1]
// L2 normalize via opNorm or via post-readback CPU norm.
```

For mean path: implementation depends on Task 3.1's choice.

- [ ] **Step 4: Run test, expect pass**

Run: `bun test tests/encoder-pool-normalize.test.ts -t "GPU pool"`
Expected: pass.

- [ ] **Step 5: Run `make checkall`**

Run: `make checkall`
Expected: all green.

- [ ] **Step 6: Rebuild as needed**

Same as Phase 2 Task 2.2 Step 6.

- [ ] **Step 7: Commit**

```bash
git add src/inference/encoder-inference.ts <other>
git commit -m "feat(encoder): L2 GPU-side pool, shrink readback to [E]"
```

### Task 3.3: Re-measure + gate

Mirrors Task 2.3 + 2.4 verbatim (`embed-perf-2026-04-27-L2/`, gate-decision logic, TODO.md scratch). Tag `p3-L2-shipped-§D` if shipped, else revert.

---

## Phase 4 — Lever 3: `embedBatch` public API

### Task 4.1: Add `EncoderInference.embedBatch`

**Files:**
- Modify: `src/inference/encoder-inference.ts`
- Create: `tests/embed-batch.test.ts`

- [ ] **Step 1: Write the parity test**

```typescript
import { describe, it, expect } from "bun:test";
// Bun has no navigator.gpu; this test runs against a mocked GgmlWasm
// that records the order of embed calls. Assertions:
//   - embedBatch([]) returns []
//   - embedBatch([t]) returns [embed(t)] (same Float32Array bytes)
//   - embedBatch([a, b, c]) returns [embed(a), embed(b), embed(c)]
//   - embedBatch serializes (no overlapping forward() calls in mock)
```

- [ ] **Step 2: Run test, expect fail**

Run: `bun test tests/embed-batch.test.ts`
Expected: fail with "embedBatch is not a function".

- [ ] **Step 3: Implement `EncoderInference.embedBatch`**

Add to `EncoderInference`:

```typescript
async embedBatch(tokenIdsList: Int32Array[]): Promise<Float32Array[]> {
	if (!this.weights) throw new Error("weights not loaded");
	const results: Float32Array[] = [];
	for (const ids of tokenIdsList) {
		results.push(await this.embed(ids));
	}
	return results;
}
```

This is the **sequential** implementation per the design's Phase 4 scope. Concatenated-graph batched compute is non-goals.

- [ ] **Step 4: Run test, expect pass**

Run: `bun test tests/embed-batch.test.ts`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/inference/encoder-inference.ts tests/embed-batch.test.ts
git commit -m "feat(encoder): EncoderInference.embedBatch (sequential)"
```

### Task 4.2: Public `engine.embedBatch`

**Files:**
- Modify: `src/core/engine.ts`

- [ ] **Step 1: Add the method**

Locate `embed(modelId, text)` (line 429 of `engine.ts`). Add immediately after it:

```typescript
/**
 * Compute L2-normalized sentence embeddings for an array of texts using
 * a registered bidirectional-encoder model. Sequential per-text compute;
 * use this when you have multiple texts to embed and want lower
 * end-to-end latency than calling `embed()` in a loop (the loop pays
 * lock-acquisition + graph-rebuild overhead per call; this method
 * pays it once).
 */
async embedBatch(modelId: string, texts: string[]): Promise<Float32Array[]> {
	const entry = this.models.get(modelId);
	if (!entry) throw new Error(`Model "${modelId}" not loaded`);
	const enc = this.encoderEngines.get(modelId);
	if (!enc) {
		throw new Error(
			`embedBatch() requires a bidirectional encoder model; "${modelId}" is architecture "${entry.hyperparams.architecture}"`,
		);
	}
	const idsList = texts.map((t) => new Int32Array(entry.tokenizer.encode(t)));
	return enc.embedBatch(idsList);
}
```

- [ ] **Step 2: Add a test for `engine.embedBatch`**

Extend `tests/engine-streaming-api.test.ts` (or similar engine-surface test) with a small mock that asserts `engine.embedBatch` resolves to N `Float32Array`s for an N-element input.

- [ ] **Step 3: Run `make checkall`**

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/core/engine.ts tests/<engine test>
git commit -m "feat(engine): public embedBatch(modelId, texts) API"
```

### Task 4.3: Wire embedBatch into the harness for true G2 measurement

**Files:**
- Modify: `smoke-test/real-model-page.js` (encoder branch from Task 1.4)

- [ ] **Step 1: Replace the sequential-loop in the encoder branch's batch mode with `embedBatch`**

In the `embedPerfMode === "batch"` branch in the encoder section, replace:

```javascript
for (const txt of texts) {
	await smokeEngine.embed(smokeEngineHandleId, txt);
}
```

with:

```javascript
await smokeEngine.embedBatch(smokeEngineHandleId, texts);
```

inside the `t0 / t1` measurement scope. Leave the warmup pass as a sequential loop (or convert to an embedBatch warmup — same effect).

- [ ] **Step 2: Rebuild bundle**

```bash
bun build src/index.ts --outfile smoke-test/webllm-bundle.js --target browser
make smoke-restart
```

- [ ] **Step 3: Re-run harness in batch mode**

```bash
make embed-perf EMBED_PERF_ARGS="--mode batch --out eval/reports/embed-perf-2026-04-27-L3/"
```

- [ ] **Step 4: Compare against the Phase-1-or-L2 batch baseline**

Compute texts/sec ratio. G2 requires `texts/sec_after / texts/sec_before ≥ 1.5×` on at least one model.

- [ ] **Step 5: Commit**

```bash
git add smoke-test/real-model-page.js eval/reports/embed-perf-2026-04-27-L3/
git commit -m "feat(smoke): route batch-mode embedPerf through embedBatch"
```

### Task 4.4: G2 + G3 gate

Mirrors Task 2.4. Tag `p4-L3-shipped-§D` or revert.

---

## Phase 5 — Lever 4 / opportunistic

### Task 5.1: Re-rank levers from latest profile data

- [ ] **Step 1: Re-run profile-mode with shipped levers in place**

```bash
make embed-perf EMBED_PERF_ARGS="--profile --out eval/reports/embed-perf-2026-04-27-postL3/"
```

- [ ] **Step 2: Identify the dominant remaining bucket**

If `graphComputeMs` is now ≥80% of step total → matmul/encode kernel work (mirrors §10/§17 — likely off-table for encoder unless something specific surfaces).

If `downloadResultMs` is still ≥10% → revisit L2 with a tighter pool (e.g. ship the partial CLS-only fallback but mean still has work).

If `build+alloc+upload` is ≥10% → leaf upload optimization (smaller dispatches, batched leaf writes, etc.).

- [ ] **Step 3: Decide ship-or-skip**

If a clear lever exists with predictable G1 win → write a focused implementation task (mirroring 2.2 / 3.2) and run it.

If nothing dominates and remaining buckets are within noise → close Phase 5 with a "no shippable lever found" writeup.

### Task 5.2 — Optional implementation block

Only run if Task 5.1 picked a lever. Same TDD shape as 2.2 / 3.2.

- [ ] **Step 1: Write failing test**
- [ ] **Step 2: Implement**
- [ ] **Step 3: Re-measure**
- [ ] **Step 4: Gate**
- [ ] **Step 5: Commit**

---

## Phase 6 — Closure writeup

### Task 6.1: Update TODO.md with the §<N> closure entry

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Compute the next § number**

Run: `grep -nE "^[0-9]+\. \*\*§[0-9]+|^### Completed on 2026-04-2" TODO.md | tail -5`
Find the highest existing § in the "Completed on 2026-04-26" / "2026-04-27" sections; add 1.

- [ ] **Step 2: Add the closure entry**

Append to the most recent `### Completed on YYYY-MM-DD` section a numbered entry with:
- Lever inventory (L1/L2/L3/L4) and per-lever ship-or-revert decision.
- Final perf table: pre vs post per shipped lever, per-(model, fixture, mode).
- G3 cosine and accuracy values (final).
- Raw logs path: `eval/reports/embed-perf-2026-04-27-{baseline,L1,L2,L3,postL3}/`.
- "Net characterization" paragraph mirroring §17 / §18 / §20 closures.

- [ ] **Step 3: Update Resumption checklist**

Edit the `### Resumption checklist` near line 2257 of TODO.md to note that §D is closed and what's next.

- [ ] **Step 4: Commit**

```bash
git add TODO.md
git commit -m "docs(TODO): §D encoder perf cycle CLOSED"
```

### Task 6.2: Merge feature branch to main

- [ ] **Step 1: Verify clean tree and all tests pass**

```bash
git status
make checkall
```

- [ ] **Step 2: Fast-forward merge to main**

```bash
git checkout main
git merge --ff-only feat/encoder-perf
```
Expected: fast-forward succeeds (no rebase needed since we branched off main and didn't merge from main).

If main has moved during the cycle: `git rebase main` on `feat/encoder-perf` first, re-run `make checkall`, then merge.

- [ ] **Step 3: Push (if remote tracking exists; otherwise skip)**

```bash
git push origin main
```

- [ ] **Step 4: Confirm with user before deleting branch**

Tell user: "§D shipped to main at <sha>. The feature branch `feat/encoder-perf` can be deleted with `git branch -d feat/encoder-perf` once you're ready."

(Branch deletion is destructive — leave for explicit user direction per global instructions.)

---

## Self-Review Notes

Spec coverage check:
- ✅ Phase 1 harness + baseline → Tasks 1.1–1.7
- ✅ Phase 2 ctx reuse (L1) → Tasks 2.1–2.4 with investigation gate
- ✅ Phase 3 GPU pool (L2) → Tasks 3.1–3.3 with investigation gate
- ✅ Phase 4 embedBatch (L3) → Tasks 4.1–4.4
- ✅ Phase 5 opportunistic (L4) → Tasks 5.1–5.2 (optional)
- ✅ Phase 6 closure → Tasks 6.1–6.2
- ✅ G1 single-text latency gate → 2.4, 3.3, 5.x
- ✅ G2 batch throughput gate → 4.4
- ✅ G3 cosine + accuracy gate → 1.7, 2.3 step 3-4, 3.3, 4.4, 5.x

Type/method consistency:
- `EncoderInference.embed(tokenIds: Int32Array)` (existing).
- `EncoderInference.embedBatch(tokenIdsList: Int32Array[]): Promise<Float32Array[]>` (Task 4.1).
- `engine.embed(modelId, text)` (existing).
- `engine.embedBatch(modelId, texts)` (Task 4.2).
- `EmbedPerfTrace` + `waitForEmbedPerfResult` exported from `eval/browser-smoke.ts` (Task 1.3).
- `window.__embedTraces` populated by smoke-page hooks (Tasks 1.2, 1.4, 4.3).
- `EMBED_PROMPTS` from `eval/fixtures/embed-prompts.ts` (Task 1.1) — note the smoke page inlines the same text values rather than importing this Bun-only file.

Per-task placeholder check: every step has either concrete code, concrete commands with expected outputs, or an explicit investigation step that produces a design-note artifact before code lands. Investigation steps in Phases 2 and 3 are honest about the "decide at impl time" nature flagged in the spec.
