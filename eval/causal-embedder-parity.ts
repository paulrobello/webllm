#!/usr/bin/env bun
/**
 * Bucket C parity harness. Drives the smoke page via agentchrome, calls
 * window.engine.embed(handleId, text) for each fixture × mode, compares
 * cosine similarity vs sentence-transformers reference vectors.
 *
 * Pass gate: cosine >= 0.999 on every row, |v|_2 == 1.0 ± 1e-3.
 *
 * Usage:
 *   bun eval/causal-embedder-parity.ts <modelId> <ref-file>
 *
 * Diagnostic ladder (if a row fails):
 *   1. All 10 rows fail uniformly (<0.5)         -> Signature C: prefix not applied
 *      OR tokenizer mismatch.
 *   2. Doc rows pass, query rows fail            -> Prefix LF/colon byte sequence wrong.
 *   3. All 10 rows land 0.95-0.99                -> Signature B: tap-point or norm mismatch.
 *   4. Length-monotonic degradation              -> Signature A: RoPE mode/freq_base wrong.
 *   5. Magnitude failure (|v|_2 != 1.0)          -> L2-normalize missing in embed().
 *   6. Mode-only mismatch (one mode passes)      -> investigate; shouldn't be possible.
 *
 * Requires:
 *   - smoke server up (`make smoke-serve`)
 *   - running agentchrome session (auto-launched if absent)
 */
import { readFileSync } from "node:fs";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "./browser-smoke.js";
import { getModelById } from "./models.js";

const COSINE_GATE = 0.999;
const MAGNITUDE_TOLERANCE = 1e-3;

const [, , modelId, refPath] = process.argv;
if (!modelId || !refPath) {
	console.error(
		"Usage: bun eval/causal-embedder-parity.ts <modelId> <ref-file>",
	);
	process.exit(2);
}

interface Fixture {
	row: number;
	input: string;
	mode: "document" | "query";
	vec: number[];
}

interface RefBundle {
	model: string;
	captured_with: string;
	pooling: string;
	instruction_prefix: string;
	fixtures: Fixture[];
}

const refs = JSON.parse(readFileSync(refPath, "utf8")) as RefBundle;
console.log(`Reference bundle: ${refs.model} (${refs.captured_with})`);
console.log(`Pooling: ${refs.pooling}`);
console.log(
	`Instruction prefix bytes: ${JSON.stringify(refs.instruction_prefix)}`,
);
console.log(`Fixtures: ${refs.fixtures.length}`);

function cosine(a: number[], b: number[]): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function magnitude(v: number[]): number {
	let s = 0;
	for (const x of v) s += x * x;
	return Math.sqrt(s);
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Run a JS expression in the page and return the parsed `result` field
 * from agentchrome's `js exec` JSON envelope. Caller is responsible for
 * JSON-stringifying complex return values inside the script.
 */
function jsExec(port: string, tab: string, script: string): string | undefined {
	const out = agentchrome(port, tab, ["js", "exec", script]);
	const resp = JSON.parse(out) as { result?: unknown; type?: string };
	if (typeof resp.result === "string") return resp.result;
	if (resp.result === undefined || resp.result === null) return undefined;
	return String(resp.result);
}

const model = getModelById(modelId);
if (!model) {
	throw new Error(
		`Unknown modelId "${modelId}" — register it in eval/models.ts first.`,
	);
}

await ensureSmokeServerReachable();
const { port, tab } = await resolveAgentchromeSession();
await ensureModelDownloaded(model);

const url = buildSmokeTestUrl(modelId, model.contextLength, {
	// Reuse the embedPerf=single path to drive engine load. The smoke page
	// exposes window.engine + window.handleId after engine construction;
	// the parity harness then drives engine.embed via agentchrome js exec.
	extraParams: {
		embedPerf: "single",
		embedFixture: "short",
		embedReps: "1",
		v: `${Date.now()}`,
	},
});
console.log(`Navigating to ${url}`);
agentchrome(port, tab, ["navigate", url]);

// Wait for the page to set window.engine + window.handleId.
const loadDeadline = Date.now() + 180_000;
let loaded = false;
while (Date.now() < loadDeadline) {
	const ready = jsExec(
		port,
		tab,
		`(() => Boolean(window.handleId && window.engine))()`,
	);
	if (ready === "true") {
		loaded = true;
		break;
	}
	await sleep(500);
}
if (!loaded) {
	throw new Error(`Timed out waiting for window.handleId on ${modelId}`);
}
console.log(`Loaded ${modelId}`);

let pass = 0;
const rows: {
	row: number;
	mode: string;
	cos: number;
	mag: number;
	ok: boolean;
}[] = [];

for (const fx of refs.fixtures) {
	const text =
		fx.mode === "query" ? refs.instruction_prefix + fx.input : fx.input;
	const inputJson = JSON.stringify(text);
	const script = `(async () => {
		const v = await window.engine.embed(window.handleId, ${inputJson});
		return JSON.stringify(Array.from(v));
	})()`;
	const raw = jsExec(port, tab, script);
	if (typeof raw !== "string") {
		throw new Error(`row ${fx.row} mode=${fx.mode}: empty response`);
	}
	const vec = JSON.parse(raw) as number[];
	if (vec.length !== fx.vec.length) {
		throw new Error(
			`dim mismatch row ${fx.row} mode=${fx.mode}: got ${vec.length}, ref ${fx.vec.length}`,
		);
	}
	const mag = magnitude(vec);
	const magOk = Math.abs(mag - 1.0) <= MAGNITUDE_TOLERANCE;
	const cos = cosine(vec, fx.vec);
	const ok = cos >= COSINE_GATE && magOk;
	if (ok) pass++;
	rows.push({ row: fx.row, mode: fx.mode, cos, mag, ok });
	console.log(
		`  row ${fx.row} ${fx.mode.padEnd(8)}  cos=${cos.toFixed(6)}  mag=${mag.toFixed(6)}  ${ok ? "PASS" : "FAIL"}`,
	);
}

console.log(
	`\n${pass}/${refs.fixtures.length} rows passed (gate cos >= ${COSINE_GATE}, mag |v|_2 == 1.0 +/- ${MAGNITUDE_TOLERANCE})`,
);
process.exit(pass === refs.fixtures.length ? 0 : 1);
