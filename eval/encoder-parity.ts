#!/usr/bin/env bun
/**
 * Encoder parity harness. Drives the smoke-test page via agentchrome,
 * calls window.engine.embed(handleId, text) for each fixture input,
 * compares cosine similarity vs sentence-transformers reference.
 * Pass gate: cosine >= 0.999 on every row.
 *
 * Run:
 *   bun eval/encoder-parity.ts <modelId> <ref-file> [--worker]
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

const argv = process.argv.slice(2);
let useWorker = false;
const positional: string[] = [];
for (const a of argv) {
	if (a === "--worker") {
		useWorker = true;
	} else {
		positional.push(a);
	}
}
const [modelId, refPath] = positional;
if (!modelId || !refPath) {
	console.error(
		"Usage: bun eval/encoder-parity.ts <modelId> <ref-file> [--worker]",
	);
	process.exit(2);
}

interface Ref {
	input: string;
	vec: number[];
}

const refs: Ref[] = JSON.parse(readFileSync(refPath, "utf8")) as Ref[];
const inputs = refs.map((r) => r.input);

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
	// Numbers / booleans round-trip through agentchrome as native types.
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
	// Reuse the embedPerf=single path to drive model load. The smoke page
	// exposes window.engine + window.handleId after engine construction
	// (real-model-page.js [6/8]); the parity harness then drives
	// engine.embed via agentchrome js exec — no smoke-page changes needed
	// beyond the global exposure.
	extraParams: {
		embedPerf: "single",
		embedFixture: "short",
		embedReps: "1",
		v: `${Date.now()}`,
		...(useWorker ? { worker: 1 } : {}),
	},
});
console.log(`Mode: ${useWorker ? "worker" : "main"}`);
console.log(`Navigating to ${url}`);
agentchrome(port, tab, ["navigate", url]);

// Wait for model load: the page assigns window.handleId/window.engine
// immediately after `adoptPreloadedModel` resolves (real-model-page.js
// step [6/8], ~line 501). Poll up to 180s for slower jina downloads.
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

// Diagnostic: dump alibiMaxBias the engine actually sees, if present.
const alibiProbe = jsExec(
	port,
	tab,
	`(() => {
		try {
			const eng = window.engine;
			const hid = window.handleId;
			if (!eng || !hid) return JSON.stringify({ ok: false, reason: "no engine/handleId" });
			const handles = eng.modelHandles ?? eng._modelHandles ?? eng.handles;
			let hp;
			if (handles && handles[hid]) {
				hp = handles[hid].hyperparams ?? handles[hid].parsed?.hyperparams;
			}
			if (!hp && eng.parsed) hp = eng.parsed.hyperparams;
			return JSON.stringify({
				ok: true,
				alibiMaxBias: hp?.alibiMaxBias,
				poolingType: hp?.poolingType,
				architecture: hp?.architecture,
			});
		} catch (e) {
			return JSON.stringify({ ok: false, reason: String(e) });
		}
	})()`,
);
console.log(`hyperparams probe: ${alibiProbe ?? "(unavailable)"}`);

let pass = 0;
const rows: { idx: number; input: string; cos: number; ok: boolean }[] = [];
for (let i = 0; i < inputs.length; i++) {
	const inputJson = JSON.stringify(inputs[i]);
	const script = `(async () => {
		const text = ${inputJson};
		const v = await window.engine.embed(window.handleId, text);
		return JSON.stringify(Array.from(v));
	})()`;
	const raw = jsExec(port, tab, script);
	if (typeof raw !== "string") {
		throw new Error(`row ${i}: empty/invalid response from agentchrome`);
	}
	const vec = JSON.parse(raw) as number[];
	if (vec.length !== refs[i].vec.length) {
		throw new Error(
			`dim mismatch row ${i}: got ${vec.length}, ref ${refs[i].vec.length}`,
		);
	}
	const cos = cosine(vec, refs[i].vec);
	const ok = cos >= COSINE_GATE;
	if (ok) pass++;
	rows.push({ idx: i, input: inputs[i], cos, ok });
	console.log(
		`  row ${i}  cos=${cos.toFixed(6)}  ${ok ? "PASS" : "FAIL"}  ${
			JSON.stringify(inputs[i]).slice(0, 60)
		}`,
	);
}

console.log(`\n${pass}/${inputs.length} rows passed (gate >= ${COSINE_GATE})`);
process.exit(pass === inputs.length ? 0 : 1);
