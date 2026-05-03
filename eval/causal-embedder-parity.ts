#!/usr/bin/env bun
/**
 * Bucket C/D parity harness. Drives the smoke page via agentchrome, calls
 * window.engine.embed(handleId, text) for each fixture × mode, compares
 * cosine similarity vs sentence-transformers / transformers reference vectors.
 *
 * Pass gate: cosine >= COSINE_GATE on every row, |v|_2 == 1.0 ± 1e-3.
 *
 * Gate selection: 0.999 for f16/full-precision GGUFs whose forward path
 * matches the reference's numeric precision (sentence-transformers is
 * fp32 PyTorch). 0.995 for hybrid-quant GGUFs (`defaultQuant === "hyb"`,
 * Q4_K on `token_embd` + f16 elsewhere — see CLAUDE.md "Per-binding
 * 128 MiB cap doctrine"); the Q4_K row-lookup error doesn't compound but
 * does shift the last-token state by ~1e-3 cosine on multi-token inputs.
 * Override with `--gate <value>`.
 *
 * Bucket D (embeddingCapable chat models): the model's `embeddingCapable`
 * flag triggers `?embeddingCapable=1` in the smoke URL, which routes
 * `engine.embed()` through the hidden-state tap-point path
 * (ModelInference.embed). Ref bundles for chat models have no
 * `instruction_prefix` and only document-mode fixtures.
 *
 * Usage:
 *   bun eval/causal-embedder-parity.ts <modelId> <ref-file> [--gate 0.999] [--worker]
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

const COSINE_GATE_FULL = 0.999;
const COSINE_GATE_HYBRID = 0.995;
// Q4_K_M: more aggressive than hybrid. Phi-3.5-mini empirical range:
// 0.922-0.970 across 10 fixtures vs f16 PyTorch reference.
// Gate = min_observed - 0.01 safety margin (0.922 - 0.01 = 0.912, rounded to 0.91).
const COSINE_GATE_Q4KM = 0.91;
// IQ3_M at 8B+ params: per-layer quant error accumulates across 36+ layers;
// empirical range against f16 HF reference is 0.90-0.96. Gate set at 0.90.
const COSINE_GATE_IQ3M = 0.90;
const MAGNITUDE_TOLERANCE = 1e-3;

const args = process.argv.slice(2);
let gateOverride: number | null = null;
let useWorker = false;
const positional: string[] = [];
for (let i = 0; i < args.length; i++) {
	if (args[i] === "--gate" && i + 1 < args.length) {
		const parsed = Number.parseFloat(args[i + 1]);
		if (!Number.isFinite(parsed) || parsed <= 0 || parsed > 1) {
			console.error(`Invalid --gate value "${args[i + 1]}"`);
			process.exit(2);
		}
		gateOverride = parsed;
		i++;
	} else if (args[i] === "--worker") {
		useWorker = true;
	} else {
		positional.push(args[i]);
	}
}
const [modelId, refPath] = positional;
if (!modelId || !refPath) {
	console.error(
		"Usage: bun eval/causal-embedder-parity.ts <modelId> <ref-file> [--gate 0.999] [--worker]",
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
	// Bucket C (dedicated embedders): instruction prefix string for query mode.
	// Bucket D (embeddingCapable chat models): absent / null — no prefix, doc-only.
	instruction_prefix?: string | null;
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
 *
 * Bucket C 1024-dim embeddings serialized to JSON exceed agentchrome's
 * ~16 KB inline-response cap, so the response is offloaded to a temp
 * file at `output_file` and stdout becomes `{output_file, summary, ...}`.
 * Read the temp file in that case; the JSON inside has the same envelope
 * shape as the inline path.
 */
function jsExec(port: string, tab: string, script: string): string | undefined {
	const out = agentchrome(port, tab, ["js", "exec", script]);
	const resp = JSON.parse(out) as {
		result?: unknown;
		type?: string;
		output_file?: string;
	};
	if (typeof resp.output_file === "string") {
		const inner = JSON.parse(readFileSync(resp.output_file, "utf8")) as {
			result?: unknown;
		};
		if (typeof inner.result === "string") return inner.result;
		if (inner.result === undefined || inner.result === null) return undefined;
		return String(inner.result);
	}
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

const COSINE_GATE =
	gateOverride ??
	(model.defaultQuant === "hyb"
		? COSINE_GATE_HYBRID
		: model.defaultQuant === "q4km"
			? COSINE_GATE_Q4KM
			: model.defaultQuant === "iq3m"
				? COSINE_GATE_IQ3M
				: COSINE_GATE_FULL);
console.log(
	`Gate: cos >= ${COSINE_GATE}${
		gateOverride !== null
			? " (override)"
			: model.defaultQuant === "hyb"
				? " (hybrid Q4_K on token_embd vs full-precision reference)"
				: model.defaultQuant === "q4km"
					? " (Q4_K_M: quant noise between hybrid and IQ3_M vs f16 reference)"
					: model.defaultQuant === "iq3m"
						? " (IQ3_M i-quant: quant noise accumulates across layers vs f16 reference)"
						: ""
	}`,
);
const activePooling = model.embeddingPooling ?? "last-token";
console.log(
	`Pooling: ${activePooling}${model.embeddingPooling ? " (registration override)" : " (default)"}`,
);

await ensureSmokeServerReachable();
const { port, tab } = await resolveAgentchromeSession();
await ensureModelDownloaded(model);

const url = buildSmokeTestUrl(modelId, model.contextLength, {
	// Do NOT enable `embedPerf` here. The smoke page sets window.engine +
	// window.handleId in [6/8] (engine construction), then later [8/8]
	// runs `runEmbedPerfHook` which fires several warmup/measured
	// `engine.embed()` calls. The harness polls only for the handleId and
	// would race those in-page calls — concurrent forward graphs corrupt
	// the shared WASM ctx stack and the next download asserts on
	// `tensor->buffer != nullptr`. Skipping embedPerf leaves [8/8]'s hook
	// as a no-op so the harness has exclusive control of engine.embed.
	extraParams: {
		v: `${Date.now()}`,
		ingest: "off",
		// Bucket D: route engine.embed() through the hidden-state tap-point path
		// (ModelInference.embed) for embeddingCapable chat models.
		...(model.embeddingCapable ? { embeddingCapable: "1" } : {}),
		// Bucket D pooling mode: forward the registration's pooling choice
		// to the smoke page so engine.embed dispatches with the right pool
		// (last-token by default; mean for high-anisotropy chat models).
		...(model.embeddingPooling === "mean"
			? { embeddingPooling: "mean" }
			: {}),
		...(useWorker ? { worker: "1" } : {}),
	},
});
console.log(`Mode: ${useWorker ? "worker" : "main"}`);
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
	// instruction_prefix is absent for bucket D (chat model, doc-only fixtures).
	const prefix = refs.instruction_prefix ?? "";
	const text = fx.mode === "query" && prefix ? prefix + fx.input : fx.input;
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

if (pass < refs.fixtures.length) {
	process.exit(1);
}

// ---------------------------------------------------------------------------
// 16+16 cosine-distinguishability sanity check.
// Runs after the primary parity gate passes. Catches the "tap-point picked
// the wrong layer" failure mode — a symmetrically wrong tap-point would give
// vectors that pass parity (if the bug is the same in PyTorch and WASM) but
// would produce semantically random embeddings that can't distinguish
// paraphrases from unrelated text.
//
// 16 paraphrase pairs span 16 domains (technology, sports, food, weather,
// finance, history, biology, music, travel, education, medicine, law, art,
// geography, politics, household). The 16 unrelated pairs are constructed by
// taking the first sentence of paraphrase pair i and pairing it with the
// first sentence of paraphrase pair (i + 8) mod 16 — same sentence pool, so
// the harness only embeds 32 unique sentences; offset of 8 maximizes
// cross-domain semantic distance.
//
// Pass criterion: relaxed margin — `mean(P) - mean(U) >= MARGIN_GATE`. The
// strict criterion (`min(P) > max(U)`) is kept as informational output but
// is *not* gating: even on the qwen3-8b-iq3m bucket D flagship the strict
// criterion fails on this 16+16 set due to vocabulary-overlap edge cases
// (eg "photosynthesis converts sunlight..." paraphrase scoring 0.85 while
// "photosynthesis... vs ...renewable energy subsidies" unrelated scores
// 0.74 — both mention natural systems). Mean-margin cleanly separates
// "useful but imperfect" (qwen3 +0.078 margin) from "random / negative"
// (phi-3.5-mini-q4km −0.006 to −0.027 margin under both pooling modes,
// which led to its bucket D demotion 2026-04-30).
// ---------------------------------------------------------------------------
const DISTINGUISHABILITY_MARGIN_GATE = 0.05;
const PARAPHRASE_PAIRS: ReadonlyArray<readonly [string, string]> = [
	// 0 — technology
	[
		"Compile-time type checking catches a wide class of programmer errors.",
		"Static type analysis prevents many bugs at build time.",
	],
	// 1 — sports
	[
		"The marathon runner crossed the finish line in just over two hours.",
		"After running for more than two hours, the athlete completed the long-distance race.",
	],
	// 2 — food
	[
		"She slowly stirred the simmering tomato sauce on the stove.",
		"On the burner, she gently mixed the bubbling tomato sauce.",
	],
	// 3 — weather
	[
		"A heavy thunderstorm rolled across the valley last night.",
		"Last night the valley was hit by a powerful electrical storm.",
	],
	// 4 — finance
	[
		"Stock prices fell sharply after the merger announcement.",
		"Share values dropped steeply once the merger was made public.",
	],
	// 5 — history
	[
		"The Roman Empire collapsed under pressure from invading tribes.",
		"Under sustained barbarian incursions, ancient Rome eventually fell.",
	],
	// 6 — biology
	[
		"Photosynthesis converts sunlight into chemical energy stored in glucose.",
		"Plants turn light from the sun into glucose-bound chemical energy.",
	],
	// 7 — music
	[
		"The orchestra performed Beethoven's ninth symphony to a packed hall.",
		"To a full audience, the ensemble played Beethoven's 9th symphony.",
	],
	// 8 — travel
	[
		"The flight from Tokyo to San Francisco takes roughly ten hours.",
		"It takes about ten hours to fly between Tokyo and San Francisco.",
	],
	// 9 — education
	[
		"The professor assigned a long reading list for the seminar.",
		"For the seminar, the instructor handed out an extensive reading list.",
	],
	// 10 — medicine
	[
		"The vaccine triggers an immune response without causing the disease.",
		"By prompting immunity without infection, the vaccine protects patients.",
	],
	// 11 — law
	[
		"The court ruled the contract unenforceable due to fraud.",
		"Because of fraudulent conduct, the judge struck down the agreement.",
	],
	// 12 — art
	[
		"The painter applied thick layers of oil paint to the canvas.",
		"Across the canvas, the artist built up dense strokes of oil paint.",
	],
	// 13 — geography
	[
		"The Amazon basin holds the largest rainforest on Earth.",
		"Earth's biggest tropical forest sits inside the Amazon river basin.",
	],
	// 14 — politics
	[
		"The senator introduced a bill to expand renewable energy subsidies.",
		"A new bill proposing larger subsidies for renewables was filed by the senator.",
	],
	// 15 — household
	[
		"He swept the kitchen floor before mopping the tiles.",
		"After sweeping, he mopped the tiles in the kitchen.",
	],
];

// Build 16 unrelated pairs by offsetting first-sentence indices by 8 so each
// row pairs cross-domain content (technology↔travel, sports↔education, etc.).
const UNRELATED_PAIRS: ReadonlyArray<readonly [string, string]> =
	PARAPHRASE_PAIRS.map(
		(_pair, i) =>
			[
				PARAPHRASE_PAIRS[i][0],
				PARAPHRASE_PAIRS[(i + 8) % PARAPHRASE_PAIRS.length][0],
			] as readonly [string, string],
	);

// Collect unique sentences and embed them.
const allSentences = [
	...new Set([
		...PARAPHRASE_PAIRS.flat(),
		...UNRELATED_PAIRS.flat(),
	]),
];
console.log(
	`\n4-pair distinguishability check: embedding ${allSentences.length} sentences...`,
);

async function embedSentence(text: string): Promise<number[]> {
	const inputJson = JSON.stringify(text);
	const script = `(async () => {
		const v = await window.engine.embed(window.handleId, ${inputJson});
		return JSON.stringify(Array.from(v));
	})()`;
	const raw = jsExec(port, tab, script);
	if (typeof raw !== "string") {
		throw new Error(`distinguishability: empty response for text ${JSON.stringify(text)}`);
	}
	return JSON.parse(raw) as number[];
}

const sentenceVecs = new Map<string, number[]>();
for (const s of allSentences) {
	sentenceVecs.set(s, await embedSentence(s));
}

function cosinePair(a: number[], b: number[]): number {
	return cosine(a, b);
}

const paraphraseCosines = PARAPHRASE_PAIRS.map(([a, b]) =>
	cosinePair(sentenceVecs.get(a)!, sentenceVecs.get(b)!),
);
const unrelatedCosines = UNRELATED_PAIRS.map(([a, b]) =>
	cosinePair(sentenceVecs.get(a)!, sentenceVecs.get(b)!),
);

console.log("\nParaphrase pairs (expect high cosine):");
for (let i = 0; i < PARAPHRASE_PAIRS.length; i++) {
	const [a, b] = PARAPHRASE_PAIRS[i];
	console.log(
		`  pair ${i} cos=${paraphraseCosines[i].toFixed(6)}  "${a.slice(0, 40)}..." vs "${b.slice(0, 40)}..."`,
	);
}
console.log("\nUnrelated pairs (expect low cosine):");
for (let i = 0; i < UNRELATED_PAIRS.length; i++) {
	const [a, b] = UNRELATED_PAIRS[i];
	console.log(
		`  pair ${i} cos=${unrelatedCosines[i].toFixed(6)}  "${a.slice(0, 40)}..." vs "${b.slice(0, 40)}..."`,
	);
}

const minParaphrase = Math.min(...paraphraseCosines);
const maxUnrelated = Math.max(...unrelatedCosines);
const meanParaphrase =
	paraphraseCosines.reduce((s, x) => s + x, 0) / paraphraseCosines.length;
const meanUnrelated =
	unrelatedCosines.reduce((s, x) => s + x, 0) / unrelatedCosines.length;
const meanMargin = meanParaphrase - meanUnrelated;
const strictPass = minParaphrase > maxUnrelated;
const distinguishabilityPass = meanMargin >= DISTINGUISHABILITY_MARGIN_GATE;

console.log(
	`\nDistinguishability (informational, strict): min_paraphrase=${minParaphrase.toFixed(6)} max_unrelated=${maxUnrelated.toFixed(6)} ${strictPass ? "PASS" : "FAIL"}`,
);
console.log(
	`Distinguishability (gating, mean-margin): mean_paraphrase=${meanParaphrase.toFixed(6)} mean_unrelated=${meanUnrelated.toFixed(6)} margin=${meanMargin >= 0 ? "+" : ""}${meanMargin.toFixed(6)} (gate >= ${DISTINGUISHABILITY_MARGIN_GATE.toFixed(2)}) ${distinguishabilityPass ? "PASS" : "FAIL"}`,
);

if (!distinguishabilityPass) {
	console.error(
		`FAIL: distinguishability check failed — mean(P) − mean(U) must be >= ${DISTINGUISHABILITY_MARGIN_GATE.toFixed(2)}.`,
	);
	process.exit(1);
}

process.exit(0);
