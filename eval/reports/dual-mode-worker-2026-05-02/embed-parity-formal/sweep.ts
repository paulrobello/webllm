#!/usr/bin/env bun
/**
 * Worker-vs-main embedder cosine parity sweep.
 *
 * For each target embedder, navigate the smoke page first in main mode,
 * embed each fixture, capture the vectors. Then navigate again in worker
 * mode (`?worker=1`), embed the same fixtures, capture those vectors.
 * Finally compute cosine similarity between corresponding pairs.
 *
 * Architectural expectation: cos == 1.0 exactly. Same code path on both
 * sides of the worker boundary, same WebGPU device, same upload, same
 * weights — there is no source of divergence.
 *
 * Targets (from `eval/reports/dual-mode-worker-2026-05-02/SUMMARY.md`
 * follow-up #6):
 *   - snowflake-arctic-embed-m-q0f32-b4 (encoder; gate >= 0.999)
 *   - qwen3-embedding-0.6b-hyb         (causal-LM embedder; gate >= 0.995)
 *   - qwen3-8b-iq3m                    (bucket D self-embed; gate >= 0.90)
 *
 * Usage:
 *   bun run eval/reports/dual-mode-worker-2026-05-02/embed-parity-formal/sweep.ts
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "../../../browser-smoke.js";
import { getModelById } from "../../../models.js";

const HERE = dirname(fileURLToPath(import.meta.url));

interface Target {
	modelId: string;
	gate: number;
	fixtures: string[];
	// True when bucket D — sets ?embeddingCapable=1.
	// (Read from registration; included here for clarity.)
}

const FIXTURES_GENERIC: string[] = [
	"Hello world.",
	"The quick brown fox jumps over the lazy dog.",
	"Embedding models map text into a dense vector space where semantic similarity corresponds to cosine distance, enabling efficient retrieval over large corpora.",
	"Café — naïve façade résumé piñata coöperate. 你好世界. Здравствуй мир.",
	".",
];

const FIXTURES_DOC: string[] = [
	"The cat sat on the mat.",
	"Quantum entanglement enables instantaneous correlation.",
	"Open the door and let the cool breeze in.",
	"Stock prices fell sharply after the merger announcement.",
	"Rendering ten thousand triangles per frame is now routine on integrated GPUs.",
];

const TARGETS: Target[] = [
	{
		modelId: "snowflake-arctic-embed-m-q0f32-b4",
		gate: 0.999,
		fixtures: FIXTURES_GENERIC,
	},
	{
		modelId: "qwen3-embedding-0.6b-hyb",
		gate: 0.995,
		fixtures: FIXTURES_DOC,
	},
	{
		modelId: "qwen3-8b-iq3m",
		gate: 0.9,
		fixtures: FIXTURES_DOC,
	},
];

function cosine(a: number[], b: number[]): number {
	let dot = 0;
	let na = 0;
	let nb = 0;
	const n = Math.min(a.length, b.length);
	for (let i = 0; i < n; i++) {
		dot += a[i] * b[i];
		na += a[i] * a[i];
		nb += b[i] * b[i];
	}
	return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

interface JsExecResp {
	result?: unknown;
	type?: string;
	output_file?: string;
}

function jsExec(port: string, tab: string, script: string): string | undefined {
	const out = agentchrome(port, tab, ["js", "exec", script]);
	const resp = JSON.parse(out) as JsExecResp;
	if (typeof resp.output_file === "string") {
		// Inline response too large; agentchrome dumped to a temp file.
		const fs = require("node:fs") as typeof import("node:fs");
		const inner = JSON.parse(fs.readFileSync(resp.output_file, "utf8")) as {
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

async function captureVectors(
	port: string,
	tab: string,
	target: Target,
	useWorker: boolean,
): Promise<number[][]> {
	const model = getModelById(target.modelId);
	if (!model) throw new Error(`Unknown model "${target.modelId}"`);

	const url = buildSmokeTestUrl(model.id, model.contextLength, {
		extraParams: {
			v: `${Date.now()}`,
			ingest: "off",
			...(model.embeddingCapable ? { embeddingCapable: "1" } : {}),
			...(model.embeddingPooling === "mean"
				? { embeddingPooling: "mean" }
				: {}),
			...(useWorker ? { worker: "1" } : {}),
		},
	});
	console.log(`  navigating (${useWorker ? "worker" : "main"}): ${url}`);
	agentchrome(port, tab, ["navigate", url]);

	// Wait for window.engine + window.handleId. Bucket D 8B model loads slow;
	// allow up to 10 minutes.
	const loadDeadline = Date.now() + 10 * 60 * 1000;
	let loaded = false;
	while (Date.now() < loadDeadline) {
		const ready = jsExec(
			port,
			tab,
			"(() => Boolean(window.handleId && window.engine))()",
		);
		if (ready === "true") {
			loaded = true;
			break;
		}
		await sleep(1000);
	}
	if (!loaded) {
		throw new Error(
			`Timed out waiting for window.handleId on ${target.modelId} (${useWorker ? "worker" : "main"})`,
		);
	}
	console.log(`  loaded ${target.modelId} (${useWorker ? "worker" : "main"})`);

	const vectors: number[][] = [];
	for (let i = 0; i < target.fixtures.length; i++) {
		const text = target.fixtures[i];
		const inputJson = JSON.stringify(text);
		const script = `(async () => {
			const v = await window.engine.embed(window.handleId, ${inputJson});
			return JSON.stringify(Array.from(v));
		})()`;
		const raw = jsExec(port, tab, script);
		if (typeof raw !== "string") {
			throw new Error(
				`Empty embed response for fixture ${i} on ${target.modelId} (${useWorker ? "worker" : "main"})`,
			);
		}
		const vec = JSON.parse(raw) as number[];
		vectors.push(vec);
		console.log(`    row ${i}: dim=${vec.length}`);
	}
	return vectors;
}

interface ResultRow {
	modelId: string;
	gate: number;
	dim: number;
	cosines: number[];
	minCos: number;
	meanCos: number;
	pass: boolean;
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();

	for (const t of TARGETS) {
		const m = getModelById(t.modelId);
		if (!m) throw new Error(`Unknown model "${t.modelId}"`);
		await ensureModelDownloaded(m);
	}

	const results: ResultRow[] = [];
	for (const target of TARGETS) {
		console.log(`\n=== ${target.modelId} (gate >= ${target.gate}) ===`);
		const mainVecs = await captureVectors(port, tab, target, false);
		const workerVecs = await captureVectors(port, tab, target, true);
		if (mainVecs.length !== workerVecs.length) {
			throw new Error(
				`row count mismatch for ${target.modelId}: main=${mainVecs.length}, worker=${workerVecs.length}`,
			);
		}
		const cosines = mainVecs.map((mv, i) => cosine(mv, workerVecs[i]));
		const minCos = Math.min(...cosines);
		const meanCos = cosines.reduce((a, b) => a + b, 0) / cosines.length;
		const pass = minCos >= target.gate;
		results.push({
			modelId: target.modelId,
			gate: target.gate,
			dim: mainVecs[0]?.length ?? 0,
			cosines,
			minCos,
			meanCos,
			pass,
		});
		// Persist raw vectors per model.
		const rawPath = resolve(HERE, `${target.modelId}.json`);
		writeFileSync(
			rawPath,
			JSON.stringify(
				{
					modelId: target.modelId,
					gate: target.gate,
					fixtures: target.fixtures,
					mainVectors: mainVecs,
					workerVectors: workerVecs,
					cosines,
					minCos,
					meanCos,
					pass,
				},
				null,
				2,
			),
		);
		console.log(`  cosines: ${cosines.map((c) => c.toFixed(6)).join(", ")}`);
		console.log(
			`  min=${minCos.toFixed(6)} mean=${meanCos.toFixed(6)} ${pass ? "PASS" : "FAIL"}`,
		);
	}

	// Write a small summary JSON next to the per-model raw files.
	const summaryPath = resolve(HERE, "summary.json");
	writeFileSync(summaryPath, JSON.stringify(results, null, 2));

	console.log("\n=== Final ===");
	console.log("| Model | min cos | mean cos | gate | PASS? |");
	console.log("|---|---:|---:|---:|:-:|");
	for (const r of results) {
		console.log(
			`| ${r.modelId} | ${r.minCos.toFixed(6)} | ${r.meanCos.toFixed(6)} | ${r.gate} | ${r.pass ? "PASS" : "FAIL"} |`,
		);
	}

	const allPass = results.every((r) => r.pass);
	process.exit(allPass ? 0 : 1);
}

// Ensure mkdirSync is used (silences linter when dir already exists upstream).
mkdirSync(HERE, { recursive: true });

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
	process.exit(1);
});
