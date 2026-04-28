/**
 * Encoder perf harness for §D. Drives the smoke-test page's [8/8]
 * arctic-embed engine with embedPerf URL params, captures wall-time
 * traces, and prints a per-(model, fixture, mode) summary.
 *
 * Usage:
 *   bun run eval/embed-perf.ts                                 # both models, both modes, both fixtures
 *   bun run eval/embed-perf.ts --model snowflake-arctic-embed-s-q0f32-b4
 *   bun run eval/embed-perf.ts --mode single --fixture short   # one cell
 *   bun run eval/embed-perf.ts --reps 50                       # override single-mode reps
 *   bun run eval/embed-perf.ts --profile                       # also pass profile=1 to smoke page
 *   bun run eval/embed-perf.ts --out eval/reports/embed-perf-2026-04-27/
 *
 * Requires:
 *   - smoke-test server up (`make smoke-serve`)
 *   - a running agentchrome session
 */

import { parseArgs } from "node:util";
import { mkdirSync, writeFileSync } from "node:fs";
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
	"bge-small-en-v1.5-q0f16",
	"bge-large-en-v1.5-q0f16",
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
		const m = getModelById(id);
		if (!m) throw new Error(`Unknown model "${id}"`);
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
				writeFileSync(logPath, JSON.stringify(cell, null, 2) + "\n");
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
