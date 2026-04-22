/**
 * Browser-driven inference perf benchmark.
 *
 * Unlike `bench/` (mitata micro-benchmarks running in-process with mock
 * forward passes), this measures the *real* end-to-end pipeline by
 * driving the smoke-test page in a Chrome instance via `agentchrome`
 * and scraping the tok/s + prefill/decode timings it prints.
 *
 * Usage:
 *   bun run eval/perf.ts                       # run + print table
 *   bun run eval/perf.ts --save                # also write eval/reports/perf-baseline.json
 *   bun run eval/perf.ts --model <id>          # choose a registered model (default tinyllama-1.1b-chat-q4_0)
 *   bun run eval/perf.ts --runs 5              # number of runs (default 3)
 *   bun run eval/perf.ts --port <cdp-port>     # agentchrome port (default: auto-detect)
 *   bun run eval/perf.ts --profile             # also dump per-phase decode timings
 *
 * Note: the `--profile` report is the integration-only verification seam for
 * expanded browser decode traces in Task 2; there is no narrower TS unit-test
 * seam for this path without broader refactoring.
 *
 * Requires:
 *   - smoke-test server on http://localhost:8031 (run `make smoke-serve`)
 *   - a running agentchrome session with the smoke-test page loaded
 */

import { parseArgs } from "node:util";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { getModelById, type BenchmarkModel } from "./models.js";

interface PerfRun {
	tokensGenerated: number;
	wallClockMs: number;
	totalMs: number;
	prefillMs: number;
	decodeMs: number;
	tokensPerSecond: number;
}

interface PerfReport {
	timestamp: string;
	modelId: string;
	modelName: string;
	prompt: string;
	runs: PerfRun[];
	median: PerfRun;
	notes?: string;
}

interface SmokeTestResult extends Omit<PerfRun, "wallClockMs"> {
	completionPageMs: number;
}

interface DecodeTrace {
	mode: "full" | "greedy" | "topk";
	nTokens: number;
	pastLen: number;
	ctxCreateMs: number;
	buildGraphMs: number;
	backendAllocMs: number;
	uploadLeavesMs: number;
	graphComputeMs: number;
	downloadResultMs: number;
	teardownMs: number;
	totalMs: number;
	backendProfileTotalMs?: number;
	backendMatmulMs?: number | null;
	backendAttentionMs?: number | null;
	backendEncodeOverheadMs?: number;
	backendDispatchCount?: number;
	backendBreakdownAvailable?: boolean;
}

const DEFAULT_MODEL_ID = "tinyllama-1.1b-chat-q4_0";
const DEFAULT_RUNS = 3;
const SMOKE_TEST_URL = "http://localhost:8031/real-model.html";
const BASELINE_PATH = "eval/reports/perf-baseline.json";

function main(): void {
	const { values } = parseArgs({
		options: {
			model: { type: "string", short: "m" },
			runs: { type: "string" },
			save: { type: "boolean" },
			port: { type: "string" },
			tab: { type: "string" },
			baseline: { type: "string" },
			profile: { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	const modelId = values.model ?? DEFAULT_MODEL_ID;
	const runs = values.runs ? Number.parseInt(values.runs, 10) : DEFAULT_RUNS;

	const model = getModelById(modelId);
	if (!model) {
		console.error(`Unknown model "${modelId}". Use --model with an ID from eval/models.ts.`);
		process.exit(1);
	}

	run(model, runs, {
		port: values.port,
		tab: values.tab,
		save: values.save ?? false,
		baseline: values.baseline ?? BASELINE_PATH,
		profile: values.profile ?? false,
	}).catch((err) => {
		console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
		process.exit(1);
	});
}

async function run(
	model: BenchmarkModel,
	nRuns: number,
	opts: {
		port?: string;
		tab?: string;
		save: boolean;
		baseline: string;
		profile: boolean;
	},
): Promise<void> {
	const { port, tab } = await resolveAgentchromeSession(opts.port, opts.tab);

	// Trigger N page reloads with cache busting and collect tok/s from each.
	const perfRuns: PerfRun[] = [];
	const allTraces: DecodeTrace[] = [];
	for (let i = 0; i < nRuns; i++) {
		process.stdout.write(`Run ${i + 1}/${nRuns}...`);
		const base = `${SMOKE_TEST_URL}?perf=${Date.now()}-${i}`;
		const url = opts.profile ? `${base}&profile=1` : base;
		agentchrome(port, tab, ["navigate", url]);
		const result = await waitForSmokeTestResult(port, tab);
		perfRuns.push({
			tokensGenerated: result.tokensGenerated,
			wallClockMs: result.completionPageMs,
			totalMs: result.totalMs,
			prefillMs: result.prefillMs,
			decodeMs: result.decodeMs,
			tokensPerSecond: result.tokensPerSecond,
		});
		if (opts.profile) {
			const traces = fetchDecodeTraces(port, tab);
			const greedyDecodeTraces = traces.filter(
				(t) => t.mode === "greedy" && t.nTokens === 1,
			);
			allTraces.push(...greedyDecodeTraces);
		}
		process.stdout.write(` ${result.tokensPerSecond.toFixed(1)} tok/s\n`);
	}

	const median = medianRun(perfRuns);
	const prompt = await extractSmokeTestPrompt(port, tab);

	const report: PerfReport = {
		timestamp: new Date().toISOString(),
		modelId: model.id,
		modelName: model.name,
		prompt,
		runs: perfRuns,
		median,
	};

	printTable(report);
	compareToBaseline(report, opts.baseline);
	if (opts.profile) {
		printProfileTable(allTraces);
	}

	if (opts.save) {
		mkdirSync(dirname(opts.baseline), { recursive: true });
		writeFileSync(opts.baseline, `${JSON.stringify(report, null, 2)}\n`);
		console.log(`\nBaseline written to ${opts.baseline}`);
	}
}

function medianRun(runs: PerfRun[]): PerfRun {
	const sorted = [...runs].sort((a, b) => a.tokensPerSecond - b.tokensPerSecond);
	return sorted[Math.floor(sorted.length / 2)];
}

function printTable(report: PerfReport): void {
	console.log(`\nInference perf — ${report.modelName} (${report.modelId})`);
	console.log(`Timestamp: ${report.timestamp}`);
	console.log(`Prompt:    ${JSON.stringify(report.prompt.slice(0, 80))}${report.prompt.length > 80 ? "…" : ""}`);
	console.log();
	console.log("Run  Tokens  Wall(ms)  Total(ms)  Prefill(ms)  Decode(ms)  tok/s");
	console.log("---  ------  --------  ---------  -----------  ----------  -----");
	for (let i = 0; i < report.runs.length; i++) {
		const r = report.runs[i];
		console.log(
			`${String(i + 1).padStart(3)}  ${String(r.tokensGenerated).padStart(6)}  ${String(Math.round(r.wallClockMs)).padStart(8)}  ${String(Math.round(r.totalMs)).padStart(9)}  ${String(Math.round(r.prefillMs)).padStart(11)}  ${String(Math.round(r.decodeMs)).padStart(10)}  ${r.tokensPerSecond.toFixed(1).padStart(5)}`,
		);
	}
	console.log("---  ------  --------  ---------  -----------  ----------  -----");
	const m = report.median;
	console.log(
		`p50* ${String(m.tokensGenerated).padStart(6)}  ${String(Math.round(m.wallClockMs)).padStart(8)}  ${String(Math.round(m.totalMs)).padStart(9)}  ${String(Math.round(m.prefillMs)).padStart(11)}  ${String(Math.round(m.decodeMs)).padStart(10)}  ${m.tokensPerSecond.toFixed(1).padStart(5)}`,
	);
	console.log("* row selected by median tok/s; other columns are from that run, not column-wise medians.");
}

function compareToBaseline(current: PerfReport, baselinePath: string): void {
	if (!existsSync(baselinePath)) {
		console.log(`\nNo baseline at ${baselinePath} — use --save to record this run as the baseline.`);
		return;
	}
	try {
		const baseline: PerfReport = JSON.parse(readFileSync(baselinePath, "utf-8"));
		if (baseline.modelId !== current.modelId) {
			console.log(`\nBaseline model differs (${baseline.modelId}) — skipping comparison.`);
			return;
		}
		const diff = current.median.tokensPerSecond - baseline.median.tokensPerSecond;
		const pct = (diff / baseline.median.tokensPerSecond) * 100;
		const sign = diff >= 0 ? "+" : "";
		console.log(
			`\nvs baseline (${baseline.timestamp}): ${baseline.median.tokensPerSecond.toFixed(1)} tok/s → ${current.median.tokensPerSecond.toFixed(1)} tok/s (${sign}${diff.toFixed(1)} / ${sign}${pct.toFixed(1)}%)`,
		);
	} catch (err) {
		console.log(`\nCouldn't read baseline ${baselinePath}: ${err instanceof Error ? err.message : err}`);
	}
}

// ── agentchrome helpers ────────────────────────────────────────────────────

function agentchrome(port: string, tab: string | undefined, args: string[]): string {
	const full = ["--port", port];
	if (tab) full.push("--tab", tab);
	full.push(...args);
	return execFileSync("agentchrome", full, { encoding: "utf-8" });
}

async function resolveAgentchromeSession(
	portArg?: string,
	tabArg?: string,
): Promise<{ port: string; tab: string }> {
	let port = portArg;
	if (!port) {
		// Query agentchrome's connect status for the active session port.
		const status = execFileSync("agentchrome", ["connect", "--status"], {
			encoding: "utf-8",
		});
		const parsed = JSON.parse(status);
		if (!parsed.port) {
			throw new Error(
				"No active agentchrome session. Start one with `agentchrome connect --launch` or pass --port.",
			);
		}
		port = String(parsed.port);
	}

	if (tabArg) return { port, tab: tabArg };

	const tabs = execFileSync("agentchrome", ["--port", port, "tabs", "list"], {
		encoding: "utf-8",
	});
	const list = JSON.parse(tabs) as Array<{ id: string; url: string }>;
	const smoke = list.find((t) => t.url.includes("real-model.html"));
	if (!smoke) {
		throw new Error(
			`No tab currently loaded on real-model.html. Navigate one there first, or pass --tab <TAB_ID>.`,
		);
	}
	return { port, tab: smoke.id };
}

async function waitForSmokeTestResult(
	port: string,
	tab: string,
): Promise<SmokeTestResult> {
	const out = agentchrome(port, tab, [
		"js",
		"exec",
		`(async () => {
			const pattern = /Generated (\\d+) tokens in ([0-9.]+)s \\(prefill: (\\d+)ms, decode: (\\d+)ms, ([0-9.]+) tok\\/s\\)/;
			const parse = () => {
				const t = document.getElementById("log")?.textContent ?? "";
				const m = t.match(pattern);
				if (!m) return null;
				return JSON.stringify({
					tokensGenerated: +m[1],
					totalMs: +m[2] * 1000,
					prefillMs: +m[3],
					decodeMs: +m[4],
					tokensPerSecond: +m[5],
					completionPageMs: performance.now(),
				});
			};

			const ready = parse();
			if (ready) return ready;

			return await new Promise((resolve, reject) => {
				const log = document.getElementById("log");
				if (!log) {
					reject(new Error("Missing #log element on smoke-test page"));
					return;
				}

				const timeout = setTimeout(() => {
					observer.disconnect();
					reject(new Error("Timed out waiting for smoke-test result line on the page"));
				}, 180_000);

				const observer = new MutationObserver(() => {
					const result = parse();
					if (!result) return;
					clearTimeout(timeout);
					observer.disconnect();
					resolve(result);
				});

				observer.observe(log, { childList: true, characterData: true, subtree: true });
			});
		})()`,
	]);
	try {
		const resp = JSON.parse(out) as { result?: string; type?: string };
		if (resp.result && resp.type === "string") {
			return JSON.parse(resp.result) as SmokeTestResult;
		}
	} catch {
		// fall through to the shared error below
	}
	throw new Error("Timed out waiting for smoke-test result line on the page");
}

function fetchDecodeTraces(port: string, tab: string): DecodeTrace[] {
	const out = agentchrome(port, tab, [
		"js",
		"exec",
		`(() => JSON.stringify(window.__decodeTraces ?? []))()`,
	]);
	const resp = JSON.parse(out) as { result?: string };
	if (!resp.result) return [];
	try {
		return JSON.parse(resp.result) as DecodeTrace[];
	} catch {
		return [];
	}
}

function printProfileTable(traces: DecodeTrace[]): void {
	if (traces.length === 0) {
		console.log("\nNo decode traces captured (profile mode).");
		return;
	}
	const keys: Array<keyof DecodeTrace> = [
		"ctxCreateMs",
		"buildGraphMs",
		"backendAllocMs",
		"uploadLeavesMs",
		"graphComputeMs",
		"downloadResultMs",
		"teardownMs",
		"totalMs",
	];
	console.log(`\nPer-phase decode timing (${traces.length} single-token steps)`);
	console.log("Phase              mean(ms)  median(ms)  p90(ms)  %total");
	console.log("-----------------  --------  ----------  -------  ------");
	const totalMean = traces.reduce((acc, t) => acc + t.totalMs, 0) / traces.length;
	for (const k of keys) {
		const stats = summarizeNumbers(traces.map((t) => t[k] as number));
		const pct = k === "totalMs" ? 100 : (stats.mean / totalMean) * 100;
		console.log(
			`${k.padEnd(17)}  ${stats.mean.toFixed(2).padStart(8)}  ${stats.median.toFixed(2).padStart(10)}  ${stats.p90.toFixed(2).padStart(7)}  ${pct.toFixed(1).padStart(5)}%`,
		);
	}

	printBackendAttributionTable(traces);
}

function printBackendAttributionTable(traces: DecodeTrace[]): void {
	const timedRows: Array<{
		label: string;
		values: number[];
		graphValues: number[];
	}> = [
		{
			label: "backendProfileTotalMs",
			...collectTimedRow(traces, (t) => t.backendProfileTotalMs),
		},
		{
			label: "backendEncodeOverheadMs",
			...collectTimedRow(traces, (t) => t.backendEncodeOverheadMs),
		},
		{
			label: "backendMatmulMs",
			...collectTimedRow(traces, (t) => t.backendMatmulMs),
		},
		{
			label: "backendAttentionMs",
			...collectTimedRow(traces, (t) => t.backendAttentionMs),
		},
	];
	const dispatchValues = traces
		.map((t) => t.backendDispatchCount)
		.filter((v): v is number => typeof v === "number");
	if (timedRows.every((row) => row.values.length === 0) && dispatchValues.length === 0) {
		return;
	}

	console.log("\nBackend decode attribution (when available)");
	console.log("Field                   samples  mean      median    p90       %graph");
	console.log("----------------------  -------  --------  --------  --------  ------");
	for (const row of timedRows) {
		if (row.values.length === 0) continue;
		const stats = summarizeNumbers(row.values);
		const graphStats = summarizeNumbers(row.graphValues);
		const pct = graphStats.mean > 0 ? (stats.mean / graphStats.mean) * 100 : 0;
		console.log(
			`${row.label.padEnd(22)}  ${String(row.values.length).padStart(7)}  ${stats.mean.toFixed(2).padStart(8)}  ${stats.median.toFixed(2).padStart(8)}  ${stats.p90.toFixed(2).padStart(8)}  ${pct.toFixed(1).padStart(5)}%`,
		);
	}
	if (dispatchValues.length > 0) {
		const stats = summarizeNumbers(dispatchValues);
		console.log(
			`${"backendDispatchCount".padEnd(22)}  ${String(dispatchValues.length).padStart(7)}  ${stats.mean.toFixed(1).padStart(8)}  ${stats.median.toFixed(1).padStart(8)}  ${stats.p90.toFixed(1).padStart(8)}  ${"-".padStart(6)}`,
		);
	}

	const breakdownTraces = traces.filter((t) => t.backendBreakdownAvailable === true).length;
	if (breakdownTraces > 0 && breakdownTraces !== traces.length) {
		console.log(
			`Breakdown fields were present on ${breakdownTraces}/${traces.length} decode traces; %graph uses only the matching samples for each row.`,
		);
	}
	console.log(
		"Profile note: backend attribution is collected only in --profile mode and can perturb absolute timing; use non-profile runs for representative throughput comparisons.",
	);
}

function collectTimedRow(
	traces: DecodeTrace[],
	selector: (trace: DecodeTrace) => number | null | undefined,
): { values: number[]; graphValues: number[] } {
	const values: number[] = [];
	const graphValues: number[] = [];
	for (const trace of traces) {
		const value = selector(trace);
		if (typeof value !== "number") continue;
		values.push(value);
		graphValues.push(trace.graphComputeMs);
	}
	return { values, graphValues };
}

function summarizeNumbers(values: number[]): { mean: number; median: number; p90: number } {
	const sorted = [...values].sort((a, b) => a - b);
	const mean = sorted.reduce((a, b) => a + b, 0) / sorted.length;
	return {
		mean,
		median: sorted[Math.floor(sorted.length / 2)],
		p90: sorted[Math.floor(sorted.length * 0.9)],
	};
}

async function extractSmokeTestPrompt(port: string, tab: string): Promise<string> {
	try {
		const out = agentchrome(port, tab, [
			"js",
			"exec",
			`(() => {
				const t = document.getElementById("log")?.textContent ?? "";
				const m = t.match(/User: (.+?)(?:Assistant:|\\n|$)/);
				return m ? m[1].trim() : "";
			})()`,
		]);
		const resp = JSON.parse(out);
		return typeof resp.result === "string" ? resp.result : "";
	} catch {
		return "";
	}
}

function printUsage(): void {
	console.log(`Usage: bun run eval/perf.ts [options]

Options:
  -m, --model <id>      Model to benchmark (default: ${DEFAULT_MODEL_ID})
      --runs <n>        Number of runs (default: ${DEFAULT_RUNS})
      --save            Write eval/reports/perf-baseline.json
      --port <cdp-port> Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>    Use this specific Chrome tab ID
      --baseline <path> Baseline file path (default: ${BASELINE_PATH})
      --profile         Also print per-phase decode timing from browser traces
  -h, --help            Show this help

Prereqs:
  - smoke-test server up: \`make smoke-serve\`
  - Chrome with the smoke test open via agentchrome
`);
}

main();
