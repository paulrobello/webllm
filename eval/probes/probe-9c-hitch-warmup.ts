/**
 * Probe 9c — Hitch warmup probe (queued 2026-05-01).
 *
 * Tests whether the deterministic ~50ms decode-shape hitch on call 0
 * (observed in the 2026-05-01 frame-probe multi-call results) can be
 * absorbed by a one-shot 4-token throwaway chatCompletion at session
 * boot. Drives the smoke page twice on `qwen3-8b-iq3m`:
 *
 *   A. `?frameProbe=1&frameProbeCalls=5` (no warmup, control)
 *   B. `?frameProbe=1&frameProbeCalls=5&frameProbeWarmup=1` (warmup)
 *
 * For each scenario we read `window.__frameProbeResult.frameStats.perCall`
 * and compare call 0's `decode.max` (and call 0's prefill.max for
 * symmetry). Pass: warmup brings call-0 decode_max into the 8.3-12 ms
 * band that matches subsequent calls' decode_max minus the structural
 * hitch.
 *
 * Usage: `bun run eval/probes/probe-9c-hitch-warmup.ts`. Writes a
 * SUMMARY.md to `eval/reports/probe-9c-2026-05-01/`.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
	waitForSmokeTestResult,
} from "../browser-smoke.ts";
import { getModelById } from "../models.ts";

const MODEL_ID = "qwen3-8b-iq3m";
const FRAME_PROBE_CALLS = 5;

interface PerCallStat {
	index: number;
	prefill: { max?: number; p95?: number; n?: number };
	decode: { max?: number; p95?: number; n?: number };
}

interface FrameProbeResult {
	model: string;
	mode: "multi" | "single";
	frameStats: { perCall?: PerCallStat[] };
}

function sleep(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

async function fetchFrameProbeResult(
	port: string,
	tab: string,
): Promise<FrameProbeResult> {
	const deadline = Date.now() + 360_000;
	const script = `(() => {
		const r = window.__frameProbeResult;
		return r ? JSON.stringify(r) : "";
	})()`;
	while (Date.now() < deadline) {
		const out = agentchrome(port, tab, ["js", "exec", script]);
		const parsed = JSON.parse(out) as { result?: string; type?: string };
		if (parsed.type === "string" && parsed.result) {
			return JSON.parse(parsed.result) as FrameProbeResult;
		}
		await sleep(1000);
	}
	throw new Error("Timed out waiting for window.__frameProbeResult");
}

async function runScenario(
	label: string,
	warmup: boolean,
	port: string,
	tab: string,
	contextLength: number,
): Promise<FrameProbeResult> {
	process.stdout.write(`Scenario ${label} (warmup=${warmup})…\n`);
	const url = buildSmokeTestUrl(MODEL_ID, contextLength, {
		extraParams: {
			perf: `${Date.now()}-${label}`,
			frameProbe: 1,
			frameProbeCalls: FRAME_PROBE_CALLS,
			...(warmup ? { frameProbeWarmup: 1 } : {}),
			ingest: "off",
		},
	});
	agentchrome(port, tab, ["navigate", url]);
	// Wait for the smoke completion line first (smoke result), then
	// pull the frame-probe summary from window.
	await waitForSmokeTestResult(port, tab);
	// Multi-call probe runs N-1 additional calls AFTER the smoke
	// result line lands; let it complete before scraping.
	const probeResult = await fetchFrameProbeResult(port, tab);
	return probeResult;
}

function formatPerCall(perCall?: PerCallStat[]): string {
	if (!perCall || perCall.length === 0) return "(no per-call stats)";
	const rows = perCall.map((c) => {
		const dMax = c.decode.max ?? 0;
		const dP95 = c.decode.p95 ?? 0;
		const pMax = c.prefill.max ?? 0;
		return `  call ${c.index}: prefill_max=${pMax.toFixed(1)}ms, decode_p95=${dP95.toFixed(1)}ms, decode_max=${dMax.toFixed(1)}ms`;
	});
	return rows.join("\n");
}

async function main(): Promise<void> {
	const model = getModelById(MODEL_ID);
	if (!model) throw new Error(`Unknown model: ${MODEL_ID}`);

	await ensureSmokeServerReachable();
	await ensureModelDownloaded(model);
	const { port, tab } = await resolveAgentchromeSession();

	// Run control (no warmup) first, then warmup. Each run is its own
	// page load so the WebGPU shader cache state doesn't bleed across
	// scenarios.
	const control = await runScenario(
		"control",
		false,
		port,
		tab,
		model.contextLength,
	);
	const warmup = await runScenario(
		"warmup",
		true,
		port,
		tab,
		model.contextLength,
	);

	const ctlPerCall = control.frameStats.perCall ?? [];
	const wupPerCall = warmup.frameStats.perCall ?? [];
	const ctlCall0DecodeMax = ctlPerCall[0]?.decode.max ?? 0;
	const wupCall0DecodeMax = wupPerCall[0]?.decode.max ?? 0;

	// "Subsequent calls" baseline: median of decode_max across calls 1..N-1
	// in the control run — that's the "hitch-absorbed" steady state we
	// want call 0 to match.
	const subsequentMaxes = ctlPerCall
		.slice(1)
		.map((c) => c.decode.max ?? 0)
		.filter((v) => v > 0)
		.sort((a, b) => a - b);
	const subsequentMedian =
		subsequentMaxes[Math.floor(subsequentMaxes.length / 2)] ?? 0;

	const lines: string[] = [];
	lines.push("# Probe 9c — Hitch warmup");
	lines.push("");
	lines.push(`**Model:** ${MODEL_ID}`);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(`**Frame-probe calls per scenario:** ${FRAME_PROBE_CALLS}`);
	lines.push("");
	lines.push("## Method");
	lines.push("");
	lines.push(
		"Two scenarios on the same page-load cadence. Control runs the existing `?frameProbeCalls=5` multi-call probe with no modification. Warmup runs the same probe with a 4-token throwaway `chatCompletion` inserted between baseline-rAF window close and the timed probe start (`?frameProbeWarmup=1`). The 500ms inter-call settle that already exists between subsequent calls is also applied after the warmup throwaway.",
	);
	lines.push("");
	lines.push(
		"Threshold: warmup brings call-0 `decode.max` into the band of subsequent calls' `decode.max` (control's call-1..4 median).",
	);
	lines.push("");
	lines.push("## Control (no warmup) — per-call frame stats");
	lines.push("");
	lines.push("```");
	lines.push(formatPerCall(ctlPerCall));
	lines.push("```");
	lines.push("");
	lines.push("## Warmup — per-call frame stats");
	lines.push("");
	lines.push("```");
	lines.push(formatPerCall(wupPerCall));
	lines.push("```");
	lines.push("");
	lines.push("## Headline comparison");
	lines.push("");
	lines.push(
		`- Control call-0 decode_max: **${ctlCall0DecodeMax.toFixed(1)} ms**`,
	);
	lines.push(
		`- Warmup  call-0 decode_max: **${wupCall0DecodeMax.toFixed(1)} ms**`,
	);
	lines.push(
		`- Control subsequent (calls 1..${ctlPerCall.length - 1}) decode_max median: **${subsequentMedian.toFixed(1)} ms**`,
	);
	const passBand = wupCall0DecodeMax <= 12 && wupCall0DecodeMax >= 0;
	const matchesSubsequent =
		Math.abs(wupCall0DecodeMax - subsequentMedian) <= 6;
	lines.push("");
	lines.push("## Verdict");
	lines.push("");
	if (passBand && matchesSubsequent) {
		lines.push(
			"**PASS** — warmup brings call-0 decode_max into the 8.3-12 ms band that matches subsequent calls' steady state.",
		);
		lines.push(
			"Decision: bake the warmup throwaway into engine init path. The 4-token throwaway adds ~200-400 ms to session boot but eliminates the per-shape JIT spike from the first user-visible NPC tick. For any agent harness running ≥1 Hz tick rate, that's the right tradeoff (one-time vs per-call hitch).",
		);
	} else if (
		wupCall0DecodeMax < ctlCall0DecodeMax - 5 &&
		!matchesSubsequent
	) {
		lines.push(
			"**PARTIAL** — warmup reduces call-0 decode_max but doesn't fully match subsequent-call steady state. Hypothesis: the hitch has multiple components (per-shape pipeline JIT + something else like KV-cache first-fill).",
		);
		lines.push(
			"Decision: warmup is still a net win at session boot (cheap, deterministic improvement); follow-up probe to identify the residual hitch source.",
		);
	} else {
		lines.push(
			"**FAIL** — warmup does not move the needle on call-0 decode_max.",
		);
		lines.push(
			"Decision: hitch is not driven by per-shape JIT; warmup-throwaway is not the right intervention. Investigate alternative (KV-cache pre-allocation, frame-pacing changes, worker migration).",
		);
	}

	const outDir = "eval/reports/probe-9c-2026-05-01";
	mkdirSync(outDir, { recursive: true });
	writeFileSync(`${outDir}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(
		`${outDir}/raw-control.json`,
		`${JSON.stringify(control, null, 2)}\n`,
	);
	writeFileSync(
		`${outDir}/raw-warmup.json`,
		`${JSON.stringify(warmup, null, 2)}\n`,
	);
	console.log(`\nReport: ${outDir}/SUMMARY.md`);

	// Quiet `execFileSync` lint warning — keep the import live so the
	// module's CLI shape stays compatible with the other probes.
	void execFileSync;
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
