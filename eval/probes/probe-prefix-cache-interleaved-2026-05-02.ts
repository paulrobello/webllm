/**
 * Prefix-cache interleaved validation probe.
 *
 * Round-robins NPC ticks (NPC_1 t1 → NPC_2 t1 → NPC_3 t1 → NPC_4 t1 →
 * NPC_1 t2 → NPC_2 t2 → ...) with per-NPC distinct ~1100-token
 * personas. The personas embed each NPC's id very early so longest-
 * shared-prefix between any two siblings is just the small framework
 * intro. After NPC_4 tick-1, the engine's per-model session-tracker
 * holds NPC_4's KV; when NPC_1 tick-2 fires, Pattern A must re-prefill
 * the entire NPC_1 persona because the session tracker can't
 * preserve it. Pattern B reloads NPC_1's per-conv KV snapshot
 * (~1.3 s post-batch) and prefills only the divergent tail.
 *
 * This is the matrix the at-scale probe (`probe-prefix-cache-at-scale-
 * 2026-05-01.ts`) was structurally unable to demonstrate — sequential
 * NPCs let Pattern A's session tracker preserve each conversation's
 * own state.
 *
 * Verdict criteria (wall-time only — prefillMs on the conv path is
 * structurally biased low):
 *   - PASS:    pattern B tick-2 wall ≥ 30% faster than pattern A
 *   - PARTIAL: pattern B faster but < 30%
 *   - FAIL:    pattern B slower or wash
 *
 * Usage: `bun run eval/probes/probe-prefix-cache-interleaved-2026-05-02.ts`.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "../browser-smoke.ts";

const MODEL_ID = "qwen3-8b-iq3m";
const REPORT_DIR = "eval/reports/prefix-cache-interleaved-2026-05-02";

interface PerCall {
	npcId: string;
	tick: 1 | 2;
	prefillMs: number;
	wallMs: number;
	output: string;
}

interface ProbeResult {
	model: string;
	patternA: PerCall[];
	patternB: PerCall[];
}

function sleep(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

async function fetchProbeResult(
	port: string,
	tab: string,
	timeoutMs = 1_800_000,
): Promise<ProbeResult> {
	const deadline = Date.now() + timeoutMs;
	const script = `(() => {
		const r = window.__probePrefixCacheInterleavedResult;
		return r ? JSON.stringify(r) : "";
	})()`;
	while (Date.now() < deadline) {
		const out = agentchrome(port, tab, ["js", "exec", script]);
		const parsed = JSON.parse(out) as { result?: string; type?: string };
		if (parsed.type === "string" && parsed.result) {
			return JSON.parse(parsed.result) as ProbeResult;
		}
		await sleep(2000);
	}
	throw new Error(
		"Timed out waiting for window.__probePrefixCacheInterleavedResult",
	);
}

function median(xs: number[]): number {
	const s = xs.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
	if (s.length === 0) return 0;
	return s[Math.floor(s.length / 2)] ?? 0;
}

function tableRow(call: PerCall): string {
	const out = (call.output ?? "").replace(/\s+/g, " ").trim().slice(0, 32);
	return `  ${call.npcId.padEnd(12)} tick=${call.tick}  prefill=${call.prefillMs.toFixed(1).padStart(8)}ms  wall=${call.wallMs.toFixed(1).padStart(8)}ms  output="${out}"`;
}

interface Verdict {
	tag: "PASS" | "PARTIAL" | "FAIL";
	rationale: string;
}

function verdictFor(
	tick2BWallMedian: number,
	tick2AWallMedian: number,
): Verdict {
	const wallSavingsPct =
		tick2AWallMedian > 0
			? ((tick2AWallMedian - tick2BWallMedian) / tick2AWallMedian) * 100
			: 0;
	if (wallSavingsPct >= 30) {
		return {
			tag: "PASS",
			rationale: `Pattern B tick-2 wall ${tick2BWallMedian.toFixed(0)} ms vs A's ${tick2AWallMedian.toFixed(0)} ms — ${wallSavingsPct.toFixed(0)}% savings. Per-conv prefix cache delivers in the interleaved regime.`,
		};
	}
	if (wallSavingsPct > 0) {
		return {
			tag: "PARTIAL",
			rationale: `Pattern B tick-2 wall is ${wallSavingsPct.toFixed(1)}% faster than A. Prefix cache delivers but overhead still significant.`,
		};
	}
	return {
		tag: "FAIL",
		rationale: `Pattern B tick-2 wall is ${(-wallSavingsPct).toFixed(1)}% SLOWER than A even in the interleaved regime — the per-conv save+load cost still exceeds the re-prefill cost the session tracker can't avoid.`,
	};
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();

	const url = `http://localhost:8031/real-model.html?model=${MODEL_ID}&probe=prefix-cache-interleaved&fa=on&ingest=off&v=${Date.now()}`;
	console.log(`Driving prefix-cache interleaved probe: ${url}`);
	agentchrome(port, tab, ["navigate", url]);

	const probe = await fetchProbeResult(port, tab);

	const tick1AWall = probe.patternA
		.filter((r) => r.tick === 1)
		.map((r) => r.wallMs);
	const tick1BWall = probe.patternB
		.filter((r) => r.tick === 1)
		.map((r) => r.wallMs);
	const tick2A = probe.patternA
		.filter((r) => r.tick === 2)
		.map((r) => r.prefillMs);
	const tick2B = probe.patternB
		.filter((r) => r.tick === 2)
		.map((r) => r.prefillMs);
	const tick2AWall = probe.patternA
		.filter((r) => r.tick === 2)
		.map((r) => r.wallMs);
	const tick2BWall = probe.patternB
		.filter((r) => r.tick === 2)
		.map((r) => r.wallMs);

	const tick1AWallMedian = median(tick1AWall);
	const tick1BWallMedian = median(tick1BWall);
	const tick2AMedian = median(tick2A);
	const tick2BMedian = median(tick2B);
	const tick2AWallMedian = median(tick2AWall);
	const tick2BWallMedian = median(tick2BWall);

	const verdict = verdictFor(tick2BWallMedian, tick2AWallMedian);

	const wallDeltaMs = tick2BWallMedian - tick2AWallMedian;
	const wallDeltaPct =
		tick2AWallMedian > 0 ? (wallDeltaMs / tick2AWallMedian) * 100 : 0;
	const deltaSign = wallDeltaMs >= 0 ? "+" : "";
	const deltaQual =
		wallDeltaMs >= 0 ? "vs Pattern A tick-2" : "faster than Pattern A tick-2";

	const lines: string[] = [];
	lines.push("# Prefix-cache interleaved validation — qwen3-8b-iq3m");
	lines.push("");
	lines.push(
		`**Model:** ${probe.model} (~1100-token per-NPC distinct persona; 4 NPC × 2 ticks × 2 patterns; round-robin matrix; maxTokens=32; FA on)`,
	);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(
		`**Verdict:** **${verdict.tag}** — Pattern B tick-2 wall is ${deltaSign}${wallDeltaMs.toFixed(0)} ms / ${deltaSign}${wallDeltaPct.toFixed(1)}% ${deltaQual}.`,
	);
	lines.push("");
	lines.push("## Headline (wall-time)");
	lines.push("");
	lines.push(
		"| pattern | tick-1 wall (median) | tick-2 wall (median) | tick-2 prefillMs (median) |",
	);
	lines.push("|---|---|---|---|");
	lines.push(
		`| A (\`chatCompletion(modelId, ...)\`) | ${tick1AWallMedian.toFixed(1)} ms | ${tick2AWallMedian.toFixed(1)} ms | ${tick2AMedian.toFixed(1)} ms |`,
	);
	lines.push(
		`| B (\`chatCompletion(conv, ...)\`)    | ${tick1BWallMedian.toFixed(1)} ms | ${tick2BWallMedian.toFixed(1)} ms | ${tick2BMedian.toFixed(1)} ms |`,
	);
	lines.push("");
	lines.push("## Why this matrix is different");
	lines.push("");
	lines.push(
		"The at-scale probe (`probe-prefix-cache-at-scale-2026-05-01`) ran 4 NPCs × 2 ticks **sequentially per NPC** (`NPC_1 t1 → NPC_1 t2 → NPC_2 t1 → ...`). Pattern A's per-model session tracker preserved each NPC's own KV across its own t1 → t2 boundary, so tick-2 only re-prefilled the divergent tail (~80 tokens). Pattern B paid an extra save+load on top, making it ~12-20% slower regardless of prefix size.",
	);
	lines.push("");
	lines.push(
		"This probe **interleaves** by round-robining all tick-1s first, then all tick-2s, with **per-NPC distinct personas** (~1100 tokens each, NPC id embedded in the first sentence). After NPC_4 tick-1 finishes, the session tracker holds NPC_4's KV. When NPC_1 tick-2 fires, the longest-shared-token-prefix with the cached state is just the small framework intro (~30 tokens). Pattern A is forced to re-prefill the entire NPC_1 persona (~1100 tokens, ~14 s at 12.5 ms/token) on every tick-2 call.",
	);
	lines.push("");
	lines.push(
		"Pattern B reloads the per-conv snapshot for that NPC and prefills only the divergent tail.",
	);
	lines.push("");
	lines.push("## Pattern A per-call detail (round-robin order)");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternA) lines.push(tableRow(c));
	lines.push("```");
	lines.push("");
	lines.push("## Pattern B per-call detail (round-robin order)");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternB) lines.push(tableRow(c));
	lines.push("```");
	lines.push("");
	lines.push("## Verdict rationale");
	lines.push("");
	lines.push(`**${verdict.tag}** — ${verdict.rationale}`);
	lines.push("");
	lines.push("## Caveats");
	lines.push("");
	lines.push("- **Single-run probe.** No averaging across page loads.");
	lines.push(
		"- **prefillMs on the conv path is structurally biased low** (manual mid-prefill happens before `generateTextStream`'s timed window). Wall-time is the source of truth.",
	);
	lines.push(
		"- **`?fa=on` is required.** Conversation handles only work in FA mode.",
	);
	lines.push(
		"- **Personas are programmatically generated** (paragraph repeated 6×). Each NPC's persona is internally repetitive but distinct from siblings via the embedded id.",
	);

	mkdirSync(REPORT_DIR, { recursive: true });
	writeFileSync(`${REPORT_DIR}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(
		`${REPORT_DIR}/raw.json`,
		`${JSON.stringify(probe, null, 2)}\n`,
	);
	console.log(`\nVerdict: ${verdict.tag}`);
	console.log(
		`  Pattern A tick-2 median prefill: ${tick2AMedian.toFixed(1)} ms, wall: ${tick2AWallMedian.toFixed(1)} ms`,
	);
	console.log(
		`  Pattern B tick-2 median prefill: ${tick2BMedian.toFixed(1)} ms, wall: ${tick2BWallMedian.toFixed(1)} ms`,
	);
	console.log(`Report: ${REPORT_DIR}/SUMMARY.md`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
