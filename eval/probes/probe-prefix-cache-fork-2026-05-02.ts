/**
 * forkConversation cross-conv prefix sharing probe.
 *
 * Measures the first-tick-per-NPC prefill savings delivered by
 * `WebLLM.forkConversation`. Pattern X (baseline): each NPC creates a
 * fresh conversation; first chatCompletion prefills the entire
 * ~1325-token shared system prefix. Pattern Y (forked): a base conv
 * is primed with the shared prefix once, then forked per NPC; each
 * fork's first chatCompletion finds the shared prefix in the
 * inherited snapshot via the longest-shared-token-prefix walk and
 * prefills only the divergent NPC tail.
 *
 * Verdict criteria (wall-time only — prefillMs on the conv path is
 * structurally biased low):
 *   - PASS:    pattern Y first-tick wall ≥ 50% faster than pattern X
 *   - PARTIAL: faster but < 50%
 *   - FAIL:    not faster
 *
 * Usage: `bun run eval/probes/probe-prefix-cache-fork-2026-05-02.ts`.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "../browser-smoke.ts";

const MODEL_ID = "qwen3-8b-iq3m";
const REPORT_DIR = "eval/reports/prefix-cache-fork-2026-05-02";

interface PerCall {
	npcId: string;
	prefillMs: number;
	wallMs: number;
	output: string;
}

interface ProbeResult {
	model: string;
	baseTickMs: number;
	patternX: PerCall[];
	patternY: PerCall[];
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
		const r = window.__probePrefixCacheForkResult;
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
	throw new Error("Timed out waiting for window.__probePrefixCacheForkResult");
}

function median(xs: number[]): number {
	const s = xs.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
	if (s.length === 0) return 0;
	return s[Math.floor(s.length / 2)] ?? 0;
}

function tableRow(call: PerCall): string {
	const out = (call.output ?? "").replace(/\s+/g, " ").trim().slice(0, 32);
	return `  ${call.npcId.padEnd(12)} prefill=${call.prefillMs.toFixed(1).padStart(8)}ms  wall=${call.wallMs.toFixed(1).padStart(8)}ms  output="${out}"`;
}

interface Verdict {
	tag: "PASS" | "PARTIAL" | "FAIL";
	rationale: string;
}

function verdictFor(yWallMedian: number, xWallMedian: number): Verdict {
	const savingsPct =
		xWallMedian > 0 ? ((xWallMedian - yWallMedian) / xWallMedian) * 100 : 0;
	if (savingsPct >= 50) {
		return {
			tag: "PASS",
			rationale: `Pattern Y first-tick wall ${yWallMedian.toFixed(0)} ms vs X's ${xWallMedian.toFixed(0)} ms — ${savingsPct.toFixed(0)}% savings. forkConversation delivers cross-conv prefix sharing as designed.`,
		};
	}
	if (savingsPct > 0) {
		return {
			tag: "PARTIAL",
			rationale: `Pattern Y first-tick wall is ${savingsPct.toFixed(1)}% faster than X. Fork delivers but overhead ate more than expected.`,
		};
	}
	return {
		tag: "FAIL",
		rationale: `Pattern Y first-tick wall is ${(-savingsPct).toFixed(1)}% SLOWER than X — the snapshot inheritance + load cost exceeds the prefill savings.`,
	};
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();

	const url = `http://localhost:8031/real-model.html?model=${MODEL_ID}&probe=prefix-cache-fork&fa=on&ingest=off&v=${Date.now()}`;
	console.log(`Driving prefix-cache fork probe: ${url}`);
	agentchrome(port, tab, ["navigate", url]);

	const probe = await fetchProbeResult(port, tab);

	const xPrefill = probe.patternX.map((r) => r.prefillMs);
	const yPrefill = probe.patternY.map((r) => r.prefillMs);
	const xWall = probe.patternX.map((r) => r.wallMs);
	const yWall = probe.patternY.map((r) => r.wallMs);

	const xPrefillMedian = median(xPrefill);
	const yPrefillMedian = median(yPrefill);
	const xWallMedian = median(xWall);
	const yWallMedian = median(yWall);

	const verdict = verdictFor(yWallMedian, xWallMedian);

	const wallSavedMs = xWallMedian - yWallMedian;
	const wallSavedPct = xWallMedian > 0 ? (wallSavedMs / xWallMedian) * 100 : 0;
	const totalSavingsAcrossNpcs =
		(xWallMedian - yWallMedian) * probe.patternY.length;

	const lines: string[] = [];
	lines.push("# forkConversation cross-conv prefix sharing — qwen3-8b-iq3m");
	lines.push("");
	lines.push(
		`**Model:** ${probe.model} (~1325-token shared system prefix; 4 NPCs × first-tick × 2 patterns; maxTokens=32; FA on)`,
	);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(
		`**Verdict:** **${verdict.tag}** — Pattern Y first-tick wall is ${wallSavedMs >= 0 ? "-" : "+"}${Math.abs(wallSavedMs).toFixed(0)} ms / ${wallSavedMs >= 0 ? "-" : "+"}${Math.abs(wallSavedPct).toFixed(1)}% vs Pattern X.`,
	);
	lines.push("");
	lines.push("## Headline (wall-time, first-tick-per-NPC)");
	lines.push("");
	lines.push(
		"| pattern | wall (median) | prefillMs (median) | savings vs X |",
	);
	lines.push("|---|---|---|---|");
	lines.push(
		`| X (baseline: \`createConversation\` per NPC) | ${xWallMedian.toFixed(1)} ms | ${xPrefillMedian.toFixed(1)} ms | — |`,
	);
	lines.push(
		`| Y (forked: \`forkConversation\` from primed base) | ${yWallMedian.toFixed(1)} ms | ${yPrefillMedian.toFixed(1)} ms | ${wallSavedMs >= 0 ? "−" : "+"}${Math.abs(wallSavedMs).toFixed(0)} ms / ${wallSavedMs >= 0 ? "−" : "+"}${Math.abs(wallSavedPct).toFixed(1)}% |`,
	);
	lines.push("");
	lines.push(
		`**Total wall-time saved across ${probe.patternY.length} NPCs:** ${(totalSavingsAcrossNpcs / 1000).toFixed(1)} s.`,
	);
	lines.push("");
	lines.push(`**Base prime tick:** ${probe.baseTickMs.toFixed(0)} ms (paid once, amortized across all forks).`);
	lines.push("");
	lines.push("## Why this matrix");
	lines.push("");
	lines.push(
		"Pattern X is the realistic naive baseline for spawning multiple agents that share a system prompt: each agent gets a fresh conversation, so the first call to each pays the full ~1325-token system prefill (~14.5 s on Qwen3-8B IQ3_M). With 4 NPCs spawned at session start, that's ~58 s of cumulative wall-time on agent boot.",
	);
	lines.push("");
	lines.push(
		"Pattern Y uses `forkConversation`: drive a base conversation through `[system]` once (the prime tick), then fork it per agent. Each fork inherits the base's KV snapshot via deep copy. The fork's first chatCompletion's longest-shared-token-prefix walk finds `sharedLen ≈ |system|` and prefills only the divergent agent-specific tail. Cost per NPC: ~`load(snapshot)` + `prefill(tail)` + `decode` ≈ snapshot reload + small.",
	);
	lines.push("");
	lines.push(
		"The pattern X / pattern Y separation includes a `resetConversation(modelHandleId)` between them so the engine's per-model session-tracker cache (`engine.ts`) doesn't carry pattern X's KV into pattern Y's base prime — that would mask the fork win by giving the base prime a session-tracker hit instead of a real prefill.",
	);
	lines.push("");
	lines.push("## Pattern X per-call detail (baseline)");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternX) lines.push(tableRow(c));
	lines.push("```");
	lines.push("");
	lines.push("## Pattern Y per-call detail (forked)");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternY) lines.push(tableRow(c));
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
		"- **Base prime tick** (paying the shared prefix once) is reported separately. For the fork pattern to be a net win, `baseTickMs + N × forkedWall < N × baselineWall`, i.e., the first NPC's effective cost is `baseTickMs + forkedWall` (one full prefill plus a fork load); each additional NPC pays only `forkedWall`.",
	);
	lines.push(
		"- **`?fa=on` is required.** Conversation handles only work in FA mode.",
	);

	mkdirSync(REPORT_DIR, { recursive: true });
	writeFileSync(`${REPORT_DIR}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(
		`${REPORT_DIR}/raw.json`,
		`${JSON.stringify(probe, null, 2)}\n`,
	);
	console.log(`\nVerdict: ${verdict.tag}`);
	console.log(
		`  Pattern X first-tick median wall: ${xWallMedian.toFixed(1)} ms (prefill ${xPrefillMedian.toFixed(1)} ms)`,
	);
	console.log(
		`  Pattern Y first-tick median wall: ${yWallMedian.toFixed(1)} ms (prefill ${yPrefillMedian.toFixed(1)} ms)`,
	);
	console.log(
		`  Base prime tick: ${probe.baseTickMs.toFixed(0)} ms (paid once)`,
	);
	console.log(`Report: ${REPORT_DIR}/SUMMARY.md`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
