/**
 * Probe 9b — Batched-prompt vs sequential N-NPC scaling (queued
 * 2026-05-01).
 *
 * For an N=4 NPC scenario, drives `qwen3-8b-iq3m` through two
 * decision patterns at matched token budgets:
 *   (i) N sequential `chatCompletion` calls each deciding one NPC
 *  (ii) one `chatCompletion` call with all N observations, asked
 *       for `[{npc_id, action}, ...]`
 *
 * Both scenarios run on the same warm engine in a single page load
 * (sequential first, then batched after a 500 ms settle). Page logic
 * lives in `smoke-test/real-model-page.js` behind `?probe=9b` and
 * posts `window.__probe9bResult` for this runner to scrape.
 *
 * Threshold: batched ≥0.7× tool-call quality at ≤0.4× total wall
 * time vs sequential. Decision: which scaling pattern is canonical
 * for the agent harness.
 *
 * Usage: `bun run eval/probes/probe-9b-batched-vs-seq.ts`. Writes a
 * SUMMARY.md to `eval/reports/probe-9b-2026-05-01/`.
 */

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
const TOOLS = ["move", "speak", "attack", "use_item", "trade"];

interface SequentialCall {
	npcId: string;
	wallMs: number;
	prefillMs: number;
	genTokens: number;
	output: string;
}

interface Probe9bResult {
	model: string;
	sequential: { totalWallMs: number; perCall: SequentialCall[] };
	batched: {
		wallMs: number;
		prefillMs: number;
		genTokens: number;
		output: string;
	};
}

function sleep(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

async function fetch9bResult(
	port: string,
	tab: string,
): Promise<Probe9bResult> {
	const deadline = Date.now() + 360_000;
	const script = `(() => {
		const r = window.__probe9bResult;
		return r ? JSON.stringify(r) : "";
	})()`;
	while (Date.now() < deadline) {
		const out = agentchrome(port, tab, ["js", "exec", script]);
		const parsed = JSON.parse(out) as { result?: string; type?: string };
		if (parsed.type === "string" && parsed.result) {
			return JSON.parse(parsed.result) as Probe9bResult;
		}
		await sleep(1500);
	}
	throw new Error("Timed out waiting for window.__probe9bResult");
}

function scoreSequential(seq: Probe9bResult["sequential"]): {
	correct: number;
	perCall: { npcId: string; output: string; matched: string | null }[];
} {
	const perCall = seq.perCall.map((r) => {
		const lower = r.output.toLowerCase();
		const matched = TOOLS.find((t) => lower.includes(t)) ?? null;
		return { npcId: r.npcId, output: r.output, matched };
	});
	const correct = perCall.filter((c) => c.matched !== null).length;
	return { correct, perCall };
}

function scoreBatched(batched: Probe9bResult["batched"]): {
	correct: number;
	parsed: { npc_id?: string; action?: string }[];
	rawOutput: string;
} {
	const text = batched.output;
	// Try strict JSON first. Fall back to regex extraction for the
	// `"action": "<tool>"` pattern if the model emits malformed JSON.
	let parsed: { npc_id?: string; action?: string }[] = [];
	const arrayStart = text.indexOf("[");
	const arrayEnd = text.lastIndexOf("]");
	if (arrayStart >= 0 && arrayEnd > arrayStart) {
		const slice = text.slice(arrayStart, arrayEnd + 1);
		try {
			parsed = JSON.parse(slice);
		} catch {
			parsed = [];
		}
	}
	if (parsed.length === 0) {
		const re = /"action"\s*:\s*"([^"]+)"/g;
		const matches: { action: string }[] = [];
		let m: RegExpExecArray | null;
		while ((m = re.exec(text)) !== null) {
			matches.push({ action: m[1] ?? "" });
		}
		parsed = matches;
	}
	const correct = parsed.filter((p) => {
		const a = (p.action ?? "").toLowerCase();
		return TOOLS.some((t) => a.includes(t));
	}).length;
	return { correct, parsed, rawOutput: text };
}

async function main(): Promise<void> {
	const model = getModelById(MODEL_ID);
	if (!model) throw new Error(`Unknown model: ${MODEL_ID}`);

	await ensureSmokeServerReachable();
	await ensureModelDownloaded(model);
	const { port, tab } = await resolveAgentchromeSession();

	const url = buildSmokeTestUrl(MODEL_ID, model.contextLength, {
		extraParams: {
			perf: `${Date.now()}-9b`,
			probe: "9b",
			ingest: "off",
		},
	});
	console.log(`Driving smoke page: ${MODEL_ID}…`);
	agentchrome(port, tab, ["navigate", url]);
	// Wait for the [7/8] smoke result first (smoke warmup), then the
	// in-page probe-9b runner posts window.__probe9bResult.
	await waitForSmokeTestResult(port, tab);
	const probe = await fetch9bResult(port, tab);

	const seqScore = scoreSequential(probe.sequential);
	const batScore = scoreBatched(probe.batched);

	const seqQuality = seqScore.correct / probe.sequential.perCall.length;
	const batQuality = batScore.correct / probe.sequential.perCall.length;
	const wallRatio = probe.batched.wallMs / probe.sequential.totalWallMs;
	const qualityRatio = batQuality === 0 ? 0 : batQuality / Math.max(seqQuality, 1e-6);

	const lines: string[] = [];
	lines.push("# Probe 9b — Batched-prompt vs sequential");
	lines.push("");
	lines.push(`**Model:** ${MODEL_ID}`);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(`**N (NPCs):** ${probe.sequential.perCall.length}`);
	lines.push("");
	lines.push("## Method");
	lines.push("");
	lines.push(
		"Single warm engine. Run sequential pattern first (4 separate `chatCompletion` calls, one per NPC, max 8 tokens each), then a 500ms settle, then the batched pattern (1 `chatCompletion` with all 4 observations, max 96 tokens, asked for a JSON array of `{npc_id, action}`).",
	);
	lines.push("");
	lines.push(
		"Correctness: each NPC's response counts as correct if its output (sequential: per-call output; batched: parsed JSON `action` field) contains a tool name from the allowed set {move, speak, attack, use_item, trade}.",
	);
	lines.push("");
	lines.push("## Sequential — per-call results");
	lines.push("");
	lines.push("| NPC | wall (ms) | prefill (ms) | genTokens | output | matched |");
	lines.push("|---|---:|---:|---:|---|---|");
	for (let i = 0; i < probe.sequential.perCall.length; i++) {
		const r = probe.sequential.perCall[i];
		const s = seqScore.perCall[i];
		if (!r || !s) continue;
		const out = r.output.replace(/\n/g, " ").slice(0, 60);
		lines.push(
			`| ${r.npcId} | ${r.wallMs.toFixed(0)} | ${r.prefillMs.toFixed(0)} | ${r.genTokens} | ${JSON.stringify(out)} | ${s.matched ?? "—"} |`,
		);
	}
	lines.push("");
	lines.push(
		`**Sequential total wall (excluding settles):** ${probe.sequential.totalWallMs.toFixed(0)} ms`,
	);
	lines.push(`**Sequential correct:** ${seqScore.correct} / ${probe.sequential.perCall.length}`);
	lines.push("");
	lines.push("## Batched — single-call result");
	lines.push("");
	lines.push(
		`- wall: **${probe.batched.wallMs.toFixed(0)} ms** · prefill: ${probe.batched.prefillMs.toFixed(0)} ms · genTokens: ${probe.batched.genTokens}`,
	);
	lines.push("");
	lines.push("Raw output:");
	lines.push("");
	lines.push("```");
	lines.push(probe.batched.output.slice(0, 800));
	lines.push("```");
	lines.push("");
	lines.push("Parsed entries:");
	lines.push("");
	lines.push("```json");
	lines.push(JSON.stringify(batScore.parsed, null, 2));
	lines.push("```");
	lines.push("");
	lines.push(`**Batched correct:** ${batScore.correct} / ${probe.sequential.perCall.length}`);
	lines.push("");
	lines.push("## Headline ratios");
	lines.push("");
	lines.push(
		`- Sequential quality: ${(seqQuality * 100).toFixed(0)}% · Batched quality: ${(batQuality * 100).toFixed(0)}%`,
	);
	lines.push(
		`- Quality ratio (batched / sequential): ${qualityRatio.toFixed(2)} (threshold ≥0.70)`,
	);
	lines.push(
		`- Wall ratio (batched / sequential): ${wallRatio.toFixed(2)} (threshold ≤0.40)`,
	);
	lines.push("");
	lines.push("## Verdict");
	lines.push("");
	const passQ = qualityRatio >= 0.7;
	const passW = wallRatio <= 0.4;
	if (passQ && passW) {
		lines.push(
			"**PASS** — batched dispatch beats both thresholds. Single-call multi-NPC dispatch is canonical for the agent harness; it's both materially faster (≤0.4× wall) and quality-neutral or better (≥0.7× of sequential's correctness).",
		);
	} else if (!passQ && passW) {
		lines.push(
			"**FAIL on quality** — batched is fast enough but loses too much per-NPC correctness. Decision: keep sequential as canonical; revisit batched only if a structured-output enforcer (constrained decoding) closes the quality gap.",
		);
	} else if (passQ && !passW) {
		lines.push(
			"**FAIL on wall** — batched maintains quality but doesn't save the projected ≥60% wall time. Decision: sequential remains canonical; the per-call structural overhead (KV-cache reset etc., per probe 9c) is the dominant cost rather than per-token decode.",
		);
	} else {
		lines.push(
			"**FAIL on both** — batched is neither materially faster nor quality-neutral. Decision: sequential is the canonical pattern.",
		);
	}

	const outDir = "eval/reports/probe-9b-2026-05-01";
	mkdirSync(outDir, { recursive: true });
	writeFileSync(`${outDir}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(`${outDir}/raw.json`, `${JSON.stringify(probe, null, 2)}\n`);
	console.log(`\nReport: ${outDir}/SUMMARY.md`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
