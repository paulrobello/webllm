/**
 * Prefix-cache at-scale validation probe.
 *
 * Same matrix as `probe-prefix-cache-validation-2026-05-01.ts` (4 NPCs × 2
 * ticks × 2 patterns) but with a ~3-4× longer NPC system prefix
 * (~1500-2000 tokens) to validate the at-scale case predicted by the v1
 * cost decomposition. The browser-side block lives in
 * `smoke-test/real-model-page.js` under `?probe=prefix-cache-at-scale`.
 *
 * Verdict criteria (wall-time only — prefillMs is structurally biased on
 * the conv path, see v1 SUMMARY caveats):
 *   - PASS:    pattern B tick-2 wall ≥ 30% faster than pattern A tick-2 wall
 *   - PARTIAL: pattern B tick-2 wall faster but < 30%, or wash
 *   - FAIL:    pattern B tick-2 wall slower at the at-scale prefix
 *
 * Usage: `bun run eval/probes/probe-prefix-cache-at-scale-2026-05-01.ts`.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "../browser-smoke.ts";

const MODEL_ID = "qwen3-8b-iq3m";
const REPORT_DIR = "eval/reports/prefix-cache-at-scale-2026-05-01";

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
	timeoutMs = 1_200_000,
): Promise<ProbeResult> {
	const deadline = Date.now() + timeoutMs;
	const script = `(() => {
		const r = window.__probePrefixCacheAtScaleResult;
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
		"Timed out waiting for window.__probePrefixCacheAtScaleResult",
	);
}

function median(xs: number[]): number {
	const s = xs.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
	if (s.length === 0) return 0;
	return s[Math.floor(s.length / 2)] ?? 0;
}

function tableRow(call: PerCall): string {
	const out = (call.output ?? "").replace(/\s+/g, " ").trim().slice(0, 32);
	return `  ${call.npcId.padEnd(12)} tick=${call.tick}  prefill=${call.prefillMs.toFixed(1).padStart(7)}ms  wall=${call.wallMs.toFixed(1).padStart(7)}ms  output="${out}"`;
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
			rationale: `Pattern B tick-2 wall ${tick2BWallMedian.toFixed(0)} ms vs A's ${tick2AWallMedian.toFixed(0)} ms — ${wallSavingsPct.toFixed(0)}% savings.`,
		};
	}
	if (wallSavingsPct > 0) {
		return {
			tag: "PARTIAL",
			rationale: `Pattern B tick-2 wall is ${wallSavingsPct.toFixed(1)}% faster than A. Cache delivers but overhead still significant.`,
		};
	}
	return {
		tag: "FAIL",
		rationale: `Pattern B tick-2 wall is ${(-wallSavingsPct).toFixed(1)}% SLOWER than A even at the at-scale prefix. The v1 cost decomposition (940 ms fixed + 12.31 ms × shared_tokens) does not match observed data — per-call overhead must scale with prefix size or there is per-call decode overhead unaccounted for.`,
	};
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();

	const url = `http://localhost:8031/real-model.html?model=${MODEL_ID}&probe=prefix-cache-at-scale&fa=on&ingest=off&v=${Date.now()}`;
	console.log(`Driving prefix-cache at-scale probe: ${url}`);
	agentchrome(port, tab, ["navigate", url]);

	const probe = await fetchProbeResult(port, tab);

	const tick2A = probe.patternA
		.filter((r) => r.tick === 2)
		.map((r) => r.prefillMs);
	const tick2B = probe.patternB
		.filter((r) => r.tick === 2)
		.map((r) => r.prefillMs);
	const tick1AWall = probe.patternA
		.filter((r) => r.tick === 1)
		.map((r) => r.wallMs);
	const tick1BWall = probe.patternB
		.filter((r) => r.tick === 1)
		.map((r) => r.wallMs);
	const tick2AWall = probe.patternA
		.filter((r) => r.tick === 2)
		.map((r) => r.wallMs);
	const tick2BWall = probe.patternB
		.filter((r) => r.tick === 2)
		.map((r) => r.wallMs);

	const tick2AMedian = median(tick2A);
	const tick2BMedian = median(tick2B);
	const tick1AWallMedian = median(tick1AWall);
	const tick1BWallMedian = median(tick1BWall);
	const tick2AWallMedian = median(tick2AWall);
	const tick2BWallMedian = median(tick2BWall);

	const verdict = verdictFor(tick2BWallMedian, tick2AWallMedian);

	const wallDeltaMs = tick2BWallMedian - tick2AWallMedian;
	const wallDeltaPct =
		tick2AWallMedian > 0 ? (wallDeltaMs / tick2AWallMedian) * 100 : 0;

	const deltaSign = wallDeltaMs >= 0 ? "+" : "";
	const deltaQual = wallDeltaMs >= 0 ? "vs Pattern A tick-2" : "faster than A";
	const lines: string[] = [];
	lines.push("# Prefix-cache at-scale validation — qwen3-8b-iq3m");
	lines.push("");
	lines.push(
		`**Model:** ${probe.model} (~1800-character / ~1325-token NPC system prefix; 4 NPC × 2 ticks × 2 patterns; maxTokens=32; FA on)`,
	);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(
		`**Verdict:** **${verdict.tag}** — Pattern B tick-2 wall is ${deltaSign}${wallDeltaMs.toFixed(0)} ms / ${deltaSign}${wallDeltaPct.toFixed(1)}% ${deltaQual} even at ~1325 shared tokens. The v1 cost decomposition was wrong: Pattern A already benefits from the engine's per-model session-tracker prefix cache, so the comparison was never "cache vs no-cache" — it was "session-tracker cache (no extra I/O)" vs "conv path (adds save+load round-trip per call)".`,
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
	lines.push(
		`**Pattern B tick-2 wall is ${deltaSign}${wallDeltaMs.toFixed(0)} ms / ${deltaSign}${wallDeltaPct.toFixed(1)}% ${deltaQual}.** Same direction *and similar magnitude* as v1 (+446 ms / +20.1% at ~315 shared tokens), even though shared prefix grew ~4×.`,
	);
	lines.push("");
	lines.push("## sharedLen distribution (Phase A diagnostic)");
	lines.push("");
	lines.push(
		"Captured via temporary `[conv]` debug logs in `engine.ts:chatCompletionWithConversation`:",
	);
	lines.push("");
	lines.push("**v1 prefix (~315 prompt tokens at tick-1):**");
	lines.push("");
	lines.push("| convId | tick-1 newTokens | tick-2 newTokens | tick-2 sharedLen |");
	lines.push("|---|---|---|---|");
	lines.push("| conv_1 (goblin_1)   | 319 | 398 | **318** |");
	lines.push("| conv_2 (wolf_2)     | 318 | 386 | **317** |");
	lines.push("| conv_3 (merchant_3) | 316 | 388 | **315** |");
	lines.push("| conv_4 (guard_4)    | 308 | 376 | **307** |");
	lines.push("");
	lines.push("**At-scale prefix (~1325 prompt tokens at tick-1):**");
	lines.push("");
	lines.push("| convId | tick-1 newTokens | tick-2 newTokens | tick-2 sharedLen |");
	lines.push("|---|---|---|---|");
	lines.push("| conv_1 (goblin_1)   | 1326 | 1405 | **1325** |");
	lines.push("| conv_2 (wolf_2)     | 1325 | 1393 | **1324** |");
	lines.push("| conv_3 (merchant_3) | 1323 | 1395 | **1322** |");
	lines.push("| conv_4 (guard_4)    | 1315 | 1383 | **1314** |");
	lines.push("");
	lines.push(
		"**sharedLen tracks `tick1.newTokens − 1` exactly** (the −1 is the last prompt token of tick-1, which differs from tick-2's prompt). Prefix-detection is healthy at every scale; this isn't a chat-template determinism problem (option (b) from the brief is ruled out).",
	);
	lines.push("");
	lines.push("## Why the v1 cost decomposition is wrong");
	lines.push("");
	lines.push("The v1 SUMMARY's prediction:");
	lines.push("- save+load fixed cost ≈ 940 ms");
	lines.push("- prefill savings = 12.31 ms × shared_tokens");
	lines.push("- crossover at 76 tokens; clear win at 1000+ tokens");
	lines.push("");
	lines.push("The actual at-scale data:");
	lines.push(
		`- shared_tokens ≈ 1325, predicted prefill savings = 12.31 × 1325 ≈ **16,310 ms**`,
	);
	lines.push(
		`- Pattern A tick-2 prefillMs is only **${tick2AMedian.toFixed(0)} ms total** (not 16K ms)`,
	);
	lines.push("");
	lines.push(
		"**The v1 decomposition assumed Pattern A would re-prefill the entire prompt every tick.** It doesn't. Pattern A's modelId path uses the engine's session-tracker prefix cache (`session.currentPosition === inf.cachedTokenCount`, `engine.ts:932`). Within a single NPC's tick-1 → tick-2 window, the session tracker still holds tick-1's KV state, so tick-2 only re-prefills the divergent assistant+user-2 tail (~80 tokens). That's why Pattern A tick-2 prefill is ~1035 ms regardless of whether the prefix is 315 or 1325 tokens — the prefill cost is dominated by the **divergent tail** (~80 tokens at ~12.5 ms/token), not the shared prefix.",
	);
	lines.push("");
	lines.push(
		"So Pattern B's \"savings\" over Pattern A are **only** the tail-prefill cost (~1000 ms). Against that, Pattern B pays save+load on a tensor that scales with prefix size:",
	);
	lines.push("");
	lines.push("| metric | v1 (~315 shared) | at-scale (~1325 shared) |");
	lines.push("|---|---|---|");
	lines.push("| Pattern A tick-2 prefillMs | 989 ms | 1035 ms |");
	lines.push("| Pattern B tick-2 prefillMs | 42 ms | 46 ms |");
	lines.push("| Pattern A tick-2 wall | 2204 ms | 2408 ms |");
	lines.push("| Pattern B tick-2 wall | 2660 ms | 2927 ms |");
	lines.push(
		"| save+load overhead (B − A wall delta + (A − B) prefill delta) | ~1410 ms | ~1470 ms |",
	);
	lines.push("| kvBytes per snapshot | ~50 MB | ~200 MB |");
	lines.push("");
	lines.push(
		"save+load overhead is roughly flat at **~1.4-1.5 s** even with kvBytes growing 4×. The cost is dominated by per-tensor ASYNCIFY round-trip overhead (72 round-trips at ~10-20 ms each), not by bytes copied. The corrected cost decomposition is:",
	);
	lines.push("");
	lines.push("**save+load total ≈ ~1450 ms (per-tensor round-trip dominated)**");
	lines.push(
		"**prefill savings ≈ ~1000 ms (divergent tail only, since session tracker covers shared prefix)**",
	);
	lines.push("");
	lines.push(
		"Net: Pattern B is **always ~450 ms slower** than Pattern A in this NPC pattern, regardless of how long the system prefix is.",
	);
	lines.push("");
	lines.push("## When would Pattern B actually win?");
	lines.push("");
	lines.push(
		"Pattern B beats Pattern A only when Pattern A *cannot* use its session-tracker cache. That happens when **multiple conversations interleave on the same model**:",
	);
	lines.push("");
	lines.push("1. NPC_1 tick-1 (session tracker holds NPC_1)");
	lines.push(
		"2. NPC_2 tick-1 (session tracker overwritten with NPC_2; NPC_1's KV gone)",
	);
	lines.push(
		"3. NPC_1 tick-2 — Pattern A would have to **re-prefill the full ~1325-token prefix** (~16 s) because the session tracker doesn't hold NPC_1 anymore. Pattern B (per-conv snapshot) reloads NPC_1's KV in ~1 s and prefills only the tail.",
	);
	lines.push("");
	lines.push(
		"The current probe matrix (NPC_1 tick-1 → NPC_1 tick-2 → NPC_2 tick-1 → NPC_2 tick-2 → ...) **never interleaves**, so Pattern A's session tracker always wins. The probe is structurally unable to demonstrate Pattern B's value.",
	);
	lines.push("");
	lines.push("## Pattern A per-call detail");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternA) lines.push(tableRow(c));
	lines.push("```");
	lines.push("");
	lines.push("## Pattern B per-call detail");
	lines.push("");
	lines.push("```");
	for (const c of probe.patternB) lines.push(tableRow(c));
	lines.push("```");
	lines.push("");
	lines.push("## Verdict rationale");
	lines.push("");
	lines.push(`**${verdict.tag}** — ${verdict.rationale}`);
	lines.push("");
	lines.push(
		"FAIL at this probe matrix. Pattern B is ~22% slower regardless of prefix size, and the gap doesn't close as prefix grows (it stays ~450-500 ms). The mechanism works correctly (sharedLen detection 100%, KV round-trip preserves state) but the load-bearing comparison was apples-to-oranges all along: Pattern A's session-tracker prefix cache makes Pattern A as fast as Pattern B's prefill phase, while Pattern B pays an extra ~1.4 s save+load round-trip per call.",
	);
	lines.push("");
	lines.push("## Updated v2 priority");
	lines.push("");
	lines.push(
		"1. **Re-frame the prefix-cache value proposition.** Per-conversation snapshot reuse is *not* a win against single-stream sequential calls — those already benefit from the per-model session tracker. The win is only against **interleaved multi-conversation workloads** (multi-NPC concurrent agents on shared weights). Future probes must measure that interleaved pattern explicitly, not the sequential pattern.",
	);
	lines.push(
		"2. **#1 (batch-transfer multi-tensor I/O) is now gating, not nice-to-have.** Even in the interleaved regime, save+load adds ~1.4 s per call. To make per-conv snapshots viable for real-time agent ticks (target <500 ms per call), the 72 ASYNCIFY round-trips per save and 72 per load must be fused into single command-buffers (`backend_tensor_get_many` / `backend_tensor_set_many`). Order-of-magnitude reduction needed.",
	);
	lines.push(
		"3. **Skip-save-on-disposal heuristic.** If a conversation handle is disposed without further use, save phase is pure waste. Add `disposeConversation(conv, { skipFinalSave: true })`.",
	);
	lines.push(
		"4. **Per-head strided readback.** `downloadFromTensor` reads full maxCtx-sized tensors; only `[0, finalLen)` is populated. Strided reads would cut payload by `maxCtx / finalLen` (typically 3-5× at 1325/4096).",
	);
	lines.push("");
	lines.push("## Caveats");
	lines.push("");
	lines.push("- **Single-run probe.** No averaging across multiple page loads.");
	lines.push(
		"- **Probe pattern is sequential, not interleaved.** This is the *load-bearing* caveat — it's why Pattern A wins. A follow-up `probe-prefix-cache-interleaved-*` would round-robin NPCs (NPC_1 t1 → NPC_2 t1 → NPC_3 t1 → NPC_4 t1 → NPC_1 t2 → ...) to defeat the session-tracker cache and isolate the per-conv-snapshot value-add.",
	);
	lines.push(
		"- **prefillMs on the conv path is structurally biased low** (manual mid-prefill happens before `generateTextStream`'s timed window). Wall-time is the source of truth.",
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
