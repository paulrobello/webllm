/**
 * Probe 9d — Worker-resident hitch (queued 2026-05-01).
 *
 * Spike: drive a Worker-resident inference path with the same
 * multi-call scheme used by the main-thread frame-probe (probe 9c
 * baseline). Measure per-call decode_max from the main thread's
 * rAF loop while the engine runs entirely off-thread. If the
 * deterministic ~42 ms decode hitch from probe 9c persists in the
 * main-thread render loop, the hitch is structural per-call work
 * (postMessage + JS scheduling jitter), not GPU-queue contention
 * with main-thread WASM compute. If the hitch disappears, item 10
 * (dual-mode worker support) is load-bearing.
 *
 * Compares against:
 *   - probe-9c control's call-0 decode_max ≈ 41.7 ms (main-thread
 *     engine, same multi-call shape, slightly different model).
 *
 * Model: `qwen3-0.6b-q4f16` (610 MB GGUF — fits a single ArrayBuffer
 * fetch; uses public `WebLLM.loadModelFromBuffer`). qwen3-8b-iq3m
 * isn't worker-portable yet without rebuilding the smoke page's
 * heap-streaming loader inside the worker; this spike answers the
 * structural question with a smaller model rather than spending
 * additional cycles on the loader port.
 *
 * Usage: `bun run eval/probes/probe-9d-worker-hitch.ts`. Writes
 * SUMMARY.md to `eval/reports/probe-9d-2026-05-01/`.
 */

import { mkdirSync, writeFileSync } from "node:fs";
import {
	agentchrome,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
} from "../browser-smoke.ts";

const MODEL_ID = "qwen3-0.6b-q4f16";
const NUM_CALLS = 5;
const CONTEXT_LENGTH = 4096;

interface PerCallStat {
	index: number;
	prefill: { max?: number; p95?: number };
	decode: { max?: number; p95?: number };
}

interface Probe9dResult {
	model: string;
	callCount: number;
	frameStats: { perCall?: PerCallStat[] };
}

function sleep(ms: number): Promise<void> {
	return new Promise((r) => setTimeout(r, ms));
}

async function fetchProbeResult(
	port: string,
	tab: string,
): Promise<Probe9dResult> {
	const deadline = Date.now() + 600_000;
	const script = `(() => {
		const r = window.__probe9dResult;
		return r ? JSON.stringify(r) : "";
	})()`;
	while (Date.now() < deadline) {
		const out = agentchrome(port, tab, ["js", "exec", script]);
		const parsed = JSON.parse(out) as { result?: string; type?: string };
		if (parsed.type === "string" && parsed.result) {
			return JSON.parse(parsed.result) as Probe9dResult;
		}
		await sleep(2000);
	}
	throw new Error("Timed out waiting for window.__probe9dResult");
}

function formatPerCall(perCall?: PerCallStat[]): string {
	if (!perCall || perCall.length === 0) return "(no per-call stats)";
	return perCall
		.map((c) => {
			const d = c.decode.max ?? 0;
			const dP95 = c.decode.p95 ?? 0;
			const p = c.prefill.max ?? 0;
			return `  call ${c.index}: prefill_max=${p.toFixed(1)}ms, decode_p95=${dP95.toFixed(1)}ms, decode_max=${d.toFixed(1)}ms`;
		})
		.join("\n");
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();

	const url = `http://localhost:8031/probe-9d.html?model=${MODEL_ID}&ctx=${CONTEXT_LENGTH}&calls=${NUM_CALLS}&v=${Date.now()}`;
	console.log(`Driving probe-9d page (worker-resident engine)…`);
	agentchrome(port, tab, ["navigate", url]);

	const probe = await fetchProbeResult(port, tab);
	const perCall = probe.frameStats.perCall ?? [];
	const decodeMaxes = perCall
		.map((c) => c.decode.max ?? 0)
		.filter((v) => v > 0);
	const sorted = [...decodeMaxes].sort((a, b) => a - b);
	const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
	const min = sorted[0] ?? 0;
	const max = sorted[sorted.length - 1] ?? 0;

	const lines: string[] = [];
	lines.push("# Probe 9d — Worker-resident hitch");
	lines.push("");
	lines.push(`**Model:** ${MODEL_ID} (worker-resident)`);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(`**Calls:** ${NUM_CALLS}`);
	lines.push("");
	lines.push("## Method");
	lines.push("");
	lines.push(
		"Standalone main page `smoke-test/probe-9d.html` boots a module Worker (`probe-9d-worker.js`) that imports `./webllm-bundle.js`, fetches the GGUF as an ArrayBuffer, and instantiates the engine via `WebLLM.loadModelFromBuffer`. The main thread runs the frame-probe rAF tracker (idle cube, 120 Hz baseline target) and dispatches 5 sequential `chatCompletion` calls to the worker via `postMessage`. The Worker collects all stream chunks internally and posts a single `chat-done` reply (so the rAF loop measures the postMessage round-trip + engine work without per-chunk message traffic distorting the timing).",
	);
	lines.push("");
	lines.push(
		"Probe 9c's control (main-thread engine, qwen3-8b-iq3m) measured deterministic per-call `decode_max ≈ 41.7 ms`. If a Worker move absorbs the hitch, this probe should land call-0..N decode_max in the 8-12 ms band that matches the rAF baseline.",
	);
	lines.push("");
	lines.push("## Per-call rAF stats (main-thread perspective)");
	lines.push("");
	lines.push("```");
	lines.push(formatPerCall(perCall));
	lines.push("```");
	lines.push("");
	lines.push("## Headline");
	lines.push("");
	lines.push(`- decode_max min: **${min.toFixed(1)} ms**`);
	lines.push(`- decode_max median: **${median.toFixed(1)} ms**`);
	lines.push(`- decode_max max: **${max.toFixed(1)} ms**`);
	lines.push("");
	lines.push("## Verdict");
	lines.push("");
	if (median <= 12) {
		lines.push(
			"**PASS** — main-thread render loop's per-call decode_max is in the rAF baseline band. Moving the engine to a Worker absorbs the structural hitch.",
		);
		lines.push(
			"Decision: item 10 (dual-mode main + worker deployment) is the load-bearing path forward for ≥1 Hz NPC tick rates with concurrent Three.js rendering. Spec needed for the postMessage bridge, KV-cache lifecycle, and embedder/chat parity.",
		);
	} else if (median <= 25) {
		lines.push(
			"**PARTIAL** — Worker reduces but doesn't eliminate the hitch. Some component of the per-call cost lives outside the engine work (e.g., Chrome's GPU process scheduling between the worker and main thread, postMessage queue draining), or the smaller model masks part of the structural cost that an 8B model would surface.",
		);
		lines.push(
			"Decision: dual-mode is still net-win but won't be a clean fix. Run an 8B-aware spike (requires porting the smoke page's heap-streaming loader into the worker) to confirm the residual at scale.",
		);
	} else {
		lines.push(
			"**FAIL** — Worker move does not absorb the hitch from the main-thread render loop's perspective. The per-call ~42 ms cost survives a thread move, meaning it's structural in WebGPU command-queue scheduling or main-thread postMessage handling rather than JS-thread compute.",
		);
		lines.push(
			"Decision: item 10 (dual-mode worker) is NOT the right intervention for the NPC + Three.js coexistence case. Investigate alternatives: WebGPU command-queue priorities, frame-pacing changes, OffscreenCanvas to move rendering off main, or accepting the hitch as a fixed cost and budgeting tick rate around it.",
		);
	}
	lines.push("");
	lines.push("## Caveats");
	lines.push("");
	lines.push(
		"- **Smaller model than probe 9c.** `qwen3-0.6b-q4f16` was used for spike tractability; `qwen3-8b-iq3m` (where the hitch was originally measured) requires the smoke page's heap-streaming loader inside the worker, which is a substantial separate piece of work. The structural hitch is per-call (independent of model size at probe-9c's measurement), so the smaller model should answer the structural question — but if the verdict above is borderline, an 8B re-run is the next step.",
	);
	lines.push(
		"- **Single-run probe.** No averaging across multiple page loads. Frame-stats are derived from N=5 calls in one session.",
	);
	lines.push(
		"- **Engine init in worker is cold.** Worker model load + WebGPU adapter acquisition happens during the page load, not during the timed probe. The frame-probe baseline only starts after `init-done` fires.",
	);
	lines.push(
		"- **No baseline-control comparison in this report.** The 41.7 ms reference comes from probe 9c on qwen3-8b-iq3m; there's an apples-to-oranges component here. To make the comparison fully clean, run a control with qwen3-0.6b-q4f16 on the main-thread `?frameProbeCalls=5` mode and use that as the reference. Queued as follow-up if the verdict here is PASS or PARTIAL.",
	);

	const outDir = "eval/reports/probe-9d-2026-05-01";
	mkdirSync(outDir, { recursive: true });
	writeFileSync(`${outDir}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(`${outDir}/raw.json`, `${JSON.stringify(probe, null, 2)}\n`);
	console.log(`\nReport: ${outDir}/SUMMARY.md`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
