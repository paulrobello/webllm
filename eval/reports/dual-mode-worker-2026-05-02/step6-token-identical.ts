/**
 * Step 6 of the dual-mode worker deployment closure: cross-mode token-identical
 * A/B with greedy sampling. Drives the smoke page through 5 prompts in both
 * worker=0 and worker=1 modes (temperature=0, topK=1 — deterministic greedy)
 * and compares the assistant text byte-for-byte.
 *
 * Run: bun run eval/reports/dual-mode-worker-2026-05-02/step6-token-identical.ts
 */

import {
	agentchrome,
	buildSmokeTestUrl,
	ensureSmokeServerReachable,
	resolveAgentchromeSession,
	waitForSmokeTestResult,
} from "../../browser-smoke.js";
import { getModelById } from "../../models.js";
import { writeFileSync } from "node:fs";

const PROMPTS = [
	"What is the capital of France?",
	"List the first three prime numbers.",
	"What is 7 plus 5?",
	"Name three primary colors.",
	"What sound does a dog make?",
] as const;

const MODEL_ID = "qwen3-0.6b-q4f16";

interface ProbeResult {
	prompt: string;
	mode: "main" | "worker";
	assistantText: string;
	tokensGenerated: number;
}

function evalStringResult(port: string, tab: string, script: string): string {
	const out = agentchrome(port, tab, ["js", "exec", script]);
	const resp = JSON.parse(out) as { result?: string };
	return resp.result ?? "";
}

async function captureAssistantText(
	port: string,
	tab: string,
): Promise<string> {
	// The smoke page logs each entry as a separate <div>. textContent
	// concatenates them with no separator, so `Assistant: ...` sits
	// directly after the prior `User: ...` line text. Find the
	// last `Assistant:` marker, then slice up to the next known
	// log marker (`[8/8]` or `[sys]` or end of string).
	const script = `(() => {
		const t = document.getElementById("log")?.textContent ?? "";
		const idx = t.lastIndexOf("Assistant:");
		if (idx < 0) return "";
		const start = idx + "Assistant:".length;
		const after = t.slice(start);
		// Stop at next log marker — pages append [8/8], [sys], [ingest] etc.
		const m = after.match(/\\[(?:[0-9]\\/[0-9]|sys|ingest|frameProbe)/);
		const stop = m ? m.index : after.length;
		return after.slice(0, stop).replace(/^\\s+/, "").replace(/\\s+$/, "");
	})()`;
	return evalStringResult(port, tab, script);
}

async function runProbe(
	port: string,
	tab: string,
	prompt: string,
	worker: boolean,
): Promise<ProbeResult> {
	const model = getModelById(MODEL_ID);
	if (!model) throw new Error(`Unknown model ${MODEL_ID}`);
	const url = buildSmokeTestUrl(model.id, model.contextLength, {
		extraParams: {
			perf: `${Date.now()}-${Math.floor(Math.random() * 1000)}`,
			temp: 0,
			topK: 1,
			topP: 1,
			rep: 1,
			max: 32,
			prompt,
			...(worker ? { worker: 1 } : {}),
		},
	});
	agentchrome(port, tab, ["navigate", url]);
	const result = await waitForSmokeTestResult(port, tab);
	const text = await captureAssistantText(port, tab);
	return {
		prompt,
		mode: worker ? "worker" : "main",
		assistantText: text,
		tokensGenerated: result.tokensGenerated,
	};
}

async function main(): Promise<void> {
	await ensureSmokeServerReachable();
	const { port, tab } = await resolveAgentchromeSession();
	const results: ProbeResult[] = [];
	for (const prompt of PROMPTS) {
		console.log(`\nPrompt: ${JSON.stringify(prompt)}`);
		const main = await runProbe(port, tab, prompt, false);
		console.log(`  main:   ${main.tokensGenerated} tok | ${JSON.stringify(main.assistantText.slice(0, 80))}`);
		results.push(main);
		const worker = await runProbe(port, tab, prompt, true);
		console.log(`  worker: ${worker.tokensGenerated} tok | ${JSON.stringify(worker.assistantText.slice(0, 80))}`);
		results.push(worker);
		const match = main.assistantText === worker.assistantText;
		console.log(`  byte-identical: ${match ? "YES" : "NO"}`);
	}
	let allMatch = true;
	const summary: string[] = ["# Step 6: Cross-mode token-identical A/B (greedy)\n"];
	summary.push(`Model: ${MODEL_ID}`);
	summary.push("Sampling: temp=0, topK=1, topP=1, rep=1, max=32 tokens");
	summary.push(`Prompts: ${PROMPTS.length}`);
	summary.push("");
	summary.push("| # | Prompt | Match | tokens(main/worker) |");
	summary.push("|---|---|:-:|:-:|");
	for (let i = 0; i < PROMPTS.length; i++) {
		const m = results[i * 2];
		const w = results[i * 2 + 1];
		const match = m.assistantText === w.assistantText;
		if (!match) allMatch = false;
		summary.push(
			`| ${i + 1} | ${JSON.stringify(m.prompt)} | ${match ? "YES" : "NO"} | ${m.tokensGenerated}/${w.tokensGenerated} |`,
		);
	}
	summary.push("");
	summary.push(`## Verdict: ${allMatch ? "PASS — all byte-identical" : "FAIL — divergence detected"}`);
	summary.push("");
	summary.push("## Per-prompt outputs\n");
	for (let i = 0; i < PROMPTS.length; i++) {
		const m = results[i * 2];
		const w = results[i * 2 + 1];
		summary.push(`### Prompt ${i + 1}: ${JSON.stringify(m.prompt)}`);
		summary.push("```");
		summary.push(`main:   ${m.assistantText}`);
		summary.push(`worker: ${w.assistantText}`);
		summary.push("```");
		summary.push("");
	}
	const outDir = "eval/reports/dual-mode-worker-2026-05-02";
	writeFileSync(`${outDir}/step6-results.md`, summary.join("\n") + "\n");
	writeFileSync(`${outDir}/step6-results.json`, JSON.stringify(results, null, 2) + "\n");
	console.log(`\nVerdict: ${allMatch ? "PASS" : "FAIL"}`);
	console.log(`Wrote: ${outDir}/step6-results.{md,json}`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
