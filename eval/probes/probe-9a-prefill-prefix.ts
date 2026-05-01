/**
 * Probe 9a — Prefill prefix-cache decomposition (queued 2026-05-01).
 *
 * Drives `qwen3-8b-iq3m` through three NPC-shaped prompts that share an
 * explicit (PREFIX, TAIL) split, captures prefillMs and tokensIn for
 * each, and fits `prefillMs = a·prefix_tokens + b·tail_tokens + c` to
 * answer: does prefix caching meaningfully shorten an NPC-tick
 * prefill? Pass-meaningful threshold: prefix tokens contribute ≥50% of
 * prefill latency at the canonical NPC prompt size (P=400, T=40).
 *
 * Usage: `bun run eval/probes/probe-9a-prefill-prefix.ts`. Writes a
 * SUMMARY.md to `eval/reports/probe-9a-2026-05-01/`.
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
import { LONG_PROMPTS } from "../fixtures/long-prompts.ts";
import { getModelById } from "../models.ts";

const MODEL_ID = "qwen3-8b-iq3m";
const RUNS_PER_FIXTURE = 3;
const FIXTURES = [
	"probe9a-Pshort-Tshort",
	"probe9a-Pshort-Tlong",
	"probe9a-Plong-Tshort",
] as const;

interface FixtureRun {
	fixture: (typeof FIXTURES)[number];
	tokensIn: number;
	prefillMs: number;
}

async function main(): Promise<void> {
	const model = getModelById(MODEL_ID);
	if (!model) throw new Error(`Unknown model: ${MODEL_ID}`);

	await ensureSmokeServerReachable();
	await ensureModelDownloaded(model);
	const { port, tab } = await resolveAgentchromeSession();

	const runs: FixtureRun[] = [];
	for (const fixture of FIXTURES) {
		const promptText = LONG_PROMPTS[fixture];
		if (!promptText) throw new Error(`Missing fixture text: ${fixture}`);
		for (let i = 0; i < RUNS_PER_FIXTURE; i++) {
			process.stdout.write(`${fixture} run ${i + 1}/${RUNS_PER_FIXTURE}...`);
			const url = buildSmokeTestUrl(model.id, model.contextLength, {
				extraParams: {
					perf: `${Date.now()}-${i}`,
					prompt: promptText,
					max: 16,
					ingest: "off",
				},
			});
			agentchrome(port, tab, ["navigate", url]);
			const r = await waitForSmokeTestResult(port, tab);
			if (r.tokensIn === undefined) {
				throw new Error(
					"Smoke result missing tokensIn — make sure smoke-test/real-model-page.js logs `tokensIn=N` and the regex captures it.",
				);
			}
			runs.push({ fixture, tokensIn: r.tokensIn, prefillMs: r.prefillMs });
			process.stdout.write(
				` tokensIn=${r.tokensIn}, prefill=${r.prefillMs}ms\n`,
			);
		}
	}

	// Compute per-fixture median (prefillMs) and observed tokensIn.
	const byFixture = new Map<
		string,
		{ tokensIn: number; medianPrefillMs: number; samples: number[] }
	>();
	for (const fixture of FIXTURES) {
		const fr = runs.filter((r) => r.fixture === fixture);
		const sorted = fr.map((r) => r.prefillMs).sort((a, b) => a - b);
		const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
		const tokensIn = fr[0]?.tokensIn ?? 0;
		byFixture.set(fixture, {
			tokensIn,
			medianPrefillMs: median,
			samples: sorted,
		});
	}

	// Decompose each fixture's tokensIn into (prefix_tokens, tail_tokens).
	// We don't know the exact prefix-vs-tail split per the BPE tokenizer,
	// but the PREFIX_LONG construction extends PREFIX_SHORT by appending
	// extra text — so the *delta* between Plong-Tshort and Pshort-Tshort
	// is purely the prefix-extension token count. Same logic for tails.
	const ss = byFixture.get("probe9a-Pshort-Tshort");
	const sl = byFixture.get("probe9a-Pshort-Tlong");
	const ls = byFixture.get("probe9a-Plong-Tshort");
	if (!ss || !sl || !ls) throw new Error("Missing fixture data");

	const prefixDeltaTokens = ls.tokensIn - ss.tokensIn;
	const tailDeltaTokens = sl.tokensIn - ss.tokensIn;
	const prefixDeltaMs = ls.medianPrefillMs - ss.medianPrefillMs;
	const tailDeltaMs = sl.medianPrefillMs - ss.medianPrefillMs;

	const a = prefixDeltaMs / prefixDeltaTokens; // ms per prefix-extension token
	const b = tailDeltaMs / tailDeltaTokens; // ms per tail-extension token

	// At the SS fixture, prefillMs = a·tokensIn_ss + 0 + c (treating SS
	// as the baseline with prefix_short, tail_short collapsed). We can't
	// cleanly separate the baseline-prefix contribution from c using
	// only differences — so we report a/b as marginal token costs and
	// use them to reason about extrapolation rather than absolute share.
	// More robust: use the SS fixture's per-token marginal (assume a≈b
	// across the prompt is consistent) and report the share at typical
	// NPC prompt sizes.

	const NPC_PREFIX = 400;
	const NPC_TAIL = 40;
	const projectedPrefillMs = a * NPC_PREFIX + b * NPC_TAIL;
	const prefixShare =
		(a * NPC_PREFIX) / (a * NPC_PREFIX + b * NPC_TAIL);

	const lines: string[] = [];
	lines.push("# Probe 9a — Prefill prefix-cache decomposition");
	lines.push("");
	lines.push(`**Model:** ${MODEL_ID}`);
	lines.push(`**Date:** ${new Date().toISOString().slice(0, 10)}`);
	lines.push(`**Runs per fixture:** ${RUNS_PER_FIXTURE}`);
	lines.push("");
	lines.push("## Per-fixture medians");
	lines.push("");
	lines.push("| Fixture | tokensIn | prefill samples (ms) | median (ms) |");
	lines.push("|---|---:|---|---:|");
	for (const fixture of FIXTURES) {
		const entry = byFixture.get(fixture);
		if (!entry) continue;
		lines.push(
			`| ${fixture} | ${entry.tokensIn} | ${entry.samples.join(", ")} | ${entry.medianPrefillMs} |`,
		);
	}
	lines.push("");
	lines.push("## Marginal token costs");
	lines.push("");
	lines.push(
		`- Prefix delta: ${prefixDeltaTokens} tokens → ${prefixDeltaMs.toFixed(1)} ms ⇒ **a = ${a.toFixed(3)} ms / prefix-token**`,
	);
	lines.push(
		`- Tail delta:   ${tailDeltaTokens} tokens → ${tailDeltaMs.toFixed(1)} ms ⇒ **b = ${b.toFixed(3)} ms / tail-token**`,
	);
	lines.push(
		`- Ratio b/a = ${(b / a).toFixed(2)} (1.0 means prefill is purely linear in total tokens; >1 would mean tail tokens are more expensive — e.g. a context-length-superlinear effect; <1 unlikely)`,
	);
	lines.push("");
	lines.push("## Projection at canonical NPC prompt (P=400, T=40)");
	lines.push("");
	lines.push(
		`- Predicted prefill: ${projectedPrefillMs.toFixed(0)} ms (a·400 + b·40, ignoring constant c).`,
	);
	lines.push(
		`- Prefix's share of prefill latency: **${(prefixShare * 100).toFixed(1)}%**`,
	);
	lines.push("");
	lines.push("## Verdict");
	lines.push("");
	if (prefixShare >= 0.5) {
		lines.push(
			"**PASS** — prefix accounts for ≥50% of the canonical NPC-tick prefill cost.",
		);
		lines.push(
			"KV-cache-per-conversation-on-shared-weights multiplexing (currently deferred per CLAUDE.md) becomes a load-bearing work item for the NPC harness: prefix caching could roughly halve per-tick prefill latency.",
		);
	} else {
		lines.push(
			"**FAIL** — prefix accounts for <50% of canonical NPC-tick prefill.",
		);
		lines.push(
			"Prefix caching would not meaningfully shorten ticks; KV-cache multiplexing remains correctly deferred.",
		);
	}

	const outDir = "eval/reports/probe-9a-2026-05-01";
	mkdirSync(outDir, { recursive: true });
	writeFileSync(`${outDir}/SUMMARY.md`, `${lines.join("\n")}\n`);
	writeFileSync(
		`${outDir}/raw-runs.json`,
		`${JSON.stringify(runs, null, 2)}\n`,
	);
	console.log(`\nReport: ${outDir}/SUMMARY.md`);
}

main().catch((err) => {
	console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
	process.exit(1);
});
