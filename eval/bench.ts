/**
 * Combined bench driver: runs the browser-driven speed pass AND the offline
 * accuracy pass against the same profile, tagged with a shared `benchId` so
 * both halves show up side-by-side on the live dashboard.
 *
 * Speed (real WebGPU) → eval/chat-smoke.ts → run_* events
 * Accuracy (offline via Character) → eval/cli.ts → eval_* events
 */
import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import { LIVE_BENCH_URL_ENV, resolveLiveBenchUrl } from "./live-client.js";
import {
	getSmokeProfile,
	getSmokeProfileSet,
	listSmokeProfiles,
	listSmokeProfileSets,
	type SmokeProfile,
} from "./smoke-profiles.js";

interface BenchCase {
	profile: SmokeProfile;
	benchId: string;
}

function expandProfileArg(arg: string): string[] {
	const items = arg
		.split(",")
		.map((s) => s.trim())
		.filter(Boolean);
	const expanded: string[] = [];
	for (const item of items) {
		const set = getSmokeProfileSet(item);
		if (set) {
			for (const name of set) expanded.push(name);
		} else {
			expanded.push(item);
		}
	}
	return Array.from(new Set(expanded));
}

function main(): void {
	const { values } = parseArgs({
		options: {
			profiles: { type: "string", short: "p" },
			dimension: { type: "string", short: "d" },
			"skip-speed": { type: "boolean" },
			"skip-accuracy": { type: "boolean" },
			"live-bench-url": { type: "string" },
			"list-profiles": { type: "boolean" },
			"fail-fast": { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	if (values["list-profiles"]) {
		console.log("Profiles:");
		for (const name of listSmokeProfiles()) console.log(`  ${name}`);
		console.log("\nProfile sets:");
		for (const name of listSmokeProfileSets()) console.log(`  ${name}`);
		process.exit(0);
	}

	if (!values.profiles) {
		console.error(
			"Error: --profiles is required (comma-separated profile names or set names)",
		);
		printUsage();
		process.exit(1);
	}

	const names = expandProfileArg(values.profiles);
	if (names.length === 0) {
		console.error("Error: --profiles expanded to an empty list");
		process.exit(1);
	}

	const cases: BenchCase[] = names.map((name) => {
		const profile = getSmokeProfile(name);
		if (!profile) {
			console.error(
				`Error: unknown profile "${name}". Available: ${listSmokeProfiles().join(", ")}`,
			);
			process.exit(1);
		}
		return {
			profile,
			benchId: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
		};
	});

	const liveUrl = resolveLiveBenchUrl(values["live-bench-url"]);
	const childEnv = { ...process.env };
	if (liveUrl) childEnv[LIVE_BENCH_URL_ENV] = liveUrl;

	const results: Array<{ label: string; ok: boolean; phase: string }> = [];
	const doSpeed = !values["skip-speed"];
	const doAccuracy = !values["skip-accuracy"];
	const failFast = values["fail-fast"] ?? false;

	for (const entry of cases) {
		const label = entry.profile.name;
		console.log(`\n═══ ${label} ═══`);

		if (doSpeed) {
			console.log(`\n--- speed: ${label} ---`);
			const speedArgs = [
				"run",
				"eval/chat-smoke.ts",
				"--profile",
				entry.profile.name,
			];
			const ok = runChild(speedArgs, childEnv);
			results.push({ label, ok, phase: "speed" });
			if (!ok && failFast) break;
		}

		if (doAccuracy) {
			console.log(`\n--- accuracy: ${label} ---`);
			// Prefer the browser-backed path (real WebGPU per task) when the
			// dashboard backend is available; fall back to the offline CLI
			// otherwise (which will return no-engine errors per task, honest
			// but not useful).
			const accArgs = liveUrl
				? [
						"run",
						"eval/browser-eval.ts",
						"--profile",
						entry.profile.name,
						"--live-bench-url",
						liveUrl,
					]
				: [
						"run",
						"eval/cli.ts",
						"--profile",
						entry.profile.name,
						"--eval-id",
						entry.benchId,
					];
			if (values.dimension) accArgs.push("--dimension", values.dimension);
			const ok = runChild(accArgs, childEnv);
			results.push({ label, ok, phase: "accuracy" });
			if (!ok && failFast) break;
		}
	}

	const failed = results.filter((r) => !r.ok).length;
	console.log("\nBench summary");
	console.log(`Total:   ${results.length}`);
	console.log(`Passed:  ${results.length - failed}`);
	console.log(`Failed:  ${failed}`);
	for (const r of results) {
		console.log(`  [${r.ok ? "PASS" : "FAIL"}] ${r.label} · ${r.phase}`);
	}
	if (failed > 0) process.exit(1);
}

function runChild(args: string[], env: NodeJS.ProcessEnv): boolean {
	try {
		execFileSync("bun", args, { stdio: "inherit", env });
		return true;
	} catch {
		return false;
	}
}

function printUsage(): void {
	console.log(`Usage: bun run eval/bench.ts [options]

Runs both the browser speed pass and the offline accuracy pass against each
profile, tagged with a shared benchId so they appear paired in the dashboard.

Options:
  -p, --profiles <csv>   Profile names or set names (e.g. "llama-vs-qwen" or
                         "qwen3-0.6b-off-warm,llama-3.2-1b-warm")
  -d, --dimension <name> Limit accuracy pass to one dimension (tool-calling,
                         reasoning, instruction-following, embedding)
      --skip-speed       Accuracy only (no browser)
      --skip-accuracy    Speed only (skip Character-based offline pass)
      --live-bench-url <url>  Stream events to dashboard (env: ${LIVE_BENCH_URL_ENV})
      --list-profiles    List available profiles + profile sets
      --fail-fast        Stop on first failure
  -h, --help             Show this help

Examples:
  bun run eval/bench.ts --profiles qwen3-0.6b-off-warm
  bun run eval/bench.ts --profiles llama-vs-qwen --dimension tool-calling
  WEBLLM_LIVE_BENCH_URL=http://localhost:8033 bun run eval/bench.ts --profiles full
`);
}

if (import.meta.main) {
	main();
}
