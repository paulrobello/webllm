/**
 * Combined bench driver: runs the browser-driven speed pass AND the offline
 * accuracy pass against the same profile, tagged with a shared `benchId` so
 * both halves show up side-by-side on the live dashboard.
 *
 * Speed (real WebGPU) → eval/chat-smoke.ts → run_* events
 * Accuracy (offline via Character) → eval/cli.ts → eval_* events
 */
import { execFileSync, spawn } from "node:child_process";
import { parseArgs } from "node:util";
import {
	LIVE_BENCH_URL_ENV,
	publishBenchSessionComplete,
	publishBenchSessionStarted,
	resolveLiveBenchUrl,
} from "./live-client.js";
import { resolveProfileModel } from "./smoke-profiles.js";
import {
	getSmokeProfile,
	getSmokeProfileSet,
	listSmokeProfiles,
	listSmokeProfileSets,
	type SmokeProfile,
} from "./smoke-profiles.js";

/**
 * Env vars threaded from the parent bench process to child harnesses
 * (eval/browser-eval.ts, eval/cli.ts, eval/chat-smoke.ts) so per-model
 * eval events can be tagged with the parent session and the per-task
 * sampling temperature can be overridden uniformly across the whole
 * accuracy pass.
 */
const BENCH_SESSION_ID_ENV = "WEBLLM_BENCH_SESSION_ID";
const BENCH_EVAL_TEMP_ENV = "WEBLLM_BENCH_EVAL_TEMPERATURE";

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

async function main(): Promise<void> {
	const { values } = parseArgs({
		options: {
			profiles: { type: "string", short: "p" },
			dimension: { type: "string", short: "d" },
			"skip-speed": { type: "boolean" },
			"skip-accuracy": { type: "boolean" },
			"live-bench-url": { type: "string" },
			"list-profiles": { type: "boolean" },
			"fail-fast": { type: "boolean" },
			worker: { type: "boolean" },
			"eval-temperature": { type: "string" },
			"no-retry-on-wedge": { type: "boolean" },
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
	const childEnv: Record<string, string> = { ...process.env } as Record<
		string,
		string
	>;
	if (liveUrl) childEnv[LIVE_BENCH_URL_ENV] = liveUrl;

	// Pin the accuracy-pass sampling temperature unless the caller explicitly
	// overrode it. Default 0 (greedy) — eliminates the per-task sampling
	// variance that swung tinyllama 0.35 ↔ 0.23 across April runs. Children
	// read this from env and override profile.temperature for tasks (speed
	// pass keeps its profile-native temperature so tok/s headlines aren't
	// affected). See also: docs/BENCHMARKS.md greedy-eval cutover.
	const evalTemperatureRaw = values["eval-temperature"];
	const evalTemperature =
		evalTemperatureRaw !== undefined
			? Number.parseFloat(evalTemperatureRaw)
			: 0;
	if (!Number.isFinite(evalTemperature) || evalTemperature < 0) {
		console.error(
			`Error: --eval-temperature must be a non-negative number (got "${evalTemperatureRaw}")`,
		);
		process.exit(1);
	}
	childEnv[BENCH_EVAL_TEMP_ENV] = String(evalTemperature);

	const sessionId = `sess-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
	childEnv[BENCH_SESSION_ID_ENV] = sessionId;

	if (liveUrl) {
		const modelIds = cases
			.map((c) => {
				const model = resolveProfileModel(c.profile);
				return model?.id;
			})
			.filter((id): id is string => typeof id === "string");
		await publishBenchSessionStarted(liveUrl, {
			sessionId,
			startedAt: new Date().toISOString(),
			totalModels: cases.length,
			modelIds,
			profileNames: cases.map((c) => c.profile.name),
			evalTemperature,
		});
		console.log(
			`[bench] session ${sessionId} · ${cases.length} model(s) · eval temperature ${evalTemperature}`,
		);
	}

	const results: Array<{
		label: string;
		ok: boolean;
		phase: string;
		reason?: string;
		retried?: boolean;
	}> = [];
	const doSpeed = !values["skip-speed"];
	const doAccuracy = !values["skip-accuracy"];
	const failFast = values["fail-fast"] ?? false;
	const retryOnWedge = !(values["no-retry-on-wedge"] ?? false);

	for (const entry of cases) {
		const label = entry.profile.name;
		console.log(`\n═══ ${label} ═══`);

		// Embedding-only profiles (encoder models like Arctic-Embed) don't
		// generate text. The speed phase runs `chat-smoke` which would fail
		// on a model that has no causal-LM path; skip it. Accuracy is still
		// useful — it runs the embedding-dimension tasks and writes cosine
		// scores to the dashboard.
		const isEmbedding = entry.profile.embedding === true;

		if (doSpeed && !isEmbedding) {
			console.log(`\n--- speed: ${label} ---`);
			const speedArgs = [
				"run",
				"eval/chat-smoke.ts",
				"--profile",
				entry.profile.name,
			];
			if (values.worker) speedArgs.push("--worker");
			const { ok, reason, retried } = await runChildWithWedgeRetry(
				speedArgs,
				childEnv,
				`${label} · speed`,
				retryOnWedge,
			);
			results.push({ label, ok, phase: "speed", reason, retried });
			if (!ok && failFast) break;
		} else if (doSpeed && isEmbedding) {
			console.log(`\n--- speed: skipped (embedding profile) ---`);
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
			const { ok, reason, retried } = await runChildWithWedgeRetry(
				accArgs,
				childEnv,
				`${label} · accuracy`,
				retryOnWedge,
			);
			results.push({ label, ok, phase: "accuracy", reason, retried });
			if (!ok && failFast) break;
		}
	}

	const failed = results.filter((r) => !r.ok).length;
	console.log("\nBench summary");
	console.log(`Total:   ${results.length}`);
	console.log(`Passed:  ${results.length - failed}`);
	console.log(`Failed:  ${failed}`);
	for (const r of results) {
		const tag = `[${r.ok ? "PASS" : "FAIL"}] ${r.label} · ${r.phase}${
			r.retried ? " (retried after wedge)" : ""
		}`;
		console.log(r.ok || !r.reason ? `  ${tag}` : `  ${tag} — ${r.reason}`);
	}

	if (liveUrl) {
		// Count models that have at least one passing phase. Per-phase failures
		// (a speed pass that crashed but accuracy succeeded, or vice versa)
		// still count toward the model's contribution to the session.
		const modelLabels = new Set(cases.map((c) => c.profile.name));
		const passedLabels = new Set(
			results.filter((r) => r.ok).map((r) => r.label),
		);
		const completedModels = Array.from(modelLabels).filter((l) =>
			passedLabels.has(l),
		).length;
		await publishBenchSessionComplete(liveUrl, {
			sessionId,
			completedAt: new Date().toISOString(),
			totalModels: cases.length,
			completedModels,
			failedModels: cases.length - completedModels,
		});
	}

	if (failed > 0) process.exit(1);
}

// Run a bench sub-task as a child Bun process. stdout/stderr are mirrored
// live to this process (so the user sees progress in real time), and we
// also keep the last few lines of each stream so the bench summary can
// surface a one-line failure reason without forcing the user to scroll
// back through the matrix output.
function runChild(
	args: string[],
	env: NodeJS.ProcessEnv,
): Promise<{ ok: boolean; reason?: string }> {
	const TAIL_LINES = 8;
	const tail = { out: [] as string[], err: [] as string[] };

	return new Promise((resolve) => {
		const child = spawn("bun", args, {
			env,
			stdio: ["inherit", "pipe", "pipe"],
		});
		const wire = (
			src: NodeJS.ReadableStream,
			dst: NodeJS.WriteStream,
			bucket: string[],
		) => {
			let leftover = "";
			src.on("data", (chunk: Buffer | string) => {
				const text = typeof chunk === "string" ? chunk : chunk.toString("utf-8");
				dst.write(text);
				const combined = leftover + text;
				const parts = combined.split("\n");
				leftover = parts.pop() ?? "";
				for (const line of parts) {
					if (!line.trim()) continue;
					bucket.push(line);
					if (bucket.length > TAIL_LINES) bucket.shift();
				}
			});
			src.on("end", () => {
				if (leftover.trim()) {
					bucket.push(leftover);
					if (bucket.length > TAIL_LINES) bucket.shift();
				}
			});
		};
		if (child.stdout) wire(child.stdout, process.stdout, tail.out);
		if (child.stderr) wire(child.stderr, process.stderr, tail.err);
		child.on("error", (err) => {
			resolve({ ok: false, reason: err.message });
		});
		child.on("close", (code, signal) => {
			if (code === 0 && !signal) {
				resolve({ ok: true });
				return;
			}
			// Build a one-line reason. Prefer a "Fatal:" / "Error:" line from
			// stderr; fall back to the last non-empty stderr line; fall back
			// to the last stdout line.
			const lines = [...tail.err, ...tail.out];
			const fatal = lines.find((l) =>
				/^(fatal|error)[: ]/i.test(l.trim()),
			);
			let reason = (fatal ?? lines[lines.length - 1] ?? "").trim();
			if (signal) {
				reason = `killed by signal ${signal}${reason ? ` — ${reason}` : ""}`;
			}
			if (reason.length > 200) reason = `${reason.slice(0, 197)}…`;
			resolve({
				ok: false,
				reason: reason || `exited with code ${code ?? "?"}`,
			});
		});
	});
}

// Patterns that indicate the browser/WASM engine is wedged and a fresh
// Chrome session should clear it. Pattern A (Chrome/Dawn WebGPU device
// loss under cumulative GPU pressure) is the documented upstream cause —
// see commit a09be3a closure. The abort guard f3cbca9 surfaces it as
// EngineDeadError; speed-pass timeouts and accuracy "no signal" stalls
// are the same root cause manifesting at different stages.
const WEDGE_REASON_PATTERNS: readonly RegExp[] = [
	/EngineDeadError/i,
	/consecutive task errors/i,
	/no signal from page/i,
	/Timed out waiting for smoke-test result line/i,
	/browser bench stalled/i,
	/Smoke chat regression produced empty/i,
];

function isWedgeReason(reason: string | undefined): boolean {
	if (!reason) return false;
	return WEDGE_REASON_PATTERNS.some((p) => p.test(reason));
}

// Kill the agentchrome session so the next child's resolveAgentchromeSession
// launches a fresh Chrome process — clears accumulated GPU-process pressure
// and any wedged WebGPU device. Best-effort; the next child's launch path
// auto-handles a missing session anyway.
function restartChromeSession(): void {
	try {
		execFileSync("agentchrome", ["connect", "--disconnect"], {
			stdio: ["ignore", "ignore", "ignore"],
			timeout: 10_000,
		});
	} catch {
		// Already-disconnected or agentchrome missing — next child handles it.
	}
}

async function runChildWithWedgeRetry(
	args: string[],
	env: NodeJS.ProcessEnv,
	label: string,
	retryEnabled: boolean,
): Promise<{ ok: boolean; reason?: string; retried?: boolean }> {
	const first = await runChild(args, env);
	if (first.ok) return first;
	if (!retryEnabled || !isWedgeReason(first.reason)) return first;

	console.log(
		`\n[bench] ${label} wedged (reason: ${first.reason}) — restarting Chrome and retrying once`,
	);
	restartChromeSession();
	// Brief pause for the OS to reap the Chrome process before the next
	// child probes the CDP port.
	await new Promise<void>((resolve) => setTimeout(resolve, 1000));

	const retry = await runChild(args, env);
	return { ...retry, retried: true };
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
      --worker           Run engine inside a DedicatedWorker (speed pass only)
      --eval-temperature <n>  Sampling temperature for the accuracy pass
                              (default: 0 / greedy). Pre-2026-05 history was
                              captured at the profile's native temperature
                              (often 0.6); current canonical baselines are
                              greedy. Override only when you specifically
                              want temp-stratified scores.
      --no-retry-on-wedge  Skip the automatic Chrome restart + single retry
                              when a sub-task fails with a wedge signature
                              (EngineDeadError, "no signal from page",
                              speed-pass timeout). Default is to retry once,
                              which converts most upstream Chrome/Dawn
                              WebGPU-device-loss wedges into PASSes.
  -h, --help             Show this help

Examples:
  bun run eval/bench.ts --profiles qwen3-0.6b-off-warm
  bun run eval/bench.ts --profiles llama-vs-qwen --dimension tool-calling
  WEBLLM_LIVE_BENCH_URL=http://localhost:8033 bun run eval/bench.ts --profiles full
`);
}

if (import.meta.main) {
	main().catch((err) => {
		console.error(err instanceof Error ? err.stack ?? err.message : err);
		process.exit(1);
	});
}
