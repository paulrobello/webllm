import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import type { SmokeTestPage } from "./browser-smoke.js";
import {
	getSmokeProfile,
	getSmokeProfileSet,
	listSmokeProfiles,
	listSmokeProfileSets,
	type SmokeProfile,
} from "./smoke-profiles.js";
import { LIVE_BENCH_URL_ENV } from "./live-client.js";
import { SMOKE_RUNS_DIR_ENV } from "./smoke-runs.js";

const DEFAULT_MODELS = ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"];
const DEFAULT_PAGES: SmokeTestPage[] = ["smoke", "debug"];
const DEFAULT_PROMPT = "hello";
const DEFAULT_PRESET = "fast";
const PRESET_NAMES = ["fast", "full", "qwen-only", "smoke-only"] as const;
const THINKING_MODES = ["off", "on"] as const;

export type ThinkingMode = (typeof THINKING_MODES)[number];
type MatrixPresetName = (typeof PRESET_NAMES)[number];

export interface MatrixCase {
	model: string;
	page: SmokeTestPage;
	prompt: string;
	thinking: ThinkingMode;
}

export function modelSupportsThinking(modelId: string): boolean {
	return modelId.startsWith("qwen3-");
}

export function getMatrixPreset(name: string): {
	models: string[];
	pages: SmokeTestPage[];
	thinkingModes: ThinkingMode[];
} {
	switch (name as MatrixPresetName) {
		case "fast":
			return {
				models: [...DEFAULT_MODELS],
				pages: [...DEFAULT_PAGES],
				thinkingModes: ["off"],
			};
		case "full":
			return {
				models: [...DEFAULT_MODELS],
				pages: [...DEFAULT_PAGES],
				thinkingModes: ["off", "on"],
			};
		case "qwen-only":
			return {
				models: ["qwen3-0.6b-q4f16"],
				pages: [...DEFAULT_PAGES],
				thinkingModes: ["off", "on"],
			};
		case "smoke-only":
			return {
				models: [...DEFAULT_MODELS],
				pages: ["smoke"],
				thinkingModes: ["off"],
			};
		default:
			console.error(
				`Unknown preset "${name}". Use ${PRESET_NAMES.join(", ")}.`,
			);
			process.exit(1);
	}
}

export function buildMatrixPlan({
	models,
	pages,
	prompt,
	thinkingModes,
}: {
	models: string[];
	pages: SmokeTestPage[];
	prompt: string;
	thinkingModes: ThinkingMode[];
}): MatrixCase[] {
	const cases: MatrixCase[] = [];
	for (const model of models) {
		const modes = thinkingModes.filter(
			(mode) => mode === "off" || modelSupportsThinking(model),
		);
		const effectiveModes = modes.length > 0 ? modes : (["off"] as ThinkingMode[]);
		for (const page of pages) {
			for (const thinking of effectiveModes) {
				cases.push({ model, page, prompt, thinking });
			}
		}
	}
	return cases;
}

interface MatrixRun {
	label: string;
	args: string[];
}

function main(): void {
	const { values } = parseArgs({
		options: {
			preset: { type: "string" },
			models: { type: "string" },
			pages: { type: "string" },
			thinking: { type: "string" },
			profiles: { type: "string" },
			prompt: { type: "string", short: "p" },
			"fail-fast": { type: "boolean" },
			"list-profiles": { type: "boolean" },
			"runs-dir": { type: "string" },
			"no-save": { type: "boolean" },
			"live-bench-url": { type: "string" },
			port: { type: "string" },
			tab: { type: "string" },
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

	const pages = parsePages(
		values.pages,
		getMatrixPreset(values.preset ?? DEFAULT_PRESET).pages,
	);
	const sharedArgs = buildSharedArgs(values.port, values.tab);
	const failFast = values["fail-fast"] ?? false;

	const saveArgs: string[] = [];
	if (values["no-save"]) saveArgs.push("--no-save");
	if (values["runs-dir"])
		saveArgs.push("--runs-dir", values["runs-dir"] as string);
	const childSharedArgs = [...sharedArgs, ...saveArgs];

	const runs: MatrixRun[] = values.profiles
		? buildProfileRuns(values.profiles, pages, childSharedArgs)
		: buildCaseRuns(values, pages, childSharedArgs);

	const childEnv = { ...process.env };
	if (values["runs-dir"]) {
		childEnv[SMOKE_RUNS_DIR_ENV] = values["runs-dir"] as string;
	}
	if (values["live-bench-url"]) {
		childEnv[LIVE_BENCH_URL_ENV] = values["live-bench-url"] as string;
	}

	const results: Array<{ label: string; ok: boolean }> = [];
	for (const run of runs) {
		console.log(`\n=== ${run.label} ===`);
		try {
			execFileSync("bun", run.args, { stdio: "inherit", env: childEnv });
			results.push({ label: run.label, ok: true });
		} catch {
			results.push({ label: run.label, ok: false });
			if (failFast) break;
		}
	}

	const failed = results.filter((r) => !r.ok).length;
	console.log("\nMatrix summary");
	console.log(`Total:   ${results.length}`);
	console.log(`Passed:  ${results.length - failed}`);
	console.log(`Failed:  ${failed}`);
	for (const result of results) {
		console.log(`  [${result.ok ? "PASS" : "FAIL"}] ${result.label}`);
	}
	if (failed > 0) process.exit(1);
}

function buildCaseRuns(
	values: {
		preset?: string;
		models?: string;
		pages?: string;
		thinking?: string;
		prompt?: string;
	},
	pages: SmokeTestPage[],
	sharedArgs: string[],
): MatrixRun[] {
	const preset = getMatrixPreset(values.preset ?? DEFAULT_PRESET);
	const models = parseCsv(values.models, preset.models);
	const thinkingModes = parseThinkingModes(
		values.thinking,
		preset.thinkingModes,
	);
	const prompt = values.prompt ?? DEFAULT_PROMPT;
	const plan = buildMatrixPlan({ models, pages, prompt, thinkingModes });
	return plan.map((entry) => ({
		label: `${entry.model} · ${entry.page} · thinking=${entry.thinking}`,
		args: [
			"run",
			"eval/chat-smoke.ts",
			"--model",
			entry.model,
			"--page",
			entry.page,
			"--prompt",
			entry.prompt,
			"--thinking",
			entry.thinking,
			...sharedArgs,
		],
	}));
}

function buildProfileRuns(
	profilesArg: string,
	pages: SmokeTestPage[],
	sharedArgs: string[],
): MatrixRun[] {
	const expanded = expandProfileNames(profilesArg);
	if (expanded.length === 0) {
		console.error("--profiles requires a comma-separated list of names");
		process.exit(1);
	}
	const profiles: SmokeProfile[] = expanded.map((name) => {
		const p = getSmokeProfile(name);
		if (!p) {
			console.error(
				`Unknown profile "${name}". Available: ${listSmokeProfiles().join(", ")}`,
			);
			process.exit(1);
		}
		return p;
	});
	const runs: MatrixRun[] = [];
	for (const profile of profiles) {
		for (const page of pages) {
			runs.push({
				label: `${profile.name} · ${page}`,
				args: [
					"run",
					"eval/chat-smoke.ts",
					"--profile",
					profile.name,
					"--page",
					page,
					...sharedArgs,
				],
			});
		}
	}
	return runs;
}

/**
 * Expand a --profiles arg into a de-duplicated list of profile names.
 * Each comma-separated entry is either a profile-set name (expands to
 * the set's members) or a raw profile name (kept as-is).
 */
export function expandProfileNames(arg: string): string[] {
	const items = parseCsv(arg, []);
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

function parseCsv(value: string | undefined, fallback: string[]): string[] {
	if (!value) return fallback;
	return value
		.split(",")
		.map((item) => item.trim())
		.filter(Boolean);
}

function parsePages(
	value: string | undefined,
	fallback: SmokeTestPage[],
): SmokeTestPage[] {
	const pages = parseCsv(value, fallback);
	for (const page of pages) {
		if (page !== "smoke" && page !== "debug") {
			console.error(`Unknown page "${page}". Use smoke or debug.`);
			process.exit(1);
		}
	}
	return pages as SmokeTestPage[];
}

function parseThinkingModes(
	value: string | undefined,
	fallback: ThinkingMode[],
): ThinkingMode[] {
	const modes = parseCsv(value, fallback);
	for (const mode of modes) {
		if (mode !== "off" && mode !== "on") {
			console.error(`Unknown thinking mode "${mode}". Use off or on.`);
			process.exit(1);
		}
	}
	return modes as ThinkingMode[];
}

function buildSharedArgs(
	port: string | undefined,
	tab: string | undefined,
): string[] {
	const args: string[] = [];
	if (port) args.push("--port", port);
	if (tab) args.push("--tab", tab);
	return args;
}

function printUsage(): void {
	console.log(`Usage: bun run eval/chat-smoke-matrix.ts [options]

Options:
      --preset <name>    Matrix preset: ${PRESET_NAMES.join(", ")} (default: ${DEFAULT_PRESET})
      --models <csv>     Comma-separated model IDs (default: ${DEFAULT_MODELS.join(",")})
      --pages <csv>      Comma-separated pages: smoke,debug (default: ${DEFAULT_PAGES.join(",")})
      --thinking <csv>   Thinking modes: off,on (non-Qwen3 models auto-collapse to off)
      --profiles <csv>   Profile names from eval/smoke-profiles.ts (supersedes preset/models/thinking)
      --list-profiles    List available profiles and exit
  -p, --prompt <text>    Interactive chat prompt (default: ${JSON.stringify(DEFAULT_PROMPT)})
      --fail-fast        Stop after the first failing case
      --port <cdp-port>  Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>     Use this specific Chrome tab ID
  -h, --help             Show this help
`);
}

if (import.meta.main) {
	main();
}
