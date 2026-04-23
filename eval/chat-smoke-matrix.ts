import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import type { SmokeTestPage } from "./browser-smoke.js";

const DEFAULT_MODELS = ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"];
const DEFAULT_PAGES: SmokeTestPage[] = ["smoke", "debug"];
const DEFAULT_PROMPT = "hello";
const DEFAULT_PRESET = "fast";
const PRESET_NAMES = ["fast", "full", "qwen-only", "smoke-only"] as const;

type MatrixPresetName = (typeof PRESET_NAMES)[number];

export interface MatrixCase {
	model: string;
	page: SmokeTestPage;
	prompt: string;
}

export function getMatrixPreset(name: string): {
	models: string[];
	pages: SmokeTestPage[];
} {
	switch (name as MatrixPresetName) {
		case "fast":
			return {
				models: [...DEFAULT_MODELS],
				pages: [...DEFAULT_PAGES],
			};
		case "full":
			return {
				models: [...DEFAULT_MODELS],
				pages: [...DEFAULT_PAGES],
			};
		case "qwen-only":
			return {
				models: ["qwen3-0.6b-q4f16"],
				pages: [...DEFAULT_PAGES],
			};
		case "smoke-only":
			return {
				models: [...DEFAULT_MODELS],
				pages: ["smoke"],
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
}: {
	models: string[];
	pages: SmokeTestPage[];
	prompt: string;
}): MatrixCase[] {
	return models.flatMap((model) =>
		pages.map((page) => ({
			model,
			page,
			prompt,
		})),
	);
}

function main(): void {
	const { values } = parseArgs({
		options: {
			preset: { type: "string" },
			models: { type: "string" },
			pages: { type: "string" },
			prompt: { type: "string", short: "p" },
			"fail-fast": { type: "boolean" },
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

	const preset = getMatrixPreset(values.preset ?? DEFAULT_PRESET);
	const models = parseCsv(values.models, preset.models);
	const pages = parsePages(values.pages, preset.pages);
	const prompt = values.prompt ?? DEFAULT_PROMPT;
	const plan = buildMatrixPlan({ models, pages, prompt });
	const sharedArgs = buildSharedArgs(values.port, values.tab);
	const failFast = values["fail-fast"] ?? false;

	const failures: MatrixCase[] = [];
	const results: Array<{ entry: MatrixCase; ok: boolean }> = [];
	for (const entry of plan) {
		const args = [
			"run",
			"eval/chat-smoke.ts",
			"--model",
			entry.model,
			"--page",
			entry.page,
			"--prompt",
			entry.prompt,
			...sharedArgs,
		];
		console.log(`\n=== ${entry.model} · ${entry.page} ===`);
		try {
			execFileSync("bun", args, {
				stdio: "inherit",
			});
			results.push({ entry, ok: true });
		} catch {
			failures.push(entry);
			results.push({ entry, ok: false });
			if (failFast) {
				break;
			}
		}
	}

	console.log("\nMatrix summary");
	console.log(`Total:   ${results.length}`);
	console.log(`Passed:  ${results.length - failures.length}`);
	console.log(`Failed:  ${failures.length}`);
	for (const result of results) {
		console.log(
			`  [${result.ok ? "PASS" : "FAIL"}] ${result.entry.model} · ${result.entry.page}`,
		);
	}
	if (failures.length > 0) {
		process.exit(1);
	}
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
      --preset <name>   Matrix preset: ${PRESET_NAMES.join(", ")} (default: ${DEFAULT_PRESET})
      --models <csv>    Comma-separated model IDs (default: ${DEFAULT_MODELS.join(",")})
      --pages <csv>     Comma-separated pages: smoke,debug (default: ${DEFAULT_PAGES.join(",")})
  -p, --prompt <text>   Interactive chat prompt (default: ${JSON.stringify(DEFAULT_PROMPT)})
      --fail-fast       Stop after the first failing case
      --port <cdp-port> Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>    Use this specific Chrome tab ID
  -h, --help            Show this help
`);
}

if (import.meta.main) {
	main();
}
