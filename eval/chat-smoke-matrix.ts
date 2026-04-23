import { execFileSync } from "node:child_process";
import { parseArgs } from "node:util";
import type { SmokeTestPage } from "./browser-smoke.js";

const DEFAULT_MODELS = ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"];
const DEFAULT_PAGES: SmokeTestPage[] = ["smoke", "debug"];
const DEFAULT_PROMPT = "hello";

export interface MatrixCase {
	model: string;
	page: SmokeTestPage;
	prompt: string;
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
			models: { type: "string" },
			pages: { type: "string" },
			prompt: { type: "string", short: "p" },
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

	const models = parseCsv(values.models, DEFAULT_MODELS);
	const pages = parsePages(values.pages);
	const prompt = values.prompt ?? DEFAULT_PROMPT;
	const plan = buildMatrixPlan({ models, pages, prompt });
	const sharedArgs = buildSharedArgs(values.port, values.tab);

	const failures: MatrixCase[] = [];
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
		} catch {
			failures.push(entry);
		}
	}

	console.log("\nMatrix summary");
	console.log(`Total:   ${plan.length}`);
	console.log(`Passed:  ${plan.length - failures.length}`);
	console.log(`Failed:  ${failures.length}`);
	if (failures.length > 0) {
		for (const failure of failures) {
			console.log(`  - ${failure.model} · ${failure.page}`);
		}
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

function parsePages(value: string | undefined): SmokeTestPage[] {
	const pages = parseCsv(value, DEFAULT_PAGES);
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
      --models <csv>    Comma-separated model IDs (default: ${DEFAULT_MODELS.join(",")})
      --pages <csv>     Comma-separated pages: smoke,debug (default: ${DEFAULT_PAGES.join(",")})
  -p, --prompt <text>   Interactive chat prompt (default: ${JSON.stringify(DEFAULT_PROMPT)})
      --port <cdp-port> Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>    Use this specific Chrome tab ID
  -h, --help            Show this help
`);
}

if (import.meta.main) {
	main();
}
