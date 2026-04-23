import { parseArgs } from "node:util";
import {
	buildSmokeTestUrl,
	ensureModelDownloaded,
	resolveAgentchromeSession,
	runSmokeChatTurn,
	waitForSmokeTestResult,
	agentchrome,
	type SmokeTestPage,
} from "./browser-smoke.js";
import { getModelById } from "./models.js";

const DEFAULT_MODEL_ID = "qwen3-0.6b-q4f16";
const DEFAULT_PROMPT = "hello";
const DEFAULT_PAGE: SmokeTestPage = "smoke";

function main(): void {
	const { values } = parseArgs({
		options: {
			model: { type: "string", short: "m" },
			prompt: { type: "string", short: "p" },
			page: { type: "string" },
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

	const modelId = values.model ?? DEFAULT_MODEL_ID;
	const prompt = values.prompt ?? DEFAULT_PROMPT;
	const page = parsePage(values.page);
	const model = getModelById(modelId);
	if (!model) {
		console.error(
			`Unknown model "${modelId}". Use --model with an ID from eval/models.ts.`,
		);
		process.exit(1);
	}

	run(model, prompt, page, values.port, values.tab).catch(
		(err) => {
			console.error(
				`Fatal: ${err instanceof Error ? err.message : String(err)}`,
			);
			process.exit(1);
		},
	);
}

async function run(
	model: NonNullable<ReturnType<typeof getModelById>>,
	prompt: string,
	page: SmokeTestPage,
	portArg?: string,
	tabArg?: string,
): Promise<void> {
	await ensureModelDownloaded(model);
	const { port, tab } = await resolveAgentchromeSession(portArg, tabArg, page);
	const url = buildSmokeTestUrl(model.id, model.contextLength, {
		page,
		extraParams: {
			chatSmoke: Date.now(),
		},
	});

	console.log(`Navigating to ${url}`);
	agentchrome(port, tab, ["navigate", url]);
	await waitForSmokeTestResult(port, tab);

	console.log(`Running interactive chat regression with prompt ${JSON.stringify(prompt)}`);
	const result = await runSmokeChatTurn(port, tab, prompt);
	if (!result.assistantText.trim()) {
		throw new Error("Smoke chat regression produced empty assistant text");
	}
	if (!result.finishReason.trim()) {
		throw new Error("Smoke chat regression produced no finish reason");
	}

	console.log("\nSmoke chat regression passed");
	console.log(`Model:         ${model.id}`);
	console.log(`Page:          ${page}`);
	console.log(`Prompt:        ${JSON.stringify(prompt)}`);
	console.log(`Finish reason: ${result.finishReason}`);
	console.log(`Assistant:     ${result.assistantText}`);
}

function parsePage(value: string | undefined): SmokeTestPage {
	if (!value) return DEFAULT_PAGE;
	if (value === "smoke" || value === "debug") return value;
	console.error(`Unknown page "${value}". Use --page smoke or --page debug.`);
	process.exit(1);
}

function printUsage(): void {
	console.log(`Usage: bun run eval/chat-smoke.ts [options]

Options:
  -m, --model <id>      Model to test (default: ${DEFAULT_MODEL_ID})
  -p, --prompt <text>   Interactive chat prompt (default: ${JSON.stringify(DEFAULT_PROMPT)})
      --page <name>     Page to test: smoke or debug (default: ${DEFAULT_PAGE})
      --port <cdp-port> Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>    Use this specific Chrome tab ID
  -h, --help            Show this help

Prereqs:
  - smoke-test server up: \`make smoke-serve\`
  - Chrome with the smoke test open via agentchrome
`);
}

main();
