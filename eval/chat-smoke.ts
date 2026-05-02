import { parseArgs } from "node:util";
import {
	agentchrome,
	buildSmokeTestUrl,
	ensureModelDownloaded,
	ensureSmokeServerReachable,
	extractSmokeTestAssistant,
	resolveAgentchromeSession,
	runSmokeChatTurn,
	type SmokeTestPage,
	waitForSmokeTestResult,
} from "./browser-smoke.js";
import { getModelById } from "./models.js";
import {
	getSmokeProfile,
	listSmokeProfiles,
	profileToUrlParams,
	resolveProfileModel,
	type SmokeProfile,
} from "./smoke-profiles.js";
import {
	publishRunComplete,
	publishRunFailed,
	publishRunStarted,
	resolveLiveBenchUrl,
} from "./live-client.js";
import {
	buildSmokeRunRecord,
	resolveSmokeRunsDir,
	type SmokeRunParams,
	writeSmokeRunRecord,
} from "./smoke-runs.js";

const DEFAULT_MODEL_ID = "qwen3-0.6b-q4f16";
const DEFAULT_PROMPT = "hello";
const DEFAULT_PAGE: SmokeTestPage = "smoke";
const DEFAULT_THINKING: ThinkingMode = "off";

type ThinkingMode = "off" | "on";

function main(): void {
	const { values } = parseArgs({
		options: {
			model: { type: "string", short: "m" },
			profile: { type: "string" },
			prompt: { type: "string", short: "p" },
			page: { type: "string" },
			thinking: { type: "string" },
			port: { type: "string" },
			tab: { type: "string" },
			help: { type: "boolean", short: "h" },
			profiles: { type: "boolean" },
			"runs-dir": { type: "string" },
			"no-save": { type: "boolean" },
			"live-bench-url": { type: "string" },
			worker: { type: "boolean" },
		},
		strict: true,
	});

	if (values.help) {
		printUsage();
		process.exit(0);
	}

	if (values.profiles) {
		for (const name of listSmokeProfiles()) console.log(name);
		process.exit(0);
	}

	const page = parsePage(values.page);

	const profile = values.profile ? resolveProfile(values.profile) : null;

	const modelId = profile?.model ?? values.model ?? DEFAULT_MODEL_ID;
	const prompt = values.prompt ?? profile?.prompt ?? DEFAULT_PROMPT;
	const thinking = values.thinking
		? parseThinking(values.thinking)
		: (profile?.thinking ?? DEFAULT_THINKING);

	const model = profile
		? resolveProfileModel(profile)
		: getModelById(modelId);
	if (!model) {
		console.error(
			`Unknown model "${modelId}". Use --model with an ID from eval/models.ts.`,
		);
		process.exit(1);
	}

	run(model, prompt, page, thinking, profile, {
		port: values.port,
		tab: values.tab,
		runsDir: values["runs-dir"],
		save: !values["no-save"],
		liveBenchUrl: resolveLiveBenchUrl(values["live-bench-url"]),
		worker: values.worker ?? false,
	}).catch((err) => {
		console.error(`Fatal: ${err instanceof Error ? err.message : String(err)}`);
		process.exit(1);
	});
}

async function run(
	model: NonNullable<ReturnType<typeof getModelById>>,
	prompt: string,
	page: SmokeTestPage,
	thinking: ThinkingMode,
	profile: SmokeProfile | null,
	opts: {
		port?: string;
		tab?: string;
		runsDir?: string;
		save: boolean;
		liveBenchUrl: string | null;
		worker: boolean;
	},
): Promise<void> {
	const runId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
	await ensureSmokeServerReachable(page);
	await ensureModelDownloaded(model);
	const { port, tab } = await resolveAgentchromeSession(
		opts.port,
		opts.tab,
		page,
	);

	if (opts.liveBenchUrl) {
		await publishRunStarted(opts.liveBenchUrl, {
			runId,
			profile: profile?.name,
			model: model.id,
			page,
			thinking,
			prompt,
		});
	}
	const extraParams: Record<string, string | number | boolean> = {
		chatSmoke: Date.now(),
	};
	if (thinking === "on") extraParams.thinking = 1;
	if (opts.worker) extraParams.worker = 1;
	// Tell the page where to POST its system profile so we can scrape
	// the resulting systemId for the run_complete record.
	if (opts.liveBenchUrl) extraParams.ingest = opts.liveBenchUrl;
	if (profile) {
		Object.assign(extraParams, profileToUrlParams(profile));
		extraParams.profile = profile.name;
	}
	extraParams.prompt = prompt;
	const contextLength = profile?.contextLength ?? model.contextLength;
	const url = buildSmokeTestUrl(model.id, contextLength, {
		page,
		extraParams,
	});

	console.log(`Navigating to ${url}`);
	agentchrome(port, tab, ["navigate", url]);
	let oneShotResult: Awaited<ReturnType<typeof waitForSmokeTestResult>>;
	let oneShotAssistant: string;
	let chatResult: Awaited<ReturnType<typeof runSmokeChatTurn>>;
	try {
		oneShotResult = await waitForSmokeTestResult(port, tab);
		oneShotAssistant = await extractSmokeTestAssistant(port, tab);
		console.log(
			`Running interactive chat regression with prompt ${JSON.stringify(prompt)}`,
		);
		chatResult = await runSmokeChatTurn(port, tab, prompt);
		if (!chatResult.assistantText.trim()) {
			throw new Error("Smoke chat regression produced empty assistant text");
		}
		if (!chatResult.finishReason.trim()) {
			throw new Error("Smoke chat regression produced no finish reason");
		}
	} catch (err) {
		if (opts.liveBenchUrl) {
			await publishRunFailed(opts.liveBenchUrl, {
				runId,
				profile: profile?.name,
				model: model.id,
				error: err instanceof Error ? err.message : String(err),
			});
		}
		throw err;
	}

	console.log("\nSmoke chat regression passed");
	if (profile) console.log(`Profile:       ${profile.name}`);
	console.log(`Model:         ${model.id}`);
	console.log(`Page:          ${page}`);
	console.log(`Thinking:      ${thinking}`);
	console.log(`Prompt:        ${JSON.stringify(prompt)}`);
	console.log(`Finish reason: ${chatResult.finishReason}`);
	console.log(`Assistant:     ${chatResult.assistantText}`);

	const params: SmokeRunParams = {
		maxTokens: profile?.maxTokens,
		temperature: profile?.temperature,
		topK: profile?.topK,
		topP: profile?.topP,
		repetitionPenalty: profile?.repetitionPenalty,
		seed: profile?.seed,
	};
	const record = buildSmokeRunRecord({
		profile,
		modelId: model.id,
		page,
		thinking,
		mode: opts.worker ? "worker" : "main",
		prompt,
		contextLength,
		params,
		oneShotResult,
		oneShotAssistantText: oneShotAssistant,
		oneShotFinishReason: oneShotResult.finishReason,
		interactiveResult: chatResult,
	});
	const systemId = scrapeSystemId(port, tab);
	if (systemId) (record as { systemId?: string }).systemId = systemId;

	if (opts.save) {
		const path = writeSmokeRunRecord(record, resolveSmokeRunsDir(opts.runsDir));
		console.log(`Run record:    ${path}`);
	}

	if (opts.liveBenchUrl) {
		await publishRunComplete(opts.liveBenchUrl, { ...record, runId });
	}
}

function scrapeSystemId(port: string, tab: string): string | null {
	try {
		const out = agentchrome(port, tab, [
			"js",
			"exec",
			"window.__webllmSystemId ?? null",
		]);
		const resp = JSON.parse(out) as { result?: unknown };
		return typeof resp.result === "string" ? resp.result : null;
	} catch {
		return null;
	}
}

function parsePage(value: string | undefined): SmokeTestPage {
	if (!value) return DEFAULT_PAGE;
	if (value === "smoke" || value === "debug") return value;
	console.error(`Unknown page "${value}". Use --page smoke or --page debug.`);
	process.exit(1);
}

function parseThinking(value: string | undefined): ThinkingMode {
	if (!value) return DEFAULT_THINKING;
	if (value === "off" || value === "on") return value;
	console.error(
		`Unknown thinking mode "${value}". Use --thinking off or --thinking on.`,
	);
	process.exit(1);
}

function resolveProfile(name: string): SmokeProfile {
	const profile = getSmokeProfile(name);
	if (!profile) {
		console.error(
			`Unknown profile "${name}". Available: ${listSmokeProfiles().join(", ")}`,
		);
		process.exit(1);
	}
	return profile;
}

function printUsage(): void {
	console.log(`Usage: bun run eval/chat-smoke.ts [options]

Options:
      --profile <name>  Named profile from eval/smoke-profiles.ts (supersedes --model/--thinking/--prompt defaults)
      --profiles        List available profiles and exit
  -m, --model <id>      Model to test (default: ${DEFAULT_MODEL_ID})
  -p, --prompt <text>   Interactive chat prompt (default: ${JSON.stringify(DEFAULT_PROMPT)})
      --page <name>     Page to test: smoke or debug (default: ${DEFAULT_PAGE})
      --thinking <mode> Thinking mode: off or on (default: ${DEFAULT_THINKING})
      --port <cdp-port> Use this agentchrome CDP port instead of auto-detecting
      --tab <tab-id>    Use this specific Chrome tab ID
      --runs-dir <dir>  Directory for run records (default: eval/reports/smoke-runs)
      --no-save         Skip writing a JSON run record
      --live-bench-url <url>  Push events to live dashboard backend (env: WEBLLM_LIVE_BENCH_URL)
      --worker          Run engine inside a DedicatedWorker via WebLLMProxy
  -h, --help            Show this help

Prereqs:
  - smoke-test server up: \`make smoke-serve\`
  - Chrome with the smoke test open via agentchrome
`);
}

main();
