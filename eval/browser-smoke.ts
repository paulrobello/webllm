import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import type { BenchmarkModel } from "./models.js";

export const SMOKE_TEST_URL = "http://localhost:8031/real-model.html";

export function buildSmokeTestUrl(
	modelId: string,
	contextLength: number,
	extraParams: Record<string, string | number | boolean> = {},
): string {
	const params = new URLSearchParams({
		model: modelId,
		ctx: String(contextLength),
	});
	for (const [key, value] of Object.entries(extraParams)) {
		params.set(key, String(value));
	}
	return `${SMOKE_TEST_URL}?${params.toString()}`;
}

export interface SmokeTestResult {
	tokensGenerated: number;
	totalMs: number;
	prefillMs: number;
	decodeMs: number;
	tokensPerSecond: number;
	completionPageMs: number;
}

export function agentchrome(
	port: string,
	tab: string | undefined,
	args: string[],
): string {
	const full = ["--port", port];
	if (tab) full.push("--tab", tab);
	full.push(...args);
	return execFileSync("agentchrome", full, { encoding: "utf-8" });
}

function sleep(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}

function evalStringResult(
	port: string,
	tab: string,
	script: string,
): string | undefined {
	const out = agentchrome(port, tab, ["js", "exec", script]);
	const resp = JSON.parse(out) as { result?: string; type?: string };
	if (resp.type === "string" && typeof resp.result === "string") {
		return resp.result;
	}
	return undefined;
}

export async function ensureModelDownloaded(
	model: BenchmarkModel,
): Promise<void> {
	const destDir = "smoke-test/models";
	const destPath = `${destDir}/${model.id}.gguf`;
	if (existsSync(destPath)) {
		return;
	}

	console.log(`\nModel ${model.id} not found locally. Preparing to download...`);
	mkdirSync(destDir, { recursive: true });

	const repoUrl = model.ggufUrl;
	if (!repoUrl.startsWith("https://huggingface.co/")) {
		throw new Error(`Unsupported model URL format: ${repoUrl}`);
	}

	const repoName = repoUrl.replace("https://huggingface.co/", "");
	const apiUrl = `https://huggingface.co/api/models/${repoName}/tree/main`;

	const res = await fetch(apiUrl);
	if (!res.ok) {
		throw new Error(
			`Failed to fetch model tree from Hugging Face: HTTP ${res.status}`,
		);
	}

	const files = (await res.json()) as Array<{ path: string; size: number }>;
	const preferredQuants = [
		model.defaultQuant.toLowerCase(),
		"q4_k_m",
		"q4_0",
		"q4_1",
		"q5_k_m",
	];

	let chosenFile: { path: string; size: number } | undefined;
	for (const quant of preferredQuants) {
		chosenFile = files.find(
			(file) =>
				file.path.toLowerCase().endsWith(".gguf") &&
				file.path.toLowerCase().includes(quant),
		);
		if (chosenFile) break;
	}

	if (!chosenFile) {
		chosenFile = files.find((file) =>
			file.path.toLowerCase().endsWith(".gguf"),
		);
	}

	if (!chosenFile) {
		throw new Error(`Could not find any .gguf files in ${repoUrl}`);
	}

	const downloadUrl = `https://huggingface.co/${repoName}/resolve/main/${chosenFile.path}`;
	console.log(
		`Downloading ${chosenFile.path} (${(chosenFile.size / 1024 / 1024).toFixed(1)} MB) to ${destPath}...`,
	);

	execFileSync("curl", ["-L", "--progress-bar", "-o", destPath, downloadUrl], {
		stdio: "inherit",
	});
	console.log("Download complete.\n");
}

export async function resolveAgentchromeSession(
	portArg?: string,
	tabArg?: string,
): Promise<{ port: string; tab: string }> {
	let port = portArg;
	if (!port) {
		const status = execFileSync("agentchrome", ["connect", "--status"], {
			encoding: "utf-8",
		});
		const parsed = JSON.parse(status) as { port?: string | number };
		if (!parsed.port) {
			throw new Error(
				"No active agentchrome session. Start one with `agentchrome connect --launch` or pass --port.",
			);
		}
		port = String(parsed.port);
	}

	if (tabArg) return { port, tab: tabArg };

	const tabs = execFileSync("agentchrome", ["--port", port, "tabs", "list"], {
		encoding: "utf-8",
	});
	const list = JSON.parse(tabs) as Array<{ id: string; url: string }>;
	const smoke = list.find((entry) => entry.url.includes("real-model.html"));
	if (!smoke) {
		throw new Error(
			"No tab currently loaded on real-model.html. Navigate one there first, or pass --tab <TAB_ID>.",
		);
	}
	return { port, tab: smoke.id };
}

export async function waitForSmokeTestResult(
	port: string,
	tab: string,
): Promise<SmokeTestResult> {
	const deadline = Date.now() + 180_000;
	const script = `(() => {
		const pattern = new RegExp("Generated (\\\\d+) tokens in ([0-9.]+)s \\\\(prefill: (\\\\d+)ms, decode: (\\\\d+)ms, ([0-9.]+) tok\\\\/s(?:, finish=[^)]+)?\\\\)");
		const t = document.getElementById("log")?.textContent ?? "";
		const m = t.match(pattern);
		if (!m) return "";
		return JSON.stringify({
			tokensGenerated: +m[1],
			totalMs: +m[2] * 1000,
			prefillMs: +m[3],
			decodeMs: +m[4],
			tokensPerSecond: +m[5],
			completionPageMs: performance.now(),
		});
	})()`;
	let lastError: unknown;
	while (Date.now() < deadline) {
		try {
			const result = evalStringResult(port, tab, script);
			if (result) {
				return JSON.parse(result) as SmokeTestResult;
			}
		} catch (error) {
			lastError = error;
		}
		await sleep(1000);
	}
	throw new Error(
		`Timed out waiting for smoke-test result line on the page${lastError ? ` (${String(lastError)})` : ""}`,
	);
}

export async function extractSmokeTestPrompt(
	port: string,
	tab: string,
): Promise<string> {
	try {
		const out = agentchrome(port, tab, [
			"js",
			"exec",
			`(() => {
				const t = document.getElementById("log")?.textContent ?? "";
				const m = t.match(/User: (.+?)(?:Assistant:|\\n|$)/);
				return m ? m[1].trim() : "";
			})()`,
		]);
		const resp = JSON.parse(out) as { result?: string };
		return typeof resp.result === "string" ? resp.result : "";
	} catch {
		return "";
	}
}

export interface ChatSmokeResult {
	assistantText: string;
	finishReason: string;
	chatOutput: string;
}

export async function runSmokeChatTurn(
	port: string,
	tab: string,
	prompt: string,
): Promise<ChatSmokeResult> {
	const escapedPrompt = JSON.stringify(prompt);
	agentchrome(port, tab, [
		"js",
		"exec",
		`(() => {
			const promptText = ${escapedPrompt};
			const input = document.getElementById("chat-input");
			const button = document.getElementById("chat-btn");
			const output = document.getElementById("chat-output");
			if (!input || !button || !output) {
				throw new Error("Missing chat elements on smoke-test page");
			}
			window.__chatSmokeInitialText = output.textContent ?? "";
			input.value = promptText;
			button.click();
			return "started";
		})()`,
	]);
	const deadline = Date.now() + 180_000;
	const parseScript = `(() => {
		const output = document.getElementById("chat-output");
		if (!output) {
			throw new Error("Missing #chat-output element on smoke-test page");
		}
		const initialText = window.__chatSmokeInitialText ?? "";
		const currentText = output.textContent ?? "";
		const delta = currentText.slice(initialText.length);
		if (!delta) return "";
		if (delta.includes("Aborted(") || delta.includes("[Error:")) {
			return JSON.stringify({
				ok: false,
				error: delta.trim(),
				chatOutput: currentText,
			});
		}
		const assistantPattern = new RegExp("Assistant: ([\\\\s\\\\S]*?)(?:\\\\n\\\\(|$)");
		const finishPattern = new RegExp("finish=([^)\\\\n]+)");
		const assistantMatch = delta.match(assistantPattern);
		const finishMatch = delta.match(finishPattern);
		const assistantText = assistantMatch?.[1]?.trim() ?? "";
		if (!assistantText || !finishMatch) return "";
		return JSON.stringify({
			ok: true,
			assistantText,
			finishReason: finishMatch[1].trim(),
			chatOutput: currentText,
		});
	})()`;
	let lastError: unknown;
	while (Date.now() < deadline) {
		let result: string | undefined;
		try {
			result = evalStringResult(port, tab, parseScript);
		} catch (error) {
			lastError = error;
		}
		if (!result) {
			await sleep(1000);
			continue;
		}
		const parsed = JSON.parse(result) as
			| {
					ok: true;
					assistantText: string;
					finishReason: string;
					chatOutput: string;
			  }
			| { ok: false; error: string; chatOutput: string };
		if (!parsed.ok) {
			throw new Error(parsed.error);
		}
		return parsed;
	}
	throw new Error(
		`Timed out waiting for smoke-test chat output${lastError ? ` (${String(lastError)})` : ""}`,
	);
}
