import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, renameSync, statSync, unlinkSync } from "node:fs";
import type { BenchmarkModel } from "./models.js";

export const SMOKE_TEST_URL = "http://localhost:8031/real-model.html";
export const DEBUG_SMOKE_TEST_URL = "http://localhost:8031/real-model-debug.html";
export type SmokeTestPage = "smoke" | "debug";

export function getSmokeTestBaseUrl(page: SmokeTestPage = "smoke"): string {
	return page === "debug" ? DEBUG_SMOKE_TEST_URL : SMOKE_TEST_URL;
}

/**
 * Probe the smoke-test static server. Returns true iff /real-model.html
 * responds 200 within the timeout. Used by bench drivers as a fast-fail
 * preflight when the user runs them directly (Make targets that depend
 * on `smoke-restart` would have already started the server, but direct
 * `bun run eval/...` invocations skip Make entirely).
 */
export async function checkSmokeServer(
	url: string = SMOKE_TEST_URL,
	timeoutMs = 2000,
): Promise<boolean> {
	const controller = new AbortController();
	const t = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const res = await fetch(url, { signal: controller.signal });
		return res.ok;
	} catch {
		return false;
	} finally {
		clearTimeout(t);
	}
}

export async function ensureSmokeServerReachable(
	page: SmokeTestPage = "smoke",
): Promise<void> {
	const probeUrl = getSmokeTestBaseUrl(page);
	const ok = await checkSmokeServer(probeUrl);
	if (ok) return;
	throw new Error(
		[
			`smoke-test server not reachable at ${probeUrl}.`,
			"Start it with `make smoke-serve` (or `make smoke-restart`) in another terminal.",
		].join(" "),
	);
}

export function buildSmokeTestUrl(
	modelId: string,
	contextLength: number,
	options: {
		page?: SmokeTestPage;
		extraParams?: Record<string, string | number | boolean>;
	} = {},
): string {
	const { page = "smoke", extraParams = {} } = options;
	const params = new URLSearchParams({
		model: modelId,
		ctx: String(contextLength),
	});
	for (const [key, value] of Object.entries(extraParams)) {
		params.set(key, String(value));
	}
	return `${getSmokeTestBaseUrl(page)}?${params.toString()}`;
}

export interface SmokeTestResult {
	tokensGenerated: number;
	totalMs: number;
	prefillMs: number;
	decodeMs: number;
	tokensPerSecond: number;
	completionPageMs: number;
	finishReason?: string;
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
	const partPath = `${destPath}.part`;
	// A leftover .part from an interrupted prior download must always be
	// retried — never reused — since curl was writing to it byte-by-byte.
	if (existsSync(partPath)) {
		console.log(
			`Found stale partial download at ${partPath}; removing before retry.`,
		);
		unlinkSync(partPath);
	}

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
	// Explicit `ggufFilePattern` wins over the MLC quant probes — used when
	// the GGUF repo's filenames don't follow MLC's `q…` naming (e.g. BERT
	// encoder GGUFs ship as `*-F16.GGUF`).
	const preferredQuants = [
		...(model.ggufFilePattern ? [model.ggufFilePattern.toLowerCase()] : []),
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

	// Trust the local cache only after verifying the size matches the
	// remote manifest — guards against a previous run being killed mid-
	// download with the destination written but truncated.
	if (existsSync(destPath)) {
		const localSize = statSync(destPath).size;
		if (localSize === chosenFile.size) {
			return;
		}
		console.log(
			`Local cache at ${destPath} is ${localSize} bytes but remote reports ${chosenFile.size}; refetching.`,
		);
		unlinkSync(destPath);
	} else {
		console.log(
			`\nModel ${model.id} not found locally. Preparing to download...`,
		);
	}

	const downloadUrl = `https://huggingface.co/${repoName}/resolve/main/${chosenFile.path}`;
	console.log(
		`Downloading ${chosenFile.path} (${(chosenFile.size / 1024 / 1024).toFixed(1)} MB) to ${destPath}...`,
	);

	// Atomic-write: curl writes to .part, then rename on success. A SIGTERM
	// during the curl leaves only the .part, which the top-of-function
	// guard removes before re-attempting — never a half-written destPath.
	execFileSync("curl", ["-L", "--progress-bar", "-o", partPath, downloadUrl], {
		stdio: "inherit",
	});
	const partSize = statSync(partPath).size;
	if (partSize !== chosenFile.size) {
		unlinkSync(partPath);
		throw new Error(
			`Downloaded ${partSize} bytes but remote reports ${chosenFile.size}; refusing to publish a truncated cache file.`,
		);
	}
	renameSync(partPath, destPath);
	console.log("Download complete.\n");
}

function readAgentchromePort(): string | null {
	try {
		const out = execFileSync("agentchrome", ["connect", "--status"], {
			encoding: "utf-8",
		});
		const parsed = JSON.parse(out) as {
			port?: string | number;
			active?: boolean;
			reachable?: boolean;
		};
		if (parsed.port && parsed.reachable !== false) return String(parsed.port);
		return null;
	} catch {
		return null;
	}
}

async function ensureAgentchromePort(): Promise<{
	port: string;
	launched: boolean;
}> {
	const existing = readAgentchromePort();
	if (existing) return { port: existing, launched: false };

	console.log("agentchrome: no active session — launching a headed Chrome…");
	try {
		// agentchrome defaults to headed; `--headless` would opt out. We need
		// headed because WebGPU doesn't work reliably in headless.
		execFileSync("agentchrome", ["connect", "--launch"], {
			stdio: ["ignore", "inherit", "inherit"],
		});
	} catch {
		// `connect --launch` can exit non-zero if another connect is racing us;
		// the poll below is authoritative.
	}

	const deadline = Date.now() + 15_000;
	while (Date.now() < deadline) {
		const port = readAgentchromePort();
		if (port) return { port, launched: true };
		await sleep(500);
	}
	throw new Error(
		"agentchrome: failed to establish a session within 15s. Try `agentchrome connect --launch` manually, then re-run.",
	);
}

/**
 * Probe an existing Chrome tab for WebGPU support. Returns true if
 * `navigator.gpu.requestAdapter()` resolves to an adapter, false otherwise.
 * Used only on the session-reuse path — when we launched Chrome ourselves
 * we use agentchrome's default (headed) and trust it.
 */
async function tabHasWebGpu(port: string, tab: string): Promise<boolean> {
	const script = `(async () => {
		if (!navigator.gpu || typeof navigator.gpu.requestAdapter !== "function") {
			return JSON.stringify({ ok: false, reason: "no-navigator-gpu" });
		}
		try {
			const adapter = await navigator.gpu.requestAdapter();
			return JSON.stringify({ ok: !!adapter, reason: adapter ? "ok" : "no-adapter" });
		} catch (err) {
			return JSON.stringify({ ok: false, reason: String(err?.message ?? err) });
		}
	})()`;
	try {
		const out = agentchrome(port, tab, ["js", "exec", script]);
		const resp = JSON.parse(out) as { result?: string };
		if (typeof resp.result !== "string") return false;
		const parsed = JSON.parse(resp.result) as { ok: boolean; reason: string };
		if (!parsed.ok) {
			console.warn(`agentchrome: tab WebGPU probe failed — ${parsed.reason}`);
		}
		return parsed.ok;
	} catch (err) {
		console.warn(
			`agentchrome: WebGPU probe threw (${err instanceof Error ? err.message : String(err)}); assuming no GPU`,
		);
		return false;
	}
}

function listAgentchromeTabs(
	port: string,
): Array<{ id: string; url: string }> {
	const raw = execFileSync("agentchrome", ["--port", port, "tabs", "list"], {
		encoding: "utf-8",
	});
	return JSON.parse(raw) as Array<{ id: string; url: string }>;
}

function findSmokeTab(
	tabs: Array<{ id: string; url: string }>,
	page: SmokeTestPage,
): string | null {
	const pagePath =
		page === "debug" ? "real-model-debug.html" : "real-model.html";
	const exact = tabs.find((entry) => entry.url.includes(pagePath));
	if (exact) return exact.id;
	const any = tabs.find(
		(entry) =>
			entry.url.includes("real-model.html") ||
			entry.url.includes("real-model-debug.html"),
	);
	return any?.id ?? null;
}

async function createSmokeTab(
	port: string,
	page: SmokeTestPage,
): Promise<string> {
	const pagePath =
		page === "debug" ? "real-model-debug.html" : "real-model.html";
	const bootUrl = `http://localhost:8031/${pagePath}`;
	let createdId: string | null = null;
	try {
		const raw = execFileSync(
			"agentchrome",
			["--port", port, "tabs", "create", bootUrl],
			{ encoding: "utf-8" },
		);
		try {
			const parsed = JSON.parse(raw) as { id?: string };
			if (typeof parsed.id === "string") createdId = parsed.id;
		} catch {
			// Not every agentchrome build returns JSON; fall back to re-listing.
		}
	} catch (err) {
		throw new Error(
			`agentchrome: failed to create a smoke tab on port ${port}: ${err instanceof Error ? err.message : String(err)}`,
		);
	}
	if (createdId) return createdId;

	const deadline = Date.now() + 5000;
	while (Date.now() < deadline) {
		const tabs = listAgentchromeTabs(port);
		const match = findSmokeTab(tabs, page);
		if (match) return match;
		await sleep(250);
	}
	throw new Error(
		`agentchrome: created a tab but could not locate it by URL match on port ${port}`,
	);
}

export async function resolveAgentchromeSession(
	portArg?: string,
	tabArg?: string,
	page: SmokeTestPage = "smoke",
): Promise<{ port: string; tab: string }> {
	let port: string;
	let launchedByUs: boolean;
	if (portArg) {
		port = portArg;
		launchedByUs = false;
	} else {
		const resolved = await ensureAgentchromePort();
		port = resolved.port;
		launchedByUs = resolved.launched;
	}

	let tab = await resolveTab(port, tabArg, page);

	// WebGPU requires headed Chrome. Our own launches use agentchrome's
	// default (headed) so we trust them. On reuse, probe — and if the
	// probe fails (almost always a headless or CDP-frozen leftover),
	// recycle the session once with a fresh launch before giving up. If
	// the user pinned --port/--tab explicitly, don't second-guess them.
	if (!launchedByUs && !portArg) {
		const gpu = await tabHasWebGpu(port, tab);
		if (!gpu) {
			console.warn(
				"agentchrome: reused Chrome session has no WebGPU adapter — recycling and relaunching once…",
			);
			stopAgentchromeSession(port);
			const fresh = await ensureAgentchromePort();
			if (!fresh.launched) {
				throw new Error(
					"agentchrome: failed to launch a fresh session after recycling. Run `make agentchrome-stop` and try again.",
				);
			}
			port = fresh.port;
			tab = await resolveTab(port, undefined, page);
			const gpu2 = await tabHasWebGpu(port, tab);
			if (!gpu2) {
				throw new Error(
					[
						"agentchrome: fresh Chrome session still has no WebGPU adapter.",
						"This Chrome build / profile cannot run WebGPU. Run `make agentchrome-stop`,",
						"then verify your Chrome supports WebGPU (chrome://gpu) and re-run.",
					].join("\n  "),
				);
			}
		}
	}

	return { port, tab };
}

async function resolveTab(
	port: string,
	tabArg: string | undefined,
	page: SmokeTestPage,
): Promise<string> {
	if (tabArg) return tabArg;
	const existing = findSmokeTab(listAgentchromeTabs(port), page);
	if (existing) return existing;
	console.log(
		`agentchrome: no smoke tab on port ${port} — creating one on the smoke-test page…`,
	);
	return await createSmokeTab(port, page);
}

function stopAgentchromeSession(port: string): void {
	try {
		execFileSync("agentchrome", ["connect", "--disconnect"], {
			stdio: ["ignore", "ignore", "ignore"],
		});
	} catch {
		// best-effort
	}
	try {
		// Kill anything still listening on the CDP port (the launched Chrome).
		execFileSync("sh", ["-c", `lsof -ti:${port} | xargs kill 2>/dev/null`], {
			stdio: ["ignore", "ignore", "ignore"],
		});
	} catch {
		// best-effort
	}
}

export async function waitForSmokeTestResult(
	port: string,
	tab: string,
): Promise<SmokeTestResult> {
	const deadline = Date.now() + 360_000;
	const script = `(() => {
		const pattern = new RegExp("Generated (\\\\d+) tokens in ([0-9.]+)s \\\\(prefill: (\\\\d+)ms, decode: (\\\\d+)ms, ([0-9.]+) tok\\\\/s(?:, finish=([^)]+))?\\\\)");
		const t = document.getElementById("log")?.textContent ?? "";
		const m = t.match(pattern);
		if (!m) return "";
		return JSON.stringify({
			tokensGenerated: +m[1],
			totalMs: +m[2] * 1000,
			prefillMs: +m[3],
			decodeMs: +m[4],
			tokensPerSecond: +m[5],
			finishReason: m[6] ? m[6].trim() : undefined,
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

export interface EmbedPerfTrace {
	mode: "single" | "batch";
	fixture: string;
	wallMs: number;
	rep?: number;
	trial?: number;
	count?: number;
}

/**
 * Wait for the smoke page's embedPerf loop to finish and return the
 * collected traces. Distinguished from waitForSmokeTestResult by
 * looking for the "[embedPerf] mode=…" log line and pulling
 * window.__embedTraces.
 */
export async function waitForEmbedPerfResult(
	port: string,
	tab: string,
): Promise<EmbedPerfTrace[]> {
	const deadline = Date.now() + 360_000;
	const doneScript = `(() => {
		const t = document.getElementById("log")?.textContent ?? "";
		return t.includes("[embedPerf] mode=") ? "1" : "";
	})()`;
	let lastError: unknown;
	while (Date.now() < deadline) {
		try {
			const done = evalStringResult(port, tab, doneScript);
			if (done === "1") {
				const out = agentchrome(port, tab, [
					"js",
					"exec",
					`(() => JSON.stringify(window.__embedTraces ?? []))()`,
				]);
				const resp = JSON.parse(out) as { result?: string; output_file?: string };
				const payload = typeof resp.result === "string" ? resp.result : "";
				return JSON.parse(payload || "[]") as EmbedPerfTrace[];
			}
		} catch (error) {
			lastError = error;
		}
		await new Promise((resolve) => setTimeout(resolve, 1000));
	}
	throw new Error(
		`Timed out waiting for embedPerf result line${lastError ? ` (${String(lastError)})` : ""}`,
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

export async function extractSmokeTestAssistant(
	port: string,
	tab: string,
): Promise<string> {
	try {
		const out = agentchrome(port, tab, [
			"js",
			"exec",
			`(() => {
				const steps = document.querySelectorAll("#log .step");
				for (let i = steps.length - 1; i >= 0; i--) {
					const text = steps[i].textContent ?? "";
					const m = text.match(/^Assistant:\\s*([\\s\\S]*)$/);
					if (m) return m[1].trim();
				}
				return "";
			})()`,
		]);
		const resp = JSON.parse(out) as { result?: string };
		return typeof resp.result === "string" ? resp.result : "";
	} catch {
		return "";
	}
}

export interface ChatSmokeMetrics {
	genTokens?: number;
	tokensPerSecond?: number;
	totalMs?: number;
}

export interface ChatSmokeResult {
	assistantText: string;
	finishReason: string;
	chatOutput: string;
	metrics: ChatSmokeMetrics;
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
	const deadline = Date.now() + 360_000;
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
		const metricsPattern = new RegExp("\\\\((\\\\d+) tokens, ([0-9.]+) tok/s, ([0-9.]+)s, finish=");
		const assistantMatch = delta.match(assistantPattern);
		const finishMatch = delta.match(finishPattern);
		const metricsMatch = delta.match(metricsPattern);
		const assistantText = assistantMatch?.[1]?.trim() ?? "";
		if (!assistantText || !finishMatch) return "";
		const metrics = metricsMatch
			? {
				genTokens: +metricsMatch[1],
				tokensPerSecond: +metricsMatch[2],
				totalMs: +metricsMatch[3] * 1000,
			}
			: {};
		return JSON.stringify({
			ok: true,
			assistantText,
			finishReason: finishMatch[1].trim(),
			chatOutput: currentText,
			metrics,
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
					metrics: ChatSmokeMetrics;
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
