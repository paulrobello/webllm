export function shouldAutoInsertBos(tokenizerConfig) {
	return tokenizerConfig?.addBosToken !== false;
}

export function shouldRunSmokeDiagnostics(params) {
	return params.has("debug") || params.has("diag");
}

export function getSmokePageCopy(debugMode) {
	return debugMode
		? {
				title: "WebLLM Real Model Debug",
				subtitle: "Load model, generate text, and run deep diagnostics",
			}
		: {
				title: "WebLLM Real Model Test",
				subtitle: "Load model and generate text",
			};
}

export function getSmokePageShellMarkup() {
	return `
<h1 id="title">WebLLM Real Model Test</h1>
<p id="subtitle" class="subtitle">Load model and generate text</p>

<div id="progress-bar"><div id="progress-fill"><span id="progress-text">0%</span></div></div>
<div id="log"></div>

<div id="chat-container">
<div id="chat-output"></div>
<input id="chat-input" type="text" placeholder="Type a message..." disabled>
<button id="chat-btn" disabled>Generate</button>
</div>`.trim();
}

export function getSmokeChatOptions(
	parsed,
	detectChatTemplate,
	chatTemplate,
	overrides = {},
) {
	if (
		parsed?.hyperparams?.architecture === "qwen3" &&
		detectChatTemplate(chatTemplate ?? "") === "chatml"
	) {
		return overrides.enableThinking === true ? {} : { enableThinking: false };
	}
	return {};
}

export function getThinkingModeFromParams(params) {
	const v = params?.get?.("thinking");
	return v === "1" || v === "on" || v === "true";
}

/**
 * Returns true iff the parsed model has a chat template that wires up
 * thinking-block control. This matches the engine's gate in
 * `chat-template.ts::shouldCloseThinkBlock` (template must reference
 * both `enable_thinking` and `<think>`) and is the canonical signal
 * for "this model is actually a thinking model". Llama, Mistral,
 * Qwen2/2.5, SmolLM2, etc. all return false here. Qwen3 returns true.
 *
 * BERT-arch encoders never support thinking — they have no
 * generative loop at all. Catch them up front for a cleaner error.
 */
export function modelSupportsThinking(parsed) {
	if (!parsed?.hyperparams || !parsed.tokenizerConfig) return false;
	if (parsed.hyperparams.architecture === "bert") return false;
	const tmpl = parsed.tokenizerConfig.chatTemplate ?? "";
	return tmpl.includes("enable_thinking") && tmpl.includes("<think>");
}

export function getSmokeSamplingConfig(
	parsed,
	detectChatTemplate,
	chatTemplate,
	chatOptions = {},
) {
	const isQwen3Chatml =
		parsed?.hyperparams?.architecture === "qwen3" &&
		detectChatTemplate(chatTemplate ?? "") === "chatml";
	if (isQwen3Chatml && chatOptions.enableThinking === false) {
		return {
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
			seed: 12345,
		};
	}
	if (isQwen3Chatml) {
		return {
			temperature: 0.6,
			topK: 20,
			topP: 0.95,
			repetitionPenalty: 1.05,
			seed: 12345,
		};
	}
	return {
		temperature: 0.7,
		topK: 40,
		topP: 0.95,
		repetitionPenalty: 1.05,
		seed: 12345,
	};
}

export function createSmokeSamplerFactory({
	Sampler,
	parsedModel,
	detectChatTemplate,
	samplingOverrides = {},
}) {
	return function makeSmokeSampler(chatTemplate, chatOptions = {}) {
		const base = getSmokeSamplingConfig(
			parsedModel,
			detectChatTemplate,
			chatTemplate,
			chatOptions,
		);
		return new Sampler({ ...base, ...samplingOverrides });
	};
}

export function getSmokeSamplingOverridesFromParams(params) {
	const overrides = {};
	const num = (key) => {
		const raw = params?.get?.(key);
		if (raw === null || raw === undefined || raw === "") return undefined;
		const v = Number(raw);
		return Number.isFinite(v) ? v : undefined;
	};
	const temperature = num("temp");
	if (temperature !== undefined) overrides.temperature = temperature;
	const topK = num("topK");
	if (topK !== undefined) overrides.topK = Math.floor(topK);
	const topP = num("topP");
	if (topP !== undefined) overrides.topP = topP;
	const repetitionPenalty = num("rep");
	if (repetitionPenalty !== undefined)
		overrides.repetitionPenalty = repetitionPenalty;
	const seed = num("seed");
	if (seed !== undefined) overrides.seed = Math.floor(seed);
	return overrides;
}

export function sanitizeDisplayText(text) {
	return text
		.replace(/^\s+/, "")
		.replace(/\n{4,}/g, "\n\n")
		.trimStart();
}

export function buildSmokePrompt(
	userMessage,
	chatOptions,
	encodeChatPrompt,
	tokenizer,
) {
	return {
		mode: "chat",
		tokens: encodeChatPrompt(
			[{ role: "user", content: userMessage }],
			tokenizer,
			chatOptions,
		),
	};
}

export function findSingleTokenProbe(tokenizer, candidates) {
	for (const text of candidates) {
		const ids = tokenizer.encode(text);
		if (ids.length === 1) return { text, ids };
	}
	return null;
}

/**
 * Smoke completion runner: thin adapter over the library's
 * `engine.chatCompletion` so the smoke page exercises the same decode
 * pipeline that public consumers do. Output shape matches the legacy
 * runner so callers (`real-model-page.js`, `real-model-runtime.js`,
 * smoke-bench profiling) keep working.
 */
export function createSmokeCompletionRunner({
	engine,
	handleId,
	inference,
	tokenizer,
	log,
	profileMode = false,
}) {
	return async function runCompletion({
		label,
		messages,
		samplingConfig,
		maxTokens,
		chatOptions = {},
	}) {
		const tStart = performance.now();
		log("running", `  ${label} messages: ${messages.length}`);

		const generatedIds = [];
		let stats = null;
		let yieldCount = 0;

		for await (const chunk of engine.chatCompletion(handleId, messages, {
			...samplingConfig,
			maxTokens,
			enableThinking: chatOptions.enableThinking,
		})) {
			if (chunk.done) {
				stats = chunk.stats ?? null;
				continue;
			}
			if (chunk.tokenId === undefined) continue;
			yieldCount++;
			generatedIds.push(chunk.tokenId);
			// Match the legacy runner: only push decode-step traces, not the
			// prefill trace (yieldCount === 1 covers the prefill+first-sample).
			if (yieldCount > 1 && profileMode && inference?.lastTrace) {
				window.__decodeTraces.push({ ...inference.lastTrace });
			}
		}

		const totalTime = performance.now() - tStart;
		const prefillMs = stats?.timeToFirstTokenMs ?? 0;
		const finishReason = stats?.finishReason ?? "max-tokens";
		const outputText = tokenizer.decode(generatedIds);
		const rawOutputText = tokenizer.decode(generatedIds, {
			includeSpecialTokens: true,
		});

		return {
			outputText,
			rawOutputText,
			displayOutputText: sanitizeDisplayText(outputText),
			genTokens: generatedIds.length,
			finishReason,
			prefillMs,
			genTime: Math.max(0, totalTime - prefillMs),
			totalTime,
		};
	};
}

export function createPrefillComparisonRunner({ inference, tokenizer, log }) {
	return async function compareBatchVsSequentialPrefill(label, promptTokens) {
		inference.resetKVCache();
		const batchLogits = await inference.forward(
			new Int32Array(promptTokens),
			new Int32Array(promptTokens.map((_, i) => i)),
		);

		inference.resetKVCache();
		let seqLogits = null;
		for (let i = 0; i < promptTokens.length; i++) {
			seqLogits = await inference.forward(
				new Int32Array([promptTokens[i]]),
				new Int32Array([i]),
			);
		}

		if (!seqLogits) {
			throw new Error(`no sequential logits for ${label}`);
		}

		let diffCount = 0;
		let maxAbsDiff = 0;
		let sumAbsDiff = 0;
		for (let i = 0; i < batchLogits.length; i++) {
			const diff = Math.abs(batchLogits[i] - seqLogits[i]);
			if (diff > 1e-5) diffCount++;
			if (diff > maxAbsDiff) maxAbsDiff = diff;
			sumAbsDiff += diff;
		}

		const topBatch = Array.from(batchLogits)
			.map((value, index) => [index, value])
			.sort((a, b) => b[1] - a[1])
			.slice(0, 5)
			.map(([tokenId, value]) => {
				const token = tokenizer.getToken(tokenId);
				return `${tokenId}:"${token ? token.text : "?"}"(${value.toFixed(2)})`;
			});
		const topSeq = Array.from(seqLogits)
			.map((value, index) => [index, value])
			.sort((a, b) => b[1] - a[1])
			.slice(0, 5)
			.map(([tokenId, value]) => {
				const token = tokenizer.getToken(tokenId);
				return `${tokenId}:"${token ? token.text : "?"}"(${value.toFixed(2)})`;
			});

		log(
			"running",
			`  ${label} batch-vs-seq diff=${diffCount}/${batchLogits.length} maxAbs=${maxAbsDiff.toExponential(3)} meanAbs=${(sumAbsDiff / batchLogits.length).toExponential(3)}`,
		);
		log("running", `  ${label} batch top5: ${topBatch.join(", ")}`);
		log("running", `  ${label} seq   top5: ${topSeq.join(", ")}`);
		inference.resetKVCache();
	};
}
