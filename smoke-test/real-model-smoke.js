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

export function getSmokeChatOptions(parsed, detectChatTemplate, chatTemplate) {
	if (
		parsed?.hyperparams?.architecture === "qwen3" &&
		detectChatTemplate(chatTemplate ?? "") === "chatml"
	) {
		return { enableThinking: false };
	}
	return {};
}

export function getSmokeSamplingConfig(
	parsed,
	detectChatTemplate,
	chatTemplate,
	chatOptions = {},
) {
	if (
		parsed?.hyperparams?.architecture === "qwen3" &&
		detectChatTemplate(chatTemplate ?? "") === "chatml" &&
		chatOptions.enableThinking === false
	) {
		return {
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
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
}) {
	return function makeSmokeSampler(chatTemplate, chatOptions = {}) {
		return new Sampler(
			getSmokeSamplingConfig(
				parsedModel,
				detectChatTemplate,
				chatTemplate,
				chatOptions,
			),
		);
	};
}

export function sanitizeDisplayText(text) {
	return text.replace(/^\s+/, "").replace(/\n{4,}/g, "\n\n").trimStart();
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

function decodeForDebug(tokenizer, tokenIds, includeSpecialTokens = false) {
	return tokenizer.decode(tokenIds, { includeSpecialTokens });
}

function getForbiddenReentryTokens(parsed, detectChatTemplate, tokenizer) {
	if (parsed?.hyperparams?.architecture !== "qwen3") return [];
	const chatTemplate = parsed?.tokenizerConfig?.chatTemplate ?? "";
	if (detectChatTemplate(chatTemplate) !== "chatml") return [];
	const imStartId = tokenizer.getId("<|im_start|>");
	return imStartId === undefined ? [] : [imStartId];
}

function getThinkingTokenIds(parsed, tokenizer) {
	if (parsed?.hyperparams?.architecture !== "qwen3") return null;
	const thinkOpenId = tokenizer.getId("<think>");
	const thinkCloseId = tokenizer.getId("</think>");
	if (thinkOpenId === undefined || thinkCloseId === undefined) return null;
	return { thinkOpenId, thinkCloseId };
}

function getMaskedTokensWhileThinking(thinkingTokenIds, tokenizer) {
	if (!thinkingTokenIds) return [];
	const imStartId = tokenizer.getId("<|im_start|>");
	const imEndId = tokenizer.getId("<|im_end|>");
	return [thinkingTokenIds.thinkOpenId, imStartId, imEndId].filter(
		(id) => id !== undefined,
	);
}

function getMaskedTokensAfterThinkingUntilAnswer(thinkingTokenIds, tokenizer) {
	if (!thinkingTokenIds) return [];
	const imStartId = tokenizer.getId("<|im_start|>");
	const imEndId = tokenizer.getId("<|im_end|>");
	const toolCallOpenId = tokenizer.getId("<tool_call>");
	const toolCallCloseId = tokenizer.getId("</tool_call>");
	const toolResponseOpenId = tokenizer.getId("<tool_response>");
	const toolResponseCloseId = tokenizer.getId("</tool_response>");
	return [
		thinkingTokenIds.thinkOpenId,
		imStartId,
		imEndId,
		toolCallOpenId,
		toolCallCloseId,
		toolResponseOpenId,
		toolResponseCloseId,
	].filter((id) => id !== undefined);
}

function maskTokenLogits(logits, tokenIds) {
	for (const tokenId of tokenIds) {
		if (tokenId >= 0 && tokenId < logits.length) {
			logits[tokenId] = -Infinity;
		}
	}
}

function isVisibleTextToken(tokenizer, tokenId) {
	const token = tokenizer.getToken(tokenId);
	if (!token) return false;
	if (token.attr & (4 | 8)) return false;
	return tokenizer.decode([tokenId]).trim().length > 0;
}

function isWhitespaceOnlyTextToken(tokenizer, tokenId) {
	const token = tokenizer.getToken(tokenId);
	if (!token) return false;
	if (token.attr & (4 | 8)) return false;
	return tokenizer.decode([tokenId]).trim().length === 0;
}

export function createSmokeCompletionRunner({
	parsed,
	tokenizer,
	inference,
	detectChatTemplate,
	log,
	profileMode = false,
}) {
	return async function runCompletion(label, promptTokens, sampler, maxGen = 64) {
		const tStart = performance.now();
		const logits = await inference.forward(
			new Int32Array(promptTokens),
			new Int32Array(promptTokens.map((_, i) => i)),
		);
		const prefillMs = performance.now() - tStart;
		const forbiddenReentryTokens = new Set(
			getForbiddenReentryTokens(parsed, detectChatTemplate, tokenizer),
		);
		const thinkingTokenIds = getThinkingTokenIds(parsed, tokenizer);
		const maskedTokensWhileThinking = getMaskedTokensWhileThinking(
			thinkingTokenIds,
			tokenizer,
		);
		const maskedTokensAfterThinkingUntilAnswer =
			getMaskedTokensAfterThinkingUntilAnswer(thinkingTokenIds, tokenizer);
		const requireVisibleAnswerAfterThinking = thinkingTokenIds !== null;
		let thinkDepth = 0;
		let hasVisibleAnswerText = false;
		let waitingForVisibleAnswer = false;

		log("running", `  ${label} prompt tokens: ${promptTokens.length}`);
		{
			const arr = Array.from(logits).map((v, i) => [i, v]);
			arr.sort((a, b) => b[1] - a[1]);
			const top = arr.slice(0, 10).map(([id, v]) => {
				const token = tokenizer.getToken(id);
				return `${id}:"${token ? token.text : "?"}"(${v.toFixed(2)})`;
			});
			log("running", `  ${label} prefill top10: ${top.join(", ")}`);
		}

		const eosId = parsed.tokenizerConfig.eosTokenId;
		const generated = [];
		const recent = [...promptTokens];
		let finishReason = "max-tokens";
		sampler.applyRepetitionPenalty(logits, recent.slice(-64));
		let sampledId = sampler.sample(logits);
		if (thinkingTokenIds && sampledId === thinkingTokenIds.thinkOpenId) {
			thinkDepth = 1;
		} else if (
			thinkingTokenIds &&
			sampledId === thinkingTokenIds.thinkCloseId
		) {
			return {
				outputText: "",
				rawOutputText: "",
				displayOutputText: "",
				genTokens: 0,
				finishReason: "stop-token",
				prefillMs,
				genTime: performance.now() - tStart - prefillMs,
				totalTime: performance.now() - tStart,
			};
		}
		generated.push(sampledId);
		recent.push(sampledId);

		for (let step = 0; step < maxGen; step++) {
			if (sampledId === eosId) {
				finishReason = "eos";
				break;
			}
			const pos = promptTokens.length + step;
			if (sampler.isGreedy && sampler.noPenalty) {
				const result = await inference.forwardDecode(
					new Int32Array([sampledId]),
					new Int32Array([pos]),
					"greedy",
				);
				if (profileMode && inference.lastTrace) {
					window.__decodeTraces.push({ ...inference.lastTrace });
				}
				sampledId = result.tokenId;
			} else {
				const stepLogits = await inference.forward(
					new Int32Array([sampledId]),
					new Int32Array([pos]),
				);
				if (profileMode && inference.lastTrace) {
					window.__decodeTraces.push({ ...inference.lastTrace });
				}
				sampler.applyRepetitionPenalty(stepLogits, recent.slice(-64));
				if (thinkDepth > 0) {
					maskTokenLogits(stepLogits, maskedTokensWhileThinking);
				} else if (waitingForVisibleAnswer) {
					maskTokenLogits(stepLogits, maskedTokensAfterThinkingUntilAnswer);
					if (requireVisibleAnswerAfterThinking) {
						maskTokenLogits(stepLogits, [eosId]);
					}
				} else if (hasVisibleAnswerText) {
					maskTokenLogits(stepLogits, maskedTokensAfterThinkingUntilAnswer);
				}
				sampledId = sampler.sample(stepLogits);
				if (waitingForVisibleAnswer) {
					const maskedWhitespaceOnly = new Set();
					while (
						isWhitespaceOnlyTextToken(tokenizer, sampledId) &&
						!maskedWhitespaceOnly.has(sampledId)
					) {
						maskedWhitespaceOnly.add(sampledId);
						maskTokenLogits(stepLogits, [sampledId]);
						sampledId = sampler.sample(stepLogits);
					}
				}
			}
			if (sampledId === eosId) {
				finishReason = "eos";
				break;
			}
			if (thinkingTokenIds && sampledId === thinkingTokenIds.thinkOpenId) {
				if (thinkDepth > 0) {
					finishReason = "stop-token";
					break;
				}
				thinkDepth++;
			} else if (
				thinkingTokenIds &&
				sampledId === thinkingTokenIds.thinkCloseId
			) {
				if (thinkDepth === 0) {
					finishReason = "stop-token";
					break;
				}
				thinkDepth = Math.max(0, thinkDepth - 1);
				if (thinkDepth === 0) {
					waitingForVisibleAnswer = requireVisibleAnswerAfterThinking;
				}
			}
			if (
				waitingForVisibleAnswer &&
				!hasVisibleAnswerText &&
				isVisibleTextToken(tokenizer, sampledId)
			) {
				hasVisibleAnswerText = true;
				waitingForVisibleAnswer = false;
			}
			if (generated.length >= 1 && forbiddenReentryTokens.has(sampledId)) {
				finishReason = "stop-token";
				break;
			}
			generated.push(sampledId);
			recent.push(sampledId);
		}

		const totalTime = performance.now() - tStart;
		const outputText = decodeForDebug(tokenizer, generated);
		const rawOutputText = decodeForDebug(tokenizer, generated, true);
		return {
			outputText,
			rawOutputText,
			displayOutputText: sanitizeDisplayText(outputText),
			genTokens: generated.length,
			finishReason,
			prefillMs,
			genTime: totalTime - prefillMs,
			totalTime,
		};
	};
}

export function createPrefillComparisonRunner({
	inference,
	tokenizer,
	log,
}) {
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
