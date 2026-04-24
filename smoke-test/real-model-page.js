export async function runRealModelPage({ debugMode = false } = {}) {
	const assetSuffix = window.location.search || "";
	const {
		GgufParser,
		GgmlWasm,
		ModelInference,
		ModelLoader,
		Sampler,
		Tokenizer,
		WebLLM,
		collectBrowserSystemProfile,
		detectChatTemplate,
		encodeChatPrompt,
		runTasks,
		score,
	} = await import(`./webllm-bundle.js${assetSuffix}`);
	const { runInteractiveChatTurn } = await import(
		`./real-model-runtime.js${assetSuffix}`
	);
	const {
		buildSmokePrompt,
		createPrefillComparisonRunner,
		createSmokeCompletionRunner,
		createSmokeSamplerFactory,
		findSingleTokenProbe,
		getSmokeChatOptions,
		getSmokePageCopy,
		getSmokePageShellMarkup,
		getSmokeSamplingOverridesFromParams,
		getThinkingModeFromParams,
		shouldAutoInsertBos,
	} = await import(`./real-model-smoke.js${assetSuffix}`);

	const params = new URLSearchParams(window.location.search);
	const thinkingEnabled = getThinkingModeFromParams(params);
	const maxTokensParam = Number(params.get("max"));
	const maxTokensOverride =
		Number.isFinite(maxTokensParam) && maxTokensParam > 0
			? Math.floor(maxTokensParam)
			: null;
	const samplingOverrides = getSmokeSamplingOverridesFromParams(params);
	const promptOverride = params.get("prompt");
	const profileName = params.get("profile");
	const benchTaskListId = params.get("bench");
	const benchIngestUrl = params.get("ingest") || "";
	const DEFAULT_MODEL_ID = "qwen3-0.6b-q4f16";
	const DEFAULT_CONTEXT_LENGTH = 4096;
	const modelId = params.get("model") || DEFAULT_MODEL_ID;
	const requestedContextLength = Number(
		params.get("ctx") || DEFAULT_CONTEXT_LENGTH,
	);
	const modelUrl = `./models/${modelId}.gguf`;
	document.body.innerHTML = getSmokePageShellMarkup();

	const logEl = document.getElementById("log");
	const titleEl = document.getElementById("title");
	const subtitleEl = document.getElementById("subtitle");
	const progressEl = document.getElementById("progress-bar");
	const progressFill = document.getElementById("progress-fill");
	const progressText = document.getElementById("progress-text");
	const chatContainer = document.getElementById("chat-container");
	const chatOutput = document.getElementById("chat-output");
	const chatInput = document.getElementById("chat-input");
	const chatBtn = document.getElementById("chat-btn");

	const pageCopy = getSmokePageCopy(debugMode);
	document.title = pageCopy.title;
	titleEl.textContent = pageCopy.title;
	subtitleEl.textContent = pageCopy.subtitle;

	const modeBar = document.createElement("div");
	modeBar.id = "mode-bar";
	const modePill = document.createElement("span");
	modePill.className = `mode-pill ${thinkingEnabled ? "on" : "off"}`;
	modePill.textContent = `Thinking: ${thinkingEnabled ? "ON" : "OFF"}`;
	const modeToggle = document.createElement("a");
	const toggleParams = new URLSearchParams(window.location.search);
	if (thinkingEnabled) toggleParams.delete("thinking");
	else toggleParams.set("thinking", "1");
	toggleParams.set("v", String(Date.now()));
	modeToggle.href = `?${toggleParams.toString()}`;
	modeToggle.className = "mode-toggle";
	modeToggle.textContent = thinkingEnabled ? "switch off" : "switch on";
	modeBar.appendChild(modePill);
	modeBar.appendChild(modeToggle);
	if (profileName) {
		const profilePill = document.createElement("span");
		profilePill.className = "mode-pill profile";
		profilePill.textContent = `Profile: ${profileName}`;
		modeBar.appendChild(profilePill);
	}
	subtitleEl.insertAdjacentElement("afterend", modeBar);

	function log(cls, msg) {
		const el = document.createElement("div");
		el.className = `step ${cls}`;
		el.textContent = msg;
		logEl.appendChild(el);
		logEl.scrollTop = logEl.scrollHeight;
	}

	function setProgress(pct) {
		progressFill.style.width = `${pct}%`;
		progressText.textContent = `${pct.toFixed(0)}%`;
	}

	let inference = null;
	let tokenizer = null;
	let parsedModel = null;
	let wasmInstance = null;
	let interactiveRunCompletion = null;
	let makeSmokeSampler = null;

	async function loadAndTest() {
		const t0 = performance.now();
		const profileMode = new URLSearchParams(window.location.search).has("profile");

		log("running", "[1/7] Fetching model...");
		progressEl.style.display = "block";
		let modelBuffer;
		try {
			const resp = await fetch(modelUrl);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const total = Number(resp.headers.get("content-length") || 0);
			const reader = resp.body.getReader();
			const chunks = [];
			let received = 0;
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				chunks.push(value);
				received += value.length;
				if (total > 0) setProgress((received / total) * 30);
			}
			modelBuffer = new Uint8Array(received);
			let off = 0;
			for (const chunk of chunks) {
				modelBuffer.set(chunk, off);
				off += chunk.length;
			}
			modelBuffer = modelBuffer.buffer;
			log(
				"pass",
				`[1/7] Model fetched: ${(received / 1e6).toFixed(1)} MB in ${((performance.now() - t0) / 1000).toFixed(1)}s`,
			);
		} catch (e) {
			log("fail", `[1/7] Fetch failed: ${e.message}`);
			return;
		}

		log("running", "[2/7] Parsing GGUF...");
		let ggufCtx;
		let parsed;
		try {
			ggufCtx = GgufParser.parse(modelBuffer);
			parsed = ModelLoader.parseModel(modelBuffer);
			parsedModel = parsed;
			const hp = parsed.hyperparams;
			const subtitleContextLength =
				requestedContextLength > 0
					? Math.min(parsed.kvCacheConfig.maxContextLength, requestedContextLength)
					: parsed.kvCacheConfig.maxContextLength;
			subtitleEl.textContent =
				`${pageCopy.subtitle} · Model: ${modelId} · arch=${hp.architecture} · ctx=${subtitleContextLength}`;
			log(
				"pass",
				`[2/7] GGUF parsed: arch=${hp.architecture} emb=${hp.embeddingLength} heads=${hp.headCount}/${hp.headCountKv} layers=${hp.layerCount} vocab=${hp.vocabularySize} ctx=${hp.contextLength}`,
			);
		} catch (e) {
			log("fail", `[2/7] Parse failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		log("running", "[3/7] Initializing WebGPU backend...");
		try {
			const wasm = new GgmlWasm();
			wasmInstance = wasm;
			await wasm.init({ wasmUrl: `./webllm-wasm.js${assetSuffix}` });
			log("pass", "[3/7] WebGPU backend initialized");

			log("running", "[4/7] Loading weights into GPU...");
			setProgress(35);
			inference = new ModelInference(wasm, parsed.hyperparams);
			if (profileMode) {
				inference.traceEnabled = true;
				window.__decodeTraces = [];
			}
			const t1 = performance.now();
			inference.loadWeights(ggufCtx, modelBuffer);
			const weightTime = ((performance.now() - t1) / 1000).toFixed(1);
			log("pass", `[4/7] Weights loaded in ${weightTime}s`);
			setProgress(80);
		} catch (e) {
			log("fail", `[3/7-4/7] Init/load failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		log("running", "[5/7] Initializing KV cache...");
		try {
			const kvContextLength =
				requestedContextLength > 0
					? Math.min(parsed.kvCacheConfig.maxContextLength, requestedContextLength)
					: parsed.kvCacheConfig.maxContextLength;
			await inference.initKVCache(kvContextLength);
			log(
				"pass",
				`[5/7] KV cache: ${kvContextLength} slots x ${parsed.hyperparams.layerCount} layers`,
			);
			setProgress(85);
		} catch (e) {
			log("fail", `[5/7] KV cache failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		log("running", "[6/7] Creating tokenizer...");
		try {
			tokenizer = new Tokenizer(parsed.tokenizerConfig);
			const testEncode = tokenizer.encode("hello");
			log(
				"pass",
				`[6/7] Tokenizer ready: vocab=${tokenizer.vocabSize}, encode("hello")=[${testEncode}]`,
			);
			window.inference = inference;
			window.tokenizer = tokenizer;
			window.parsedModel = parsed;
		} catch (e) {
			log("fail", `[6/7] Tokenizer failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		if (debugMode) {
			log("running", "[debug] Deep diagnostics enabled");

			log("running", "[7-norm] F32 norm weight probe...");
			try {
				const attn = await inference.debugReadNormWeight("attn0", 8);
				const ffn = await inference.debugReadNormWeight("ffn0", 8);
				const out = await inference.debugReadNormWeight("output", 8);
				log(
					"running",
					`  blk.0.attn_norm first8: [${Array.from(attn).map((v) => v.toFixed(4)).join(",")}]`,
				);
				log(
					"running",
					`  blk.0.ffn_norm  first8: [${Array.from(ffn).map((v) => v.toFixed(4)).join(",")}]`,
				);
				log(
					"running",
					`  output_norm     first8: [${Array.from(out).map((v) => v.toFixed(4)).join(",")}]`,
				);
			} catch (e) {
				log("fail", `[7-norm] failed: ${e.message}`);
			}

			log("running", "[7-debug] KV write probe...");
			try {
				inference.resetKVCache();
				const probeTokens = tokenizer.encode("hello");
				await inference.forward(
					new Int32Array(probeTokens),
					new Int32Array(probeTokens.map((_, i) => i)),
				);
				const k0 = await inference.debugReadKCache(0, 64 * 4, 0);
				const nzK = Array.from(k0).filter((v) => Math.abs(v) > 1e-9).length;
				const sumAbsK = Array.from(k0).reduce((sum, v) => sum + Math.abs(v), 0);
				log(
					"running",
					`  kv.k[layer=0][pos=0][head=0] nonzero=${nzK}/64 sumAbs=${sumAbsK.toFixed(4)}`,
				);
				const v0 = await inference.debugReadVCache(0, 64 * 4, 0);
				const nzV = Array.from(v0).filter((v) => Math.abs(v) > 1e-9).length;
				const sumAbsV = Array.from(v0).reduce((sum, v) => sum + Math.abs(v), 0);
				log(
					"running",
					`  kv.v[layer=0][pos=0..63][dim=0][head=0] nonzero=${nzV}/64 sumAbs=${sumAbsV.toFixed(4)} first4=[${Array.from(v0.slice(0, 4)).map((v) => v.toFixed(4)).join(",")}]`,
				);
			} catch (e) {
				log("fail", `[7-debug] KV probe failed: ${e.message}\n${e.stack || ""}`);
			}

			log("running", "[7-sanity] single-token continuation...");
			try {
				inference.resetKVCache();
				const sanityText = shouldAutoInsertBos(parsed.tokenizerConfig)
					? "The"
					: "Hello";
				const sanityIds = tokenizer.encode(sanityText);
				const sanityTokens = shouldAutoInsertBos(parsed.tokenizerConfig)
					? [tokenizer.bosId, ...sanityIds]
					: sanityIds;
				const logits = await inference.forward(
					new Int32Array(sanityTokens),
					new Int32Array(sanityTokens.map((_, i) => i)),
				);
				const arr = Array.from(logits).map((v, i) => [i, v]);
				arr.sort((a, b) => b[1] - a[1]);
				const top5 = arr.slice(0, 5).map(([id, v]) => {
					const token = tokenizer.getToken(id);
					return `${id}:"${token ? token.text : "?"}"(${v.toFixed(2)})`;
				});
				log("running", `  ${JSON.stringify(sanityText)} → ${top5.join(", ")}`);
			} catch (e) {
				log("fail", `[7-sanity] failed: ${e.message}`);
			}

			log("running", "[7a] KV history differential...");
			try {
				const probeInfo = findSingleTokenProbe(tokenizer, [
					"?",
					".",
					"!",
					" a",
					" I",
				]);
				const prefixAInfo = findSingleTokenProbe(tokenizer, [
					"hello",
					"Hello",
					"The",
				]);
				const prefixBInfo = findSingleTokenProbe(tokenizer, [
					"world",
					"time",
					"story",
				]);
				if (!probeInfo || !prefixAInfo || !prefixBInfo) {
					throw new Error("could not find single-token probes for this tokenizer");
				}

				async function logitsAfterPrefix(prefix, probe) {
					inference.resetKVCache();
					await inference.forward(
						new Int32Array(prefix),
						new Int32Array(prefix.map((_, i) => i)),
					);
					return inference.forward(
						new Int32Array(probe),
						new Int32Array([prefix.length]),
					);
				}

				const lA = await logitsAfterPrefix(prefixAInfo.ids, probeInfo.ids);
				const lB = await logitsAfterPrefix(prefixBInfo.ids, probeInfo.ids);
				let diffCount = 0;
				let maxAbsDiff = 0;
				for (let i = 0; i < lA.length; i++) {
					const diff = Math.abs(lA[i] - lB[i]);
					if (diff > 1e-5) diffCount++;
					if (diff > maxAbsDiff) maxAbsDiff = diff;
				}
				const topA = Array.from(lA).reduce((m, v, i, a) => (a[m] >= v ? m : i), 0);
				const topB = Array.from(lB).reduce((m, v, i, a) => (a[m] >= v ? m : i), 0);
				const verdict = diffCount > 100 ? "HISTORY MATTERS" : "HISTORY IGNORED";
				log(
					"running",
					`  probe=${JSON.stringify(probeInfo.text)} [${probeInfo.ids}] prefixA=${JSON.stringify(prefixAInfo.text)} prefixB=${JSON.stringify(prefixBInfo.text)}`,
				);
				log(
					"running",
					`  diff logits=${diffCount}/${lA.length}, maxAbsDiff=${maxAbsDiff.toExponential(3)}`,
				);
				log(
					"running",
					`  topA=${topA} (${lA[topA].toFixed(3)})  topB=${topB} (${lB[topB].toFixed(3)})  => ${verdict}`,
				);
				inference.resetKVCache();
			} catch (e) {
				log("fail", `[7a] KV diagnostic failed: ${e.message}\n${e.stack || ""}`);
			}
		}

		log("running", "[7/7] Generating text...");
		setProgress(90);
		try {
			if (debugMode) {
				for (const prompt of ["The", "The quick brown", "Hello, how are you"]) {
					inference.resetKVCache();
					const tks = shouldAutoInsertBos(parsed.tokenizerConfig)
						? [tokenizer.bosId, ...tokenizer.encode(prompt)]
						: tokenizer.encode(prompt);
					const lgts = await inference.forward(
						new Int32Array(tks),
						new Int32Array(tks.map((_, i) => i)),
					);
					const arr = Array.from(lgts).map((v, i) => [i, v]);
					arr.sort((a, b) => b[1] - a[1]);
					const top = arr.slice(0, 5).map(([id, v]) => {
						const token = tokenizer.getToken(id);
						return `${id}:"${token ? token.text : "?"}"(${v.toFixed(2)})`;
					});
					log("running", `  ${JSON.stringify(prompt)} → ${top.join(", ")}`);
				}
				inference.resetKVCache();
			}

			const userMessage = promptOverride || "Tell one short joke.";
			const chatTmpl = parsed.tokenizerConfig.chatTemplate;
			makeSmokeSampler = createSmokeSamplerFactory({
				Sampler,
				parsedModel: parsed,
				detectChatTemplate,
				samplingOverrides,
			});
			const smokeChatOptions = getSmokeChatOptions(
				parsed,
				detectChatTemplate,
				chatTmpl,
				{ enableThinking: thinkingEnabled },
			);
			const smokePrompt = buildSmokePrompt(
				userMessage,
				smokeChatOptions,
				encodeChatPrompt,
				tokenizer,
			);
			const compareBatchVsSequentialPrefill = createPrefillComparisonRunner({
				inference,
				tokenizer,
				log,
			});

			if (debugMode) {
				await compareBatchVsSequentialPrefill(smokePrompt.mode, smokePrompt.tokens);
			}

			const runCompletion = createSmokeCompletionRunner({
				parsed,
				tokenizer,
				inference,
				detectChatTemplate,
				log,
				profileMode,
			});
			interactiveRunCompletion = runCompletion;

			const smokeSampler = makeSmokeSampler(chatTmpl, smokeChatOptions);
			const smokeMaxTokens =
				maxTokensOverride ?? (thinkingEnabled ? 1024 : 64);
			const smokeResult = await runCompletion(
				smokePrompt.mode,
				smokePrompt.tokens,
				smokeSampler,
				smokeMaxTokens,
			);

			log(
				"pass",
				`[7/7] Generated ${smokeResult.genTokens} tokens in ${(smokeResult.totalTime / 1000).toFixed(1)}s (prefill: ${smokeResult.prefillMs.toFixed(0)}ms, decode: ${smokeResult.genTime.toFixed(0)}ms, ${(smokeResult.genTokens / (smokeResult.genTime / 1000)).toFixed(1)} tok/s, finish=${smokeResult.finishReason})`,
			);
			log("pass", `User: ${userMessage}`);
			const assistantText = thinkingEnabled
				? smokeResult.rawOutputText ||
					smokeResult.displayOutputText ||
					smokeResult.outputText
				: smokeResult.displayOutputText || smokeResult.outputText;
			log("pass", `Assistant: ${assistantText}`);
			setProgress(100);
		} catch (e) {
			log("fail", `[7/7] Generation failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		// Best-effort: collect + register the system profile whenever an
		// ingest URL is known. Speed runs (chat-smoke) read the resulting
		// window.__webllmSystemId via agentchrome scrape; bench mode reads
		// it directly. Failure here never blocks the run — it's metadata.
		if (benchIngestUrl && !window.__webllmSystemId) {
			try {
				if (navigator.gpu) {
					const sysAdapter = await navigator.gpu.requestAdapter();
					if (sysAdapter) {
						const profile = await collectBrowserSystemProfile(sysAdapter);
						window.__webllmSystemId = profile.systemId;
						await fetch(`${benchIngestUrl}/system-profiles`, {
							method: "POST",
							headers: { "content-type": "application/json" },
							body: JSON.stringify(profile),
						});
						log(
							"pass",
							`[sys] ${profile.gpuVendor ?? "?"} · ${profile.gpuArchitecture ?? "?"} · Chrome ${profile.chromeVersion ?? "?"} (id ${profile.systemId})`,
						);
					}
				}
			} catch (err) {
				console.warn(`system-profile collection failed: ${err}`);
			}
		}

		if (benchTaskListId) {
			if (!benchIngestUrl) {
				log(
					"fail",
					"[bench] ?bench=<id> requires ?ingest=<live-server-url> too",
				);
				window.__benchStatus = { done: true, error: "missing ingest url" };
				return;
			}
			try {
				if (!wasmInstance || !inference || !parsedModel) {
					throw new Error(
						"bench mode reached after [7/7] but wasm/inference/parsed are unset — did the smoke steps fail silently?",
					);
				}
				const { runBenchMode } = await import(
					`./real-model-bench.js${assetSuffix}`
				);
				await runBenchMode({
					WebLLM,
					runTasks,
					score,
					collectBrowserSystemProfile,
					wasm: wasmInstance,
					inference,
					parsed: parsedModel,
					modelId,
					taskListId: benchTaskListId,
					ingestUrl: benchIngestUrl,
					log,
					setProgress,
					profileName,
					thinking: thinkingEnabled,
					params: {
						contextLength: Number.isFinite(requestedContextLength)
							? requestedContextLength
							: undefined,
						maxTokens: maxTokensOverride ?? undefined,
						temperature: samplingOverrides.temperature,
						topK: samplingOverrides.topK,
						topP: samplingOverrides.topP,
						repetitionPenalty: samplingOverrides.repetitionPenalty,
						seed: samplingOverrides.seed,
					},
				});
			} catch (e) {
				log("fail", `[bench] failed: ${e.message}\n${e.stack || ""}`);
				window.__benchStatus = {
					...(window.__benchStatus ?? {}),
					done: true,
					error: e.message,
				};
			}
			// Bench mode replaces interactive chat — don't reveal the chat input.
			return;
		}

		chatContainer.style.display = "block";
		chatInput.disabled = false;
		chatBtn.disabled = false;
		chatInput.focus();
	}

	async function runChat() {
		const text = chatInput.value.trim();
		if (!text) return;
		chatBtn.disabled = true;
		chatInput.disabled = true;
		chatOutput.textContent += `User: ${text}\nAssistant: `;

		try {
			if (!window._chatSession) {
				window._chatSession = {
					position: 0,
					history: [],
					messages: [],
					prevCount: 0,
				};
			}
			const { session, result } = await runInteractiveChatTurn({
				text,
				session: window._chatSession,
				parsedModel,
				tokenizer,
				inference,
				interactiveRunCompletion,
				makeSmokeSampler,
				getSmokeChatOptions: (nextParsedModel, chatTemplate) =>
					getSmokeChatOptions(nextParsedModel, detectChatTemplate, chatTemplate, {
						enableThinking: thinkingEnabled,
					}),
				encodeChatPrompt,
			});
			window._chatSession = session;
			const renderedText = thinkingEnabled
				? result.rawText || result.fullText
				: result.fullText;
			chatOutput.textContent = chatOutput.textContent.replace(
				/Assistant: [\s\S]*$/,
				`Assistant: ${renderedText}`,
			);

			const elapsed = (result.totalTime / 1000).toFixed(2);
			const tps =
				result.genTime > 0
					? (result.genTokens / (result.genTime / 1000)).toFixed(1)
					: "0.0";
			chatOutput.textContent += `\n(${result.genTokens} tokens, ${tps} tok/s, ${elapsed}s, finish=${result.finishReason})\n\n`;
		} catch (e) {
			chatOutput.textContent += `[Error: ${e.message}]\n\n`;
		}

		chatBtn.disabled = false;
		chatInput.disabled = false;
		chatInput.value = "";
		chatInput.focus();
	}

	chatBtn.addEventListener("click", runChat);
	chatInput.addEventListener("keydown", (e) => {
		if (e.key === "Enter" && !chatBtn.disabled) runChat();
	});

	await loadAndTest();
}
