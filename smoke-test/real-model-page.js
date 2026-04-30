// §D embed-perf measurement helper. No-op when `mode` is null/unrecognized,
// so existing smoke runs (which never set ?embedPerf=...) are unaffected.
// Driven externally by eval/embed-perf.ts via cache-busted page navigations
// that scrape `window.__embedTraces` after the page settles.
async function runEmbedPerfHook(engine, handleId, mode, reps, fixture, log) {
	if (mode !== "single" && mode !== "batch") return;

	const SHORT_TEXT = "happy";
	const LONG_TEXT =
		"Compilers translate human-readable source code into instructions a " +
		"computer can execute. The translation usually runs in several stages: " +
		"a lexer breaks the input into tokens, a parser assembles those tokens " +
		"into a syntax tree, a semantic analyser checks the tree for meaning, " +
		"and a code generator emits machine code or bytecode for some target " +
		"architecture. Modern compilers add an optimiser between the analyser " +
		"and the generator that reorders, inlines, and rewrites the program in " +
		"ways that preserve its observable behaviour while reducing its runtime " +
		"or code size.";
	const batchMixed = [];
	for (let i = 0; i < 32; i++) batchMixed.push(SHORT_TEXT);
	for (let i = 0; i < 32; i++) batchMixed.push(LONG_TEXT);

	const fixtureMap = {
		short: [SHORT_TEXT],
		long: [LONG_TEXT],
		batchMixed: batchMixed,
	};
	const texts = fixtureMap[fixture] ?? [SHORT_TEXT];

	window.__embedTraces = [];

	if (mode === "single") {
		const fixtureText = texts[0];
		// 5-rep warmup
		for (let i = 0; i < 5; i++) {
			await engine.embed(handleId, fixtureText);
		}
		// Measured reps
		for (let i = 0; i < reps; i++) {
			const t0 = performance.now();
			await engine.embed(handleId, fixtureText);
			const t1 = performance.now();
			window.__embedTraces.push({
				mode: "single",
				fixture: fixture,
				rep: i,
				wallMs: t1 - t0,
			});
		}
	} else {
		// batch mode — Phase 1 baseline runs sequentially via engine.embed.
		// Phase 4 will swap this for engine.embedBatch when that lands.
		for (const txt of texts) {
			await engine.embed(handleId, txt);
		}
		const trials = Number.isFinite(reps) && reps > 0 ? reps : 3;
		for (let trial = 0; trial < trials; trial++) {
			const t0 = performance.now();
			for (const txt of texts) {
				await engine.embed(handleId, txt);
			}
			const t1 = performance.now();
			window.__embedTraces.push({
				mode: "batch",
				fixture: fixture,
				trial: trial,
				count: texts.length,
				wallMs: t1 - t0,
			});
		}
	}

	log(
		"pass",
		`[embedPerf] mode=${mode} fixture=${fixture} traces=${window.__embedTraces.length}`,
	);
}

export async function runRealModelPage({ debugMode = false } = {}) {
	const assetSuffix = window.location.search || "";
	const {
		CausalLMEmbedder,
		EncoderInference,
		GgufParser,
		GgmlWasm,
		ModelInference,
		ModelLoader,
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
		findSingleTokenProbe,
		getSmokeChatOptions,
		getSmokePageCopy,
		getSmokePageShellMarkup,
		getSmokeSamplingConfig,
		getSmokeSamplingOverridesFromParams,
		getThinkingModeFromParams,
		modelSupportsThinking,
		shouldAutoInsertBos,
	} = await import(`./real-model-smoke.js${assetSuffix}`);

	const params = new URLSearchParams(window.location.search);
	// `?wasm=mem64` toggles the wasm64 binary; default loads webllm-wasm.js
	// (wasm32, no regression for ≤3.5 GiB models).
	//
	// **This page is the manual debug surface.** No size-aware auto-routing
	// here — for >4 GiB models pass `&wasm=mem64` explicitly. Without it,
	// the wasm32 heap can't fit the GGUF and `wasm.malloc(N)` returns null
	// at step 2.
	//
	// The eval harness (`make bench-browser-eval`, `make smoke-bench`) auto-
	// routes via `profileToUrlParams` in `eval/smoke-profiles.ts`, which
	// mirrors `pickWasmUrl` in `src/core/engine.ts` (the canonical
	// modelByteLength-driven decision used by the public `WebLLM.from*`
	// constructors at runtime).
	const wasmVariant =
		params.get("wasm") === "mem64" ? "webllm-wasm-mem64.js" : "webllm-wasm.js";
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
	// Resolve dashboard ingest URL with a "default-on" policy:
	//   • `?ingest=<url>`   → use that URL.
	//   • `?ingest=off`     → disable ingest entirely.
	//   • (no `?ingest=`)   → default to http://localhost:8033 so plain
	//                         smoke runs auto-record. Failed POSTs are
	//                         best-effort and never block the run.
	const ingestParam = params.get("ingest");
	const benchIngestUrl = (() => {
		if (ingestParam === null) return "http://localhost:8033";
		const trimmed = ingestParam.trim();
		if (trimmed.length === 0) return "";
		if (trimmed.toLowerCase() === "off") return "";
		return trimmed;
	})();
	const DEFAULT_MODEL_ID = "qwen3-0.6b-q4f16";
	const DEFAULT_CONTEXT_LENGTH = 4096;
	const modelId = params.get("model") || DEFAULT_MODEL_ID;
	const requestedContextLength = Number(
		params.get("ctx") || DEFAULT_CONTEXT_LENGTH,
	);
	const modelUrl = `./models/${modelId}.gguf`;
	// Speculative-decoding lever: when `?drafter=<id>` is set the page
	// brings up a second model alongside the target and routes chat
	// through CompletionConfig.drafter / draftLength so ship-gate
	// measurements exercise the public API end-to-end.
	const drafterId = params.get("drafter") || null;
	const drafterDraftLengthParam = Number(params.get("draftLength"));
	const drafterDraftLength =
		Number.isFinite(drafterDraftLengthParam) && drafterDraftLengthParam > 0
			? Math.floor(drafterDraftLengthParam)
			: null;
	// §4 Flash Attention gate: ?fa=on toggles ggml_flash_attn_ext + the
	// FA-ready V-cache layout. Default off — preserves §18-revert behavior.
	const flashAttnEnabled = params.get("fa") === "on";
	// Bucket D embedding gate: ?embeddingCapable=1 dispatches to the hidden-state
	// embedding path when adopted into the engine. Gates parity testing for
	// causal-LM-derived embedders (Qwen3-Embedding, etc).
	const embeddingCapable =
		params.get("embeddingCapable") === "1" ||
		params.get("embeddingCapable") === "true";
	// Bucket D pooling mode: ?embeddingPooling=mean overrides the default
	// last-token pool. Used for chat models with high last-token anisotropy
	// (e.g., Phi-3.5-mini). Ignored unless ?embeddingCapable=1.
	const embeddingPoolingParam = params.get("embeddingPooling");
	const embeddingPooling =
		embeddingPoolingParam === "mean" ? "mean" : "last-token";
	// §22 prefill-tiling diagnostic: ?diagnoseAlloc=1 dumps WebGPU device
	// limits to #log at startup. No engine work — caller follows up with a
	// long-prefill request and watches console for the abort to capture
	// the offending buffer size.
	const diagnoseAlloc = params.get("diagnoseAlloc") === "1";
	// §22 prefill-tiling gate: `?prefillTile=N` forces a specific tile size.
	// When the URL param is absent, we leave the ctor opt undefined and let
	// `ModelInference`'s hyperparam-derived default decide (see §30 +
	// `computeDefaultPrefillTileSize` in src/inference/model-inference.ts).
	// Pass `?prefillTile=0` to force-disable on a 7B+ model.
	const prefillTileParam = params.get("prefillTile");
	let prefillTileOverride; // undefined → ctor heuristic decides
	if (prefillTileParam !== null) {
		const raw = Number(prefillTileParam);
		prefillTileOverride =
			Number.isFinite(raw) && raw >= 0 ? Math.floor(raw) : 0;
	}
	// §D embed-perf measurement loop (driven by eval/embed-perf.ts harness).
	// `embedPerf` enables the hook; null = no-op (existing smoke unaffected).
	const embedPerfMode = params.get("embedPerf"); // null | "single" | "batch"
	const embedRepsRaw = params.get("embedReps");
	const embedReps = embedRepsRaw ? Number.parseInt(embedRepsRaw, 10) : 30;
	const embedFixture = params.get("embedFixture") ?? "short";
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
	const faPill = document.createElement("span");
	faPill.className = `mode-pill ${flashAttnEnabled ? "on" : "off"}`;
	faPill.textContent = `FA: ${flashAttnEnabled ? "ON" : "OFF"}`;
	modeBar.appendChild(faPill);
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
	let smokeEngine = null;
	let smokeEngineHandleId = null;
	let drafterHandleId = null;

	async function loadAndTest() {
		const t0 = performance.now();
		const profileMode = new URLSearchParams(window.location.search).has("profile");

		// WebGPU init moved to step 1: subsequent steps stream the GGUF
		// directly into the WASM heap, so the heap must exist first.
		// JS-heap `new Uint8Array(N)` caps at ~2 GiB on Chrome and fails
		// for >2 GiB GGUFs (4B Q4 ≈ 2.27 GiB). A `Uint8Array` *view* over
		// the WASM-backed ArrayBuffer can exceed 2 GiB.
		log("running", "[1/8] Initializing WebGPU backend...");
		let wasm;
		try {
			wasm = new GgmlWasm();
			wasmInstance = wasm;
			await wasm.init({ wasmUrl: `./${wasmVariant}${assetSuffix}` });
			log("pass", "[1/8] WebGPU backend initialized");
		} catch (e) {
			log("fail", `[1/8] WebGPU init failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		if (diagnoseAlloc && navigator.gpu) {
			try {
				const adapter = await navigator.gpu.requestAdapter();
				const device = await adapter.requestDevice();
				const lim = device.limits;
				const dump = {
					maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
					maxBufferSize: lim.maxBufferSize,
					maxStorageBuffersPerShaderStage: lim.maxStorageBuffersPerShaderStage,
					maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
					maxBindGroups: lim.maxBindGroups,
				};
				log("running", `[diagnoseAlloc] WebGPU device limits: ${JSON.stringify(dump)}`);
				device.destroy();
			} catch (e) {
				log("fail", `[diagnoseAlloc] failed to query device: ${e.message}`);
			}
		}

		log("running", "[2/8] Fetching model...");
		progressEl.style.display = "block";
		// Saved (ptr, length) is the source of truth: WASM memory growth
		// during ctxCreate / backendAllocCtxTensors detaches any prior
		// Uint8Array view, so views must be re-derived from the live
		// `wasm.heapU8.buffer` at each use via the dataAt callback below.
		let modelPtr = 0;
		let modelByteLength = 0;
		const fetchStart = performance.now();
		try {
			const resp = await fetch(modelUrl);
			if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
			const total = Number(resp.headers.get("content-length") || 0);
			if (total <= 0) {
				throw new Error(
					"missing content-length on model response; streaming into WASM heap requires it",
				);
			}
			modelPtr = wasm.malloc(total);
			if (!modelPtr) throw new Error(`wasm malloc(${total}) returned null`);
			modelByteLength = total;
			const reader = resp.body.getReader();
			let received = 0;
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				// Re-fetch heapU8 each chunk: malloc above and any future
				// growth invalidate prior buffer references. Module.HEAPU8
				// is re-bound by Emscripten on grow; the getter returns
				// the current view.
				wasm.heapU8.set(value, modelPtr + received);
				received += value.length;
				setProgress((received / total) * 30);
			}
			if (received !== total) {
				throw new Error(
					`short read: expected ${total} bytes, got ${received}`,
				);
			}
			log(
				"pass",
				`[2/8] Model fetched: ${(received / 1e6).toFixed(1)} MB in ${((performance.now() - fetchStart) / 1000).toFixed(1)}s`,
			);
		} catch (e) {
			log("fail", `[2/8] Fetch failed: ${e.message}`);
			if (modelPtr) wasm.free(modelPtr);
			return;
		}

		// Build the model-region accessor. Re-derives a fresh sub-view of
		// HEAPU8 on every call so callers can't accidentally hold a
		// stale reference across a memory-grow event.
		const modelDataAt = (off, len) =>
			new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);

		log("running", "[3/8] Parsing GGUF...");
		let ggufCtx;
		let parsed;
		try {
			const fullView = modelDataAt(0, modelByteLength);
			ggufCtx = GgufParser.parse(fullView);
			parsed = ModelLoader.parseModel(fullView);
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
				`[3/8] GGUF parsed: arch=${hp.architecture} emb=${hp.embeddingLength} heads=${hp.headCount}/${hp.headCountKv} layers=${hp.layerCount} vocab=${hp.vocabularySize} ctx=${hp.contextLength}`,
			);
		} catch (e) {
			log("fail", `[3/8] Parse failed: ${e.message}\n${e.stack || ""}`);
			wasm.free(modelPtr);
			return;
		}

		// Reject upfront if `?thinking=1` was set on a model whose chat
		// template doesn't actually wire up thinking. Without this guard
		// the engine silently runs with non-thinking semantics but the
		// dashboard records `thinking: "on"`, polluting thinking-on/off
		// comparison panels with runs that aren't real thinking. Encoder
		// models also can't think (no generative loop). Fail fast — we
		// haven't loaded weights to GPU yet so the abort is cheap.
		if (thinkingEnabled && !modelSupportsThinking(parsed)) {
			const archLabel = parsed.hyperparams.architecture;
			log(
				"fail",
				`Thinking mode requested (?thinking=1) but model "${modelId}" ` +
					`(arch=${archLabel}) does not support thinking. Only Qwen3-family ` +
					`models with chat templates that reference both \`enable_thinking\` ` +
					`and \`<think>\` are real thinking models. Remove the ` +
					`thinking flag (or use the "switch off" link above) and reload.`,
			);
			setProgress(0);
			wasm.free(modelPtr);
			return;
		}

		// BERT-arch GGUFs are bidirectional encoders — they have no causal
		// generation, no KV cache, and a different weight set than llama-style
		// causal LMs. Steps [4-5,7] are routed through EncoderInference for
		// these models; the causal-LM ModelInference path would crash with
		// `Weight "output_norm.weight" not found` because that tensor only
		// exists on causal models. Mirrors `isEncoderArchitecture` from
		// src/core/types.ts — keep both in sync when adding encoder archs.
		const isEncoderModel =
			parsed.hyperparams.architecture === "bert" ||
			parsed.hyperparams.architecture === "nomic-bert" ||
			parsed.hyperparams.architecture === "jina-bert-v2";
		// Causal-LM-derived embedders (e.g. Qwen3-Embedding) share the qwen3
		// weight layout but are pooled at the last token and never decode.
		// They route through CausalLMEmbedder, skip the KV cache, and use
		// engine.embed() for warmup/generation gates — same as encoders
		// from the smoke page's perspective. Mirrors
		// `isCausalEmbedderArchitecture` from src/core/types.ts.
		const isCausalEmbedderModel =
			parsed.hyperparams.architecture === "qwen3-embedding";
		// Unified gate for everything the smoke page treats identically
		// across encoder + causal-embedder paths: skip KV init, warm up via
		// embed(), skip [7/8] generation, skip the [8/8] second-engine
		// reference encoder load.
		const isEmbedderModel = isEncoderModel || isCausalEmbedderModel;
		// Bucket D: embeddingCapable chat model running in embed-perf bench
		// mode. Shares the normal ModelInference load path (KV cache init +
		// chatCompletion warmup) but skips step-7 chat generation and routes
		// step-8 to the embed-perf hook on the already-loaded engine rather
		// than loading the arctic-embed-s reference encoder.
		const isBucketDEmbedPerf = embeddingCapable && embedPerfMode !== null;

		log("running", "[4/8] Loading weights into GPU...");
		let loadFailed = false;
		try {
			setProgress(35);
			if (isEncoderModel) {
				inference = new EncoderInference(wasm, parsed.hyperparams);
			} else if (isCausalEmbedderModel) {
				inference = new CausalLMEmbedder(wasm, parsed.hyperparams);
			} else {
				inference = new ModelInference(wasm, parsed.hyperparams, {
					flashAttn: flashAttnEnabled,
					prefillTileSize: prefillTileOverride,
				});
				if (profileMode) {
					inference.traceEnabled = true;
					window.__decodeTraces = [];
				}
				const resolvedTile = inference.prefillTileSize;
				if (resolvedTile > 0) {
					const tilePill = document.createElement("span");
					tilePill.className = "mode-pill on";
					tilePill.textContent = `tile: ${resolvedTile}`;
					modeBar.appendChild(tilePill);
				}
			}
			const t1 = performance.now();
			inference.loadWeights(ggufCtx, modelDataAt);
			const weightTime = ((performance.now() - t1) / 1000).toFixed(1);
			log("pass", `[4/8] Weights loaded in ${weightTime}s`);
			setProgress(80);
		} catch (e) {
			loadFailed = true;
			log("fail", `[4/8] Load failed: ${e.message}\n${e.stack || ""}`);
		} finally {
			// Weights are uploaded to GPU buffers; the staging copy in
			// the WASM heap is no longer needed. Freeing immediately
			// reclaims ~model-file-size of WASM memory before the KV
			// cache and graph buffers get allocated.
			wasm.free(modelPtr);
			modelPtr = 0;
		}
		if (loadFailed) return;

		if (isEmbedderModel) {
			log(
				"pass",
				`[5/8] KV cache: skipped (embedder model — no autoregressive cache)`,
			);
			setProgress(85);
		} else {
			log("running", "[5/8] Initializing KV cache...");
			try {
				const kvContextLength =
					requestedContextLength > 0
						? Math.min(parsed.kvCacheConfig.maxContextLength, requestedContextLength)
						: parsed.kvCacheConfig.maxContextLength;
				await inference.initKVCache(kvContextLength);
				log(
					"pass",
					`[5/8] KV cache: ${kvContextLength} slots x ${parsed.hyperparams.layerCount} layers`,
				);
				setProgress(85);
			} catch (e) {
				log("fail", `[5/8] KV cache failed: ${e.message}\n${e.stack || ""}`);
				return;
			}
		}

		log("running", "[6/8] Creating tokenizer...");
		try {
			tokenizer = new Tokenizer(parsed.tokenizerConfig);
			const testEncode = tokenizer.encode("hello");
			log(
				"pass",
				`[6/8] Tokenizer ready: vocab=${tokenizer.vocabSize}, encode("hello")=[${testEncode}]`,
			);
			window.inference = inference;
			window.tokenizer = tokenizer;
			window.parsedModel = parsed;
		} catch (e) {
			log("fail", `[6/8] Tokenizer failed: ${e.message}\n${e.stack || ""}`);
			return;
		}

		// Build the WebLLM engine and adopt the already-loaded inference
		// pipeline. Routes [7/8], the interactive chat box, and bench mode
		// through `engine.chatCompletion` / `engine.embed` so the smoke
		// page exercises the same decode path public consumers do.
		try {
			if (!navigator.gpu) {
				throw new Error("navigator.gpu unavailable; smoke test needs WebGPU");
			}
			smokeEngine = await WebLLM.init({
				memoryBudget: 2_000_000_000,
			});
			const smokeEngineHandle = await smokeEngine.adoptPreloadedModel(
				modelId,
				{ wasm: wasmInstance, inference, parsed },
				{ embeddingCapable, embeddingPooling },
			);
			smokeEngineHandleId = smokeEngineHandle.id;
			// Expose for external harnesses (e.g. eval/encoder-parity.ts):
			// the parity harness drives engine.embed via agentchrome js-exec.
			window.engine = smokeEngine;
			window.handleId = smokeEngineHandleId;
		} catch (e) {
			log(
				"fail",
				`[6/8] Engine construction failed: ${e.message}\n${e.stack || ""}`,
			);
			return;
		}

		// Optional drafter for speculative decoding. Mirrors the target's
		// load path: fetch the GGUF, bring up a *separate* `GgmlWasm` heap
		// (heaps are per-model in this codebase — sharing one between
		// target + drafter would let one model's malloc invalidate the
		// other's HEAPU8 view), construct a `ModelInference`, and adopt
		// it under a second handle id on the same engine.
		if (drafterId) {
			log("running", `[drafter] Loading ${drafterId}...`);
			const drafterUrl = `./models/${drafterId}.gguf${assetSuffix}`;
			let drafterPtr = 0;
			let drafterByteLength = 0;
			let drafterWasm = null;
			try {
				drafterWasm = new GgmlWasm();
				await drafterWasm.init({ wasmUrl: `./${wasmVariant}${assetSuffix}` });
				const drafterResp = await fetch(drafterUrl);
				if (!drafterResp.ok) {
					throw new Error(`HTTP ${drafterResp.status} fetching ${drafterUrl}`);
				}
				const drafterTotal = Number(
					drafterResp.headers.get("content-length") || 0,
				);
				if (drafterTotal <= 0) {
					throw new Error(
						"missing content-length on drafter response; streaming into WASM heap requires it",
					);
				}
				drafterPtr = drafterWasm.malloc(drafterTotal);
				if (!drafterPtr) {
					throw new Error(`drafter wasm malloc(${drafterTotal}) returned null`);
				}
				drafterByteLength = drafterTotal;
				const drafterReader = drafterResp.body.getReader();
				let drafterReceived = 0;
				while (true) {
					const { done, value } = await drafterReader.read();
					if (done) break;
					// Re-derive heapU8 each chunk — same memory-grow rationale
					// as the target fetch above.
					drafterWasm.heapU8.set(value, drafterPtr + drafterReceived);
					drafterReceived += value.length;
				}
				if (drafterReceived !== drafterTotal) {
					throw new Error(
						`drafter short read: expected ${drafterTotal} bytes, got ${drafterReceived}`,
					);
				}
				const drafterDataAt = (off, len) =>
					new Uint8Array(
						drafterWasm.heapU8.buffer,
						drafterPtr + off,
						len,
					);
				const drafterFullView = drafterDataAt(0, drafterByteLength);
				const drafterGgufCtx = GgufParser.parse(drafterFullView);
				const drafterParsed = ModelLoader.parseModel(drafterFullView);
				const drafterInference = new ModelInference(
					drafterWasm,
					drafterParsed.hyperparams,
					{ prefillTileSize: prefillTileOverride },
				);
				drafterInference.loadWeights(drafterGgufCtx, drafterDataAt);
				const drafterCtxLen =
					requestedContextLength > 0
						? Math.min(
								drafterParsed.kvCacheConfig.maxContextLength,
								requestedContextLength,
							)
						: drafterParsed.kvCacheConfig.maxContextLength;
				await drafterInference.initKVCache(drafterCtxLen);
				// Free the staging copy now that weights are on the GPU —
				// same reasoning as the target free at [4/8].
				drafterWasm.free(drafterPtr);
				drafterPtr = 0;
				// `loadModel` (called inside `adoptPreloadedModel`) mints a
				// synthetic handle id; that's the key under which the drafter
				// gets registered in `inferenceEngines`. The user-facing
				// drafter name is just a registry hint — pass `handle.id`
				// (not `drafterId`) into `CompletionConfig.drafter` below.
				const drafterHandle = await smokeEngine.adoptPreloadedModel(
					drafterId,
					{
						wasm: drafterWasm,
						inference: drafterInference,
						parsed: drafterParsed,
					},
				);
				drafterHandleId = drafterHandle.id;
				log(
					"pass",
					`[drafter] ${drafterId} loaded (ctx=${drafterCtxLen}, draftLength=${drafterDraftLength ?? "default"})`,
				);
			} catch (e) {
				if (drafterPtr && drafterWasm) drafterWasm.free(drafterPtr);
				log(
					"fail",
					`[drafter] Load failed: ${e.message}\n${e.stack || ""}`,
				);
				return;
			}
		}

		// Shader-cache warmup. Cold WebGPU pipelines compile on first dispatch,
		// which on Apple Metal can take ~15s for a transformer graph and
		// dominates any subsequent tok/s measurement. Run a tiny generation /
		// embedding here so [7/8], the bench harness, and the interactive chat
		// box all measure steady-state numbers. Result is discarded; KV cache
		// is reset automatically by the next `chatCompletion` call.
		//
		// Uses realistic sampling (temperature 0.6, topK 40, repetition
		// penalty 1.05) so the topk decode pipeline compiles here. A
		// `temperature: 0` warmup would only compile the greedy/full pipeline
		// and the first realistic-sampler measurement would still pay topk
		// shader-compile cost. Two warmup tokens are enough — first decode
		// step covers compile, second confirms steady state.
		try {
			const warmupStart = performance.now();
			if (isEmbedderModel) {
				await smokeEngine.embed(smokeEngineHandleId, "warmup");
			} else {
				const stream = smokeEngine.chatCompletion(
					smokeEngineHandleId,
					[{ role: "user", content: "hi" }],
					{
						maxTokens: 2,
						temperature: 0.6,
						topK: 40,
						repetitionPenalty: 1.05,
					},
				);
				// Drain the async iterator so all decode steps run.
				// biome-ignore lint/suspicious/noEmptyBlockStatements: drain
				for await (const _ of stream) {
				}
			}
			const warmupMs = performance.now() - warmupStart;
			log("pass", `[6/8] Shader-cache warmup complete in ${warmupMs.toFixed(0)}ms`);
		} catch (e) {
			// Warmup failure is non-fatal — log and continue. The downstream
			// timed steps will still run; they'll just include cold-shader
			// compilation cost in their tok/s.
			log(
				"running",
				`[6/8] Shader-cache warmup failed (continuing): ${e.message}`,
			);
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

		if (isEmbedderModel || isBucketDEmbedPerf) {
			log(
				"pass",
				"[7/8] Generation: skipped (embedder model — bench mode runs embedding tasks instead)",
			);
			setProgress(90);
		} else {
		log("running", "[7/8] Generating text...");
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
				engine: smokeEngine,
				handleId: smokeEngineHandleId,
				inference,
				tokenizer,
				log,
				profileMode,
			});
			interactiveRunCompletion = runCompletion;

			const smokeSamplingConfig = {
				...getSmokeSamplingConfig(
					parsed,
					detectChatTemplate,
					chatTmpl,
					smokeChatOptions,
				),
				...samplingOverrides,
				// Optional spec-decode lever — `runCompletion` spreads the
				// sampling config into `engine.chatCompletion`'s
				// `CompletionConfig`, which already accepts these fields.
				// Use the synthetic handle id, not the user-facing name —
				// the engine's drafter gate lives in `inferenceEngines`,
				// keyed by handle id.
				...(drafterHandleId ? { drafter: drafterHandleId } : {}),
				...(drafterDraftLength !== null
					? { draftLength: drafterDraftLength }
					: {}),
			};
			const smokeMaxTokens =
				maxTokensOverride ?? (thinkingEnabled ? 1024 : 64);
			const smokeResult = await runCompletion({
				label: smokePrompt.mode,
				messages: [{ role: "user", content: userMessage }],
				samplingConfig: smokeSamplingConfig,
				maxTokens: smokeMaxTokens,
				chatOptions: smokeChatOptions,
			});

			log(
				"pass",
				`[7/8] Generated ${smokeResult.genTokens} tokens in ${(smokeResult.totalTime / 1000).toFixed(1)}s (prefill: ${smokeResult.prefillMs.toFixed(0)}ms, decode: ${smokeResult.genTime.toFixed(0)}ms, ${(smokeResult.genTokens / (smokeResult.genTime / 1000)).toFixed(1)} tok/s, finish=${smokeResult.finishReason})`,
			);
			log("pass", `User: ${userMessage}`);
			const assistantText = thinkingEnabled
				? smokeResult.rawOutputText ||
					smokeResult.displayOutputText ||
					smokeResult.outputText
				: smokeResult.displayOutputText || smokeResult.outputText;
			log("pass", `Assistant: ${assistantText}`);
			// Stash a SmokeRunRecord so the post-[8/8] dashboard ingest hook
			// can POST `run_complete`. We snapshot here (instead of re-deriving
			// later) because `userMessage` and `smokeResult` are scoped to
			// this try-block.
			const decodeMs = Math.round(smokeResult.genTime ?? 0);
			const totalMs = Math.round(smokeResult.totalTime ?? 0);
			const prefillMs = Math.round(smokeResult.prefillMs ?? 0);
			const tps =
				decodeMs > 0
					? Math.round(((smokeResult.genTokens ?? 0) / (decodeMs / 1000)) * 10) / 10
					: 0;
			window.__webllmSmokeRunId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
			window.__webllmSmokeRecord = {
				schemaVersion: 1,
				timestamp: new Date().toISOString(),
				profile: profileName ?? undefined,
				model: modelId,
				page: "smoke",
				thinking: thinkingEnabled ? "on" : "off",
				prompt: userMessage,
				params: {
					contextLength: Number.isFinite(requestedContextLength)
						? requestedContextLength
						: undefined,
					...samplingOverrides,
				},
				oneShot: {
					assistantText,
					finishReason: smokeResult.finishReason,
					genTokens: smokeResult.genTokens ?? 0,
					prefillMs,
					decodeMs,
					totalMs,
					tokensPerSecond: tps,
				},
			};
			setProgress(100);
		} catch (e) {
			log("fail", `[7/8] Generation failed: ${e.message}\n${e.stack || ""}`);
			return;
		}
		}

		// The [8/8] embed smoke check loads a *second* engine on a known-good
		// arctic-embed-s F16 GGUF. For embedder bench runs we already have a
		// pooling pipeline loaded (steps 1-6 above) — re-downloading the
		// reference GGUF would only add noise. Skip when the page itself is
		// already embedder-driven (encoder or causal-embedder), or a bucket D
		// embeddingCapable chat model in embed-perf bench mode.
		if (isEmbedderModel || isBucketDEmbedPerf) {
			log(
				"pass",
				"[8/8] Reference encoder check: skipped (page is already running an embedder model)",
			);
			await runEmbedPerfHook(
				smokeEngine,
				smokeEngineHandleId,
				embedPerfMode,
				embedReps,
				embedFixture,
				log,
			);
		} else {
		log(
			"running",
			"[8/8] Loading Arctic-Embed-s and computing embedding...",
		);
		try {
			const embedUrl = `./models/snowflake-arctic-embed-s-f16.GGUF${assetSuffix}`;
			const embedResp = await fetch(embedUrl);
			if (!embedResp.ok) {
				throw new Error(`HTTP ${embedResp.status} fetching ${embedUrl}`);
			}
			const embedBuf = await embedResp.arrayBuffer();
			if (!navigator.gpu) {
				throw new Error("navigator.gpu unavailable");
			}
			const embedAdapter = await navigator.gpu.requestAdapter();
			if (!embedAdapter) {
				throw new Error("requestAdapter() returned null");
			}
			const embedDevice = await embedAdapter.requestDevice();
			const { engine: engine2, handle: embedHandle } =
				await WebLLM.loadModelFromBuffer(
					embedBuf,
					"arctic-s",
					{ device: embedDevice, memoryBudget: 500_000_000 },
					`./${wasmVariant}${assetSuffix}`,
				);
			const va = await engine2.embed(embedHandle.id, "happy");
			const vb = await engine2.embed(embedHandle.id, "joyful");
			window.embedA = va;
			window.embedB = vb;

			if (va.length !== 384) {
				throw new Error(`expected 384-dim vector, got ${va.length}`);
			}
			if (!Number.isFinite(va[0])) {
				throw new Error(`embedding[0] is not finite (=${va[0]})`);
			}
			let na = 0;
			for (let i = 0; i < va.length; i++) na += va[i] * va[i];
			const norm = Math.sqrt(na);
			if (!(norm >= 0.99 && norm <= 1.01)) {
				throw new Error(`‖v‖=${norm.toFixed(4)} not in [0.99, 1.01]`);
			}
			let dot = 0;
			let nb = 0;
			for (let i = 0; i < va.length; i++) {
				dot += va[i] * vb[i];
				nb += vb[i] * vb[i];
			}
			const cosine =
				na === 0 || nb === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
			if (!(cosine >= 0.75)) {
				throw new Error(
					`cosine=${cosine.toFixed(4)} below 0.75 threshold`,
				);
			}
			// Guard against the "all inputs produce identical vectors" pathology —
			// would pass the ≥0.75 check trivially but indicates broken tokenization
			// or a position/token-type embedding bug.
			if (cosine > 0.999) {
				throw new Error(
					`cosine=${cosine.toFixed(6)} suspiciously close to 1.0 for distinct synonyms — likely identical-vectors bug`,
				);
			}
			log(
				"pass",
				`[8/8] embed('happy') · embed('joyful') cosine=${cosine.toFixed(2)} (>=0.75 expected, ‖v‖=${norm.toFixed(2)})`,
			);
			await runEmbedPerfHook(
				engine2,
				embedHandle.id,
				embedPerfMode,
				embedReps,
				embedFixture,
				log,
			);
		} catch (e) {
			log("fail", `[8/8] embed failed: ${e.message}\n${e.stack || ""}`);
			throw e;
		}
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

		// Best-effort: post the [7/8] one-shot run as run_complete to the
		// dashboard. Default-on (see `?ingest=` resolution above); a
		// connection refused on a stopped dashboard is logged once and
		// swallowed. Bench mode posts its own eval_complete; we still post
		// the one-shot here so a single page load yields one row in `runs`
		// regardless of whether bench mode was requested.
		if (benchIngestUrl && window.__webllmSmokeRecord) {
			try {
				const payload = {
					...window.__webllmSmokeRecord,
					runId: window.__webllmSmokeRunId,
				};
				if (window.__webllmSystemId) payload.systemId = window.__webllmSystemId;
				const res = await fetch(`${benchIngestUrl}/ingest?kind=run_complete`, {
					method: "POST",
					headers: { "content-type": "application/json" },
					body: JSON.stringify(payload),
				});
				if (res.ok) {
					log(
						"pass",
						`[ingest] run_complete → ${benchIngestUrl} (runId ${window.__webllmSmokeRunId})`,
					);
				} else {
					log(
						"fail",
						`[ingest] run_complete failed: HTTP ${res.status} ${res.statusText}`,
					);
				}
			} catch (err) {
				log(
					"fail",
					`[ingest] dashboard not reachable at ${benchIngestUrl} — skipping (use ?ingest=off to silence): ${err?.message ?? String(err)}`,
				);
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
						"bench mode reached after [7/8] but wasm/inference/parsed are unset — did the smoke steps fail silently?",
					);
				}
				const { runBenchMode } = await import(
					`./real-model-bench.js${assetSuffix}`
				);
				if (!smokeEngine || !smokeEngineHandleId) {
					throw new Error(
						"bench mode reached without a constructed engine — did [7/8] fail silently?",
					);
				}
				await runBenchMode({
					engine: smokeEngine,
					handleId: smokeEngineHandleId,
					runTasks,
					score,
					collectBrowserSystemProfile,
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
				detectChatTemplate,
				interactiveRunCompletion,
				getSmokeChatOptions: (nextParsedModel, chatTemplate) =>
					getSmokeChatOptions(nextParsedModel, detectChatTemplate, chatTemplate, {
						enableThinking: thinkingEnabled,
					}),
				getSmokeSamplingConfig,
				samplingOverrides,
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
