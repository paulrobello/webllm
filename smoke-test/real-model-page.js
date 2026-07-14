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
	// `?backend=jsep` (P2-v2 prototype) swaps the bundle + WASM artifacts
	// to the JSEP-style variant. Default is the production legacy backend.
	// The flag also threads through to `WebLLM.init({ backend })` below so
	// `engine.maybeInstallJsep()` wires the seven `Module.jsep*` callbacks
	// onto the freshly-initialized `GgmlWasm`.
	const _earlyParams = new URLSearchParams(window.location.search);
	const useJsepBackend = _earlyParams.get("backend") === "jsep";
	const bundleName = useJsepBackend
		? "webllm-bundle-jsep.js"
		: "webllm-bundle.js";
	// ARC-003: deep inference internals (ModelInference, GgmlWasm, GgufParser,
	// detectChatTemplate, encodeChatPrompt) were moved off the public barrel
	// (`src/index.ts`) into `src/internal.ts`. The smoke harness pulls them
	// from the dedicated `webllm-internal.js` bundle; the public surface
	// (WebLLM, error classes, sampling profiles, types) stays in
	// `webllm-bundle.js`. Both bundles are rebuilt by the `smoke-test`
	// Makefile target.
	const {
		CausalLMEmbedder,
		EncoderInference,
		ModelLoader,
		Tokenizer,
		WebLLM,
		collectBrowserSystemProfile,
		runTasks,
		score,
	} = await import(`./${bundleName}${assetSuffix}`);
	const {
		GgufParser,
		GgmlWasm,
		ModelInference,
		detectChatTemplate,
		encodeChatPrompt,
	} = await import(`./webllm-internal.js${assetSuffix}`);
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
	const wasmVariant = useJsepBackend
		? "webllm-wasm-jsep.js"
		: params.get("wasm") === "mem64"
			? "webllm-wasm-mem64.js"
			: "webllm-wasm.js";
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
	const benchSessionId = params.get("session");
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
	// Dual-mode worker deployment: `?worker=1` runs the engine inside a
	// DedicatedWorker (same-bundle re-entry via `WebLLMProxy`). The
	// public TS surface is identical between main-thread and worker
	// modes; only the host context differs. The dashboard records
	// `mode=worker|main` for cross-mode A/B perf comparison.
	const useWorker = params.get("worker") === "1";
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
	// Frame-probe gate: ?frameProbe=1 spins a small WebGL2 cube via rAF in
	// parallel with [7/8] generation and reports per-phase frame deltas
	// (baseline / prefill / decode / post). Answers the agent + Three.js
	// coexistence question without running the full Three.js stack. See
	// `smoke-test/frame-probe.js`.
	const frameProbeEnabled = params.get("frameProbe") === "1";
	const frameProbeBaselineMsRaw = Number(params.get("frameProbeBaselineMs"));
	const frameProbeBaselineMs =
		Number.isFinite(frameProbeBaselineMsRaw) && frameProbeBaselineMsRaw > 0
			? Math.floor(frameProbeBaselineMsRaw)
			: 3000;
	// `?scene=<url>` switches the probe from the trivial cube to a real
	// Three.js GLTF scene loaded at that URL. Use this to measure the
	// "agent + Three.js coexistence" question with real GPU contention,
	// not just main-thread scheduling.
	const frameProbeSceneUrl = params.get("scene") || null;
	// `?frameProbeCalls=N` runs N back-to-back chatCompletion calls in the
	// same page load and reports per-call hitch distribution. Answers the
	// question "is the decode hitch deterministic per-call or jitter?"
	// — see `summarizeFrameProbeMulti` in `./frame-probe.js`.
	const frameProbeCallsRaw = Number(params.get("frameProbeCalls"));
	const frameProbeCalls =
		Number.isFinite(frameProbeCallsRaw) && frameProbeCallsRaw > 1
			? Math.floor(frameProbeCallsRaw)
			: 1;
	// `?frameProbeWarmup=1` runs a throwaway 4-token chatCompletion AFTER
	// the baseline rAF window but BEFORE timing the multi-call probe.
	// Tests probe-9c hypothesis: that the deterministic ~50ms call-0
	// decode-shape hitch is caused by per-shape pipeline JIT and can be
	// absorbed by a one-shot warmup at session boot.
	const frameProbeWarmup = params.get("frameProbeWarmup") === "1";
	// `?probe=9b` runs probe-9b (batched-prompt vs N sequential calls)
	// after the regular [7/8] smoke flow. Posts the result to
	// `window.__probe9bResult`; the Bun runner scrapes both scenarios.
	const probe9bEnabled = params.get("probe") === "9b";
	// `?probe=prefix-cache` runs the prefix-cache validation probe
	// (4 NPCs × 2 ticks each, pattern A: no handles / pattern B: with
	// per-NPC ConversationHandles). Posts the timing matrix to
	// `window.__probePrefixCacheResult` for the Bun driver at
	// `eval/probes/probe-prefix-cache-validation-2026-05-01.ts`.
	const probePrefixCacheEnabled = params.get("probe") === "prefix-cache";
	// `?probe=prefix-cache-at-scale` runs the same matrix but with a
	// ~3-4× longer NPC system prefix (~1500-2000 tokens) to validate the
	// at-scale case. Posts to `window.__probePrefixCacheAtScaleResult`.
	const probePrefixCacheAtScaleEnabled =
		params.get("probe") === "prefix-cache-at-scale";
	// `?probe=prefix-cache-interleaved` defeats the per-model session-
	// tracker prefix cache by round-robining NPCs (NPC_1 t1 → NPC_2 t1 →
	// NPC_3 t1 → NPC_4 t1 → NPC_1 t2 → NPC_2 t2 → ...) with per-NPC
	// distinct ~1200-token personas, so each NPC's prompt diverges very
	// early from any sibling's. Pattern A's session tracker can only
	// preserve the small shared framework intro; tick-2 calls must
	// re-prefill the entire per-NPC persona. Pattern B reloads the
	// per-conv KV snapshot and prefills only the tail. Posts to
	// `window.__probePrefixCacheInterleavedResult`.
	const probePrefixCacheInterleavedEnabled =
		params.get("probe") === "prefix-cache-interleaved";
	// `?probe=prefix-cache-fork` measures the cross-conv prefix sharing
	// win delivered by `WebLLM.forkConversation`. Pattern X (baseline):
	// each NPC creates a fresh conv via createConversation; first
	// chatCompletion prefills the full ~1325-token shared system
	// prefix. Pattern Y (forked): one base conv is primed with the
	// shared system prefix, then forked per NPC; each fork's first
	// chatCompletion finds the shared prefix in the inherited
	// snapshot via the longest-shared-token-prefix walk and prefills
	// only the divergent tail. Posts to
	// `window.__probePrefixCacheForkResult`.
	const probePrefixCacheForkEnabled =
		params.get("probe") === "prefix-cache-fork";
	const frameProbeModule = frameProbeEnabled
		? await import(`./frame-probe.js${assetSuffix}`)
		: null;
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
		const _t0 = performance.now();
		const profileMode = new URLSearchParams(window.location.search).has(
			"perfTrace",
		);

		// WebGPU init moved to step 1: subsequent steps stream the GGUF
		// directly into the WASM heap, so the heap must exist first.
		// JS-heap `new Uint8Array(N)` caps at ~2 GiB on Chrome and fails
		// for >2 GiB GGUFs (4B Q4 ≈ 2.27 GiB). A `Uint8Array` *view* over
		// the WASM-backed ArrayBuffer can exceed 2 GiB.
		//
		// Worker mode (`?worker=1`): the main thread has no WASM heap —
		// the engine's worker side calls `wasm.init()` internally inside
		// `loadModelFromBuffer`. We still log a [1/8] PASS line so the
		// page shell shows step continuity; the actual init happens at
		// [6/8] when the buffer is transferred.
		log("running", "[1/8] Initializing WebGPU backend...");
		let wasm = null;
		if (useWorker) {
			log("pass", "[1/8] WebGPU backend init delegated to worker");
		} else {
			try {
				wasm = new GgmlWasm();
				wasmInstance = wasm;
				await wasm.init({ wasmUrl: `./${wasmVariant}${assetSuffix}` });
				// JSEP callback wiring (P2-v2 prototype). Must land BEFORE the
				// first model load — `ggml_backend_jsep_alloc_buffer` reads
				// `Module.jsepAlloc` at weight-upload time. The smoke page
				// owns the externally-constructed `GgmlWasm` and uses
				// `engine.adoptPreloadedModel`, which bypasses the engine's
				// internal `maybeInstallJsep` (only `loadModelFromBuffer` /
				// `loadModelFromUrl` invoke it). Install the callbacks here
				// so the JSEP backend's `Module.jsep*` hooks are wired
				// before [4/8] weight upload.
				if (useJsepBackend) {
					if (!navigator.gpu) {
						throw new Error(
							"backend=jsep requires WebGPU; navigator.gpu is unavailable",
						);
					}
					const jsepAdapter = await navigator.gpu.requestAdapter();
					if (!jsepAdapter) {
						throw new Error("backend=jsep could not acquire a GPUAdapter");
					}
					const jsepDevice = await jsepAdapter.requestDevice();
					await wasm.installJsepCallbacks(jsepDevice);
					log("pass", "[1/8] JSEP callbacks installed");
				}
				log("pass", "[1/8] WebGPU backend initialized");
			} catch (e) {
				log("fail", `[1/8] WebGPU init failed: ${e.message}\n${e.stack || ""}`);
				return;
			}
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
				log(
					"running",
					`[diagnoseAlloc] WebGPU device limits: ${JSON.stringify(dump)}`,
				);
				device.destroy();
			} catch (e) {
				log("fail", `[diagnoseAlloc] failed to query device: ${e.message}`);
			}
		}

		log("running", "[2/8] Fetching model...");
		progressEl.style.display = "block";
		// Main-thread mode: stream into the WASM heap so we never hold a
		// full JS-heap copy of the GGUF (the wasm32 build's `Uint8Array`
		// max length and the JS heap's >2 GiB awkwardness both bite for
		// 7B+ models).
		//
		// Worker mode: the worker fetches the GGUF directly into its own
		// WASM heap via `engine.loadModelFromUrl`. The engine returns the
		// parsed metadata (hyperparams, tokenizerConfig, kvCacheConfig)
		// alongside the model handle, so the smoke page no longer parses
		// GGUF main-side at all. We do the WebLLM.init + loadModelFromUrl
		// at [2/8] in worker mode so subsequent steps can populate `parsed`
		// from `result.metadata` and reuse all downstream metadata-driven
		// logic (subtitle, KV-cache log, modelSupportsThinking gate, ctx
		// clamp, tokenizer construction, chat-template branching) without
		// branching on mode.
		let modelPtr = 0;
		let modelByteLength = 0;
		const fetchStart = performance.now();
		try {
			if (useWorker) {
				if (!navigator.gpu) {
					throw new Error("navigator.gpu unavailable; smoke test needs WebGPU");
				}
				const ctxLenForLoad =
					requestedContextLength > 0 ? requestedContextLength : 0;
				log(
					"running",
					`[2/8] Worker fetching model from ${modelUrl} (streaming into worker WASM heap)...`,
				);
				smokeEngine = await WebLLM.init({
					memoryBudget: 2_000_000_000,
					maxConversations: 8,
					worker: true,
					...(useJsepBackend ? { backend: "jsep" } : {}),
				});
				const result = await smokeEngine.loadModelFromUrl(
					modelUrl,
					modelId,
					`./${wasmVariant}${assetSuffix}`,
					{
						priority: 0,
						// `loadModelFromUrl` clamps to maxContextLength internally
						// when contextLength > 0; passing 0 leaves the engine to
						// pick the GGUF default. We don't yet know the GGUF max
						// from the main side (that's the whole point of this
						// refactor — no main-side parse), so pass through the
						// requested value as-is.
						...(ctxLenForLoad > 0 ? { contextLength: ctxLenForLoad } : {}),
						embeddingCapable,
						embeddingPooling,
					},
				);
				parsedModel = result.metadata;
				smokeEngineHandleId = result.handle.id;
				modelByteLength = 0; // unknown without main-side fetch; not used in worker mode
				window.engine = smokeEngine;
				window.handleId = smokeEngineHandleId;
				log(
					"pass",
					`[2/8] Worker model load complete in ${((performance.now() - fetchStart) / 1000).toFixed(1)}s`,
				);
				setProgress(40);
			} else {
				const resp = await fetch(modelUrl);
				if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
				const total = Number(resp.headers.get("content-length") || 0);
				if (total <= 0) {
					throw new Error(
						"missing content-length on model response; streaming into WASM heap requires it",
					);
				}
				modelByteLength = total;
				const reader = resp.body.getReader();
				let received = 0;
				modelPtr = wasm.malloc(total);
				if (!modelPtr) throw new Error(`wasm malloc(${total}) returned null`);
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
			}
		} catch (e) {
			log("fail", `[2/8] Fetch failed: ${e.message}\n${e.stack || ""}`);
			if (modelPtr) wasm.free(modelPtr);
			return;
		}

		// Main-thread `modelDataAt`: re-derives a fresh sub-view of HEAPU8
		// on every call so callers can't accidentally hold a stale reference
		// across a memory-grow event. Worker mode never needs `modelDataAt`
		// — the worker side parsed the GGUF and uploaded weights itself.
		const modelDataAt = useWorker
			? null
			: (off, len) => new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);

		log("running", "[3/8] Parsing GGUF...");
		let ggufCtx;
		let parsed;
		try {
			if (useWorker) {
				// Worker mode: the engine's `loadModelFromUrl` at [2/8]
				// returned the parsed metadata directly. No main-side parse.
				parsed = parsedModel;
				ggufCtx = null;
			} else {
				const fullView = modelDataAt(0, modelByteLength);
				ggufCtx = GgufParser.parse(fullView);
				parsed = ModelLoader.parseModel(fullView);
				parsedModel = parsed;
			}
			const hp = parsed.hyperparams;
			const subtitleContextLength =
				requestedContextLength > 0
					? Math.min(
							parsed.kvCacheConfig.maxContextLength,
							requestedContextLength,
						)
					: parsed.kvCacheConfig.maxContextLength;
			subtitleEl.textContent = `${pageCopy.subtitle} · Model: ${modelId} · arch=${hp.architecture} · ctx=${subtitleContextLength}`;
			log(
				"pass",
				`[3/8] GGUF parsed: arch=${hp.architecture} emb=${hp.embeddingLength} heads=${hp.headCount}/${hp.headCountKv} layers=${hp.layerCount} vocab=${hp.vocabularySize} ctx=${hp.contextLength}`,
			);
		} catch (e) {
			log("fail", `[3/8] Parse failed: ${e.message}\n${e.stack || ""}`);
			if (!useWorker && modelPtr) wasm.free(modelPtr);
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
			if (!useWorker && modelPtr) wasm.free(modelPtr);
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

		// Worker mode: weights upload + KV-cache init already happened
		// inside `engine.loadModelFromUrl` at [2/8]. We log [4/8] and
		// [5/8] PASS lines here to preserve step continuity for the page
		// shell and any downstream log-scraping.
		if (useWorker) {
			log("pass", "[4/8] Weights upload delegated to worker");
			setProgress(80);
			log(
				"pass",
				`[5/8] KV cache: delegated to worker (ctx=${
					requestedContextLength > 0
						? Math.min(
								parsed.kvCacheConfig.maxContextLength,
								requestedContextLength,
							)
						: parsed.kvCacheConfig.maxContextLength
				})`,
			);
			setProgress(85);
		} else {
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
							? Math.min(
									parsed.kvCacheConfig.maxContextLength,
									requestedContextLength,
								)
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

		// Build the WebLLM engine.
		//
		// Main-thread mode: adopt the already-built `inference` pipeline
		// via `adoptPreloadedModel` so the engine reuses the same WASM /
		// inference / parsed instances we constructed above.
		//
		// Worker mode: the engine + model were already constructed at
		// [2/8] (engine returns parsed metadata as part of the load
		// result, so the smoke page never parses GGUF main-side). Skip
		// this block; `smokeEngine` / `smokeEngineHandleId` / `window.engine`
		// are already set.
		if (!useWorker) {
			try {
				if (!navigator.gpu) {
					throw new Error("navigator.gpu unavailable; smoke test needs WebGPU");
				}
				smokeEngine = await WebLLM.init({
					memoryBudget: 2_000_000_000,
					// Default is 4; the prefix-cache-fork probe runs 1 base
					// + 4 forks = 5 concurrent conversations.
					maxConversations: 8,
					worker: false,
					...(useJsepBackend ? { backend: "jsep" } : {}),
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
		}

		// Optional drafter for speculative decoding. Mirrors the target's
		// load path: fetch the GGUF, bring up a *separate* `GgmlWasm` heap
		// (heaps are per-model in this codebase — sharing one between
		// target + drafter would let one model's malloc invalidate the
		// other's HEAPU8 view), construct a `ModelInference`, and adopt
		// it under a second handle id on the same engine.
		//
		// Worker mode mirrors the target's [6/8] path: fetch into a JS-
		// heap ArrayBuffer, transfer to the worker via `loadModelFromBuffer`,
		// and let the worker re-parse + build inference. The drafter ends
		// up registered on the same in-worker engine instance as the
		// target, so `CompletionConfig.drafter` resolution is unchanged.
		if (drafterId) {
			log("running", `[drafter] Loading ${drafterId}...`);
			const drafterUrl = `./models/${drafterId}.gguf${assetSuffix}`;
			if (useWorker) {
				try {
					const drafterResp = await fetch(drafterUrl);
					if (!drafterResp.ok) {
						throw new Error(
							`HTTP ${drafterResp.status} fetching ${drafterUrl}`,
						);
					}
					// Guardrail: `arrayBuffer()` materializes the full body in
					// a JS-heap ArrayBuffer, which V8 caps at ~3.5 GB per
					// allocation (the same bug Path A fixed for the main
					// model by switching to `engine.loadModelFromUrl`).
					// Drafters in the canonical sweep are <= 1 GB, well under
					// the V8 cap. Retained `loadModelFromBuffer` here for the
					// small ergonomic win of skipping the prefix-parse two-step
					// the main model now needs. If a future >3.5 GB drafter is
					// introduced, switch this branch to `loadModelFromUrl`
					// (the engine support is already there). The check below
					// fails fast with a clear message instead of letting the
					// allocation throw an opaque RangeError.
					const V8_ARRAY_BUFFER_CAP = 3.5 * 1024 * 1024 * 1024;
					const drafterTotal = Number(
						drafterResp.headers.get("content-length") || 0,
					);
					if (drafterTotal > V8_ARRAY_BUFFER_CAP) {
						throw new Error(
							`drafter ${drafterId} is ${(drafterTotal / 1e9).toFixed(1)} GB; ` +
								`worker-mode drafter path uses arrayBuffer() which would trip ` +
								`V8's ArrayBuffer cap. Switch the drafter to loadModelFromUrl ` +
								`for drafters > 3.5 GB (TODO).`,
						);
					}
					const drafterBuf = await drafterResp.arrayBuffer();
					// Peek at metadata main-side just for the ctx-length log
					// line; the worker does the load-bearing parse itself.
					const drafterParsed = ModelLoader.parseModel(
						new Uint8Array(drafterBuf),
					);
					const drafterCtxLen =
						requestedContextLength > 0
							? Math.min(
									drafterParsed.kvCacheConfig.maxContextLength,
									requestedContextLength,
								)
							: drafterParsed.kvCacheConfig.maxContextLength;
					const drafterResult = await smokeEngine.loadModelFromBuffer(
						drafterBuf,
						drafterId,
						`./${wasmVariant}${assetSuffix}`,
						{ priority: 0, contextLength: drafterCtxLen },
					);
					drafterHandleId = drafterResult.handle.id;
					log(
						"pass",
						`[drafter] ${drafterId} loaded (ctx=${drafterCtxLen}, draftLength=${drafterDraftLength ?? "default"})`,
					);
				} catch (e) {
					log("fail", `[drafter] Load failed: ${e.message}\n${e.stack || ""}`);
					return;
				}
			} else {
				let drafterPtr = 0;
				let drafterByteLength = 0;
				let drafterWasm = null;
				try {
					drafterWasm = new GgmlWasm();
					await drafterWasm.init({
						wasmUrl: `./${wasmVariant}${assetSuffix}`,
					});
					const drafterResp = await fetch(drafterUrl);
					if (!drafterResp.ok) {
						throw new Error(
							`HTTP ${drafterResp.status} fetching ${drafterUrl}`,
						);
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
						throw new Error(
							`drafter wasm malloc(${drafterTotal}) returned null`,
						);
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
						new Uint8Array(drafterWasm.heapU8.buffer, drafterPtr + off, len);
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
					log("fail", `[drafter] Load failed: ${e.message}\n${e.stack || ""}`);
					return;
				}
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
				for await (const _ of stream) {
				}
			}
			const warmupMs = performance.now() - warmupStart;
			log(
				"pass",
				`[6/8] Shader-cache warmup complete in ${warmupMs.toFixed(0)}ms`,
			);
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
					`  blk.0.attn_norm first8: [${Array.from(attn)
						.map((v) => v.toFixed(4))
						.join(",")}]`,
				);
				log(
					"running",
					`  blk.0.ffn_norm  first8: [${Array.from(ffn)
						.map((v) => v.toFixed(4))
						.join(",")}]`,
				);
				log(
					"running",
					`  output_norm     first8: [${Array.from(out)
						.map((v) => v.toFixed(4))
						.join(",")}]`,
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
					`  kv.v[layer=0][pos=0..63][dim=0][head=0] nonzero=${nzV}/64 sumAbs=${sumAbsV.toFixed(4)} first4=[${Array.from(
						v0.slice(0, 4),
					)
						.map((v) => v.toFixed(4))
						.join(",")}]`,
				);
			} catch (e) {
				log(
					"fail",
					`[7-debug] KV probe failed: ${e.message}\n${e.stack || ""}`,
				);
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
					throw new Error(
						"could not find single-token probes for this tokenizer",
					);
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
				const topA = Array.from(lA).reduce(
					(m, v, i, a) => (a[m] >= v ? m : i),
					0,
				);
				const topB = Array.from(lB).reduce(
					(m, v, i, a) => (a[m] >= v ? m : i),
					0,
				);
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
				log(
					"fail",
					`[7a] KV diagnostic failed: ${e.message}\n${e.stack || ""}`,
				);
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
			// Frame-probe: start the rAF logger BEFORE chat completion so the
			// baseline window (configurable via ?frameProbeBaselineMs=) sits
			// outside any inference work. Phase segmentation is wall-clock based
			// — see `summarizeFrameProbe` in `./frame-probe.js`.
			let frameProbeCtl = null;
			let frameProbeChatStart = 0;
			if (frameProbeEnabled && frameProbeModule) {
				if (frameProbeSceneUrl) {
					log("running", `[frameProbe] loading scene: ${frameProbeSceneUrl}`);
				}
				frameProbeCtl = await frameProbeModule.startFrameProbe({
					sceneUrl: frameProbeSceneUrl,
				});
				if (frameProbeCtl.unsupported) {
					log("warn", "[frameProbe] WebGL2 unavailable — probe disabled");
					frameProbeCtl = null;
				} else {
					const info = frameProbeCtl.sceneInfo;
					if (info?.kind === "gltf") {
						log(
							"pass",
							`[frameProbe] scene loaded: ${info.triangles.toLocaleString()} tri in ${info.loadMs}ms`,
						);
					}
					log(
						"running",
						`[frameProbe] baseline rAF (${frameProbeBaselineMs}ms idle)…`,
					);
					await new Promise((r) => setTimeout(r, frameProbeBaselineMs));
					frameProbeChatStart = performance.now();
				}
			}
			try {
				if (debugMode) {
					for (const prompt of [
						"The",
						"The quick brown",
						"Hello, how are you",
					]) {
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
					await compareBatchVsSequentialPrefill(
						smokePrompt.mode,
						smokePrompt.tokens,
					);
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
				// Probe 9c: optional warmup throwaway. Runs a 4-token
				// chatCompletion BEFORE the timed multi-call probe so the
				// per-shape pipeline JIT cost is amortized off the timed
				// path. After warmup, re-anchor `frameProbeChatStart` so
				// frame-probe segmentation excludes the warmup window.
				if (frameProbeWarmup && frameProbeCtl) {
					log("running", "[frameProbe] warmup throwaway (4 tokens)…");
					try {
						await runCompletion({
							label: smokePrompt.mode,
							messages: [{ role: "user", content: userMessage }],
							samplingConfig: smokeSamplingConfig,
							maxTokens: 4,
							chatOptions: smokeChatOptions,
						});
					} catch (e) {
						log("warn", `[frameProbe] warmup failed: ${e.message}`);
					}
					// Settle so warmup's GPU queue drains before the timed
					// probe; matches the inter-call gap in multi-call mode.
					await new Promise((r) => setTimeout(r, 500));
					frameProbeChatStart = performance.now();
				}
				const smokeResult = await runCompletion({
					label: smokePrompt.mode,
					messages: [{ role: "user", content: userMessage }],
					samplingConfig: smokeSamplingConfig,
					maxTokens: smokeMaxTokens,
					chatOptions: smokeChatOptions,
				});

				log(
					"pass",
					`[7/8] Generated ${smokeResult.genTokens} tokens in ${(smokeResult.totalTime / 1000).toFixed(1)}s (prefill: ${smokeResult.prefillMs.toFixed(0)}ms, decode: ${smokeResult.genTime.toFixed(0)}ms, ${(smokeResult.genTokens / (smokeResult.genTime / 1000)).toFixed(1)} tok/s, finish=${smokeResult.finishReason}, tokensIn=${smokePrompt.tokens.length})`,
				);
				log("pass", `User: ${userMessage}`);
				const assistantText = thinkingEnabled
					? smokeResult.rawOutputText ||
						smokeResult.displayOutputText ||
						smokeResult.outputText
					: smokeResult.displayOutputText || smokeResult.outputText;
				log("pass", `Assistant: ${assistantText}`);
				// Frame-probe wrap: capture tEnd, hold a 1s post window, then
				// segment + report. Surfaced both to the page log and to
				// `window.__frameProbeResult` for agentchrome scrape.
				//
				// Multi-call mode (`?frameProbeCalls=N`, N>1): after the smoke
				// call above, runs N-1 additional chatCompletion calls (same
				// prompt, fresh KV state per call) so the per-call decode-hitch
				// distribution can be inspected. The first call's result is
				// preserved as the smoke record.
				if (frameProbeCtl && frameProbeModule) {
					const probeCalls = [
						{
							tStart: frameProbeChatStart,
							prefillMs: smokeResult.prefillMs ?? 0,
							tEnd: performance.now(),
							genTokens: smokeResult.genTokens ?? 0,
						},
					];
					if (frameProbeCalls > 1) {
						log(
							"running",
							`[frameProbe] running ${frameProbeCalls - 1} additional call(s) for hitch-distribution analysis…`,
						);
						for (let i = 1; i < frameProbeCalls; i++) {
							// Inter-call settle so the prior post window doesn't
							// bleed into the next prefill. 500ms is long enough
							// for the GPU queue to drain on the test scene.
							await new Promise((r) => setTimeout(r, 500));
							const tStartI = performance.now();
							let resultI;
							try {
								resultI = await runCompletion({
									label: smokePrompt.mode,
									messages: [{ role: "user", content: userMessage }],
									samplingConfig: smokeSamplingConfig,
									maxTokens: smokeMaxTokens,
									chatOptions: smokeChatOptions,
								});
							} catch (e) {
								log(
									"warn",
									`[frameProbe] call ${i + 1}/${frameProbeCalls} failed: ${e.message}`,
								);
								break;
							}
							const tEndI = performance.now();
							probeCalls.push({
								tStart: tStartI,
								prefillMs: resultI.prefillMs ?? 0,
								tEnd: tEndI,
								genTokens: resultI.genTokens ?? 0,
							});
							log(
								"running",
								`[frameProbe] call ${i + 1}/${frameProbeCalls}: ${resultI.genTokens}t in ${resultI.totalTime.toFixed(0)}ms (prefill ${resultI.prefillMs.toFixed(0)}ms, ${(resultI.genTokens / (resultI.genTime / 1000)).toFixed(1)} tok/s)`,
							);
						}
					}
					await new Promise((r) => setTimeout(r, 1000));
					frameProbeCtl.stop({ removeOverlay: false });
					const isMulti = probeCalls.length > 1;
					let summary;
					if (isMulti) {
						summary = frameProbeModule.summarizeFrameProbeMulti({
							samples: frameProbeCtl.samples,
							tBaselineStart: frameProbeChatStart - frameProbeBaselineMs,
							calls: probeCalls,
						});
						for (const line of frameProbeModule.formatFrameProbeMultiReport(
							summary,
						)) {
							log("running", line);
						}
					} else {
						summary = frameProbeModule.summarizeFrameProbe({
							samples: frameProbeCtl.samples,
							tStart: frameProbeChatStart,
							prefillMs: smokeResult.prefillMs,
							tEnd: probeCalls[0].tEnd,
						});
						for (const line of frameProbeModule.formatFrameProbeReport(
							summary,
						)) {
							log("running", line);
						}
					}
					const decodeMsFp = smokeResult.genTime ?? 0;
					const baseFrameStats = isMulti
						? {
								baseline: summary.baseline,
								post: summary.post,
								perCall: summary.calls,
							}
						: {
								baseline: summary.baseline,
								prefill: summary.prefill,
								decode: summary.decode,
								post: summary.post,
							};
					window.__frameProbeResult = {
						model: modelId,
						mode: isMulti ? "multi" : "single",
						callCount: probeCalls.length,
						decodeTokens: smokeResult.genTokens ?? 0,
						prefillMs: smokeResult.prefillMs ?? 0,
						decodeMs: decodeMsFp,
						tokensPerSec:
							decodeMsFp > 0
								? (smokeResult.genTokens ?? 0) / (decodeMsFp / 1000)
								: 0,
						frameStats: baseFrameStats,
						verdict: isMulti ? null : summary.verdict,
						sceneInfo: frameProbeCtl.sceneInfo,
						calls: isMulti ? probeCalls : undefined,
						sample: (
							smokeResult.displayOutputText ||
							smokeResult.outputText ||
							""
						).slice(0, 240),
					};
				}
				// Probe 9b: batched vs sequential N-NPC scaling. Runs AFTER
				// the [7/8] smoke flow has warmed the engine. Posts the
				// timing + raw output of both scenarios to
				// `window.__probe9bResult` for the Bun runner to scrape and
				// score. No public-API change — uses `engine.chatCompletion`
				// directly, same path the smoke flow uses.
				if (probe9bEnabled) {
					log("running", "[probe9b] running batched vs sequential scenarios…");
					const NPC_PREFIX_9B =
						"You are an NPC AI controller for a fantasy MMO. Available tools: move, speak, attack, use_item, trade. Each NPC has stats hp, mp, level, position. Pick exactly one tool name as the action.";
					const NPCS_9B = [
						{
							id: "goblin_1",
							obs: "Goblin sees Hero approaching at distance 8, hp 22/40. Hero is hostile.",
						},
						{
							id: "wolf_2",
							obs: "Wolf sees a wounded rabbit at distance 5, hp 30/30. Hungry.",
						},
						{
							id: "merchant_3",
							obs: "Merchant has new wares. Player Hero approaching with 200 gold, neutral stance.",
						},
						{
							id: "guard_4",
							obs: "Guard sees suspicious player Thief sneaking near treasure room.",
						},
					];

					async function runOnce(prompt, maxTokens) {
						const tStart = performance.now();
						const tokens = await runCompletion({
							label: "probe9b",
							messages: [{ role: "user", content: prompt }],
							samplingConfig: smokeSamplingConfig,
							maxTokens,
							chatOptions: smokeChatOptions,
						});
						const tEnd = performance.now();
						return {
							wallMs: tEnd - tStart,
							prefillMs: tokens.prefillMs,
							genTokens: tokens.genTokens,
							output: tokens.outputText ?? tokens.displayOutputText ?? "",
						};
					}

					// Inter-call settle so KV-cache reset + GPU queue drain
					// don't bleed into the next prefill.
					const settle = () => new Promise((r) => setTimeout(r, 500));

					const seqRuns = [];
					const seqTotalStart = performance.now();
					for (const npc of NPCS_9B) {
						const prompt = `${NPC_PREFIX_9B}\n\nNPC: ${npc.id}\nObservation: ${npc.obs}\n\nReply with one word — the tool name:`;
						seqRuns.push({ npcId: npc.id, ...(await runOnce(prompt, 8)) });
						await settle();
					}
					const seqTotalWallMs =
						performance.now() - seqTotalStart - (NPCS_9B.length - 1) * 500;

					await settle();

					const batchedObsList = NPCS_9B.map((n) => `- ${n.id}: ${n.obs}`).join(
						"\n",
					);
					const batchedPrompt = `${NPC_PREFIX_9B}\n\nDecide a tool action for each NPC below.\n${batchedObsList}\n\nReply with a JSON array of objects, one per NPC, e.g. [{"npc_id":"goblin_1","action":"attack"}, ...]:`;
					const batchedRun = await runOnce(batchedPrompt, 96);

					window.__probe9bResult = {
						model: modelId,
						sequential: {
							totalWallMs: seqTotalWallMs,
							perCall: seqRuns,
						},
						batched: {
							wallMs: batchedRun.wallMs,
							prefillMs: batchedRun.prefillMs,
							genTokens: batchedRun.genTokens,
							output: batchedRun.output,
						},
					};
					log(
						"pass",
						`[probe9b] sequential total=${seqTotalWallMs.toFixed(0)}ms (${NPCS_9B.length} calls), batched=${batchedRun.wallMs.toFixed(0)}ms`,
					);
				}

				// Probe prefix-cache: validates that conversation-handle-mode
				// `chatCompletion(conv, ...)` actually skips the shared prefix
				// on a 2nd tick. Two patterns × 4 NPCs × 2 ticks per NPC = 16
				// timed calls. Pattern A uses the modelId path (full re-prefill
				// every call); pattern B uses per-NPC ConversationHandles.
				// PASS = pattern B's tick-2 median lands in 75-150 ms band.
				if (probePrefixCacheEnabled || probePrefixCacheAtScaleEnabled) {
					const probeTag = probePrefixCacheAtScaleEnabled
						? "probe-prefix-cache-at-scale"
						: "probe-prefix-cache";
					log(
						"running",
						`[${probeTag}] running pattern A (no handles) then pattern B (with handles)…`,
					);

					const NPC_PREFIX_BASE =
						"You are an NPC AI controller for a fantasy MMO. Available tools: move, speak, attack, use_item, trade. Each NPC has stats hp, mp, level, position. Pick exactly one tool name as the action. Detailed tool reference. move(x, y): walk the NPC to grid coordinates (x, y); fails if path is blocked, slowed by terrain. speak(text): emit a short utterance audible to NPCs and players within 12 tiles; logs to chat. attack(target): initiate combat with target NPC or player id; honors faction rules and aggro tables. use_item(item): consume from inventory; potions restore hp/mp, scrolls cast spells, food triggers regen ticks. trade(player): open trade window with target player id; both parties must accept. Stat semantics: hp is current health out of max_hp, depletes from damage and regenerates outside combat; mp is mana for spells, regenerates faster than hp; level scales damage and resists; position is current grid cell as (x, y); inventory is a list of item ids. Decision rules: prefer survival over aggression below 30% hp, prefer engagement above 70% hp, fall back to flee if outnumbered three to one or more, never break neutrality with same-faction NPCs.";
					const NPC_PREFIX_AT_SCALE_TAIL =
						" Combat formulas: damage = (attacker.attack × roll(0.85, 1.15)) − defender.defense; critical = roll(0.05) doubles damage; magic resist applies after physical reduction; armor pen = max(0, attacker.armorPen − defender.armor × 0.5). Faction relations: orcs vs humans (-3); elves vs orcs (-2); humans vs elves (+1); dwarves vs orcs (-2); dwarves vs elves (-1); all factions neutral to merchants and guards; bounty hunters honor contracts above factions. Status effects: poison ticks 5 hp/turn for 3 turns; stun blocks attack and movement for 1 turn; haste doubles speed for 2 turns; bleed ticks 3 hp/turn for 4 turns; burn ticks 4 hp/turn for 2 turns and disables ice spells; freeze halts movement for 2 turns; charm flips faction temporarily; silence disables spell-cast for 3 turns; root anchors position for 2 turns. Loot tables: low-tier mobs drop 1-3 gp + 30% chance common item; mid-tier add 50% rare drop; bosses always drop legendary + 100-500 gp; chests scale with dungeon depth; quest items bypass random rolls and always drop. Aggro mechanics: damage taken adds threat = damage; healing adds threat = healing × 0.5; threat decays 10%/turn outside combat; taunt forces +200 threat; stealth halves all threat generation; pets generate threat scaled by 0.5. Inventory rules: max 50 slots; weight cap = 10 × strength; over-cap halves movement and disables sprint; equipped gear does not count toward slot count but counts toward weight; consumables stack to 99; quest items have dedicated tab. Trade rules: equal-faction trade tax 5%; cross-faction tax 15%; criminal-status disables trade entirely; black-market merchants accept stolen goods at 30% discount; bartering skill reduces tax by up to 5%. Speech rules: NPCs respond to mentioned proper nouns within 2-tile range; ambient chatter triggers every 10 turns of inactivity; greeting lines vary by faction stance; aggressive NPCs taunt before melee engagement; merchants advertise wares every 30 turns. World map: continents Aerthos (north), Vorden (east), Mirrowyn (south), Karaduun (west), Skyreach (sky-bound floating islands); 7 capitals (Aerthos: Ivormere, Whitestone; Vorden: Hexspire; Mirrowyn: Sunhold, Tideglass; Karaduun: Dustforge; Skyreach: Aerie); 24 minor towns; tunnels link via teleport gates aligned to leyline nodes. Day/night: 24 in-game hours = 1 real hour; nocturnal mobs +20% attack at night, day-active +20% perception by day; dawn and dusk are spawn windows for rare mobs; lunar phases gate werewolf transformations and certain quest triggers. Weather rules: rain slows fire spells by 25% and boosts water spells; snow halves movement on outdoor tiles and applies cold-vulnerability +10%; sandstorms blind ranged attacks beyond 4 tiles; fog masks stealth detection by +30%; thunderstorms power-spike lightning casters by 15% but interrupt long channels on 5% chance per turn. Crafting rules: tier-1 recipes need 1 common + 1 raw material; tier-2 add a refined component; legendary recipes require a quest-rare core; failed crafts return 50% of inputs; specialization perks reduce material cost by up to 20%. Quest log rules: at most 25 active quests; abandoned quests have a 24-hour cooldown; daily resets at server midnight; weekly raids cap at 7 entries; bounty boards refresh every 4 hours. Movement and pathing: NPCs use A* over a 4-connected grid; line-of-sight raycasts ignore translucent props; pets hold a leash radius of 12 tiles unless aggro overrides; mounted NPCs cost 2× weight but move at 2× speed. Animation timing: attack windup 250 ms, recovery 350 ms; cast windup scales with spell tier 200-1500 ms; interrupt windows occur during the first 60% of windup; dodge i-frames last 240 ms; block reduces damage by 50% for 200 ms after press. Audio cues: footsteps audible within 6 tiles indoors and 9 tiles outdoors; spellcasts emit element-specific cues; combat music kicks in within 12 tiles of any aggro source; merchant jingles play within 4 tiles of stalls.";
					const NPC_PREFIX_PFX = probePrefixCacheAtScaleEnabled
						? NPC_PREFIX_BASE + NPC_PREFIX_AT_SCALE_TAIL
						: NPC_PREFIX_BASE;
					const NPCS_PFX = [
						{
							id: "goblin_1",
							obs1: "Goblin sees Hero approaching at distance 8, hp 22/40. Hero is hostile.",
							obs2: "Hero is now at distance 4 and drew a sword. Goblin hp 22/40.",
						},
						{
							id: "wolf_2",
							obs1: "Wolf sees a wounded rabbit at distance 5, hp 30/30. Hungry.",
							obs2: "Rabbit fled into bushes. A second hunter wolf approached.",
						},
						{
							id: "merchant_3",
							obs1: "Merchant has new wares. Player Hero approaching with 200 gold, neutral stance.",
							obs2: "Hero asked about the rare potion. Hero gold balance now 80.",
						},
						{
							id: "guard_4",
							obs1: "Guard sees suspicious player Thief sneaking near treasure room.",
							obs2: "Thief drew a dagger and lunged toward the chest.",
						},
					];

					// Tick 1: [system, user(obs1)]
					// Tick 2: [system, user(obs1), assistant(canned reply), user(obs2)]
					// The shared prefix in tick-2 against tick-1's snapshot covers
					// [system, user(obs1)] in token space; tick 2 only has to
					// prefill the divergent [assistant, user(obs2)] tail.
					const buildTick1 = (npc) => [
						{ role: "system", content: NPC_PREFIX_PFX },
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs1}\n\nReply with one word — the tool name:`,
						},
					];
					const buildTick2 = (npc, prevAssistant) => [
						{ role: "system", content: NPC_PREFIX_PFX },
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs1}\n\nReply with one word — the tool name:`,
						},
						{ role: "assistant", content: prevAssistant },
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs2}\n\nReply with one word — the tool name:`,
						},
					];

					// Inter-call settle so KV-cache reset + GPU queue drain don't
					// bleed into the next prefill (matches probe9b cadence).
					const settle = () => new Promise((r) => setTimeout(r, 500));

					async function runChat(arg, messages) {
						const tStart = performance.now();
						const stream = smokeEngine.chatCompletion(arg, messages, {
							maxTokens: 32,
						});
						let prefillMs = 0;
						let firstTokenAt = 0;
						let firstChunkSeen = false;
						const generatedIds = [];
						let stats = null;
						for await (const chunk of stream) {
							if (chunk.done) {
								stats = chunk.stats ?? null;
								continue;
							}
							if (!firstChunkSeen) {
								firstTokenAt = performance.now();
								prefillMs = firstTokenAt - tStart;
								firstChunkSeen = true;
							}
							if (chunk.tokenId !== undefined) {
								generatedIds.push(chunk.tokenId);
							}
						}
						const wallMs = performance.now() - tStart;
						// Prefer the engine's official prefill timing (covers the
						// pure prefill phase, no first-decode overhead).
						if (
							stats &&
							typeof stats.timeToFirstTokenMs === "number" &&
							stats.timeToFirstTokenMs > 0
						) {
							prefillMs = stats.timeToFirstTokenMs;
						}
						const outputText = tokenizer.decode(generatedIds);
						return { wallMs, prefillMs, output: outputText };
					}

					// Pattern A: handle-id path with no ConversationHandle. Each
					// call sees full re-prefill via the engine's per-model session
					// tracker. NB: the smoke page registers the model under a
					// synthetic `handleId` (not the user-facing config `modelId`),
					// so we drive the engine via `smokeEngineHandleId`.
					const engineHandleId = smokeEngineHandleId;
					const patternA = [];
					smokeEngine.resetModelSession(engineHandleId);
					await settle();
					for (const npc of NPCS_PFX) {
						const r1 = await runChat(engineHandleId, buildTick1(npc));
						patternA.push({
							npcId: npc.id,
							tick: 1,
							prefillMs: r1.prefillMs,
							wallMs: r1.wallMs,
							output: r1.output,
						});
						await settle();
						const r2 = await runChat(
							engineHandleId,
							buildTick2(npc, r1.output),
						);
						patternA.push({
							npcId: npc.id,
							tick: 2,
							prefillMs: r2.prefillMs,
							wallMs: r2.wallMs,
							output: r2.output,
						});
						await settle();
					}
					smokeEngine.resetModelSession(engineHandleId);
					await settle();

					// Pattern B: per-NPC ConversationHandle. Tick 1 populates the
					// snapshot; tick 2 should hit the prefix cache.
					const patternB = [];
					const convs = await Promise.all(
						NPCS_PFX.map(() => smokeEngine.createConversation(engineHandleId)),
					);
					try {
						for (let i = 0; i < NPCS_PFX.length; i++) {
							const npc = NPCS_PFX[i];
							const r1 = await runChat(convs[i], buildTick1(npc));
							patternB.push({
								npcId: npc.id,
								tick: 1,
								prefillMs: r1.prefillMs,
								wallMs: r1.wallMs,
								output: r1.output,
							});
							await settle();
							const r2 = await runChat(convs[i], buildTick2(npc, r1.output));
							patternB.push({
								npcId: npc.id,
								tick: 2,
								prefillMs: r2.prefillMs,
								wallMs: r2.wallMs,
								output: r2.output,
							});
							await settle();
						}
					} finally {
						for (const c of convs) await smokeEngine.disposeConversation(c);
					}

					const median = (xs) => {
						const s = xs
							.filter((v) => Number.isFinite(v))
							.sort((a, b) => a - b);
						if (s.length === 0) return 0;
						return s[Math.floor(s.length / 2)];
					};
					const tick2A = patternA
						.filter((r) => r.tick === 2)
						.map((r) => r.prefillMs);
					const tick2B = patternB
						.filter((r) => r.tick === 2)
						.map((r) => r.prefillMs);

					const probeResultObj = {
						model: modelId,
						patternA,
						patternB,
					};
					if (probePrefixCacheAtScaleEnabled) {
						window.__probePrefixCacheAtScaleResult = probeResultObj;
					} else {
						window.__probePrefixCacheResult = probeResultObj;
					}

					log(
						"pass",
						`[${probeTag}] tick-2 medians: A=${median(tick2A).toFixed(0)}ms, B=${median(tick2B).toFixed(0)}ms`,
					);
				}

				// Probe prefix-cache-interleaved: defeats Pattern A's session-
				// tracker cache by round-robining NPC ticks. Per-NPC personas
				// are distinct from token-3 onward (NPC name appears in the
				// first sentence), so longest-shared-prefix between any two
				// sibling NPCs is just the small shared framework intro. After
				// NPC_4 tick-1, the session tracker holds NPC_4's KV; NPC_1
				// tick-2 must re-prefill the entire NPC_1 persona (~1100
				// tokens). Pattern B reloads NPC_1's per-conv snapshot in
				// ~1.4 s (post-batch) and prefills only the divergent tail.
				if (probePrefixCacheInterleavedEnabled) {
					const probeTag = "probe-prefix-cache-interleaved";
					log(
						"running",
						`[${probeTag}] running interleaved pattern A then pattern B…`,
					);

					// Tiny shared framework intro — small enough that all of
					// Pattern A's session-tracker savings come from this one
					// block, regardless of which NPC was last computed.
					const FRAMEWORK_INTRO =
						"You are an NPC AI controller. Pick exactly one tool name as the action. Available tools: move, speak, attack, use_item, trade.";

					// Build a per-NPC persona by repeating a paragraph that
					// embeds the NPC's id very early. The first occurrence of
					// the id forces token-level divergence from any sibling
					// NPC's persona; everything after that lives in the
					// divergent tail and must be re-prefilled when the session
					// tracker holds a different NPC's KV.
					function buildPersona(npcId, role, locale, era) {
						const para = `Persona for ${npcId} (role ${role}, locale ${locale}, era ${era}). ${npcId} was raised in the ${locale} territories during the ${era} cycle, where ${role} duties shaped a particular discipline of restraint, observation, and reaction. ${npcId} maintains the customary tactical doctrine of its kin: prefer survival below thirty percent vitality, prefer engagement above seventy percent vitality, fall back to flee if outnumbered three to one or more, never break neutrality with same-faction agents. ${npcId} keeps a personal log of grievances, of bargains, of debts, of small mercies — these influence which tool ${npcId} reaches for first when pressed. ${npcId} prefers tools that reflect the ethic of ${role}: a guard chooses confrontation, a merchant chooses speech, a hunter chooses pursuit, a trader chooses bargain. ${npcId}'s ${era}-cycle training instructs that the first response is rarely the correct one, and that observing the field for an extra heartbeat distinguishes survivors from casualties.`;
						// Repeat 6× so persona lands at ~1100 tokens.
						return Array.from({ length: 6 }, () => para).join(" ");
					}

					const NPCS_INTERLEAVED = [
						{
							id: "goblin_1",
							persona: buildPersona(
								"goblin_1",
								"raider",
								"Bonewood",
								"Skullsplit",
							),
							obs1: "Goblin sees Hero approaching at distance 8, hp 22/40. Hero is hostile.",
							obs2: "Hero is now at distance 4 and drew a sword. Goblin hp 22/40.",
						},
						{
							id: "wolf_2",
							persona: buildPersona(
								"wolf_2",
								"hunter",
								"Frostmoor",
								"Longwinter",
							),
							obs1: "Wolf sees a wounded rabbit at distance 5, hp 30/30. Hungry.",
							obs2: "Rabbit fled into bushes. A second hunter wolf approached.",
						},
						{
							id: "merchant_3",
							persona: buildPersona(
								"merchant_3",
								"trader",
								"Sunhold",
								"Goldspring",
							),
							obs1: "Merchant has new wares. Player Hero approaching with 200 gold, neutral stance.",
							obs2: "Hero asked about the rare potion. Hero gold balance now 80.",
						},
						{
							id: "guard_4",
							persona: buildPersona(
								"guard_4",
								"sentinel",
								"Ivormere",
								"Stoneward",
							),
							obs1: "Guard sees suspicious player Thief sneaking near treasure room.",
							obs2: "Thief drew a dagger and lunged toward the chest.",
						},
					];

					const buildTick1 = (npc) => [
						{
							role: "system",
							content: `${FRAMEWORK_INTRO}\n\n${npc.persona}`,
						},
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs1}\n\nReply with one word — the tool name:`,
						},
					];
					const buildTick2 = (npc, prevAssistant) => [
						{
							role: "system",
							content: `${FRAMEWORK_INTRO}\n\n${npc.persona}`,
						},
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs1}\n\nReply with one word — the tool name:`,
						},
						{ role: "assistant", content: prevAssistant },
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs2}\n\nReply with one word — the tool name:`,
						},
					];

					const settle = () => new Promise((r) => setTimeout(r, 500));

					async function runChat(arg, messages) {
						const tStart = performance.now();
						const stream = smokeEngine.chatCompletion(arg, messages, {
							maxTokens: 32,
						});
						let prefillMs = 0;
						let firstTokenAt = 0;
						let firstChunkSeen = false;
						const generatedIds = [];
						let stats = null;
						for await (const chunk of stream) {
							if (chunk.done) {
								stats = chunk.stats ?? null;
								continue;
							}
							if (!firstChunkSeen) {
								firstTokenAt = performance.now();
								prefillMs = firstTokenAt - tStart;
								firstChunkSeen = true;
							}
							if (chunk.tokenId !== undefined) {
								generatedIds.push(chunk.tokenId);
							}
						}
						const wallMs = performance.now() - tStart;
						if (
							stats &&
							typeof stats.timeToFirstTokenMs === "number" &&
							stats.timeToFirstTokenMs > 0
						) {
							prefillMs = stats.timeToFirstTokenMs;
						}
						const outputText = tokenizer.decode(generatedIds);
						return { wallMs, prefillMs, output: outputText };
					}

					const engineHandleId = smokeEngineHandleId;

					// Pattern A interleaved: round-robin all tick-1s, then all
					// tick-2s. Session tracker is forced to invalidate the
					// per-NPC persona on every cross-NPC call.
					const patternA = [];
					const patternARecallCues = new Array(NPCS_INTERLEAVED.length);
					smokeEngine.resetModelSession(engineHandleId);
					await settle();
					for (let i = 0; i < NPCS_INTERLEAVED.length; i++) {
						const npc = NPCS_INTERLEAVED[i];
						const r1 = await runChat(engineHandleId, buildTick1(npc));
						patternA.push({
							npcId: npc.id,
							tick: 1,
							prefillMs: r1.prefillMs,
							wallMs: r1.wallMs,
							output: r1.output,
						});
						patternARecallCues[i] = r1.output;
						await settle();
					}
					for (let i = 0; i < NPCS_INTERLEAVED.length; i++) {
						const npc = NPCS_INTERLEAVED[i];
						const r2 = await runChat(
							engineHandleId,
							buildTick2(npc, patternARecallCues[i]),
						);
						patternA.push({
							npcId: npc.id,
							tick: 2,
							prefillMs: r2.prefillMs,
							wallMs: r2.wallMs,
							output: r2.output,
						});
						await settle();
					}
					smokeEngine.resetModelSession(engineHandleId);
					await settle();

					// Pattern B interleaved: same matrix, per-NPC handles. Each
					// NPC's tick-2 reloads its own KV snapshot rather than
					// re-prefilling the persona.
					const patternB = [];
					const patternBRecallCues = new Array(NPCS_INTERLEAVED.length);
					const convs = await Promise.all(
						NPCS_INTERLEAVED.map(() =>
							smokeEngine.createConversation(engineHandleId),
						),
					);
					try {
						for (let i = 0; i < NPCS_INTERLEAVED.length; i++) {
							const npc = NPCS_INTERLEAVED[i];
							const r1 = await runChat(convs[i], buildTick1(npc));
							patternB.push({
								npcId: npc.id,
								tick: 1,
								prefillMs: r1.prefillMs,
								wallMs: r1.wallMs,
								output: r1.output,
							});
							patternBRecallCues[i] = r1.output;
							await settle();
						}
						for (let i = 0; i < NPCS_INTERLEAVED.length; i++) {
							const npc = NPCS_INTERLEAVED[i];
							const r2 = await runChat(
								convs[i],
								buildTick2(npc, patternBRecallCues[i]),
							);
							patternB.push({
								npcId: npc.id,
								tick: 2,
								prefillMs: r2.prefillMs,
								wallMs: r2.wallMs,
								output: r2.output,
							});
							await settle();
						}
					} finally {
						for (const c of convs) await smokeEngine.disposeConversation(c);
					}

					const med = (xs) => {
						const s = xs
							.filter((v) => Number.isFinite(v))
							.sort((a, b) => a - b);
						if (s.length === 0) return 0;
						return s[Math.floor(s.length / 2)];
					};
					const tick2A = patternA
						.filter((r) => r.tick === 2)
						.map((r) => r.prefillMs);
					const tick2B = patternB
						.filter((r) => r.tick === 2)
						.map((r) => r.prefillMs);
					window.__probePrefixCacheInterleavedResult = {
						model: modelId,
						patternA,
						patternB,
					};
					log(
						"pass",
						`[${probeTag}] tick-2 medians: A=${med(tick2A).toFixed(0)}ms, B=${med(tick2B).toFixed(0)}ms`,
					);
				}

				// Probe prefix-cache-fork: measures forkConversation's
				// cross-conv prefix sharing win on the first-tick-per-NPC.
				// Pattern X (baseline): each NPC creates a fresh conv;
				// first chatCompletion prefills the entire ~1325-token
				// shared system prefix. Pattern Y (forked): a base conv
				// is primed with the shared prefix once, then forked per
				// NPC; each fork's first chatCompletion finds the shared
				// prefix in the inherited snapshot and prefills only the
				// divergent tail.
				if (probePrefixCacheForkEnabled) {
					const probeTag = "probe-prefix-cache-fork";
					log(
						"running",
						`[${probeTag}] running pattern X (baseline) then pattern Y (forked)…`,
					);

					const SHARED_SYSTEM =
						"You are an NPC AI controller for a fantasy MMO. Available tools: move, speak, attack, use_item, trade. Each NPC has stats hp, mp, level, position. Pick exactly one tool name as the action. Detailed tool reference. move(x, y): walk the NPC to grid coordinates (x, y); fails if path is blocked, slowed by terrain. speak(text): emit a short utterance audible to NPCs and players within 12 tiles; logs to chat. attack(target): initiate combat with target NPC or player id; honors faction rules and aggro tables. use_item(item): consume from inventory; potions restore hp/mp, scrolls cast spells, food triggers regen ticks. trade(player): open trade window with target player id; both parties must accept. Stat semantics: hp is current health out of max_hp, depletes from damage and regenerates outside combat; mp is mana for spells, regenerates faster than hp; level scales damage and resists; position is current grid cell as (x, y); inventory is a list of item ids. Decision rules: prefer survival over aggression below 30% hp, prefer engagement above 70% hp, fall back to flee if outnumbered three to one or more, never break neutrality with same-faction NPCs. Combat formulas: damage = (attacker.attack × roll(0.85, 1.15)) − defender.defense; critical = roll(0.05) doubles damage; magic resist applies after physical reduction; armor pen = max(0, attacker.armorPen − defender.armor × 0.5). Faction relations: orcs vs humans (-3); elves vs orcs (-2); humans vs elves (+1); dwarves vs orcs (-2); dwarves vs elves (-1); all factions neutral to merchants and guards; bounty hunters honor contracts above factions. Status effects: poison ticks 5 hp/turn for 3 turns; stun blocks attack and movement for 1 turn; haste doubles speed for 2 turns; bleed ticks 3 hp/turn for 4 turns; burn ticks 4 hp/turn for 2 turns and disables ice spells; freeze halts movement for 2 turns; charm flips faction temporarily; silence disables spell-cast for 3 turns; root anchors position for 2 turns. Loot tables: low-tier mobs drop 1-3 gp + 30% chance common item; mid-tier add 50% rare drop; bosses always drop legendary + 100-500 gp; chests scale with dungeon depth; quest items bypass random rolls and always drop. Aggro mechanics: damage taken adds threat = damage; healing adds threat = healing × 0.5; threat decays 10%/turn outside combat; taunt forces +200 threat; stealth halves all threat generation; pets generate threat scaled by 0.5.";
					const NPCS_FORK = [
						{
							id: "goblin_1",
							obs: "Goblin sees Hero approaching at distance 8, hp 22/40. Hero is hostile.",
						},
						{
							id: "wolf_2",
							obs: "Wolf sees a wounded rabbit at distance 5, hp 30/30. Hungry.",
						},
						{
							id: "merchant_3",
							obs: "Merchant has new wares. Player Hero approaching with 200 gold, neutral stance.",
						},
						{
							id: "guard_4",
							obs: "Guard sees suspicious player Thief sneaking near treasure room.",
						},
					];

					const buildFirstTick = (npc) => [
						{ role: "system", content: SHARED_SYSTEM },
						{
							role: "user",
							content: `NPC: ${npc.id}\nObservation: ${npc.obs}\n\nReply with one word — the tool name:`,
						},
					];

					const settle = () => new Promise((r) => setTimeout(r, 500));

					async function runChatFork(arg, messages) {
						const tStart = performance.now();
						const stream = smokeEngine.chatCompletion(arg, messages, {
							maxTokens: 32,
						});
						let prefillMs = 0;
						let firstTokenAt = 0;
						let firstChunkSeen = false;
						const generatedIds = [];
						let stats = null;
						for await (const chunk of stream) {
							if (chunk.done) {
								stats = chunk.stats ?? null;
								continue;
							}
							if (!firstChunkSeen) {
								firstTokenAt = performance.now();
								prefillMs = firstTokenAt - tStart;
								firstChunkSeen = true;
							}
							if (chunk.tokenId !== undefined) {
								generatedIds.push(chunk.tokenId);
							}
						}
						const wallMs = performance.now() - tStart;
						if (
							stats &&
							typeof stats.timeToFirstTokenMs === "number" &&
							stats.timeToFirstTokenMs > 0
						) {
							prefillMs = stats.timeToFirstTokenMs;
						}
						const outputText = tokenizer.decode(generatedIds);
						return { wallMs, prefillMs, output: outputText };
					}

					const engineHandleId = smokeEngineHandleId;

					// Pattern X (baseline): each NPC gets a fresh conv. First
					// chatCompletion has no snapshot, so it prefills the entire
					// shared prefix from scratch.
					const patternX = [];
					const baselineConvs = await Promise.all(
						NPCS_FORK.map(() => smokeEngine.createConversation(engineHandleId)),
					);
					try {
						for (let i = 0; i < NPCS_FORK.length; i++) {
							const npc = NPCS_FORK[i];
							const r = await runChatFork(
								baselineConvs[i],
								buildFirstTick(npc),
							);
							patternX.push({
								npcId: npc.id,
								prefillMs: r.prefillMs,
								wallMs: r.wallMs,
								output: r.output,
							});
							await settle();
						}
					} finally {
						for (const c of baselineConvs)
							await smokeEngine.disposeConversation(c);
					}

					// Reset the per-model session tracker so its KV doesn't bleed
					// into pattern Y. (Without this, the engine's session tracker
					// would already hold the shared prefix from pattern X's last
					// call, masking the fork win — pattern Y's "base prime" call
					// would be a session-tracker hit, not a fresh prefill.)
					smokeEngine.resetModelSession(engineHandleId);
					await settle();

					// Pattern Y (forked): prime a base conv with the shared
					// prefix, fork per NPC, drive the same first-tick
					// chatCompletion. The fork's first call should find the
					// shared prefix in the inherited snapshot and prefill only
					// the divergent NPC tail.
					const baseConv = await smokeEngine.createConversation(engineHandleId);
					let baseTickMs = 0;
					try {
						// Prime base with [system, user="ping"]. The "ping" user
						// message is generic — every fork's first call will
						// diverge at the first NPC-specific token, sharing only
						// the [system] prefix. This is the realistic spawn pattern.
						const baseStart = performance.now();
						await runChatFork(baseConv, [
							{ role: "system", content: SHARED_SYSTEM },
							{ role: "user", content: "ping" },
						]);
						baseTickMs = performance.now() - baseStart;

						const patternY = [];
						const forkConvs = await Promise.all(
							NPCS_FORK.map(() => smokeEngine.forkConversation(baseConv)),
						);
						try {
							for (let i = 0; i < NPCS_FORK.length; i++) {
								const npc = NPCS_FORK[i];
								const r = await runChatFork(forkConvs[i], buildFirstTick(npc));
								patternY.push({
									npcId: npc.id,
									prefillMs: r.prefillMs,
									wallMs: r.wallMs,
									output: r.output,
								});
								await settle();
							}
						} finally {
							for (const c of forkConvs)
								await smokeEngine.disposeConversation(c);
						}

						const med = (xs) => {
							const s = xs
								.filter((v) => Number.isFinite(v))
								.sort((a, b) => a - b);
							if (s.length === 0) return 0;
							return s[Math.floor(s.length / 2)];
						};
						window.__probePrefixCacheForkResult = {
							model: modelId,
							baseTickMs,
							patternX,
							patternY,
						};
						log(
							"pass",
							`[${probeTag}] median wall: X=${med(patternX.map((r) => r.wallMs)).toFixed(0)}ms, Y=${med(patternY.map((r) => r.wallMs)).toFixed(0)}ms`,
						);
					} finally {
						await smokeEngine.disposeConversation(baseConv);
					}
				}

				// Stash a SmokeRunRecord so the post-[8/8] dashboard ingest hook
				// can POST `run_complete`. We snapshot here (instead of re-deriving
				// later) because `userMessage` and `smokeResult` are scoped to
				// this try-block.
				const decodeMs = Math.round(smokeResult.genTime ?? 0);
				const totalMs = Math.round(smokeResult.totalTime ?? 0);
				const prefillMs = Math.round(smokeResult.prefillMs ?? 0);
				const tps =
					decodeMs > 0
						? Math.round(
								((smokeResult.genTokens ?? 0) / (decodeMs / 1000)) * 10,
							) / 10
						: 0;
				window.__webllmSmokeRunId = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
				window.__webllmSmokeRecord = {
					schemaVersion: 1,
					timestamp: new Date().toISOString(),
					profile: profileName ?? undefined,
					model: modelId,
					page: "smoke",
					thinking: thinkingEnabled ? "on" : "off",
					mode: useWorker ? "worker" : "main",
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
				if (frameProbeCtl) frameProbeCtl.stop({ removeOverlay: true });
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
			log("running", "[8/8] Loading Arctic-Embed-s and computing embedding...");
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
					throw new Error(`cosine=${cosine.toFixed(4)} below 0.75 threshold`);
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
				// Worker mode: `wasmInstance` and `inference` live worker-side
				// and are null on the main thread — only `parsedModel` (parsed
				// from the GGUF before transfer at [6/8]) is required here.
				if (!parsedModel) {
					throw new Error(
						"bench mode reached after [7/8] but parsed metadata is unset — did the smoke steps fail silently?",
					);
				}
				if (!useWorker && (!wasmInstance || !inference)) {
					throw new Error(
						"bench mode (main-thread) reached but wasm/inference are unset — did the smoke steps fail silently?",
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
					sessionId: benchSessionId,
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
					getSmokeChatOptions(
						nextParsedModel,
						detectChatTemplate,
						chatTemplate,
						{
							enableThinking: thinkingEnabled,
						},
					),
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
