// Parity-capture harness — drives ModelInference.forwardWithLayerTaps on a
// loaded GGUF and POSTs the result as canonical JSON to the Bun capture
// server (eval/tools/parity-capture/capture-server.ts).
//
// URL params:
//   ?model=<id>         — GGUF basename under ./models/ (no extension).
//   ?prompt=<text>      — prompt text to tokenize and forward.
//   ?inputIds=N,N,N     — bypass WebLLM tokenizer entirely; use these IDs as
//                         the prefill input. Takes precedence over ?prompt+
//                         ?addBos. Use this to feed the SAME token IDs that
//                         HF saw, so the parity comparison isolates the
//                         forward-pass numerics from tokenizer differences.
//   ?addBos=1           — prepend BOS to tokenIds if not already present
//                         (mirrors HF capture's --add-bos flag). Ignored
//                         when ?inputIds= is set.
//   ?capture=<url>      — capture-server URL (default http://localhost:8035/capture).
//                         "off" disables POST (use for browser-only smoke).
//   ?topK=<n>           — top-K logits (default 16).
//   ?wasm=mem64         — load the mem64 WASM target (for >3.5 GiB models).
//
// The page logs each step to #log and emits a [PARITY-DONE] marker on
// success so callers can scrape the log to confirm capture completion.

export async function runParityCapture() {
	const assetSuffix = window.location.search || "";
	const params = new URLSearchParams(window.location.search);

	const modelId = params.get("model");
	if (!modelId) {
		document.body.textContent = "Missing required param: ?model=<id>";
		throw new Error("missing ?model");
	}
	const prompt = params.get("prompt") || "The capital of France is";
	const addBos = params.get("addBos") === "1";
	const inputIdsParam = params.get("inputIds");
	const inputIdsOverride = inputIdsParam
		? inputIdsParam
				.split(",")
				.map((s) => Number(s.trim()))
				.filter((n) => Number.isFinite(n) && n >= 0)
		: null;
	const captureParam = params.get("capture");
	const captureUrl =
		captureParam && captureParam.toLowerCase() === "off"
			? null
			: captureParam || "http://localhost:8035/capture";
	const topK = Math.max(1, Number(params.get("topK") || 16));

	const wasmVariant =
		params.get("wasm") === "mem64" ? "webllm-wasm-mem64.js" : "webllm-wasm.js";

	const bundleName = "webllm-bundle.js";

	// Minimal page shell — we don't need the full chat UI.
	document.body.innerHTML = `
		<div class="container">
			<h1>WebLLM Parity Capture</h1>
			<p>Model: <code>${modelId}</code> · Prompt: <code>${escapeHtml(prompt)}</code></p>
			<div id="log"></div>
		</div>
	`;
	const logEl = document.getElementById("log");
	function log(cls, msg) {
		const el = document.createElement("div");
		el.className = `step ${cls}`;
		el.textContent = msg;
		logEl.appendChild(el);
		logEl.scrollTop = logEl.scrollHeight;
	}

	const { GgmlWasm, GgufParser, ModelInference, ModelLoader, Tokenizer } =
		await import(`./${bundleName}${assetSuffix}`);

	log("running", "[1/6] Initializing WebGPU backend...");
	const wasm = new GgmlWasm();
	try {
		await wasm.init({ wasmUrl: `./${wasmVariant}${assetSuffix}` });
		log("pass", "[1/6] WebGPU backend init complete");
	} catch (e) {
		log("fail", `[1/6] WASM init failed: ${e.message}`);
		throw e;
	}

	const modelUrl = `./models/${modelId}.gguf`;
	let modelPtr = 0;
	let modelByteLength = 0;
	log("running", `[2/6] Streaming GGUF from ${modelUrl}...`);
	try {
		const resp = await fetch(modelUrl);
		if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
		const total = Number(resp.headers.get("content-length") || 0);
		if (total <= 0) {
			throw new Error("missing content-length on model response");
		}
		modelByteLength = total;
		const reader = resp.body.getReader();
		let received = 0;
		modelPtr = wasm.malloc(total);
		if (!modelPtr) throw new Error(`wasm malloc(${total}) returned null`);
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			wasm.heapU8.set(value, modelPtr + received);
			received += value.length;
		}
		if (received !== total) {
			throw new Error(`short read: expected ${total} bytes, got ${received}`);
		}
		log(
			"pass",
			`[2/6] GGUF streamed: ${(received / 1e6).toFixed(1)} MB`,
		);
	} catch (e) {
		log("fail", `[2/6] Fetch failed: ${e.message}`);
		if (modelPtr) wasm.free(modelPtr);
		throw e;
	}

	const modelDataAt = (off, len) =>
		new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);

	log("running", "[3/6] Parsing GGUF...");
	let parsed;
	let ggufCtx;
	try {
		const fullView = modelDataAt(0, modelByteLength);
		ggufCtx = GgufParser.parse(fullView);
		parsed = ModelLoader.parseModel(fullView);
		log(
			"pass",
			`[3/6] arch=${parsed.hyperparams.architecture} emb=${parsed.hyperparams.embeddingLength} layers=${parsed.hyperparams.layerCount} vocab=${parsed.hyperparams.vocabularySize}`,
		);
	} catch (e) {
		log("fail", `[3/6] Parse failed: ${e.message}`);
		wasm.free(modelPtr);
		throw e;
	}

	log("running", "[4/6] Loading weights to GPU (FA off for parity)...");
	let inference;
	try {
		inference = new ModelInference(wasm, parsed.hyperparams, {
			flashAttn: false,
		});
		inference.loadWeights(ggufCtx, modelDataAt);
		log("pass", "[4/6] Weights uploaded");
	} catch (e) {
		log("fail", `[4/6] Weights load failed: ${e.message}\n${e.stack || ""}`);
		wasm.free(modelPtr);
		throw e;
	} finally {
		wasm.free(modelPtr);
		modelPtr = 0;
	}

	log("running", "[5/6] Tokenizing + tapped forward...");
	let capture;
	try {
		let ids;
		if (inputIdsOverride) {
			ids = inputIdsOverride;
			log(
				"running",
				`[5/6] inputIds (override; tokenizer bypassed) len=${ids.length}: [${ids.join(",")}]`,
			);
		} else {
			const tokenizer = new Tokenizer(parsed.tokenizerConfig);
			ids = tokenizer.encode(prompt);
			const bosId = tokenizer.bosTokenId ?? null;
			if (addBos && bosId !== null) {
				if (ids.length === 0 || ids[0] !== bosId) {
					ids = [bosId, ...ids];
				}
			}
			log("running", `[5/6] tokenIds (len=${ids.length}): [${ids.join(",")}]`);
		}
		const tokenIdsArr = new Int32Array(ids);
		const t0 = performance.now();
		const tapped = await inference.forwardWithLayerTaps(tokenIdsArr, { topK });
		const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
		log(
			"pass",
			`[5/6] Tap forward complete in ${elapsed}s: ${tapped.perLayerResidual.length} layers, embDim=${tapped.finalNormHidden.length}, topK=${tapped.logitsTop16.ids.length}`,
		);

		capture = {
			model: modelId,
			captured_with: "webllm",
			captured_at: new Date().toISOString(),
			torch_dtype: null,
			add_bos: addBos,
			prompt,
			input_token_ids: Array.from(tokenIdsArr),
			n_layer: tapped.perLayerResidual.length,
			n_embd: tapped.finalNormHidden.length,
			embedding_output_last_token: Array.from(tapped.embeddingOutput),
			per_layer_residual_last_token: tapped.perLayerResidual.map((arr) =>
				Array.from(arr),
			),
			final_norm_hidden_last_token: Array.from(tapped.finalNormHidden),
			logits_top16: {
				ids: Array.from(tapped.logitsTop16.ids),
				values: Array.from(tapped.logitsTop16.values),
			},
		};
		window.__parityCapture = capture;
	} catch (e) {
		log("fail", `[5/6] Tap forward failed: ${e.message}\n${e.stack || ""}`);
		throw e;
	}

	if (captureUrl) {
		log("running", `[6/6] POSTing capture to ${captureUrl}...`);
		try {
			const resp = await fetch(captureUrl, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ payload: capture }),
			});
			if (!resp.ok) {
				const body = await resp.text();
				throw new Error(`HTTP ${resp.status}: ${body}`);
			}
			const j = await resp.json();
			log("pass", `[6/6] POST ok → ${j.path} (${j.bytes} bytes)`);
		} catch (e) {
			log("fail", `[6/6] POST failed: ${e.message}`);
			throw e;
		}
	} else {
		log(
			"pass",
			`[6/6] capture=off — JSON available at window.__parityCapture (size=${JSON.stringify(capture).length} chars)`,
		);
	}

	log("pass", "[PARITY-DONE]");
}

function escapeHtml(s) {
	return s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}
