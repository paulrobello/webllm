// FA prefill discriminator probe — invokes ModelInference.forwardVerify
// (which calls forwardAllPositions directly) with flashAttn=true to bisect
// the Gemma 4 + FA chat.html regression. forwardWithLayerTaps already
// succeeds at FA=true (parity-capture probe); forwardSingle/forwardDecode
// trap at FA=true via chat.html. This probe isolates the third forward
// method (forwardAllPositions, used by speculative-decoding K+1 verify).
//
// URL params:
//   ?model=<id>      GGUF basename under ./models/ (required).
//   ?prompt=<text>   prompt to tokenize (default "The capital of France is").
//   ?fa=off          disable FA (default on — the probe's whole point).
//   ?ctx=N           KV cache contextLength (default 4096 — matches chat.html).
//   ?path=verify|forward   which forward to invoke (default verify →
//                          forwardAllPositions; forward → forwardSingle via inf.forward()).
//   ?wasm=mem64      load mem64 WASM target.

export async function runFaPrefillProbe() {
	const assetSuffix = window.location.search || "";
	const params = new URLSearchParams(window.location.search);
	const modelId = params.get("model");
	if (!modelId) {
		document.body.textContent = "Missing required param: ?model=<id>";
		throw new Error("missing ?model");
	}
	const prompt = params.get("prompt") || "The capital of France is";
	const flashAttn = params.get("fa") !== "off";
	const ctxLenOverride = Math.max(64, Number(params.get("ctx") || 4096));
	const pathParam = params.get("path") === "forward" ? "forward" : "verify";
	const wasmVariant =
		params.get("wasm") === "mem64" ? "webllm-wasm-mem64.js" : "webllm-wasm.js";
	const bundleName = "webllm-bundle.js";

	document.body.innerHTML = `
		<div class="container">
			<h1>WebLLM FA Prefill Probe</h1>
			<p>Model: <code>${modelId}</code> · FA: <code>${flashAttn}</code> · Prompt: <code>${escapeHtml(prompt)}</code></p>
			<div id="log"></div>
		</div>
	`;
	const logEl = document.getElementById("log");
	const log = (cls, msg) => {
		const el = document.createElement("div");
		el.className = `step ${cls}`;
		el.textContent = msg;
		logEl.appendChild(el);
	};

	const { GgmlWasm, GgufParser, ModelInference, ModelLoader, Tokenizer } =
		await import(`./${bundleName}${assetSuffix}`);

	log("running", "[1/6] WebGPU + WASM init...");
	const wasm = new GgmlWasm();
	await wasm.init({ wasmUrl: `./${wasmVariant}${assetSuffix}` });
	log("pass", "[1/6] backend ready");

	const modelUrl = `./models/${modelId}.gguf`;
	log("running", `[2/6] streaming GGUF from ${modelUrl}...`);
	const resp = await fetch(modelUrl);
	if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
	const total = Number(resp.headers.get("content-length") || 0);
	if (total <= 0) throw new Error("missing content-length");
	const modelPtr = wasm.malloc(total);
	if (!modelPtr) throw new Error(`malloc(${total}) failed`);
	const reader = resp.body.getReader();
	let received = 0;
	while (true) {
		const { done, value } = await reader.read();
		if (done) break;
		wasm.heapU8.set(value, modelPtr + received);
		received += value.length;
	}
	if (received !== total) throw new Error(`short read ${received}/${total}`);
	log("pass", `[2/6] streamed ${(received / 1e6).toFixed(1)} MB`);

	const modelDataAt = (off, len) =>
		new Uint8Array(wasm.heapU8.buffer, modelPtr + off, len);

	log("running", "[3/6] parsing GGUF...");
	const fullView = modelDataAt(0, total);
	const ggufCtx = GgufParser.parse(fullView);
	const parsed = ModelLoader.parseModel(fullView);
	log(
		"pass",
		`[3/6] arch=${parsed.hyperparams.architecture} layers=${parsed.hyperparams.layerCount} fa=${flashAttn}`,
	);

	log("running", "[4/6] loading weights + init KV...");
	let inference;
	try {
		inference = new ModelInference(wasm, parsed.hyperparams, { flashAttn });
		inference.loadWeights(ggufCtx, modelDataAt);
		const ctxLen = Math.min(
			parsed.kvCacheConfig.maxContextLength,
			ctxLenOverride,
		);
		await inference.initKVCache(ctxLen);
		log("pass", `[4/6] weights uploaded, KV ctxLen=${ctxLen}`);
	} catch (e) {
		log("fail", `[4/6] failed: ${e.message}\n${e.stack || ""}`);
		wasm.free(modelPtr);
		throw e;
	} finally {
		wasm.free(modelPtr);
	}

	log("running", "[5/6] tokenize...");
	const tokenizer = new Tokenizer(parsed.tokenizerConfig);
	const ids = tokenizer.encode(prompt);
	if (ids.length < 2) {
		throw new Error(
			`forwardAllPositions requires nTokens >= 2; got ${ids.length}`,
		);
	}
	const tokenIdsArr = new Int32Array(ids);
	const positions = new Int32Array(ids.length);
	for (let i = 0; i < ids.length; i++) positions[i] = i;
	log("pass", `[5/6] nTokens=${ids.length}: [${ids.join(",")}]`);

	const pathLabel =
		pathParam === "forward"
			? "forward() → forwardSingle"
			: "forwardVerify → forwardAllPositions";
	log("running", `[6/6] ${pathLabel}...`);
	const t0 = performance.now();
	try {
		const logits =
			pathParam === "forward"
				? await inference.forward(tokenIdsArr, positions)
				: await inference.forwardVerify(tokenIdsArr, positions);
		const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
		const V = parsed.hyperparams.vocabularySize;
		const rows = logits.length / V;
		// Argmax of last row — convenient sanity-check value.
		// forward() returns 1 row (last position only); forwardVerify returns N rows.
		let argmax = 0;
		let best = -Infinity;
		const off = rows === 1 ? 0 : (ids.length - 1) * V;
		for (let i = 0; i < V; i++) {
			const v = logits[off + i];
			if (v > best) {
				best = v;
				argmax = i;
			}
		}
		log(
			"pass",
			`[6/6] OK in ${elapsed}s · rows=${rows} · last-row argmax=${argmax} (logit=${best.toFixed(3)})`,
		);
		log("pass", "[FA-PREFILL-PROBE-DONE-PASS]");
		window.__faPrefillProbe = { ok: true, elapsed, argmax, best, rows };
	} catch (e) {
		const elapsed = ((performance.now() - t0) / 1000).toFixed(3);
		log(
			"fail",
			`[6/6] TRAPPED after ${elapsed}s: ${e.message}\n${e.stack || ""}`,
		);
		log("fail", "[FA-PREFILL-PROBE-DONE-FAIL]");
		window.__faPrefillProbe = {
			ok: false,
			elapsed,
			error: e.message,
			stack: e.stack,
		};
		throw e;
	}
}

function escapeHtml(s) {
	return s
		.replace(/&/g, "&amp;")
		.replace(/</g, "&lt;")
		.replace(/>/g, "&gt;")
		.replace(/"/g, "&quot;");
}
