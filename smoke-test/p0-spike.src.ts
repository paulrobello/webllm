// Tier 3 P0 spike harness — load TinyLlama through llama_decode and assert
// top-1 == " Paris" (id 3681). Hardcoded prompt token IDs come from
// eval/reports/p0-spike-2026-05-05/PROMPT-FIXTURE.md (P1 will replace
// with llama_tokenize).
//
// Bundled to smoke-test/p0-spike.js via:
//   bun build smoke-test/p0-spike.src.ts --outfile smoke-test/p0-spike.js \
//     --target browser

import { createLlamaBridge } from "../src/inference/llama-bridge.js";

// Fixture from PROMPT-FIXTURE.md
const PROMPT_TOKEN_IDS = [1, 450, 7483, 310, 3444, 338];
const EXPECTED_PARIS_ID = 3681;
const GGUF_URL = "/models/tinyllama-1.1b-chat-q4_0.gguf";

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	// Also mirror to console so agentchrome console-follow captures it.
	console.log(msg);
}

async function runSpike(): Promise<void> {
	try {
		log("[1/6] Initializing WASM module...");
		// Dynamic import — the smoke-serve root is smoke-test/, so this resolves
		// relative to the served HTML (./webllm-wasm.js).
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm.js")).default;
		const mod: any = await createModule();

		log("[2/6] Initializing WebGPU backend...");
		// _webgpu_init is the existing async-via-ASYNCIFY backend init —
		// must be awaited (returns a Promise on the suspend path).
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}

		log(`[3/6] Fetching TinyLlama Q4_0 GGUF from ${GGUF_URL}...`);
		const resp = await fetch(GGUF_URL);
		if (!resp.ok) {
			log(`fetch failed: ${resp.status} ${resp.statusText}`, "fail");
			return;
		}
		const buf = new Uint8Array(await resp.arrayBuffer());
		log(`     loaded ${(buf.byteLength / 1024 / 1024).toFixed(1)} MiB`);

		log("[4/6] Loading model + creating context...");
		const bridge = createLlamaBridge(mod);
		const t0 = performance.now();
		const model = await bridge.loadModel(buf);
		const tLoad = performance.now() - t0;
		const vocab = bridge.nVocab(model);
		log(`     model loaded in ${tLoad.toFixed(0)} ms; vocab = ${vocab}`);
		const ctx = bridge.createContext(model, { nCtx: 512 });

		log(`[5/6] Decoding prompt (${PROMPT_TOKEN_IDS.length} tokens)...`);
		const promptTokens = new Int32Array(PROMPT_TOKEN_IDS);
		const tDecode = performance.now();
		const status = await bridge.decode(ctx, promptTokens, 0);
		const tDecodeMs = performance.now() - tDecode;
		log(`     llama_decode status = ${status} (${tDecodeMs.toFixed(0)} ms)`);
		if (status !== 0) {
			log(`     llama_decode FAILED with status ${status}`, "fail");
			bridge.freeContext(ctx);
			bridge.freeModel(model);
			return;
		}

		log("[6/6] Reading logits + argmax...");
		const logits = bridge.getLogits(ctx, model);
		let topId = 0;
		let topVal = -Infinity;
		for (let i = 0; i < logits.length; i++) {
			if (logits[i] > topVal) {
				topVal = logits[i];
				topId = i;
			}
		}
		log(`     top-1 token id = ${topId} (logit ${topVal.toFixed(3)})`);
		log(`     expected " Paris" id ${EXPECTED_PARIS_ID}`);

		if (topId === EXPECTED_PARIS_ID) {
			log("PASS — top-1 matches \" Paris\"", "pass");
		} else {
			log(`FAIL — got id ${topId} instead of ${EXPECTED_PARIS_ID}`, "fail");
		}

		bridge.freeContext(ctx);
		bridge.freeModel(model);
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runSpike();
