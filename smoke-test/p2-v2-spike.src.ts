// P2-v2 Phase 2 follow-on spike harness — exercise the JSEP backend
// end-to-end via webllm_decode and measure the T3 gate metrics.
//
// Loads tinyllama-1.1b-chat-q4_0.gguf through the JSEP-built WASM
// (`webllm-wasm-jsep.js`), installs JSEP callbacks, decodes the
// "The capital of France is" 6-token prompt, then greedy-decodes 5
// continuation tokens. After the loop, snapshots `module.__jsep.counters`
// and reports per-token wall + EM_ASM crossings/token.
//
// Bundled to smoke-test/p2-v2-spike.js via:
//   bun build smoke-test/p2-v2-spike.src.ts --outfile smoke-test/p2-v2-spike.js \
//     --target browser

import { installJsepCallbacks } from "../src/inference/jsep/index.js";
import { createLlamaBridge } from "../src/inference/llama-bridge.js";

// Same fixture as p0-spike.src.ts — "The capital of France is".
const PROMPT_TOKEN_IDS = [1, 450, 7483, 310, 3444, 338];
const N_GENERATE = 5;
const GGUF_URL = "/models/tinyllama-1.1b-chat-q4_0.gguf";

function log(msg: string, cls = ""): void {
	const el = document.getElementById("log");
	if (!el) return;
	const line = document.createElement("div");
	if (cls) line.className = cls;
	line.textContent = msg;
	el.appendChild(line);
	console.log(msg);
}

async function runSpike(): Promise<void> {
	try {
		log("[1/8] Initializing JSEP WASM module...");
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import("./webllm-wasm-jsep.js")).default;
		const mod: any = await createModule();

		log("[2/8] Acquiring WebGPU device...");
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) {
			log("no WebGPU adapter", "fail");
			return;
		}
		const device = await adapter.requestDevice();

		log("[3/8] Installing JSEP callbacks (must precede webllm_load_model)...");
		installJsepCallbacks(mod, device);

		log("[4/8] Initializing ggml-webgpu backend...");
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}

		log(`[5/8] Fetching GGUF from ${GGUF_URL}...`);
		const resp = await fetch(GGUF_URL);
		if (!resp.ok) {
			log(`fetch failed: ${resp.status} ${resp.statusText}`, "fail");
			return;
		}
		const buf = new Uint8Array(await resp.arrayBuffer());
		log(`     loaded ${(buf.byteLength / 1024 / 1024).toFixed(1)} MiB`);

		log("[6/8] Loading model + creating context...");
		const bridge = createLlamaBridge(mod);
		const t0 = performance.now();
		const model = await bridge.loadModel(buf);
		const tLoadMs = performance.now() - t0;
		const vocab = bridge.nVocab(model);
		log(`     model loaded in ${tLoadMs.toFixed(0)} ms; vocab = ${vocab}`);
		const ctx = await bridge.createContext(model, { nCtx: 512 });

		// Snapshot counters AFTER model load so model-load JSEP traffic
		// (alloc bursts, weight uploads) doesn't pollute per-token rate.
		const counter0: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		log(`     counters@load = ${JSON.stringify(counter0)}`);

		log(`[7/8] Decoding prompt (${PROMPT_TOKEN_IDS.length} tokens)...`);
		const promptTokens = new Int32Array(PROMPT_TOKEN_IDS);
		const tPrefillStart = performance.now();
		let status = await bridge.decode(ctx, promptTokens, 0);
		const tPrefillMs = performance.now() - tPrefillStart;
		if (status !== 0) {
			log(`     prefill failed status=${status}`, "fail");
			bridge.freeContext(ctx);
			bridge.freeModel(model);
			return;
		}
		log(`     prefill ${tPrefillMs.toFixed(0)} ms`);

		log(`[8/8] Greedy decoding ${N_GENERATE} tokens...`);
		const generatedIds: number[] = [];
		let nPast = PROMPT_TOKEN_IDS.length;
		const tDecodeStart = performance.now();
		for (let step = 0; step < N_GENERATE; step++) {
			const logits = await bridge.getLogits(ctx, model);
			let topId = 0;
			let topVal = -Infinity;
			for (let i = 0; i < logits.length; i++) {
				if (logits[i] > topVal) {
					topVal = logits[i];
					topId = i;
				}
			}
			generatedIds.push(topId);
			const single = new Int32Array([topId]);
			status = await bridge.decode(ctx, single, nPast);
			nPast++;
			if (status !== 0) {
				log(`decode step ${step} failed status=${status}`, "fail");
				break;
			}
		}
		const tDecodeMs = performance.now() - tDecodeStart;

		const counter1: Record<string, number> = {
			...(mod.__jsep?.counters ?? {}),
		};
		const deltas: Record<string, number> = {};
		for (const k of Object.keys(counter1)) {
			deltas[k] = (counter1[k] ?? 0) - (counter0[k] ?? 0);
		}

		const perTokenMs = tDecodeMs / N_GENERATE;
		const totalCrossings =
			(deltas.alloc ?? 0) +
			(deltas.free ?? 0) +
			(deltas.write ?? 0) +
			(deltas.read ?? 0) +
			(deltas.clear ?? 0) +
			(deltas.runOp ?? 0) +
			(deltas.sync ?? 0);
		const crossingsPerToken = totalCrossings / N_GENERATE;

		// Detokenize generated ids into a readable string.
		let generatedText = "";
		try {
			generatedText = bridge.detokenize(model, new Int32Array(generatedIds));
		} catch (e) {
			generatedText = `<detokenize failed: ${(e as Error).message}>`;
		}

		log(`GENERATED_TOKENS = ${JSON.stringify(generatedIds)}`);
		log(`GENERATED_TEXT = ${JSON.stringify(generatedText)}`);
		log(`PER_TOKEN_MS = ${perTokenMs.toFixed(2)}`);
		log(`COUNTER_DELTAS = ${JSON.stringify(deltas)}`);
		log(`CROSSINGS_PER_TOKEN = ${crossingsPerToken.toFixed(1)}`);
		log(`TOTAL_DECODE_MS = ${tDecodeMs.toFixed(0)}`);
		log(`TOTAL_PREFILL_MS = ${tPrefillMs.toFixed(0)}`);
		log(`MODEL_LOAD_MS = ${tLoadMs.toFixed(0)}`);
		log("DONE", "pass");

		// Expose for agentchrome `js exec` readout.
		(window as any).__spikeResult = {
			generatedIds,
			generatedText,
			perTokenMs,
			deltas,
			crossingsPerToken,
			totalDecodeMs: tDecodeMs,
			totalPrefillMs: tPrefillMs,
			modelLoadMs: tLoadMs,
			counter0,
			counter1,
		};

		bridge.freeContext(ctx);
		bridge.freeModel(model);
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runSpike();
