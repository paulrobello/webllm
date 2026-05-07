// Stage 4.17 Probe 7 — non-JSEP reference run for the per-node first8
// diff. Loads TinyLlama through the production webllm-wasm.js (same
// llama.cpp + ggml-webgpu the production webllm chat path uses, known
// to produce " Paris" correctly), arms the cb_eval node dump, runs the
// same prefill + greedy 5-decode the JSEP spike runs, then exposes the
// captured CHECKPOINT lines on `window.__refCheckpoints` for an
// agentchrome readout.
//
// Bundle:
//   bun build smoke-test/p2-v2-ref-probe.src.ts \
//     --outfile smoke-test/p2-v2-ref-probe.js --target browser
//
// Run side-by-side with smoke-test/p2-v2-spike.html; diff first8
// across matching checkpoint indices to localize where JSEP diverges.

import { createLlamaBridge } from "../src/inference/llama-bridge.js";

// Same fixture as p2-v2-spike.src.ts and p0-spike.src.ts —
// "The capital of France is".
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

async function runRefProbe(): Promise<void> {
	try {
		log("[1/7] Initializing non-JSEP WASM module...");
		// Inherit the page's `?v=...` query so the dynamic import doesn't
		// hit a stale cached copy of webllm-wasm.js (CLAUDE.md regression
		// lesson: cache-busting must propagate to imported assets).
		const cacheBust = window.location.search || "";
		// @ts-ignore — Emscripten output, no .d.ts
		const createModule = (await import(`./webllm-wasm.js${cacheBust}`)).default;
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});

		log("[2/7] Initializing WebGPU backend...");
		const initStatus = await mod._webgpu_init();
		if (initStatus !== 0) {
			log(`webgpu_init returned ${initStatus}`, "fail");
			return;
		}

		log(`[3/7] Fetching ${GGUF_URL}...`);
		const resp = await fetch(GGUF_URL);
		if (!resp.ok) {
			log(`fetch failed: ${resp.status}`, "fail");
			return;
		}
		const buf = new Uint8Array(await resp.arrayBuffer());
		log(`     loaded ${(buf.byteLength / 1024 / 1024).toFixed(1)} MiB`);

		log("[4/7] Loading model + creating context...");
		const bridge = createLlamaBridge(mod);
		const tLoadStart = performance.now();
		const model = await bridge.loadModel(buf);
		const tLoadMs = performance.now() - tLoadStart;
		const vocab = bridge.nVocab(model);
		const ctx = await bridge.createContext(model, { nCtx: 512 });
		log(`     vocab=${vocab}, load=${tLoadMs.toFixed(0)} ms`);

		// Stage 4.17 Probe 7 — same dump arming as JSEP spike.
		mod._webllm_enable_node_dump(200);

		log(`[5/7] Prefill (${PROMPT_TOKEN_IDS.length} tokens)...`);
		const promptTokens = new Int32Array(PROMPT_TOKEN_IDS);
		const tPrefillStart = performance.now();
		const status = await bridge.decode(ctx, promptTokens, 0);
		const tPrefillMs = performance.now() - tPrefillStart;
		if (status !== 0) {
			log(`     prefill failed status=${status}`, "fail");
			bridge.freeContext(ctx);
			bridge.freeModel(model);
			return;
		}
		log(`     prefill ${tPrefillMs.toFixed(0)} ms`);

		log(`[6/7] Greedy ${N_GENERATE}-token decode...`);
		const generatedIds: number[] = [];
		let nPast = PROMPT_TOKEN_IDS.length;
		const tDecodeStart = performance.now();
		const logitsStep0Stats: { topId: number; topVal: number } = {
			topId: -1,
			topVal: -Infinity,
		};
		for (let step = 0; step < N_GENERATE; ++step) {
			const logits = await bridge.getLogits(ctx, model);
			let topId = 0;
			let topVal = -Infinity;
			for (let i = 0; i < logits.length; i++) {
				if (logits[i] > topVal) {
					topVal = logits[i];
					topId = i;
				}
			}
			if (step === 0) {
				logitsStep0Stats.topId = topId;
				logitsStep0Stats.topVal = topVal;
			}
			generatedIds.push(topId);
			const single = new Int32Array([topId]);
			const dStatus = await bridge.decode(ctx, single, nPast);
			if (dStatus !== 0) {
				log(`     decode step ${step} failed status=${dStatus}`, "fail");
				break;
			}
			nPast++;
		}
		const tDecodeMs = performance.now() - tDecodeStart;
		const perTokenMs = tDecodeMs / N_GENERATE;
		log(`     decode ${tDecodeMs.toFixed(0)} ms (${perTokenMs.toFixed(2)} ms/tok)`);

		log("[7/7] Capturing checkpoints + summary...");
		const checkpointLines = ((window as any).__stderrLines as string[])
			.filter((s) => s.includes("[CHECKPOINT"));
		(window as any).__refCheckpoints = checkpointLines;
		log(`CHECKPOINT_COUNT = ${checkpointLines.length}`);
		for (const line of checkpointLines) log(line);
		log(`LOGIT_STATS_STEP0 = ${JSON.stringify(logitsStep0Stats)}`);
		log(`GENERATED_TOKENS = ${JSON.stringify(generatedIds)}`);
		log(`PER_TOKEN_MS = ${perTokenMs.toFixed(2)}`);
		log(`TOTAL_PREFILL_MS = ${tPrefillMs.toFixed(0)}`);
		log(`MODEL_LOAD_MS = ${tLoadMs.toFixed(0)}`);
		log("DONE", "pass");

		(window as any).__refResult = {
			generatedIds,
			perTokenMs,
			totalPrefillMs: tPrefillMs,
			modelLoadMs: tLoadMs,
			checkpointCount: checkpointLines.length,
			logitStats: logitsStep0Stats,
		};

		bridge.freeContext(ctx);
		bridge.freeModel(model);
	} catch (err: unknown) {
		const e = err as Error;
		log(`FAIL — ${e.message}\n${e.stack ?? ""}`, "fail");
	}
}

void runRefProbe();
