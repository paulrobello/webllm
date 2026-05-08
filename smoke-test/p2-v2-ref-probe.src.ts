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

// Stage 4.36 — canonical-6 reference capture. `?model=<key>` URL param
// matches the JSEP spike's registry; non-JSEP webllm-wasm.js is the
// parity reference for `generatedIds[0]`.
const MODEL_REGISTRY: Record<
	string,
	{ ggufUrl: string; promptText: string }
> = {
	tinyllama: {
		ggufUrl: "/models/tinyllama-1.1b-chat-q4_0.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-0.6b": {
		ggufUrl: "/models/qwen3-0.6b-q4f16.gguf",
		promptText: "The capital of France is",
	},
	"qwen3-1.7b": {
		ggufUrl: "/models/qwen3-1.7b-q4f16.gguf",
		promptText: "The capital of France is",
	},
};

function resolveModelKey(): string {
	const params = new URLSearchParams(window.location.search);
	return params.get("model") ?? "tinyllama";
}

const MODEL_KEY = resolveModelKey();
const MODEL_ENTRY = MODEL_REGISTRY[MODEL_KEY];
if (!MODEL_ENTRY) {
	throw new Error(
		`Unknown model key '${MODEL_KEY}'. Known keys: ${Object.keys(MODEL_REGISTRY).join(", ")}`,
	);
}
const N_GENERATE = 5;
const GGUF_URL = MODEL_ENTRY.ggufUrl;

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

		// Stage 4.36 — tokenize the prompt per-model. Replaces the
		// previously hardcoded TinyLlama token IDs.
		const tokenizedPrompt = bridge.tokenize(model, MODEL_ENTRY.promptText, {
			addBos: true,
			parseSpecial: true,
		});
		const promptTokenIds = Array.from(tokenizedPrompt);
		log(
			`     [stage4.36] model=${MODEL_KEY} promptText="${MODEL_ENTRY.promptText}" promptIds=${JSON.stringify(promptTokenIds)}`,
		);
		log(`[5/7] Prefill (${promptTokenIds.length} tokens)...`);
		const promptTokens = new Int32Array(promptTokenIds);
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
		let nPast = promptTokenIds.length;
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

		// Stage 4.31 Probe 18 Shape A — parse the same `[CHECKPOINT-FULL ...]`
		// lines on the non-JSEP reference path. The diff against the JSEP
		// spike's `__stage431Stats` is computed offline.
		const stage431Pat =
			/\[CHECKPOINT-FULL idx=(\d+) name=(\S+) n_elements=(\d+) finite=(\d+) mean=(\S+) abs_max=(\S+) abs_min=(\S+) nan=(\d+) inf=(\d+)\]/;
		const stage431Stats: Array<{
			idx: number;
			name: string;
			n_elements: number;
			finite: number;
			mean: number;
			abs_max: number;
			abs_min: number;
			nan: number;
			inf: number;
		}> = [];
		for (const line of (window as any).__stderrLines as string[]) {
			const m = line.match(stage431Pat);
			if (!m) continue;
			stage431Stats.push({
				idx: +m[1],
				name: m[2],
				n_elements: +m[3],
				finite: +m[4],
				mean: Number(m[5]),
				abs_max: Number(m[6]),
				abs_min: Number(m[7]),
				nan: +m[8],
				inf: +m[9],
			});
		}
		(window as any).__stage431Stats = stage431Stats;
		log(`STAGE431_STATS_COUNT = ${stage431Stats.length}`);
		for (const s of stage431Stats) {
			log(
				`[STAGE-4.31] idx=${s.idx} name=${s.name} n=${s.n_elements} ` +
					`finite=${s.finite} mean=${s.mean} abs_max=${s.abs_max} ` +
					`abs_min=${s.abs_min} nan=${s.nan} inf=${s.inf}`,
			);
		}
		log(`MODEL_KEY = ${MODEL_KEY}`);
		log(`PROMPT_TEXT = ${JSON.stringify(MODEL_ENTRY.promptText)}`);
		log(`PROMPT_IDS = ${JSON.stringify(promptTokenIds)}`);
		log(`LOGIT_STATS_STEP0 = ${JSON.stringify(logitsStep0Stats)}`);
		log(`GENERATED_TOKENS = ${JSON.stringify(generatedIds)}`);
		log(`PER_TOKEN_MS = ${perTokenMs.toFixed(2)}`);
		log(`TOTAL_PREFILL_MS = ${tPrefillMs.toFixed(0)}`);
		log(`MODEL_LOAD_MS = ${tLoadMs.toFixed(0)}`);
		log("DONE", "pass");

		try {
			await fetch("http://localhost:8032/STAGE-4.33-ref.txt", {
				method: "POST",
				body: ((window as any).__stderrLines as string[]).join("\n")
			});
		} catch (e) {
			console.error("Failed to POST logs:", e);
		}

		(window as any).__refResult = {
			modelKey: MODEL_KEY,
			promptText: MODEL_ENTRY.promptText,
			promptIds: promptTokenIds,
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
