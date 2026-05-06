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
import {
	dispatchMatmul,
	GGML_TYPE_F32,
	GGML_TYPE_Q4_K,
	type JsepOpDescriptor,
} from "../src/inference/jsep/ops/matmul.js";
import {
	GGML_OP_MUL_MAT,
	GGML_OP_RMS_NORM,
} from "../src/inference/jsep/index.js";
import { dispatchRmsNorm } from "../src/inference/jsep/ops/rms-norm.js";

// Same fixture as p0-spike.src.ts — "The capital of France is".
const PROMPT_TOKEN_IDS = [1, 450, 7483, 310, 3444, 338];
const N_GENERATE = 5;
const GGUF_URL = "/models/tinyllama-1.1b-chat-q4_0.gguf";

// ----- Q4_K kernel self-test ------------------------------------------------

function f32ToF16Bits(f: number): number {
	const buf = new ArrayBuffer(4);
	new Float32Array(buf)[0] = f;
	const u32 = new Uint32Array(buf)[0];
	const sign = (u32 >> 31) & 0x1;
	const expF32 = (u32 >> 23) & 0xff;
	const mantF32 = u32 & 0x7fffff;
	if (expF32 === 0xff) return (sign << 15) | 0x7c00 | (mantF32 ? 1 : 0);
	const expF16 = expF32 - 127 + 15;
	if (expF16 >= 0x1f) return (sign << 15) | 0x7c00;
	if (expF16 <= 0) return sign << 15;
	return (sign << 15) | (expF16 << 10) | (mantF32 >> 13);
}

function f16BitsToF32(bits: number): number {
	const sign = (bits >> 15) & 0x1;
	const exp = (bits >> 10) & 0x1f;
	const mant = bits & 0x3ff;
	if (exp === 0) {
		if (mant === 0) return sign ? -0 : 0;
		const v = 2 ** -14 * (mant / 1024);
		return sign ? -v : v;
	}
	if (exp === 0x1f) return mant ? NaN : sign ? -Infinity : Infinity;
	const val = 2 ** (exp - 15) * (1 + mant / 1024);
	return sign ? -val : val;
}

// Pack a single Q4_K super-block (256 elems / 144 bytes) given explicit
// (d, dmin, sc[8], m[8], nibbles[256]).
function packQ4_KSingle(
	d: number,
	dmin: number,
	sc: Uint8Array,
	m: Uint8Array,
	nibbles: Uint8Array,
): { bytes: Uint8Array; dequant: Float32Array } {
	const bytes = new Uint8Array(144);
	const view = new DataView(bytes.buffer);
	view.setUint16(0, f32ToF16Bits(d), true);
	view.setUint16(2, f32ToF16Bits(dmin), true);
	for (let j = 0; j < 4; j++) {
		bytes[4 + j] = (sc[j] & 0x3f) | (((sc[j + 4] >> 4) & 0x3) << 6);
		bytes[4 + j + 4] = (m[j] & 0x3f) | (((m[j + 4] >> 4) & 0x3) << 6);
		bytes[4 + j + 8] = (sc[j + 4] & 0xf) | ((m[j + 4] & 0xf) << 4);
	}
	for (let pair = 0; pair < 4; pair++) {
		for (let l = 0; l < 32; l++) {
			const lo = nibbles[pair * 64 + l] & 0xf;
			const hi = nibbles[pair * 64 + 32 + l] & 0xf;
			bytes[16 + pair * 32 + l] = lo | (hi << 4);
		}
	}
	const dequant = new Float32Array(256);
	const decodedD = f16BitsToF32(f32ToF16Bits(d));
	const decodedDmin = f16BitsToF32(f32ToF16Bits(dmin));
	for (let pair = 0; pair < 4; pair++) {
		const is0 = pair * 2;
		const is1 = pair * 2 + 1;
		const d1 = decodedD * sc[is0];
		const m1 = decodedDmin * m[is0];
		const d2 = decodedD * sc[is1];
		const m2 = decodedDmin * m[is1];
		for (let l = 0; l < 32; l++) {
			dequant[pair * 64 + l] = d1 * nibbles[pair * 64 + l] - m1;
			dequant[pair * 64 + 32 + l] = d2 * nibbles[pair * 64 + 32 + l] - m2;
		}
	}
	return { bytes, dequant };
}

async function runQ4KSelfTest(
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
): Promise<void> {
	const M = 1;
	const K = 256;
	const N = 1;
	const sc = new Uint8Array([5, 12, 19, 26, 33, 40, 47, 54]);
	const m = new Uint8Array([3, 7, 11, 15, 19, 23, 27, 31]);
	const nibbles = new Uint8Array(256);
	for (let i = 0; i < 256; i++) nibbles[i] = i & 0xf;
	const { bytes: q4Bytes, dequant: src0Dequant } = packQ4_KSingle(
		0.05,
		0.01,
		sc,
		m,
		nibbles,
	);
	const src1 = new Float32Array(N * K);
	for (let i = 0; i < src1.length; i++) src1[i] = ((i % 13) - 6) * 0.15;

	let reference = 0;
	for (let k = 0; k < K; k++) reference += src0Dequant[k] * src1[k];

	const h0 = runtime.dataManager.alloc(q4Bytes.byteLength);
	const h1 = runtime.dataManager.alloc(src1.byteLength);
	const hd = runtime.dataManager.alloc(M * N * 4);
	const rec0 = runtime.dataManager.get(h0);
	const rec1 = runtime.dataManager.get(h1);
	runtime.device.queue.writeBuffer(
		rec0.buffer,
		0,
		q4Bytes,
		0,
		q4Bytes.byteLength,
	);
	runtime.device.queue.writeBuffer(
		rec1.buffer,
		0,
		new Uint8Array(src1.buffer),
		0,
		src1.byteLength,
	);

	const desc: JsepOpDescriptor = {
		op: GGML_OP_MUL_MAT,
		nSrc: 2,
		dst: {
			bufHandle: hd,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [M, N, 1, 1],
			nb: [4, 4, 4 * M, 4 * M * N],
		},
		srcs: [
			{
				bufHandle: h0,
				offset: 0,
				type: GGML_TYPE_Q4_K,
				ne: [K, M, 1, 1],
				nb: [144, 144 * (K / 256), 144 * (K / 256) * M, 0],
			},
			{
				bufHandle: h1,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [K, N, 1, 1],
				nb: [4, 4 * K, 4 * K * N, 0],
			},
		],
	};
	const status = dispatchMatmul(runtime, desc);
	runtime.encoderBatcher.flush();

	const recd = runtime.dataManager.get(hd);
	const staging = runtime.device.createBuffer({
		size: M * N * 4,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(recd.buffer, 0, staging, 0, M * N * 4);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, M * N * 4);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();
	const passLog = {
		status,
		got: got[0],
		reference,
		delta: Math.abs(got[0] - reference),
		dequantFirst4: Array.from(src0Dequant.slice(0, 4)),
		dequantLast4: Array.from(src0Dequant.slice(-4)),
	};
	log(`Q4K_SELFTEST = ${JSON.stringify(passLog)}`);
	(window as any).__q4kSelfTest = passLog;

	runtime.dataManager.free(h0);
	runtime.dataManager.free(h1);
	runtime.dataManager.free(hd);
}

// ----- RMS_NORM real-shape self-test ---------------------------------------
//
// Stage 3.5 Step 1: drive `dispatchRmsNorm` with a row whose column count
// matches TinyLlama's first model RMS_NORM (cols=2048). The kernel was
// written and validated only via the synthetic golden-test harness in
// Phase 2. If RMS_NORM produces zero (or NaN, or wrong-magnitude) output
// for non-zero input on real-model shapes, the residual stream collapses
// to zero on the first layer's first norm — fully consistent with the
// observed Outcome C all-zero logits.

interface MallocModule {
	_malloc: (n: number) => number;
	_free: (p: number) => void;
	HEAPU8: Uint8Array;
	HEAPF32: Float32Array;
}

async function runRmsNormSelfTest(
	mod: MallocModule,
	runtime: import("../src/inference/jsep/index.js").JsepRuntime,
): Promise<void> {
	const rows = 1;
	const cols = 2048;
	const eps = 1e-5;

	// Non-zero input pattern that exercises both signs and a non-trivial
	// magnitude. Mean(x²) is ~0.21 for this pattern, so inv_rms is near 2.18.
	const x = new Float32Array(rows * cols);
	for (let i = 0; i < x.length; i++) x[i] = ((i % 17) - 8) * 0.1;

	// CPU reference.
	let sumSq = 0;
	for (let i = 0; i < cols; i++) sumSq += x[i] * x[i];
	const invRms = 1 / Math.sqrt(sumSq / cols + eps);
	const reference = new Float32Array(cols);
	for (let i = 0; i < cols; i++) reference[i] = x[i] * invRms;

	const totalBytes = x.byteLength;
	const hX = runtime.dataManager.alloc(totalBytes);
	const hOut = runtime.dataManager.alloc(totalBytes);

	// Upload x into the JSEP-side input buffer.
	runtime.device.queue.writeBuffer(
		runtime.dataManager.get(hX).buffer,
		0,
		new Uint8Array(x.buffer),
		0,
		totalBytes,
	);

	// `dispatchRmsNorm` reads eps from `heapBuffer[opParamsPtr]` as f32, so
	// allocate 4 bytes in the wasm heap and write eps there. This mirrors
	// the production path where `op_params[0]` is a node field on the
	// wasm side.
	const opParamsPtr = mod._malloc(4);
	mod.HEAPF32[opParamsPtr >>> 2] = eps;

	const desc: JsepOpDescriptor = {
		op: GGML_OP_RMS_NORM,
		nSrc: 1,
		dst: {
			bufHandle: hOut,
			offset: 0,
			type: GGML_TYPE_F32,
			ne: [cols, rows, 1, 1],
			nb: [4, 4 * cols, 4 * cols * rows, 0],
		},
		srcs: [
			{
				bufHandle: hX,
				offset: 0,
				type: GGML_TYPE_F32,
				ne: [cols, rows, 1, 1],
				nb: [4, 4 * cols, 4 * cols * rows, 0],
			},
		],
	};

	const status = dispatchRmsNorm(
		{
			device: runtime.device,
			dataManager: runtime.dataManager,
			encoderBatcher: runtime.encoderBatcher,
			pipelineCache: runtime.pipelineCache,
			bindGroupLayoutCache: runtime.bindGroupLayoutCache,
		},
		desc,
		opParamsPtr,
		mod.HEAPU8.buffer,
	);
	runtime.encoderBatcher.flush();

	const recOut = runtime.dataManager.get(hOut);
	const staging = runtime.device.createBuffer({
		size: totalBytes,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const enc = runtime.device.createCommandEncoder();
	enc.copyBufferToBuffer(recOut.buffer, 0, staging, 0, totalBytes);
	runtime.device.queue.submit([enc.finish()]);
	await staging.mapAsync(GPUMapMode.READ, 0, totalBytes);
	const got = new Float32Array(staging.getMappedRange().slice(0));
	staging.unmap();
	staging.destroy();

	let maxAbsDelta = 0;
	let hasNaN = false;
	let hasInf = false;
	let zeroCount = 0;
	for (let i = 0; i < cols; i++) {
		const v = got[i];
		if (Number.isNaN(v)) hasNaN = true;
		else if (!Number.isFinite(v)) hasInf = true;
		else if (v === 0) zeroCount++;
		const d = Math.abs(v - reference[i]);
		if (d > maxAbsDelta) maxAbsDelta = d;
	}

	const passLog = {
		status,
		invRms,
		first4Got: Array.from(got.slice(0, 4)),
		first4Ref: Array.from(reference.slice(0, 4)),
		last4Got: Array.from(got.slice(-4)),
		last4Ref: Array.from(reference.slice(-4)),
		maxAbsDelta,
		hasNaN,
		hasInf,
		zeroCount,
	};
	log(`RMSNORM_SELFTEST = ${JSON.stringify(passLog)}`);
	(window as any).__rmsNormSelfTest = passLog;

	mod._free(opParamsPtr);
	runtime.dataManager.free(hX);
	runtime.dataManager.free(hOut);
}

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
		(window as any).__stderrLines = [];
		const mod: any = await createModule({
			printErr: (s: string) => {
				(window as any).__stderrLines.push(s);
				console.error(s);
			},
		});

		log("[2/8] Acquiring WebGPU device...");
		const adapter = await navigator.gpu?.requestAdapter();
		if (!adapter) {
			log("no WebGPU adapter", "fail");
			return;
		}
		const device = await adapter.requestDevice();

		log("[3/8] Installing JSEP callbacks (must precede webllm_load_model)...");
		const runtime = installJsepCallbacks(mod, device);

		// Stage 3 self-test: hand-crafted Q4_K matmul against a CPU reference
		// to localize whether the all-zeros logits chain comes from the Q4_K
		// kernel or from elsewhere (RMS_NORM, splits, CPY).
		await runQ4KSelfTest(runtime);

		// Stage 3.5 Step 1 self-test: real-model RMS_NORM shape (cols=2048,
		// matches TinyLlama first norm). Disambiguates whether the residual
		// stream is being zeroed at the first norm vs further downstream.
		await runRmsNormSelfTest(mod as MallocModule, runtime);

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
		const logitStats: Array<{
			step: number;
			first8: number[];
			topVal: number;
			topId: number;
			hasNaN: boolean;
			hasInf: boolean;
			finiteCount: number;
			minFinite: number;
			maxFinite: number;
		}> = [];
		for (let step = 0; step < N_GENERATE; step++) {
			const logits = await bridge.getLogits(ctx, model);
			let topId = 0;
			let topVal = -Infinity;
			let hasNaN = false;
			let hasInf = false;
			let finiteCount = 0;
			let minFinite = Infinity;
			let maxFinite = -Infinity;
			for (let i = 0; i < logits.length; i++) {
				const v = logits[i];
				if (Number.isNaN(v)) {
					hasNaN = true;
				} else if (!Number.isFinite(v)) {
					hasInf = true;
				} else {
					finiteCount++;
					if (v < minFinite) minFinite = v;
					if (v > maxFinite) maxFinite = v;
				}
				if (v > topVal) {
					topVal = v;
					topId = i;
				}
			}
			if (step === 0) {
				logitStats.push({
					step,
					first8: Array.from(logits.slice(0, 8)),
					topVal,
					topId,
					hasNaN,
					hasInf,
					finiteCount,
					minFinite,
					maxFinite,
				});
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

		log(`LOGIT_STATS_STEP0 = ${JSON.stringify(logitStats[0])}`);
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
