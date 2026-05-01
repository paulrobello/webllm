import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";
import { GgufParser } from "../src/models/gguf-parser.js";
import type { GgufContext } from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

const TINYLLAMA = resolve("smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf");

// Real forward passes require the WebGPU-backed WASM build, which only
// runs in a browser. In Bun we skip — the smoke harness covers this path
// end-to-end. We also skip if the local TinyLlama GGUF is missing.
const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
const SHOULD_SKIP = !HAS_WEBGPU || !existsSync(TINYLLAMA);

describe.skipIf(SHOULD_SKIP)("ModelInference KV snapshot round-trip", () => {
	test("serialize → reset → load reproduces logits at next position", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({} as Parameters<typeof wasm.init>[0]);
		const inf = new ModelInference(wasm, parsed.hyperparams, {
			flashAttn: true,
		});
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		// Hand-crafted token ids — the actual token doesn't matter, only that
		// forward is deterministic given identical inputs.
		const idArr = [1, 22172, 4321, 5678, 9012];
		const N = idArr.length;
		const ids = new Int32Array(idArr);
		const positions = new Int32Array(N);
		for (let i = 0; i < N; i++) positions[i] = i;

		// Reference path: fresh prefill, then forward at position N.
		inf.resetKVCache();
		await inf.forward(ids, positions);
		expect(inf.cachedTokenCount).toBe(N);
		const referenceLogits = await inf.forward(
			new Int32Array([idArr[N - 1]]),
			new Int32Array([N]),
		);

		// Snapshot path: same prefill, serialize, reset, load, forward.
		inf.resetKVCache();
		await inf.forward(ids, positions);
		const snapshot = inf.serializeKVCache(N);
		expect(snapshot.byteLength).toBeGreaterThan(0);
		inf.resetKVCache();
		expect(inf.cachedTokenCount).toBe(0);
		inf.loadKVCache(snapshot, N);
		expect(inf.cachedTokenCount).toBe(N);
		const restoredLogits = await inf.forward(
			new Int32Array([idArr[N - 1]]),
			new Int32Array([N]),
		);

		expect(restoredLogits.length).toBe(referenceLogits.length);
		let maxAbsDiff = 0;
		for (let i = 0; i < referenceLogits.length; i++) {
			const d = Math.abs(restoredLogits[i] - referenceLogits[i]);
			if (d > maxAbsDiff) maxAbsDiff = d;
		}
		expect(maxAbsDiff).toBeLessThan(1e-3);

		await inf.dispose();
		await wasm.shutdown();
	});

	test("partial serialize: serialize(N-1) then prefill of last token == original", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({} as Parameters<typeof wasm.init>[0]);
		const inf = new ModelInference(wasm, parsed.hyperparams, {
			flashAttn: true,
		});
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		const idArr = [1, 22172, 4321, 5678, 9012];
		const N = idArr.length;

		// Reference: prefill all N, then forward last token at position N.
		inf.resetKVCache();
		const allIds = new Int32Array(idArr);
		const allPos = new Int32Array(N);
		for (let i = 0; i < N; i++) allPos[i] = i;
		await inf.forward(allIds, allPos);
		const refLogits = await inf.forward(
			new Int32Array([idArr[N - 1]]),
			new Int32Array([N]),
		);

		// Snapshot: prefill first N-1, serialize at N-1, reset, load, then
		// prefill the Nth token at position N-1, then forward last token at N.
		inf.resetKVCache();
		const headIds = new Int32Array(idArr.slice(0, N - 1));
		const headPos = new Int32Array(N - 1);
		for (let i = 0; i < N - 1; i++) headPos[i] = i;
		await inf.forward(headIds, headPos);
		const partial = inf.serializeKVCache(N - 1);
		inf.resetKVCache();
		inf.loadKVCache(partial, N - 1);
		expect(inf.cachedTokenCount).toBe(N - 1);
		await inf.forward(new Int32Array([idArr[N - 1]]), new Int32Array([N - 1]));
		const tailLogits = await inf.forward(
			new Int32Array([idArr[N - 1]]),
			new Int32Array([N]),
		);

		expect(tailLogits.length).toBe(refLogits.length);
		let maxAbsDiff = 0;
		for (let i = 0; i < refLogits.length; i++) {
			const d = Math.abs(tailLogits[i] - refLogits[i]);
			if (d > maxAbsDiff) maxAbsDiff = d;
		}
		expect(maxAbsDiff).toBeLessThan(1e-3);

		await inf.dispose();
		await wasm.shutdown();
	});

	test("loadKVCache rejects mismatched byte length", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({} as Parameters<typeof wasm.init>[0]);
		const inf = new ModelInference(wasm, parsed.hyperparams, {
			flashAttn: true,
		});
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		const tooSmall = new Uint8Array(8);
		expect(() => inf.loadKVCache(tooSmall, 4)).toThrow(/byte/i);

		await inf.dispose();
		await wasm.shutdown();
	});
});
