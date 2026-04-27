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

describe.skipIf(SHOULD_SKIP)("ModelInference.forwardVerify", () => {
	test("returns same last-row logits as forward() for nTokens=2", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({});
		const inf = new ModelInference(wasm, parsed.hyperparams);
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		const ids = new Int32Array([1, 22172]);
		const pos = new Int32Array([0, 1]);

		const lastRow = await inf.forward(ids, pos);
		expect(inf.cachedTokenCount).toBe(2);

		inf.resetKVCache();
		const allRows = await inf.forwardVerify(ids, pos);
		expect(inf.cachedTokenCount).toBe(2);

		const vocab = parsed.hyperparams.vocabularySize;
		expect(allRows.length).toBe(2 * vocab);

		const lastRowFromVerify = allRows.subarray(vocab, 2 * vocab);
		let maxAbsDiff = 0;
		for (let i = 0; i < vocab; i++) {
			const d = Math.abs(lastRowFromVerify[i] - lastRow[i]);
			if (d > maxAbsDiff) maxAbsDiff = d;
		}
		expect(maxAbsDiff).toBeLessThan(1e-3);

		await inf.dispose();
		await wasm.shutdown();
	});

	test("rejects nTokens < 2", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({});
		const inf = new ModelInference(wasm, parsed.hyperparams);
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		await expect(
			inf.forwardVerify(new Int32Array([1]), new Int32Array([0])),
		).rejects.toThrow("nTokens >= 2");

		await inf.dispose();
		await wasm.shutdown();
	});
});
