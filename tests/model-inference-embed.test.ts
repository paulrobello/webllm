import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";
import { GgufParser } from "../src/models/gguf-parser.js";
import type { GgufContext } from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

const TINYLLAMA = resolve("smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf");
const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
const SHOULD_SKIP = !HAS_WEBGPU || !existsSync(TINYLLAMA);

function hashF32Buffer(arr: Float32Array): string {
	const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
	let h = 0x811c9dc5;
	for (let i = 0; i < bytes.length; i++) {
		h ^= bytes[i];
		h = Math.imul(h, 0x01000193);
	}
	return (h >>> 0).toString(16);
}

describe.skipIf(SHOULD_SKIP)("ModelInference.embed", () => {
	test("does not perturb KV cache or chat logits", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({});
		const inf = new ModelInference(wasm, parsed.hyperparams);
		inf.loadWeights(ggufCtx, view);
		inf.initKVCache(64);

		const chatA = new Int32Array([1, 22172, 920]);
		const posA = new Int32Array([0, 1, 2]);
		const chatB = new Int32Array([530]);
		const posB = new Int32Array([3]);

		await inf.forward(chatA, posA);
		const baselineLogits = await inf.forward(chatB, posB);
		const baselineHash = hashF32Buffer(baselineLogits);

		inf.resetKVCache();
		await inf.forward(chatA, posA);
		expect(inf.cachedTokenCount).toBe(3);

		const embedIds = new Int32Array([100, 200, 300]);
		const embedVec = await inf.embed(embedIds);
		expect(embedVec.length).toBe(parsed.hyperparams.embeddingLength);
		expect(inf.cachedTokenCount).toBe(3);

		let sq = 0;
		for (let i = 0; i < embedVec.length; i++) sq += embedVec[i] * embedVec[i];
		expect(Math.abs(Math.sqrt(sq) - 1.0)).toBeLessThan(1e-5);

		const afterEmbedLogits = await inf.forward(chatB, posB);
		expect(hashF32Buffer(afterEmbedLogits)).toBe(baselineHash);

		await inf.dispose();
		await wasm.shutdown();
	});
});
