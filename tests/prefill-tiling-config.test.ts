import { describe, expect, test } from "bun:test";
import type { ModelHyperparams } from "../src/core/types.js";
import type { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";

const STUB_WASM = {} as unknown as GgmlWasm;
const STUB_HP = {
	architecture: "llama",
	layerCount: 32,
	embeddingLength: 4096,
	headCount: 32,
	headCountKv: 32,
	embeddingHeadLength: 128,
	feedForwardLength: 11008,
	vocabularySize: 32000,
	contextLength: 2048,
	ropeFreqBase: 10000,
	ropeFreqScale: 1.0,
} as unknown as ModelHyperparams;

describe("prefillTileSize ctor option", () => {
	test("defaults to 0", () => {
		const inf = new ModelInference(STUB_WASM, STUB_HP);
		expect(inf.prefillTileSize).toBe(0);
	});

	test("accepts a positive integer", () => {
		const inf = new ModelInference(STUB_WASM, STUB_HP, {
			prefillTileSize: 256,
		});
		expect(inf.prefillTileSize).toBe(256);
	});

	test("0 is explicitly allowed (means disabled)", () => {
		const inf = new ModelInference(STUB_WASM, STUB_HP, { prefillTileSize: 0 });
		expect(inf.prefillTileSize).toBe(0);
	});

	test("rejects negative values", () => {
		expect(
			() => new ModelInference(STUB_WASM, STUB_HP, { prefillTileSize: -1 }),
		).toThrow(/must be >= 0/);
	});

	test("coexists with flashAttn option", () => {
		const inf = new ModelInference(STUB_WASM, STUB_HP, {
			flashAttn: true,
			prefillTileSize: 128,
		});
		expect(inf.flashAttn).toBe(true);
		expect(inf.prefillTileSize).toBe(128);
	});
});
