import { describe, expect, test } from "bun:test";
import type { ModelHyperparams } from "../src/core/types.js";
import type { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";

const STUB_WASM = {} as unknown as GgmlWasm;
// Sub-7B shape (TinyLlama-1.1B): layerCount=22, embeddingLength=2048.
// Both gates of the §30 heuristic fail, so `defaults to 0` still
// reflects "no override AND heuristic returns 0." Boundary tests
// below override these fields explicitly to exercise the gates.
const STUB_HP = {
	architecture: "llama",
	layerCount: 22,
	embeddingLength: 2048,
	headCount: 32,
	headCountKv: 32,
	embeddingHeadLength: 64,
	feedForwardLength: 5632,
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

describe("prefillTileSize heuristic default", () => {
	test("layerCount>=32 AND embeddingLength>=4096 → 128", () => {
		const hp = {
			...STUB_HP,
			layerCount: 32,
			embeddingLength: 4096,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp);
		expect(inf.prefillTileSize).toBe(128);
	});

	test("either gate fails → 0", () => {
		const cases: Array<Partial<ModelHyperparams>> = [
			{ layerCount: 31, embeddingLength: 4096 }, // layer below
			{ layerCount: 32, embeddingLength: 2048 }, // emb below
			{ layerCount: 16, embeddingLength: 2048 }, // both below
		];
		for (const overrides of cases) {
			const hp = { ...STUB_HP, ...overrides } as ModelHyperparams;
			const inf = new ModelInference(STUB_WASM, hp);
			expect(inf.prefillTileSize).toBe(0);
		}
	});

	test("explicit prefillTileSize: 0 overrides heuristic-128", () => {
		const hp = {
			...STUB_HP,
			layerCount: 32,
			embeddingLength: 4096,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { prefillTileSize: 0 });
		expect(inf.prefillTileSize).toBe(0);
	});
});
