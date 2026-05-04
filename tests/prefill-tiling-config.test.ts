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
	test("layerCount>=32 → 128 regardless of embeddingLength", () => {
		// Boundary at 32 layers (Mistral 7B / Llama-3.1-8B canonical).
		const cases: Array<{ layerCount: number; embeddingLength: number }> = [
			{ layerCount: 32, embeddingLength: 4096 }, // Mistral 7B / Llama-3.1-8B
			{ layerCount: 36, embeddingLength: 2560 }, // qwen3-4b — was bypassed by the AND gate, hits §22 abort on tc-005
			{ layerCount: 36, embeddingLength: 4096 }, // qwen3-8b
			{ layerCount: 32, embeddingLength: 2048 }, // hypothetical 32-layer model with smaller emb — still high enough graph pressure
		];
		for (const overrides of cases) {
			const hp = { ...STUB_HP, ...overrides } as ModelHyperparams;
			const inf = new ModelInference(STUB_WASM, hp);
			expect(inf.prefillTileSize).toBe(128);
		}
	});

	test("layerCount<32 → 0 (no tiling needed)", () => {
		// Sub-32-layer models stay below the graph allocator's per-tile
		// budget at the seq lengths the bench exercises. Adding tiling
		// overhead unnecessarily would just slow them down.
		const cases: Array<{ layerCount: number; embeddingLength: number }> = [
			{ layerCount: 28, embeddingLength: 2048 }, // qwen3-1.7b
			{ layerCount: 28, embeddingLength: 2560 }, // qwen3-4b would have been here if it were 28 layers
			{ layerCount: 22, embeddingLength: 2048 }, // tinyllama
			{ layerCount: 16, embeddingLength: 2048 }, // smollm2-1.7b
			{ layerCount: 31, embeddingLength: 4096 }, // hypothetical just-below boundary
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
