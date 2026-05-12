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

describe("prefillTileSize FA VEC clamp (head_dim > 128)", () => {
	// The clamp activates only when flashAttn === true AND any layer's
	// head_dim exceeds 128. See computeDefaultPrefillTileSize() doc for
	// the upstream FA path-selection assert chain at
	// ggml-webgpu-shader-lib.hpp:734 + :2560.

	test("FA on + head_dim > 128 (Gemma 4 global) → clamp to 16", () => {
		// Gemma 4 E2B: 35 layers, head_dim 256 (SWA) / 512 (global).
		// §22 heuristic would return 128; FA VEC clamp brings it down to 16.
		const hp = {
			...STUB_HP,
			layerCount: 35,
			embeddingLength: 1536,
			embeddingHeadLength: 512,
			embeddingHeadLengthPerLayer: [
				...Array(15).fill(512),
				...Array(20).fill(256),
			],
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { flashAttn: true });
		expect(inf.prefillTileSize).toBe(16);
	});

	test("FA on + uniform head_dim 256 (Gemma 2 / Gemma 3) → clamp to 16", () => {
		// Gemma 2 / Gemma 3 use uniform head_dim 256; no per-layer override.
		const hp = {
			...STUB_HP,
			layerCount: 26,
			embeddingLength: 2304,
			embeddingHeadLength: 256,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { flashAttn: true });
		// layerCount<32 → §22 returns 0; FA clamp still kicks in.
		expect(inf.prefillTileSize).toBe(16);
	});

	test("FA on + head_dim <= 128 (Llama / Qwen / Mistral) → §22 default", () => {
		// TILE / SUBGROUP_MATRIX FA paths fit fine at head_dim <= 128.
		const cases: Array<{
			layerCount: number;
			embeddingHeadLength: number;
			expected: number;
		}> = [
			{ layerCount: 22, embeddingHeadLength: 64, expected: 0 }, // TinyLlama
			{ layerCount: 32, embeddingHeadLength: 128, expected: 128 }, // Llama-3.1-8B
			{ layerCount: 32, embeddingHeadLength: 128, expected: 128 }, // Mistral 7B
			{ layerCount: 28, embeddingHeadLength: 128, expected: 0 }, // Qwen3-1.7B
		];
		for (const c of cases) {
			const hp = {
				...STUB_HP,
				layerCount: c.layerCount,
				embeddingHeadLength: c.embeddingHeadLength,
			} as ModelHyperparams;
			const inf = new ModelInference(STUB_WASM, hp, { flashAttn: true });
			expect(inf.prefillTileSize).toBe(c.expected);
		}
	});

	test("FA off + head_dim > 128 → §22 default (no clamp)", () => {
		// Manual matmul path doesn't go through FA at all; no clamp needed.
		const hp = {
			...STUB_HP,
			layerCount: 35,
			embeddingHeadLength: 512,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { flashAttn: false });
		expect(inf.prefillTileSize).toBe(128);
	});

	test("FA on + Gemma 4 + §22 yields 128 → clamp wins via min", () => {
		// Sanity check on the min() guard: §22 returns 128, FA clamp wants 16,
		// final value must be 16 (the smaller).
		const hp = {
			...STUB_HP,
			layerCount: 35,
			embeddingHeadLength: 512,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, { flashAttn: true });
		expect(inf.prefillTileSize).toBe(16);
	});

	test("explicit prefillTileSize override beats FA clamp too", () => {
		// The override surface should win unconditionally — caller may know
		// something the heuristic doesn't (e.g. running on a backend with a
		// patched FA dispatch).
		const hp = {
			...STUB_HP,
			layerCount: 35,
			embeddingHeadLength: 512,
		} as ModelHyperparams;
		const inf = new ModelInference(STUB_WASM, hp, {
			flashAttn: true,
			prefillTileSize: 64,
		});
		expect(inf.prefillTileSize).toBe(64);
	});
});
