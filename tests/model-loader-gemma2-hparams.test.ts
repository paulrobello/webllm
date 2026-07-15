import { describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { ModelLoader } from "../src/models/model-loader.js";

const GEMMA2 = "smoke-test/models/gemma-2-2b-q4f16.gguf";

describe.skipIf(!existsSync(GEMMA2))(
	"ModelLoader populates Gemma 2 SWA hparams (Stage 4.2)",
	() => {
		it("derives alternating SWA pattern from period (default 2)", () => {
			const buf = readFileSync(GEMMA2);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;

			expect(hp.architecture).toBe("gemma2");

			// Per upstream gemma2.cpp + llama-hparams.cpp set_swa_pattern with
			// dense_first=false and period=2: swa_layers[il] = (il % 2 < 1) →
			// even layers SWA (true), odd layers global (false).
			expect(hp.slidingWindowPattern).toBeDefined();
			expect(hp.slidingWindowPattern).toHaveLength(hp.layerCount);
			expect(hp.slidingWindowPattern?.[0]).toBe(true); // SWA
			expect(hp.slidingWindowPattern?.[1]).toBe(false); // global
			expect(hp.slidingWindowPattern?.[2]).toBe(true);
			expect(hp.slidingWindowPattern?.[3]).toBe(false);
			// Tail invariant
			expect(hp.slidingWindowPattern?.[hp.layerCount - 1]).toBe(
				(hp.layerCount - 1) % 2 < 1,
			);
		});

		it("reads sliding window size (Gemma 2 default 4096)", () => {
			const buf = readFileSync(GEMMA2);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;

			// bartowski Gemma-2-2b GGUF stores attention.sliding_window=4096.
			expect(hp.slidingWindowSize).toBe(4096);
		});

		it("leaves per-layer geometry arrays absent (uniform Gemma 2)", () => {
			const buf = readFileSync(GEMMA2);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;

			// Gemma 2 has uniform head_count / head_dim / FFN / rope across
			// layers; the per-layer arrays must NOT be populated so downstream
			// dispatch falls back to scalar hp.embeddingHeadLength / headCount /
			// feedForwardLength / ropeFreqBase / ropeDimensionCount.
			expect(hp.embeddingHeadLengthPerLayer).toBeUndefined();
			expect(hp.headCountPerLayer).toBeUndefined();
			expect(hp.headCountKvPerLayer).toBeUndefined();
			expect(hp.feedForwardLengthPerLayer).toBeUndefined();
			expect(hp.ropeDimensionCountPerLayer).toBeUndefined();
			expect(hp.ropeFreqBasePerLayer).toBeUndefined();
		});

		it("does not enable KV reuse (Gemma 2 has no shared-KV layers)", () => {
			const buf = readFileSync(GEMMA2);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;

			expect(hp.sharedKvLayers).toBeUndefined();
			expect(hp.kvReuseFromLayer).toBeUndefined();
		});
	},
);
