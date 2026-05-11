import { describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { ModelLoader } from "../../src/models/model-loader.js";

const GEMMA4 = "smoke-test/models/gemma-4-e2b-it-q4km.gguf";

describe.skipIf(!existsSync(GEMMA4))(
	"ModelLoader populates Gemma 4 per-layer hparams",
	() => {
		it("populates all per-layer arrays from GGUF", () => {
			const buf = readFileSync(GEMMA4);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;

			expect(hp.architecture).toBe("gemma4");
			expect(hp.layerCount).toBe(35);

			// Pattern (T,T,T,T,F) × 7 → indices 4, 9, 14, 19, 24, 29, 34 = global
			expect(hp.slidingWindowPattern).toBeDefined();
			expect(hp.slidingWindowPattern).toHaveLength(35);
			expect(hp.slidingWindowPattern?.[4]).toBe(false); // global
			expect(hp.slidingWindowPattern?.[0]).toBe(true); // SWA

			// head_dim: 512 global, 256 SWA
			expect(hp.embeddingHeadLengthPerLayer).toBeDefined();
			expect(hp.embeddingHeadLengthPerLayer?.[0]).toBe(256);
			expect(hp.embeddingHeadLengthPerLayer?.[4]).toBe(512);

			// rope_dim: 512 global, 256 SWA
			expect(hp.ropeDimensionCountPerLayer?.[0]).toBe(256);
			expect(hp.ropeDimensionCountPerLayer?.[4]).toBe(512);

			// freq_base: 1e6 global, 1e4 SWA
			expect(hp.ropeFreqBasePerLayer?.[0]).toBe(10_000);
			expect(hp.ropeFreqBasePerLayer?.[4]).toBe(1_000_000);

			// FFN: 6144 layers 0-14, 12288 layers 15-34
			expect(hp.feedForwardLengthPerLayer).toBeDefined();
			expect(hp.feedForwardLengthPerLayer?.[0]).toBe(6144);
			expect(hp.feedForwardLengthPerLayer?.[14]).toBe(6144);
			expect(hp.feedForwardLengthPerLayer?.[15]).toBe(12288);
			expect(hp.feedForwardLengthPerLayer?.[34]).toBe(12288);

			// Other Gemma 4 fields
			expect(hp.slidingWindowSize).toBe(512);
			expect(hp.sharedKvLayers).toBe(20);
			expect(hp.finalLogitSoftcap).toBe(30);
		});
	},
);
