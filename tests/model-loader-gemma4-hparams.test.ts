import { describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { ModelLoader } from "../src/models/model-loader.js";

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

		it("loads PLE tensors and pleDim when arch is gemma4", () => {
			const buf = readFileSync(GEMMA4);
			const parsed = ModelLoader.parseModel(buf);
			const hp = parsed.hyperparams;
			const ple = parsed.pleTensors;

			// pleDim exposed on hyperparams
			expect(hp.pleDim).toBe(256);

			// All three PLE tensor descriptors are present
			expect(ple).toBeDefined();
			expect(ple?.perLayerEmbed).toBeDefined();
			expect(ple?.perLayerProj).toBeDefined();
			expect(ple?.perLayerProjNorm).toBeDefined();

			// per_layer_token_embd.weight: [8960, 262144] Q5_K
			// 8960 = 256 (pleDim) × 35 (layerCount)
			expect(ple?.perLayerEmbed.name).toBe("per_layer_token_embd.weight");
			expect(ple?.perLayerEmbed.dimensions).toHaveLength(2);
			expect(ple?.perLayerEmbed.dimensions[0]).toBe(8960); // pleDim × layerCount
			expect(ple?.perLayerEmbed.dimensions[1]).toBe(262144); // vocabSize

			// per_layer_model_proj.weight: [1536, 8960] BF16
			expect(ple?.perLayerProj.name).toBe("per_layer_model_proj.weight");
			expect(ple?.perLayerProj.dimensions).toHaveLength(2);
			expect(ple?.perLayerProj.dimensions[0]).toBe(1536); // hiddenDim
			expect(ple?.perLayerProj.dimensions[1]).toBe(8960); // pleDim × layerCount

			// per_layer_proj_norm.weight: [256] F32 (RMSNorm scale)
			expect(ple?.perLayerProjNorm.name).toBe("per_layer_proj_norm.weight");
			expect(ple?.perLayerProjNorm.dimensions).toHaveLength(1);
			expect(ple?.perLayerProjNorm.dimensions[0]).toBe(256); // pleDim
		});

		it("loads per-block gated-PLE tensors when arch is gemma4", () => {
			// Tasks 3.2a + 3.2b + 3.2c: AltUp / Laurel / gated-PLE per-block tensor exposure.
			//
			// The Q4KM GGUF contains gated-PLE per-block tensors (blk.L.inp_gate.weight,
			// blk.L.proj.weight, blk.L.post_norm.weight) but does NOT contain AltUp or
			// Laurel tensors. This is expected — the AltUp/Laurel tensors are absent from
			// this quantized export. The detection gate (altUpGlobal presence) correctly
			// identifies non-AltUp GGUFs without false-positives.
			//
			// Tensor names confirmed via:
			//   - GGUF probe (probe output: blk.{0..34}.inp_gate.weight etc., count=35)
			//   - llama.cpp src/llama-arch.cpp (LLM_TENSOR_PER_LAYER_INP_GATE → "blk.%d.inp_gate",
			//     LLM_TENSOR_PER_LAYER_PROJ → "blk.%d.proj",
			//     LLM_TENSOR_PER_LAYER_POST_NORM → "blk.%d.post_norm")
			const buf = readFileSync(GEMMA4);
			const parsed = ModelLoader.parseModel(buf);
			const pb = parsed.gemma3nPerBlock;

			// gemma3nPerBlock is populated (gated-PLE tensors are present)
			expect(pb).toBeDefined();

			// Gated-PLE per-block arrays — present in all Gemma 3N GGUFs, length === layerCount
			expect(pb?.pleInpGate).toHaveLength(35);
			expect(pb?.plePerBlockProj).toHaveLength(35);
			expect(pb?.plePostNorm).toHaveLength(35);

			// Verify layer 0 tensor descriptors are non-null with correct names
			expect(pb?.pleInpGate[0]).toBeDefined();
			expect(pb?.pleInpGate[0].name).toBe("blk.0.inp_gate.weight");
			expect(pb?.pleInpGate[0].dimensions).toEqual([1536, 256]); // [n_embd, pleDim]

			expect(pb?.plePerBlockProj[0]).toBeDefined();
			expect(pb?.plePerBlockProj[0].name).toBe("blk.0.proj.weight");
			expect(pb?.plePerBlockProj[0].dimensions).toEqual([256, 1536]); // [pleDim, n_embd]

			expect(pb?.plePostNorm[0]).toBeDefined();
			expect(pb?.plePostNorm[0].name).toBe("blk.0.post_norm.weight");
			expect(pb?.plePostNorm[0].dimensions).toEqual([1536]); // [n_embd]

			// Verify last layer (34) is also loaded
			expect(pb?.pleInpGate[34]).toBeDefined();
			expect(pb?.pleInpGate[34].name).toBe("blk.34.inp_gate.weight");

			// AltUp global — absent from this Q4KM GGUF (stripped export)
			// The detection gate for Gemma 3N code paths is altUpGlobal presence
			expect(parsed.altUpGlobal).toBeUndefined();

			// AltUp and Laurel per-block sub-arrays — absent from this Q4KM GGUF
			expect(pb?.altupCorrectCoef).toBeUndefined();
			expect(pb?.altupCorrectScale).toBeUndefined();
			expect(pb?.altupPredictCoef).toBeUndefined();
			expect(pb?.altupRouter).toBeUndefined();
			expect(pb?.altupRouterNorm).toBeUndefined();
			expect(pb?.laurelL).toBeUndefined();
			expect(pb?.laurelR).toBeUndefined();
			expect(pb?.laurelPostNorm).toBeUndefined();
		});

		it("returns undefined gemma3nPerBlock for non-Gemma-4 architectures", () => {
			// Non-Gemma-4 GGUFs must not populate the Gemma 3N tensor groups.
			// Use a Llama GGUF as the non-Gemma-4 control.
			const LLAMA = "smoke-test/models/llama-3.2-1b-q4f16.gguf";
			if (!existsSync(LLAMA)) return; // skip if not available
			const buf = readFileSync(LLAMA);
			const parsed = ModelLoader.parseModel(buf);
			expect(parsed.altUpGlobal).toBeUndefined();
			expect(parsed.gemma3nPerBlock).toBeUndefined();
		});
	},
);
