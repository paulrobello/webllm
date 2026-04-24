import { describe, expect, test } from "bun:test";
import type { ModelHyperparams } from "../src/core/types.js";
import { EncoderInference } from "../src/inference/encoder-inference.js";

describe("EncoderInference construction", () => {
	test("rejects non-bert hyperparams", () => {
		const hp: ModelHyperparams = {
			architecture: "llama",
			contextLength: 512,
			embeddingLength: 384,
			headCount: 12,
			headCountKv: 12,
			layerCount: 12,
			vocabularySize: 30522,
			embeddingHeadLength: 32,
			feedForwardLength: 1536,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
		};
		expect(() => new EncoderInference({} as never, hp)).toThrow(
			/requires architecture "bert"/,
		);
	});
});
