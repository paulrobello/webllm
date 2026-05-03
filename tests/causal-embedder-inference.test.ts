import { describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import {
	CAUSAL_EMBEDDER_ARCHITECTURES,
	isCausalEmbedderArchitecture,
	type ModelHyperparams,
} from "../src/core/types.js";
import { CausalLMEmbedder } from "../src/inference/causal-embedder-inference.js";

describe("isCausalEmbedderArchitecture", () => {
	test("returns true for causal-embedder archs", () => {
		expect(isCausalEmbedderArchitecture("qwen3-embedding")).toBe(true);
	});
	test("returns false for encoder + plain causal archs", () => {
		for (const a of [
			"bert",
			"nomic-bert",
			"jina-bert-v2",
			"llama",
			"mistral",
			"qwen",
			"qwen2",
			"qwen3",
			"phi",
			"gemma",
			"mixtral",
			"deepseek",
		] as const) {
			expect(isCausalEmbedderArchitecture(a)).toBe(false);
		}
	});
	test("CAUSAL_EMBEDDER_ARCHITECTURES tuple matches helper truth-table", () => {
		expect(CAUSAL_EMBEDDER_ARCHITECTURES).toEqual(["qwen3-embedding"]);
	});
});

describe("CausalLMEmbedder construction", () => {
	function makeHp(arch: ModelHyperparams["architecture"]): ModelHyperparams {
		return {
			architecture: arch,
			contextLength: 32768,
			embeddingLength: 1024,
			headCount: 16,
			headCountKv: 8,
			layerCount: 28,
			vocabularySize: 151669,
			embeddingHeadLength: 128,
			feedForwardLength: 3072,
			ropeFreqBase: 1000000,
			ropeScale: 1,
			normEpsilon: 1e-6,
			expertCount: 0,
			expertUsedCount: 0,
			quantType: "F16",
			poolingType: "last-token",
		};
	}

	test("rejects encoder hyperparams", () => {
		expect(() => new CausalLMEmbedder({} as never, makeHp("bert"))).toThrow(
			/does not support architecture "bert"/,
		);
	});

	test("rejects plain causal-LM hyperparams", () => {
		expect(() => new CausalLMEmbedder({} as never, makeHp("llama"))).toThrow(
			/does not support architecture "llama"/,
		);
	});

	test("accepts qwen3-embedding", () => {
		expect(
			() => new CausalLMEmbedder({} as never, makeHp("qwen3-embedding")),
		).not.toThrow();
	});
});

const FIXTURE_PATH =
	"eval/reports/bucket-c-probe-2026-04-29/cache/qwen3-embedding-0.6b.gguf";
const HAS_FIXTURE = existsSync(FIXTURE_PATH);

describe("CausalLMEmbedder GGUF fixture", () => {
	test.skipIf(!HAS_FIXTURE)(
		"Qwen3-Embedding-0.6B GGUF fixture is present in the bucket-c probe cache",
		() => {
			expect(HAS_FIXTURE).toBe(true);
		},
	);
});
