import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
	type GenerationConfig,
	Generator,
} from "../src/inference/generation.js";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";
import { ModelInference } from "../src/inference/model-inference.js";
import { Sampler } from "../src/inference/sampler.js";
import { SpeculativeGenerator } from "../src/inference/speculative.js";
import { Tokenizer } from "../src/inference/tokenizer.js";
import { GgufParser } from "../src/models/gguf-parser.js";
import { InferenceSession } from "../src/models/inference-session.js";
import { ModelLoader } from "../src/models/model-loader.js";

const TINYLLAMA = resolve("smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf");

// Real forward passes require the WebGPU-backed WASM build, which only runs
// in a browser. In Bun we skip — the smoke harness covers this path
// end-to-end. We also skip if the local TinyLlama GGUF is missing.
const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as { gpu?: unknown }).gpu !== "undefined";
const SHOULD_SKIP = !HAS_WEBGPU || !existsSync(TINYLLAMA);

describe.skipIf(SHOULD_SKIP)("SpeculativeGenerator integration", () => {
	test("greedy spec on tinyllama (self-draft) matches greedy non-spec output", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view);

		// Two independent ModelInference instances on the same GGUF — same
		// tokenizer / vocab / EOS so vocab parity holds. This exercises the
		// spec path without needing a real second model in CI.
		const wasmA = new GgmlWasm();
		await wasmA.init({});
		const target = new ModelInference(wasmA, parsed.hyperparams);
		target.loadWeights(ggufCtx, view);
		target.initKVCache(128);

		const wasmB = new GgmlWasm();
		await wasmB.init({});
		const drafter = new ModelInference(wasmB, parsed.hyperparams);
		drafter.loadWeights(ggufCtx, view);
		drafter.initKVCache(128);

		const tokenizer = new Tokenizer(parsed.tokenizerConfig);
		const promptIds = tokenizer.encode("Hello");
		const config: GenerationConfig = {
			prompt: "Hello",
			maxTokens: 12,
			temperature: 0,
			topK: 0,
			topP: 1.0,
			repetitionPenalty: 1.0,
		};

		// Spec path.
		const specSampler = new Sampler({ temperature: 0, seed: 1 });
		const specTokens: number[] = [];
		const specGen = SpeculativeGenerator.generate({
			promptTokenIds: promptIds,
			target,
			drafter,
			tokenizer,
			sampler: specSampler,
			config,
			eosTokenId: tokenizer.eosId,
			draftLength: 4,
		});
		for await (const id of specGen) specTokens.push(id);

		// Non-spec path on a fresh target instance.
		const wasmC = new GgmlWasm();
		await wasmC.init({});
		const baseline = new ModelInference(wasmC, parsed.hyperparams);
		baseline.loadWeights(ggufCtx, view);
		baseline.initKVCache(128);
		const baselineSampler = new Sampler({ temperature: 0, seed: 1 });
		const baselineSession = new InferenceSession(
			{
				maxTokens: 12,
				temperature: 0,
				topK: 0,
				topP: 1.0,
				repetitionPenalty: 1.0,
				contextOverflowPolicy: "stop",
			},
			0,
		);
		const baselineTokens: number[] = [];
		const baselineGen = Generator.generate(
			promptIds,
			baselineSampler,
			baselineSession,
			tokenizer.eosId,
			(ids, positions) =>
				baseline.forward(new Int32Array(ids), new Int32Array(positions)),
			config,
			undefined,
			undefined,
		);
		for await (const id of baselineGen) baselineTokens.push(id);

		// Assertion: both paths emit the same token sequence under greedy.
		expect(specTokens).toEqual(baselineTokens);

		await target.dispose();
		await wasmA.shutdown();
		await drafter.dispose();
		await wasmB.shutdown();
		await baseline.dispose();
		await wasmC.shutdown();
	});

	test("KV rollback after partial accept", async () => {
		const data = readFileSync(TINYLLAMA);
		const view = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view);

		const wasmA = new GgmlWasm();
		await wasmA.init({});
		const target = new ModelInference(wasmA, parsed.hyperparams);
		target.loadWeights(ggufCtx, view);
		target.initKVCache(128);

		const wasmB = new GgmlWasm();
		await wasmB.init({});
		const drafter = new ModelInference(wasmB, parsed.hyperparams);
		drafter.loadWeights(ggufCtx, view);
		drafter.initKVCache(128);

		const tokenizer = new Tokenizer(parsed.tokenizerConfig);
		const promptIds = tokenizer.encode("Once");
		const sampler = new Sampler({ temperature: 0.7, topK: 20, seed: 42 });
		const config: GenerationConfig = {
			prompt: "Once",
			maxTokens: 8,
			temperature: 0.7,
			topK: 20,
			topP: 0.9,
			repetitionPenalty: 1.1,
		};

		const tokens: number[] = [];
		for await (const id of SpeculativeGenerator.generate({
			promptTokenIds: promptIds,
			target,
			drafter,
			tokenizer,
			sampler,
			config,
			eosTokenId: tokenizer.eosId,
			draftLength: 4,
		}))
			tokens.push(id);

		// After completion, both caches sit at promptLen + tokens.length - 1.
		// The last yielded token's KV slot is written by the next step's first
		// drafter forward, not yet — so we are short by exactly one.
		expect(target.cachedTokenCount).toBe(promptIds.length + tokens.length - 1);
		expect(drafter.cachedTokenCount).toBe(promptIds.length + tokens.length - 1);

		await target.dispose();
		await wasmA.shutdown();
		await drafter.dispose();
		await wasmB.shutdown();
	});
});
