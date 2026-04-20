import { describe, expect, test } from "bun:test";
import {
	type GenerationConfig,
	Generator,
} from "../src/inference/generation.js";
import { Sampler } from "../src/inference/sampler.js";
import {
	InferenceSession,
	type InferenceSessionConfig,
} from "../src/models/inference-session.js";

function mockForwardPass(
	_tokenIds: number[],
	_positions: number[],
): Promise<Float32Array> {
	const vocabSize = 10;
	const logits = new Float32Array(vocabSize);
	logits[3] = 10.0; // token 3 has highest logit
	logits[2] = -100; // EOS very low
	return Promise.resolve(logits);
}

const BASE_SESSION_CONFIG: InferenceSessionConfig = {
	maxTokens: 100,
	temperature: 0,
	topK: 40,
	topP: 1,
	repetitionPenalty: 1,
	contextOverflowPolicy: "stop",
};

describe("Generator", () => {
	test("yields tokens from generation", async () => {
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 5,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBeGreaterThan(0);
		expect(tokens.every((t) => t === 3)).toBe(true);
	});

	test("stops on EOS token", async () => {
		async function eosForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			logits[2] = 100.0;
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			eosForward,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens).toEqual([2]);
	});

	test("stops on maxTokens", async () => {
		const sampler = new Sampler({ temperature: 0 });
		// Session maxTokens must accommodate prompt (1) + generated (3) = 4
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 10 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 3,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		)) {
			tokens.push(token);
		}
		expect(tokens.length).toBe(3);
	});

	test("returns generation stats", async () => {
		const sampler = new Sampler({ temperature: 0 });
		// Session maxTokens must accommodate prompt (1) + generated (3) = 4
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 10 },
			0,
		);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 3,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
		};

		let result: Awaited<ReturnType<typeof Generator.generate>["return"]>;
		const gen = Generator.generate(
			[1],
			sampler,
			session,
			2,
			mockForwardPass,
			config,
		);
		while (true) {
			const { value, done } = await gen.next();
			if (done) {
				result = value;
				break;
			}
		}
		expect(result).toBeDefined();
		expect(result?.tokenCount).toBe(3);
		expect(result?.tokensPerSecond).toBeGreaterThan(0);
		expect(result?.timeToFirstTokenMs).toBeGreaterThanOrEqual(0);
	});

	test("stops on custom stop tokens", async () => {
		let callCount = 0;
		async function stopTokenForward(): Promise<Float32Array> {
			const logits = new Float32Array(10);
			callCount++;
			// First call yields token 5, second call yields token 7 (stop token)
			if (callCount === 1) {
				logits[5] = 100.0;
			} else {
				logits[7] = 100.0;
			}
			return logits;
		}
		const sampler = new Sampler({ temperature: 0 });
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		const config: GenerationConfig = {
			prompt: "test",
			maxTokens: 100,
			temperature: 0,
			topK: 40,
			topP: 1,
			repetitionPenalty: 1,
			stopTokens: [7],
		};

		const tokens: number[] = [];
		for await (const token of Generator.generate(
			[1],
			sampler,
			session,
			99,
			stopTokenForward,
			config,
		)) {
			tokens.push(token);
		}
		// Should stop when token 7 is sampled (the stop token)
		expect(tokens).toContain(5);
		expect(tokens).not.toContain(7);
	});
});
