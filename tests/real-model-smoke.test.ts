import { expect, test } from "bun:test";
import {
	buildSmokePrompt,
	createPrefillComparisonRunner,
	createSmokeSamplerFactory,
	getSmokeChatOptions,
	shouldAutoInsertBos,
} from "../smoke-test/real-model-smoke.js";

test("qwen chatml smoke helpers disable thinking and use qwen sampling defaults", () => {
	const parsedModel = {
		hyperparams: { architecture: "qwen3" },
		tokenizerConfig: { chatTemplate: "<|im_start|>assistant" },
	};
	const samplerConfigs: Array<Record<string, unknown>> = [];
	class FakeSampler {
		constructor(config: Record<string, unknown>) {
			samplerConfigs.push(config);
		}
	}

	const chatOptions = getSmokeChatOptions(
		parsedModel,
		(template: string) =>
			template.includes("<|im_start|>") ? "chatml" : "unknown",
		parsedModel.tokenizerConfig.chatTemplate,
	);
	expect(chatOptions).toEqual({ enableThinking: false });

	const makeSmokeSampler = createSmokeSamplerFactory({
		Sampler: FakeSampler,
		parsedModel,
		detectChatTemplate: (template: string) =>
			template.includes("<|im_start|>") ? "chatml" : "unknown",
	});
	makeSmokeSampler(parsedModel.tokenizerConfig.chatTemplate, chatOptions);
	expect(samplerConfigs).toEqual([
		{
			temperature: 0.7,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
			seed: 12345,
		},
	]);
});

test("smoke prompt helpers respect BOS policy and build a chat prompt", () => {
	expect(shouldAutoInsertBos({ addBosToken: false })).toBe(false);
	expect(shouldAutoInsertBos({ addBosToken: true })).toBe(true);
	expect(shouldAutoInsertBos(undefined)).toBe(true);

	const prompt = buildSmokePrompt(
		"Tell one short joke.",
		{ enableThinking: false },
		(messages, tokenizer, chatOptions) => {
			expect(messages).toEqual([
				{ role: "user", content: "Tell one short joke." },
			]);
			expect(tokenizer).toEqual({ id: "fake-tokenizer" });
			expect(chatOptions).toEqual({ enableThinking: false });
			return [7, 8, 9];
		},
		{ id: "fake-tokenizer" },
	);

	expect(prompt).toEqual({
		mode: "chat",
		tokens: [7, 8, 9],
	});
});

test("prefill comparison helper logs batch versus sequential diagnostics", async () => {
	const logs: Array<{ cls: string; msg: string }> = [];
	const tokenizer = {
		getToken(id: number) {
			return { text: `tok${id}` };
		},
	};
	let resetCount = 0;
	const inference = {
		resetKVCache() {
			resetCount++;
		},
		async forward(tokens: Int32Array) {
			if (tokens.length > 1) {
				return new Float32Array([0.5, 0.2, 0.1]);
			}
			if (tokens[0] === 7) {
				return new Float32Array([0.45, 0.25, 0.1]);
			}
			return new Float32Array([0.5, 0.2, 0.1]);
		},
	};

	const compareBatchVsSequentialPrefill = createPrefillComparisonRunner({
		inference,
		tokenizer,
		log(cls: string, msg: string) {
			logs.push({ cls, msg });
		},
	});

	await compareBatchVsSequentialPrefill("chat", [3, 7]);

	expect(resetCount).toBe(3);
	expect(logs).toHaveLength(3);
	expect(logs[0]).toEqual({
		cls: "running",
		msg: expect.stringContaining("chat batch-vs-seq diff=2/3"),
	});
	expect(logs[1]).toEqual({
		cls: "running",
		msg: '  chat batch top5: 0:"tok0"(0.50), 1:"tok1"(0.20), 2:"tok2"(0.10)',
	});
	expect(logs[2]).toEqual({
		cls: "running",
		msg: '  chat seq   top5: 0:"tok0"(0.45), 1:"tok1"(0.25), 2:"tok2"(0.10)',
	});
});
