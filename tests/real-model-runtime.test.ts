import { expect, test } from "bun:test";
import { runInteractiveChatTurn } from "../smoke-test/real-model-runtime.js";

test("runInteractiveChatTurn routes through the engine completion runner with seeded sampler config", async () => {
	const session = {
		messages: [] as Array<{ role: string; content: string }>,
	};
	const completionCalls: Array<{
		label: string;
		messages: Array<{ role: string; content: string }>;
		samplingConfig: Record<string, unknown>;
		maxTokens: number;
		chatOptions: Record<string, unknown>;
	}> = [];

	const { result, session: nextSession } = await runInteractiveChatTurn({
		text: " hello ",
		session,
		parsedModel: {
			hyperparams: { architecture: "qwen3" },
			tokenizerConfig: { chatTemplate: "<|im_start|>assistant" },
		},
		detectChatTemplate: (template: string) =>
			template.includes("<|im_start|>") ? "chatml" : "unknown",
		getSmokeChatOptions: () => ({ enableThinking: false }),
		getSmokeSamplingConfig: (
			_parsed: unknown,
			_detect: unknown,
			_chatTemplate: unknown,
			chatOptions: { enableThinking?: boolean },
		) => ({
			temperature: chatOptions.enableThinking === false ? 0.7 : 0.6,
			topK: 20,
			topP: 0.8,
			repetitionPenalty: 1.1,
			seed: 12345,
		}),
		samplingOverrides: { topK: 32 },
		async interactiveRunCompletion({
			label,
			messages,
			samplingConfig,
			maxTokens,
			chatOptions,
		}) {
			completionCalls.push({
				label,
				messages: [...messages],
				samplingConfig,
				maxTokens,
				chatOptions,
			});
			return {
				outputText: "Hello there!",
				rawOutputText: "Hello there!",
				displayOutputText: "Hello there!",
				genTokens: 2,
				genTime: 50,
				totalTime: 120,
				prefillMs: 70,
				finishReason: "eos",
			};
		},
	});

	expect(completionCalls).toEqual([
		{
			label: "chat-interactive",
			messages: [{ role: "user", content: "hello" }],
			samplingConfig: {
				temperature: 0.7,
				topK: 32,
				topP: 0.8,
				repetitionPenalty: 1.1,
				seed: 12345,
			},
			maxTokens: 100,
			chatOptions: { enableThinking: false },
		},
	]);
	expect(result.fullText).toBe("Hello there!");
	expect(result.finishReason).toBe("eos");
	expect(nextSession.messages).toEqual([
		{ role: "user", content: "hello" },
		{ role: "assistant", content: "Hello there!" },
	]);
});
