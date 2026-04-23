import { expect, test } from "bun:test";
import { runInteractiveChatTurn } from "../smoke-test/real-model-runtime.js";

test("runInteractiveChatTurn returns visible assistant text without surfacing smoke harness errors", async () => {
	const resetCalls: string[] = [];
	const inference = {
		resetKVCache() {
			resetCalls.push("reset");
		},
	};
	const session = {
		position: 11,
		history: [99],
		messages: [] as Array<{ role: string; content: string }>,
		prevCount: 0,
	};
	const makeSmokeSamplerCalls: Array<{
		chatTemplate: string;
		chatOptions: Record<string, unknown>;
	}> = [];
	const completionCalls: Array<{
		label: string;
		promptTokens: number[];
		maxTokens: number;
		chatOptions: Record<string, unknown>;
	}> = [];

	const { result, session: nextSession } = await runInteractiveChatTurn({
		text: " hello ",
		session,
		parsedModel: {
			tokenizerConfig: {
				chatTemplate: "<|im_start|>assistant",
			},
		},
		tokenizer: { id: "fake-tokenizer" },
		inference,
		makeSmokeSampler(chatTemplate, chatOptions) {
			makeSmokeSamplerCalls.push({ chatTemplate, chatOptions });
			return { id: "fake-sampler" };
		},
		getSmokeChatOptions() {
			return { enableThinking: false };
		},
		encodeChatPrompt(messages, _tokenizer, chatOptions) {
			expect(messages).toEqual([{ role: "user", content: "hello" }]);
			expect(chatOptions).toEqual({ enableThinking: false });
			return [3, 5, 8];
		},
		async interactiveRunCompletion(
			label,
			promptTokens,
			_sampler,
			maxTokens,
			chatOptions,
		) {
			completionCalls.push({ label, promptTokens, maxTokens, chatOptions });
			return {
				outputText: "Hello there!",
				displayOutputText: "",
				genTokens: 2,
				genTime: 50,
				totalTime: 120,
				finishReason: "eos",
			};
		},
	});

	expect(resetCalls).toEqual(["reset"]);
	expect(makeSmokeSamplerCalls).toEqual([
		{
			chatTemplate: "<|im_start|>assistant",
			chatOptions: { enableThinking: false },
		},
	]);
	expect(completionCalls).toEqual([
		{
			label: "chat-interactive",
			promptTokens: [3, 5, 8],
			maxTokens: 100,
			chatOptions: { enableThinking: false },
		},
	]);
	expect(result.fullText).toBe("Hello there!");
	expect(result.finishReason).toBe("eos");
	expect(nextSession.position).toBe(0);
	expect(nextSession.history).toEqual([]);
	expect(nextSession.prevCount).toBe(1);
	expect(nextSession.messages).toEqual([
		{ role: "user", content: "hello" },
		{ role: "assistant", content: "Hello there!" },
	]);
});
