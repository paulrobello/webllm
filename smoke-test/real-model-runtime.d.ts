// Ambient declaration for the smoke-test runtime helper.
export function runInteractiveChatTurn(opts: unknown): Promise<{
	result: {
		fullText: string;
		finishReason: string;
		[key: string]: unknown;
	};
	session: { messages: Array<{ role: string; content: string }> };
}>;
