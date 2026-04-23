export async function runInteractiveChatTurn({
	text,
	session,
	parsedModel,
	tokenizer,
	inference,
	interactiveRunCompletion,
	makeSmokeSampler,
	getSmokeChatOptions,
	encodeChatPrompt,
}) {
	const trimmedText = text.trim();
	if (!trimmedText) {
		throw new Error("Interactive chat requires non-empty user text");
	}
	if (typeof interactiveRunCompletion !== "function") {
		throw new Error("interactive completion helper not ready");
	}

	const nextSession = session ?? {
		position: 0,
		history: [],
		messages: [],
		prevCount: 0,
	};
	const chatTemplate = parsedModel?.tokenizerConfig?.chatTemplate;
	const chatOptions = getSmokeChatOptions(parsedModel, chatTemplate);
	const sampler = makeSmokeSampler(chatTemplate, chatOptions);

	nextSession.messages.push({ role: "user", content: trimmedText });
	const promptTokens = encodeChatPrompt(
		nextSession.messages,
		tokenizer,
		chatOptions,
	);

	inference.resetKVCache();
	nextSession.position = 0;
	nextSession.history = [];
	nextSession.prevCount = nextSession.messages.length;

	const result = await interactiveRunCompletion(
		"chat-interactive",
		promptTokens,
		sampler,
		100,
		chatOptions,
	);
	const fullText = result.displayOutputText || result.outputText;
	nextSession.messages.push({ role: "assistant", content: fullText });

	return {
		session: nextSession,
		result: {
			...result,
			fullText,
		},
	};
}
