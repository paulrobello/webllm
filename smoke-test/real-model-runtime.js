export async function runInteractiveChatTurn({
	text,
	session,
	parsedModel,
	detectChatTemplate,
	interactiveRunCompletion,
	getSmokeChatOptions,
	getSmokeSamplingConfig,
	samplingOverrides = {},
}) {
	const trimmedText = text.trim();
	if (!trimmedText) {
		throw new Error("Interactive chat requires non-empty user text");
	}
	if (typeof interactiveRunCompletion !== "function") {
		throw new Error("interactive completion helper not ready");
	}

	const nextSession = session ?? { messages: [] };
	const chatTemplate = parsedModel?.tokenizerConfig?.chatTemplate;
	const chatOptions = getSmokeChatOptions(parsedModel, chatTemplate);
	const samplingConfig = {
		...getSmokeSamplingConfig(
			parsedModel,
			detectChatTemplate,
			chatTemplate,
			chatOptions,
		),
		...samplingOverrides,
	};

	nextSession.messages.push({ role: "user", content: trimmedText });
	const thinkingOn = chatOptions.enableThinking !== false;
	const maxTokens = thinkingOn ? 1024 : 100;
	const result = await interactiveRunCompletion({
		label: "chat-interactive",
		messages: nextSession.messages,
		samplingConfig,
		maxTokens,
		chatOptions,
	});
	const fullText = result.displayOutputText || result.outputText;
	const rawText = result.rawOutputText || fullText;
	nextSession.messages.push({ role: "assistant", content: fullText });

	return {
		session: nextSession,
		result: {
			...result,
			fullText,
			rawText,
		},
	};
}
