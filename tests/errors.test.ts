import { describe, expect, test } from "bun:test";
import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationPoolFullError,
	WebLLMError,
} from "../src/core/errors.js";

describe("conversation errors", () => {
	test("ConversationNotFoundError carries id + code", () => {
		const e = new ConversationNotFoundError("conv_3");
		expect(e).toBeInstanceOf(WebLLMError);
		expect(e.code).toBe("CONVERSATION_NOT_FOUND");
		expect(e.conversationId).toBe("conv_3");
		expect(e.name).toBe("ConversationNotFoundError");
	});

	test("ConversationPoolFullError carries live ids", () => {
		const e = new ConversationPoolFullError(["conv_1", "conv_2"]);
		expect(e.code).toBe("CONVERSATION_POOL_FULL");
		expect(e.liveConversationIds).toEqual(["conv_1", "conv_2"]);
	});

	test("ConversationContextOverflowError carries token counts", () => {
		const e = new ConversationContextOverflowError("conv_1", 4100, 4096);
		expect(e.code).toBe("CONVERSATION_CONTEXT_OVERFLOW");
		expect(e.requestedTokens).toBe(4100);
		expect(e.maxContextTokens).toBe(4096);
	});

	test("ConversationBusyError carries id", () => {
		const e = new ConversationBusyError("conv_2");
		expect(e.code).toBe("CONVERSATION_BUSY");
		expect(e.conversationId).toBe("conv_2");
	});
});
