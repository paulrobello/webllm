/**
 * Serialize/reconstruct WebLLMError subclasses across the postMessage
 * boundary. structuredClone drops class identity, so the worker side
 * serializes to a flat shape with a `code` field and the proxy side
 * rebuilds the matching subclass via a single switch.
 *
 * Mirror-drift sentinel: tests/webllm-error-codec.test.ts parametrizes
 * over a factory table that mirrors errors.ts. Adding a new WebLLMError
 * subclass requires updating both this codec and the test factory.
 */

import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
	WebLLMError,
	type WebLLMErrorCode,
} from "./errors.js";
import type { SerializedError } from "./worker-bridge.js";

export function serializeError(e: unknown): SerializedError {
	if (e instanceof WebLLMError) {
		const out: SerializedError = {
			code: e.code,
			message: e.message,
			stack: e.stack,
		};
		if (e instanceof ModelNotFoundError) out.modelId = e.modelId;
		else if (e instanceof ModelNotLoadedError) out.modelId = e.modelId;
		else if (e instanceof InferenceEngineMissingError) out.modelId = e.modelId;
		else if (e instanceof EncoderRequiredError) {
			out.modelId = e.modelId;
			out.architecture = e.architecture;
		} else if (e instanceof ConversationNotFoundError)
			out.conversationId = e.conversationId;
		else if (e instanceof ConversationNotPopulatedError)
			out.conversationId = e.conversationId;
		else if (e instanceof ConversationPoolFullError)
			out.liveConversationIds = [...e.liveConversationIds];
		else if (e instanceof ConversationContextOverflowError) {
			out.conversationId = e.conversationId;
			out.requestedTokens = e.requestedTokens;
			out.maxContextTokens = e.maxContextTokens;
		} else if (e instanceof ConversationBusyError)
			out.conversationId = e.conversationId;
		// SpeculativeDecodingReservedError carries no extra fields.
		return out;
	}
	if (e instanceof Error) {
		return { code: "GENERIC", message: e.message, stack: e.stack };
	}
	return { code: "GENERIC", message: `non-Error thrown: ${String(e)}` };
}

export function reconstructError(s: SerializedError): WebLLMError | Error {
	switch (s.code) {
		case "MODEL_NOT_FOUND":
			return attachStack(new ModelNotFoundError(s.modelId ?? ""), s);
		case "MODEL_NOT_LOADED":
			return attachStack(new ModelNotLoadedError(s.modelId ?? ""), s);
		case "INFERENCE_ENGINE_MISSING":
			return attachStack(new InferenceEngineMissingError(s.modelId ?? ""), s);
		case "ENCODER_REQUIRED":
			return attachStack(
				new EncoderRequiredError(s.modelId ?? "", s.architecture ?? ""),
				s,
				/* preserveMessage */ true,
			);
		case "SPECULATIVE_DECODING_RESERVED":
			return attachStack(new SpeculativeDecodingReservedError(), s);
		case "CONVERSATION_NOT_FOUND":
			return attachStack(
				new ConversationNotFoundError(s.conversationId ?? ""),
				s,
			);
		case "CONVERSATION_NOT_POPULATED":
			return attachStack(
				new ConversationNotPopulatedError(s.conversationId ?? ""),
				s,
			);
		case "CONVERSATION_POOL_FULL":
			return attachStack(
				new ConversationPoolFullError(s.liveConversationIds ?? []),
				s,
			);
		case "CONVERSATION_CONTEXT_OVERFLOW":
			return attachStack(
				new ConversationContextOverflowError(
					s.conversationId ?? "",
					s.requestedTokens ?? 0,
					s.maxContextTokens ?? 0,
				),
				s,
			);
		case "CONVERSATION_BUSY":
			return attachStack(new ConversationBusyError(s.conversationId ?? ""), s);
		case "DISPOSED": {
			const e = new WebLLMError(
				s.message,
				"DISPOSED" as unknown as WebLLMErrorCode,
			);
			if (s.stack) e.stack = s.stack;
			return e;
		}
		default: {
			const e = new Error(s.message);
			if (s.stack) e.stack = s.stack;
			return e;
		}
	}
}

function attachStack(
	e: WebLLMError,
	s: SerializedError,
	preserveMessage = false,
): WebLLMError {
	if (preserveMessage) {
		(e as { message: string }).message = s.message;
	}
	if (s.stack) e.stack = s.stack;
	return e;
}
