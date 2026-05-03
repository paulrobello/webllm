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

import type {
	CorruptBlobReason,
	IncompatibleConversationReason,
	PersistenceIOReason,
	PersistenceUnavailableReason,
} from "./errors.js";
import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
	CorruptBlobError,
	EncoderRequiredError,
	IncompatibleConversationError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	PersistenceIOError,
	PersistenceQuotaError,
	PersistenceUnavailableError,
	SpeculativeDecodingReservedError,
	WebLLMError,
	type WebLLMErrorCode,
} from "./errors.js";
import type { SerializedError } from "./worker-bridge.js";

function serializeCause(
	c: unknown,
): { message: string; name?: string } | undefined {
	if (c instanceof Error) return { message: c.message, name: c.name };
	if (typeof c === "string") return { message: c };
	return undefined;
}

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
		else if (e instanceof IncompatibleConversationError) {
			out.reason = e.reason;
			out.details = e.details;
		} else if (e instanceof CorruptBlobError) {
			out.reason = e.reason;
			out.details = e.details;
		} else if (e instanceof PersistenceUnavailableError) {
			out.reason = e.reason;
			out.cause = serializeCause(e.cause);
		} else if (e instanceof PersistenceQuotaError) {
			out.attemptedBytes = e.attemptedBytes;
		} else if (e instanceof PersistenceIOError) {
			out.reason = e.reason;
			out.cause = serializeCause(e.cause);
		}
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
		case "INCOMPATIBLE_CONVERSATION":
			return attachStack(
				new IncompatibleConversationError(
					(s.reason ?? "schema-mismatch") as IncompatibleConversationReason,
					s.details ?? {},
				),
				s,
			);
		case "CORRUPT_BLOB":
			return attachStack(
				new CorruptBlobError(
					(s.reason ?? "bad-magic") as CorruptBlobReason,
					s.details ?? {},
				),
				s,
			);
		case "PERSISTENCE_UNAVAILABLE":
			return attachStack(
				new PersistenceUnavailableError(
					(s.reason ?? "indexeddb-missing") as PersistenceUnavailableReason,
					s.cause,
				),
				s,
			);
		case "PERSISTENCE_QUOTA":
			return attachStack(new PersistenceQuotaError(s.attemptedBytes ?? 0), s);
		case "PERSISTENCE_IO":
			return attachStack(
				new PersistenceIOError(
					(s.reason ?? "io-failure") as PersistenceIOReason,
					s.cause,
				),
				s,
			);
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
