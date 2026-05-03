import { describe, expect, test } from "bun:test";
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
} from "../src/core/errors.js";
import {
	reconstructError,
	serializeError,
} from "../src/core/webllm-error-codec.js";

// Factory table that mirrors errors.ts. Adding a new WebLLMError subclass
// without updating both this table and the codec switch fails the
// "round-trips every code" test below.
const FACTORIES: Record<WebLLMErrorCode, () => WebLLMError> = {
	MODEL_NOT_FOUND: () => new ModelNotFoundError("m1"),
	MODEL_NOT_LOADED: () => new ModelNotLoadedError("m1"),
	INFERENCE_ENGINE_MISSING: () => new InferenceEngineMissingError("m1"),
	ENCODER_REQUIRED: () =>
		new EncoderRequiredError("m1", "qwen3", "use chatCompletion"),
	SPECULATIVE_DECODING_RESERVED: () => new SpeculativeDecodingReservedError(),
	CONVERSATION_NOT_FOUND: () => new ConversationNotFoundError("c1"),
	CONVERSATION_NOT_POPULATED: () => new ConversationNotPopulatedError("c1"),
	CONVERSATION_POOL_FULL: () => new ConversationPoolFullError(["c1", "c2"]),
	CONVERSATION_CONTEXT_OVERFLOW: () =>
		new ConversationContextOverflowError("c1", 5000, 4096),
	CONVERSATION_BUSY: () => new ConversationBusyError("c1"),
	INCOMPATIBLE_CONVERSATION: () =>
		new IncompatibleConversationError("schema-mismatch", { got: 99, want: 1 }),
	CORRUPT_BLOB: () =>
		new CorruptBlobError("bad-magic", { firstFour: [0, 0, 0, 0] }),
	PERSISTENCE_UNAVAILABLE: () =>
		new PersistenceUnavailableError("indexeddb-missing"),
	PERSISTENCE_QUOTA: () => new PersistenceQuotaError(123_456_789),
	PERSISTENCE_IO: () =>
		new PersistenceIOError("io-failure", new Error("disk full")),
};

describe("webllm-error-codec — mirror-drift sentinel", () => {
	test("every WebLLMErrorCode has a factory entry", () => {
		const codes: WebLLMErrorCode[] = [
			"MODEL_NOT_FOUND",
			"MODEL_NOT_LOADED",
			"INFERENCE_ENGINE_MISSING",
			"ENCODER_REQUIRED",
			"SPECULATIVE_DECODING_RESERVED",
			"CONVERSATION_NOT_FOUND",
			"CONVERSATION_NOT_POPULATED",
			"CONVERSATION_POOL_FULL",
			"CONVERSATION_CONTEXT_OVERFLOW",
			"CONVERSATION_BUSY",
			"INCOMPATIBLE_CONVERSATION",
			"CORRUPT_BLOB",
			"PERSISTENCE_UNAVAILABLE",
			"PERSISTENCE_QUOTA",
			"PERSISTENCE_IO",
		];
		for (const c of codes) {
			expect(FACTORIES[c]).toBeDefined();
		}
	});

	for (const [code, factory] of Object.entries(FACTORIES)) {
		test(`round-trip preserves instanceof, code, message for ${code}`, () => {
			const original = factory();
			const wire = JSON.parse(JSON.stringify(serializeError(original)));
			const rebuilt = reconstructError(wire);
			expect(rebuilt).toBeInstanceOf(WebLLMError);
			expect(rebuilt).toBeInstanceOf(original.constructor);
			expect((rebuilt as WebLLMError).code).toBe(original.code);
			expect(rebuilt.message).toBe(original.message);
		});
	}

	test("ModelNotFoundError preserves modelId field", () => {
		const e = new ModelNotFoundError("qwen3-8b-iq3m");
		const r = reconstructError(
			JSON.parse(JSON.stringify(serializeError(e))),
		) as ModelNotFoundError;
		expect(r.modelId).toBe("qwen3-8b-iq3m");
	});

	test("ConversationContextOverflowError preserves all numeric fields", () => {
		const e = new ConversationContextOverflowError("c-x", 5123, 4096);
		const r = reconstructError(
			JSON.parse(JSON.stringify(serializeError(e))),
		) as ConversationContextOverflowError;
		expect(r.conversationId).toBe("c-x");
		expect(r.requestedTokens).toBe(5123);
		expect(r.maxContextTokens).toBe(4096);
	});

	test("ConversationPoolFullError preserves liveConversationIds array", () => {
		const e = new ConversationPoolFullError(["c1", "c2", "c3"]);
		const r = reconstructError(
			JSON.parse(JSON.stringify(serializeError(e))),
		) as ConversationPoolFullError;
		expect([...r.liveConversationIds]).toEqual(["c1", "c2", "c3"]);
	});

	test("EncoderRequiredError preserves architecture field", () => {
		const e = new EncoderRequiredError("m1", "qwen3", "hint");
		const r = reconstructError(
			JSON.parse(JSON.stringify(serializeError(e))),
		) as EncoderRequiredError;
		expect(r.architecture).toBe("qwen3");
	});

	test("IncompatibleConversationError preserves reason + details", () => {
		const err = new IncompatibleConversationError("fingerprint-mismatch", {
			field: "nLayer",
			got: 28,
			want: 32,
		});
		const round = reconstructError(serializeError(err));
		expect(round).toBeInstanceOf(IncompatibleConversationError);
		const ic = round as IncompatibleConversationError;
		expect(ic.reason).toBe("fingerprint-mismatch");
		expect(ic.details).toEqual({ field: "nLayer", got: 28, want: 32 });
	});

	test("CorruptBlobError preserves reason + details", () => {
		const err = new CorruptBlobError("byte-size-mismatch", {
			got: 99,
			want: 100,
		});
		const round = reconstructError(serializeError(err));
		expect(round).toBeInstanceOf(CorruptBlobError);
		const cb = round as CorruptBlobError;
		expect(cb.reason).toBe("byte-size-mismatch");
		expect(cb.details).toEqual({ got: 99, want: 100 });
	});

	test("PersistenceQuotaError preserves attemptedBytes", () => {
		const err = new PersistenceQuotaError(987_654_321);
		const round = reconstructError(serializeError(err));
		expect(round).toBeInstanceOf(PersistenceQuotaError);
		expect((round as PersistenceQuotaError).attemptedBytes).toBe(987_654_321);
	});

	test("PersistenceUnavailableError preserves reason", () => {
		const err = new PersistenceUnavailableError("indexeddb-blocked");
		const round = reconstructError(serializeError(err));
		expect(round).toBeInstanceOf(PersistenceUnavailableError);
		expect((round as PersistenceUnavailableError).reason).toBe(
			"indexeddb-blocked",
		);
	});

	test("PersistenceIOError preserves reason + cause message", () => {
		const err = new PersistenceIOError(
			"transaction-aborted",
			new Error("abort"),
		);
		const round = reconstructError(serializeError(err));
		expect(round).toBeInstanceOf(PersistenceIOError);
		const io = round as PersistenceIOError;
		expect(io.reason).toBe("transaction-aborted");
		// After codec round-trip, cause is the structured-clone shape {message, name?},
		// not the original Error instance.
		expect((io.cause as { message: string }).message).toBe("abort");
	});

	test("non-WebLLMError throws round-trip as plain Error with GENERIC code", () => {
		const e = new RangeError("boom");
		const wire = JSON.parse(JSON.stringify(serializeError(e)));
		expect(wire.code).toBe("GENERIC");
		const r = reconstructError(wire);
		expect(r).not.toBeInstanceOf(WebLLMError);
		expect(r.message).toBe("boom");
	});

	test("DISPOSED code reconstructs as WebLLMError with DISPOSED code (used for crash/dispose paths)", () => {
		const wire = { code: "DISPOSED" as const, message: "engine disposed" };
		const r = reconstructError(wire) as WebLLMError;
		expect(r).toBeInstanceOf(WebLLMError);
		expect(r.code).toBe("DISPOSED" as unknown as WebLLMErrorCode);
		expect(r.message).toBe("engine disposed");
	});

	test("non-Error thrown values serialize with GENERIC code and string-coerced message", () => {
		const wire = serializeError("string thrown");
		expect(wire.code).toBe("GENERIC");
		expect(wire.message).toContain("string thrown");
	});
});
