/**
 * Error taxonomy for the public WebLLM API.
 *
 * All errors thrown by public engine/character/generation surfaces extend
 * {@link WebLLMError}, which carries a stable `code` string suitable for
 * programmatic dispatch. Existing `instanceof Error` and try/catch flows
 * continue to work unchanged — these are additive subclasses.
 */

/** Stable code values exposed on {@link WebLLMError.code}. */
export type WebLLMErrorCode =
	| "MODEL_NOT_FOUND"
	| "MODEL_NOT_LOADED"
	| "INFERENCE_ENGINE_MISSING"
	| "ENCODER_REQUIRED"
	| "SPECULATIVE_DECODING_RESERVED"
	| "CONVERSATION_NOT_FOUND"
	| "CONVERSATION_NOT_POPULATED"
	| "CONVERSATION_POOL_FULL"
	| "CONVERSATION_CONTEXT_OVERFLOW"
	| "CONVERSATION_BUSY"
	| "INCOMPATIBLE_CONVERSATION"
	| "CORRUPT_BLOB"
	| "PERSISTENCE_UNAVAILABLE"
	| "PERSISTENCE_QUOTA"
	| "PERSISTENCE_IO";

/** Base class for all errors thrown by the public WebLLM API. */
export class WebLLMError extends Error {
	readonly code: WebLLMErrorCode;
	constructor(
		message: string,
		code: WebLLMErrorCode,
		options?: ErrorOptions,
	) {
		super(message, options);
		this.name = "WebLLMError";
		this.code = code;
	}
}

/** Thrown when an operation references a model id that the engine has no record of. */
export class ModelNotFoundError extends WebLLMError {
	readonly modelId: string;
	constructor(modelId: string) {
		super(`Model "${modelId}" not found`, "MODEL_NOT_FOUND");
		this.name = "ModelNotFoundError";
		this.modelId = modelId;
	}
}

/**
 * Thrown when a model handle exists but its weights / tokenizer have not
 * finished loading — typically because the consumer obtained the handle from
 * a low-level path that bypassed `loadModelFromBuffer`.
 */
export class ModelNotLoadedError extends WebLLMError {
	readonly modelId: string;
	constructor(modelId: string) {
		super(`Model "${modelId}" not fully loaded`, "MODEL_NOT_LOADED");
		this.name = "ModelNotLoadedError";
		this.modelId = modelId;
	}
}

/**
 * Thrown when a chat / generation call is routed to a model whose inference
 * pipeline is unavailable (e.g. encoder model fed to `chatCompletion`).
 */
export class InferenceEngineMissingError extends WebLLMError {
	readonly modelId: string;
	constructor(modelId: string) {
		super(
			`No inference engine for model "${modelId}"`,
			"INFERENCE_ENGINE_MISSING",
		);
		this.name = "InferenceEngineMissingError";
		this.modelId = modelId;
	}
}

/**
 * Thrown when `engine.embed()` is called against a model whose architecture
 * is not a bidirectional encoder (e.g. a causal-LM passed by mistake).
 */
export class EncoderRequiredError extends WebLLMError {
	readonly modelId: string;
	readonly architecture: string;
	constructor(modelId: string, architecture: string, hint?: string) {
		const message = hint
			? `embed() requires a bidirectional encoder model; "${modelId}" is architecture "${architecture}". ${hint}`
			: `embed() requires a bidirectional encoder model; "${modelId}" is architecture "${architecture}"`;
		super(message, "ENCODER_REQUIRED");
		this.name = "EncoderRequiredError";
		this.modelId = modelId;
		this.architecture = architecture;
	}
}

/**
 * Thrown when a consumer passes `drafter` to `chatCompletion` /
 * `generateStream`. Speculative decoding is reserved in v1 — see TODO.md §19
 * and `docs/superpowers/specs/2026-04-26-speculative-decoding-design.md` for
 * the measurement (0.20× vs baseline at K=4 on qwen3-8b/qwen3-0.6b).
 */
export class SpeculativeDecodingReservedError extends WebLLMError {
	constructor() {
		super(
			"Speculative decoding is reserved in v1 (measured 0.20× vs baseline at K=4 on qwen3-8b/qwen3-0.6b on 2026-04-26). See TODO.md §19 and docs/superpowers/specs/2026-04-26-speculative-decoding-design.md.",
			"SPECULATIVE_DECODING_RESERVED",
		);
		this.name = "SpeculativeDecodingReservedError";
	}
}

/**
 * Thrown when a caller references a conversation id that the engine has no
 * record of — disposed, never created, or invalidated by a model unload.
 */
export class ConversationNotFoundError extends WebLLMError {
	readonly conversationId: string;
	constructor(conversationId: string) {
		super(
			`Conversation "${conversationId}" not found (disposed, never created, or model unloaded)`,
			"CONVERSATION_NOT_FOUND",
		);
		this.name = "ConversationNotFoundError";
		this.conversationId = conversationId;
	}
}

/**
 * Thrown when `engine.forkConversation(srcConv)` is called against a
 * conversation that has no KV snapshot yet (i.e., never had a successful
 * `chatCompletion` call). A fork would have nothing to inherit.
 */
export class ConversationNotPopulatedError extends WebLLMError {
	readonly conversationId: string;
	constructor(conversationId: string) {
		super(
			`Conversation "${conversationId}" has no KV snapshot yet — drive at least one chatCompletion call before forking`,
			"CONVERSATION_NOT_POPULATED",
		);
		this.name = "ConversationNotPopulatedError";
		this.conversationId = conversationId;
	}
}

/**
 * Thrown when `engine.createConversation()` is called while the per-engine
 * conversation pool is at its configured cap. Caller must dispose an existing
 * conversation before creating another.
 */
export class ConversationPoolFullError extends WebLLMError {
	readonly liveConversationIds: readonly string[];
	constructor(liveConversationIds: readonly string[]) {
		super(
			`Conversation pool full (${liveConversationIds.length} live: ${liveConversationIds.join(", ")}). Dispose one before creating another.`,
			"CONVERSATION_POOL_FULL",
		);
		this.name = "ConversationPoolFullError";
		this.liveConversationIds = liveConversationIds;
	}
}

/**
 * Thrown when appending a message to a conversation would exceed the model's
 * configured context window.
 */
export class ConversationContextOverflowError extends WebLLMError {
	readonly conversationId: string;
	readonly requestedTokens: number;
	readonly maxContextTokens: number;
	constructor(
		conversationId: string,
		requestedTokens: number,
		maxContextTokens: number,
	) {
		super(
			`Conversation "${conversationId}" would exceed context: ${requestedTokens} tokens > ${maxContextTokens} max`,
			"CONVERSATION_CONTEXT_OVERFLOW",
		);
		this.name = "ConversationContextOverflowError";
		this.conversationId = conversationId;
		this.requestedTokens = requestedTokens;
		this.maxContextTokens = maxContextTokens;
	}
}

/**
 * Thrown when a second `chatCompletion` / `generateStream` call lands on a
 * conversation that already has an in-flight call. Conversations are
 * single-writer.
 */
export class ConversationBusyError extends WebLLMError {
	readonly conversationId: string;
	constructor(conversationId: string) {
		super(
			`Conversation "${conversationId}" has an in-flight chatCompletion call`,
			"CONVERSATION_BUSY",
		);
		this.name = "ConversationBusyError";
		this.conversationId = conversationId;
	}
}

export type IncompatibleConversationReason =
	| "schema-mismatch"
	| "fingerprint-mismatch"
	| "tokenizer-mismatch";

export class IncompatibleConversationError extends WebLLMError {
	readonly reason: IncompatibleConversationReason;
	readonly details: Record<string, unknown>;
	constructor(
		reason: IncompatibleConversationReason,
		details: Record<string, unknown>,
	) {
		super(
			`incompatible persisted conversation: ${reason}`,
			"INCOMPATIBLE_CONVERSATION",
		);
		this.name = "IncompatibleConversationError";
		this.reason = reason;
		this.details = details;
	}
}

export type CorruptBlobReason =
	| "bad-magic"
	| "bad-header-len"
	| "bad-header-json"
	| "byte-size-mismatch";

export class CorruptBlobError extends WebLLMError {
	readonly reason: CorruptBlobReason;
	readonly details: Record<string, unknown>;
	constructor(reason: CorruptBlobReason, details: Record<string, unknown>) {
		super(`corrupt persisted-conversation blob: ${reason}`, "CORRUPT_BLOB");
		this.name = "CorruptBlobError";
		this.reason = reason;
		this.details = details;
	}
}

export type PersistenceUnavailableReason =
	| "indexeddb-missing"
	| "indexeddb-blocked"
	| "open-failed";

export class PersistenceUnavailableError extends WebLLMError {
	readonly reason: PersistenceUnavailableReason;
	constructor(reason: PersistenceUnavailableReason, cause?: unknown) {
		super(
			`persistence unavailable: ${reason}`,
			"PERSISTENCE_UNAVAILABLE",
			cause !== undefined ? { cause } : undefined,
		);
		this.name = "PersistenceUnavailableError";
		this.reason = reason;
	}
}

export class PersistenceQuotaError extends WebLLMError {
	readonly attemptedBytes: number;
	constructor(attemptedBytes: number) {
		super(
			`persistence quota exceeded (attempted ${attemptedBytes} bytes)`,
			"PERSISTENCE_QUOTA",
		);
		this.name = "PersistenceQuotaError";
		this.attemptedBytes = attemptedBytes;
	}
}

export type PersistenceIOReason = "io-failure" | "transaction-aborted";

export class PersistenceIOError extends WebLLMError {
	readonly reason: PersistenceIOReason;
	constructor(reason: PersistenceIOReason, cause: unknown) {
		super(`persistence IO error: ${reason}`, "PERSISTENCE_IO", { cause });
		this.name = "PersistenceIOError";
		this.reason = reason;
	}
}
