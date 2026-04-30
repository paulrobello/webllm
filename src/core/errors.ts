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
	| "SPECULATIVE_DECODING_RESERVED";

/** Base class for all errors thrown by the public WebLLM API. */
export class WebLLMError extends Error {
	readonly code: WebLLMErrorCode;
	constructor(message: string, code: WebLLMErrorCode) {
		super(message);
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
