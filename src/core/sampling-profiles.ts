/**
 * Sampling-parameter profiles surfaced to consumers via `CompletionConfig.sampling`.
 *
 * `QWEN_THINKING_DEFAULTS` matches Qwen3's recommended parameters when the
 * model is invoked in thinking mode (default for ChatML-templated Qwen3
 * variants). `QWEN_NON_THINKING_DEFAULTS` is the official non-thinking
 * profile applied when `enableThinking: false` is set.
 *
 * Engine auto-applies these when `architecture` starts with `"qwen"` and the
 * chat template is ChatML and `sampling` is `"auto"` (default). Consumers
 * can also force a profile via `sampling: "qwen-thinking"` /
 * `"qwen-default"` regardless of architecture, or opt out with
 * `sampling: "raw"`.
 */
export const QWEN_THINKING_DEFAULTS = Object.freeze({
	temperature: 0.6,
	topK: 20,
	topP: 0.95,
	repetitionPenalty: 1.05,
} as const);

export const QWEN_NON_THINKING_DEFAULTS = Object.freeze({
	temperature: 0.7,
	topK: 20,
	topP: 0.8,
	repetitionPenalty: 1.1,
} as const);

/** Sampling mode dispatch on `CompletionConfig.sampling`. */
export type SamplingMode = "auto" | "qwen-thinking" | "qwen-default" | "raw";

/** Inputs to {@link resolveSamplingParams}. */
export interface SamplingResolutionInput {
	/** Mode from `CompletionConfig.sampling`. Default `"auto"` when unset. */
	samplingMode: SamplingMode;
	/**
	 * True when the loaded model is a Qwen-family architecture with a
	 * ChatML chat template — the only context where `"auto"` applies a
	 * profile.
	 */
	isQwenChatml: boolean;
	/**
	 * Reasoning toggle from the chat config. `false` selects the
	 * non-thinking profile under `"auto"` + Qwen+ChatML; `true` /
	 * `undefined` select the thinking profile.
	 */
	enableThinking?: boolean;
	/**
	 * Consumer-provided sampler overrides. Each defined value wins over
	 * the resolved profile and the engine fallback.
	 */
	consumer: {
		temperature?: number;
		topK?: number;
		topP?: number;
		repetitionPenalty?: number;
	};
}

/** Resolved sampler params after profile selection + consumer override. */
export interface ResolvedSamplingParams {
	temperature: number;
	topK: number;
	topP: number;
	repetitionPenalty: number;
}

/**
 * Pure resolver for `CompletionConfig.sampling` dispatch. Used by
 * `engine.generateStream` to compute the effective sampler params for a
 * generation request.
 *
 * Precedence (highest first): consumer override → forced profile (from
 * `"qwen-thinking"` / `"qwen-default"`) → auto profile (from `"auto"`
 * when Qwen+ChatML) → engine fallback (`1.0` / `0` / `1.0` / `1.0`).
 *
 * Mode `"raw"` produces no profile, falling straight through to consumer
 * overrides + engine fallbacks.
 */
export function resolveSamplingParams(
	input: SamplingResolutionInput,
): ResolvedSamplingParams {
	const { samplingMode, isQwenChatml, enableThinking, consumer } = input;
	const applyAutoQwen = samplingMode === "auto" && isQwenChatml;
	const forcedProfile =
		samplingMode === "qwen-thinking"
			? QWEN_THINKING_DEFAULTS
			: samplingMode === "qwen-default"
				? QWEN_NON_THINKING_DEFAULTS
				: null;
	const autoProfile = applyAutoQwen
		? enableThinking === false
			? QWEN_NON_THINKING_DEFAULTS
			: QWEN_THINKING_DEFAULTS
		: null;
	const activeProfile = forcedProfile ?? autoProfile;
	return {
		temperature: consumer.temperature ?? activeProfile?.temperature ?? 1.0,
		topK: consumer.topK ?? activeProfile?.topK ?? 0,
		topP: consumer.topP ?? activeProfile?.topP ?? 1.0,
		repetitionPenalty:
			consumer.repetitionPenalty ?? activeProfile?.repetitionPenalty ?? 1.0,
	};
}
