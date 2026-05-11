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

/**
 * Phi-3 / Phi-3.5 default sampling — greedy, per the upstream model
 * card. Phi-3.5-mini was trained on a heavily multilingual corpus and
 * any non-zero temperature causes language drift on short English
 * prompts (observed: same English prompt produced Spanish / Russian /
 * Japanese continuations across consecutive turns at temperature 0.4).
 * Greedy keeps Phi-3.5 in the prompt's language deterministically;
 * users wanting creative variety can dial temperature up via the
 * Settings panel.
 */
export const PHI3_DEFAULTS = Object.freeze({
	temperature: 0,
	topK: 0,
	topP: 1,
	repetitionPenalty: 1.1,
} as const);

/**
 * Mistral-Instruct (v0.1 / v0.2 / v0.3) default sampling — the official
 * recommendation from MistralAI for instruct fine-tunes is `T=0.7`,
 * `top_p=0.95`. Without these, the engine fallback (`T=1.0` /
 * unconstrained) makes Mistral skip the special `</s>` (EOS) token in
 * favour of higher-probability text continuations, producing multi-turn
 * self-dialogue instead of a single bounded reply. The `[INST]…[/INST]`
 * template family detected as `"llama2"` without a `<<SYS>>` envelope is
 * Mistral-Instruct (true Llama-2 keeps `<<SYS>>`).
 */
export const MISTRAL_DEFAULTS = Object.freeze({
	temperature: 0.7,
	topK: 0,
	topP: 0.95,
	repetitionPenalty: 1.0,
} as const);

/**
 * Gemma 4 default sampling. Google's reference inference pipeline uses
 * `temperature=1.0`, `top_p=0.95`, `top_k=64` for instruction-tuned
 * Gemma 4 checkpoints. For benchmarking the project pins temperature 0
 * via the `--eval-temperature` override; runtime chat callers get the
 * higher-temperature spec defaults unless they explicitly override.
 */
export const GEMMA4_DEFAULTS = Object.freeze({
	temperature: 1.0,
	topK: 64,
	topP: 0.95,
	repetitionPenalty: 1.0,
} as const);

/** Sampling mode dispatch on `CompletionConfig.sampling`. */
export type SamplingMode =
	| "auto"
	| "qwen-thinking"
	| "qwen-default"
	| "phi3"
	| "mistral"
	| "gemma4"
	| "raw";

/** Inputs to {@link resolveSamplingParams}. */
export interface SamplingResolutionInput {
	/** Mode from `CompletionConfig.sampling`. Default `"auto"` when unset. */
	samplingMode: SamplingMode;
	/**
	 * True when the loaded model is a Qwen-family architecture with a
	 * ChatML chat template — selects the Qwen `"auto"` profile.
	 */
	isQwenChatml: boolean;
	/**
	 * True when the loaded model is a Phi-3 / Phi-3.5 architecture —
	 * selects the Phi-3 `"auto"` profile. (Phi-3 has more conservative
	 * recommended sampling than the engine fallback.)
	 */
	isPhi3?: boolean;
	/**
	 * True when the loaded model is a Mistral-Instruct family model
	 * (template detected as `"llama2"` without a `<<SYS>>` envelope).
	 * Selects Mistral's official recommended sampling under `"auto"`.
	 */
	isMistral?: boolean;
	/**
	 * True when the loaded model is a Gemma 4 architecture (any size).
	 * Selects the Gemma 4 `"auto"` profile (T=1.0 / top_p=0.95 / top_k=64).
	 */
	isGemma4?: boolean;
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
 * `"qwen-thinking"` / `"qwen-default"` / `"phi3"` / `"mistral"`) →
 * auto profile (selected by family flags) → engine fallback (`1.0` /
 * `0` / `1.0` / `1.0`).
 *
 * Mode `"raw"` produces no profile, falling straight through to consumer
 * overrides + engine fallbacks.
 */
export function resolveSamplingParams(
	input: SamplingResolutionInput,
): ResolvedSamplingParams {
	const {
		samplingMode,
		isQwenChatml,
		isPhi3,
		isMistral,
		isGemma4,
		enableThinking,
		consumer,
	} = input;
	const applyAutoQwen = samplingMode === "auto" && isQwenChatml;
	const applyAutoPhi3 = samplingMode === "auto" && !!isPhi3 && !isQwenChatml;
	const applyAutoMistral =
		samplingMode === "auto" && !!isMistral && !isQwenChatml && !isPhi3;
	const applyAutoGemma4 =
		samplingMode === "auto" &&
		!!isGemma4 &&
		!isQwenChatml &&
		!isPhi3 &&
		!isMistral;
	const forcedProfile =
		samplingMode === "qwen-thinking"
			? QWEN_THINKING_DEFAULTS
			: samplingMode === "qwen-default"
				? QWEN_NON_THINKING_DEFAULTS
				: samplingMode === "phi3"
					? PHI3_DEFAULTS
					: samplingMode === "mistral"
						? MISTRAL_DEFAULTS
						: samplingMode === "gemma4"
							? GEMMA4_DEFAULTS
							: null;
	const autoProfile = applyAutoQwen
		? enableThinking === false
			? QWEN_NON_THINKING_DEFAULTS
			: QWEN_THINKING_DEFAULTS
		: applyAutoPhi3
			? PHI3_DEFAULTS
			: applyAutoMistral
				? MISTRAL_DEFAULTS
				: applyAutoGemma4
					? GEMMA4_DEFAULTS
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
