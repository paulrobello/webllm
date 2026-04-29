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
export const QWEN_THINKING_DEFAULTS = {
	temperature: 0.6,
	topK: 20,
	topP: 0.95,
	repetitionPenalty: 1.05,
} as const;

export const QWEN_NON_THINKING_DEFAULTS = {
	temperature: 0.7,
	topK: 20,
	topP: 0.8,
	repetitionPenalty: 1.1,
} as const;
