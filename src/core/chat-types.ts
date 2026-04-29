import type { GenerationFinishReason } from "../inference/generation.js";

/** A single message in a conversation. */
export interface ChatMessage {
	readonly role: "system" | "user" | "assistant";
	readonly content: string;
}

/**
 * Tool / function schema passed into the chat template. Just the shape the
 * model needs in the prompt — handlers, responses, etc. live on the
 * Character/ToolSystem side and aren't serialised here.
 */
export interface ChatToolSchema {
	name: string;
	description: string;
	parameters: Record<
		string,
		{
			type: "string" | "number" | "integer" | "boolean" | "array" | "object";
			description?: string;
			required?: boolean;
		}
	>;
}

/** Configuration for a chat completion request. */
export interface CompletionConfig {
	/** Maximum number of tokens to generate. Default: 512 */
	maxTokens?: number;
	/** Qwen-style reasoning toggle for chat templates that support it. */
	enableThinking?: boolean;
	/** Sampling temperature. 0 = greedy. Default: 1.0 */
	temperature?: number;
	/** Top-K filtering. 0 = disabled. Default: 0 */
	topK?: number;
	/** Top-P (nucleus) filtering. 1.0 = disabled. Default: 1.0 */
	topP?: number;
	/** Repetition penalty. 1.0 = disabled. Default: 1.0 */
	repetitionPenalty?: number;
	/**
	 * How to apply sampling-parameter defaults.
	 * - `"auto"` (default): apply Qwen profiles when architecture starts
	 *   with `"qwen"` and chat template is ChatML; otherwise use
	 *   consumer-provided values.
	 * - `"qwen-thinking"`: force `QWEN_THINKING_DEFAULTS` regardless of
	 *   architecture.
	 * - `"qwen-default"`: force `QWEN_NON_THINKING_DEFAULTS` regardless
	 *   of architecture.
	 * - `"raw"`: skip auto-application; use only consumer-provided values
	 *   (with engine fallbacks for unspecified fields).
	 *
	 * Consumer-provided field values always override profile defaults.
	 * Example: `sampling: "qwen-thinking", temperature: 0.9` applies the
	 * qwen profile then overrides temperature with 0.9.
	 */
	sampling?: "auto" | "qwen-thinking" | "qwen-default" | "raw";
	/**
	 * Optional PRNG seed for deterministic sampling. When omitted the
	 * sampler falls back to `Math.random`. Useful for reproducible
	 * smoke/bench runs.
	 */
	seed?: number;
	/** AbortSignal to cancel generation mid-stream. */
	signal?: AbortSignal;
	/** Custom stop token IDs that halt generation. */
	stopTokenIds?: readonly number[];
	/**
	 * Tool schemas to surface to the model via the chat template. For ChatML
	 * templates (Qwen3, Hermes) this injects a `<tools>...</tools>` block
	 * and tool-call instructions before the user's system message.
	 * Templates that don't support tool blocks ignore this.
	 */
	tools?: readonly ChatToolSchema[];
	/**
	 * Reserved in v1. Setting this throws — measurement on 2026-04-26
	 * (qwen3-8b-iq3m via qwen3-0.6b-q4f16, K=4) produced 3.0 vs 15.3 tok/s
	 * baseline (0.20×); verify-readback overhead and per-step drafter
	 * forwards dwarf the savings. Driver, sampler helpers, forwardVerify,
	 * and tests remain in tree (`src/inference/speculative.ts` etc.) for a
	 * future v2 lever (dynamic K, multi-tokenizer drafters, GPU-resident
	 * verify reduction).
	 *
	 * See TODO.md §19 and
	 * docs/superpowers/specs/2026-04-26-speculative-decoding-design.md.
	 */
	drafter?: string;
	/** Reserved in v1 — see `drafter` above. */
	draftLength?: number;
}

/** Input accepted by the public streaming API. */
export type StreamInput = string | ChatMessage[];

/** Configuration for the public streaming API. */
export type StreamConfig = CompletionConfig;

/** Statistics from a completed streamed generation. */
export interface StreamStats {
	/** Number of tokens generated (excluding prompt). */
	readonly tokenCount: number;
	/** Throughput in tokens per second. */
	readonly tokensPerSecond: number;
	/** Milliseconds from generation start to first sampled token. */
	readonly timeToFirstTokenMs: number;
	/** Total wall-clock time in ms. */
	readonly totalMs: number;
	/** Full response text. */
	readonly text: string;
	/** Why generation stopped. */
	readonly finishReason: GenerationFinishReason;
}

/** Each yield from generateStream() / chatCompletion(). */
export interface StreamChunk {
	/** Incremental text fragment (empty string on final done=true chunk). */
	readonly text: string;
	/** Generated token ID for incremental chunks; omitted on the final chunk. */
	readonly tokenId?: number;
	/** True only on the final yield, which carries stats. */
	readonly done: boolean;
	/** Present and populated only when done=true. */
	readonly stats?: StreamStats;
}

export type CompletionChunk = StreamChunk;
export type CompletionStats = StreamStats;
