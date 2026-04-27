import type { GenerationFinishReason } from "../inference/generation.js";

/** A single message in a conversation. */
export interface ChatMessage {
	role: "system" | "user" | "assistant";
	content: string;
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
		{ type: string; description?: string; required?: boolean }
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
	 * Optional PRNG seed for deterministic sampling. When omitted the
	 * sampler falls back to `Math.random`. Useful for reproducible
	 * smoke/bench runs.
	 */
	seed?: number;
	/** AbortSignal to cancel generation mid-stream. */
	signal?: AbortSignal;
	/** Custom stop token IDs that halt generation. */
	stopTokenIds?: number[];
	/**
	 * Tool schemas to surface to the model via the chat template. For ChatML
	 * templates (Qwen3, Hermes) this injects a `<tools>...</tools>` block
	 * and tool-call instructions before the user's system message.
	 * Templates that don't support tool blocks ignore this.
	 */
	tools?: ChatToolSchema[];
	/**
	 * If set, route the generation through the speculative-decode driver
	 * using the named drafter model (must already be loaded via
	 * `loadModel`). The drafter must share the target's tokenizer (vocab
	 * size, BOS, EOS) and the request must not configure any steering
	 * fields (Qwen3 thinking masks etc.) — otherwise routing throws.
	 *
	 * See docs/superpowers/specs/2026-04-26-speculative-decoding-design.md.
	 */
	drafter?: string;
	/** Speculative-decode draft burst length (default 4, range [2, 32]). */
	draftLength?: number;
}

/** Input accepted by the public streaming API. */
export type StreamInput = string | ChatMessage[];

/** Configuration for the public streaming API. */
export type StreamConfig = CompletionConfig;

/** Statistics from a completed streamed generation. */
export interface StreamStats {
	/** Number of tokens generated (excluding prompt). */
	tokenCount: number;
	/** Throughput in tokens per second. */
	tokensPerSecond: number;
	/** Milliseconds from generation start to first sampled token. */
	timeToFirstTokenMs: number;
	/** Total wall-clock time in ms. */
	totalMs: number;
	/** Full response text. */
	text: string;
	/** Why generation stopped. */
	finishReason: GenerationFinishReason;
}

/** Each yield from generateStream() / chatCompletion(). */
export interface StreamChunk {
	/** Incremental text fragment (empty string on final done=true chunk). */
	text: string;
	/** Generated token ID for incremental chunks; omitted on the final chunk. */
	tokenId?: number;
	/** True only on the final yield, which carries stats. */
	done: boolean;
	/** Present and populated only when done=true. */
	stats?: StreamStats;
}

export type CompletionChunk = StreamChunk;
export type CompletionStats = StreamStats;
