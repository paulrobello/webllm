/** A single message in a conversation. */
export interface ChatMessage {
	role: "system" | "user" | "assistant";
	content: string;
}

/** Configuration for a chat completion request. */
export interface CompletionConfig {
	/** Maximum number of tokens to generate. Default: 512 */
	maxTokens?: number;
	/** Sampling temperature. 0 = greedy. Default: 1.0 */
	temperature?: number;
	/** Top-K filtering. 0 = disabled. Default: 0 */
	topK?: number;
	/** Top-P (nucleus) filtering. 1.0 = disabled. Default: 1.0 */
	topP?: number;
	/** Repetition penalty. 1.0 = disabled. Default: 1.0 */
	repetitionPenalty?: number;
	/** AbortSignal to cancel generation mid-stream. */
	signal?: AbortSignal;
	/** Custom stop token IDs that halt generation. */
	stopTokenIds?: number[];
}

/** Each yield from chatCompletion(). */
export interface CompletionChunk {
	/** Incremental text fragment (empty string on final done=true chunk). */
	text: string;
	/** True only on the final yield, which carries stats. */
	done: boolean;
	/** Present and populated only when done=true. */
	stats?: CompletionStats;
}

/** Statistics from a completed chat completion. */
export interface CompletionStats {
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
}
