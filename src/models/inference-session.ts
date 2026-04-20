/** Configuration for creating an InferenceSession. */
export interface InferenceSessionConfig {
	/** Maximum number of tokens the session can hold. */
	maxTokens: number;
	/** Sampling temperature (unused by session but stored for reference). */
	temperature: number;
	/** Top-K parameter (unused by session but stored for reference). */
	topK: number;
	/** Top-P parameter (unused by session but stored for reference). */
	topP: number;
	/** Repetition penalty (unused by session but stored for reference). */
	repetitionPenalty: number;
	/** Policy when context length exceeds maxTokens. */
	contextOverflowPolicy: "stop" | "truncate";
}

/**
 * Tracks token position and history for a single inference sequence.
 *
 * Manages the autoregressive state: current position, accumulated token IDs,
 * and stop-condition logic for the generation loop.
 */
export class InferenceSession {
	private sequenceId: number;
	private position: number;
	private tokenHistory: number[];
	private config: InferenceSessionConfig;

	/**
	 * @param config - Session parameters including max tokens and overflow policy.
	 * @param sequenceId - Unique identifier for this inference sequence.
	 */
	constructor(config: InferenceSessionConfig, sequenceId: number) {
		this.config = config;
		this.sequenceId = sequenceId;
		this.position = 0;
		this.tokenHistory = [];
	}

	get currentPosition(): number {
		return this.position;
	}

	get tokens(): readonly number[] {
		return this.tokenHistory;
	}

	get id(): number {
		return this.sequenceId;
	}

	get isFull(): boolean {
		return this.position >= this.config.maxTokens;
	}

	/**
	 * Advance the current token position by n steps.
	 *
	 * @param n - Number of positions to advance.
	 */
	advance(n: number): void {
		this.position += n;
	}

	/**
	 * Append a token ID to the history.
	 *
	 * @param tokenId - Generated or prompt token ID to record.
	 */
	pushToken(tokenId: number): void {
		this.tokenHistory.push(tokenId);
	}

	/** Reset position and token history to the initial state. */
	reset(): void {
		this.position = 0;
		this.tokenHistory = [];
	}

	/**
	 * Determine whether generation should stop.
	 *
	 * @param tokenId - Last sampled token ID.
	 * @param eosTokenId - End-of-sequence token ID.
	 * @returns True if EOS reached or token limit exceeded.
	 */
	shouldStop(tokenId: number, eosTokenId: number): boolean {
		if (tokenId === eosTokenId) return true;
		if (this.tokenHistory.length >= this.config.maxTokens) return true;
		return false;
	}
}
