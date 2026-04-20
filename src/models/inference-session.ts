export interface InferenceSessionConfig {
	maxTokens: number;
	temperature: number;
	topK: number;
	topP: number;
	repetitionPenalty: number;
	contextOverflowPolicy: "stop" | "truncate";
}

export class InferenceSession {
	private sequenceId: number;
	private position: number;
	private tokenHistory: number[];
	private config: InferenceSessionConfig;

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

	advance(n: number): void {
		this.position += n;
	}

	pushToken(tokenId: number): void {
		this.tokenHistory.push(tokenId);
	}

	reset(): void {
		this.position = 0;
		this.tokenHistory = [];
	}

	shouldStop(tokenId: number, eosTokenId: number): boolean {
		if (tokenId === eosTokenId) return true;
		if (this.tokenHistory.length >= this.config.maxTokens) return true;
		return false;
	}
}
