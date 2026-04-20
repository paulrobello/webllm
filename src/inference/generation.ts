import type { InferenceSession } from "../models/inference-session.js";
import type { Sampler } from "./sampler.js";

/** Configuration for a single generation request. */
export interface GenerationConfig {
	/** The input prompt text (pre-tokenization). */
	prompt: string;
	/** Maximum number of tokens to generate. */
	maxTokens: number;
	/** Sampling temperature. 0 = greedy. */
	temperature: number;
	/** Top-K sampling parameter. 0 = disabled. */
	topK: number;
	/** Top-P (nucleus) sampling parameter. 1.0 = disabled. */
	topP: number;
	/** Repetition penalty multiplier. 1.0 = disabled. */
	repetitionPenalty: number;
	/** Optional custom stop token IDs that halt generation. */
	stopTokens?: number[];
}

/** Statistics and output from a completed generation run. */
export interface GenerationResult {
	/** All tokens produced during generation (excluding prompt). */
	tokens: number[];
	/** Decoded output text (requires tokenizer integration). */
	text: string;
	/** Number of generated tokens. */
	tokenCount: number;
	/** Throughput in tokens per second. */
	tokensPerSecond: number;
	/** Latency from generation start to first sampled token, in ms. */
	timeToFirstTokenMs: number;
}

/**
 * Autoregressive generation loop.
 *
 * Ties together tokenization, forward passes, sampling, and session management
 * into a single async generator that yields tokens as they are produced.
 *
 * biome-ignore lint/complexity/noStaticOnlyClass: Class provides namespace for generation; may gain instance state in future phases.
 */
export class Generator {
	/**
	 * Run an autoregressive generation loop.
	 *
	 * @param promptTokenIds - Pre-tokenized prompt token IDs.
	 * @param sampler - Sampler instance for selecting next tokens.
	 * @param session - Inference session tracking position and token history.
	 * @param eosTokenId - End-of-sequence token ID.
	 * @param forwardPass - Function that runs the model forward pass and returns logits.
	 * @param config - Generation configuration.
	 * @yields Sampled token IDs (one at a time, excluding prompt tokens).
	 * @returns Generation statistics after the loop completes.
	 */
	static async *generate(
		promptTokenIds: number[],
		sampler: Sampler,
		session: InferenceSession,
		eosTokenId: number,
		forwardPass: (
			tokenIds: number[],
			positions: number[],
		) => Promise<Float32Array>,
		config: GenerationConfig,
	): AsyncGenerator<number, GenerationResult> {
		const startTime = performance.now();

		// 1. Prefill: process all prompt tokens at once
		const promptPositions = promptTokenIds.map(
			(_: number, i: number) => session.currentPosition + i,
		);
		const logits = await forwardPass(promptTokenIds, promptPositions);
		session.advance(promptTokenIds.length);
		for (const id of promptTokenIds) session.pushToken(id);

		// 2. Sample first token from prefill logits
		const firstTokenTime = performance.now();
		let sampledId = sampler.sample(logits);
		yield sampledId;
		session.pushToken(sampledId);
		let generatedCount = 1;

		// 3. Autoregressive decode loop
		while (!session.shouldStop(sampledId, eosTokenId)) {
			if (generatedCount >= config.maxTokens) break;
			if (config.stopTokens?.includes(sampledId)) break;

			const stepLogits = await forwardPass(
				[sampledId],
				[session.currentPosition],
			);
			session.advance(1);
			sampledId = sampler.sample(stepLogits);
			generatedCount++;

			if (config.stopTokens?.includes(sampledId)) break;
			if (sampledId === eosTokenId) break;
			yield sampledId;
			session.pushToken(sampledId);
		}

		// 4. Return stats
		const elapsed = performance.now() - startTime;
		const totalGenerated = session.tokens.length - promptTokenIds.length;
		return {
			tokens: [...session.tokens],
			text: "",
			tokenCount: totalGenerated,
			tokensPerSecond: totalGenerated / (elapsed / 1000),
			timeToFirstTokenMs: firstTokenTime - startTime,
		};
	}
}
