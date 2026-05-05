import type { InferenceSession } from "../models/inference-session.js";
import type { DecodeMode, DecodeResult } from "./model-inference.js";
import type { Sampler } from "./sampler.js";
import {
	StreamingDecoder,
	TokenAttribute,
	type Tokenizer,
} from "./tokenizer.js";

/** Configuration for a single generation request. Public API surface. */
export interface GenerationConfig {
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
	/** Optional AbortSignal to cancel generation mid-stream. */
	signal?: AbortSignal;
}

/**
 * Internal options for `Generator.generate` / `generateTextStream`. Extends
 * the public `GenerationConfig` with chat-control steering fields (Qwen3
 * thinking-block masks, leading-whitespace enforcement, etc.). Engine builds
 * this internally and never exposes it on the public API.
 */
export interface InternalGenerationOptions extends GenerationConfig {
	/**
	 * Optional token IDs that should terminate generation if produced after the
	 * first generated token. Used to contain malformed chat-control reentry.
	 */
	forbiddenReentryTokens?: number[];
	/** Optional token ID for opening a thinking block. */
	thinkingOpenTokenId?: number;
	/** Optional token ID for closing a thinking block. */
	thinkingCloseTokenId?: number;
	/**
	 * When true, treat repeated `<think>` or a stray `</think>` as malformed and
	 * stop generation.
	 */
	enforceSingleThinkBlock?: boolean;
	/**
	 * Optional token IDs to suppress from sampling while inside an open thinking
	 * block. Used to steer malformed Qwen chat outputs toward `</think>` or
	 * normal reasoning tokens instead of repeating control markers.
	 */
	maskedTokensWhileThinking?: number[];
	/**
	 * Optional tokenizer used for lightweight token classification during
	 * generation-time steering.
	 */
	tokenizer?: Tokenizer;
	/**
	 * Optional token IDs to suppress after a think block closes and before any
	 * visible assistant answer text has been emitted.
	 */
	maskedTokensAfterThinkingUntilAnswer?: number[];
	/**
	 * When true, suppress EOS and custom stop tokens after `</think>` until at
	 * least one visible answer token has been emitted.
	 */
	requireVisibleAnswerAfterThinking?: boolean;
	/**
	 * When true, suppress EOS and custom stop tokens from the start of
	 * generation until at least one visible answer token has been emitted.
	 */
	requireVisibleAnswerBeforeStop?: boolean;
	/**
	 * When true, suppress whitespace-only text tokens after `</think>` until a
	 * visible answer token has been emitted.
	 */
	suppressWhitespaceOnlyAfterThinking?: boolean;
	/**
	 * When true, suppress whitespace-only text tokens from the start of
	 * generation until a visible answer token has been emitted.
	 */
	suppressWhitespaceOnlyUntilAnswer?: boolean;
	/**
	 * Optional token IDs to suppress after visible assistant answer text has
	 * started, preventing relapse into control-token scaffolding.
	 */
	maskedTokensAfterAnswerStarts?: number[];
	/**
	 * When true, the first token sampled after `</think>` closes is forced to
	 * begin with whitespace by masking + resampling until a non-control token
	 * whose decoded text starts with `\s` is produced. One-shot: applies only
	 * to the first post-`</think>` step, then defers to
	 * `suppressWhitespaceOnlyAfterThinking` for subsequent steps. Prevents
	 * run-on output like `</think>The answer ...`.
	 */
	requireLeadingWhitespaceAfterThinking?: boolean;
}

export type GenerationFinishReason =
	/** Generation cancelled via AbortSignal. */
	| "aborted"
	/** End-of-sequence token sampled. */
	| "eos"
	/** maxTokens budget exhausted. */
	| "max-tokens"
	/** A custom `stopTokens` entry was sampled. */
	| "stop-token";

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
	/** Why generation stopped. */
	finishReason: GenerationFinishReason;
}

export interface GenerationStreamChunk {
	/** Incremental decoded text for a generated token. */
	text: string;
	/** Token ID for incremental chunks; omitted on the final chunk. */
	tokenId?: number;
	/** True only on the final yield carrying completion metadata. */
	done: boolean;
	/** Present and populated only when done=true. */
	stats?: GenerationStreamResult;
}

export interface GenerationStreamResult extends GenerationResult {
	/** Total wall-clock time in ms. */
	totalMs: number;
}

export interface GenerationStreamOptions {
	promptTokenIds: number[];
	sampler: Sampler;
	session: InferenceSession;
	eosTokenId: number;
	tokenizer: Tokenizer;
	forwardPass: (
		tokenIds: number[],
		positions: number[],
	) => Float32Array | Promise<Float32Array>;
	config: InternalGenerationOptions;
	forwardDecode?: (
		tokenIds: number[],
		positions: number[],
		mode: DecodeMode,
		topK?: number,
	) => Promise<DecodeResult>;
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
	 * @param config - Generation configuration (signal travels via `config.signal`).
	 * @param forwardDecode - Optional GPU-side decode function for reduced readback.
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
		) => Float32Array | Promise<Float32Array>,
		config: InternalGenerationOptions,
		forwardDecode?: (
			tokenIds: number[],
			positions: number[],
			mode: DecodeMode,
			topK?: number,
		) => Promise<DecodeResult>,
	): AsyncGenerator<number, GenerationResult> {
		const startTime = performance.now();
		let finishReason: GenerationFinishReason | undefined;
		const forbiddenReentryTokens = new Set(config.forbiddenReentryTokens ?? []);
		const thinkOpenTokenId = config.thinkingOpenTokenId;
		const thinkCloseTokenId = config.thinkingCloseTokenId;
		const maskedTokensWhileThinking = config.maskedTokensWhileThinking ?? [];
		const maskedTokensAfterThinkingUntilAnswer =
			config.maskedTokensAfterThinkingUntilAnswer ?? [];
		const maskedTokensAfterAnswerStarts =
			config.maskedTokensAfterAnswerStarts ?? [];
		const requireVisibleAnswerAfterThinking =
			config.requireVisibleAnswerAfterThinking === true;
		const requireVisibleAnswerBeforeStop =
			config.requireVisibleAnswerBeforeStop === true;
		const suppressWhitespaceOnlyAfterThinking =
			config.suppressWhitespaceOnlyAfterThinking === true;
		const suppressWhitespaceOnlyUntilAnswer =
			config.suppressWhitespaceOnlyUntilAnswer === true;
		const requireLeadingWhitespaceAfterThinking =
			config.requireLeadingWhitespaceAfterThinking === true;
		let thinkDepth = 0;
		let thinkClosed = false;
		let hasVisibleAnswerText = false;
		let waitingForVisibleAnswer = requireVisibleAnswerBeforeStop;
		let requireLeadingWhitespaceForNextStep = false;

		if (config.signal?.aborted) {
			const elapsed = performance.now() - startTime;
			return {
				tokens: [...session.tokens],
				text: "",
				tokenCount: 0,
				tokensPerSecond: 0,
				timeToFirstTokenMs: elapsed,
				finishReason: "aborted",
			};
		}

		// 1. Prefill: process all prompt tokens at once
		const promptPositions = promptTokenIds.map(
			(_: number, i: number) => session.currentPosition + i,
		);
		const logits = await forwardPass(promptTokenIds, promptPositions);
		session.advance(promptTokenIds.length);
		for (const id of promptTokenIds) session.pushToken(id);

		if (config.signal?.aborted) {
			const elapsed = performance.now() - startTime;
			return {
				tokens: [...session.tokens],
				text: "",
				tokenCount: 0,
				tokensPerSecond: 0,
				timeToFirstTokenMs: elapsed,
				finishReason: "aborted",
			};
		}

		// 2. Sample first token from prefill logits
		const firstTokenTime = performance.now();
		const recentTokens = [...promptTokenIds];
		sampler.applyRepetitionPenalty(logits, recentTokens.slice(-64));
		if (waitingForVisibleAnswer) {
			maskTokenLogits(logits, [...(config.stopTokens ?? []), eosTokenId]);
		}
		let sampledId = sampleVisibleAnswerToken(
			logits,
			sampler,
			config.tokenizer,
			waitingForVisibleAnswer,
			false,
			suppressWhitespaceOnlyUntilAnswer,
		);
		if (sampledId === thinkOpenTokenId) {
			thinkDepth = 1;
		} else if (
			config.enforceSingleThinkBlock &&
			sampledId === thinkCloseTokenId
		) {
			return {
				tokens: [...session.tokens],
				text: "",
				tokenCount: 0,
				tokensPerSecond: 0,
				timeToFirstTokenMs: firstTokenTime - startTime,
				finishReason: "stop-token",
			};
		}
		if (
			waitingForVisibleAnswer &&
			isVisibleTextToken(config.tokenizer, sampledId)
		) {
			hasVisibleAnswerText = true;
			waitingForVisibleAnswer = false;
		}
		yield sampledId;
		session.pushToken(sampledId);
		recentTokens.push(sampledId);
		let generatedCount = 1;

		// Decode-mode selection is dynamic per step. When no steering state
		// is active (no open think block, not waiting for visible answer
		// post-`</think>`, and no during-answer scaffolding mask) the topk
		// fast path is safe even when the config configures steering — the
		// masks would be no-ops for this step. This is what gives qwen3
		// thinking-off and tinyllama realistic-sampling runs the topk
		// throughput when steering happens to be inactive.
		//
		// When steering IS active, qwen3 mask sets almost never land in the
		// top-K of full-vocab logits (measured 0.31% top-K hit rate / 0.41%
		// top-(K+10) hit rate across 982 mask-token instances). So we ask
		// the GPU for K + STEERING_TOPK_BUFFER candidates and CPU-filter
		// the masked indices instead of paying for a full 32K/152K-vocab
		// readback + JS sampling pipeline. Waiting-for-visible-answer
		// steps stay on the full path because their whitespace-guard
		// resampling needs full-vocab access.
		const greedyOk = sampler.isGreedy && sampler.noPenalty;
		const topkOk = sampler.topK > 0 && !greedyOk;
		const decodeStep = forwardDecode;
		const STEERING_TOPK_BUFFER = 10;

		// 3. Autoregressive decode loop
		while (!session.shouldStop(sampledId, eosTokenId)) {
			if (config.signal?.aborted) {
				finishReason = "aborted";
				break;
			}
			if (generatedCount >= config.maxTokens) {
				finishReason = "max-tokens";
				break;
			}

			const steeringActive =
				thinkDepth > 0 || waitingForVisibleAnswer || hasVisibleAnswerText;
			// `waitingForVisibleAnswer` needs whitespace-guard resampling on
			// full logits; everything else can use topk + CPU mask filter.
			const steeringTopkOk =
				topkOk && steeringActive && !waitingForVisibleAnswer;
			const useTopK = topkOk && (!steeringActive || steeringTopkOk);
			const gpuMode: DecodeMode = greedyOk
				? "greedy"
				: useTopK
					? "topk"
					: "full";

			if (decodeStep && gpuMode === "greedy") {
				// GPU ARGMAX — no CPU sampling needed
				const result = await decodeStep(
					[sampledId],
					[session.currentPosition],
					"greedy",
				);
				if (result.tokenId === undefined) {
					throw new Error("forwardDecode(greedy) returned no tokenId");
				}
				session.advance(1);
				if (config.signal?.aborted) break;
				sampledId = result.tokenId;
			} else if (decodeStep && gpuMode === "topk") {
				// GPU TOP_K + CPU sampling on reduced set. When steering is
				// active we request K + STEERING_TOPK_BUFFER candidates so
				// CPU-side mask filtering still leaves enough room.
				const requestedK = steeringTopkOk
					? sampler.topK + STEERING_TOPK_BUFFER
					: sampler.topK;
				const result = await decodeStep(
					[sampledId],
					[session.currentPosition],
					"topk",
					requestedK,
				);
				if (!result.topKIndices || !result.topKValues) {
					throw new Error("forwardDecode(topk) returned incomplete top-k data");
				}
				session.advance(1);
				if (config.signal?.aborted) break;

				let indices = result.topKIndices;
				let values = result.topKValues;
				if (steeringTopkOk) {
					const activeMask =
						thinkDepth > 0
							? maskedTokensWhileThinking
							: hasVisibleAnswerText
								? maskedTokensAfterAnswerStarts
								: [];
					if (activeMask.length > 0) {
						const masked = new Set(activeMask);
						const keepIdx: number[] = [];
						const keepVal: number[] = [];
						for (let i = 0; i < indices.length; i++) {
							if (!masked.has(indices[i])) {
								keepIdx.push(indices[i]);
								keepVal.push(values[i]);
							}
						}
						// If the entire pool is masked (essentially never — measured
						// 0% across 982 mask checks) we degrade gracefully by
						// sampling from the unfiltered pool for this one step
						// rather than firing a redundant full-vocab readback.
						if (keepIdx.length > 0) {
							indices = new Int32Array(keepIdx);
							values = new Float32Array(keepVal);
						}
					}
				}

				sampledId = sampler.sampleFromTopK(
					indices,
					values,
					recentTokens.slice(-64),
				);
			} else if (decodeStep && gpuMode === "full") {
				// GPU full logits + full CPU sampling pipeline
				const result = await decodeStep(
					[sampledId],
					[session.currentPosition],
					"full",
				);
				if (!result.logits) {
					throw new Error("forwardDecode(full) returned no logits");
				}
				session.advance(1);
				if (config.signal?.aborted) break;
				sampler.applyRepetitionPenalty(result.logits, recentTokens.slice(-64));
				if (thinkDepth > 0) {
					maskTokenLogits(result.logits, maskedTokensWhileThinking);
				} else if (waitingForVisibleAnswer) {
					maskTokenLogits(result.logits, maskedTokensAfterThinkingUntilAnswer);
					if (
						requireVisibleAnswerAfterThinking ||
						requireVisibleAnswerBeforeStop
					) {
						maskTokenLogits(result.logits, [
							eosTokenId,
							...(config.stopTokens ?? []),
						]);
					}
				} else if (hasVisibleAnswerText) {
					maskTokenLogits(result.logits, maskedTokensAfterAnswerStarts);
				}
				const usedLeadingWsGuard =
					waitingForVisibleAnswer && requireLeadingWhitespaceForNextStep;
				sampledId = sampleVisibleAnswerToken(
					result.logits,
					sampler,
					config.tokenizer,
					waitingForVisibleAnswer,
					usedLeadingWsGuard,
					waitingForVisibleAnswer
						? thinkClosed
							? suppressWhitespaceOnlyAfterThinking
							: suppressWhitespaceOnlyUntilAnswer
						: false,
				);
				if (usedLeadingWsGuard) requireLeadingWhitespaceForNextStep = false;
			} else {
				// Fallback: original path (no forwardDecode provided)
				const stepLogits = await forwardPass(
					[sampledId],
					[session.currentPosition],
				);
				session.advance(1);
				if (config.signal?.aborted) break;
				sampler.applyRepetitionPenalty(stepLogits, recentTokens.slice(-64));
				if (thinkDepth > 0) {
					maskTokenLogits(stepLogits, maskedTokensWhileThinking);
				} else if (waitingForVisibleAnswer) {
					maskTokenLogits(stepLogits, maskedTokensAfterThinkingUntilAnswer);
					if (
						requireVisibleAnswerAfterThinking ||
						requireVisibleAnswerBeforeStop
					) {
						maskTokenLogits(stepLogits, [
							eosTokenId,
							...(config.stopTokens ?? []),
						]);
					}
				} else if (hasVisibleAnswerText) {
					maskTokenLogits(stepLogits, maskedTokensAfterAnswerStarts);
				}
				const usedLeadingWsGuard =
					waitingForVisibleAnswer && requireLeadingWhitespaceForNextStep;
				sampledId = sampleVisibleAnswerToken(
					stepLogits,
					sampler,
					config.tokenizer,
					waitingForVisibleAnswer,
					usedLeadingWsGuard,
					waitingForVisibleAnswer
						? thinkClosed
							? suppressWhitespaceOnlyAfterThinking
							: suppressWhitespaceOnlyUntilAnswer
						: false,
				);
				if (usedLeadingWsGuard) requireLeadingWhitespaceForNextStep = false;
			}

			generatedCount++;
			recentTokens.push(sampledId);

			if (sampledId === thinkOpenTokenId) {
				if (config.enforceSingleThinkBlock && thinkDepth > 0) {
					finishReason = "stop-token";
					break;
				}
				thinkDepth++;
			} else if (sampledId === thinkCloseTokenId) {
				if (config.enforceSingleThinkBlock && thinkDepth === 0) {
					finishReason = "stop-token";
					break;
				}
				thinkDepth = Math.max(0, thinkDepth - 1);
				if (thinkDepth === 0) {
					thinkClosed = true;
					waitingForVisibleAnswer = true;
					if (requireLeadingWhitespaceAfterThinking) {
						requireLeadingWhitespaceForNextStep = true;
					}
				}
			}
			if (
				waitingForVisibleAnswer &&
				!hasVisibleAnswerText &&
				isVisibleTextToken(config.tokenizer, sampledId)
			) {
				hasVisibleAnswerText = true;
				waitingForVisibleAnswer = false;
			}
			if (generatedCount > 1 && forbiddenReentryTokens.has(sampledId)) {
				finishReason = "stop-token";
				break;
			}
			if (config.stopTokens?.includes(sampledId)) {
				finishReason = "stop-token";
				break;
			}
			if (sampledId === eosTokenId) {
				finishReason = "eos";
				break;
			}
			yield sampledId;
			session.pushToken(sampledId);
		}

		// 4. Return stats
		if (!finishReason) {
			finishReason = config.signal?.aborted
				? "aborted"
				: sampledId === eosTokenId
					? "eos"
					: "max-tokens";
		}

		const elapsed = performance.now() - startTime;
		const totalGenerated = session.tokens.length - promptTokenIds.length;
		return {
			tokens: [...session.tokens],
			text: "",
			tokenCount: totalGenerated,
			tokensPerSecond: totalGenerated / (elapsed / 1000),
			timeToFirstTokenMs: firstTokenTime - startTime,
			finishReason,
		};
	}
}

function maskTokenLogits(logits: Float32Array, tokenIds: number[]): void {
	for (const tokenId of tokenIds) {
		if (tokenId >= 0 && tokenId < logits.length) {
			logits[tokenId] = -Infinity;
		}
	}
}

function isVisibleTextToken(
	tokenizer: Tokenizer | undefined,
	tokenId: number,
): boolean {
	if (
		!tokenizer ||
		typeof tokenizer.getToken !== "function" ||
		typeof tokenizer.decode !== "function"
	) {
		return false;
	}
	const token = tokenizer.getToken(tokenId);
	if (!token) return false;
	if (token.attr & (TokenAttribute.CONTROL | TokenAttribute.USER_DEFINED)) {
		return false;
	}
	return tokenizer.decode([tokenId]).trim().length > 0;
}

function isWhitespaceOnlyTextToken(
	tokenizer: Tokenizer | undefined,
	tokenId: number,
): boolean {
	if (
		!tokenizer ||
		typeof tokenizer.getToken !== "function" ||
		typeof tokenizer.decode !== "function"
	) {
		return false;
	}
	const token = tokenizer.getToken(tokenId);
	if (!token) return false;
	if (token.attr & (TokenAttribute.CONTROL | TokenAttribute.USER_DEFINED)) {
		return false;
	}
	return tokenizer.decode([tokenId]).trim().length === 0;
}

function tokenStartsWithWhitespace(
	tokenizer: Tokenizer | undefined,
	tokenId: number,
): boolean {
	if (
		!tokenizer ||
		typeof tokenizer.getToken !== "function" ||
		typeof tokenizer.decode !== "function"
	) {
		return false;
	}
	const token = tokenizer.getToken(tokenId);
	if (!token) return false;
	if (token.attr & (TokenAttribute.CONTROL | TokenAttribute.USER_DEFINED)) {
		return false;
	}
	const text = tokenizer.decode([tokenId]);
	if (text.length === 0) return false;
	return /^\s/.test(text);
}

function sampleVisibleAnswerToken(
	logits: Float32Array,
	sampler: Sampler,
	tokenizer: Tokenizer | undefined,
	guardPostThinkAnswerStart: boolean,
	requireLeadingWhitespace: boolean,
	suppressWhitespaceOnly: boolean,
): number {
	let sampledId = sampler.sample(logits);
	if (!guardPostThinkAnswerStart) return sampledId;

	if (requireLeadingWhitespace) {
		const masked = new Set<number>();
		while (
			!tokenStartsWithWhitespace(tokenizer, sampledId) &&
			!masked.has(sampledId)
		) {
			masked.add(sampledId);
			maskTokenLogits(logits, [sampledId]);
			sampledId = sampler.sample(logits);
		}
		return sampledId;
	}

	if (suppressWhitespaceOnly) {
		const masked = new Set<number>();
		while (
			isWhitespaceOnlyTextToken(tokenizer, sampledId) &&
			!masked.has(sampledId)
		) {
			masked.add(sampledId);
			maskTokenLogits(logits, [sampledId]);
			sampledId = sampler.sample(logits);
		}
	}
	return sampledId;
}

export async function* generateTextStream({
	promptTokenIds,
	sampler,
	session,
	eosTokenId,
	tokenizer,
	forwardPass,
	config,
	forwardDecode,
}: GenerationStreamOptions): AsyncGenerator<
	GenerationStreamChunk,
	GenerationStreamResult
> {
	const startTime = performance.now();
	const decoder = new StreamingDecoder(tokenizer);
	const gen = Generator.generate(
		promptTokenIds,
		sampler,
		session,
		eosTokenId,
		forwardPass,
		config,
		forwardDecode,
	);

	while (true) {
		const next = await gen.next();
		if (next.done) {
			const stats: GenerationStreamResult = {
				...next.value,
				text: decoder.text,
				totalMs: performance.now() - startTime,
			};
			yield { text: "", done: true, stats };
			return stats;
		}

		const tokenId = next.value;
		yield {
			text: decoder.push(tokenId),
			tokenId,
			done: false,
		};
	}
}
