import type {
	GenerationConfig,
	GenerationFinishReason,
	GenerationResult,
} from "./generation.js";
import type { ModelInference } from "./model-inference.js";
import type { Sampler } from "./sampler.js";
import { StreamingDecoder, type Tokenizer } from "./tokenizer.js";

/**
 * Per-stream warn-once state. The driver constructs one of these per stream
 * and passes it into every `acceptPrefix` call. The function mutates the
 * `degenerateResidualWarned` flag so a degenerate residual emits exactly one
 * warning per stream (matching spec §9.3) instead of once per process.
 */
export interface SpeculativeWarnState {
	degenerateResidualWarned: boolean;
}

/** Construct fresh warn state for a new stream. */
export function newSpeculativeWarnState(): SpeculativeWarnState {
	return { degenerateResidualWarned: false };
}

/** Result of `acceptPrefix`. */
export interface AcceptPrefixResult {
	/** Number of drafted tokens accepted (0..K). */
	acceptedCount: number;
	/**
	 * The next visible token to emit beyond the accepted prefix:
	 * - On rejection: residual-distribution sample.
	 * - On full accept (acceptedCount === K): bonus token from target row K-1.
	 *
	 * `null` when the accept loop terminated on a finish condition before
	 * drawing a residual / bonus sample (the trigger was an accepted draft
	 * token already included in the yielded prefix, or maxTokens was hit
	 * before the residual / bonus sample could be drawn).
	 */
	finalSampledId: number | null;
	/**
	 * Set when the accept loop terminated due to a finish condition (EOS,
	 * custom stop-token id, or maxTokens exhaustion). `null` when the loop
	 * terminated normally.
	 */
	finishReason: GenerationFinishReason | null;
}

interface AcceptPrefixArgs {
	draftTokens: number[];
	draftDistros: Float32Array[];
	targetDistros: Float32Array[];
	sampler: Sampler;
	warnState: SpeculativeWarnState;
	eosTokenId: number;
	stopTokens: ReadonlySet<number>;
	maxTokens: number;
	generatedCount: number;
}

/**
 * Standard Leviathan-et-al rejection sampling for speculative decoding.
 *
 * For each drafted token id at position k, computes
 * `r = min(1, p_target[id] / p_draft[id])` and accepts iff a fresh uniform
 * draw `u < r`. On first rejection, samples a final visible token from the
 * residual distribution `q[i] = max(0, p_target[i] - p_draft[i])`
 * normalized. If all K accept, samples a bonus token from
 * `targetDistros[K-1]` (free token from the parallel verify).
 *
 * Pure modulo `warnState.degenerateResidualWarned`, which it sets to true
 * the first time a degenerate residual is encountered in the stream so the
 * fallback warning fires exactly once per stream.
 *
 * Walks accepted tokens left-to-right and truncates on EOS, a custom
 * stop-token id, or maxTokens exhaustion. On finish-condition truncation,
 * `finalSampledId` is `null` when the trigger was an accepted draft token
 * (already in the yielded prefix) or when maxTokens is hit before the
 * residual / bonus sample. `finalSampledId` is non-null when the trigger
 * was the residual or bonus sample itself.
 */
export function acceptPrefix(args: AcceptPrefixArgs): AcceptPrefixResult {
	const {
		draftTokens,
		draftDistros,
		targetDistros,
		sampler,
		warnState,
		eosTokenId,
		stopTokens,
		maxTokens,
		generatedCount,
	} = args;
	const K = draftTokens.length;
	if (K === 0) {
		throw new Error("acceptPrefix: K=0 is invalid");
	}
	if (draftDistros.length !== K || targetDistros.length !== K) {
		throw new Error(
			`acceptPrefix: distro count mismatch (K=${K}, draft=${draftDistros.length}, target=${targetDistros.length})`,
		);
	}

	// Threshold convention for the maxTokens checks below:
	//   post-emit: `generatedCount + acceptedCount >= maxTokens`
	//     — acceptedCount has just been incremented to count the token we
	//       accepted; we've now produced exactly `generatedCount + acceptedCount`
	//       tokens total. Equality means we hit the budget exactly.
	//   pre-emit:  `generatedCount + acceptedCount + 1 > maxTokens`
	//     — the +1 accounts for the residual / bonus sample we're *about to*
	//       draw and yield. Strict `>` reads as "would exceed budget if we
	//       drew it"; equality is fine, we yield it then stop.
	//   post-residual / post-bonus: same `+1 >= maxTokens` form as post-emit.
	// The two pre-emit `>` checks are nearly dead code (the post-emit `>=`
	// check at line ~114 establishes the invariant they could fire on for
	// k > 0); kept as defensive guards against caller-supplied
	// `generatedCount >= maxTokens`.
	let acceptedCount = 0;
	for (let k = 0; k < K; k++) {
		const id = draftTokens[k];
		const pT = targetDistros[k][id] ?? 0;
		const pD = draftDistros[k][id] ?? 0;
		const r = pD > 0 ? Math.min(1, pT / pD) : 0;
		const u = sampler.rand();
		if (u < r) {
			acceptedCount++;
			// Check finish conditions on the just-accepted token.
			if (id === eosTokenId || stopTokens.has(id)) {
				return {
					acceptedCount,
					finalSampledId: null,
					finishReason: "stop-token",
				};
			}
			if (generatedCount + acceptedCount >= maxTokens) {
				return {
					acceptedCount,
					finalSampledId: null,
					finishReason: "max-tokens",
				};
			}
			continue;
		}
		// Reject: check max-tokens BEFORE drawing residual sample.
		if (generatedCount + acceptedCount + 1 > maxTokens) {
			return {
				acceptedCount,
				finalSampledId: null,
				finishReason: "max-tokens",
			};
		}
		const finalSampledId = sampleResidual(
			targetDistros[k],
			draftDistros[k],
			sampler,
			warnState,
		);
		// The residual sample itself can be EOS / stop.
		if (finalSampledId === eosTokenId || stopTokens.has(finalSampledId)) {
			return { acceptedCount, finalSampledId, finishReason: "stop-token" };
		}
		if (generatedCount + acceptedCount + 1 >= maxTokens) {
			return { acceptedCount, finalSampledId, finishReason: "max-tokens" };
		}
		return { acceptedCount, finalSampledId, finishReason: null };
	}
	// All K accepted: bonus sample.
	if (generatedCount + acceptedCount + 1 > maxTokens) {
		return { acceptedCount, finalSampledId: null, finishReason: "max-tokens" };
	}
	const finalSampledId = sampler.sampleFromDistribution(targetDistros[K - 1]);
	if (finalSampledId === eosTokenId || stopTokens.has(finalSampledId)) {
		return { acceptedCount, finalSampledId, finishReason: "stop-token" };
	}
	if (generatedCount + acceptedCount + 1 >= maxTokens) {
		return { acceptedCount, finalSampledId, finishReason: "max-tokens" };
	}
	return { acceptedCount, finalSampledId, finishReason: null };
}

/**
 * Sample from the residual distribution
 * `q[i] = max(0, p_target[i] - p_draft[i])` normalized to sum to 1.
 *
 * If the residual is degenerate (Σq < 1e-9), fall back to sampling from
 * `p_target` and emit one warning per stream via `warnState`.
 */
function sampleResidual(
	pTarget: Float32Array,
	pDraft: Float32Array,
	sampler: Sampler,
	warnState: SpeculativeWarnState,
): number {
	const n = pTarget.length;
	const q = new Float32Array(n);
	let sum = 0;
	for (let i = 0; i < n; i++) {
		const v = Math.max(0, pTarget[i] - pDraft[i]);
		q[i] = v;
		sum += v;
	}
	if (sum < 1e-9) {
		if (!warnState.degenerateResidualWarned) {
			console.warn(
				"speculative: residual distribution degenerate (Σq < 1e-9); falling back to p_target sample",
			);
			warnState.degenerateResidualWarned = true;
		}
		return sampler.sampleFromDistribution(pTarget);
	}
	for (let i = 0; i < n; i++) q[i] /= sum;
	return sampler.sampleFromDistribution(q);
}

/** Options passed to `SpeculativeGenerator.generate`. */
export interface SpeculativeGenerateOptions {
	/** Pre-tokenized prompt token IDs. */
	promptTokenIds: number[];
	/** Loaded target ModelInference. */
	target: ModelInference;
	/** Loaded drafter ModelInference (must share the target's tokenizer). */
	drafter: ModelInference;
	/** Tokenizer for streaming-text decoding. */
	tokenizer: Tokenizer;
	/** User's Sampler (seeded; both drafter and target draws + rejection rolls
	 *  consume from this same instance for determinism). */
	sampler: Sampler;
	/** Generation config (maxTokens, temperature, topK, topP, repetitionPenalty,
	 *  stopTokens). Steering fields must be empty — engagement gate at the
	 *  engine level guarantees this. */
	config: GenerationConfig;
	/** EOS token id for the shared tokenizer. */
	eosTokenId: number;
	/** Draft-burst length K (≥ 2). */
	draftLength: number;
	/** Optional abort signal. Honored at three points per spec step (top of
	 *  loop, after draft burst, mid-draft-burst). */
	signal?: AbortSignal;
}

/**
 * Speculative-decode generator: drafter proposes K tokens, target verifies
 * them in one parallel forward, rejection-sampling accepts a prefix and
 * emits a residual or bonus token.
 *
 * Yields each emitted token id one at a time, returns a `GenerationResult`
 * with `tokens`, `text`, `tokenCount`, `tokensPerSecond`,
 * `timeToFirstTokenMs`, `finishReason` — same shape as
 * `Generator.generate`'s return so the engine adapter can treat both paths
 * symmetrically.
 *
 * After each spec step, rolls drafter and target `nCached` back to the new
 * `pastLen` so the next step's KV state is consistent. On full accept + bonus
 * (emitted = K + 1), both caches stay at `pastLen_before + K` since the bonus
 * token's KV slot will be written by the next step's first drafter forward
 * — same shape as the prefill→decode handoff.
 *
 * Honors `opts.signal` at three points per step: top of decode loop, mid
 * draft burst (after each drafter forward), and post-draft (before verify).
 * On abort the function returns a `GenerationResult` with
 * `finishReason: "aborted"` and the already-yielded `tokenCount`.
 */
export const SpeculativeGenerator = {
	async *generate(
		opts: SpeculativeGenerateOptions,
	): AsyncGenerator<number, GenerationResult> {
		const {
			promptTokenIds,
			target,
			drafter,
			tokenizer,
			sampler,
			config,
			eosTokenId,
			draftLength,
			signal,
		} = opts;
		const K = draftLength;
		if (K < 2) {
			throw new Error(`SpeculativeGenerator: draftLength=${K} must be >= 2`);
		}
		const stopTokens = new Set(config.stopTokens ?? []);
		const startTime = performance.now();
		const warnState = newSpeculativeWarnState();

		// === Prefill on target ===
		const promptPositions = promptTokenIds.map((_, i) => i);
		const targetPrefillLogits = await target.forward(
			new Int32Array(promptTokenIds),
			new Int32Array(promptPositions),
		);

		// Sample first emitted token from the target's prefill last-position
		// logits, with repetition penalty applied first (mirrors
		// Generator.generate's prefill path).
		const recentTokens: number[] = [...promptTokenIds];
		sampler.applyRepetitionPenalty(
			targetPrefillLogits,
			recentTokens.slice(-64),
		);
		const firstDistro = sampler.computeDistribution(targetPrefillLogits);
		let lastEmittedId = sampler.sampleFromDistribution(firstDistro);

		const firstTokenTime = performance.now();
		yield lastEmittedId;
		recentTokens.push(lastEmittedId);
		const allTokens: number[] = [...promptTokenIds, lastEmittedId];
		let generatedCount = 1;
		let finishReason: GenerationFinishReason | undefined;

		// === Prefill on drafter ===
		// Logits discarded — we only need its KV populated to match target's.
		await drafter.forward(
			new Int32Array(promptTokenIds),
			new Int32Array(promptPositions),
		);

		let pastLen = promptTokenIds.length;

		// First-token early stop: if the prefill sample is EOS / a stop token,
		// short-circuit before running the drafter prefill / decode loop.
		// Mirrors the corresponding check in Generator.generate.
		if (lastEmittedId === eosTokenId || stopTokens.has(lastEmittedId)) {
			finishReason = "stop-token";
		}

		// === Decode loop ===
		while (finishReason === undefined) {
			if (signal?.aborted) {
				finishReason = "aborted";
				break;
			}
			if (generatedCount >= config.maxTokens) {
				finishReason = "max-tokens";
				break;
			}

			// Hoist the recent-token window once per spec step. The
			// 64-token slice is identical across the K drafter forwards and
			// the K verify rows; allocating it 2K+1 times per step wastes
			// ~9 array allocations + 9 Set constructions per step at K=4.
			const penaltyWindow = recentTokens.slice(-64);

			// --- Draft burst ---
			const draftTokens: number[] = [];
			const draftDistros: Float32Array[] = [];
			let prev = lastEmittedId;
			let aborted = false;
			for (let k = 0; k < K; k++) {
				if (signal?.aborted) {
					aborted = true;
					break;
				}
				const logits = await drafter.forward(
					new Int32Array([prev]),
					new Int32Array([pastLen + k]),
				);
				sampler.applyRepetitionPenalty(logits, penaltyWindow);
				const distro = sampler.computeDistribution(logits);
				const id = sampler.sampleFromDistribution(distro);
				draftTokens.push(id);
				draftDistros.push(distro);
				prev = id;
			}
			if (aborted) {
				finishReason = "aborted";
				// Drafter ran ahead by `draftTokens.length` cache slots; target
				// is still at pastLen (verify hasn't run). Roll drafter back to
				// match.
				drafter.truncateKVCache(target.cachedTokenCount);
				break;
			}
			if (signal?.aborted) {
				finishReason = "aborted";
				// Drafter is at pastLen + K; target is still at pastLen.
				drafter.truncateKVCache(target.cachedTokenCount);
				break;
			}

			// --- Verify on target ---
			// Inputs at positions pastLen..pastLen+K-1 are
			// [lastEmittedId, draftIds[0..K-2]] — the K tokens that go into
			// the target's KV at those positions. The last drafted token
			// (draftIds[K-1]) is what we want the target to predict at
			// position pastLen+K.
			const verifyInputs = new Int32Array(K);
			verifyInputs[0] = lastEmittedId;
			for (let k = 1; k < K; k++) verifyInputs[k] = draftTokens[k - 1];
			const verifyPositions = new Int32Array(K);
			for (let k = 0; k < K; k++) verifyPositions[k] = pastLen + k;
			const targetLogitsAll = await target.forwardVerify(
				verifyInputs,
				verifyPositions,
			);
			const vocab = targetLogitsAll.length / K;
			const targetDistros: Float32Array[] = [];
			for (let k = 0; k < K; k++) {
				const row = targetLogitsAll
					.subarray(k * vocab, (k + 1) * vocab)
					.slice();
				sampler.applyRepetitionPenalty(row, penaltyWindow);
				targetDistros.push(sampler.computeDistribution(row));
			}

			// --- Rejection sampling ---
			const result = acceptPrefix({
				draftTokens,
				draftDistros,
				targetDistros,
				sampler,
				warnState,
				eosTokenId,
				stopTokens,
				maxTokens: config.maxTokens,
				generatedCount,
			});

			// --- Yield accepted prefix + finalSampledId ---
			for (let k = 0; k < result.acceptedCount; k++) {
				const id = draftTokens[k];
				yield id;
				generatedCount++;
				recentTokens.push(id);
				allTokens.push(id);
				lastEmittedId = id;
			}
			if (result.finalSampledId !== null) {
				yield result.finalSampledId;
				generatedCount++;
				recentTokens.push(result.finalSampledId);
				allTokens.push(result.finalSampledId);
				lastEmittedId = result.finalSampledId;
			}

			pastLen +=
				result.acceptedCount + (result.finalSampledId !== null ? 1 : 0);

			// === KV rollback ===
			// On partial accept (emitted ≤ K): drafter and target nCached are at
			// pastLen_before + K; we want them at the new pastLen (= pastLen_before
			// + emitted). Roll back the unaccepted suffix.
			//
			// On full accept + bonus (emitted = K + 1): both nCached are at
			// pastLen_before + K. The bonus token's KV slot will be written on
			// the next step's first drafter forward — same shape as the
			// prefill→decode handoff. min(currentNCached, newPastLen) =
			// min(pastLen_before + K, pastLen_before + K + 1) = pastLen_before + K
			// → no actual decrement on full-accept-with-bonus.
			drafter.truncateKVCache(Math.min(drafter.cachedTokenCount, pastLen));
			target.truncateKVCache(Math.min(target.cachedTokenCount, pastLen));

			if (result.finishReason !== null) {
				finishReason = result.finishReason;
				break;
			}
			if (generatedCount >= config.maxTokens) {
				finishReason = "max-tokens";
				break;
			}
		}

		// === Build result ===
		const decoder = new StreamingDecoder(tokenizer);
		let text = "";
		for (let i = promptTokenIds.length; i < allTokens.length; i++) {
			text += decoder.push(allTokens[i]);
		}
		const decodeMs = performance.now() - firstTokenTime;
		return {
			tokens: allTokens,
			text,
			tokenCount: generatedCount,
			tokensPerSecond:
				generatedCount > 0 ? (generatedCount * 1000) / decodeMs : 0,
			timeToFirstTokenMs: firstTokenTime - startTime,
			finishReason: finishReason ?? "max-tokens",
		};
	},
};
