import type { GenerationFinishReason } from "./generation.js";
import type { Sampler } from "./sampler.js";

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
