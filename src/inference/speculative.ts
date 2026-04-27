import type { Sampler } from "./sampler.js";

/** Result of `acceptPrefix`. */
export interface AcceptPrefixResult {
	/** Number of drafted tokens accepted (0..K). */
	acceptedCount: number;
	/**
	 * The next visible token to emit beyond the accepted prefix:
	 * - On rejection: residual-distribution sample.
	 * - On full accept (acceptedCount === K): bonus token from target row K-1.
	 * Always non-null in this initial version; Task 5 adds the
	 * finish-mid-burst case where this is set to null.
	 */
	finalSampledId: number;
	/**
	 * Set when the accept loop terminated due to a finish condition. Always
	 * null in this initial version; populated in Task 5.
	 */
	finishReason: null;
}

interface AcceptPrefixArgs {
	draftTokens: number[];
	draftDistros: Float32Array[];
	targetDistros: Float32Array[];
	sampler: Sampler;
}

/**
 * Standard Leviathan-et-al rejection sampling for speculative decoding.
 *
 * For each drafted token id at position k, computes
 * `r = min(1, p_target[id] / p_draft[id])` and accepts iff a fresh uniform draw
 * `u < r`. On first rejection, samples a final visible token from the residual
 * distribution `q[i] = max(0, p_target[i] - p_draft[i])` normalized. If all K
 * accept, samples a bonus token from `targetDistros[K-1]` (free token from the
 * parallel verify).
 *
 * Pure function. Reads RNG via `sampler.rand()`. Returns
 * `(acceptedCount, finalSampledId)` for the caller to yield in order.
 */
export function acceptPrefix(args: AcceptPrefixArgs): AcceptPrefixResult {
	const { draftTokens, draftDistros, targetDistros, sampler } = args;
	const K = draftTokens.length;
	if (K === 0) {
		throw new Error("acceptPrefix: K=0 is invalid");
	}
	if (draftDistros.length !== K || targetDistros.length !== K) {
		throw new Error(
			`acceptPrefix: distro count mismatch (K=${K}, draft=${draftDistros.length}, target=${targetDistros.length})`,
		);
	}

	let acceptedCount = 0;
	for (let k = 0; k < K; k++) {
		const id = draftTokens[k];
		const pT = targetDistros[k][id] ?? 0;
		const pD = draftDistros[k][id] ?? 0;
		const r = pD > 0 ? Math.min(1, pT / pD) : 0;
		const u = sampler.rand();
		if (u < r) {
			acceptedCount++;
			continue;
		}
		const finalSampledId = sampleResidual(
			targetDistros[k],
			draftDistros[k],
			sampler,
		);
		return { acceptedCount, finalSampledId, finishReason: null };
	}
	const finalSampledId = sampler.sampleFromDistribution(targetDistros[K - 1]);
	return { acceptedCount, finalSampledId, finishReason: null };
}

/**
 * Sample from the residual distribution `q[i] = max(0, p_target[i] - p_draft[i])`
 * normalized to sum to 1. If the residual is degenerate (Σq < 1e-9), fall back
 * to sampling from `p_target` and warn once per process.
 */
let degenerateResidualWarned = false;
export function _resetDegenerateResidualWarning(): void {
	degenerateResidualWarned = false;
}

function sampleResidual(
	pTarget: Float32Array,
	pDraft: Float32Array,
	sampler: Sampler,
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
		if (!degenerateResidualWarned) {
			console.warn(
				"speculative: residual distribution degenerate (Σq < 1e-9); falling back to p_target sample",
			);
			degenerateResidualWarned = true;
		}
		return sampler.sampleFromDistribution(pTarget);
	}
	for (let i = 0; i < n; i++) q[i] /= sum;
	return sampler.sampleFromDistribution(q);
}
