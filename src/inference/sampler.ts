/** Optional configuration for constructing a Sampler instance. */
export interface SamplerConfig {
	/** Sampling temperature. 0 = greedy decoding, 1.0 = default. */
	temperature?: number;
	/** Top-K filtering cutoff. 0 = disabled. */
	topK?: number;
	/** Top-P (nucleus) cumulative probability threshold. 1.0 = disabled. */
	topP?: number;
	/** Repetition penalty multiplier applied to recently seen tokens. 1.0 = disabled. */
	repetitionPenalty?: number;
	/** PRNG seed for deterministic sampling. Omit for Math.random. */
	seed?: number;
}

/**
 * Token sampler with temperature, top-k, top-p, and repetition penalty, optional seeded PRNG.
 *
 * Applies a multi-stage filtering pipeline to raw logits: temperature scaling,
 * top-K truncation, top-P (nucleus) filtering, then weighted random sampling.
 */
export class Sampler {
	private temperature: number;
	private topK: number;
	private topP: number;
	private repetitionPenalty: number;
	private rng: () => number;

	/**
	 * @param config - Sampler configuration. All fields optional; defaults to greedy-friendly values.
	 */
	constructor(config: SamplerConfig = {}) {
		this.temperature = config.temperature ?? 1.0;
		this.topK = config.topK ?? 0;
		this.topP = config.topP ?? 1.0;
		this.repetitionPenalty = config.repetitionPenalty ?? 1.0;

		if (config.seed !== undefined) {
			let s0 = config.seed;
			let s1 = config.seed ^ 0xdeadbeef;
			let s2 = config.seed ^ 0xcafebabe;
			let s3 = config.seed ^ 0x12345678;
			this.rng = () => {
				const result = Math.imul(s0, 5) ^ (s0 << 7) ^ (s0 << 13);
				const t = s1 << 9;
				s2 ^= s0;
				s3 ^= s1;
				s1 ^= s2;
				s0 ^= s3;
				s2 ^= t;
				s3 = (s3 << 11) | (s3 >>> 21);
				return (result >>> 0) / 4294967296;
			};
		} else {
			this.rng = Math.random;
		}
	}

	/**
	 * Run the full sampling pipeline on raw logits and return a token index.
	 *
	 * @param logits - Raw unnormalized log probabilities from the model.
	 * @returns Sampled token index.
	 */
	sample(logits: Float32Array): number {
		if (this.temperature === 0) return argmax(logits);
		const scaled = this.applyTemperature(logits);
		const filtered = this.applyTopK(scaled);
		const topPFiltered = this.applyTopP(filtered);
		const probs = softmax(topPFiltered);
		let r = this.rng();
		for (let i = 0; i < probs.length; i++) {
			r -= probs[i];
			if (r <= 0) return i;
		}
		return probs.length - 1;
	}

	/**
	 * Scale logits by 1/temperature.
	 *
	 * @param logits - Raw logits.
	 * @returns Temperature-scaled copy of logits.
	 */
	applyTemperature(logits: Float32Array): Float32Array {
		if (this.temperature === 1.0) return new Float32Array(logits);
		const result = new Float32Array(logits.length);
		for (let i = 0; i < logits.length; i++)
			result[i] = logits[i] / this.temperature;
		return result;
	}

	/**
	 * Retain only the top-K highest logits, setting the rest to -Infinity.
	 *
	 * @param logits - Input logits.
	 * @returns Filtered copy with non-top-K entries set to -Infinity.
	 */
	applyTopK(logits: Float32Array): Float32Array {
		if (this.topK === 0 || this.topK >= logits.length)
			return new Float32Array(logits);
		const indices = Array.from({ length: logits.length }, (_, i) => i);
		indices.sort((a, b) => logits[b] - logits[a]);
		const result = new Float32Array(logits.length).fill(-Infinity);
		for (let i = 0; i < this.topK; i++) result[indices[i]] = logits[indices[i]];
		return result;
	}

	/**
	 * Retain the smallest set of top logits whose cumulative probability >= topP.
	 *
	 * @param logits - Input logits.
	 * @returns Filtered copy with excluded entries set to -Infinity.
	 */
	applyTopP(logits: Float32Array): Float32Array {
		if (this.topP >= 1.0) return new Float32Array(logits);
		const probs = softmax(logits);
		const indices = Array.from({ length: probs.length }, (_, i) => i);
		indices.sort((a, b) => probs[b] - probs[a]);
		let cumulative = 0;
		const result = new Float32Array(logits.length).fill(-Infinity);
		for (const idx of indices) {
			cumulative += probs[idx];
			result[idx] = logits[idx];
			if (cumulative >= this.topP) break;
		}
		return result;
	}

	/**
	 * Penalize logits for recently seen tokens to reduce repetition.
	 * Mutates the input array in place.
	 *
	 * @param logits - Logits array to mutate.
	 * @param recentTokens - Token IDs that should be penalized.
	 */
	applyRepetitionPenalty(logits: Float32Array, recentTokens: number[]): void {
		if (this.repetitionPenalty === 1.0) return;
		const seen = new Set(recentTokens);
		for (const idx of seen) {
			if (idx < 0 || idx >= logits.length) continue;
			if (logits[idx] > 0) logits[idx] /= this.repetitionPenalty;
			else logits[idx] *= this.repetitionPenalty;
		}
	}
}

/**
 * Return the index of the maximum value in a Float32Array.
 *
 * @param arr - Input array.
 * @returns Index of the largest element.
 */
function argmax(arr: Float32Array): number {
	let maxIdx = 0;
	let maxVal = arr[0];
	for (let i = 1; i < arr.length; i++) {
		if (arr[i] > maxVal) {
			maxVal = arr[i];
			maxIdx = i;
		}
	}
	return maxIdx;
}

/**
 * Compute softmax probabilities from logits in a numerically stable way.
 *
 * @param logits - Raw logits.
 * @returns Probability distribution (same length, sums to 1).
 */
function softmax(logits: Float32Array): Float32Array {
	const max = Math.max(...logits);
	const exps = new Float32Array(logits.length);
	let sum = 0;
	for (let i = 0; i < logits.length; i++) {
		exps[i] = Math.exp(logits[i] - max);
		sum += exps[i];
	}
	const probs = new Float32Array(logits.length);
	for (let i = 0; i < logits.length; i++) probs[i] = exps[i] / sum;
	return probs;
}
