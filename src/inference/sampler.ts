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
	temperature: number;
	topK: number;
	topP: number;
	repetitionPenalty: number;
	rng: () => number;

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

	/** Whether greedy mode (temperature=0) is active. */
	get isGreedy(): boolean {
		return this.temperature === 0;
	}

	/** Whether no repetition penalty is applied. */
	get noPenalty(): boolean {
		return this.repetitionPenalty === 1.0;
	}

	/**
	 * Run the full sampling pipeline on raw logits and return a token index.
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
	 * Sample from a GPU-reduced top-K set (indices + values).
	 * Operates only on the k candidates — skips full-vocab sort/softmax.
	 */
	sampleFromTopK(
		indices: Int32Array,
		values: Float32Array,
		recentTokens: number[],
	): number {
		if (this.repetitionPenalty !== 1.0) {
			const recentSet = new Set(recentTokens);
			for (let i = 0; i < indices.length; i++) {
				if (recentSet.has(indices[i])) {
					values[i] =
						values[i] > 0
							? values[i] / this.repetitionPenalty
							: values[i] * this.repetitionPenalty;
				}
			}
		}

		if (this.temperature === 0) {
			let maxIdx = 0;
			for (let i = 1; i < values.length; i++) {
				if (values[i] > values[maxIdx]) maxIdx = i;
			}
			return indices[maxIdx];
		}

		const scaled = new Float32Array(values.length);
		for (let i = 0; i < values.length; i++)
			scaled[i] = values[i] / this.temperature;

		if (this.topP < 1.0) {
			const probs = softmax(scaled);
			const order = Array.from({ length: probs.length }, (_, i) => i);
			order.sort((a, b) => probs[b] - probs[a]);
			let cumulative = 0;
			const filtered = new Float32Array(values.length).fill(-Infinity);
			for (const idx of order) {
				cumulative += probs[idx];
				filtered[idx] = scaled[idx];
				if (cumulative >= this.topP) break;
			}
			return sampleProbs(softmax(filtered), indices, this.rng);
		}

		return sampleProbs(softmax(scaled), indices, this.rng);
	}

	/**
	 * Scale logits by 1/temperature.
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

function softmax(logits: Float32Array): Float32Array {
	let max = logits[0] ?? -Infinity;
	for (let i = 1; i < logits.length; i++) {
		if (logits[i] > max) max = logits[i];
	}
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

/** Sample from probs using indices as the token ID mapping. */
function sampleProbs(
	probs: Float32Array,
	indices: Int32Array,
	rng: () => number,
): number {
	let r = rng();
	for (let i = 0; i < probs.length; i++) {
		r -= probs[i];
		if (r <= 0) return indices[i];
	}
	return indices[probs.length - 1];
}
