import { afterEach, describe, expect, test } from "bun:test";
import { Sampler } from "../src/inference/sampler.js";
import {
	_resetDegenerateResidualWarning,
	acceptPrefix,
} from "../src/inference/speculative.js";

afterEach(() => _resetDegenerateResidualWarning());

function uniform(n: number): Float32Array {
	const arr = new Float32Array(n);
	for (let i = 0; i < n; i++) arr[i] = 1 / n;
	return arr;
}

function onehot(n: number, idx: number): Float32Array {
	const arr = new Float32Array(n);
	arr[idx] = 1.0;
	return arr;
}

describe("acceptPrefix — core math", () => {
	test("accepts all K when p_target == p_draft", () => {
		const N = 10000;
		const K = 4;
		const sampler = new Sampler({ seed: 1 });
		let totalAccepted = 0;
		for (let i = 0; i < N; i++) {
			const dist = uniform(8);
			const result = acceptPrefix({
				draftTokens: [0, 1, 2, 3],
				draftDistros: [dist, dist, dist, dist],
				targetDistros: [dist, dist, dist, dist],
				sampler,
			});
			totalAccepted += result.acceptedCount;
		}
		const rate = totalAccepted / (N * K);
		expect(rate).toBeGreaterThan(0.98);
	});

	test("rejects with probability 1 on disjoint support", () => {
		const N = 10000;
		const sampler = new Sampler({ seed: 2 });
		const draft = onehot(4, 0);
		const target = onehot(4, 1);
		let totalAccepted = 0;
		const finalIdCounts = [0, 0, 0, 0];
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0],
				draftDistros: [draft],
				targetDistros: [target],
				sampler,
			});
			totalAccepted += result.acceptedCount;
			finalIdCounts[result.finalSampledId]++;
		}
		expect(totalAccepted).toBe(0);
		expect(finalIdCounts[1]).toBe(N);
	});

	test("acceptance rate matches min(1, pT/pD)", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 3 });
		const draft = new Float32Array([0.8, 0.2, 0, 0]);
		const target = new Float32Array([0.4, 0.6, 0, 0]);
		let accepted = 0;
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0],
				draftDistros: [draft],
				targetDistros: [target],
				sampler,
			});
			accepted += result.acceptedCount;
		}
		const rate = accepted / N;
		expect(Math.abs(rate - 0.5)).toBeLessThan(0.02);
	});

	test("residual distribution is correct on rejection", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 4 });
		const draft = onehot(4, 0);
		const target = new Float32Array([0, 0.5, 0.5, 0]);
		const counts = [0, 0, 0, 0];
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0],
				draftDistros: [draft],
				targetDistros: [target],
				sampler,
			});
			counts[result.finalSampledId]++;
		}
		expect(counts[0]).toBe(0);
		expect(counts[3]).toBe(0);
		expect(Math.abs(counts[1] / N - 0.5)).toBeLessThan(0.02);
		expect(Math.abs(counts[2] / N - 0.5)).toBeLessThan(0.02);
	});

	test("bonus token sampled from target row K-1 on full accept", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 5 });
		const matched = uniform(4);
		// bonusTarget[3] = 0.4 > matched[3] = 0.25, so drafting token 3 at the
		// last position guarantees acceptance (r = min(1, 0.4/0.25) = 1) while
		// leaving the bonus distribution as [0.1, 0.2, 0.3, 0.4].
		const bonusTarget = new Float32Array([0.1, 0.2, 0.3, 0.4]);
		const counts = [0, 0, 0, 0];
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0, 3],
				draftDistros: [matched, matched],
				targetDistros: [matched, bonusTarget],
				sampler,
			});
			expect(result.acceptedCount).toBe(2);
			counts[result.finalSampledId]++;
		}
		expect(Math.abs(counts[0] / N - 0.1)).toBeLessThan(0.02);
		expect(Math.abs(counts[1] / N - 0.2)).toBeLessThan(0.02);
		expect(Math.abs(counts[2] / N - 0.3)).toBeLessThan(0.02);
		expect(Math.abs(counts[3] / N - 0.4)).toBeLessThan(0.02);
	});

	test("seed determinism", () => {
		const distA = uniform(4);
		const distB = new Float32Array([0.5, 0.3, 0.1, 0.1]);
		const samplerA = new Sampler({ seed: 7 });
		const samplerB = new Sampler({ seed: 7 });
		const r1 = acceptPrefix({
			draftTokens: [0, 1],
			draftDistros: [distA, distA],
			targetDistros: [distB, distB],
			sampler: samplerA,
		});
		const r2 = acceptPrefix({
			draftTokens: [0, 1],
			draftDistros: [distA, distA],
			targetDistros: [distB, distB],
			sampler: samplerB,
		});
		expect(r1).toEqual(r2);
	});
});
