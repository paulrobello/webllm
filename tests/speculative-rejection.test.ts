import { describe, expect, test } from "bun:test";
import { Sampler } from "../src/inference/sampler.js";
import {
	acceptPrefix,
	newSpeculativeWarnState,
} from "../src/inference/speculative.js";

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
		const warnState = newSpeculativeWarnState();
		let totalAccepted = 0;
		for (let i = 0; i < N; i++) {
			const dist = uniform(8);
			const result = acceptPrefix({
				draftTokens: [0, 1, 2, 3],
				draftDistros: [dist, dist, dist, dist],
				targetDistros: [dist, dist, dist, dist],
				sampler,
				warnState,
				eosTokenId: -1,
				stopTokens: new Set<number>(),
				maxTokens: 100000,
				generatedCount: 0,
			});
			totalAccepted += result.acceptedCount;
		}
		const rate = totalAccepted / (N * K);
		expect(rate).toBeGreaterThan(0.98);
	});

	test("rejects with probability 1 on disjoint support", () => {
		const N = 10000;
		const sampler = new Sampler({ seed: 2 });
		const warnState = newSpeculativeWarnState();
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
				warnState,
				eosTokenId: -1,
				stopTokens: new Set<number>(),
				maxTokens: 100000,
				generatedCount: 0,
			});
			totalAccepted += result.acceptedCount;
			if (result.finalSampledId !== null)
				finalIdCounts[result.finalSampledId]++;
		}
		expect(totalAccepted).toBe(0);
		expect(finalIdCounts[1]).toBe(N);
	});

	test("acceptance rate matches min(1, pT/pD)", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 3 });
		const warnState = newSpeculativeWarnState();
		const draft = new Float32Array([0.8, 0.2, 0, 0]);
		const target = new Float32Array([0.4, 0.6, 0, 0]);
		let accepted = 0;
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0],
				draftDistros: [draft],
				targetDistros: [target],
				sampler,
				warnState,
				eosTokenId: -1,
				stopTokens: new Set<number>(),
				maxTokens: 100000,
				generatedCount: 0,
			});
			accepted += result.acceptedCount;
		}
		const rate = accepted / N;
		expect(Math.abs(rate - 0.5)).toBeLessThan(0.02);
	});

	test("residual distribution is correct on rejection", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 4 });
		const warnState = newSpeculativeWarnState();
		const draft = onehot(4, 0);
		const target = new Float32Array([0, 0.5, 0.5, 0]);
		const counts = [0, 0, 0, 0];
		for (let i = 0; i < N; i++) {
			const result = acceptPrefix({
				draftTokens: [0],
				draftDistros: [draft],
				targetDistros: [target],
				sampler,
				warnState,
				eosTokenId: -1,
				stopTokens: new Set<number>(),
				maxTokens: 100000,
				generatedCount: 0,
			});
			if (result.finalSampledId !== null) counts[result.finalSampledId]++;
		}
		expect(counts[0]).toBe(0);
		expect(counts[3]).toBe(0);
		expect(Math.abs(counts[1] / N - 0.5)).toBeLessThan(0.02);
		expect(Math.abs(counts[2] / N - 0.5)).toBeLessThan(0.02);
	});

	test("bonus token sampled from target row K-1 on full accept", () => {
		const N = 20000;
		const sampler = new Sampler({ seed: 5 });
		const warnState = newSpeculativeWarnState();
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
				warnState,
				eosTokenId: -1,
				stopTokens: new Set<number>(),
				maxTokens: 100000,
				generatedCount: 0,
			});
			expect(result.acceptedCount).toBe(2);
			if (result.finalSampledId !== null) counts[result.finalSampledId]++;
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
			warnState: newSpeculativeWarnState(),
			eosTokenId: -1,
			stopTokens: new Set<number>(),
			maxTokens: 100000,
			generatedCount: 0,
		});
		const r2 = acceptPrefix({
			draftTokens: [0, 1],
			draftDistros: [distA, distA],
			targetDistros: [distB, distB],
			sampler: samplerB,
			warnState: newSpeculativeWarnState(),
			eosTokenId: -1,
			stopTokens: new Set<number>(),
			maxTokens: 100000,
			generatedCount: 0,
		});
		expect(r1).toEqual(r2);
	});
});

describe("acceptPrefix — finish conditions", () => {
	const dist = uniform(8);

	test("EOS at draftIds[1] truncates accepted to 2 with stop-token", () => {
		const sampler = new Sampler({ seed: 100 });
		const result = acceptPrefix({
			draftTokens: [3, 7, 5, 4],
			draftDistros: [dist, dist, dist, dist],
			targetDistros: [dist, dist, dist, dist],
			sampler,
			warnState: newSpeculativeWarnState(),
			eosTokenId: 7,
			stopTokens: new Set(),
			maxTokens: 100,
			generatedCount: 0,
		});
		expect(result.acceptedCount).toBe(2);
		expect(result.finalSampledId).toBe(null);
		expect(result.finishReason).toBe("stop-token");
	});

	test("stop-token in custom set truncates", () => {
		const sampler = new Sampler({ seed: 101 });
		const result = acceptPrefix({
			draftTokens: [3, 6, 5, 4],
			draftDistros: [dist, dist, dist, dist],
			targetDistros: [dist, dist, dist, dist],
			sampler,
			warnState: newSpeculativeWarnState(),
			eosTokenId: 0,
			stopTokens: new Set([6]),
			maxTokens: 100,
			generatedCount: 0,
		});
		expect(result.acceptedCount).toBe(2);
		expect(result.finishReason).toBe("stop-token");
	});

	test("maxTokens limit truncates accepted prefix", () => {
		const sampler = new Sampler({ seed: 102 });
		const result = acceptPrefix({
			draftTokens: [3, 7, 5, 4],
			draftDistros: [dist, dist, dist, dist],
			targetDistros: [dist, dist, dist, dist],
			sampler,
			warnState: newSpeculativeWarnState(),
			eosTokenId: -1,
			stopTokens: new Set(),
			maxTokens: 12,
			generatedCount: 10,
		});
		expect(result.acceptedCount).toBe(2);
		expect(result.finalSampledId).toBe(null);
		expect(result.finishReason).toBe("max-tokens");
	});

	test("EOS as residual sample sets stop-token with non-null finalId", () => {
		const sampler = new Sampler({ seed: 103 });
		const draft = onehot(8, 0);
		const target = onehot(8, 7);
		const result = acceptPrefix({
			draftTokens: [0],
			draftDistros: [draft],
			targetDistros: [target],
			sampler,
			warnState: newSpeculativeWarnState(),
			eosTokenId: 7,
			stopTokens: new Set(),
			maxTokens: 100,
			generatedCount: 0,
		});
		expect(result.acceptedCount).toBe(0);
		expect(result.finalSampledId).toBe(7);
		expect(result.finishReason).toBe("stop-token");
	});

	test("bonus token EOS sets stop-token after full accept", () => {
		const sampler = new Sampler({ seed: 104 });
		const matched = uniform(8);
		// targetDistros[1] gives draft id 1 enough mass to always accept
		// (pT=0.4, pD=0.125, r=1) while concentrating bonus mass on EOS=5.
		const bonusFavorEos = new Float32Array([0, 0.4, 0, 0, 0, 0.6, 0, 0]);
		const result = acceptPrefix({
			draftTokens: [0, 1],
			draftDistros: [matched, matched],
			targetDistros: [matched, bonusFavorEos],
			sampler,
			warnState: newSpeculativeWarnState(),
			eosTokenId: 5,
			stopTokens: new Set(),
			maxTokens: 100,
			generatedCount: 0,
		});
		expect(result.acceptedCount).toBe(2);
		expect(result.finalSampledId).toBe(5);
		expect(result.finishReason).toBe("stop-token");
	});
});
