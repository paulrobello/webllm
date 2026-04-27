import { describe, expect, test } from "bun:test";
import { Sampler } from "../src/inference/sampler.js";

describe("Sampler", () => {
	test("greedy sampling picks highest logit", () => {
		const sampler = new Sampler({ temperature: 0 });
		const logits = new Float32Array([0.1, 0.5, 0.9, 0.3]);
		expect(sampler.sample(logits)).toBe(2);
	});

	test("temperature scaling works", () => {
		const sampler = new Sampler({ temperature: 2.0 });
		const logits = new Float32Array([1.0, 2.0]);
		const scaled = sampler.applyTemperature(logits);
		expect(scaled[0]).toBeCloseTo(0.5, 5);
		expect(scaled[1]).toBeCloseTo(1.0, 5);
	});

	test("top-k filters to k highest logits", () => {
		const sampler = new Sampler({ temperature: 1.0, topK: 2 });
		const logits = new Float32Array([0.1, 0.5, 0.9, 0.3]);
		const filtered = sampler.applyTopK(logits);
		expect(filtered[0]).toBe(-Infinity);
		expect(filtered[1]).toBeCloseTo(0.5, 5);
		expect(filtered[2]).toBeCloseTo(0.9, 5);
		expect(filtered[3]).toBe(-Infinity);
	});

	test("top-p filters by cumulative probability", () => {
		const sampler = new Sampler({ temperature: 1.0, topP: 0.5 });
		const logits = new Float32Array([0.1, 0.2, 3.0, 0.1]);
		const filtered = sampler.applyTopP(logits);
		expect(filtered[2]).toBe(3.0);
	});

	test("repetition penalty penalizes repeated tokens", () => {
		const sampler = new Sampler({
			temperature: 1.0,
			repetitionPenalty: 1.5,
		});
		const logits = new Float32Array([1.0, 2.0, 3.0]);
		sampler.applyRepetitionPenalty(logits, [2]);
		expect(logits[2]).toBeCloseTo(2.0, 5);
	});

	test("deterministic with fixed seed", () => {
		const sampler1 = new Sampler({ temperature: 1.0, seed: 42 });
		const sampler2 = new Sampler({ temperature: 1.0, seed: 42 });
		const logits = new Float32Array([0.1, 0.5, 0.9, 0.3, 0.7]);
		const results1: number[] = [];
		const results2: number[] = [];
		for (let i = 0; i < 100; i++) {
			results1.push(sampler1.sample(new Float32Array(logits)));
			results2.push(sampler2.sample(new Float32Array(logits)));
		}
		expect(results1).toEqual(results2);
	});
});

describe("Sampler.computeDistribution", () => {
	test("returns one-hot at argmax for temperature=0", () => {
		const sampler = new Sampler({ temperature: 0 });
		const logits = new Float32Array([1.0, 5.0, 3.0, 2.0]);
		const probs = sampler.computeDistribution(logits);
		expect(probs[1]).toBe(1.0);
		expect(probs[0]).toBe(0);
		expect(probs[2]).toBe(0);
		expect(probs[3]).toBe(0);
	});

	test("temperature=1 produces softmax over all logits", () => {
		const sampler = new Sampler({ temperature: 1.0 });
		const logits = new Float32Array([1.0, 1.0, 1.0, 1.0]);
		const probs = sampler.computeDistribution(logits);
		for (const p of probs) expect(p).toBeCloseTo(0.25, 3);
		const sum = probs.reduce((a, b) => a + b, 0);
		expect(sum).toBeCloseTo(1.0, 4);
	});

	test("topK filters all but the top K", () => {
		const sampler = new Sampler({ temperature: 1.0, topK: 2 });
		const logits = new Float32Array([1.0, 5.0, 3.0, 2.0]);
		const probs = sampler.computeDistribution(logits);
		expect(probs[0]).toBe(0);
		expect(probs[3]).toBe(0);
		expect(probs[1] + probs[2]).toBeCloseTo(1.0, 4);
		expect(probs[1]).toBeGreaterThan(probs[2]);
	});

	test("sum of probs is 1.0 with topK + topP both active", () => {
		const sampler = new Sampler({ temperature: 0.7, topK: 3, topP: 0.9 });
		const logits = new Float32Array([1.0, 5.0, 3.0, 2.0, 0.5]);
		const probs = sampler.computeDistribution(logits);
		const sum = probs.reduce((a, b) => a + b, 0);
		expect(sum).toBeCloseTo(1.0, 4);
	});
});

describe("Sampler.sampleFromDistribution", () => {
	test("draws id 0 when probs[0]=1", () => {
		const sampler = new Sampler({ seed: 42 });
		const probs = new Float32Array([1.0, 0.0, 0.0]);
		for (let i = 0; i < 10; i++) {
			expect(sampler.sampleFromDistribution(probs)).toBe(0);
		}
	});

	test("draws ids from distro at expected frequency", () => {
		const sampler = new Sampler({ seed: 12345 });
		const probs = new Float32Array([0.6, 0.4, 0.0]);
		const counts = [0, 0, 0];
		const N = 10000;
		for (let i = 0; i < N; i++) {
			counts[sampler.sampleFromDistribution(probs)]++;
		}
		expect(counts[0] / N).toBeCloseTo(0.6, 1);
		expect(counts[1] / N).toBeCloseTo(0.4, 1);
		expect(counts[2]).toBe(0);
	});

	test("rand() is deterministic across runs given same seed", () => {
		const a = new Sampler({ seed: 999 });
		const b = new Sampler({ seed: 999 });
		for (let i = 0; i < 5; i++) {
			expect(a.rand()).toBe(b.rand());
		}
	});
});
