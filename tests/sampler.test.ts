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
