import { describe, expect, test } from "bun:test";
import {
	scoreCosineSimilarity,
	scoreCosineSimilarityDetails,
} from "../src/evaluation/scorer.js";

test("identical vectors score 1.0", () => {
	const v = new Float32Array([1, 2, 3, 4]);
	expect(scoreCosineSimilarity(v, v)).toBeCloseTo(1.0, 5);
});

test("opposite vectors score 0.0", () => {
	const a = new Float32Array([1, 0, 0]);
	const b = new Float32Array([-1, 0, 0]);
	expect(scoreCosineSimilarity(a, b)).toBeCloseTo(0.0, 5);
});

test("orthogonal vectors score 0.5", () => {
	const a = new Float32Array([1, 0]);
	const b = new Float32Array([0, 1]);
	expect(scoreCosineSimilarity(a, b)).toBeCloseTo(0.5, 5);
});

test("parallel vectors of different magnitudes score 1.0", () => {
	const a = new Float32Array([1, 2, 3]);
	const b = new Float32Array([2, 4, 6]);
	expect(scoreCosineSimilarity(a, b)).toBeCloseTo(1.0, 5);
});

test("zero vector scores 0", () => {
	const a = new Float32Array([0, 0, 0]);
	const b = new Float32Array([1, 2, 3]);
	expect(scoreCosineSimilarity(a, b)).toBe(0);
});

test("length mismatch throws", () => {
	const a = new Float32Array([1, 2, 3]);
	const b = new Float32Array([1, 2]);
	expect(() => scoreCosineSimilarity(a, b)).toThrow(/length mismatch/);
});

test("score stays within [0, 1] for floating-point noise", () => {
	// Near-identical vectors that could overshoot cosine=1 via FP error
	const a = new Float32Array(128);
	const b = new Float32Array(128);
	for (let i = 0; i < 128; i++) {
		a[i] = Math.sin(i);
		b[i] = Math.sin(i) + 1e-9 * Math.cos(i);
	}
	const s = scoreCosineSimilarity(a, b);
	expect(s).toBeGreaterThanOrEqual(0);
	expect(s).toBeLessThanOrEqual(1);
});

describe("scoreCosineSimilarityDetails", () => {
	test("returns raw cosine + remapped score", () => {
		const a = new Float32Array([1, 0, 0]);
		const b = new Float32Array([0, 1, 0]);
		const r = scoreCosineSimilarityDetails(a, b);
		expect(r.cosine).toBeCloseTo(0);
		expect(r.score).toBeCloseTo(0.5);
	});

	test("identical vectors return cosine=1, score=1", () => {
		const a = new Float32Array([0.6, 0.8]);
		const b = new Float32Array([0.6, 0.8]);
		const r = scoreCosineSimilarityDetails(a, b);
		expect(r.cosine).toBeCloseTo(1);
		expect(r.score).toBeCloseTo(1);
	});

	test("opposite vectors return cosine=-1, score=0", () => {
		const a = new Float32Array([1, 0]);
		const b = new Float32Array([-1, 0]);
		const r = scoreCosineSimilarityDetails(a, b);
		expect(r.cosine).toBeCloseTo(-1);
		expect(r.score).toBeCloseTo(0);
	});

	test("zero-norm vectors return cosine=0, score=0", () => {
		const a = new Float32Array([0, 0]);
		const b = new Float32Array([1, 1]);
		const r = scoreCosineSimilarityDetails(a, b);
		expect(r.cosine).toBe(0);
		expect(r.score).toBe(0);
	});

	test("throws on length mismatch", () => {
		expect(() =>
			scoreCosineSimilarityDetails(
				new Float32Array([1]),
				new Float32Array([1, 2]),
			),
		).toThrow(/length mismatch/);
	});
});
