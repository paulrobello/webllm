import { beforeEach, expect, test } from "bun:test";
import {
	getCustomScorer,
	hasCustomScorer,
	listCustomScorers,
	registerCustomScorer,
} from "../src/evaluation/custom-scorers.js";
import { score } from "../src/evaluation/scorer.js";
import type { EvalTask } from "../src/evaluation/types.js";

const TEST_NAME = "__unit-test-scorer";

beforeEach(() => {
	// Re-register fresh per test — registry is module-global.
	registerCustomScorer(TEST_NAME, () => 0);
});

test("registerCustomScorer / getCustomScorer / hasCustomScorer round-trip", () => {
	registerCustomScorer(TEST_NAME, () => 0.5);
	expect(hasCustomScorer(TEST_NAME)).toBe(true);
	const fn = getCustomScorer(TEST_NAME);
	expect(fn?.("anything", "anything")).toBe(0.5);
});

test("listCustomScorers includes our test name", () => {
	expect(listCustomScorers()).toContain(TEST_NAME);
});

test("score() resolves a custom task via the registry", () => {
	registerCustomScorer(TEST_NAME, (output) =>
		output.includes("banana") ? 1 : 0,
	);
	const task: EvalTask = {
		id: "t",
		dimension: "instruction-following",
		description: "d",
		systemPrompt: "s",
		input: "i",
		expected: "e",
		scoring: { type: "custom", name: TEST_NAME },
		difficulty: "easy",
	};
	expect(score("apple banana cherry", task)).toBe(1);
	expect(score("apple cherry", task)).toBe(0);
});

test("score() returns 0 for an unregistered custom name", () => {
	const task: EvalTask = {
		id: "t",
		dimension: "reasoning",
		description: "d",
		systemPrompt: "s",
		input: "i",
		expected: "e",
		scoring: { type: "custom", name: "__does-not-exist" },
		difficulty: "easy",
	};
	expect(score("whatever", task)).toBe(0);
});

test("shipped scorer-registrations populate the 13 expected names", async () => {
	await import("../eval/tasks/scorer-registrations.js");
	const expected = [
		"rs-012-water-jug",
		"in-002-one-sentence",
		"in-006-numbered-5-items",
		"in-008-avoid-forbidden-words",
		"in-009-photosynthesis-3-bullets",
		"in-011-alice-json",
		"in-012-three-questions-order",
		"emb-003-fast-synonyms",
		"emb-004-hot-antonyms",
		"emb-005-foot-analogy",
		"emb-006-fish-vegetables-grouping",
		"emb-009-puppy-kitten",
		"emb-012-light-sense-disambiguation",
	];
	for (const name of expected) {
		expect(hasCustomScorer(name)).toBe(true);
	}
});

test("shipped scorer rs-012-water-jug gives partial credit correctly", async () => {
	await import("../eval/tasks/scorer-registrations.js");
	const fn = getCustomScorer("rs-012-water-jug");
	expect(fn).toBeDefined();
	if (!fn) return;
	expect(fn("fill the 5, pour, empty, leaving 4", "")).toBe(1);
	expect(fn("fill the 5, pour", "")).toBe(0.5);
	expect(fn("no relevant words", "")).toBe(0);
});
