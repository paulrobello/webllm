import { describe, expect, test } from "bun:test";
import {
	InferenceSession,
	type InferenceSessionConfig,
} from "../src/models/inference-session.js";

const BASE_SESSION_CONFIG: InferenceSessionConfig = {
	maxTokens: 50,
	temperature: 0.7,
	topK: 40,
	topP: 0.9,
	repetitionPenalty: 1.1,
	contextOverflowPolicy: "stop",
};

describe("InferenceSession", () => {
	test("starts at position 0", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		expect(session.currentPosition).toBe(0);
	});

	test("advance moves position", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		session.advance(5);
		expect(session.currentPosition).toBe(5);
	});

	test("pushToken adds to history", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		session.pushToken(42);
		session.pushToken(43);
		expect(session.tokens).toEqual([42, 43]);
	});

	test("shouldStop returns true for EOS", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		expect(session.shouldStop(2, 2)).toBe(true);
	});

	test("shouldStop returns true when maxTokens reached", () => {
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 2 },
			0,
		);
		session.pushToken(1);
		session.pushToken(2);
		expect(session.shouldStop(3, 99)).toBe(true);
	});

	test("shouldStop returns false during normal generation", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		expect(session.shouldStop(42, 2)).toBe(false);
	});

	test("reset clears state", () => {
		const session = new InferenceSession(BASE_SESSION_CONFIG, 0);
		session.advance(10);
		session.pushToken(1);
		session.reset();
		expect(session.currentPosition).toBe(0);
		expect(session.tokens).toEqual([]);
	});

	test("isFull when exceeding context", () => {
		const session = new InferenceSession(
			{ ...BASE_SESSION_CONFIG, maxTokens: 3 },
			0,
		);
		session.advance(3);
		expect(session.isFull).toBe(true);
	});
});
