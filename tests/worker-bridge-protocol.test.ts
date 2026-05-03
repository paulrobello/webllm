import { describe, expect, test } from "bun:test";
import {
	makeRequestId,
	makeStreamId,
	type ProxyToWorker,
	type SerializedError,
	type WorkerToProxy,
} from "../src/core/worker-bridge.js";

describe("worker-bridge", () => {
	test("makeRequestId returns monotonically increasing positive integers", () => {
		const a = makeRequestId();
		const b = makeRequestId();
		const c = makeRequestId();
		expect(a).toBeGreaterThan(0);
		expect(b).toBeGreaterThan(a);
		expect(c).toBeGreaterThan(b);
	});

	test("makeStreamId returns monotonically increasing positive integers, separate counter from request ids", () => {
		const a = makeStreamId();
		const b = makeStreamId();
		expect(a).toBeGreaterThan(0);
		expect(b).toBeGreaterThan(a);
	});

	test("ProxyToWorker variants survive structuredClone", () => {
		const samples: ProxyToWorker[] = [
			{ type: "init", id: 1, config: { memoryBudget: 8e9 } },
			{ type: "method-call", id: 2, name: "embed", args: ["m1", "hello"] },
			{
				type: "stream-start",
				streamId: 3,
				name: "chatCompletion",
				args: ["m1", [{ role: "user", content: "hi" }], {}],
			},
			{ type: "stream-cancel", streamId: 3 },
			{ type: "dispose", id: 4 },
		];
		for (const s of samples) {
			const cloned = structuredClone(s);
			expect(cloned).toEqual(s);
		}
	});

	test("WorkerToProxy variants survive structuredClone", () => {
		const err: SerializedError = {
			code: "MODEL_NOT_FOUND",
			message: "x",
			modelId: "m1",
		};
		const samples: WorkerToProxy[] = [
			{ type: "init-done", id: 1 },
			{ type: "method-result", id: 2, value: { id: "h1" } },
			{ type: "method-error", id: 2, error: err },
			{
				type: "stream-chunk",
				streamId: 3,
				chunk: { text: "hi", tokenId: 42, done: false },
			},
			{
				type: "stream-chunks",
				streamId: 3,
				chunks: [
					{ text: "hi", tokenId: 42, done: false },
					{ text: " there", tokenId: 43, done: false },
				],
			},
			{ type: "stream-done", streamId: 3 },
			{ type: "stream-error", streamId: 3, error: err },
			{ type: "log", level: "info", message: "ok" },
		];
		for (const s of samples) {
			const cloned = structuredClone(s);
			expect(cloned).toEqual(s);
		}
	});
});
