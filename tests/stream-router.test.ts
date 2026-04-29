import { describe, expect, test } from "bun:test";
import { StreamRouter } from "../src/inference/stream-router.js";

describe("StreamRouter", () => {
	test("emits tokens to registered consumer", async () => {
		const router = new StreamRouter<string>();
		const consumer = router.createConsumer("test");
		router.emit("test", "Hello");
		router.emit("test", " ");
		router.emit("test", "world");
		router.close("test");
		const tokens: string[] = [];
		for await (const token of consumer) tokens.push(token);
		expect(tokens).toEqual(["Hello", " ", "world"]);
	});

	test("supports multiple consumers", async () => {
		const router = new StreamRouter<string>();
		const consumer1 = router.createConsumer("c1");
		const consumer2 = router.createConsumer("c2");
		router.emit("c1", "a");
		router.emit("c2", "b");
		router.close("c1");
		router.close("c2");
		const tokens1: string[] = [];
		for await (const token of consumer1) tokens1.push(token);
		const tokens2: string[] = [];
		for await (const token of consumer2) tokens2.push(token);
		expect(tokens1).toEqual(["a"]);
		expect(tokens2).toEqual(["b"]);
	});

	test("interrupt cancels stream", async () => {
		const router = new StreamRouter<string>();
		const consumer = router.createConsumer("test");
		router.emit("test", "start");
		router.interrupt("test");
		const tokens: string[] = [];
		for await (const token of consumer) tokens.push(token);
		expect(tokens).toEqual(["start"]);
	});

	test("removeConsumer cleans up", () => {
		const router = new StreamRouter<string>();
		router.createConsumer("test");
		router.removeConsumer("test");
		expect(router.hasConsumer("test")).toBe(false);
	});

	test("getQueueDepth reports pending items", () => {
		const router = new StreamRouter<string>();
		router.createConsumer("test");
		expect(router.getQueueDepth("test")).toBe(0);
		router.emit("test", "a");
		router.emit("test", "b");
		router.emit("test", "c");
		expect(router.getQueueDepth("test")).toBe(3);
		expect(router.getQueueDepth("missing")).toBe(0);
	});

	test("maxQueueDepth interrupts a stalled consumer", async () => {
		const router = new StreamRouter<number>({ maxQueueDepth: 2 });
		const consumer = router.createConsumer("slow");
		router.emit("slow", 1);
		router.emit("slow", 2);
		router.emit("slow", 3); // Exceeds cap → consumer interrupted.
		router.emit("slow", 4); // Dropped (consumer interrupted).
		const drained: number[] = [];
		for await (const v of consumer) drained.push(v);
		expect(drained).toEqual([1, 2]);
	});
});
