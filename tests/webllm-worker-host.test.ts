import { describe, expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import { startWorkerHost } from "../src/core/webllm-worker-host.js";
import type {
	ProxyToWorker,
	RequestId,
	WorkerToProxy,
} from "../src/core/worker-bridge.js";

// Minimal fake engine implementing the surface the host reflects against.
function makeFakeEngine() {
	return {
		// non-streaming method, returns plain JSON
		async embed(modelId: string, text: string): Promise<Float32Array> {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return new Float32Array([text.length, 0.5, 0.25]);
		},
		// non-streaming method that throws WebLLMError
		async createConversation(modelId: string) {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return { id: "c1", modelHandleId: modelId };
		},
		// streaming method (async generator)
		async *chatCompletion(_modelId: string, _msgs: unknown[]) {
			yield { text: "hi", tokenId: 1, done: false };
			yield { text: " there", tokenId: 2, done: false };
			yield { text: "", done: true, stats: { decodeTokensPerSec: 99 } };
		},
		async dispose() {
			/* no-op */
		},
	};
}

interface FakeChannel {
	hostInbox: ProxyToWorker[];
	proxyInbox: WorkerToProxy[];
	postToHost(m: ProxyToWorker): void;
}

function makeChannel(): {
	channel: FakeChannel;
	postToProxy: (m: WorkerToProxy) => void;
} {
	const proxyInbox: WorkerToProxy[] = [];
	const hostInbox: ProxyToWorker[] = [];
	return {
		channel: {
			hostInbox,
			proxyInbox,
			postToHost(m) {
				hostInbox.push(m);
			},
		},
		postToProxy: (m: WorkerToProxy) => {
			proxyInbox.push(m);
		},
	};
}

describe("webllm-worker-host", () => {
	test("method-call success returns method-result", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "method-call",
			id: 1,
			name: "embed",
			args: ["m1", "abc"],
		});
		// allow microtasks to settle
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox).toEqual([
			{
				type: "method-result",
				id: 1,
				value: new Float32Array([3, 0.5, 0.25]),
			},
		]);
	});

	test("method-call WebLLMError serializes to method-error with code", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "method-call",
			id: 2,
			name: "embed",
			args: ["missing", "x"],
		});
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox).toHaveLength(1);
		const reply = channel.proxyInbox[0];
		expect(reply.type).toBe("method-error");
		if (reply.type === "method-error") {
			expect(reply.id).toBe(2);
			expect(reply.error.code).toBe("MODEL_NOT_FOUND");
			expect(reply.error.modelId).toBe("missing");
		}
	});

	test("stream-start drains async iterator into stream-chunks + stream-done", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 7,
			name: "chatCompletion",
			args: ["m1", [{ role: "user", content: "hi" }], {}],
		});
		// drain microtasks
		for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));
		// 3 back-to-back chunks coalesce and the stream terminates with
		// stream-done. Assert the timing-robust invariants (all chunks
		// delivered in order, clean termination, no legacy singular posts)
		// rather than the exact batch count: under event-loop contention the
		// host's 16 ms time-cap (BATCH_MAX_MS in webllm-worker-host.ts)
		// legitimately splits even back-to-back yields into multiple
		// stream-chunks batches — chunks still arrive in order, so it is
		// correct behavior. Pinning exactly one batch made this test ~22%
		// flaky on loaded runners.
		const batches = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunks",
		) as Array<Extract<WorkerToProxy, { type: "stream-chunks" }>>;
		const allChunks = batches.flatMap((b) => b.chunks);
		expect(allChunks).toHaveLength(3);
		expect(allChunks.map((c) => c.text)).toEqual(["hi", " there", ""]);
		expect(channel.proxyInbox[channel.proxyInbox.length - 1]?.type).toBe(
			"stream-done",
		);
		// No legacy singular stream-chunk posts.
		expect(
			channel.proxyInbox.filter((m) => m.type === "stream-chunk"),
		).toHaveLength(0);
	});

	test("stream-cancel posted immediately after stream-start aborts the stream (no race)", async () => {
		const { channel, postToProxy } = makeChannel();
		// Engine that yields slowly so we have a chance to see whether the
		// cancel landed. Yields are spaced out one microtask each.
		const engine = {
			async *chatCompletion(
				_modelId: string,
				_msgs: unknown[],
				config?: { signal?: AbortSignal },
			) {
				const sig = config?.signal;
				for (let i = 0; i < 100; i++) {
					if (sig?.aborted) return;
					yield { text: String(i), tokenId: i, done: false };
					await new Promise((r) => setTimeout(r, 0));
				}
				yield { text: "", done: true };
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});

		// Synchronously: start stream, then immediately cancel.
		channel.postToHost({
			type: "stream-start",
			streamId: 42,
			name: "chatCompletion",
			args: ["m1", [{ role: "user", content: "go" }], {}],
		});
		channel.postToHost({ type: "stream-cancel", streamId: 42 });

		// Drain microtasks; the stream should have aborted before producing
		// anywhere near 100 chunks.
		for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0));

		const chunkCount = channel.proxyInbox
			.filter((m) => m.type === "stream-chunks")
			.reduce(
				(n, m) =>
					n +
					(m as Extract<WorkerToProxy, { type: "stream-chunks" }>).chunks
						.length,
				0,
			);
		const lastType = channel.proxyInbox[channel.proxyInbox.length - 1]?.type;
		// Far less than 100 chunks if cancellation worked.
		expect(chunkCount).toBeLessThan(20);
		// The stream still terminates cleanly with stream-done (the host
		// posts stream-done after the loop exits).
		expect(lastType).toBe("stream-done");
	});

	test("stream-start coalesces chunks into stream-chunks batches (size-cap)", async () => {
		// 16 chunks yielded back-to-back should land in 2 batches of 8 (the
		// size-cap path), not 16 individual stream-chunk posts. All 16 chunk
		// payloads must arrive at the proxy in original order.
		const { channel, postToProxy } = makeChannel();
		const engine = {
			async *chatCompletion(_modelId: string, _msgs: unknown[]) {
				for (let i = 0; i < 16; i++) {
					yield { text: String(i), tokenId: i, done: false };
				}
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 11,
			name: "chatCompletion",
			args: ["m1", [{ role: "user", content: "go" }], {}],
		});
		// Drain microtasks.
		for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0));

		// No legacy singular stream-chunk posts.
		const singletons = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunk",
		);
		expect(singletons).toHaveLength(0);

		const batches = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunks",
		) as Array<Extract<WorkerToProxy, { type: "stream-chunks" }>>;
		// Coalescing should at least halve the message count: 16 chunks at
		// cap=8 → ≥2 size-driven batches, and far fewer than 16 under any
		// realistic timing. The upper bound tolerates event-loop stalls (the
		// 16 ms time-cap can split back-to-back yields into a few extra
		// batches) while still catching a flush-every-chunk regression (which
		// would produce 16 singleton batches). See the sibling "drains async
		// iterator" test for why exact batch counts are not robust here.
		expect(batches.length).toBeGreaterThanOrEqual(1);
		expect(batches.length).toBeLessThanOrEqual(8);
		// Total chunks across all batches must equal 16 in order.
		const allChunks = batches.flatMap((b) => b.chunks);
		expect(allChunks).toHaveLength(16);
		expect(allChunks.map((c) => c.tokenId)).toEqual(
			Array.from({ length: 16 }, (_, i) => i),
		);
		// stream-done arrives after all chunks.
		const lastType = channel.proxyInbox[channel.proxyInbox.length - 1]?.type;
		expect(lastType).toBe("stream-done");
	});

	test("stream-start flushes via time-cap when chunks are spaced apart", async () => {
		// Yields spaced 20 ms apart (> BATCH_MAX_MS=16) must NOT all coalesce
		// into a single big batch. The flush check runs after each push, so
		// the time-cap fires when chunk N+1 lands more than 16 ms after the
		// last flush — flushing chunks 1..N. With 8 yields spaced 20 ms apart,
		// every chunk after the first crosses the time boundary, so we
		// expect roughly one batch per chunk (modulo the trailing tail flush
		// at stream-done). The size-cap (8 chunks) never fires here.
		//
		// Defends against a regression where the time-cap branch is silently
		// lost (e.g. an `else` accidentally swallowing the size+time
		// disjunction) — under that bug all 8 chunks would coalesce into a
		// single batch via the trailing tail flush.
		const { channel, postToProxy } = makeChannel();
		const engine = {
			async *chatCompletion(_modelId: string, _msgs: unknown[]) {
				for (let i = 0; i < 8; i++) {
					yield { text: String(i), tokenId: i, done: false };
					await new Promise((r) => setTimeout(r, 20));
				}
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 99,
			name: "chatCompletion",
			args: ["m1", [{ role: "user", content: "go" }], {}],
		});
		// Total minimum wall is ~160 ms across 8 yields; leave generous
		// headroom so the trailing flush + stream-done land before assertion.
		await new Promise((r) => setTimeout(r, 400));

		const batches = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunks",
		) as Array<Extract<WorkerToProxy, { type: "stream-chunks" }>>;
		const totalChunks = batches.reduce((n, b) => n + b.chunks.length, 0);
		expect(totalChunks).toBe(8);
		// Time-cap test: spaced yields must NOT coalesce into one batch.
		// Under a working time-cap path we expect ~8 batches (one per yield);
		// under regression with a broken time-cap we'd see exactly 1 batch.
		// Use a generous floor (>= 4) to absorb scheduler jitter without
		// allowing the regression mode (1 batch) to slip through.
		expect(batches.length).toBeGreaterThanOrEqual(4);
		// Order must still be preserved across batches.
		const allChunks = batches.flatMap((b) => b.chunks);
		expect(allChunks.map((c) => c.tokenId)).toEqual(
			Array.from({ length: 8 }, (_, i) => i),
		);
		// stream-done after all batches.
		const lastType = channel.proxyInbox[channel.proxyInbox.length - 1]?.type;
		expect(lastType).toBe("stream-done");
	});

	test("stream-start flushes residual buffer before stream-done", async () => {
		// 5 chunks yielded — fewer than BATCH_MAX_CHUNKS=8. Without a final
		// flush, no `stream-chunks` would be posted at all. Verify the trailing
		// flush before stream-done delivers them.
		const { channel, postToProxy } = makeChannel();
		const engine = {
			async *chatCompletion(_modelId: string, _msgs: unknown[]) {
				for (let i = 0; i < 5; i++) {
					yield { text: String(i), tokenId: i, done: false };
				}
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 12,
			name: "chatCompletion",
			args: ["m1", [], {}],
		});
		for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));

		const batches = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunks",
		) as Array<Extract<WorkerToProxy, { type: "stream-chunks" }>>;
		// The trailing flush before stream-done must deliver all 5 chunks
		// (the regression it guards is "no trailing flush → chunks dropped").
		// Assert all 5 arrive in order rather than exactly one batch — under
		// event-loop stalls the 16 ms time-cap can split the back-to-back
		// yields into multiple batches, but the total is invariant. See the
		// sibling "drains async iterator" test.
		const allChunks = batches.flatMap((b) => b.chunks);
		expect(allChunks).toHaveLength(5);
		expect(allChunks.map((c) => c.tokenId)).toEqual([0, 1, 2, 3, 4]);
		expect(channel.proxyInbox[channel.proxyInbox.length - 1]?.type).toBe(
			"stream-done",
		);
	});

	test("stream-error path flushes pending chunks before posting the error", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = {
			async *chatCompletion(_modelId: string, _msgs: unknown[]) {
				yield { text: "a", tokenId: 1, done: false };
				yield { text: "b", tokenId: 2, done: false };
				throw new Error("boom");
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "stream-start",
			streamId: 13,
			name: "chatCompletion",
			args: ["m1", [], {}],
		});
		for (let i = 0; i < 10; i++) await new Promise((r) => setTimeout(r, 0));

		const batches = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunks",
		) as Array<Extract<WorkerToProxy, { type: "stream-chunks" }>>;
		const allChunks = batches.flatMap((b) => b.chunks);
		expect(allChunks.map((c) => c.tokenId)).toEqual([1, 2]);
		const lastType = channel.proxyInbox[channel.proxyInbox.length - 1]?.type;
		expect(lastType).toBe("stream-error");
	});

	test("exportConversation result is transferred (envelope.transfer populated)", async () => {
		const sentMessages: Array<{
			msg: WorkerToProxy;
			transfer?: Transferable[] | undefined;
		}> = [];
		const handlers = { receive: null as ((m: ProxyToWorker) => void) | null };
		const engine = {
			async exportConversation() {
				return new Uint8Array([1, 2, 3, 4, 5]);
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: (msg, transfer) => sentMessages.push({ msg, transfer }),
			receive: (h) => {
				handlers.receive = h;
			},
		});
		handlers.receive?.({
			type: "method-call",
			id: 1 as RequestId,
			name: "exportConversation",
			args: [{ id: "c1", modelHandleId: "m1" }],
		});
		await new Promise((r) => setTimeout(r, 0));
		const result = sentMessages.find((e) => e.msg.type === "method-result");
		expect(result).toBeDefined();
		expect(result?.transfer?.length).toBe(1);
		expect(result?.transfer?.[0]).toBeInstanceOf(ArrayBuffer);
	});

	test("non-allowlisted method does NOT populate transfer list", async () => {
		const sentMessages: Array<{
			msg: WorkerToProxy;
			transfer?: Transferable[] | undefined;
		}> = [];
		const handlers = { receive: null as ((m: ProxyToWorker) => void) | null };
		const engine = {
			async embed() {
				return new Float32Array([1, 2, 3]);
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: (msg, transfer) => sentMessages.push({ msg, transfer }),
			receive: (h) => {
				handlers.receive = h;
			},
		});
		handlers.receive?.({
			type: "method-call",
			id: 1 as RequestId,
			name: "embed",
			args: ["m1", "hi"],
		});
		await new Promise((r) => setTimeout(r, 0));
		const result = sentMessages.find((e) => e.msg.type === "method-result");
		expect(result?.transfer).toBeUndefined();
	});

	test("unknown method returns method-error with GENERIC code", async () => {
		const { channel, postToProxy } = makeChannel();
		const engine = makeFakeEngine();
		startWorkerHost({
			engine,
			postMessage: postToProxy,
			receive(handler) {
				channel.postToHost = (m) => handler(m);
			},
		});
		channel.postToHost({
			type: "method-call",
			id: 9,
			name: "definitelyDoesNotExist",
			args: [],
		});
		await new Promise((r) => setTimeout(r, 0));
		expect(channel.proxyInbox[0].type).toBe("method-error");
		if (channel.proxyInbox[0].type === "method-error") {
			expect(channel.proxyInbox[0].error.code).toBe("GENERIC");
		}
	});
});
