import { describe, expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import { WebLLMProxy } from "../src/core/webllm-proxy.js";
import { startWorkerHost } from "../src/core/webllm-worker-host.js";
import type {
	ProxyToWorker,
	WorkerToProxy,
} from "../src/core/worker-bridge.js";

// Build an in-process channel that wires WebLLMProxy ↔ startWorkerHost
// without ever constructing a real Worker. Each side has its own
// receive handler; sends are delivered as microtasks.
function makeInProcessChannel(): {
	worker: {
		postMessage: (m: ProxyToWorker) => void;
		addEventListener: (
			event: "message" | "error" | "messageerror",
			h: (e: { data: WorkerToProxy } | Event) => void,
		) => void;
		terminate: () => void;
	};
	hostPost: (m: WorkerToProxy) => void;
	hostReceive: (handler: (m: ProxyToWorker) => void) => void;
} {
	let toHost: ((m: ProxyToWorker) => void) | null = null;
	let toProxy: ((e: { data: WorkerToProxy }) => void) | null = null;
	const worker = {
		postMessage(m: ProxyToWorker) {
			queueMicrotask(() => toHost?.(m));
		},
		addEventListener(
			event: "message" | "error" | "messageerror",
			h: (e: { data: WorkerToProxy } | Event) => void,
		) {
			if (event === "message") {
				toProxy = h as (e: { data: WorkerToProxy }) => void;
			}
		},
		terminate() {},
	};
	const hostPost = (m: WorkerToProxy) => {
		queueMicrotask(() => toProxy?.({ data: m }));
	};
	const hostReceive = (handler: (m: ProxyToWorker) => void) => {
		toHost = handler;
	};
	return { worker, hostPost, hostReceive };
}

function makeFakeEngine() {
	return {
		async embed(modelId: string): Promise<Float32Array> {
			if (modelId === "missing") throw new ModelNotFoundError(modelId);
			return new Float32Array([1, 2, 3]);
		},
		async createConversation(modelId: string) {
			return { id: `c-${modelId}`, modelHandleId: modelId };
		},
		async disposeConversation(_conv: unknown) {
			return undefined;
		},
		async dispose() {},
	};
}

describe("WebLLMProxy — non-streaming", () => {
	test("embed round-trip returns the worker's value", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const v = await proxy.embed("m1", "hello");
		expect(Array.from(v)).toEqual([1, 2, 3]);
	});

	test("embed surfaces typed ModelNotFoundError main-thread", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		await expect(proxy.embed("missing", "x")).rejects.toBeInstanceOf(
			ModelNotFoundError,
		);
	});

	test("loadModelFromUrl round-trip strips inference and returns handle + metadata", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		// The worker host strips the non-cloneable `inference` field from
		// the return value of loadModelFromBuffer / loadModelFromUrl, but
		// preserves `metadata` (pure data — hyperparams, tokenizerConfig,
		// kvCacheConfig). The fake engine here returns all three; the
		// proxy should see `handle` + `metadata` and not `inference`.
		// Stand-in for `LoadedModelMetadata` — the test only cares that the
		// nested object survives postMessage's structured clone end-to-end,
		// not that it's a fully-populated `ModelHyperparams` shape.
		const fakeMetadata = {
			hyperparams: { architecture: "qwen3", layerCount: 28 },
			tokenizerConfig: { vocabSize: 151936 },
			kvCacheConfig: { maxContextLength: 4096 },
		} as const;
		const engine = {
			async loadModelFromUrl(
				_url: string,
				name: string,
				_wasmUrl?: string,
				_options?: unknown,
			) {
				return {
					handle: { id: `h-${name}` },
					// Plain object stand-in for ModelInference — would be a
					// non-cloneable class instance in the real path.
					inference: { __nonCloneable: () => {} },
					metadata: fakeMetadata,
				};
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const result = await proxy.loadModelFromUrl(
			"https://example.test/model.gguf",
			"qwen3-test",
		);
		expect(result.handle.id).toBe("h-qwen3-test");
		// inference is stripped by the worker host's sanitizer; the
		// resulting object lacks the field rather than carrying an empty
		// placeholder.
		expect("inference" in result).toBe(false);
		// metadata is preserved end-to-end.
		expect(result.metadata as unknown).toEqual(fakeMetadata);
	});

	test("createConversation returns the worker's handle", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const h = await proxy.createConversation("m1");
		expect(h).toEqual({ id: "c-m1", modelHandleId: "m1" });
	});

	test("dispose terminates the worker and rejects subsequent calls", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		let terminated = false;
		const origTerminate = worker.terminate;
		worker.terminate = () => {
			terminated = true;
			origTerminate();
		};
		startWorkerHost({
			engine: makeFakeEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		await proxy.dispose();
		expect(terminated).toBe(true);
		await expect(proxy.embed("m1", "x")).rejects.toThrow(/dispose/i);
	});

	test("in-flight calls reject when dispose is called mid-flight", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		// Slow engine: embed never resolves while we kick off dispose.
		const engine = {
			embed(): Promise<Float32Array> {
				return new Promise(() => {
					/* never resolves */
				});
			},
			async dispose() {},
		};
		startWorkerHost({
			engine,
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const inFlight = proxy.embed("m1", "x");
		// Don't await; race a dispose against the never-resolving embed.
		await proxy.dispose();
		await expect(inFlight).rejects.toThrow(/dispose/i);
	});
});

describe("WebLLMProxy — streaming", () => {
	function makeStreamingEngine() {
		return {
			async *chatCompletion(
				_modelId: string,
				_msgs: unknown[],
				config?: { signal?: AbortSignal },
			) {
				const sig = config?.signal;
				const chunks = [
					{ text: "a", tokenId: 1, done: false },
					{ text: "b", tokenId: 2, done: false },
					{ text: "c", tokenId: 3, done: false },
					{ text: "", done: true, stats: { decodeTokensPerSec: 100 } },
				];
				for (const c of chunks) {
					if (sig?.aborted) return;
					await new Promise((r) => setTimeout(r, 0));
					yield c;
				}
			},
			async dispose() {},
		};
	}

	test("chatCompletion yields all chunks in order", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeStreamingEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const seen: string[] = [];
		for await (const chunk of proxy.chatCompletion("m1", [
			{ role: "user", content: "hi" },
		])) {
			if (chunk.text) seen.push(chunk.text);
		}
		expect(seen).toEqual(["a", "b", "c"]);
	});

	test("chatCompletion early-break sends stream-cancel", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({
			engine: makeStreamingEngine(),
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		let count = 0;
		for await (const chunk of proxy.chatCompletion("m1", [
			{ role: "user", content: "hi" },
		])) {
			count += 1;
			if (chunk.tokenId === 1) break;
		}
		expect(count).toBe(1);
	});

	test("stream-chunks round-trip: 5 chunks injected as single batch yield in order", async () => {
		// Build the channel manually so we can capture the streamId from the
		// host-bound stream-start message and inject a coalesced batch with
		// the same id. We drive the proxy directly here (no host loop) to
		// exercise the stream-chunks dispatch path in isolation.
		const handlers: {
			toProxy?: (e: { data: WorkerToProxy }) => void;
		} = {};
		let capturedStreamId: number | null = null;
		const worker = {
			postMessage(m: ProxyToWorker) {
				if (m.type === "stream-start") capturedStreamId = m.streamId;
			},
			addEventListener(
				event: "message" | "error" | "messageerror",
				h: (e: { data: WorkerToProxy } | Event) => void,
			) {
				if (event === "message") {
					handlers.toProxy = h as (e: { data: WorkerToProxy }) => void;
				}
			},
			terminate() {},
		};
		const proxy = await WebLLMProxy.fromWorker(worker);
		const iter = proxy.chatCompletion("m1", [{ role: "user", content: "hi" }]);
		// Settle the microtask that posts stream-start to the worker.
		await new Promise((r) => setTimeout(r, 0));
		expect(capturedStreamId).not.toBeNull();
		const sid = capturedStreamId as unknown as number;
		const dispatch = handlers.toProxy;
		if (!dispatch) throw new Error("proxy never registered message handler");

		// Inject a single batch of 5 chunks via the message dispatcher.
		dispatch({
			data: {
				type: "stream-chunks",
				streamId: sid,
				chunks: [
					{ text: "a", tokenId: 1, done: false },
					{ text: "b", tokenId: 2, done: false },
					{ text: "c", tokenId: 3, done: false },
					{ text: "d", tokenId: 4, done: false },
					{ text: "e", tokenId: 5, done: false },
				],
			},
		});
		dispatch({ data: { type: "stream-done", streamId: sid } });

		const collected: string[] = [];
		for await (const c of iter) {
			if (c.text) collected.push(c.text);
		}
		expect(collected).toEqual(["a", "b", "c", "d", "e"]);
	});

	test("chatCompletion propagates worker-side stream-error as typed error", async () => {
		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		const { ModelNotFoundError } = await import("../src/core/errors.js");
		startWorkerHost({
			engine: {
				// biome-ignore lint/correctness/useYield: intentional throw before yield
				async *chatCompletion() {
					throw new ModelNotFoundError("m1");
				},
				async dispose() {},
			},
			postMessage: hostPost,
			receive: hostReceive,
		});
		const proxy = await WebLLMProxy.fromWorker(worker);
		const consume = async () => {
			for await (const _c of proxy.chatCompletion("m1", [
				{ role: "user", content: "hi" },
			])) {
				/* drain */
			}
		};
		await expect(consume()).rejects.toBeInstanceOf(ModelNotFoundError);
	});
});

// Probe #6 (worker-mode + ConversationPool integration): the surface
// sentinel proves the proxy mirrors all four conversation methods, but
// no test previously round-tripped a `ConversationHandle` through
// structured clone *and* exercised the conv-overload of `chatCompletion`.
// The lifecycle test below drives:
//   create → chatCompletion(conv) → fork → chatCompletion(forked) → dispose
// against a fake engine that owns a real `ConversationPool`. If the
// handle's `id` / `modelHandleId` survives postMessage and reflect-
// dispatch finds all four engine methods, this test passes; if either
// link is broken, it fails with a localizable error.
describe("WebLLMProxy — conversation lifecycle", () => {
	test("create → chatCompletion(conv) → fork → chatCompletion(fork) → dispose round-trips through the proxy", async () => {
		const { ConversationPool } = await import(
			"../src/core/conversation-pool.js"
		);
		const { ConversationNotFoundError } = await import("../src/core/errors.js");
		const pool = new ConversationPool({ maxConversations: 4 });
		const seenConvIds: string[] = [];

		const engine = {
			async createConversation(modelHandleId: string) {
				return pool.create(modelHandleId);
			},
			async forkConversation(
				src: import("../src/core/conversation-pool.js").ConversationHandle,
			) {
				return pool.fork(src);
			},
			async disposeConversation(
				conv: import("../src/core/conversation-pool.js").ConversationHandle,
			) {
				pool.dispose(conv);
			},
			async *chatCompletion(
				first:
					| string
					| import("../src/core/conversation-pool.js").ConversationHandle,
				_msgs: unknown[],
			) {
				if (typeof first === "string") {
					throw new Error("test only exercises conv-handle overload");
				}
				pool.assertExists(first);
				seenConvIds.push(first.id);
				// Seed a snapshot so subsequent forkConversation succeeds.
				pool.set(first, {
					conversationId: first.id,
					modelHandleId: first.modelHandleId,
					tokenIds: [1, 2, 3],
					kvBytes: new Uint8Array(8),
					byteSize: 8,
					lastAccessMs: Date.now(),
				});
				yield { text: "x", tokenId: 1, done: false };
				yield { text: "", done: true, stats: { decodeTokensPerSec: 1 } };
			},
			async dispose() {},
		};

		const { worker, hostPost, hostReceive } = makeInProcessChannel();
		startWorkerHost({ engine, postMessage: hostPost, receive: hostReceive });
		const proxy = await WebLLMProxy.fromWorker(worker);

		const conv1 = await proxy.createConversation("m1");
		expect(conv1).toEqual({ id: "conv_1", modelHandleId: "m1" });

		for await (const _c of proxy.chatCompletion(conv1, [
			{ role: "user", content: "hi" },
		])) {
			/* drain */
		}

		const conv2 = await proxy.forkConversation(conv1);
		expect(conv2).toEqual({ id: "conv_2", modelHandleId: "m1" });

		for await (const _c of proxy.chatCompletion(conv2, [
			{ role: "user", content: "hi" },
		])) {
			/* drain */
		}

		expect(seenConvIds).toEqual(["conv_1", "conv_2"]);

		await proxy.disposeConversation(conv1);
		// Disposed handle: chatCompletion(conv1, ...) should now reject with
		// ConversationNotFoundError, surfacing through the typed-error codec.
		const drainDisposed = async () => {
			for await (const _c of proxy.chatCompletion(conv1, [
				{ role: "user", content: "hi" },
			])) {
				/* drain */
			}
		};
		await expect(drainDisposed()).rejects.toBeInstanceOf(
			ConversationNotFoundError,
		);
	});
});
