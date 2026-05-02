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
