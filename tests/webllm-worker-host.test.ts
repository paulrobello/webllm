import { describe, expect, test } from "bun:test";
import { ModelNotFoundError } from "../src/core/errors.js";
import { startWorkerHost } from "../src/core/webllm-worker-host.js";
import type {
	ProxyToWorker,
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

	test("stream-start drains async iterator into stream-chunk + stream-done", async () => {
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
		const types = channel.proxyInbox.map((m) => m.type);
		expect(types).toEqual([
			"stream-chunk",
			"stream-chunk",
			"stream-chunk",
			"stream-done",
		]);
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

		const chunkCount = channel.proxyInbox.filter(
			(m) => m.type === "stream-chunk",
		).length;
		const lastType = channel.proxyInbox[channel.proxyInbox.length - 1]?.type;
		// Far less than 100 chunks if cancellation worked.
		expect(chunkCount).toBeLessThan(20);
		// The stream still terminates cleanly with stream-done (the host
		// posts stream-done after the loop exits).
		expect(lastType).toBe("stream-done");
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
