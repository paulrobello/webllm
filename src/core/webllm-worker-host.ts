/**
 * Worker-side message handler. Constructs a real WebLLM engine and
 * routes ProxyToWorker messages by name via reflect-dispatch.
 *
 * Streaming methods (`chatCompletion`, `generateStream`) drain their
 * async iterator and emit one `stream-chunk` per yield, terminating
 * with `stream-done`. A `stream-cancel` message wires through an
 * AbortController on the args' last `config` argument.
 *
 * Decoupled from `Worker` globals so it can be unit-tested with an
 * in-process channel (see tests/webllm-worker-host.test.ts).
 */

import type { GenerationStreamChunk } from "../inference/generation.js";
import { serializeError } from "./webllm-error-codec.js";
import type {
	ProxyToWorker,
	StreamId,
	WorkerToProxy,
} from "./worker-bridge.js";

export interface WorkerHostOptions {
	/** Engine instance to dispatch RPCs against. */
	// biome-ignore lint/suspicious/noExplicitAny: reflect-dispatch by name
	engine: any;
	/** Send a message to the proxy (main thread). */
	postMessage(m: WorkerToProxy): void;
	/** Subscribe to incoming proxy messages. */
	receive(handler: (m: ProxyToWorker) => void): void;
	/** Optional logger (default: silent). */
	log?(level: "info" | "warn" | "error", message: string): void;
}

export interface WorkerHostHandle {
	/** Stop receiving messages. Does not stop in-flight streams. */
	close(): void;
}

export function startWorkerHost(opts: WorkerHostOptions): WorkerHostHandle {
	const aborts = new Map<StreamId, AbortController>();
	let closed = false;

	opts.receive((msg) => {
		if (closed) return;
		switch (msg.type) {
			case "method-call":
				void handleMethodCall(msg);
				return;
			case "stream-start":
				void handleStreamStart(msg);
				return;
			case "stream-cancel": {
				const ac = aborts.get(msg.streamId);
				if (ac) ac.abort();
				return;
			}
			case "init":
			case "dispose":
				// `init` and `dispose` are typically handled by the bundle
				// entry that owns engine construction. Surfaces here only
				// in the unit-test path; route via method-call/result.
				return;
		}
	});

	async function handleMethodCall(
		msg: Extract<ProxyToWorker, { type: "method-call" }>,
	) {
		try {
			const fn = opts.engine[msg.name];
			if (typeof fn !== "function") {
				throw new Error(`unknown engine method: ${msg.name}`);
			}
			const value = await fn.apply(opts.engine, msg.args);
			opts.postMessage({ type: "method-result", id: msg.id, value });
		} catch (e) {
			opts.postMessage({
				type: "method-error",
				id: msg.id,
				error: serializeError(e),
			});
		}
	}

	async function handleStreamStart(
		msg: Extract<ProxyToWorker, { type: "stream-start" }>,
	) {
		const ac = new AbortController();
		aborts.set(msg.streamId, ac);
		// Inject signal into config (last arg, expected to be an
		// optional config object). If args is shorter or last arg is
		// not an object, append a fresh config; consumers tolerate
		// extra fields on config per chat-types.ts.
		const args = [...msg.args];
		const lastIdx = args.length - 1;
		const last = args[lastIdx];
		if (last && typeof last === "object" && !Array.isArray(last)) {
			args[lastIdx] = { ...(last as object), signal: ac.signal };
		} else {
			args.push({ signal: ac.signal });
		}
		try {
			const fn = opts.engine[msg.name];
			if (typeof fn !== "function") {
				throw new Error(`unknown engine streaming method: ${msg.name}`);
			}
			const iter = fn.apply(opts.engine, args) as AsyncIterable<unknown>;
			for await (const chunk of iter) {
				if (ac.signal.aborted) break;
				opts.postMessage({
					type: "stream-chunk",
					streamId: msg.streamId,
					chunk: chunk as GenerationStreamChunk,
				});
			}
			opts.postMessage({ type: "stream-done", streamId: msg.streamId });
		} catch (e) {
			opts.postMessage({
				type: "stream-error",
				streamId: msg.streamId,
				error: serializeError(e),
			});
		} finally {
			aborts.delete(msg.streamId);
		}
	}

	return {
		close() {
			closed = true;
		},
	};
}
