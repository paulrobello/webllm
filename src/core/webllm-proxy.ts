/**
 * Main-thread façade that mirrors the public WebLLM surface over a
 * postMessage channel. See spec §"Components" #2.
 *
 * Non-streaming methods marshal as a single method-call/method-result
 * round trip. Streaming methods are added in a follow-up commit (Task 6)
 * and back the AsyncIterableIterator they return with a per-stream queue.
 */

import type {
	ConversationHandle,
	ConversationOptions,
} from "./conversation-pool.js";
import type { ModelHandle, WebLLMConfig } from "./types.js";
import { reconstructError } from "./webllm-error-codec.js";
import {
	makeRequestId,
	type ProxyToWorker,
	type RequestId,
	type SerializedError,
	type WorkerToProxy,
} from "./worker-bridge.js";

interface MinimalWorker {
	postMessage(m: ProxyToWorker, transfer?: Transferable[]): void;
	addEventListener(
		event: "message" | "error" | "messageerror",
		h: (e: { data: WorkerToProxy } | Event) => void,
	): void;
	terminate(): void;
}

export class WebLLMProxy {
	private worker: MinimalWorker;
	private pending = new Map<
		RequestId,
		{ resolve: (v: unknown) => void; reject: (e: unknown) => void }
	>();
	private disposed = false;

	private constructor(worker: MinimalWorker) {
		this.worker = worker;
		worker.addEventListener("message", (e) => {
			if (e && typeof e === "object" && "data" in e) {
				this.handleMessage((e as { data: WorkerToProxy }).data);
			}
		});
		worker.addEventListener("error", () => {
			this.handleCrash("worker error event");
		});
		worker.addEventListener("messageerror", () => {
			this.handleCrash("worker messageerror event");
		});
	}

	/**
	 * Public init path used by `WebLLM.init({ worker: true })`. Constructs
	 * a Worker, sends the init RPC, awaits init-done.
	 */
	static async init(config: WebLLMConfig): Promise<WebLLMProxy> {
		const worker = new Worker(new URL(import.meta.url, import.meta.url), {
			type: "module",
		}) as unknown as MinimalWorker;
		const proxy = new WebLLMProxy(worker);
		await proxy.callInit(config);
		return proxy;
	}

	/**
	 * Test-only entry: hand in a pre-wired worker (e.g. an in-process
	 * channel). Skips the init RPC because the unit-test worker host
	 * (`startWorkerHost`) treats `init` as a no-op — engine construction
	 * is the test's responsibility, not the proxy's.
	 */
	static async fromWorker(worker: MinimalWorker): Promise<WebLLMProxy> {
		return new WebLLMProxy(worker);
	}

	private async callInit(config: WebLLMConfig): Promise<void> {
		// `WebLLMConfig` does not yet carry a `worker` field (Task 8 adds it);
		// the destructure below is a forward-compat no-op for today.
		const { worker: _w, ...rest } = config as WebLLMConfig & {
			worker?: unknown;
		};
		void _w;
		await this.request<void>({
			type: "init",
			id: makeRequestId(),
			config: rest,
		});
	}

	// ────────── public WebLLM surface (non-streaming) ──────────

	// Instance signature in engine.ts:1132 is (data, name, wasmUrl?, options?)
	// returning { handle, inference }. The proxy mirrors that exactly.
	loadModelFromBuffer = (
		data: ArrayBuffer,
		name: string,
		wasmUrl?: string,
		options?: unknown,
	): Promise<{ handle: ModelHandle; inference: unknown }> =>
		this.callMethod<{ handle: ModelHandle; inference: unknown }>(
			"loadModelFromBuffer",
			[data, name, wasmUrl, options],
			[data],
		);
	unloadModel = (id: string) => this.callMethod<void>("unloadModel", [id]);
	embed = (modelId: string, text: string) =>
		this.callMethod<Float32Array>("embed", [modelId, text]);
	chat = (modelId: string, prompt: string, config?: unknown) =>
		this.callMethod<string>("chat", [modelId, prompt, config]);
	createConversation = (
		modelHandleId: string,
		opts?: ConversationOptions,
	): Promise<ConversationHandle> =>
		this.callMethod<ConversationHandle>("createConversation", [
			modelHandleId,
			opts,
		]);
	disposeConversation = (conv: ConversationHandle): Promise<void> =>
		this.callMethod<void>("disposeConversation", [conv]);
	forkConversation = (src: ConversationHandle): Promise<ConversationHandle> =>
		this.callMethod<ConversationHandle>("forkConversation", [src]);

	async dispose(): Promise<void> {
		if (this.disposed) return;
		// Fire-and-forget: send the dispose RPC but do not await a response.
		// The worker may shut down its engine before responding, and we
		// tear down the proxy unconditionally on this call. Any in-flight
		// pending requests are rejected via `handleCrash` below.
		try {
			this.worker.postMessage({ type: "dispose", id: makeRequestId() });
		} catch {
			// ignore — we're tearing down anyway
		}
		this.handleCrash("dispose");
	}

	// ────────── private plumbing ──────────

	private callMethod<T>(
		name: string,
		args: unknown[],
		transfer?: Transferable[],
	): Promise<T> {
		if (this.disposed) {
			return Promise.reject(new Error("WebLLM proxy disposed"));
		}
		return this.request<T>(
			{ type: "method-call", id: makeRequestId(), name, args },
			transfer,
		);
	}

	private request<T>(
		msg: ProxyToWorker,
		transfer?: Transferable[],
	): Promise<T> {
		return new Promise<T>((resolve, reject) => {
			const id = "id" in msg ? msg.id : -1;
			if (id < 0) {
				reject(new Error("internal: request without id"));
				return;
			}
			this.pending.set(id, {
				resolve: (v) => resolve(v as T),
				reject,
			});
			try {
				if (transfer && transfer.length > 0) {
					this.worker.postMessage(msg, transfer);
				} else {
					this.worker.postMessage(msg);
				}
			} catch (e) {
				this.pending.delete(id);
				reject(e);
			}
		});
	}

	private handleMessage(m: WorkerToProxy): void {
		switch (m.type) {
			case "init-done": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.resolve(undefined);
				}
				return;
			}
			case "method-result": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.resolve(m.value);
				}
				return;
			}
			case "method-error": {
				const p = this.pending.get(m.id);
				if (p) {
					this.pending.delete(m.id);
					p.reject(reconstructError(m.error));
				}
				return;
			}
			case "stream-chunk":
			case "stream-done":
			case "stream-error":
				// Routed by the streaming task (Task 6) — leave a no-op
				// hook here for now so the unit tests for non-streaming
				// methods don't trip over unhandled-event warnings.
				return;
			case "log":
				return;
		}
	}

	private handleCrash(reason: string): void {
		if (this.disposed) return;
		this.disposed = true;
		const err: SerializedError = {
			code: "DISPOSED",
			message: `engine disposed (${reason})`,
		};
		const rebuilt = reconstructError(err);
		for (const p of this.pending.values()) {
			p.reject(rebuilt);
		}
		this.pending.clear();
		try {
			this.worker.terminate();
		} catch {
			// ignore
		}
	}
}
