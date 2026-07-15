/**
 * Main-thread façade that mirrors the public WebLLM surface over a
 * postMessage channel. See spec §"Components" #2.
 *
 * Non-streaming methods marshal as a single method-call/method-result
 * round trip. Streaming methods are added in a follow-up commit (Task 6)
 * and back the AsyncIterableIterator they return with a per-stream queue.
 */

import type { CausalLMEmbedder } from "../inference/causal-embedder-inference.js";
import type { EncoderInference } from "../inference/encoder-inference.js";
import type {
	GenerationConfig,
	GenerationStreamChunk,
} from "../inference/generation.js";
import type { ModelInference } from "../inference/model-inference.js";
import type {
	ChatMessage,
	CompletionConfig,
	StreamConfig,
	StreamInput,
} from "./chat-types.js";
import type {
	ConversationHandle,
	ConversationOptions,
} from "./conversation-pool.js";
import type { WebLLM } from "./engine.js";
import type {
	LoadedModelMetadata,
	ModelHandle,
	ModelLoadOptions,
	WebLLMConfig,
} from "./types.js";
import { reconstructError } from "./webllm-error-codec.js";
import {
	makeRequestId,
	makeStreamId,
	type ProxyToWorker,
	type RequestId,
	type SerializedError,
	type StreamId,
	type WorkerToProxy,
} from "./worker-bridge.js";

/**
 * Compile-time surface contract between {@link WebLLM} and {@link WebLLMProxy}
 * (ARC-004). Every public WebLLM method that the proxy mirrors appears here,
 * so a method added to WebLLM without a matching proxy arm (or vice versa) is
 * a `tsc` error — complementing the runtime `tests/webllm-proxy-surface.test.ts`
 * sentinel. The cast in `WebLLM.init` (`as unknown as Promise<WebLLM>`) stays
 * for now; this type narrows the surface on the proxy side.
 *
 * Excluded (covered by the runtime surface sentinel, not this Pick):
 *   - `tokenize` — engine method is synchronous (`readonly number[]`); the
 *     proxy's worker-RPC implementation is necessarily async
 *     (`Promise<readonly number[]>`). The signatures are fundamentally
 *     incompatible.
 *   - `chatCompletion`, `generateStream` — engine returns
 *     `AsyncGenerator<T, void>`; the proxy's hand-built worker-RPC iter is
 *     `AsyncIterableIterator<GenerationStreamChunk>`. The project's TS lib
 *     requires `[Symbol.asyncDispose]` on `AsyncGenerator<T>` but does not
 *     expose `Symbol.asyncDispose` itself (TS2741 / TS2550 gap), so widening
 *     the proxy's iter to `AsyncGenerator` is not satisfiable. Revisit once
 *     the project's lib target exposes `Symbol.asyncDispose`.
 */
type WebLLMSurface = Pick<
	Omit<
		WebLLM,
		// tokenize is sync on the engine; the proxy is inherently async.
		| "tokenize"
		// Streaming iters can't satisfy AsyncGenerator's asyncDispose
		// requirement under the project's current TS lib (see docstring).
		| "chatCompletion"
		| "generateStream"
	>,
	| "loadModelFromBuffer"
	| "loadModelFromUrl"
	| "unloadModel"
	| "resetModelSession"
	| "resetConversation"
	| "embed"
	| "chat"
	| "createConversation"
	| "disposeConversation"
	| "forkConversation"
	| "exportConversation"
	| "importConversation"
	| "dispose"
>;

/**
 * Return shape shared by `loadModelFromBuffer` / `loadModelFromUrl`. Matches
 * the engine's signature exactly so {@link WebLLMSurface} is satisfied. The
 * worker-host sanitizer strips the non-cloneable `inference` field before the
 * reply crosses the postMessage boundary (the proxy never reconstructs a live
 * pipeline object); the union type is the public contract, the runtime value
 * for `inference` in worker mode is a stripped sentinel.
 */
type LoadResult = {
	handle: ModelHandle;
	inference: ModelInference | EncoderInference | CausalLMEmbedder;
	metadata: LoadedModelMetadata;
};

/**
 * Drop the non-cloneable `signal: AbortSignal` field from a config arg
 * before posting it to the worker. The host injects its own signal tied
 * to `stream-cancel` (`webllm-worker-host.ts:141`); the user's signal is
 * wired back via {@link wireSignalToStream} so cancellation propagates.
 *
 * Returns the input unchanged when there is no signal to strip — keeps
 * `undefined` / non-object configs working.
 */
function stripSignal(config: unknown): unknown {
	if (!config || typeof config !== "object" || Array.isArray(config)) {
		return config;
	}
	const obj = config as Record<string, unknown>;
	if (!("signal" in obj)) return config;
	const { signal: _signal, ...rest } = obj;
	return rest;
}

/** Pull the `signal` field out of a config arg, if any. */
function extractSignal(config: unknown): AbortSignal | undefined {
	if (!config || typeof config !== "object" || Array.isArray(config)) {
		return undefined;
	}
	const sig = (config as Record<string, unknown>).signal;
	return sig instanceof AbortSignal ? sig : undefined;
}

/**
 * Pull the non-cloneable streaming callbacks (`onToken` / `onThinking`) off a
 * config arg so the proxy can re-invoke them per arriving chunk on the main
 * side. Functions do not survive `postMessage` (structured clone drops them),
 * so without this capture-and-strip the callbacks would silently vanish.
 * Mirrors {@link extractSignal}. Returns only the callbacks that are actually
 * functions.
 */
function extractCallbacks(config: unknown): {
	onToken?: (delta: string) => void;
	onThinking?: (delta: string) => void;
} {
	if (!config || typeof config !== "object" || Array.isArray(config)) {
		return {};
	}
	const obj = config as Record<string, unknown>;
	const out: {
		onToken?: (delta: string) => void;
		onThinking?: (delta: string) => void;
	} = {};
	if (typeof obj.onToken === "function") {
		out.onToken = obj.onToken as (delta: string) => void;
	}
	if (typeof obj.onThinking === "function") {
		out.onThinking = obj.onThinking as (delta: string) => void;
	}
	return out;
}

/**
 * Drop the non-cloneable streaming callbacks from a config arg before it
 * crosses `postMessage`. Companion to {@link extractCallbacks}; mirrors
 * {@link stripSignal}. Returns the input unchanged when there is nothing to
 * strip.
 */
function stripCallbacks(config: unknown): unknown {
	if (!config || typeof config !== "object" || Array.isArray(config)) {
		return config;
	}
	const obj = config as Record<string, unknown>;
	if (!("onToken" in obj) && !("onThinking" in obj)) return config;
	const { onToken: _t, onThinking: _th, ...rest } = obj;
	return rest;
}

interface MinimalWorker {
	postMessage(m: ProxyToWorker, transfer?: Transferable[]): void;
	addEventListener(
		event: "message" | "error" | "messageerror",
		h: (e: { data: WorkerToProxy } | Event) => void,
	): void;
	terminate(): void;
}

export class WebLLMProxy implements WebLLMSurface {
	private worker: MinimalWorker;
	private pending = new Map<
		RequestId,
		{ resolve: (v: unknown) => void; reject: (e: unknown) => void }
	>();
	private streams = new Map<
		StreamId,
		{
			queue: GenerationStreamChunk[];
			waiters: Array<{
				resolve: (r: IteratorResult<GenerationStreamChunk>) => void;
				reject: (e: unknown) => void;
			}>;
			errored: unknown | null;
			done: boolean;
			/**
			 * Streaming callbacks captured off the config before posting
			 * (functions don't cross `postMessage`). Re-invoked per arriving
			 * chunk by {@link WebLLMProxy.invokeStreamCallbacks}.
			 */
			onToken?: (delta: string) => void;
			onThinking?: (delta: string) => void;
			/**
			 * True once the stream has reached a terminal state
			 * (done / error / cancel). Checked before invoking callbacks so
			 * no onToken/onThinking fires after terminal resolution — the
			 * abort-ordering invariant (ENH-002 Task 2 §4).
			 */
			callbacksReleased: boolean;
		}
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
	// returning { handle, inference, metadata }. The worker host sanitizer
	// strips the non-cloneable `inference` field before reply; `metadata`
	// is pure data (hyperparams + tokenizerConfig + kvCacheConfig) and
	// survives postMessage's structured clone.
	loadModelFromBuffer = (
		data: ArrayBuffer | Uint8Array,
		name: string,
		wasmUrl?: string,
		options?: Partial<ModelLoadOptions>,
	): Promise<LoadResult> =>
		this.callMethod<LoadResult>(
			"loadModelFromBuffer",
			[data, name, wasmUrl, options],
			[data],
		);
	// `url` and `name` are cheap strings; no Transferables needed. The
	// worker fetches + streams into its own WASM heap so >3.5 GB models
	// don't have to land in a main-thread JS-heap ArrayBuffer first.
	loadModelFromUrl = (
		url: string,
		name: string,
		wasmUrl?: string,
		options?: Partial<ModelLoadOptions>,
		onProgress?: (received: number, total: number) => void,
	): Promise<LoadResult> => {
		void onProgress;
		return this.callMethod<LoadResult>("loadModelFromUrl", [
			url,
			name,
			wasmUrl,
			options,
		]);
	};
	unloadModel = (id: string) => this.callMethod<void>("unloadModel", [id]);
	resetModelSession = (modelId: string) =>
		this.callMethod<void>("resetModelSession", [modelId]);
	/** @deprecated Use resetModelSession. */
	resetConversation = (modelId: string) =>
		this.callMethod<void>("resetConversation", [modelId]);
	embed = (modelId: string, text: string) =>
		this.callMethod<Float32Array>("embed", [modelId, text]);
	chat = (
		modelId: string,
		prompt: string,
		config?: Partial<GenerationConfig>,
	) => this.callMethod<string>("chat", [modelId, prompt, stripSignal(config)]);
	chatCompletion = (
		modelOrConv: string | ConversationHandle,
		messages: ChatMessage[],
		config?: CompletionConfig,
	): AsyncIterableIterator<GenerationStreamChunk> =>
		this.startStream(
			"chatCompletion",
			[modelOrConv, messages, stripCallbacks(stripSignal(config))],
			extractSignal(config),
			extractCallbacks(config),
		);
	generateStream = (
		modelId: string,
		input: StreamInput,
		config?: StreamConfig,
	): AsyncIterableIterator<GenerationStreamChunk> =>
		this.startStream(
			"generateStream",
			[modelId, input, stripCallbacks(stripSignal(config))],
			extractSignal(config),
			extractCallbacks(config),
		);
	tokenize = (
		modelHandleId: string,
		text: string,
	): Promise<readonly number[]> =>
		this.callMethod<readonly number[]>("tokenize", [modelHandleId, text]);
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
	exportConversation = (conv: ConversationHandle): Promise<Uint8Array> =>
		this.callMethod<Uint8Array>("exportConversation", [conv]);
	importConversation = (
		modelHandleId: string,
		blob: Uint8Array,
		options?: ConversationOptions,
	): Promise<ConversationHandle> =>
		this.callMethod<ConversationHandle>(
			"importConversation",
			[modelHandleId, blob, options],
			[blob.buffer as ArrayBuffer],
		);

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
			case "stream-chunk": {
				const s = this.streams.get(m.streamId);
				if (!s) return;
				this.invokeStreamCallbacks(s, m.chunk);
				if (s.waiters.length > 0) {
					const w = s.waiters.shift();
					w?.resolve({ value: m.chunk, done: false });
				} else {
					s.queue.push(m.chunk);
				}
				return;
			}
			case "stream-chunks": {
				// Coalesced batch from the worker-host. Fan chunks back out
				// one-at-a-time so the public AsyncIterableIterator surface
				// is unchanged. Order is preserved: waiters drain first
				// (oldest-first), then the rest queue.
				const s = this.streams.get(m.streamId);
				if (!s) return;
				for (const chunk of m.chunks) {
					this.invokeStreamCallbacks(s, chunk);
					if (s.waiters.length > 0) {
						const w = s.waiters.shift();
						w?.resolve({ value: chunk, done: false });
					} else {
						s.queue.push(chunk);
					}
				}
				return;
			}
			case "stream-done": {
				const s = this.streams.get(m.streamId);
				if (!s) return;
				s.done = true;
				this.releaseStreamCallbacks(s);
				const waiters = s.waiters;
				s.waiters = [];
				for (const w of waiters) w.resolve({ value: undefined, done: true });
				if (s.queue.length === 0) {
					this.streams.delete(m.streamId);
				}
				return;
			}
			case "stream-error": {
				const s = this.streams.get(m.streamId);
				if (!s) return;
				const err = reconstructError(m.error);
				s.errored = err;
				this.releaseStreamCallbacks(s);
				// Reject any pending waiters directly — otherwise the for-await
				// loop terminates on a stale done=true and never re-enters next().
				const waiters = s.waiters;
				s.waiters = [];
				for (const w of waiters) w.reject(err);
				if (s.queue.length === 0) {
					this.streams.delete(m.streamId);
				}
				return;
			}
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
		for (const s of this.streams.values()) {
			s.errored = rebuilt;
			const waiters = s.waiters;
			s.waiters = [];
			for (const w of waiters) w.reject(rebuilt);
		}
		this.streams.clear();
		try {
			this.worker.terminate();
		} catch {
			// ignore
		}
	}

	/**
	 * Re-invoke the captured `onToken` / `onThinking` for one arriving chunk.
	 * Called from the `stream-chunk` / `stream-chunks` handlers BEFORE the
	 * chunk is enqueued / handed to a waiter, so callbacks fire at arrival
	 * time (mirroring the inline contract where the Generator fires them
	 * synchronously inside `gen.next()`, independent of consumer pull rate).
	 *
	 * Guards:
	 * - Skips entirely once the stream is terminal
	 *   ({@link StreamEntry.callbacksReleased}) — abort ordering.
	 * - Fires `onToken` only when `chunk.text` is non-empty (the terminal
	 *   done chunk carries `text: ""` and must not spuriously fire).
	 * - Fires `onThinking` only when `chunk.thinkingText` is present.
	 */
	private invokeStreamCallbacks(
		s: {
			callbacksReleased: boolean;
			onToken?: (delta: string) => void;
			onThinking?: (delta: string) => void;
		},
		chunk: GenerationStreamChunk,
	): void {
		if (s.callbacksReleased) return;
		if (chunk.text && s.onToken) s.onToken(chunk.text);
		if (chunk.thinkingText && s.onThinking) s.onThinking(chunk.thinkingText);
	}

	/**
	 * Mark a stream's callbacks as released and drop the stored references so
	 * no later-arriving chunk can invoke them and the closures don't leak.
	 * Called on every terminal path: `stream-done`, `stream-error`, and
	 * `cancel()` (user abort). Idempotent — the `callbacksReleased` guard
	 * makes a second call a no-op.
	 *
	 * `exactOptionalPropertyTypes`: assigning `undefined` to an optional
	 * property is an error, so the references are cleared via `delete`.
	 */
	private releaseStreamCallbacks(s: {
		callbacksReleased: boolean;
		onToken?: (delta: string) => void;
		onThinking?: (delta: string) => void;
	}): void {
		if (s.callbacksReleased) return;
		s.callbacksReleased = true;
		delete s.onToken;
		delete s.onThinking;
	}

	private startStream(
		name: "chatCompletion" | "generateStream",
		args: unknown[],
		signal?: AbortSignal,
		callbacks?: {
			onToken?: (delta: string) => void;
			onThinking?: (delta: string) => void;
		},
	): AsyncIterableIterator<GenerationStreamChunk> {
		if (this.disposed) {
			const err = new Error("WebLLM proxy disposed");
			// biome-ignore lint/correctness/useYield: throw-only generator surfaces error to consumer
			return (async function* () {
				throw err;
			})();
		}
		const streamId = makeStreamId();
		this.streams.set(streamId, {
			queue: [],
			waiters: [],
			errored: null,
			done: false,
			callbacksReleased: false,
			// Conditional spread keeps the optional fields absent (not
			// `undefined`) under `exactOptionalPropertyTypes`.
			...(callbacks?.onToken ? { onToken: callbacks.onToken } : {}),
			...(callbacks?.onThinking ? { onThinking: callbacks.onThinking } : {}),
		});
		this.worker.postMessage({ type: "stream-start", streamId, name, args });
		const self = this;
		let cancelled = false;
		const cancel = (): void => {
			if (cancelled) return;
			cancelled = true;
			try {
				self.worker.postMessage({ type: "stream-cancel", streamId });
			} catch {
				// ignore — worker may be terminating
			}
			// Reject any in-flight waiters with AbortError so consumers'
			// `for await` exits promptly rather than hanging until the
			// host's `stream-done` ack catches up.
			const s = self.streams.get(streamId);
			if (s) {
				// Release callbacks BEFORE rejecting waiters so chunks that
				// arrive after this point (in-flight `stream-chunks` posts)
				// do not fire onToken/onThinking — the abort-ordering
				// invariant (ENH-002 Task 2 §4).
				self.releaseStreamCallbacks(s);
				const err = new DOMException("chatCompletion aborted", "AbortError");
				s.errored = err;
				const waiters = s.waiters;
				s.waiters = [];
				for (const w of waiters) w.reject(err);
			}
		};
		// Wire user-supplied AbortSignal to the worker's stream-cancel
		// channel. The signal itself is not cloneable (`stripSignal` removed
		// it before postMessage); this listener bridges main-side abort to
		// worker-side cancellation.
		if (signal) {
			if (signal.aborted) {
				cancel();
			} else {
				signal.addEventListener("abort", cancel, { once: true });
			}
		}
		const iter: AsyncIterableIterator<GenerationStreamChunk> = {
			[Symbol.asyncIterator]() {
				return this;
			},
			next(): Promise<IteratorResult<GenerationStreamChunk>> {
				const s = self.streams.get(streamId);
				if (!s) {
					return Promise.resolve({ value: undefined, done: true });
				}
				if (s.queue.length > 0) {
					const value = s.queue.shift() as GenerationStreamChunk;
					return Promise.resolve({ value, done: false });
				}
				if (s.errored) {
					const err = s.errored;
					self.streams.delete(streamId);
					return Promise.reject(err);
				}
				if (s.done) {
					self.streams.delete(streamId);
					return Promise.resolve({ value: undefined, done: true });
				}
				return new Promise((resolve, reject) =>
					s.waiters.push({ resolve, reject }),
				);
			},
			return(): Promise<IteratorResult<GenerationStreamChunk>> {
				cancel();
				self.streams.delete(streamId);
				return Promise.resolve({ value: undefined, done: true });
			},
			throw(e?: unknown): Promise<IteratorResult<GenerationStreamChunk>> {
				self.streams.delete(streamId);
				return Promise.reject(e);
			},
		};
		return iter;
	}
}
