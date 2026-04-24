/** Configuration for initializing the GGML WASM bridge. */
export interface GgmlWasmConfig {
	/** URL to the Emscripten-generated JS module (e.g. "webllm-wasm.js"). */
	wasmUrl: string;
}

/** GGML tensor type enum values matching ggml.h. */
export const GgmlType = {
	F32: 0,
	F16: 1,
	Q4_0: 2,
	Q4_1: 3,
	Q5_0: 6,
	Q5_1: 7,
	Q8_0: 8,
	Q8_1: 9,
	Q2_K: 10,
	Q3_K: 11,
	Q4_K: 12,
	Q5_K: 13,
	Q6_K: 14,
	Q8_K: 15,
	I32: 26,
	BF16: 30,
} as const;

/** RoPE mode flags. */
export const RopeMode = {
	NORMAL: 0,
	NEOX: 2,
} as const;

/** Opaque handle to a ggml tensor (WASM heap pointer). */
export type TensorPtr = number;
/** Opaque handle to a ggml computation graph. */
export type GraphPtr = number;
/** Opaque handle to a ggml backend buffer. */
export type BufferPtr = number;

export interface TensorDownloadTimings {
	beginMs: number;
	waitMs: number;
	finishMs: number;
	copyMs: number;
}

export interface TensorDownloadRequest {
	readonly requestId: number;
	readonly tensor: TensorPtr;
	readonly offset: number;
	readonly byteLength: number;
	readonly timings: Readonly<TensorDownloadTimings>;
	wait(): Promise<void>;
	finish(): Promise<Uint8Array>;
	cancel(): void;
}

export interface GraphComputeProfile {
	readonly totalMs: number;
	readonly matmulMs: number | null;
	readonly attentionMs: number | null;
	readonly encodeOverheadMs: number;
	readonly dispatchCount: number;
	readonly breakdownAvailable: boolean;
}

/**
 * WebAssembly bridge for GGML inference via the ggml-webgpu backend.
 *
 * The ggml-webgpu build uses ASYNCIFY=1 (not JSPI) so that only
 * webgpu_init() is async (it calls WaitAny internally). All other
 * WASM exports (tensor ops, graph compute) are synchronous — they
 * never call Asyncify.handleSleep, so the ASYNCIFY wrapper returns
 * immediately.
 *
 * Uses stack-based memory allocation to avoid issues with malloc/free
 * in ASYNCIFY builds.
 */
export class GgmlWasm {
	// biome-ignore lint/suspicious/noExplicitAny: Emscripten module has dynamic shape
	private m: any = null;
	private initialized = false;
	private readonly asyncTensorGetWaiters = new Map<
		number,
		{
			resolve: () => void;
			reject: (error: Error) => void;
		}
	>();
	private readonly asyncTensorGetStates = new Map<number, number>();
	private graphComputeQueue: Promise<void> = Promise.resolve();

	private async enqueueGraphCompute<T>(op: () => Promise<T>): Promise<T> {
		const run = this.graphComputeQueue.then(op, op);
		this.graphComputeQueue = run.then(
			() => undefined,
			() => undefined,
		);
		return await run;
	}

	private async callWithAsyncify<T>(fn: () => T): Promise<T> {
		const previousAsync = this.m.Asyncify?.currData ?? null;
		try {
			const result = fn();
			if (
				this.m.Asyncify?.currData &&
				this.m.Asyncify.currData !== previousAsync
			) {
				return await this.m.Asyncify.whenDone();
			}
			return result;
		} catch (error) {
			if (
				this.m.Asyncify?.currData &&
				this.m.Asyncify.currData !== previousAsync
			) {
				return await this.m.Asyncify.whenDone();
			}
			throw error;
		}
	}

	private now(): number {
		return globalThis.performance?.now() ?? Date.now();
	}

	private installAsyncTensorGetNotifier(): void {
		if (this.m === null) {
			return;
		}
		this.m.__webllmNotifyAsyncTensorGet = (
			requestId: number,
			state: number,
		) => {
			const waiter = this.asyncTensorGetWaiters.get(requestId);
			if (!waiter) {
				this.asyncTensorGetStates.set(requestId, state);
				return;
			}
			this.asyncTensorGetWaiters.delete(requestId);
			if (state === 2) {
				waiter.resolve();
				return;
			}
			waiter.reject(
				new Error(
					`Tensor download request ${requestId} failed in backend state ${state}`,
				),
			);
		};
	}

	private usesAsyncTensorGetCallbacks(): boolean {
		return this.m !== null && this.backendTensorGetAsyncCallbackSupport() === 1;
	}

	private waitForAsyncTensorGetCompletion(requestId: number): Promise<void> {
		const completedState = this.asyncTensorGetStates.get(requestId);
		if (completedState !== undefined) {
			this.asyncTensorGetStates.delete(requestId);
			if (completedState === 2) {
				return Promise.resolve();
			}
			return Promise.reject(
				new Error(
					`Tensor download request ${requestId} failed in backend state ${completedState}`,
				),
			);
		}
		return new Promise<void>((resolve, reject) => {
			this.asyncTensorGetWaiters.set(requestId, { resolve, reject });
		});
	}

	private rejectAsyncTensorGetWaiter(requestId: number, error: Error): void {
		this.asyncTensorGetStates.delete(requestId);
		const waiter = this.asyncTensorGetWaiters.get(requestId);
		if (!waiter) {
			return;
		}
		this.asyncTensorGetWaiters.delete(requestId);
		waiter.reject(error);
	}

	async init(config: GgmlWasmConfig): Promise<void> {
		const factory = (await import(config.wasmUrl)).default;
		this.m = await factory({
			printErr: (text: unknown) => {
				const message = String(text);
				if (message.includes("adapter_info:")) {
					console.info(message);
					return;
				}
				console.error(message);
				// Debug: surface WASM stderr (ggml_abort writes here) to
				// channels that survive a wedged CDP session. Title reads
				// work even when `js exec` / `console read` time out during
				// a hung main thread.
				if (typeof globalThis !== "undefined") {
					const g = globalThis as unknown as {
						__wasmStderr?: string[];
						document?: { title: string };
					};
					g.__wasmStderr ??= [];
					g.__wasmStderr.push(message);
					if (g.__wasmStderr.length > 32) g.__wasmStderr.shift();
					// Latch the FIRST non-adapter-info stderr line into the
					// title. Later lines (stack frames, Emscripten's
					// "Aborted()" wrapper) would otherwise overwrite the
					// GGML assertion that names the failure site.
					if (g.document && !g.document.title.startsWith("[wasm]")) {
						g.document.title = `[wasm] ${message}`.slice(0, 300);
					}
				}
			},
		});
		this.installAsyncTensorGetNotifier();
		const result = await this.callWithAsyncify<number>(() =>
			this.m._webgpu_init(),
		);
		if (result !== 0) {
			throw new Error(`WebGPU backend init failed (code ${result})`);
		}
		this.initialized = true;
	}

	async shutdown(): Promise<void> {
		if (!this.initialized) return;
		this.m._webgpu_shutdown();
		for (const [requestId, waiter] of this.asyncTensorGetWaiters) {
			waiter.reject(
				new Error(
					`Tensor download request ${requestId} was interrupted by shutdown`,
				),
			);
		}
		this.asyncTensorGetWaiters.clear();
		this.asyncTensorGetStates.clear();
		this.initialized = false;
		this.m = null;
	}

	// ── Memory helpers ─────────────────────────────────────────────────

	malloc(size: number): number {
		return this.m._malloc(size);
	}

	free(ptr: number): void {
		this.m._free(ptr);
	}

	stackSave(): number {
		return this.m.stackSave();
	}

	stackRestore(sp: number): void {
		this.m.stackRestore(sp);
	}

	stackAlloc(size: number): number {
		return this.m.stackAlloc(size);
	}

	withStack<T>(size: number, fn: (ptr: number) => T): T {
		const sp = this.m.stackSave();
		try {
			return fn(this.m.stackAlloc(size));
		} finally {
			this.m.stackRestore(sp);
		}
	}

	get heapU8(): Uint8Array {
		return this.m.HEAPU8;
	}

	get heapF32(): Float32Array {
		return this.m.HEAPF32;
	}

	// ── Context ──────────────────────────────────────────────────────────

	ctxCreate(memSize: number): number {
		const rc = this.m._ctx_create(memSize);
		if (rc < 0) throw new Error(`ctx_create failed (${rc})`);
		return rc;
	}

	ctxFree(): void {
		this.m._ctx_free();
	}

	// ── Tensor creation ──────────────────────────────────────────────────

	tensorNew1d(type: number, ne0: number): TensorPtr {
		return this.m._tensor_new_1d(type, ne0);
	}

	tensorNew2d(type: number, ne0: number, ne1: number): TensorPtr {
		return this.m._tensor_new_2d(type, ne0, ne1);
	}

	tensorNew3d(type: number, ne0: number, ne1: number, ne2: number): TensorPtr {
		return this.m._tensor_new_3d(type, ne0, ne1, ne2);
	}

	tensorNew4d(
		type: number,
		ne0: number,
		ne1: number,
		ne2: number,
		ne3: number,
	): TensorPtr {
		return this.m._tensor_new_4d(type, ne0, ne1, ne2, ne3);
	}

	tensorSetName(_tensor: TensorPtr, _name: string): void {
		// Skip naming — tensor names are only used for debugging.
		// _tensor_set_name requires stack alloc + WASM call which is
		// incompatible with ASYNCIFY. Tracked JS-side in nameToTensor.
	}

	// ── Tensor properties ────────────────────────────────────────────────

	tensorNelements(tensor: TensorPtr): number {
		return this.m._tensor_nelements(tensor);
	}

	tensorNbytes(tensor: TensorPtr): number {
		return this.m._tensor_nbytes(tensor);
	}

	tensorType(tensor: TensorPtr): number {
		return this.m._tensor_type(tensor);
	}

	tensorNe(tensor: TensorPtr, dim: number): number {
		return this.m._tensor_ne(tensor, dim);
	}

	tensorNb(tensor: TensorPtr, dim: number): number {
		return this.m._tensor_nb(tensor, dim);
	}

	tensorData(tensor: TensorPtr): number {
		return this.m._tensor_data(tensor);
	}

	// ── Tensor data I/O ──────────────────────────────────────────────────

	tensorSetData(tensor: TensorPtr, srcHeapPtr: number, size: number): void {
		this.m._tensor_set_data(tensor, srcHeapPtr, size);
	}

	tensorGetData(tensor: TensorPtr, dstHeapPtr: number, size: number): void {
		this.m._tensor_get_data(tensor, dstHeapPtr, size);
	}

	/** Upload bytes from JS ArrayBuffer to a GPU tensor via heap buffer. */
	uploadToTensor(tensor: TensorPtr, data: Uint8Array, offset = 0): void {
		const ptr = this.malloc(data.byteLength);
		try {
			this.heapU8.set(data, ptr);
			this.m._backend_tensor_set(tensor, ptr, offset, data.byteLength);
		} finally {
			this.free(ptr);
		}
	}

	/** Upload bytes chunk-by-chunk to avoid large heap allocations. */
	uploadToTensorChunked(
		tensor: TensorPtr,
		data: Uint8Array,
		chunkSize = 4 * 1024 * 1024,
	): void {
		const total = data.byteLength;
		const ptr = this.malloc(Math.min(chunkSize, total));
		try {
			for (let off = 0; off < total; off += chunkSize) {
				const end = Math.min(off + chunkSize, total);
				const slice = data.subarray(off, end);
				this.heapU8.set(slice, ptr);
				this.m._backend_tensor_set(tensor, ptr, off, slice.byteLength);
			}
		} finally {
			this.free(ptr);
		}
	}

	beginDownloadFromTensor(
		tensor: TensorPtr,
		byteLength: number,
		offset = 0,
	): TensorDownloadRequest {
		this.installAsyncTensorGetNotifier();
		const ptr = this.malloc(byteLength);
		const timings: TensorDownloadTimings = {
			beginMs: 0,
			waitMs: 0,
			finishMs: 0,
			copyMs: 0,
		};
		const beginStart = this.now();
		const requestId = this.backendTensorGetAsyncBegin(
			tensor,
			offset,
			byteLength,
		);
		timings.beginMs = this.now() - beginStart;

		let done = false;
		let cancelled = false;
		let waitPromise: Promise<void> | null = null;
		let result: Uint8Array | null = null;

		const releaseHeap = () => {
			if (done) return;
			done = true;
			this.free(ptr);
		};

		const cancelInternal = () => {
			if (cancelled || done) return;
			cancelled = true;
			this.rejectAsyncTensorGetWaiter(
				requestId,
				new Error(`Tensor download request ${requestId} was cancelled`),
			);
			try {
				this.backendTensorGetAsyncCancel(requestId);
			} catch {
				// Best effort cancellation only.
			} finally {
				releaseHeap();
			}
		};

		const ensureActive = () => {
			if (cancelled) {
				throw new Error(`Tensor download request ${requestId} was cancelled`);
			}
		};

		const wait = async () => {
			ensureActive();
			if (waitPromise !== null) {
				await waitPromise;
				return;
			}
			waitPromise = (async () => {
				const waitStart = this.now();
				if (this.usesAsyncTensorGetCallbacks()) {
					await this.waitForAsyncTensorGetCompletion(requestId);
				} else {
					while (this.backendTensorGetAsyncPoll(requestId) === 0) {
						await new Promise<void>((resolve) => setTimeout(resolve, 1));
					}
				}
				timings.waitMs = this.now() - waitStart;
			})();
			await waitPromise;
		};

		return {
			requestId,
			tensor,
			offset,
			byteLength,
			timings,
			wait,
			finish: async () => {
				ensureActive();
				if (result !== null) {
					return result;
				}
				try {
					await wait();
					ensureActive();
					const finishStart = this.now();
					this.backendTensorGetAsyncFinish(requestId, ptr, byteLength);
					timings.finishMs = this.now() - finishStart;
					const copyStart = this.now();
					result = new Uint8Array(this.heapU8.buffer, ptr, byteLength).slice();
					timings.copyMs = this.now() - copyStart;
					return result;
				} catch (error) {
					cancelInternal();
					throw error;
				} finally {
					releaseHeap();
				}
			},
			cancel: () => {
				cancelInternal();
			},
		};
	}

	/** Download tensor data from GPU to a new Uint8Array via heap buffer. */
	async downloadFromTensor(
		tensor: TensorPtr,
		byteLength: number,
		offset = 0,
	): Promise<Uint8Array> {
		const request = this.beginDownloadFromTensor(tensor, byteLength, offset);
		try {
			return await request.finish();
		} catch (error) {
			request.cancel();
			throw error;
		}
	}

	// ── Graph operations ─────────────────────────────────────────────────

	opMulMat(a: TensorPtr, b: TensorPtr): TensorPtr {
		return this.m._op_mul_mat(a, b);
	}

	opAdd(a: TensorPtr, b: TensorPtr): TensorPtr {
		return this.m._op_add(a, b);
	}

	opMul(a: TensorPtr, b: TensorPtr): TensorPtr {
		return this.m._op_mul(a, b);
	}

	opRmsNorm(x: TensorPtr, eps: number): TensorPtr {
		return this.m._op_rms_norm(x, eps);
	}

	opSilu(x: TensorPtr): TensorPtr {
		return this.m._op_silu(x);
	}

	opGelu(x: TensorPtr): TensorPtr {
		return this.m._op_gelu(x);
	}

	opRope(
		x: TensorPtr,
		pos: TensorPtr,
		nDims: number,
		mode: number,
		nCtxOrig: number,
		freqBase: number,
		freqScale: number,
		extFactor: number,
		attnFactor: number,
		betaFast: number,
		betaSlow: number,
	): TensorPtr {
		return this.m._op_rope(
			x,
			pos,
			nDims,
			mode,
			nCtxOrig,
			freqBase,
			freqScale,
			extFactor,
			attnFactor,
			betaFast,
			betaSlow,
		);
	}

	opReshape2d(x: TensorPtr, ne0: number, ne1: number): TensorPtr {
		return this.m._op_reshape_2d(x, ne0, ne1);
	}

	opReshape3d(x: TensorPtr, ne0: number, ne1: number, ne2: number): TensorPtr {
		return this.m._op_reshape_3d(x, ne0, ne1, ne2);
	}

	opPermute(
		x: TensorPtr,
		d0: number,
		d1: number,
		d2: number,
		d3: number,
	): TensorPtr {
		return this.m._op_permute(x, d0, d1, d2, d3);
	}

	opCont(x: TensorPtr): TensorPtr {
		return this.m._op_cont(x);
	}

	opView2d(
		x: TensorPtr,
		ne0: number,
		ne1: number,
		nb1: number,
		offset: number,
	): TensorPtr {
		return this.m._op_view_2d(x, ne0, ne1, nb1, offset);
	}

	opView3d(
		x: TensorPtr,
		ne0: number,
		ne1: number,
		ne2: number,
		nb1: number,
		nb2: number,
		offset: number,
	): TensorPtr {
		return this.m._op_view_3d(x, ne0, ne1, ne2, nb1, nb2, offset);
	}

	opCpy(src: TensorPtr, dst: TensorPtr): TensorPtr {
		return this.m._op_cpy(src, dst);
	}

	opSoftMax(x: TensorPtr): TensorPtr {
		return this.m._op_soft_max(x);
	}

	/** Fused SwiGLU for LLaMA FFN: silu(a) * b in one op. */
	opSwigluSplit(a: TensorPtr, b: TensorPtr): TensorPtr {
		return this.m._op_swiglu_split(a, b);
	}

	/**
	 * Fused scale + mask + softmax.
	 * @param mask - F32 tensor [ne0, ne1] broadcast over higher dims. 0 = visible,
	 *               -Infinity = masked. Pass 0 for no mask.
	 * @param scale - Multiplier applied to x before softmax.
	 * @param maxBias - ALiBi max bias. Pass 0 for standard causal attention.
	 */
	opSoftMaxExt(
		x: TensorPtr,
		mask: TensorPtr,
		scale: number,
		maxBias: number,
	): TensorPtr {
		return this.m._op_soft_max_ext(x, mask, scale, maxBias);
	}

	opScale(x: TensorPtr, s: number): TensorPtr {
		return this.m._op_scale(x, s);
	}

	opRepeat(x: TensorPtr, y: TensorPtr): TensorPtr {
		return this.m._op_repeat(x, y);
	}

	opGetRows(a: TensorPtr, b: TensorPtr): TensorPtr {
		return this.m._op_get_rows(a, b);
	}

	opArgmax(src: TensorPtr): TensorPtr {
		return this.m._op_argmax(src);
	}

	opTopK(src: TensorPtr, k: number): TensorPtr {
		return this.m._op_top_k(src, k);
	}

	opDiagMaskInf(x: TensorPtr, nPast: number): TensorPtr {
		return this.m._op_diag_mask_inf(x, nPast);
	}

	opNorm(x: TensorPtr, eps: number): TensorPtr {
		return this.m._op_norm(x, eps);
	}

	// ── Graph compute ────────────────────────────────────────────────────

	graphNew(size: number): GraphPtr {
		return this.m._graph_new(size);
	}

	graphBuildForwardExpand(graph: GraphPtr, tensor: TensorPtr): void {
		this.m._graph_build_forward_expand(graph, tensor);
	}

	async graphCompute(graph: GraphPtr): Promise<number> {
		return await this.enqueueGraphCompute(() =>
			this.callWithAsyncify<number>(() => this.m._graph_compute(graph)),
		);
	}

	setDetailedGraphComputeProfilingEnabled(enabled: boolean): void {
		this.m._webgpu_set_graph_profiling_enabled?.(enabled ? 1 : 0);
	}

	async graphComputeWithDetailedProfile(graph: GraphPtr): Promise<number> {
		return await this.enqueueGraphCompute(async () => {
			this.setDetailedGraphComputeProfilingEnabled(true);
			try {
				return await this.callWithAsyncify<number>(() =>
					this.m._graph_compute(graph),
				);
			} finally {
				this.setDetailedGraphComputeProfilingEnabled(false);
			}
		});
	}

	getLastGraphComputeProfile(): GraphComputeProfile | null {
		if (this.m._webgpu_last_graph_profile_valid?.() !== 1) {
			return null;
		}
		const breakdownAvailable =
			this.m._webgpu_last_graph_profile_breakdown_available?.() === 1;
		return {
			totalMs: this.m._webgpu_last_graph_profile_total_ms(),
			matmulMs: breakdownAvailable
				? this.m._webgpu_last_graph_profile_matmul_ms()
				: null,
			attentionMs: breakdownAvailable
				? this.m._webgpu_last_graph_profile_attention_ms()
				: null,
			encodeOverheadMs: this.m._webgpu_last_graph_profile_encode_overhead_ms(),
			dispatchCount: this.m._webgpu_last_graph_profile_dispatch_count(),
			breakdownAvailable,
		};
	}

	// ── Backend buffer ───────────────────────────────────────────────────

	backendAllocCtxTensors(): BufferPtr {
		return this.m._backend_alloc_ctx_tensors();
	}

	backendBufferFree(buffer: BufferPtr): void {
		this.m._backend_buffer_free(buffer);
	}

	backendTensorSet(
		tensor: TensorPtr,
		srcHeapPtr: number,
		offset: number,
		size: number,
	): void {
		this.m._backend_tensor_set(tensor, srcHeapPtr, offset, size);
	}

	/**
	 * Write up to three backend tensors in a single WASM call. Designed for
	 * per-forward leaf inputs (pos, tokenIds, mask) to save FFI hops. Pass a
	 * tensor pointer of 0 to skip that slot — e.g. skip the mask when it's
	 * not needed on single-token decode.
	 */
	backendTensorSet3(
		t1: TensorPtr,
		d1: number,
		sz1: number,
		t2: TensorPtr,
		d2: number,
		sz2: number,
		t3: TensorPtr,
		d3: number,
		sz3: number,
	): void {
		this.m._backend_tensor_set3(t1, d1, sz1, t2, d2, sz2, t3, d3, sz3);
	}

	async backendTensorGet(
		tensor: TensorPtr,
		dstHeapPtr: number,
		offset: number,
		size: number,
	): Promise<void> {
		await this.callWithAsyncify<void>(() =>
			this.m._backend_tensor_get(tensor, dstHeapPtr, offset, size),
		);
	}

	backendTensorGetAsyncBegin(
		tensor: TensorPtr,
		offset: number,
		size: number,
	): number {
		return this.m._backend_tensor_get_async_begin(tensor, offset, size);
	}

	backendTensorGetAsyncPoll(requestId: number): number {
		return this.m._backend_tensor_get_async_poll(requestId);
	}

	backendTensorGetAsyncFinish(
		requestId: number,
		dstHeapPtr: number,
		size: number,
	): void {
		this.m._backend_tensor_get_async_finish(requestId, dstHeapPtr, size);
	}

	backendTensorGetAsyncCancel(requestId: number): void {
		this.m._backend_tensor_get_async_cancel(requestId);
	}

	backendTensorGetAsyncCallbackSupport(): number {
		return this.m._backend_tensor_get_async_callback_support?.() ?? 0;
	}

	backendTensorAlignment(): number {
		return this.m._backend_tensor_alignment();
	}
}
