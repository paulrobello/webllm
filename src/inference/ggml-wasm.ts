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

/** Precision hint for ggml_flash_attn_ext_set_prec. */
export const GgmlPrec = {
	DEFAULT: 0,
	F32: 10,
} as const;
export type GgmlPrec = (typeof GgmlPrec)[keyof typeof GgmlPrec];

/** Byte size of an fp32 matmul output element. Used by encoders for view3d byte arithmetic. */
export const F32_BYTES = 4;

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

	/**
	 * Set during `init()`. `true` if the loaded module was built with
	 * `-sMEMORY64=1 -sWASM_BIGINT=1` — pointer-returning custom exports
	 * (`_bridge_malloc`, `_tensor_new_*`, etc.) return BigInt values
	 * that the wrappers narrow to `number` because no single allocation
	 * in this codebase exceeds 2^53 bytes (largest tensor at 30B IQ3_M
	 * ≈ 850 MB; full 13B Q4_K_S model file ≈ 7.4 GiB ≪ 2^53). See the
	 * MEMORY64 migration plan for the cap analysis.
	 */
	private is64 = false;

	/**
	 * Pointer/size_t boundary discipline.
	 *
	 * Under MEMORY64 + WASM_BIGINT, every `void*` and `size_t` arg is i64
	 * and must be a BigInt; the corresponding return values come back as
	 * BigInt and we narrow with `Number(...)`. No single allocation in
	 * this codebase exceeds 2^53 bytes (largest tensor at 30B IQ3_M
	 * ≈ 850 MB; full 13B Q4_K_S model ≈ 7.4 GiB ≪ 2^53), so the narrow
	 * is safe.
	 *
	 * Under wasm32 every `void*` and `size_t` is i32. The `>>> 0` coerces
	 * a signed i32 return to unsigned uint32 — the Emscripten linker only
	 * emits this for known-pointer-typed exports (`_malloc` / `_free`);
	 * custom exports like `_bridge_malloc`, `_tensor_new_*`, and `_op_*`
	 * (returning `void*`) are not in that list, so JS sees the raw signed
	 * i32 and any address ≥ 2^31 surfaces as a negative Number. That
	 * breaks `Uint8Array.set(_, offset)` for any 7B+ model where the
	 * heap fills past 2 GiB before the upload scratch malloc lands in
	 * the upper half. Phase 1 of the MEMORY64 migration switched these
	 * calls from `_malloc` to `_bridge_malloc` and inherited the bug;
	 * the inline `>>> 0` at every pointer-return call site is the fix.
	 *
	 * The is64 branch is inlined at every call site (rather than
	 * factored into helper methods) because the wrapped bridge ops are
	 * microsecond-scale and per-call helper dispatch dominated dispatch-
	 * heavy hot paths in profiling (see PHASE-5-PARITY.md).
	 */

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
		// Detect wasm32 vs wasm64 ABI shape: under MEMORY64 + WASM_BIGINT,
		// custom-export pointer returns are BigInt and pointer args must
		// be BigInt; under wasm32 they're Number. Try the wasm32 (Number)
		// shape first — under wasm64 the call throws TypeError because
		// the i64 arg can't accept a JS Number. The catch path retries
		// with BigInt and pins is64=true.
		try {
			const probe = this.m._bridge_malloc(0);
			this.is64 = typeof probe === "bigint";
			this.m._bridge_free(probe);
		} catch {
			const probe = this.m._bridge_malloc(0n);
			this.is64 = true;
			this.m._bridge_free(probe);
		}
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
		if (this.is64) {
			const ptr = this.m._bridge_malloc(BigInt(size));
			return Number(ptr);
		}
		// `>>> 0`: see `num()` — Emscripten's linker doesn't emit unsigned
		// coercion for custom-export pointer returns, so any malloc landing
		// above 2 GiB (common for 7B+ models) would otherwise return as a
		// negative JS Number and break Uint8Array.set offsets.
		return this.m._bridge_malloc(size) >>> 0;
	}

	free(ptr: number): void {
		if (this.is64) {
			this.m._bridge_free(BigInt(ptr));
			return;
		}
		this.m._bridge_free(ptr);
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
		// _ctx_create takes a size_t (i64 under MEMORY64) and returns int32_t.
		const rc = this.is64
			? this.m._ctx_create(BigInt(memSize))
			: this.m._ctx_create(memSize);
		if (rc < 0) throw new Error(`ctx_create failed (${rc})`);
		return rc;
	}

	ctxFree(): void {
		this.m._ctx_free();
	}

	// ── Tensor creation ──────────────────────────────────────────────────

	tensorNew1d(type: number, ne0: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._tensor_new_1d(type, ne0));
		}
		return this.m._tensor_new_1d(type, ne0) >>> 0;
	}

	tensorNew2d(type: number, ne0: number, ne1: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._tensor_new_2d(type, ne0, ne1));
		}
		return this.m._tensor_new_2d(type, ne0, ne1) >>> 0;
	}

	tensorNew3d(type: number, ne0: number, ne1: number, ne2: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._tensor_new_3d(type, ne0, ne1, ne2));
		}
		return this.m._tensor_new_3d(type, ne0, ne1, ne2) >>> 0;
	}

	tensorNew4d(
		type: number,
		ne0: number,
		ne1: number,
		ne2: number,
		ne3: number,
	): TensorPtr {
		if (this.is64) {
			return Number(this.m._tensor_new_4d(type, ne0, ne1, ne2, ne3));
		}
		return this.m._tensor_new_4d(type, ne0, ne1, ne2, ne3) >>> 0;
	}

	tensorSetName(_tensor: TensorPtr, _name: string): void {
		// Skip naming — tensor names are only used for debugging.
		// _tensor_set_name requires stack alloc + WASM call which is
		// incompatible with ASYNCIFY. Tracked JS-side in nameToTensor.
	}

	// ── Tensor properties ────────────────────────────────────────────────

	tensorNelements(tensor: TensorPtr): number {
		if (this.is64) {
			return this.m._tensor_nelements(BigInt(tensor));
		}
		return this.m._tensor_nelements(tensor);
	}

	tensorNbytes(tensor: TensorPtr): number {
		if (this.is64) {
			return this.m._tensor_nbytes(BigInt(tensor));
		}
		return this.m._tensor_nbytes(tensor);
	}

	tensorType(tensor: TensorPtr): number {
		if (this.is64) {
			return this.m._tensor_type(BigInt(tensor));
		}
		return this.m._tensor_type(tensor);
	}

	tensorNe(tensor: TensorPtr, dim: number): number {
		if (this.is64) {
			return this.m._tensor_ne(BigInt(tensor), dim);
		}
		return this.m._tensor_ne(tensor, dim);
	}

	tensorNb(tensor: TensorPtr, dim: number): number {
		if (this.is64) {
			return this.m._tensor_nb(BigInt(tensor), dim);
		}
		return this.m._tensor_nb(tensor, dim);
	}

	tensorData(tensor: TensorPtr): number {
		if (this.is64) {
			return Number(this.m._tensor_data(BigInt(tensor)));
		}
		return this.m._tensor_data(tensor) >>> 0;
	}

	// ── Tensor data I/O ──────────────────────────────────────────────────

	tensorSetData(tensor: TensorPtr, srcHeapPtr: number, size: number): void {
		if (this.is64) {
			this.m._tensor_set_data(BigInt(tensor), BigInt(srcHeapPtr), BigInt(size));
			return;
		}
		this.m._tensor_set_data(tensor, srcHeapPtr, size);
	}

	tensorGetData(tensor: TensorPtr, dstHeapPtr: number, size: number): void {
		if (this.is64) {
			this.m._tensor_get_data(BigInt(tensor), BigInt(dstHeapPtr), BigInt(size));
			return;
		}
		this.m._tensor_get_data(tensor, dstHeapPtr, size);
	}

	/** Upload bytes from JS ArrayBuffer to a GPU tensor via heap buffer. */
	uploadToTensor(tensor: TensorPtr, data: Uint8Array, offset = 0): void {
		const ptr = this.malloc(data.byteLength);
		try {
			this.heapU8.set(data, ptr);
			if (this.is64) {
				this.m._backend_tensor_set(
					BigInt(tensor),
					BigInt(ptr),
					BigInt(offset),
					BigInt(data.byteLength),
				);
			} else {
				this.m._backend_tensor_set(tensor, ptr, offset, data.byteLength);
			}
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
			if (this.is64) {
				const tensorArg = BigInt(tensor);
				const ptrArg = BigInt(ptr);
				for (let off = 0; off < total; off += chunkSize) {
					const end = Math.min(off + chunkSize, total);
					const slice = data.subarray(off, end);
					this.heapU8.set(slice, ptr);
					this.m._backend_tensor_set(
						tensorArg,
						ptrArg,
						BigInt(off),
						BigInt(slice.byteLength),
					);
				}
			} else {
				for (let off = 0; off < total; off += chunkSize) {
					const end = Math.min(off + chunkSize, total);
					const slice = data.subarray(off, end);
					this.heapU8.set(slice, ptr);
					this.m._backend_tensor_set(tensor, ptr, off, slice.byteLength);
				}
			}
		} finally {
			this.free(ptr);
		}
	}

	/** Upload `byteLength` bytes via a callback-resolved source, chunk by
	 * chunk. The callback is invoked once per chunk *after* the scratch
	 * malloc, so any heap growth from that malloc can't detach the source
	 * view between derivation and `set`. Use this when source bytes live
	 * in the WASM heap (where they share the buffer that may grow mid-
	 * upload). Static JS-heap sources should use `uploadToTensorChunked`.
	 */
	uploadRangeChunked(
		tensor: TensorPtr,
		dataAt: (srcOffset: number, byteLength: number) => Uint8Array,
		byteLength: number,
		chunkSize = 4 * 1024 * 1024,
	): void {
		const ptr = this.malloc(Math.min(chunkSize, byteLength));
		try {
			if (this.is64) {
				const tensorArg = BigInt(tensor);
				const ptrArg = BigInt(ptr);
				for (let off = 0; off < byteLength; off += chunkSize) {
					const end = Math.min(off + chunkSize, byteLength);
					const slice = dataAt(off, end - off);
					this.heapU8.set(slice, ptr);
					this.m._backend_tensor_set(
						tensorArg,
						ptrArg,
						BigInt(off),
						BigInt(slice.byteLength),
					);
				}
			} else {
				for (let off = 0; off < byteLength; off += chunkSize) {
					const end = Math.min(off + chunkSize, byteLength);
					const slice = dataAt(off, end - off);
					this.heapU8.set(slice, ptr);
					this.m._backend_tensor_set(tensor, ptr, off, slice.byteLength);
				}
			}
		} finally {
			this.free(ptr);
		}
	}

	async beginDownloadFromTensor(
		tensor: TensorPtr,
		byteLength: number,
		offset = 0,
	): Promise<TensorDownloadRequest> {
		this.installAsyncTensorGetNotifier();
		const ptr = this.malloc(byteLength);
		const timings: TensorDownloadTimings = {
			beginMs: 0,
			waitMs: 0,
			finishMs: 0,
			copyMs: 0,
		};
		const beginStart = this.now();
		const requestId = await this.backendTensorGetAsyncBegin(
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
			// Cancel returns a Promise under JSPI; fire-and-forget is fine
			// here since cancellation is best-effort and we don't block on it.
			this.backendTensorGetAsyncCancel(requestId).catch(() => {
				/* ignore */
			});
			releaseHeap();
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
					while ((await this.backendTensorGetAsyncPoll(requestId)) === 0) {
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
					await this.backendTensorGetAsyncFinish(requestId, ptr, byteLength);
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
		const request = await this.beginDownloadFromTensor(
			tensor,
			byteLength,
			offset,
		);
		try {
			return await request.finish();
		} catch (error) {
			request.cancel();
			throw error;
		}
	}

	// ── Graph operations ─────────────────────────────────────────────────

	opMulMat(a: TensorPtr, b: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_mul_mat(BigInt(a), BigInt(b)));
		}
		return this.m._op_mul_mat(a, b) >>> 0;
	}

	opAdd(a: TensorPtr, b: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_add(BigInt(a), BigInt(b)));
		}
		return this.m._op_add(a, b) >>> 0;
	}

	opMul(a: TensorPtr, b: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_mul(BigInt(a), BigInt(b)));
		}
		return this.m._op_mul(a, b) >>> 0;
	}

	opRmsNorm(x: TensorPtr, eps: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_rms_norm(BigInt(x), eps));
		}
		return this.m._op_rms_norm(x, eps) >>> 0;
	}

	opSilu(x: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_silu(BigInt(x)));
		}
		return this.m._op_silu(x) >>> 0;
	}

	opGelu(x: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_gelu(BigInt(x)));
		}
		return this.m._op_gelu(x) >>> 0;
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
		if (this.is64) {
			return Number(
				this.m._op_rope(
					BigInt(x),
					BigInt(pos),
					nDims,
					mode,
					nCtxOrig,
					freqBase,
					freqScale,
					extFactor,
					attnFactor,
					betaFast,
					betaSlow,
				),
			);
		}
		return (
			this.m._op_rope(
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
			) >>> 0
		);
	}

	opReshape2d(x: TensorPtr, ne0: number, ne1: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_reshape_2d(BigInt(x), ne0, ne1));
		}
		return this.m._op_reshape_2d(x, ne0, ne1) >>> 0;
	}

	opReshape3d(x: TensorPtr, ne0: number, ne1: number, ne2: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_reshape_3d(BigInt(x), ne0, ne1, ne2));
		}
		return this.m._op_reshape_3d(x, ne0, ne1, ne2) >>> 0;
	}

	opPermute(
		x: TensorPtr,
		d0: number,
		d1: number,
		d2: number,
		d3: number,
	): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_permute(BigInt(x), d0, d1, d2, d3));
		}
		return this.m._op_permute(x, d0, d1, d2, d3) >>> 0;
	}

	opCont(x: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_cont(BigInt(x)));
		}
		return this.m._op_cont(x) >>> 0;
	}

	opView2d(
		x: TensorPtr,
		ne0: number,
		ne1: number,
		nb1: number,
		offset: number,
	): TensorPtr {
		// offset is size_t (i64 under wasm64); ne0/ne1/nb1 stay int32_t.
		if (this.is64) {
			return Number(
				this.m._op_view_2d(BigInt(x), ne0, ne1, nb1, BigInt(offset)),
			);
		}
		return this.m._op_view_2d(x, ne0, ne1, nb1, offset) >>> 0;
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
		if (this.is64) {
			return Number(
				this.m._op_view_3d(BigInt(x), ne0, ne1, ne2, nb1, nb2, BigInt(offset)),
			);
		}
		return this.m._op_view_3d(x, ne0, ne1, ne2, nb1, nb2, offset) >>> 0;
	}

	opCpy(src: TensorPtr, dst: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_cpy(BigInt(src), BigInt(dst)));
		}
		return this.m._op_cpy(src, dst) >>> 0;
	}

	opSoftMax(x: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_soft_max(BigInt(x)));
		}
		return this.m._op_soft_max(x) >>> 0;
	}

	/** Fused SwiGLU for LLaMA FFN: silu(a) * b in one op. */
	opSwigluSplit(a: TensorPtr, b: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_swiglu_split(BigInt(a), BigInt(b)));
		}
		return this.m._op_swiglu_split(a, b) >>> 0;
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
		if (this.is64) {
			return Number(
				this.m._op_soft_max_ext(BigInt(x), BigInt(mask), scale, maxBias),
			);
		}
		return this.m._op_soft_max_ext(x, mask, scale, maxBias) >>> 0;
	}

	/**
	 * Fused scaled-dot-product attention (FLASH_ATTN_EXT).
	 *
	 * Replaces the manual opMulMat(K,Q) + opSoftMaxExt + opMulMat(V,attn) chain
	 * with a single fused op. The ggml-webgpu backend will route this to its
	 * VEC or TILE shader path at decode shapes (N=1, head_dim%32==0, K F16
	 * with vec4-aligned data offset) — `flash_attn_get_decisions` in
	 * ggml-webgpu-shader-lib.hpp picks the path.
	 *
	 * @param q     Query  [head_dim, n_tokens,    n_head]    F32
	 * @param k     Key    [head_dim, n_kv,        n_head_kv] F16 / Q4_0 / Q8_0
	 * @param v     Value  [head_dim, n_kv,        n_head_kv] same dtype as k
	 * @param mask  F16 [n_kv_padded, n_tokens] broadcast over heads (-inf masked,
	 *              0 visible). Pass 0 (null) for no mask.
	 * @param scale Typically 1/sqrt(head_dim).
	 * @param maxBias ALiBi max bias; 0 for standard causal attention.
	 * @param logitSoftcap Gemma-style soft-cap; 0 for standard models.
	 * @returns [head_dim, n_head, n_tokens] — caller must permute/reshape to
	 *          merge heads back into the residual stream.
	 */
	opFlashAttn(
		q: TensorPtr,
		k: TensorPtr,
		v: TensorPtr,
		mask: TensorPtr,
		scale: number,
		maxBias: number,
		logitSoftcap: number,
	): TensorPtr {
		if (this.is64) {
			return Number(
				this.m._op_flash_attn_ext(
					BigInt(q),
					BigInt(k),
					BigInt(v),
					BigInt(mask),
					scale,
					maxBias,
					logitSoftcap,
				),
			);
		}
		return (
			this.m._op_flash_attn_ext(q, k, v, mask, scale, maxBias, logitSoftcap) >>>
			0
		);
	}

	/** Pin the FA accumulator precision (e.g. GgmlPrec.F32 for higher precision). */
	opFlashAttnSetPrec(a: TensorPtr, prec: GgmlPrec): void {
		if (this.is64) {
			this.m._op_flash_attn_ext_set_prec(BigInt(a), prec);
			return;
		}
		this.m._op_flash_attn_ext_set_prec(a, prec);
	}

	/** Attach attention sinks (used by some Phi-3 / Qwen variants). Pass 0 for none. */
	opFlashAttnAddSinks(a: TensorPtr, sinks: TensorPtr): void {
		if (this.is64) {
			this.m._op_flash_attn_ext_add_sinks(BigInt(a), BigInt(sinks));
			return;
		}
		this.m._op_flash_attn_ext_add_sinks(a, sinks);
	}

	opScale(x: TensorPtr, s: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_scale(BigInt(x), s));
		}
		return this.m._op_scale(x, s) >>> 0;
	}

	opRepeat(x: TensorPtr, y: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_repeat(BigInt(x), BigInt(y)));
		}
		return this.m._op_repeat(x, y) >>> 0;
	}

	opGetRows(a: TensorPtr, b: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_get_rows(BigInt(a), BigInt(b)));
		}
		return this.m._op_get_rows(a, b) >>> 0;
	}

	opArgmax(src: TensorPtr): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_argmax(BigInt(src)));
		}
		return this.m._op_argmax(src) >>> 0;
	}

	opTopK(src: TensorPtr, k: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_top_k(BigInt(src), k));
		}
		return this.m._op_top_k(src, k) >>> 0;
	}

	opDiagMaskInf(x: TensorPtr, nPast: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_diag_mask_inf(BigInt(x), nPast));
		}
		return this.m._op_diag_mask_inf(x, nPast) >>> 0;
	}

	opNorm(x: TensorPtr, eps: number): TensorPtr {
		if (this.is64) {
			return Number(this.m._op_norm(BigInt(x), eps));
		}
		return this.m._op_norm(x, eps) >>> 0;
	}

	// ── Graph compute ────────────────────────────────────────────────────

	graphNew(size: number): GraphPtr {
		// _graph_new takes size_t (i64 under wasm64) and returns void* graph ptr.
		if (this.is64) {
			return Number(this.m._graph_new(BigInt(size)));
		}
		return this.m._graph_new(size) >>> 0;
	}

	graphBuildForwardExpand(graph: GraphPtr, tensor: TensorPtr): void {
		if (this.is64) {
			this.m._graph_build_forward_expand(BigInt(graph), BigInt(tensor));
			return;
		}
		this.m._graph_build_forward_expand(graph, tensor);
	}

	async graphCompute(graph: GraphPtr): Promise<number> {
		const graphArg: number | bigint = this.is64 ? BigInt(graph) : graph;
		return await this.enqueueGraphCompute(() =>
			this.callWithAsyncify<number>(() => this.m._graph_compute(graphArg)),
		);
	}

	setDetailedGraphComputeProfilingEnabled(enabled: boolean): void {
		this.m._webgpu_set_graph_profiling_enabled?.(enabled ? 1 : 0);
	}

	async graphComputeWithDetailedProfile(graph: GraphPtr): Promise<number> {
		const graphArg: number | bigint = this.is64 ? BigInt(graph) : graph;
		return await this.enqueueGraphCompute(async () => {
			this.setDetailedGraphComputeProfilingEnabled(true);
			try {
				return await this.callWithAsyncify<number>(() =>
					this.m._graph_compute(graphArg),
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
		if (this.is64) {
			return Number(this.m._backend_alloc_ctx_tensors());
		}
		return this.m._backend_alloc_ctx_tensors() >>> 0;
	}

	backendBufferFree(buffer: BufferPtr): void {
		if (this.is64) {
			this.m._backend_buffer_free(BigInt(buffer));
			return;
		}
		this.m._backend_buffer_free(buffer);
	}

	backendTensorSet(
		tensor: TensorPtr,
		srcHeapPtr: number,
		offset: number,
		size: number,
	): void {
		if (this.is64) {
			this.m._backend_tensor_set(
				BigInt(tensor),
				BigInt(srcHeapPtr),
				BigInt(offset),
				BigInt(size),
			);
			return;
		}
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
		if (this.is64) {
			this.m._backend_tensor_set3(
				BigInt(t1),
				BigInt(d1),
				BigInt(sz1),
				BigInt(t2),
				BigInt(d2),
				BigInt(sz2),
				BigInt(t3),
				BigInt(d3),
				BigInt(sz3),
			);
			return;
		}
		this.m._backend_tensor_set3(t1, d1, sz1, t2, d2, sz2, t3, d3, sz3);
	}

	async backendTensorGet(
		tensor: TensorPtr,
		dstHeapPtr: number,
		offset: number,
		size: number,
	): Promise<void> {
		if (this.is64) {
			const tensorArg = BigInt(tensor);
			const dstArg = BigInt(dstHeapPtr);
			const offsetArg = BigInt(offset);
			const sizeArg = BigInt(size);
			await this.callWithAsyncify<void>(() =>
				this.m._backend_tensor_get(tensorArg, dstArg, offsetArg, sizeArg),
			);
			return;
		}
		await this.callWithAsyncify<void>(() =>
			this.m._backend_tensor_get(tensor, dstHeapPtr, offset, size),
		);
	}

	async backendTensorGetAsyncBegin(
		tensor: TensorPtr,
		offset: number,
		size: number,
	): Promise<number> {
		// Under JSPI (b4d4b48), this export is in JSPI_EXPORTS and returns
		// a Promise<int32_t>. Callers store the resolved integer requestId
		// as a Map key in asyncTensorGetWaiters / asyncTensorGetStates;
		// without await, the Promise object is keyed instead and the
		// C++ notifier (which carries the integer id) never finds the
		// waiter — silent hang on first decode.
		if (this.is64) {
			return Number(
				await this.m._backend_tensor_get_async_begin(
					BigInt(tensor),
					BigInt(offset),
					BigInt(size),
				),
			);
		}
		return await this.m._backend_tensor_get_async_begin(tensor, offset, size);
	}

	async backendTensorGetAsyncPoll(requestId: number): Promise<number> {
		return await this.m._backend_tensor_get_async_poll(requestId);
	}

	async backendTensorGetAsyncFinish(
		requestId: number,
		dstHeapPtr: number,
		size: number,
	): Promise<void> {
		if (this.is64) {
			await this.m._backend_tensor_get_async_finish(
				requestId,
				BigInt(dstHeapPtr),
				BigInt(size),
			);
			return;
		}
		await this.m._backend_tensor_get_async_finish(requestId, dstHeapPtr, size);
	}

	async backendTensorGetAsyncCancel(requestId: number): Promise<void> {
		await this.m._backend_tensor_get_async_cancel(requestId);
	}

	backendTensorGetAsyncCallbackSupport(): number {
		return this.m._backend_tensor_get_async_callback_support?.() ?? 0;
	}

	backendTensorAlignment(): number {
		return this.m._backend_tensor_alignment();
	}
}
