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

	async init(config: GgmlWasmConfig): Promise<void> {
		const factory = (await import(config.wasmUrl)).default;
		this.m = await factory();
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

	/** Upload bytes from JS ArrayBuffer to a GPU tensor via stack buffer. */
	uploadToTensor(tensor: TensorPtr, data: Uint8Array, offset = 0): void {
		const sp = this.stackSave();
		try {
			const ptr = this.stackAlloc(data.byteLength);
			this.heapU8.set(data, ptr);
			this.m._backend_tensor_set(tensor, ptr, offset, data.byteLength);
		} finally {
			this.stackRestore(sp);
		}
	}

	/** Download tensor data from GPU to a new Uint8Array via heap buffer. */
	async downloadFromTensor(
		tensor: TensorPtr,
		byteLength: number,
		offset = 0,
	): Promise<Uint8Array> {
		const ptr = this.malloc(byteLength);
		try {
			await this.callWithAsyncify<void>(() =>
				this.m._backend_tensor_get(tensor, ptr, offset, byteLength),
			);
			return new Uint8Array(this.heapU8.buffer, ptr, byteLength).slice();
		} finally {
			this.free(ptr);
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

	opScale(x: TensorPtr, s: number): TensorPtr {
		return this.m._op_scale(x, s);
	}

	opRepeat(x: TensorPtr, y: TensorPtr): TensorPtr {
		return this.m._op_repeat(x, y);
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
		return await this.callWithAsyncify<number>(() =>
			this.m._graph_compute(graph),
		);
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

	backendTensorAlignment(): number {
		return this.m._backend_tensor_alignment();
	}
}
