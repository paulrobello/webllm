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
 * Manages the Emscripten MODULARIZE module and exposes typed methods
 * for tensor operations, graph building, and GPU compute.
 */
export class GgmlWasm {
	private m: any = null;
	private initialized = false;

	/**
	 * Load the Emscripten module and initialize the WebGPU backend.
	 */
	async init(config: GgmlWasmConfig): Promise<void> {
		const factory = (await import(config.wasmUrl)).default;
		this.m = await factory();
		const result: number = this.m._webgpu_init();
		if (result !== 0) {
			throw new Error(`WebGPU backend init failed (code ${result})`);
		}
		this.initialized = true;
	}

	/**
	 * Shut down the backend and release the WASM module.
	 */
	async shutdown(): Promise<void> {
		if (!this.initialized) return;
		this.m._webgpu_shutdown();
		this.initialized = false;
		this.m = null;
	}

	// ── WASM heap helpers ────────────────────────────────────────────────

	/** Allocate WASM heap memory. Returns a pointer. */
	malloc(size: number): number {
		return this.m._malloc(size);
	}

	/** Free WASM heap memory. */
	free(ptr: number): void {
		this.m._free(ptr);
	}

	/** Get HEAPU8 view of WASM memory. */
	get heapU8(): Uint8Array {
		return this.m.HEAPU8;
	}

	/** Get HEAPF32 view of WASM memory. */
	get heapF32(): Float32Array {
		return this.m.HEAPF32;
	}

	/** Write a string to WASM memory. Returns pointer. */
	stringToNewUTF8(str: string): number {
		return this.m.stringToUTF8(str, this.m._malloc(str.length + 1), str.length + 1);
	}

	// ── Context ──────────────────────────────────────────────────────────

	/** Create a ggml context with the given memory budget (bytes). */
	ctxCreate(memSize: number): number {
		const rc = this.m._ctx_create(memSize);
		if (rc !== 0) throw new Error(`ctx_create failed (${rc})`);
		return rc;
	}

	/** Free the current ggml context. */
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

	tensorNew4d(type: number, ne0: number, ne1: number, ne2: number, ne3: number): TensorPtr {
		return this.m._tensor_new_4d(type, ne0, ne1, ne2, ne3);
	}

	tensorSetName(tensor: TensorPtr, name: string): void {
		const namePtr = this.stringToNewUTF8(name);
		this.m._tensor_set_name(tensor, namePtr);
		this.free(namePtr);
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

	/**
	 * Copy data from a WASM heap region into a tensor.
	 * Both `srcHeapPtr` and the tensor data must be in WASM memory.
	 */
	tensorSetData(tensor: TensorPtr, srcHeapPtr: number, size: number): void {
		this.m._tensor_set_data(tensor, srcHeapPtr, size);
	}

	/**
	 * Copy tensor data to a WASM heap region.
	 */
	tensorGetData(tensor: TensorPtr, dstHeapPtr: number, size: number): void {
		this.m._tensor_get_data(tensor, dstHeapPtr, size);
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
		x: TensorPtr, nDims: number, mode: number, nCtxOrig: number,
		freqBase: number, freqScale: number, extFactor: number,
		attnFactor: number, betaFast: number, betaSlow: number,
	): TensorPtr {
		return this.m._op_rope(x, nDims, mode, nCtxOrig, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow);
	}

	opReshape2d(x: TensorPtr, ne0: number, ne1: number): TensorPtr {
		return this.m._op_reshape_2d(x, ne0, ne1);
	}

	opReshape3d(x: TensorPtr, ne0: number, ne1: number, ne2: number): TensorPtr {
		return this.m._op_reshape_3d(x, ne0, ne1, ne2);
	}

	opPermute(x: TensorPtr, d0: number, d1: number, d2: number, d3: number): TensorPtr {
		return this.m._op_permute(x, d0, d1, d2, d3);
	}

	opCont(x: TensorPtr): TensorPtr {
		return this.m._op_cont(x);
	}

	opView2d(x: TensorPtr, ne0: number, ne1: number, nb1: number, offset: number): TensorPtr {
		return this.m._op_view_2d(x, ne0, ne1, nb1, offset);
	}

	opView3d(x: TensorPtr, ne0: number, ne1: number, ne2: number, nb1: number, nb2: number, offset: number): TensorPtr {
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

	graphCompute(graph: GraphPtr): number {
		return this.m._graph_compute(graph);
	}

	// ── Backend buffer ───────────────────────────────────────────────────

	/** Allocate all tensors in the current context on the GPU backend. */
	backendAllocCtxTensors(): BufferPtr {
		return this.m._backend_alloc_ctx_tensors();
	}

	backendBufferFree(buffer: BufferPtr): void {
		this.m._backend_buffer_free(buffer);
	}

	/** Upload data from WASM heap to a tensor on the GPU. */
	backendTensorSet(tensor: TensorPtr, srcHeapPtr: number, offset: number, size: number): void {
		this.m._backend_tensor_set(tensor, srcHeapPtr, offset, size);
	}

	/** Download tensor data from GPU to WASM heap. */
	backendTensorGet(tensor: TensorPtr, dstHeapPtr: number, offset: number, size: number): void {
		this.m._backend_tensor_get(tensor, dstHeapPtr, offset, size);
	}

	backendTensorAlignment(): number {
		return this.m._backend_tensor_alignment();
	}
}
