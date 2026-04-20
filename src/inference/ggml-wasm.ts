/** Configuration for initializing the GGML WASM bridge. */
export interface GgmlWasmConfig {
	/** URL to fetch the GGML WebAssembly binary from. */
	wasmUrl: string;
	/** WebGPU device used for GPU buffer operations. */
	device: GPUDevice;
}

/**
 * WebAssembly bridge for GGML GPU tensor operations via the ggml-webgpu backend.
 *
 * Manages GPU buffer lifecycle and dispatches compute operations (matmul, attention,
 * RoPE, normalization, softmax) through WASM-exported functions.
 */
export class GgmlWasm {
	private wasm: WebAssembly.Exports | null = null;
	private device: GPUDevice | null = null;
	private bufferMap = new Map<number, GPUBuffer>();
	private nextBufferId = 0;

	/**
	 * Fetch, instantiate, and initialize the GGML WASM module with WebGPU.
	 *
	 * @param config - WASM URL and GPU device.
	 * @returns Resolves when initialization succeeds; rejects on WASM init failure.
	 */
	async init(config: GgmlWasmConfig): Promise<void> {
		this.device = config.device;
		const { instance } = await WebAssembly.instantiateStreaming(
			fetch(config.wasmUrl),
			{ env: {} },
		);
		this.wasm = instance.exports;
		const initResult = (this.wasm.webgpu_init as () => number)();
		if (initResult !== 0)
			throw new Error(`WASM init failed with code ${initResult}`);
	}

	/**
	 * Release all GPU buffers and shut down the WASM module.
	 *
	 * Safe to call multiple times; no-ops if already shut down.
	 */
	async shutdown(): Promise<void> {
		if (!this.wasm) return;
		(this.wasm.webgpu_shutdown as () => void)();
		for (const [, buffer] of this.bufferMap) buffer.destroy();
		this.bufferMap.clear();
		this.wasm = null;
		this.device = null;
	}

	/**
	 * Allocate a GPU buffer and register it by a generated ID.
	 *
	 * @param size - Buffer size in bytes.
	 * @param usage - GPUBufferUsageFlags for the new buffer.
	 * @returns Numeric buffer ID for subsequent operations.
	 */
	createBuffer(size: number, usage: GPUBufferUsageFlags): number {
		if (!this.device) throw new Error("Not initialized");
		const buffer = this.device.createBuffer({ size, usage });
		const id = this.nextBufferId++;
		this.bufferMap.set(id, buffer);
		return id;
	}

	/**
	 * Upload data into an existing GPU buffer at a given byte offset.
	 *
	 * @param id - Buffer ID returned by createBuffer.
	 * @param data - Source data to write.
	 * @param offset - Byte offset within the buffer (default 0).
	 */
	writeBuffer(id: number, data: BufferSource, offset = 0): void {
		if (!this.device) throw new Error("Not initialized");
		const buffer = this.bufferMap.get(id);
		if (!buffer) throw new Error(`Buffer ${id} not found`);
		this.device.queue.writeBuffer(
			buffer,
			offset,
			data as GPUAllowSharedBufferSource,
		);
	}

	/**
	 * Copy GPU buffer contents back to the CPU.
	 *
	 * @param id - Buffer ID to read from.
	 * @param size - Number of bytes to read.
	 * @returns ArrayBuffer with the copied data.
	 */
	async readBuffer(id: number, size: number): Promise<ArrayBuffer> {
		if (!this.device) throw new Error("Not initialized");
		const buffer = this.bufferMap.get(id);
		if (!buffer) throw new Error(`Buffer ${id} not found`);
		const staging = this.device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
		this.device.queue.submit([encoder.finish()]);
		await staging.mapAsync(GPUMapMode.READ);
		const data = staging.getMappedRange().slice(0);
		staging.unmap();
		staging.destroy();
		return data;
	}

	/**
	 * Destroy a GPU buffer and remove it from the registry.
	 *
	 * @param id - Buffer ID to destroy.
	 */
	destroyBuffer(id: number): void {
		const buffer = this.bufferMap.get(id);
		if (buffer) {
			buffer.destroy();
			this.bufferMap.delete(id);
		}
	}

	/**
	 * Dispatch a matrix multiplication kernel via WASM.
	 *
	 * @param a - Buffer ID for matrix A [M,K].
	 * @param b - Buffer ID for matrix B [K,N].
	 * @param out - Buffer ID for output matrix C [M,N].
	 * @param m - Rows of A.
	 * @param n - Columns of B.
	 * @param k - Columns of A / rows of B.
	 * @param typeA - GGML tensor type for matrix A.
	 * @param typeB - GGML tensor type for matrix B.
	 */
	mulMat(
		a: number,
		b: number,
		out: number,
		m: number,
		n: number,
		k: number,
		typeA: number,
		typeB: number,
	): void {
		if (!this.wasm) throw new Error("Not initialized");
		(this.wasm.webgpu_mul_mat as (...args: number[]) => void)(
			a,
			b,
			out,
			m,
			n,
			k,
			typeA,
			typeB,
		);
	}

	/**
	 * Dispatch a fused flash-attention kernel via WASM.
	 *
	 * @param q - Buffer ID for query tensor.
	 * @param k - Buffer ID for key tensor.
	 * @param v - Buffer ID for value tensor.
	 * @param out - Buffer ID for output tensor.
	 * @param headDim - Dimension per attention head.
	 * @param nHeads - Number of attention heads.
	 * @param seqLen - Sequence length.
	 * @param scale - Attention scale factor (typically 1/sqrt(headDim)).
	 */
	flashAttn(
		q: number,
		k: number,
		v: number,
		out: number,
		headDim: number,
		nHeads: number,
		seqLen: number,
		scale: number,
	): void {
		if (!this.wasm) throw new Error("Not initialized");
		(this.wasm.webgpu_flash_attn as (...args: number[]) => void)(
			q,
			k,
			v,
			out,
			headDim,
			nHeads,
			seqLen,
			scale,
		);
	}

	/**
	 * Apply Rotary Position Embedding (RoPE) to a tensor.
	 *
	 * @param tensor - Buffer ID for the input tensor.
	 * @param freqs - Buffer ID for precomputed frequency values.
	 * @param out - Buffer ID for the output tensor.
	 * @param dim - Dimension over which to apply RoPE.
	 * @param freqBase - Base frequency for the sinusoidal computation.
	 * @param freqScale - Scaling factor applied to frequencies.
	 */
	rope(
		tensor: number,
		freqs: number,
		out: number,
		dim: number,
		freqBase: number,
		freqScale: number,
	): void {
		if (!this.wasm) throw new Error("Not initialized");
		(this.wasm.webgpu_rope as (...args: number[]) => void)(
			tensor,
			freqs,
			out,
			dim,
			freqBase,
			freqScale,
		);
	}

	/**
	 * Apply RMS normalization: out = x * weight / sqrt(mean(x^2) + eps).
	 *
	 * @param x - Buffer ID for input tensor.
	 * @param weight - Buffer ID for learned weight vector.
	 * @param out - Buffer ID for output tensor.
	 * @param rows - Number of rows (tokens).
	 * @param cols - Number of columns (hidden dimension).
	 * @param eps - Epsilon for numerical stability.
	 */
	rmsNorm(
		x: number,
		weight: number,
		out: number,
		rows: number,
		cols: number,
		eps: number,
	): void {
		if (!this.wasm) throw new Error("Not initialized");
		(this.wasm.webgpu_rms_norm as (...args: number[]) => void)(
			x,
			weight,
			out,
			rows,
			cols,
			eps,
		);
	}

	/**
	 * Apply row-wise softmax with optional scaling.
	 *
	 * @param x - Buffer ID for input logits.
	 * @param out - Buffer ID for output probabilities.
	 * @param rows - Number of rows.
	 * @param cols - Number of columns per row.
	 * @param scale - Pre-softmax scale factor (typically 1/sqrt(headDim)).
	 */
	softmax(
		x: number,
		out: number,
		rows: number,
		cols: number,
		scale: number,
	): void {
		if (!this.wasm) throw new Error("Not initialized");
		(this.wasm.webgpu_soft_max as (...args: number[]) => void)(
			x,
			out,
			rows,
			cols,
			scale,
		);
	}
}
