export interface GgmlWasmConfig {
	wasmUrl: string;
	device: GPUDevice;
}

export class GgmlWasm {
	private wasm: WebAssembly.Exports | null = null;
	private device: GPUDevice | null = null;
	private bufferMap = new Map<number, GPUBuffer>();
	private nextBufferId = 0;

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

	async shutdown(): Promise<void> {
		if (!this.wasm) return;
		(this.wasm.webgpu_shutdown as () => void)();
		for (const [, buffer] of this.bufferMap) buffer.destroy();
		this.bufferMap.clear();
		this.wasm = null;
		this.device = null;
	}

	createBuffer(size: number, usage: GPUBufferUsageFlags): number {
		if (!this.device) throw new Error("Not initialized");
		const buffer = this.device.createBuffer({ size, usage });
		const id = this.nextBufferId++;
		this.bufferMap.set(id, buffer);
		return id;
	}

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

	destroyBuffer(id: number): void {
		const buffer = this.bufferMap.get(id);
		if (buffer) {
			buffer.destroy();
			this.bufferMap.delete(id);
		}
	}

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
