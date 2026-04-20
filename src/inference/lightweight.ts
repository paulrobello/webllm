import { ALL_SHADERS } from "./wgsl-shaders.js";

/** Configuration for constructing a LightweightModel. */
export interface LightweightModelConfig {
	/** WebGPU device for compute operations. */
	device: GPUDevice;
	/** Embedding vector dimension. */
	embeddingDim: number;
	/** Number of tokens in the vocabulary. */
	vocabularySize: number;
	/** Hidden layer dimension. */
	hiddenDim: number;
	/** Number of transformer layers. */
	layerCount: number;
	/** Maximum supported sequence length. */
	maxSequenceLength: number;
}

/** Model weight tensors required for inference. Optional fields default to tied/shared weights. */
export interface LightweightWeights {
	/** Token embedding matrix [vocabSize, embeddingDim]. */
	tokenEmbeddings: Float32Array;
	/** Positional embedding matrix [maxSeqLen, embeddingDim]. */
	positionEmbeddings?: Float32Array;
	/** Normalization layer weight vector [embeddingDim]. */
	normWeight: Float32Array;
	/** Normalization layer bias vector [embeddingDim]. */
	normBias?: Float32Array;
	/** Output projection weights (tied with tokenEmbeddings if omitted). */
	outputWeights?: Float32Array;
	/** Output projection bias. */
	outputBias?: Float32Array;
}

/**
 * Pure WGSL compute pipeline manager for sub-50M parameter models without WASM dependency.
 *
 * Creates and manages WebGPU compute pipelines and weight buffers, dispatching embedding
 * lookups and other operations entirely through WGSL shaders.
 */
export class LightweightModel {
	private device: GPUDevice;
	private config: LightweightModelConfig;
	private pipelines: Map<string, GPUComputePipeline>;
	private buffers: GPUBuffer[];
	private weightBuffers: Map<string, GPUBuffer>;
	private _loaded: boolean;

	/**
	 * @param config - Model dimension and architecture parameters.
	 */
	constructor(config: LightweightModelConfig) {
		this.config = config;
		this.device = config.device;
		this.pipelines = new Map();
		this.buffers = [];
		this.weightBuffers = new Map();
		this._loaded = false;
	}

	/**
	 * Compile all WGSL shaders into GPU compute pipelines.
	 *
	 * Must be called before loadWeights or embed.
	 */
	async init(): Promise<void> {
		for (const [name, code] of Object.entries(ALL_SHADERS)) {
			const module = this.device.createShaderModule({ code });
			const pipeline = this.device.createComputePipeline({
				layout: "auto",
				compute: { module, entryPoint: "main" },
			});
			this.pipelines.set(name, pipeline);
		}
		this._loaded = true;
	}

	/**
	 * Upload model weight tensors to GPU buffers.
	 *
	 * @param weights - Weight data to upload.
	 */
	async loadWeights(weights: LightweightWeights): Promise<void> {
		const embBuf = this.createGPUBuffer(
			weights.tokenEmbeddings,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);
		this.weightBuffers.set("tokenEmbeddings", embBuf);

		if (weights.positionEmbeddings) {
			const posBuf = this.createGPUBuffer(
				weights.positionEmbeddings,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			);
			this.weightBuffers.set("positionEmbeddings", posBuf);
		}

		const normBuf = this.createGPUBuffer(
			weights.normWeight,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);
		this.weightBuffers.set("normWeight", normBuf);

		if (weights.normBias) {
			const biasBuf = this.createGPUBuffer(
				weights.normBias,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			);
			this.weightBuffers.set("normBias", biasBuf);
		}

		if (weights.outputWeights) {
			const outBuf = this.createGPUBuffer(
				weights.outputWeights,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			);
			this.weightBuffers.set("outputWeights", outBuf);
		}

		if (weights.outputBias) {
			const biasBuf = this.createGPUBuffer(
				weights.outputBias,
				GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
			);
			this.weightBuffers.set("outputBias", biasBuf);
		}
	}

	/**
	 * Look up embedding vectors for a batch of token IDs.
	 *
	 * @param tokenIds - Token IDs to embed.
	 * @returns Flattened embedding vectors [nTokens * embeddingDim].
	 */
	async embed(tokenIds: Uint32Array): Promise<Float32Array> {
		const pipeline = this.pipelines.get("embedding_lookup");
		if (!pipeline) throw new Error("embedding_lookup pipeline not initialized");

		const nTokens = tokenIds.length;
		const embedDim = this.config.embeddingDim;

		// Create token IDs buffer
		const idBuf = this.createGPUBuffer(
			tokenIds,
			GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		);

		// Create output buffer
		const outSize = nTokens * embedDim * 4;
		const outBuf = this.device.createBuffer({
			size: outSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
		});
		this.buffers.push(outBuf);

		// Create uniform params
		const params = new ArrayBuffer(16);
		const view = new DataView(params);
		view.setUint32(0, this.config.vocabularySize, true);
		view.setUint32(4, embedDim, true);
		view.setUint32(8, nTokens, true);
		const paramsBuf = this.createUniformBuffer(params);

		// Create bind group
		const embBuf = this.weightBuffers.get("tokenEmbeddings");
		if (!embBuf) throw new Error("Token embeddings not loaded");

		const bindGroup = this.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{ binding: 0, resource: { buffer: embBuf } },
				{ binding: 1, resource: { buffer: idBuf } },
				{ binding: 2, resource: { buffer: outBuf } },
				{ binding: 3, resource: { buffer: paramsBuf } },
			],
		});

		// Encode and submit
		const encoder = this.device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(pipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(Math.ceil((nTokens * embedDim) / 64));
		pass.end();
		this.device.queue.submit([encoder.finish()]);

		// Read back
		return this.readBuffer(outBuf, outSize);
	}

	private createGPUBuffer(
		data: Float32Array | Uint32Array,
		usage: GPUBufferUsageFlags,
	): GPUBuffer {
		const buf = this.device.createBuffer({
			size: data.byteLength,
			usage: usage | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(buf, 0, data as GPUAllowSharedBufferSource);
		this.buffers.push(buf);
		return buf;
	}

	private createUniformBuffer(data: ArrayBuffer): GPUBuffer {
		const buf = this.device.createBuffer({
			size: data.byteLength,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});
		this.device.queue.writeBuffer(buf, 0, data as GPUAllowSharedBufferSource);
		this.buffers.push(buf);
		return buf;
	}

	private async readBuffer(
		buffer: GPUBuffer,
		size: number,
	): Promise<Float32Array> {
		const staging = this.device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		this.buffers.push(staging);

		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(buffer, 0, staging, 0, size);
		this.device.queue.submit([encoder.finish()]);

		await staging.mapAsync(GPUMapMode.READ);
		const data = new Float32Array(staging.getMappedRange().slice(0));
		staging.unmap();
		return data;
	}

	/** Release all GPU buffers, pipelines, and weight storage. */
	destroy(): void {
		for (const buf of this.buffers) buf.destroy();
		this.buffers = [];
		this.weightBuffers.clear();
		this.pipelines.clear();
		this._loaded = false;
	}

	get isLoaded(): boolean {
		return this._loaded;
	}
	get embeddingDim(): number {
		return this.config.embeddingDim;
	}
	get vocabularySize(): number {
		return this.config.vocabularySize;
	}
	get hiddenDim(): number {
		return this.config.hiddenDim;
	}
	get layerCount(): number {
		return this.config.layerCount;
	}
}
