/**
 * Memoized GPUComputePipeline cache for the JSEP backend.
 *
 * Keyed by an opaque string supplied by the kernel author (typically a
 * stable hash of WGSL + workgroup geometry + bind layout). The builder
 * is invoked at most once per key for the lifetime of the cache.
 *
 * @experimental The JSEP backend is active Phase-3 R&D; this class is
 * distinct from `src/core/pipeline-cache.ts` (the IndexedDB persistence
 * cache) and was renamed from `PipelineCache` to avoid the cross-module
 * name collision flagged in ARC-007.
 */

export class JsepPipelineCache {
	private readonly device: GPUDevice;
	private readonly cache = new Map<string, GPUComputePipeline>();

	constructor(device: GPUDevice) {
		this.device = device;
	}

	getOrCreate(
		key: string,
		builder: (device: GPUDevice) => GPUComputePipeline,
	): GPUComputePipeline {
		const existing = this.cache.get(key);
		if (existing) return existing;
		const created = builder(this.device);
		this.cache.set(key, created);
		return created;
	}

	/** Test/debug helper. */
	size(): number {
		return this.cache.size;
	}
}
