/**
 * GPU buffer manager for the JSEP-style backend.
 *
 * Maintains a handle → GPUBuffer map with bucketed free-list reuse.
 * Handles are integers starting at 1 — the C++ side uses 0 as the
 * sentinel-stripped state of `data == 0x2000`, so this allocator must
 * never return 0.
 *
 * Bucket sizes are powers of 4 from 1 KB to 128 MB; 128 MB matches the
 * per-binding `maxStorageBufferBindingSize` cap on Chrome/Apple.
 */

const SIZE_BUCKETS: readonly number[] = [
	1 << 10, // 1 KB
	4 << 10, // 4 KB
	16 << 10, // 16 KB
	64 << 10, // 64 KB
	256 << 10, // 256 KB
	1 << 20, // 1 MB
	4 << 20, // 4 MB
	16 << 20, // 16 MB
	64 << 20, // 64 MB
	128 << 20, // 128 MB
];

interface BufferRecord {
	buffer: GPUBuffer;
	size: number; // physical bucketed size
	bucket: number;
}

function pickBucket(size: number): { bucketIndex: number; capacity: number } {
	for (let i = 0; i < SIZE_BUCKETS.length; i++) {
		if (size <= SIZE_BUCKETS[i]) {
			return { bucketIndex: i, capacity: SIZE_BUCKETS[i] };
		}
	}
	// Above the largest bucket: don't bucket — use exact size.
	return { bucketIndex: -1, capacity: size };
}

/**
 * GPU buffer pool with handle-based access. Handles are integer keys; the
 * manager owns the underlying GPUBuffer lifecycle.
 *
 * Contract:
 * - alloc(size) returns a new handle (≥1; never 0).
 * - free(handle) returns the buffer to a bucket free-list (or destroys
 *   if oversized). Idempotent: free on an unknown handle is a no-op.
 * - After free(handle), subsequent get/write/read/clear with that handle
 *   throws "invalid handle".
 * - destroy() releases all retained buffers (live + bucketed) and
 *   invalidates the manager.
 */
export class GpuDataManager {
	private readonly device: GPUDevice;
	private readonly handles = new Map<number, BufferRecord>();
	private readonly freeBuckets: BufferRecord[][] = SIZE_BUCKETS.map(() => []);
	private nextHandle = 1;

	constructor(device: GPUDevice) {
		this.device = device;
	}

	alloc(size: number): number {
		if (size <= 0) {
			throw new Error(`GpuDataManager.alloc: invalid size ${size}`);
		}
		const { bucketIndex, capacity } = pickBucket(size);
		let record: BufferRecord;
		if (bucketIndex >= 0) {
			const pool = this.freeBuckets[bucketIndex];
			const reused = pool.pop();
			if (reused) {
				record = reused;
			} else {
				record = {
					buffer: this.device.createBuffer({
						size: capacity,
						usage:
							GPUBufferUsage.STORAGE |
							GPUBufferUsage.COPY_SRC |
							GPUBufferUsage.COPY_DST,
					}),
					size: capacity,
					bucket: bucketIndex,
				};
			}
		} else {
			// Oversized allocation — exact size, not bucketed for reuse.
			record = {
				buffer: this.device.createBuffer({
					size: capacity,
					usage:
						GPUBufferUsage.STORAGE |
						GPUBufferUsage.COPY_SRC |
						GPUBufferUsage.COPY_DST,
				}),
				size: capacity,
				bucket: -1,
			};
		}
		const handle = this.nextHandle++;
		this.handles.set(handle, record);
		return handle;
	}

	free(handle: number): void {
		const record = this.handles.get(handle);
		if (!record) return;
		this.handles.delete(handle);
		if (record.bucket >= 0) {
			this.freeBuckets[record.bucket].push(record);
		} else {
			// Oversized — destroy immediately; no reuse pool.
			record.buffer.destroy();
		}
	}

	get(handle: number): { buffer: GPUBuffer; size: number } {
		const record = this.handles.get(handle);
		if (!record) {
			throw new Error(`GpuDataManager.get: invalid handle ${handle}`);
		}
		return { buffer: record.buffer, size: record.size };
	}

	write(
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
		wasmHeap: ArrayBufferLike,
	): void {
		const record = this.handles.get(handle);
		if (!record) {
			throw new Error(`GpuDataManager.write: invalid handle ${handle}`);
		}
		// Re-derive view each call — never cache across awaits (heap-grow safety).
		const view = new Uint8Array(wasmHeap, hostPtr, size);
		this.device.queue.writeBuffer(record.buffer, offset, view, 0, size);
	}

	async readAsync(
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
		wasmHeap: ArrayBufferLike,
	): Promise<void> {
		const record = this.handles.get(handle);
		if (!record) {
			throw new Error(`GpuDataManager.readAsync: invalid handle ${handle}`);
		}
		const staging = this.device.createBuffer({
			size,
			usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
		});
		const encoder = this.device.createCommandEncoder();
		encoder.copyBufferToBuffer(record.buffer, offset, staging, 0, size);
		this.device.queue.submit([encoder.finish()]);
		await staging.mapAsync(GPUMapMode.READ, 0, size);
		const mapped = new Uint8Array(staging.getMappedRange(0, size));
		// Re-derive heap view AFTER the await (heap may have grown).
		const dest = new Uint8Array(wasmHeap, hostPtr, size);
		dest.set(mapped);
		staging.unmap();
		staging.destroy();
	}

	clear(handle: number, value: number, offset: number, size: number): void {
		const record = this.handles.get(handle);
		if (!record) {
			throw new Error(`GpuDataManager.clear: invalid handle ${handle}`);
		}
		const scratch = new Uint8Array(size);
		if (value !== 0) scratch.fill(value);
		this.device.queue.writeBuffer(record.buffer, offset, scratch, 0, size);
	}

	/** Test/debug helper: number of currently-live (allocated) handles. */
	liveHandleCount(): number {
		return this.handles.size;
	}

	/**
	 * Destroy all retained GPUBuffers (live handles + bucket free-list) and
	 * clear internal state. Call from engine teardown to release WebGPU
	 * resources promptly. After destroy(), the manager is no longer usable —
	 * subsequent alloc/get/write/read/clear/free will throw.
	 */
	destroy(): void {
		for (const record of this.handles.values()) {
			record.buffer.destroy();
		}
		this.handles.clear();

		for (const bucket of this.freeBuckets) {
			for (const record of bucket) {
				record.buffer.destroy();
			}
			bucket.length = 0;
		}
	}
}
