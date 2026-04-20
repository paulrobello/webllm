/** Tracks a single GPU buffer allocation within the memory pool. */
export interface BufferAllocation {
	readonly id: number;
	size: number;
	priority: number;
	modelId: string;
	freed: boolean;
}

/** Memory pool lifecycle event types. */
export type MemoryEvent = "pressure" | "eviction" | "allocation" | "free";
/** Callback invoked when a memory pool event fires. */
export type MemoryEventHandler = (
	event: string,
	detail: {
		allocationId?: number;
		modelId?: string;
		used: number;
		total: number;
	},
) => void;

/**
 * Model-scoped GPU memory pool with pressure detection and priority-based eviction.
 */
export class MemoryPool {
	private allocations = new Map<number, BufferAllocation>();
	private nextId = 0;
	private _budget: number;
	private _usedBytes = 0;
	private listeners = new Map<string, Set<MemoryEventHandler>>();
	private hadPressure = false;

	constructor(budget: number) {
		this._budget = budget;
	}

	/** Total byte budget for the pool. */
	get budget(): number {
		return this._budget;
	}
	/** Bytes currently allocated across all live buffers. */
	get usedBytes(): number {
		return this._usedBytes;
	}
	/** Bytes available before the budget is exhausted. */
	get remainingBytes(): number {
		return this._budget - this._usedBytes;
	}
	/** Fraction of the budget currently in use (0 to 1). */
	get pressureRatio(): number {
		return this._usedBytes / this._budget;
	}
	/** True when more than 75% of the budget is in use. */
	get isUnderPressure(): boolean {
		return this.pressureRatio > 0.75;
	}
	/** Number of live (non-freed) allocations. */
	get allocationCount(): number {
		return this.allocations.size;
	}

	/**
	 * Allocate a buffer of the given size within the pool's budget.
	 *
	 * @param size - Bytes to allocate.
	 * @param priority - Eviction priority (lower = more likely to be evicted).
	 * @param modelId - Owning model identifier.
	 * @returns Allocation ID.
	 * @throws Error if the allocation exceeds the remaining budget.
	 */
	allocate(size: number, priority = 0, modelId = "default"): number {
		if (size > this.remainingBytes) {
			throw new Error(
				`Allocation of ${size} bytes exceeds memory budget (remaining: ${this.remainingBytes})`,
			);
		}
		const id = this.nextId++;
		this.allocations.set(id, { id, size, priority, modelId, freed: false });
		this._usedBytes += size;
		this.emit("allocation", {
			allocationId: id,
			modelId,
			used: this._usedBytes,
			total: this._budget,
		});
		if (!this.hadPressure && this.isUnderPressure) {
			this.hadPressure = true;
			this.emit("pressure", { used: this._usedBytes, total: this._budget });
		}
		return id;
	}

	/**
	 * Release a previously allocated buffer.
	 *
	 * @param id - Allocation ID to free.
	 */
	free(id: number): void {
		const alloc = this.allocations.get(id);
		if (!alloc || alloc.freed) return;
		alloc.freed = true;
		this._usedBytes -= alloc.size;
		this.allocations.delete(id);
		this.emit("free", {
			allocationId: id,
			modelId: alloc.modelId,
			used: this._usedBytes,
			total: this._budget,
		});
		if (this._usedBytes < this._budget * 0.75) this.hadPressure = false;
	}

	/**
	 * Check whether the requested size fits within the remaining budget.
	 *
	 * @param size - Bytes to check.
	 * @returns True if the allocation would succeed.
	 */
	canAllocate(size: number): boolean {
		return size <= this.remainingBytes;
	}

	/**
	 * Evict the lowest-priority allocation to free space for a new allocation.
	 *
	 * @param neededSize - Bytes required for the pending allocation.
	 * @param excludeModelId - Optional model ID whose allocations should not be evicted.
	 * @returns The evicted allocation ID, or null if no suitable candidate was found.
	 */
	evictForAllocation(
		neededSize: number,
		excludeModelId?: string,
	): number | null {
		const candidates = [...this.allocations.values()]
			.filter((a) => !a.freed && a.modelId !== excludeModelId)
			.sort((a, b) => b.priority - a.priority);
		for (const candidate of candidates) {
			if (this.remainingBytes + candidate.size >= neededSize) {
				this.emit("eviction", {
					allocationId: candidate.id,
					modelId: candidate.modelId,
					used: this._usedBytes - candidate.size,
					total: this._budget,
				});
				this.free(candidate.id);
				return candidate.id;
			}
		}
		return null;
	}

	/**
	 * Get the total bytes allocated for a specific model.
	 *
	 * @param modelId - Model identifier.
	 * @returns Bytes used by the model's live allocations.
	 */
	getModelUsage(modelId: string): number {
		let total = 0;
		for (const a of this.allocations.values()) {
			if (!a.freed && a.modelId === modelId) total += a.size;
		}
		return total;
	}

	/**
	 * Get all live allocations for a specific model.
	 *
	 * @param modelId - Model identifier.
	 * @returns Array of active buffer allocations.
	 */
	getModelAllocations(modelId: string): BufferAllocation[] {
		return [...this.allocations.values()].filter(
			(a) => !a.freed && a.modelId === modelId,
		);
	}

	/**
	 * Free all allocations belonging to a model.
	 *
	 * @param modelId - Model identifier to evict entirely.
	 * @returns Total bytes freed.
	 */
	evictModel(modelId: string): number {
		const toEvict = [...this.allocations.values()].filter(
			(a) => !a.freed && a.modelId === modelId,
		);
		let freed = 0;
		for (const a of toEvict) {
			freed += a.size;
			this.free(a.id);
		}
		return freed;
	}

	/**
	 * Subscribe to a memory pool event.
	 *
	 * @param event - Event type (pressure, eviction, allocation, free).
	 * @param handler - Callback invoked with event details.
	 */
	on(event: MemoryEvent, handler: MemoryEventHandler): void {
		if (!this.listeners.has(event)) this.listeners.set(event, new Set());
		this.listeners.get(event)?.add(handler);
	}

	/**
	 * Unsubscribe from a memory pool event.
	 *
	 * @param event - Event type.
	 * @param handler - Previously registered callback.
	 */
	off(event: MemoryEvent, handler: MemoryEventHandler): void {
		this.listeners.get(event)?.delete(handler);
	}

	private emit(
		event: string,
		detail: {
			allocationId?: number;
			modelId?: string;
			used: number;
			total: number;
		},
	): void {
		for (const handler of this.listeners.get(event) ?? []) {
			handler(event, detail);
		}
	}

	/** Release all allocations and reset pool state. */
	reset(): void {
		this.allocations.clear();
		this._usedBytes = 0;
		this.hadPressure = false;
	}
}
