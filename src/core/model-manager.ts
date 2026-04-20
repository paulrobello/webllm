import type { MemoryPool } from "./memory-pool.js";
import type { ModelEntry } from "./types.js";

/**
 * Multi-model lifecycle coordinator with memory-aware priority eviction.
 */
export class ModelManager {
	private models = new Map<string, ModelEntry>();
	private memoryPool: MemoryPool;

	constructor(memoryPool: MemoryPool) {
		this.memoryPool = memoryPool;
	}

	/**
	 * Register a model entry with the manager.
	 *
	 * @param entry - The model entry to track.
	 */
	register(entry: ModelEntry): void {
		this.models.set(entry.id, entry);
	}

	/**
	 * Unregister a model, evicting its memory and resetting its KV cache.
	 *
	 * @param id - Model identifier to remove.
	 */
	async unregister(id: string): Promise<void> {
		const entry = this.models.get(id);
		if (!entry) return;
		this.memoryPool.evictModel(id);
		entry.kvCache.reset();
		this.models.delete(id);
	}

	/**
	 * Look up a model by ID.
	 *
	 * @param id - Model identifier.
	 * @returns The model entry, or undefined if not registered.
	 */
	get(id: string): ModelEntry | undefined {
		return this.models.get(id);
	}

	/** @returns All registered models sorted by ascending priority. */
	getAll(): ModelEntry[] {
		return [...this.models.values()].sort((a, b) => a.priority - b.priority);
	}

	/** @returns The highest-priority registered model, or undefined if none exist. */
	getPrimary(): ModelEntry | undefined {
		const all = this.getAll();
		return all.length > 0 ? all[0] : undefined;
	}

	/**
	 * Evict the lowest-priority model that has no active sessions.
	 *
	 * @returns The evicted model ID, or null if no model could be evicted.
	 */
	evictLowestPriority(): string | null {
		const all = this.getAll();
		if (all.length === 0) return null;
		// Evict highest priority number (= lowest priority)
		const toEvict = all[all.length - 1];
		// Don't evict if it has active sessions
		if (toEvict.activeSessions > 0) {
			// Find next lowest without active sessions
			for (let i = all.length - 2; i >= 0; i--) {
				if (all[i].activeSessions === 0) {
					const id = all[i].id;
					void this.unregister(id);
					return id;
				}
			}
			return null;
		}
		const id = toEvict.id;
		void this.unregister(id);
		return id;
	}

	/**
	 * Increment the active session count for a model, protecting it from eviction.
	 *
	 * @param modelId - Model identifier.
	 */
	addSession(modelId: string): void {
		const entry = this.models.get(modelId);
		if (entry) entry.activeSessions++;
	}

	/**
	 * Decrement the active session count for a model, re-enabling eviction when zero.
	 *
	 * @param modelId - Model identifier.
	 */
	removeSession(modelId: string): void {
		const entry = this.models.get(modelId);
		if (entry && entry.activeSessions > 0) entry.activeSessions--;
	}

	/**
	 * Check whether a model of the given size can fit, either directly or via eviction.
	 *
	 * @param size - Bytes required.
	 * @returns True if the memory pool can accommodate the allocation.
	 */
	canFit(size: number): boolean {
		return (
			this.memoryPool.canAllocate(size) ||
			this.memoryPool.evictForAllocation(size) !== null
		);
	}

	/** @returns Total bytes currently allocated across all models. */
	getTotalMemoryUsage(): number {
		return this.memoryPool.usedBytes;
	}

	/** Number of registered models. */
	get count(): number {
		return this.models.size;
	}

	/** Reset all KV caches, unregister all models, and reset the memory pool. */
	clear(): void {
		for (const [, entry] of this.models) {
			entry.kvCache.reset();
		}
		this.models.clear();
		this.memoryPool.reset();
	}
}
