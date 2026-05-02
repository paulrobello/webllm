import {
	ConversationNotFoundError,
	ConversationNotPopulatedError,
	ConversationPoolFullError,
} from "./errors.js";

export interface ConversationHandle {
	readonly id: string;
	readonly modelHandleId: string;
}

export interface ConversationOptions {
	/** Maximum KV tokens for this conversation. Default: model's contextLength. */
	maxContextTokens?: number;
}

export interface KVSnapshot {
	conversationId: string;
	modelHandleId: string;
	tokenIds: number[];
	kvBytes: Uint8Array;
	byteSize: number;
	lastAccessMs: number;
}

export interface ConversationPoolConfig {
	maxConversations: number;
}

interface PoolEntry {
	handle: ConversationHandle;
	options: ConversationOptions;
	snapshot: KVSnapshot | null;
	locked: boolean;
	/**
	 * Monotonic access ordinal — bumped on every operation that touches
	 * the entry (`get`, `set`, `options`, `tryAcquireLock`, plus initial
	 * creation). Used by `create()` to pick the LRU non-locked entry to
	 * evict when the pool is at capacity. A counter (rather than
	 * `Date.now()`) avoids ties for back-to-back operations within the
	 * same millisecond.
	 */
	accessSeq: number;
}

export class ConversationPool {
	private readonly entries = new Map<string, PoolEntry>();
	private readonly maxConversations: number;
	private nextId = 1;
	private accessCounter = 0;

	constructor(config: ConversationPoolConfig) {
		this.maxConversations = config.maxConversations;
	}

	create(
		modelHandleId: string,
		options: ConversationOptions = {},
	): ConversationHandle {
		if (this.entries.size >= this.maxConversations) {
			// LRU eviction: drop the oldest non-locked entry. If every
			// entry is locked (in-flight chatCompletion), there's nothing
			// safe to evict and we surface the full-pool error to the
			// caller. Spec follow-up #1.
			const victim = this._findLruEvictable();
			if (!victim) {
				throw new ConversationPoolFullError([...this.entries.keys()]);
			}
			this.entries.delete(victim.handle.id);
		}
		const id = `conv_${this.nextId++}`;
		const handle: ConversationHandle = { id, modelHandleId };
		this.entries.set(id, {
			handle,
			options,
			snapshot: null,
			locked: false,
			accessSeq: ++this.accessCounter,
		});
		return handle;
	}

	private _findLruEvictable(): PoolEntry | null {
		let oldest: PoolEntry | null = null;
		for (const entry of this.entries.values()) {
			if (entry.locked) continue;
			if (oldest === null || entry.accessSeq < oldest.accessSeq) {
				oldest = entry;
			}
		}
		return oldest;
	}

	/**
	 * Create a new conversation that inherits a deep copy of `src`'s
	 * snapshot. The new conversation is independent from `src` after
	 * fork — mutating either one's snapshot does not affect the other.
	 *
	 * Throws `ConversationNotFoundError` if `src` doesn't exist.
	 * Throws `ConversationNotPopulatedError` if `src` has no snapshot
	 * (never had a successful chatCompletion call).
	 *
	 * Honors the same LRU eviction rules as `create()` — may evict the
	 * oldest non-locked entry to make room. Note: `src` is itself touched
	 * during the existence check, so `src` will be the most-recently-used
	 * entry and is guaranteed not to be evicted by this fork.
	 *
	 * Spec follow-up #2 (cross-conversation prefix sharing) — copy-from-
	 * prefix-store path. The new conversation's first chatCompletion
	 * call will find the inherited snapshot via the longest-shared-token
	 * prefix walk and only prefill the divergent tail.
	 */
	fork(src: ConversationHandle): ConversationHandle {
		const srcEntry = this._requireEntry(src);
		const srcSnap = srcEntry.snapshot;
		if (!srcSnap) {
			throw new ConversationNotPopulatedError(src.id);
		}
		if (this.entries.size >= this.maxConversations) {
			const victim = this._findLruEvictable();
			if (!victim) {
				throw new ConversationPoolFullError([...this.entries.keys()]);
			}
			this.entries.delete(victim.handle.id);
		}
		const id = `conv_${this.nextId++}`;
		const handle: ConversationHandle = { id, modelHandleId: src.modelHandleId };
		const copiedBytes = new Uint8Array(srcSnap.kvBytes.byteLength);
		copiedBytes.set(srcSnap.kvBytes);
		this.entries.set(id, {
			handle,
			options: { ...srcEntry.options },
			snapshot: {
				conversationId: id,
				modelHandleId: src.modelHandleId,
				tokenIds: [...srcSnap.tokenIds],
				kvBytes: copiedBytes,
				byteSize: copiedBytes.byteLength,
				lastAccessMs: Date.now(),
			},
			locked: false,
			accessSeq: ++this.accessCounter,
		});
		return handle;
	}

	dispose(conv: ConversationHandle): void {
		this.entries.delete(conv.id);
	}

	disposeAllForModel(modelHandleId: string): void {
		for (const [id, entry] of this.entries) {
			if (entry.handle.modelHandleId === modelHandleId) {
				this.entries.delete(id);
			}
		}
	}

	/** Throws ConversationNotFoundError if the handle has been disposed or never existed. */
	assertExists(conv: ConversationHandle): void {
		this._requireEntry(conv);
	}

	get(conv: ConversationHandle): KVSnapshot | undefined {
		const entry = this.entries.get(conv.id);
		if (!entry) return undefined;
		entry.accessSeq = ++this.accessCounter;
		return entry.snapshot ?? undefined;
	}

	set(conv: ConversationHandle, snapshot: KVSnapshot): void {
		const entry = this._requireEntry(conv);
		entry.snapshot = snapshot;
	}

	options(conv: ConversationHandle): ConversationOptions {
		return this._requireEntry(conv).options;
	}

	private _requireEntry(conv: ConversationHandle): PoolEntry {
		const entry = this.entries.get(conv.id);
		if (!entry) throw new ConversationNotFoundError(conv.id);
		if (entry.handle.modelHandleId !== conv.modelHandleId) {
			throw new ConversationNotFoundError(conv.id);
		}
		entry.accessSeq = ++this.accessCounter;
		return entry;
	}

	tryAcquireLock(conv: ConversationHandle): (() => void) | null {
		const entry = this._requireEntry(conv);
		if (entry.locked) return null;
		entry.locked = true;
		let released = false;
		return () => {
			if (released) return;
			released = true;
			entry.locked = false;
		};
	}
}
