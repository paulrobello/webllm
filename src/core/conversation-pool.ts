import {
	ConversationNotFoundError,
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
}

export class ConversationPool {
	private readonly entries = new Map<string, PoolEntry>();
	private readonly maxConversations: number;
	private nextId = 1;

	constructor(config: ConversationPoolConfig) {
		this.maxConversations = config.maxConversations;
	}

	create(
		modelHandleId: string,
		options: ConversationOptions = {},
	): ConversationHandle {
		if (this.entries.size >= this.maxConversations) {
			throw new ConversationPoolFullError([...this.entries.keys()]);
		}
		const id = `conv_${this.nextId++}`;
		const handle: ConversationHandle = { id, modelHandleId };
		this.entries.set(id, {
			handle,
			options,
			snapshot: null,
			locked: false,
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

	has(conv: ConversationHandle): boolean {
		return this.entries.has(conv.id);
	}

	get(conv: ConversationHandle): KVSnapshot | undefined {
		return this.entries.get(conv.id)?.snapshot ?? undefined;
	}

	set(conv: ConversationHandle, snapshot: KVSnapshot): void {
		const entry = this.requireHandle(conv);
		entry.snapshot = snapshot;
	}

	options(conv: ConversationHandle): ConversationOptions {
		return this.requireHandle(conv).options;
	}

	requireHandle(conv: ConversationHandle): PoolEntry {
		const entry = this.entries.get(conv.id);
		if (!entry) throw new ConversationNotFoundError(conv.id);
		return entry;
	}

	tryAcquireLock(conv: ConversationHandle): (() => void) | null {
		const entry = this.requireHandle(conv);
		if (entry.locked) return null;
		entry.locked = true;
		return () => {
			entry.locked = false;
		};
	}
}
