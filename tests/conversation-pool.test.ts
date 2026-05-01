import { describe, expect, test } from "bun:test";
import { ConversationPool } from "../src/core/conversation-pool.js";
import {
	ConversationNotFoundError,
	ConversationPoolFullError,
} from "../src/core/errors.js";

describe("ConversationPool", () => {
	test("create allocates handle with stable id and modelHandleId", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("model-a");
		expect(conv.id).toMatch(/^conv_/);
		expect(conv.modelHandleId).toBe("model-a");
		expect(pool.has(conv)).toBe(true);
	});

	test("ids are unique across creates", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const a = pool.create("m");
		const b = pool.create("m");
		expect(a.id).not.toBe(b.id);
	});

	test("dispose removes from pool; subsequent get throws", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		pool.dispose(conv);
		expect(pool.has(conv)).toBe(false);
		expect(() => pool.requireHandle(conv)).toThrow(ConversationNotFoundError);
	});

	test("dispose is idempotent", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		pool.dispose(conv);
		expect(() => pool.dispose(conv)).not.toThrow();
	});

	test("create at cap throws ConversationPoolFullError with live ids", () => {
		const pool = new ConversationPool({ maxConversations: 2 });
		const a = pool.create("m");
		const b = pool.create("m");
		try {
			pool.create("m");
			throw new Error("expected throw");
		} catch (err) {
			expect(err).toBeInstanceOf(ConversationPoolFullError);
			const e = err as ConversationPoolFullError;
			expect(e.liveConversationIds).toContain(a.id);
			expect(e.liveConversationIds).toContain(b.id);
		}
	});

	test("disposeAllForModel clears matching handles only", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const a = pool.create("model-a");
		const b = pool.create("model-b");
		pool.disposeAllForModel("model-a");
		expect(pool.has(a)).toBe(false);
		expect(pool.has(b)).toBe(true);
	});

	test("set then get round-trips a snapshot", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		const snap = {
			conversationId: conv.id,
			modelHandleId: "m",
			tokenIds: [1, 2, 3],
			kvBytes: new Uint8Array(8),
			byteSize: 8,
			lastAccessMs: 0,
		};
		pool.set(conv, snap);
		expect(pool.get(conv)).toBe(snap);
	});

	test("tryAcquireLock prevents concurrent claims; release re-allows", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		const release = pool.tryAcquireLock(conv);
		expect(release).not.toBeNull();
		expect(pool.tryAcquireLock(conv)).toBeNull();
		release?.();
		const release2 = pool.tryAcquireLock(conv);
		expect(release2).not.toBeNull();
		release2?.();
	});
});
