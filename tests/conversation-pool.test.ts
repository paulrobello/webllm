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
		expect(() => pool.assertExists(conv)).not.toThrow();
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
		expect(() => pool.assertExists(conv)).toThrow(ConversationNotFoundError);
	});

	test("dispose is idempotent", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		pool.dispose(conv);
		expect(() => pool.dispose(conv)).not.toThrow();
	});

	test("ConversationPoolFullError carries live ids when raised", () => {
		// LRU eviction means create() at cap evicts rather than throwing in
		// the common case; throw only happens when every entry is locked.
		// Verify the error still carries live ids in that path.
		const pool = new ConversationPool({ maxConversations: 2 });
		const a = pool.create("m");
		const b = pool.create("m");
		const releaseA = pool.tryAcquireLock(a);
		const releaseB = pool.tryAcquireLock(b);
		expect(releaseA).not.toBeNull();
		expect(releaseB).not.toBeNull();
		try {
			pool.create("m");
			throw new Error("expected throw");
		} catch (err) {
			expect(err).toBeInstanceOf(ConversationPoolFullError);
			const e = err as ConversationPoolFullError;
			expect(e.liveConversationIds).toContain(a.id);
			expect(e.liveConversationIds).toContain(b.id);
		}
		releaseA?.();
		releaseB?.();
	});

	test("disposeAllForModel clears matching handles only", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const a = pool.create("model-a");
		const b = pool.create("model-b");
		pool.disposeAllForModel("model-a");
		expect(() => pool.assertExists(a)).toThrow(ConversationNotFoundError);
		expect(() => pool.assertExists(b)).not.toThrow();
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

	test("double-release of a lock does not steal a re-acquired lock", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("m");
		const release1 = pool.tryAcquireLock(conv);
		expect(release1).not.toBeNull();
		release1?.();
		const release2 = pool.tryAcquireLock(conv);
		expect(release2).not.toBeNull();
		release1?.(); // double-release: must be a no-op
		expect(pool.tryAcquireLock(conv)).toBeNull(); // release2 still owns the lock
		release2?.();
	});

	test("requireHandle rejects fabricated handle with wrong modelHandleId", () => {
		const pool = new ConversationPool({ maxConversations: 4 });
		const conv = pool.create("model-a");
		const fake = { id: conv.id, modelHandleId: "model-b" };
		expect(() => pool.assertExists(fake)).toThrow(ConversationNotFoundError);
	});

	describe("LRU eviction", () => {
		test("create at cap evicts oldest non-locked entry and succeeds", () => {
			const pool = new ConversationPool({ maxConversations: 2 });
			const a = pool.create("m");
			// Force a > 0 ms gap so lastAccessMs ordering is observable.
			const b = pool.create("m");
			// `a` is now older. Creating a third should evict `a`.
			const c = pool.create("m");
			expect(() => pool.assertExists(a)).toThrow(ConversationNotFoundError);
			expect(() => pool.assertExists(b)).not.toThrow();
			expect(() => pool.assertExists(c)).not.toThrow();
		});

		test("access via get updates LRU order", () => {
			const pool = new ConversationPool({ maxConversations: 2 });
			const a = pool.create("m");
			const b = pool.create("m");
			// Touch `a` so `b` becomes the oldest.
			pool.get(a);
			const c = pool.create("m");
			// `b` should be evicted (oldest), `a` survives.
			expect(() => pool.assertExists(a)).not.toThrow();
			expect(() => pool.assertExists(b)).toThrow(ConversationNotFoundError);
			expect(() => pool.assertExists(c)).not.toThrow();
		});

		test("locked entries are not evicted; throws if all locked", () => {
			const pool = new ConversationPool({ maxConversations: 2 });
			const a = pool.create("m");
			const b = pool.create("m");
			const releaseA = pool.tryAcquireLock(a);
			const releaseB = pool.tryAcquireLock(b);
			expect(releaseA).not.toBeNull();
			expect(releaseB).not.toBeNull();
			try {
				pool.create("m");
				throw new Error("expected throw");
			} catch (err) {
				expect(err).toBeInstanceOf(ConversationPoolFullError);
			}
			// Cleanup so the test doesn't leak locked entries (not strictly
			// necessary since the pool goes out of scope, but matches the
			// other tests' hygiene).
			releaseA?.();
			releaseB?.();
		});

		test("eviction skips locked entries even if older", () => {
			const pool = new ConversationPool({ maxConversations: 2 });
			const a = pool.create("m");
			const b = pool.create("m");
			// `a` is older but locked. `b` should be evicted instead.
			const releaseA = pool.tryAcquireLock(a);
			expect(releaseA).not.toBeNull();
			const c = pool.create("m");
			expect(() => pool.assertExists(a)).not.toThrow();
			expect(() => pool.assertExists(b)).toThrow(ConversationNotFoundError);
			expect(() => pool.assertExists(c)).not.toThrow();
			releaseA?.();
		});
	});
});
