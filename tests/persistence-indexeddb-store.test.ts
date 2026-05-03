import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { PersistenceUnavailableError } from "../src/core/errors.js";
import { IndexedDBConversationStore } from "../src/persistence/indexeddb-store.js";

const indexedDBAvailable = typeof indexedDB !== "undefined";
const DB_NAME = "test-conv-store";

describe.skipIf(!indexedDBAvailable)("IndexedDBConversationStore", () => {
	let store: IndexedDBConversationStore;

	beforeEach(async () => {
		await new Promise<void>((resolve) => {
			const req = indexedDB.deleteDatabase(DB_NAME);
			req.onsuccess = () => resolve();
			req.onerror = () => resolve();
			req.onblocked = () => resolve();
		});
		store = new IndexedDBConversationStore(DB_NAME);
	});

	afterEach(async () => {
		await store.close();
	});

	test("put + get round-trip", async () => {
		const blob = new Uint8Array([1, 2, 3, 4, 5]);
		await store.put("k1", blob);
		const got = await store.get("k1");
		expect(got).toEqual(blob);
	});

	test("get of missing key returns undefined", async () => {
		expect(await store.get("nope")).toBeUndefined();
	});

	test("delete removes the entry", async () => {
		await store.put("k1", new Uint8Array([7]));
		await store.delete("k1");
		expect(await store.get("k1")).toBeUndefined();
	});

	test("list returns metadata only (byteLength + savedAtMs)", async () => {
		await store.put("k1", new Uint8Array([1, 2, 3]));
		await store.put("k2", new Uint8Array(100));
		const entries = await store.list();
		expect(entries).toHaveLength(2);
		const k1 = entries.find((e) => e.key === "k1");
		expect(k1?.byteLength).toBe(3);
		expect(k1?.savedAtMs).toBeGreaterThan(0);
	});

	test("put overwrite replaces both blob and metadata", async () => {
		await store.put("k1", new Uint8Array([1, 2, 3]));
		await store.put("k1", new Uint8Array([1, 2, 3, 4, 5, 6]));
		const got = await store.get("k1");
		expect(got?.byteLength).toBe(6);
		const entries = await store.list();
		expect(entries.find((e) => e.key === "k1")?.byteLength).toBe(6);
	});

	test("clear removes all entries", async () => {
		await store.put("k1", new Uint8Array([1]));
		await store.put("k2", new Uint8Array([2]));
		await store.clear();
		expect(await store.list()).toEqual([]);
	});

	test("open() is idempotent", async () => {
		await store.open();
		await store.open();
		await store.put("k1", new Uint8Array([9]));
		expect((await store.get("k1"))?.[0]).toBe(9);
	});

	test("indexedDB-missing throws PersistenceUnavailableError", async () => {
		const original = (globalThis as { indexedDB?: unknown }).indexedDB;
		(globalThis as { indexedDB?: unknown }).indexedDB = undefined;
		try {
			const orphan = new IndexedDBConversationStore("no-idb");
			await expect(orphan.put("k", new Uint8Array([1]))).rejects.toBeInstanceOf(
				PersistenceUnavailableError,
			);
		} finally {
			(globalThis as { indexedDB?: unknown }).indexedDB = original;
		}
	});
});
