import { describe, expect, test } from "bun:test";
import { MemoryPool } from "../src/core/memory-pool.js";
import { ModelManager } from "../src/core/model-manager.js";
import type { ModelEntry, ModelHyperparams } from "../src/core/types.js";
import type { Tokenizer } from "../src/inference/tokenizer.js";
import { KVCache, type KVCacheConfig } from "../src/models/kv-cache.js";

const BASE_KV_CONFIG: KVCacheConfig = {
	nLayers: 4,
	nEmbdHeadK: 64,
	nEmbdHeadV: 64,
	nKvHead: 8,
	maxContextLength: 128,
	dataType: "f32",
};

function makeEntry(id: string, priority: number): ModelEntry {
	return {
		id,
		name: id,
		priority,
		lightweight: false,
		hyperparams: {} as ModelHyperparams,
		kvCache: new KVCache(BASE_KV_CONFIG),
		tokenizer: null as unknown as Tokenizer,
		memoryAllocations: [],
		loaded: true,
		activeSessions: 0,
	};
}

describe("ModelManager", () => {
	test("register and get model", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		const entry = makeEntry("m1", 0);
		mgr.register(entry);
		expect(mgr.get("m1")).toBe(entry);
	});

	test("getAll returns sorted by priority", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 2));
		mgr.register(makeEntry("m2", 0));
		mgr.register(makeEntry("m3", 1));
		const all = mgr.getAll();
		expect(all.map((m) => m.id)).toEqual(["m2", "m3", "m1"]);
	});

	test("getPrimary returns highest priority", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 2));
		mgr.register(makeEntry("m2", 0));
		expect(mgr.getPrimary()?.id).toBe("m2");
	});

	test("unregister removes model", async () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 0));
		await mgr.unregister("m1");
		expect(mgr.get("m1")).toBeUndefined();
		expect(mgr.count).toBe(0);
	});

	test("evictLowestPriority removes highest-number priority", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 0));
		mgr.register(makeEntry("m2", 2));
		mgr.register(makeEntry("m3", 1));
		const evicted = mgr.evictLowestPriority();
		expect(evicted).toBe("m2");
		expect(mgr.count).toBe(2);
	});

	test("session tracking", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 0));
		mgr.addSession("m1");
		mgr.addSession("m1");
		expect(mgr.get("m1")?.activeSessions).toBe(2);
		mgr.removeSession("m1");
		expect(mgr.get("m1")?.activeSessions).toBe(1);
	});

	test("clear removes all models", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		mgr.register(makeEntry("m1", 0));
		mgr.register(makeEntry("m2", 1));
		mgr.clear();
		expect(mgr.count).toBe(0);
	});

	test("count returns model count", () => {
		const mgr = new ModelManager(new MemoryPool(4096));
		expect(mgr.count).toBe(0);
		mgr.register(makeEntry("m1", 0));
		expect(mgr.count).toBe(1);
	});

	test("getTotalMemoryUsage delegates to memory pool", () => {
		const pool = new MemoryPool(4096);
		pool.allocate(1024, 0, "m1");
		const mgr = new ModelManager(pool);
		mgr.register(makeEntry("m1", 0));
		expect(mgr.getTotalMemoryUsage()).toBe(1024);
	});
});
