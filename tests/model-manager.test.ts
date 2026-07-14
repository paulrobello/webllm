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

	test("pool usedBytes > 0 after allocate and budget gate reflects usage", () => {
		// ARC-002 regression: prior to wiring the allocate side in
		// `_buildInferenceAndRegister`, `usedBytes` stayed at 0 forever and
		// the budget gate was inert. This test pins the wired behavior by
		// simulating the exact allocate call the engine makes on load.
		const pool = new MemoryPool(2048);
		const mgr = new ModelManager(pool);
		mgr.register(makeEntry("m1", 0));
		// Engine wires one allocation per loaded model, tagged with its id.
		pool.allocate(1536, 0, "m1");
		expect(pool.usedBytes).toBe(1536);
		expect(mgr.getTotalMemoryUsage()).toBe(1536);
		expect(pool.canAllocate(512)).toBe(true); // 1536 + 512 = 2048, exactly fits
		expect(pool.canAllocate(513)).toBe(false); // over by 1 byte
		// canFit goes through eviction too — with m1 registered and no
		// active sessions, pool-level eviction IS allowed and reclaims
		// the 1536 allocation as a side effect of the probe.
		expect(mgr.canFit(513)).toBe(true);
	});

	test("unregister frees the model's allocations and re-opens the budget", async () => {
		// ARC-002: unloadModel → _modelManager.unregister → memoryPool.evictModel
		// is the matching free path for the allocate wired in load.
		const pool = new MemoryPool(2048);
		const mgr = new ModelManager(pool);
		mgr.register(makeEntry("m1", 0));
		pool.allocate(1536, 0, "m1");
		expect(pool.usedBytes).toBe(1536);
		await mgr.unregister("m1");
		expect(pool.usedBytes).toBe(0);
		expect(pool.getModelUsage("m1")).toBe(0);
		// After unregister, the budget is fully available again.
		expect(mgr.canFit(2048)).toBe(true);
	});
});
