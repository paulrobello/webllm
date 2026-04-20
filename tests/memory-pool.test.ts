import { describe, expect, test } from "bun:test";
import { MemoryPool } from "../src/core/memory-pool.js";

describe("MemoryPool", () => {
	test("allocates a buffer within budget", () => {
		const pool = new MemoryPool(1024);
		const id = pool.allocate(256);
		expect(id).toBe(0);
		expect(pool.usedBytes).toBe(256);
		expect(pool.remainingBytes).toBe(768);
	});

	test("throws when allocation exceeds budget", () => {
		const pool = new MemoryPool(100);
		expect(() => pool.allocate(200)).toThrow("exceeds memory budget");
	});

	test("frees a buffer and reclaims memory", () => {
		const pool = new MemoryPool(1024);
		const id = pool.allocate(256);
		pool.free(id);
		expect(pool.usedBytes).toBe(0);
		expect(pool.remainingBytes).toBe(1024);
	});

	test("tracks multiple allocations", () => {
		const pool = new MemoryPool(1024);
		const id0 = pool.allocate(256);
		const id1 = pool.allocate(512);
		expect(pool.usedBytes).toBe(768);
		pool.free(id0);
		expect(pool.usedBytes).toBe(512);
		pool.free(id1);
		expect(pool.usedBytes).toBe(0);
	});

	test("reports memory pressure correctly", () => {
		const pool = new MemoryPool(1000);
		pool.allocate(800);
		expect(pool.pressureRatio).toBe(0.8);
		expect(pool.isUnderPressure).toBe(true);
	});

	test("evicts lowest priority allocation on pressure", () => {
		const pool = new MemoryPool(1024);
		const low = pool.allocate(512, 2);
		const _high = pool.allocate(512, 0);
		expect(pool.canAllocate(256)).toBe(false);
		const evicted = pool.evictForAllocation(256);
		expect(evicted).toBe(low);
		expect(pool.canAllocate(256)).toBe(true);
	});

	test("reset clears all allocations", () => {
		const pool = new MemoryPool(1024);
		pool.allocate(256);
		pool.allocate(512);
		pool.reset();
		expect(pool.usedBytes).toBe(0);
		expect(pool.allocationCount).toBe(0);
	});

	test("getModelUsage tracks per-model bytes", () => {
		const pool = new MemoryPool(4096);
		pool.allocate(512, 0, "model-a");
		pool.allocate(256, 0, "model-b");
		pool.allocate(1024, 0, "model-a");
		expect(pool.getModelUsage("model-a")).toBe(1536);
		expect(pool.getModelUsage("model-b")).toBe(256);
		expect(pool.getModelUsage("model-c")).toBe(0);
	});

	test("evictModel frees all allocations for model", () => {
		const pool = new MemoryPool(4096);
		pool.allocate(512, 0, "model-a");
		pool.allocate(256, 0, "model-b");
		pool.allocate(1024, 0, "model-a");
		const freed = pool.evictModel("model-a");
		expect(freed).toBe(1536);
		expect(pool.usedBytes).toBe(256);
		expect(pool.allocationCount).toBe(1);
		expect(pool.getModelUsage("model-a")).toBe(0);
	});

	test("evictForAllocation excludes model", () => {
		const pool = new MemoryPool(1024);
		const lowA = pool.allocate(512, 2, "model-a");
		pool.allocate(512, 0, "model-b");
		// Without exclusion, highest priority (model-a) would be evicted
		// With exclusion of model-b, only model-a is a candidate
		const evicted = pool.evictForAllocation(256, "model-b");
		expect(evicted).toBe(lowA);
		expect(pool.canAllocate(256)).toBe(true);
	});

	test("evictForAllocation excludes model from eviction", () => {
		const pool = new MemoryPool(1024);
		pool.allocate(512, 2, "model-a");
		const lowB = pool.allocate(512, 0, "model-b");
		// Excluding model-a means only model-b can be evicted
		const evicted = pool.evictForAllocation(256, "model-a");
		expect(evicted).toBe(lowB);
		expect(pool.canAllocate(256)).toBe(true);
	});

	test("emits pressure event when threshold crossed", () => {
		const pool = new MemoryPool(1000);
		const events: string[] = [];
		pool.on("pressure", (event) => {
			events.push(event);
		});
		pool.allocate(600); // 60% — no pressure
		expect(events).toHaveLength(0);
		pool.allocate(200); // 80% — pressure threshold crossed
		expect(events).toHaveLength(1);
		expect(events[0]).toBe("pressure");
	});

	test("does not emit duplicate pressure events while above threshold", () => {
		const pool = new MemoryPool(1000);
		const events: string[] = [];
		pool.on("pressure", (event) => {
			events.push(event);
		});
		pool.allocate(800); // 80% — first pressure
		pool.allocate(100); // 90% — no new pressure
		expect(events).toHaveLength(1);
	});

	test("emits pressure again after recovery and re-pressurizing", () => {
		const pool = new MemoryPool(1000);
		const events: string[] = [];
		pool.on("pressure", (event) => {
			events.push(event);
		});
		const id = pool.allocate(800); // pressure
		pool.free(id); // back to 0% — hadPressure resets
		expect(events).toHaveLength(1);
		pool.allocate(800); // pressure again
		expect(events).toHaveLength(2);
	});

	test("emits eviction event", () => {
		const pool = new MemoryPool(1024);
		const events: Array<{
			event: string;
			detail: { allocationId?: number; modelId?: string };
		}> = [];
		pool.on("eviction", (event, detail) => {
			events.push({ event, detail });
		});
		const low = pool.allocate(512, 2, "model-a");
		pool.allocate(512, 0, "model-b");
		pool.evictForAllocation(256, "model-b");
		expect(events).toHaveLength(1);
		expect(events[0].event).toBe("eviction");
		expect(events[0].detail.allocationId).toBe(low);
		expect(events[0].detail.modelId).toBe("model-a");
	});

	test("emits allocation event", () => {
		const pool = new MemoryPool(1024);
		const events: Array<{
			event: string;
			detail: { allocationId?: number; modelId?: string };
		}> = [];
		pool.on("allocation", (event, detail) => {
			events.push({ event, detail });
		});
		pool.allocate(256, 0, "test-model");
		expect(events).toHaveLength(1);
		expect(events[0].event).toBe("allocation");
		expect(events[0].detail.modelId).toBe("test-model");
		expect(events[0].detail.allocationId).toBe(0);
	});

	test("emits free event", () => {
		const pool = new MemoryPool(1024);
		const events: Array<{
			event: string;
			detail: { allocationId?: number; modelId?: string };
		}> = [];
		pool.on("free", (event, detail) => {
			events.push({ event, detail });
		});
		const id = pool.allocate(256, 0, "test-model");
		pool.free(id);
		expect(events).toHaveLength(1);
		expect(events[0].event).toBe("free");
		expect(events[0].detail.allocationId).toBe(id);
		expect(events[0].detail.modelId).toBe("test-model");
	});

	test("getModelAllocations returns all for model", () => {
		const pool = new MemoryPool(4096);
		pool.allocate(512, 0, "model-a");
		pool.allocate(256, 1, "model-b");
		pool.allocate(1024, 2, "model-a");
		const allocs = pool.getModelAllocations("model-a");
		expect(allocs).toHaveLength(2);
		expect(allocs[0].modelId).toBe("model-a");
		expect(allocs[1].modelId).toBe("model-a");
		const totalSize = allocs.reduce((sum, a) => sum + a.size, 0);
		expect(totalSize).toBe(1536);
	});

	test("off removes event handler", () => {
		const pool = new MemoryPool(1024);
		const events: string[] = [];
		const handler = (event: string) => {
			events.push(event);
		};
		pool.on("allocation", handler);
		pool.allocate(100); // emits allocation
		pool.off("allocation", handler);
		pool.allocate(100); // no handler
		expect(events).toHaveLength(1);
	});
});
