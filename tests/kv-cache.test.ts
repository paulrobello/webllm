import { describe, expect, test } from "bun:test";
import { KVCache, type KVCacheConfig } from "../src/models/kv-cache.js";

const BASE_CONFIG: KVCacheConfig = {
	nLayers: 4,
	nEmbdHeadK: 64,
	nEmbdHeadV: 64,
	nKvHead: 8,
	maxContextLength: 128,
	dataType: "f32",
};

describe("KVCache", () => {
	test("finds contiguous slots from start", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(5, 0);
		expect(slots).toHaveLength(5);
		expect(slots[0]).toBe(0);
		expect(slots[4]).toBe(4);
	});

	test("advances head after allocation", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots1 = cache.findSlots(10, 0);
		cache.updateSlots(
			slots1,
			Array.from({ length: 10 }, (_, i) => i),
			0,
		);
		const slots2 = cache.findSlots(5, 0);
		expect(slots2[0]).toBe(10);
	});

	test("wraps around ring buffer", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 16 });
		const slots1 = cache.findSlots(12, 0);
		cache.updateSlots(
			slots1,
			Array.from({ length: 12 }, (_, i) => i),
			0,
		);
		const slots2 = cache.findSlots(4, 1);
		expect(slots2).toHaveLength(4);
	});

	test("updateSlots sets position and sequence", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(3, 0);
		cache.updateSlots(slots, [0, 1, 2], 0);
		expect(cache.usedCellsCount).toBe(3);
		expect(cache.utilizationRatio).toBeCloseTo(3 / 128);
	});

	test("evictSequence frees cells", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(5, 0);
		cache.updateSlots(slots, [0, 1, 2, 3, 4], 0);
		cache.evictSequence(0);
		expect(cache.usedCellsCount).toBe(0);
	});

	test("setKBuffer/getKBuffer round-trip", () => {
		const cache = new KVCache(BASE_CONFIG);
		cache.setKBuffer(0, 42);
		expect(cache.getKBuffer(0)).toBe(42);
		expect(cache.getKBuffer(1)).toBeNull();
	});

	test("setVBuffer/getVBuffer round-trip", () => {
		const cache = new KVCache(BASE_CONFIG);
		cache.setVBuffer(0, 99);
		expect(cache.getVBuffer(0)).toBe(99);
		expect(cache.getVBuffer(1)).toBeNull();
	});

	test("reset clears all state", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(10, 0);
		cache.updateSlots(
			slots,
			Array.from({ length: 10 }, (_, i) => i),
			0,
		);
		cache.setKBuffer(0, 1);
		cache.setVBuffer(0, 2);
		cache.reset();
		expect(cache.usedCellsCount).toBe(0);
		expect(cache.getKBuffer(0)).toBeNull();
		expect(cache.getVBuffer(0)).toBeNull();
	});

	test("reports full when no slots available", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 8 });
		const slots = cache.findSlots(8, 0);
		cache.updateSlots(
			slots,
			Array.from({ length: 8 }, (_, i) => i),
			0,
		);
		const result = cache.findSlots(1, 1);
		expect(result).toHaveLength(0);
	});

	test("evicts only targeted sequence", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 16 });
		const slots1 = cache.findSlots(4, 0);
		cache.updateSlots(slots1, [0, 1, 2, 3], 0);
		const slots2 = cache.findSlots(4, 1);
		cache.updateSlots(slots2, [0, 1, 2, 3], 1);
		expect(cache.usedCellsCount).toBe(8);
		cache.evictSequence(0);
		expect(cache.usedCellsCount).toBe(4);
	});

	test("utilizationRatio is 0 on empty cache", () => {
		const cache = new KVCache(BASE_CONFIG);
		expect(cache.utilizationRatio).toBe(0);
	});

	test("totalCells equals maxContextLength", () => {
		const cache = new KVCache(BASE_CONFIG);
		expect(cache.totalCells).toBe(128);
	});

	test("findSlots returns empty when request exceeds buffer", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 16 });
		const result = cache.findSlots(20, 0);
		expect(result).toHaveLength(0);
	});

	test("sharePromptCells copies positions to new sequence", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 32 });
		const promptLen = 5;
		// Sequence 0: fill positions 0..4 (the "system prompt")
		const slots = cache.findSlots(promptLen, 0);
		cache.updateSlots(
			slots,
			Array.from({ length: promptLen }, (_, i) => i),
			0,
		);
		expect(cache.usedCellsCount).toBe(promptLen);

		// Share those cells with sequence 1
		const shared = cache.sharePromptCells(0, 1, promptLen);
		expect(shared).toBe(promptLen);

		// Sequence 1 should now see those cells
		const cells = cache.getSequenceCells(1);
		expect(cells).toHaveLength(promptLen);
		// Sequence 0 still sees them too
		expect(cache.getSequenceCells(0)).toHaveLength(promptLen);
	});

	test("hasPromptCache detects shared prompt", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 32 });
		const promptLen = 5;
		const slots = cache.findSlots(promptLen, 0);
		cache.updateSlots(
			slots,
			Array.from({ length: promptLen }, (_, i) => i),
			0,
		);

		// Before sharing, sequence 1 has no prompt cache
		expect(cache.hasPromptCache(1, promptLen)).toBe(false);

		cache.sharePromptCells(0, 1, promptLen);

		// After sharing, sequence 1 has a full prompt cache
		expect(cache.hasPromptCache(1, promptLen)).toBe(true);
		// Original sequence still has its cache
		expect(cache.hasPromptCache(0, promptLen)).toBe(true);
	});

	test("evictSequence only removes one sequence from shared cells", () => {
		const cache = new KVCache({ ...BASE_CONFIG, maxContextLength: 32 });
		const promptLen = 5;
		const slots = cache.findSlots(promptLen, 0);
		cache.updateSlots(
			slots,
			Array.from({ length: promptLen }, (_, i) => i),
			0,
		);
		cache.sharePromptCells(0, 1, promptLen);

		// Both sequences share the cells: usedCells = 5 (original) + 5 (shared)
		expect(cache.usedCellsCount).toBe(promptLen * 2);

		// Evict only sequence 0
		cache.evictSequence(0);

		// The cells are still occupied by sequence 1, so they should NOT be freed
		expect(cache.getSequenceCells(1)).toHaveLength(promptLen);
		// Only the share count was freed (5 shared refs removed)
		expect(cache.usedCellsCount).toBe(promptLen);

		// Now evict sequence 1 -- cells should become truly empty
		cache.evictSequence(1);
		expect(cache.usedCellsCount).toBe(0);
		expect(cache.getSequenceCells(1)).toHaveLength(0);
	});

	test("getMemoryUsage returns bytes for occupied cells", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(5, 0);
		cache.updateSlots(slots, [0, 1, 2, 3, 4], 0);
		// bytesPerCell = nLayers * (nEmbdHeadK + nEmbdHeadV) * 4 = 4 * (64+64) * 4 = 2048
		const expectedBytes = 5 * 4 * (64 + 64) * 4;
		expect(cache.getMemoryUsage()).toBe(expectedBytes);
	});

	test("getSequenceCells returns empty for unknown sequence", () => {
		const cache = new KVCache(BASE_CONFIG);
		const slots = cache.findSlots(3, 0);
		cache.updateSlots(slots, [0, 1, 2], 0);
		expect(cache.getSequenceCells(99)).toHaveLength(0);
	});
});
