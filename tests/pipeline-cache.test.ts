import { describe, expect, test } from "bun:test";
import { PipelineCache } from "../src/core/pipeline-cache.js";

// IndexedDB is not available in Bun's test runtime.
// These tests run correctly in browser environments (Chrome, Edge, Firefox).
// Use Playwright or similar for browser-based test execution.
const indexedDBAvailable = typeof indexedDB !== "undefined";

describe("PipelineCache", () => {
	test.skipIf(!indexedDBAvailable)(
		"stores and retrieves pipeline data",
		async () => {
			const cache = new PipelineCache("test-cache");
			const data = new Uint8Array([1, 2, 3, 4]);
			await cache.put("pipeline-matmul-q4_0", data);
			const retrieved = await cache.get("pipeline-matmul-q4_0");
			expect(retrieved).toEqual(data);
		},
	);

	test.skipIf(!indexedDBAvailable)(
		"returns undefined for missing key",
		async () => {
			const cache = new PipelineCache("test-cache-missing");
			const result = await cache.get("nonexistent");
			expect(result).toBeUndefined();
		},
	);

	test.skipIf(!indexedDBAvailable)("deletes a cached pipeline", async () => {
		const cache = new PipelineCache("test-cache-delete");
		await cache.put("to-delete", new Uint8Array([5, 6]));
		await cache.delete("to-delete");
		const result = await cache.get("to-delete");
		expect(result).toBeUndefined();
	});

	test.skipIf(!indexedDBAvailable)(
		"lists all cached pipeline keys",
		async () => {
			const cache = new PipelineCache("test-cache-list");
			await cache.put("a", new Uint8Array([1]));
			await cache.put("b", new Uint8Array([2]));
			const keys = await cache.keys();
			expect(keys).toContain("a");
			expect(keys).toContain("b");
		},
	);

	test.skipIf(!indexedDBAvailable)("clears all cached data", async () => {
		const cache = new PipelineCache("test-cache-clear");
		await cache.put("x", new Uint8Array([1]));
		await cache.put("y", new Uint8Array([2]));
		await cache.clear();
		const keys = await cache.keys();
		expect(keys).toHaveLength(0);
	});
});
