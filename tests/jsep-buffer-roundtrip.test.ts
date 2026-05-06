/**
 * JSEP backend — buffer roundtrip unit tests.
 *
 * Exercises the JS-side runtime scaffold against a real `GPUDevice`.
 * Bun's test runner has no WebGPU; without it, the entire suite skips
 * cleanly. Browser smoke-test coverage exercises the full path against
 * the patched WASM build.
 */

import { describe, expect, test } from "bun:test";
import { GpuDataManager } from "../src/inference/jsep/gpu-data-manager.js";
import {
	destroyJsepCallbacks,
	installJsepCallbacks,
	type JsepModule,
	STATUS_NOT_IMPLEMENTED,
} from "../src/inference/jsep/index.js";

const HAS_WEBGPU =
	typeof navigator !== "undefined" &&
	typeof (navigator as Navigator & { gpu?: unknown }).gpu !== "undefined";

async function getDevice(): Promise<GPUDevice | null> {
	if (!HAS_WEBGPU) return null;
	try {
		const adapter = await navigator.gpu.requestAdapter();
		if (!adapter) return null;
		return await adapter.requestDevice();
	} catch {
		return null;
	}
}

describe("JSEP buffer roundtrip", () => {
	if (!HAS_WEBGPU) {
		test.skip("requires WebGPU; covered by browser smoke", () => {});
		return;
	}

	test("roundtrip identity (write → read bit-exact)", async () => {
		const device = await getDevice();
		if (!device) {
			console.warn("WebGPU adapter unavailable; skipping");
			return;
		}
		const mgr = new GpuDataManager(device);
		const sizeBytes = 1024;
		const handle = mgr.alloc(sizeBytes);
		expect(handle).toBeGreaterThanOrEqual(1);

		const src = new Float32Array(sizeBytes / 4);
		for (let i = 0; i < src.length; i++) src[i] = i * 1.25 - 7;

		// Stage WASM-heap surrogate: a fresh ArrayBuffer the same shape
		// the real heap would expose. write/readAsync take the buffer
		// each call, so heap-grow safety is reproduced.
		const heap = new ArrayBuffer(8192);
		const heapView = new Uint8Array(heap);
		heapView.set(new Uint8Array(src.buffer));

		mgr.write(handle, 0, 0, sizeBytes, heap);

		// Wipe the destination region in the heap before readback so the
		// equality check actually proves data came from the GPU.
		const readPtr = 4096;
		new Uint8Array(heap, readPtr, sizeBytes).fill(0xab);

		await mgr.readAsync(handle, 0, readPtr, sizeBytes, heap);

		const got = new Float32Array(heap, readPtr, src.length);
		for (let i = 0; i < src.length; i++) {
			expect(got[i]).toBe(src[i]);
		}

		mgr.free(handle);
		device.destroy();
	});

	test("sequential writes to disjoint regions", async () => {
		const device = await getDevice();
		if (!device) return;
		const mgr = new GpuDataManager(device);
		const sizeBytes = 2048;
		const handle = mgr.alloc(sizeBytes);

		const heap = new ArrayBuffer(8192);
		const partA = new Uint8Array(heap, 0, 1024);
		const partB = new Uint8Array(heap, 1024, 1024);
		for (let i = 0; i < 1024; i++) {
			partA[i] = i & 0xff;
			partB[i] = (i + 73) & 0xff;
		}

		mgr.write(handle, 0, 0, 1024, heap);
		mgr.write(handle, 1024, 1024, 1024, heap);

		const readPtr = 4096;
		new Uint8Array(heap, readPtr, sizeBytes).fill(0);
		await mgr.readAsync(handle, 0, readPtr, sizeBytes, heap);

		const out = new Uint8Array(heap, readPtr, sizeBytes);
		for (let i = 0; i < 1024; i++) {
			expect(out[i]).toBe(i & 0xff);
			expect(out[1024 + i]).toBe((i + 73) & 0xff);
		}

		mgr.free(handle);
		device.destroy();
	});

	test("clear then write-after-clear", async () => {
		const device = await getDevice();
		if (!device) return;
		const mgr = new GpuDataManager(device);
		const sizeBytes = 512;
		const handle = mgr.alloc(sizeBytes);

		const heap = new ArrayBuffer(4096);
		new Uint8Array(heap, 0, sizeBytes).fill(1);
		mgr.write(handle, 0, 0, sizeBytes, heap);

		mgr.clear(handle, 0, 0, sizeBytes);

		const readPtr = 2048;
		new Uint8Array(heap, readPtr, sizeBytes).fill(0xff);
		await mgr.readAsync(handle, 0, readPtr, sizeBytes, heap);
		const cleared = new Uint8Array(heap, readPtr, sizeBytes);
		for (let i = 0; i < sizeBytes; i++) expect(cleared[i]).toBe(0);

		new Uint8Array(heap, 0, sizeBytes).fill(2);
		mgr.write(handle, 0, 0, sizeBytes, heap);
		new Uint8Array(heap, readPtr, sizeBytes).fill(0xff);
		await mgr.readAsync(handle, 0, readPtr, sizeBytes, heap);
		const refilled = new Uint8Array(heap, readPtr, sizeBytes);
		for (let i = 0; i < sizeBytes; i++) expect(refilled[i]).toBe(2);

		mgr.free(handle);
		device.destroy();
	});

	test("bucket reuse: free → alloc same size returns a usable buffer", async () => {
		const device = await getDevice();
		if (!device) return;
		const mgr = new GpuDataManager(device);
		const sizeBytes = 4096; // falls in the 4 KB bucket
		const h1 = mgr.alloc(sizeBytes);
		const got1 = mgr.get(h1);
		expect(got1.size).toBeGreaterThanOrEqual(sizeBytes);
		mgr.free(h1);

		const h2 = mgr.alloc(sizeBytes);
		const got2 = mgr.get(h2);
		expect(got2.size).toBeGreaterThanOrEqual(sizeBytes);
		// Don't assert handle equality — the bucket may swap buffers.

		// Confirm the reused buffer still behaves correctly.
		const heap = new ArrayBuffer(8192);
		new Uint8Array(heap, 0, sizeBytes).fill(0x5a);
		mgr.write(h2, 0, 0, sizeBytes, heap);
		new Uint8Array(heap, 4096, sizeBytes).fill(0);
		await mgr.readAsync(h2, 0, 4096, sizeBytes, heap);
		const out = new Uint8Array(heap, 4096, sizeBytes);
		for (let i = 0; i < sizeBytes; i++) expect(out[i]).toBe(0x5a);

		mgr.free(h2);
		device.destroy();
	});

	test("installJsepCallbacks wires all 7 hooks; jsepRunOp returns NOT_IMPLEMENTED", async () => {
		const device = await getDevice();
		if (!device) return;
		// Synthesize a minimal Module-shaped stub. HEAPU8 wraps a real
		// ArrayBuffer so jsepWrite/jsepRead can execute end-to-end.
		const heap = new ArrayBuffer(8192);
		const stub: JsepModule = { HEAPU8: new Uint8Array(heap) };
		installJsepCallbacks(stub, device);

		expect(typeof stub.jsepAlloc).toBe("function");
		expect(typeof stub.jsepFree).toBe("function");
		expect(typeof stub.jsepWrite).toBe("function");
		expect(typeof stub.jsepRead).toBe("function");
		expect(typeof stub.jsepClear).toBe("function");
		expect(typeof stub.jsepRunOp).toBe("function");
		expect(typeof stub.jsepSync).toBe("function");

		const handle = stub.jsepAlloc?.(256) ?? 0;
		expect(handle).toBeGreaterThanOrEqual(1);

		// jsepRunOp returns NOT_IMPLEMENTED for any non-matmul op in Task 4.
		// Build a tiny descriptor with op=GGML_OP_NONE (=0) at byte 0 of the
		// HEAPU8-backed buffer; remainder doesn't matter because we bail
		// before reading any tensor block.
		const descView = new Int32Array(heap, 0, 2);
		descView[0] = 0; // GGML_OP_NONE
		descView[1] = 0; // n_src
		const status = stub.jsepRunOp?.(0, 2, 0, 0);
		expect(status).toBe(STATUS_NOT_IMPLEMENTED);

		// jsepRead must return a Promise (JSPI relies on the thenable shape).
		const readResult = stub.jsepRead?.(handle, 0, 1024, 256);
		expect(readResult).toBeInstanceOf(Promise);
		await readResult;

		stub.jsepFree?.(handle);
		stub.jsepSync?.();

		// Double-install is now an error.
		expect(() => installJsepCallbacks(stub, device)).toThrow(
			/already installed/,
		);

		// destroyJsepCallbacks tears down cleanly + re-install works.
		destroyJsepCallbacks(stub);
		expect(stub.__jsep).toBeUndefined();
		expect(stub.jsepAlloc).toBeUndefined();
		installJsepCallbacks(stub, device);
		expect(typeof stub.jsepAlloc).toBe("function");

		device.destroy();
	});
});
