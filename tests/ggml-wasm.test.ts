import { describe, expect, test } from "bun:test";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";

type MockModule = {
	HEAPU8: Uint8Array;
	_malloc(size: number): number;
	_free(ptr: number): void;
	_backend_tensor_get(
		tensor: number,
		dstHeapPtr: number,
		offset: number,
		size: number,
	): void;
	_backend_tensor_get_async_begin?(
		tensor: number,
		offset: number,
		size: number,
	): number;
	_backend_tensor_get_async_poll?(requestId: number): number;
	_backend_tensor_get_async_finish?(
		requestId: number,
		dstHeapPtr: number,
		size: number,
	): void;
	_backend_tensor_get_async_cancel?(requestId: number): void;
};

function createWasm(overrides: Partial<MockModule> = {}) {
	const calls: string[] = [];
	const heapU8 = new Uint8Array(new ArrayBuffer(64));
	const module: MockModule = {
		HEAPU8: heapU8,
		_malloc: (size: number) => {
			calls.push(`malloc:${size}`);
			return 8;
		},
		_free: (ptr: number) => {
			calls.push(`free:${ptr}`);
		},
		_backend_tensor_get: () => {
			calls.push("blocking-get");
			throw new Error("blocking _backend_tensor_get should not be used");
		},
		...overrides,
	};

	const wasm = new GgmlWasm();
	(wasm as unknown as { m: MockModule }).m = module;
	return { wasm, heapU8, calls };
}

describe("GgmlWasm async readback wrappers", () => {
	test("forwards async readback wrapper calls to wasm exports", () => {
		const calls: string[] = [];
		const { wasm } = createWasm({
			_backend_tensor_get_async_begin: (tensor, offset, size) => {
				calls.push(`begin:${tensor}:${offset}:${size}`);
				return 41;
			},
			_backend_tensor_get_async_poll: (requestId) => {
				calls.push(`poll:${requestId}`);
				return 1;
			},
			_backend_tensor_get_async_finish: (requestId, dstHeapPtr, size) => {
				calls.push(`finish:${requestId}:${dstHeapPtr}:${size}`);
			},
			_backend_tensor_get_async_cancel: (requestId) => {
				calls.push(`cancel:${requestId}`);
			},
		});

		expect(wasm.backendTensorGetAsyncBegin(7, 12, 16)).toBe(41);
		expect(wasm.backendTensorGetAsyncPoll(41)).toBe(1);
		wasm.backendTensorGetAsyncFinish(41, 24, 16);
		wasm.backendTensorGetAsyncCancel(41);

		expect(calls).toEqual([
			"begin:7:12:16",
			"poll:41",
			"finish:41:24:16",
			"cancel:41",
		]);
	});
});

describe("GgmlWasm.downloadFromTensor", () => {
	test("polls async readback requests until ready and returns copied bytes", async () => {
		const { wasm, heapU8, calls } = createWasm({
			_backend_tensor_get_async_begin: (tensor, offset, size) => {
				calls.push(`begin:${tensor}:${offset}:${size}`);
				return 77;
			},
			_backend_tensor_get_async_poll: (requestId) => {
				calls.push(`poll:${requestId}`);
				return calls.filter((call) => call === "poll:77").length >= 3 ? 1 : 0;
			},
			_backend_tensor_get_async_finish: (requestId, dstHeapPtr, size) => {
				calls.push(`finish:${requestId}:${dstHeapPtr}:${size}`);
				heapU8.set([1, 2, 3, 4], dstHeapPtr);
			},
		});

		await expect(wasm.downloadFromTensor(7, 4, 12)).resolves.toEqual(
			new Uint8Array([1, 2, 3, 4]),
		);
		expect(calls).toEqual([
			"malloc:4",
			"begin:7:12:4",
			"poll:77",
			"poll:77",
			"poll:77",
			"finish:77:8:4",
			"free:8",
		]);
	});

	test("frees heap memory when async finish throws", async () => {
		const { wasm, calls } = createWasm({
			_backend_tensor_get_async_begin: () => 99,
			_backend_tensor_get_async_poll: () => 1,
			_backend_tensor_get_async_finish: () => {
				throw new Error("finish failed");
			},
		});

		await expect(wasm.downloadFromTensor(3, 8, 0)).rejects.toThrow(
			"finish failed",
		);
		expect(calls).toContain("free:8");
		expect(calls.at(-1)).toBe("free:8");
	});
});
