import { describe, expect, test } from "bun:test";
import { GgmlWasm } from "../src/inference/ggml-wasm.js";

type MockModule = {
	HEAPU8: Uint8Array;
	Asyncify?: {
		currData: object | null;
		whenDone(): Promise<number>;
	};
	_malloc(size: number): number;
	_free(ptr: number): void;
	_bridge_malloc(size: number): number;
	_bridge_free(ptr: number): void;
	_graph_compute?(graph: number): number;
	_webgpu_set_graph_profiling_enabled?(enabled: number): void;
	_webgpu_last_graph_profile_valid?(): number;
	_webgpu_last_graph_profile_breakdown_available?(): number;
	_webgpu_last_graph_profile_total_ms?(): number;
	_webgpu_last_graph_profile_matmul_ms?(): number;
	_webgpu_last_graph_profile_attention_ms?(): number;
	_webgpu_last_graph_profile_encode_overhead_ms?(): number;
	_webgpu_last_graph_profile_dispatch_count?(): number;
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
	_backend_tensor_get_async_callback_support?(): number;
	__webllmNotifyAsyncTensorGet?(requestId: number, state: number): void;
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
		_bridge_malloc: (size: number) => {
			calls.push(`malloc:${size}`);
			return 8;
		},
		_bridge_free: (ptr: number) => {
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

describe("GgmlWasm.malloc/free routes through bridge_malloc/bridge_free", () => {
	test("malloc calls _bridge_malloc, not _malloc", () => {
		let bridgeCalled = 0;
		let stdlibCalled = 0;
		const fakeModule = {
			_bridge_malloc: (size: number) => {
				bridgeCalled++;
				return 0xac0000 + size;
			},
			_bridge_free: (_ptr: number) => {},
			_malloc: (_size: number) => {
				stdlibCalled++;
				return 0;
			},
			_free: (_ptr: number) => {},
			HEAPU8: new Uint8Array(64),
			HEAPF32: new Float32Array(16),
		};
		const wasm = new GgmlWasm();
		// biome-ignore lint/suspicious/noExplicitAny: test injection
		(wasm as any).m = fakeModule;
		// biome-ignore lint/suspicious/noExplicitAny: test injection
		(wasm as any).is64 = false;

		const ptr = wasm.malloc(16);
		expect(ptr).toBe(0xac0010);
		wasm.free(ptr);

		expect(bridgeCalled).toBe(1);
		expect(stdlibCalled).toBe(0);
	});

	test("malloc normalizes BigInt return value to number under wasm64", () => {
		const fakeModule = {
			_bridge_malloc: (size: bigint) => {
				expect(typeof size).toBe("bigint");
				return BigInt(0xac0000) + size;
			},
			_bridge_free: (_ptr: bigint) => {},
			HEAPU8: new Uint8Array(64),
			HEAPF32: new Float32Array(16),
		};
		const wasm = new GgmlWasm();
		// biome-ignore lint/suspicious/noExplicitAny: test injection
		(wasm as any).m = fakeModule;
		// biome-ignore lint/suspicious/noExplicitAny: test injection
		(wasm as any).is64 = true;

		const ptr = wasm.malloc(16);
		expect(typeof ptr).toBe("number");
		expect(ptr).toBe(0xac0010);
		wasm.free(ptr);
	});
});

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

describe("GgmlWasm detailed graph profiling controls", () => {
	test("graphComputeWithDetailedProfile scopes the backend profiling toggle to one compute", async () => {
		const calls: string[] = [];
		const { wasm } = createWasm({
			_webgpu_set_graph_profiling_enabled: (enabled) => {
				calls.push(`profile:${enabled}`);
			},
			_graph_compute: (graph) => {
				calls.push(`compute:${graph}`);
				return 123;
			},
		});

		await expect(wasm.graphComputeWithDetailedProfile(7)).resolves.toBe(123);
		expect(calls).toEqual(["profile:1", "compute:7", "profile:0"]);
	});

	test("serializes profiled and unprofiled graph computes so profiling does not leak across overlap", async () => {
		const calls: string[] = [];
		const deferred = () => {
			let resolve!: (value: number) => void;
			const promise = new Promise<number>((innerResolve) => {
				resolve = innerResolve;
			});
			return { promise, resolve };
		};
		const pending = new Map<number, ReturnType<typeof deferred>>();
		let activeGraph: number | null = null;
		const asyncify = {
			currData: null as object | null,
			whenDone: () => {
				if (activeGraph === null) {
					throw new Error("no active graph");
				}
				const request = pending.get(activeGraph);
				if (!request) {
					throw new Error(`missing pending graph ${activeGraph}`);
				}
				return request.promise;
			},
		};
		const { wasm } = createWasm({
			Asyncify: asyncify,
			_webgpu_set_graph_profiling_enabled: (enabled) => {
				calls.push(`profile:${enabled}`);
			},
			_graph_compute: (graph) => {
				calls.push(`compute-start:${graph}`);
				const token = {};
				const request = deferred();
				pending.set(graph, request);
				activeGraph = graph;
				asyncify.currData = token;
				request.promise.then(() => {
					calls.push(`compute-end:${graph}`);
					if (asyncify.currData === token) {
						asyncify.currData = null;
					}
				});
				return graph * 10;
			},
		});

		const profiled = wasm.graphComputeWithDetailedProfile(1);
		const plain = wasm.graphCompute(2);
		await Promise.resolve();

		expect(calls).toEqual(["profile:1", "compute-start:1"]);

		pending.get(1)?.resolve(10);
		await expect(profiled).resolves.toBe(10);
		await Promise.resolve();
		expect(calls).toEqual([
			"profile:1",
			"compute-start:1",
			"compute-end:1",
			"profile:0",
			"compute-start:2",
		]);

		pending.get(2)?.resolve(20);
		await expect(plain).resolves.toBe(20);
		expect(calls).toEqual([
			"profile:1",
			"compute-start:1",
			"compute-end:1",
			"profile:0",
			"compute-start:2",
			"compute-end:2",
		]);
	});
});

describe("GgmlWasm.getLastGraphComputeProfile", () => {
	test("returns the last graph compute profiling summary when available", () => {
		const { wasm } = createWasm({
			_webgpu_last_graph_profile_valid: () => 1,
			_webgpu_last_graph_profile_breakdown_available: () => 1,
			_webgpu_last_graph_profile_total_ms: () => 12.5,
			_webgpu_last_graph_profile_matmul_ms: () => 8.25,
			_webgpu_last_graph_profile_attention_ms: () => 1.5,
			_webgpu_last_graph_profile_encode_overhead_ms: () => 0.75,
			_webgpu_last_graph_profile_dispatch_count: () => 14,
		});

		expect(wasm.getLastGraphComputeProfile()).toEqual({
			totalMs: 12.5,
			matmulMs: 8.25,
			attentionMs: 1.5,
			encodeOverheadMs: 0.75,
			dispatchCount: 14,
			breakdownAvailable: true,
		});
	});

	test("returns null breakdown timings when GPU attribution is unavailable", () => {
		const { wasm } = createWasm({
			_webgpu_last_graph_profile_valid: () => 1,
			_webgpu_last_graph_profile_breakdown_available: () => 0,
			_webgpu_last_graph_profile_total_ms: () => 12.5,
			_webgpu_last_graph_profile_matmul_ms: () => 0,
			_webgpu_last_graph_profile_attention_ms: () => 0,
			_webgpu_last_graph_profile_encode_overhead_ms: () => 0.75,
			_webgpu_last_graph_profile_dispatch_count: () => 14,
		});

		expect(wasm.getLastGraphComputeProfile()).toEqual({
			totalMs: 12.5,
			matmulMs: null,
			attentionMs: null,
			encodeOverheadMs: 0.75,
			dispatchCount: 14,
			breakdownAvailable: false,
		});
	});

	test("returns null when no graph compute profile has been recorded", () => {
		const { wasm } = createWasm({
			_webgpu_last_graph_profile_valid: () => 0,
		});

		expect(wasm.getLastGraphComputeProfile()).toBeNull();
	});
});

describe("GgmlWasm.downloadFromTensor", () => {
	test("waits for callback-driven async readback completion without JS polling", async () => {
		const { wasm, heapU8, calls } = createWasm({
			_backend_tensor_get_async_begin: (tensor, offset, size) => {
				calls.push(`begin:${tensor}:${offset}:${size}`);
				return 77;
			},
			_backend_tensor_get_async_callback_support: () => 1,
			_backend_tensor_get_async_finish: (requestId, dstHeapPtr, size) => {
				calls.push(`finish:${requestId}:${dstHeapPtr}:${size}`);
				heapU8.set([1, 2, 3, 4], dstHeapPtr);
			},
		});
		const setTimeoutCalls: number[] = [];
		const originalSetTimeout = globalThis.setTimeout;
		globalThis.setTimeout = ((handler: TimerHandler, timeout?: number) => {
			setTimeoutCalls.push(timeout ?? 0);
			return originalSetTimeout(handler, timeout);
		}) as typeof globalThis.setTimeout;

		try {
			const request = wasm.beginDownloadFromTensor(7, 4, 12);
			queueMicrotask(() => {
				(wasm as unknown as { m: MockModule }).m.__webllmNotifyAsyncTensorGet?.(
					77,
					2,
				);
			});
			await request.wait();
			await expect(request.finish()).resolves.toEqual(
				new Uint8Array([1, 2, 3, 4]),
			);
			expect(setTimeoutCalls).toEqual([]);
			expect(request.timings.beginMs).toBeGreaterThanOrEqual(0);
			expect(request.timings.waitMs).toBeGreaterThanOrEqual(0);
			expect(request.timings.finishMs).toBeGreaterThanOrEqual(0);
			expect(request.timings.copyMs).toBeGreaterThanOrEqual(0);
			expect(calls).toEqual([
				"malloc:4",
				"begin:7:12:4",
				"finish:77:8:4",
				"free:8",
			]);
		} finally {
			globalThis.setTimeout = originalSetTimeout;
		}
	});

	test("falls back to polling when callback completion is unavailable", async () => {
		const { wasm, heapU8, calls } = createWasm({
			_backend_tensor_get_async_begin: (tensor, offset, size) => {
				calls.push(`begin:${tensor}:${offset}:${size}`);
				return 78;
			},
			_backend_tensor_get_async_poll: (requestId) => {
				calls.push(`poll:${requestId}`);
				return calls.filter((call) => call === "poll:78").length >= 2 ? 1 : 0;
			},
			_backend_tensor_get_async_finish: (requestId, dstHeapPtr, size) => {
				calls.push(`finish:${requestId}:${dstHeapPtr}:${size}`);
				heapU8.set([5, 6, 7, 8], dstHeapPtr);
			},
		});

		await expect(wasm.downloadFromTensor(7, 4, 12)).resolves.toEqual(
			new Uint8Array([5, 6, 7, 8]),
		);
		expect(calls).toEqual([
			"malloc:4",
			"begin:7:12:4",
			"poll:78",
			"poll:78",
			"finish:78:8:4",
			"free:8",
		]);
	});

	test("keeps downloadFromTensor compatibility via the request wrapper", async () => {
		const { wasm, heapU8, calls } = createWasm({
			_backend_tensor_get_async_begin: (tensor, offset, size) => {
				calls.push(`begin:${tensor}:${offset}:${size}`);
				return 17;
			},
			_backend_tensor_get_async_poll: (requestId) => {
				calls.push(`poll:${requestId}`);
				return 1;
			},
			_backend_tensor_get_async_finish: (requestId, dstHeapPtr, size) => {
				calls.push(`finish:${requestId}:${dstHeapPtr}:${size}`);
				heapU8.set([9, 8, 7, 6], dstHeapPtr);
			},
		});

		await expect(wasm.downloadFromTensor(7, 4, 12)).resolves.toEqual(
			new Uint8Array([9, 8, 7, 6]),
		);
		expect(calls).toEqual([
			"malloc:4",
			"begin:7:12:4",
			"poll:17",
			"finish:17:8:4",
			"free:8",
		]);
	});

	test("cancels unfinished requests and frees heap memory when async finish throws", async () => {
		const { wasm, calls } = createWasm({
			_backend_tensor_get_async_begin: () => 99,
			_backend_tensor_get_async_poll: () => 1,
			_backend_tensor_get_async_finish: () => {
				throw new Error("finish failed");
			},
			_backend_tensor_get_async_cancel: (requestId) => {
				calls.push(`cancel:${requestId}`);
			},
		});

		await expect(wasm.downloadFromTensor(3, 8, 0)).rejects.toThrow(
			"finish failed",
		);
		expect(calls).toContain("cancel:99");
		expect(calls).toContain("free:8");
		expect(calls.at(-1)).toBe("free:8");
	});
});
