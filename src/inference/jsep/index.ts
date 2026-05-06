/**
 * Public entry point for the JSEP-style WebGPU backend runtime scaffold.
 *
 * `installJsepCallbacks(module, device)` wires the seven `Module.jsep*`
 * hooks the C++ side (`ggml-jsep.cpp`) calls via `EM_ASM` / `EM_JS`:
 *
 *   - `jsepAlloc(size)`               → integer handle (≥1)
 *   - `jsepFree(handle)`              → void
 *   - `jsepWrite(h, off, ptr, size)`  → void (sync; queues writeBuffer)
 *   - `jsepRead(h, off, ptr, size)`   → Promise<void> (async readback)
 *   - `jsepClear(h, val, off, size)`  → void
 *   - `jsepRunOp(...)`                → status code (NOT_IMPLEMENTED in Task 3)
 *   - `jsepSync()`                    → void (flush command encoder)
 *
 * `module.jsepRead` returns a Promise — JSPI's promising-wrap awaits it
 * on the wasm side because `_backend_tensor_get` is in `JSPI_EXPORTS`.
 *
 * Op dispatch is intentionally absent in Task 3: matmul (Task 4) and
 * RMS_NORM (Task 5) populate the dispatch table later; until then,
 * `jsepRunOp` returns `STATUS_NOT_IMPLEMENTED` for every op kind so the
 * C++ side falls back to the CPU backend.
 */

import { CommandEncoderBatcher } from "./command-encoder.js";
import { GpuDataManager } from "./gpu-data-manager.js";
import { PipelineCache } from "./pipeline-cache.js";

export const STATUS_OK = 0;
export const STATUS_NOT_IMPLEMENTED = 1;
export const STATUS_FAILED = -1;

/**
 * Minimal Emscripten module shape the JSEP scaffold needs. The real
 * module type carries many more fields; we narrow to the surface the
 * scaffold touches so test stubs are easy to construct.
 */
export interface JsepModule {
	HEAPU8: Uint8Array;
	jsepAlloc?: (size: number) => number;
	jsepFree?: (handle: number) => void;
	jsepWrite?: (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	) => void;
	jsepRead?: (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	) => Promise<void>;
	jsepClear?: (
		handle: number,
		value: number,
		offset: number,
		size: number,
	) => void;
	jsepRunOp?: (
		op: number,
		srcHandlesPtr: number,
		dstHandle: number,
		opParamsPtr: number,
		paramsLen: number,
	) => number;
	jsepSync?: () => void;
	__jsep?: JsepRuntime;
}

export interface JsepRuntime {
	device: GPUDevice;
	dataManager: GpuDataManager;
	encoderBatcher: CommandEncoderBatcher;
	pipelineCache: PipelineCache;
}

/**
 * Install the seven `jsep*` callbacks on a Module-like object.
 *
 * Must be called at most once per module — call `destroyJsepCallbacks`
 * first to re-install.
 *
 * `device` is taken as a parameter for testability. The Task 6 engine
 * integration pulls it from `module.preinitializedWebGPUDevice`.
 */
export function installJsepCallbacks(
	module: JsepModule,
	device: GPUDevice,
): JsepRuntime {
	if (module.__jsep) {
		throw new Error(
			"installJsepCallbacks: callbacks already installed on this module. " +
				"Call destroyJsepCallbacks(module) first if you need to re-install.",
		);
	}
	const dataManager = new GpuDataManager(device);
	const encoderBatcher = new CommandEncoderBatcher(device);
	const pipelineCache = new PipelineCache(device);

	const runtime: JsepRuntime = {
		device,
		dataManager,
		encoderBatcher,
		pipelineCache,
	};
	module.__jsep = runtime;

	module.jsepAlloc = (size: number): number => dataManager.alloc(size);

	module.jsepFree = (handle: number): void => {
		dataManager.free(handle);
	};

	module.jsepWrite = (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	): void => {
		// Re-derive the heap buffer each call — the WASM heap may have
		// grown between EM_ASM frames, detaching prior views.
		dataManager.write(handle, offset, hostPtr, size, module.HEAPU8.buffer);
	};

	module.jsepRead = (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	): Promise<void> =>
		dataManager.readAsync(handle, offset, hostPtr, size, module.HEAPU8.buffer);

	module.jsepClear = (
		handle: number,
		value: number,
		offset: number,
		size: number,
	): void => {
		dataManager.clear(handle, value, offset, size);
	};

	module.jsepRunOp = (
		_op: number,
		_srcHandlesPtr: number,
		_dstHandle: number,
		_opParamsPtr: number,
		_paramsLen: number,
	): number => {
		// No op kernels installed in Task 3. Tasks 4+5 wire matmul +
		// RMS_NORM through a dispatch table keyed on `op`.
		return STATUS_NOT_IMPLEMENTED;
	};

	module.jsepSync = (): void => {
		encoderBatcher.flush();
	};

	return runtime;
}

/**
 * Tear down a previously installed JSEP runtime. Destroys all GPU
 * resources and removes the callback hooks so installJsepCallbacks
 * can be invoked again.
 */
export function destroyJsepCallbacks(module: JsepModule): void {
	const runtime = module.__jsep;
	if (!runtime) return;
	runtime.dataManager.destroy();
	delete module.__jsep;
	delete module.jsepAlloc;
	delete module.jsepFree;
	delete module.jsepWrite;
	delete module.jsepRead;
	delete module.jsepClear;
	delete module.jsepRunOp;
	delete module.jsepSync;
}
