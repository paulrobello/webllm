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
import { dispatchMatmul, readDescriptor } from "./ops/matmul.js";
import { dispatchRmsNorm } from "./ops/rms-norm.js";
import { dispatchSetRows } from "./ops/set-rows.js";
import { PipelineCache } from "./pipeline-cache.js";

export const STATUS_OK = 0;
export const STATUS_NOT_IMPLEMENTED = 1;
export const STATUS_FAILED = -1;

// ggml_op enum values (subset; see ggml/include/ggml.h:479).
// MUL_MAT = 29 (0=NONE, DUP=1, ADD=2, ADD_ID=3, ADD1=4, ACC=5, SUB=6, MUL=7,
// DIV=8, SQR=9, SQRT=10, LOG=11, SIN=12, COS=13, SUM=14, SUM_ROWS=15,
// CUMSUM=16, MEAN=17, ARGMAX=18, COUNT_EQUAL=19, REPEAT=20, REPEAT_BACK=21,
// CONCAT=22, SILU_BACK=23, NORM=24, RMS_NORM=25, RMS_NORM_BACK=26,
// GROUP_NORM=27, L2_NORM=28, MUL_MAT=29).
// Continuing from MUL_MAT=29: MUL_MAT_ID=30, OUT_PROD=31, SCALE=32, SET=33,
// CPY=34, CONT=35, RESHAPE=36, VIEW=37, PERMUTE=38, TRANSPOSE=39,
// GET_ROWS=40, GET_ROWS_BACK=41, SET_ROWS=42.
export const GGML_OP_RMS_NORM = 25;
export const GGML_OP_MUL_MAT = 29;
export const GGML_OP_SET_ROWS = 42;

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
	/**
	 * Task 4 signature — descriptor-based:
	 *   (descriptorPtr, descriptorWords, opParamsPtr, opParamsLen) → status
	 *
	 * The descriptor packs op + n_src + (dst block) + (src blocks). See
	 * `src/inference/jsep/ops/matmul.ts::readDescriptor` for the layout.
	 */
	jsepRunOp?: (
		descriptorPtr: number,
		descriptorWords: number,
		opParamsPtr: number,
		opParamsLen: number,
	) => number;
	jsepSync?: () => void;
	__jsep?: JsepRuntime;
}

/**
 * Per-callback invocation counters. Diagnostic-only — reads as
 * `module.__jsep.counters`. Used by Task 7 to compute steady-state
 * EM_ASM crossings/token (legacy gate metric).
 */
export interface JsepCounters {
	alloc: number;
	free: number;
	write: number;
	read: number;
	clear: number;
	runOp: number;
	sync: number;
}

export interface JsepRuntime {
	device: GPUDevice;
	dataManager: GpuDataManager;
	encoderBatcher: CommandEncoderBatcher;
	pipelineCache: PipelineCache;
	// Bind-group layouts memoized per pipeline cache key. Lives on the
	// runtime (not module scope) so each `GPUDevice` owns its own cache —
	// reusing a layout from a destroyed device is a WebGPU validation
	// error.
	bindGroupLayoutCache: Map<string, GPUBindGroupLayout>;
	counters: JsepCounters;
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
	const bindGroupLayoutCache = new Map<string, GPUBindGroupLayout>();
	const counters: JsepCounters = {
		alloc: 0,
		free: 0,
		write: 0,
		read: 0,
		clear: 0,
		runOp: 0,
		sync: 0,
	};

	const runtime: JsepRuntime = {
		device,
		dataManager,
		encoderBatcher,
		pipelineCache,
		bindGroupLayoutCache,
		counters,
	};
	module.__jsep = runtime;

	module.jsepAlloc = (size: number): number => {
		counters.alloc++;
		return dataManager.alloc(size);
	};

	module.jsepFree = (handle: number): void => {
		counters.free++;
		dataManager.free(handle);
	};

	module.jsepWrite = (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	): void => {
		counters.write++;
		// Re-derive the heap buffer each call — the WASM heap may have
		// grown between EM_ASM frames, detaching prior views.
		dataManager.write(handle, offset, hostPtr, size, module.HEAPU8.buffer);
	};

	module.jsepRead = (
		handle: number,
		offset: number,
		hostPtr: number,
		size: number,
	): Promise<void> => {
		counters.read++;
		return dataManager.readAsync(
			handle,
			offset,
			hostPtr,
			size,
			module.HEAPU8.buffer,
		);
	};

	module.jsepClear = (
		handle: number,
		value: number,
		offset: number,
		size: number,
	): void => {
		counters.clear++;
		dataManager.clear(handle, value, offset, size);
	};

	module.jsepRunOp = (
		descriptorPtr: number,
		_descriptorWords: number,
		opParamsPtr: number,
		_opParamsLen: number,
	): number => {
		counters.runOp++;
		// Re-derive HEAP32 each call — the WASM heap may have grown
		// between EM_ASM frames, detaching prior views.
		const buf = module.HEAPU8.buffer;
		const heap32 = new Int32Array(buf, 0, buf.byteLength >>> 2);
		const desc = readDescriptor(heap32, descriptorPtr);
		const ctx = {
			device,
			dataManager,
			encoderBatcher,
			pipelineCache,
			bindGroupLayoutCache,
		};
		if (desc.op === GGML_OP_MUL_MAT) {
			return dispatchMatmul(ctx, desc);
		}
		if (desc.op === GGML_OP_RMS_NORM) {
			return dispatchRmsNorm(ctx, desc, opParamsPtr, buf);
		}
		if (desc.op === GGML_OP_SET_ROWS) {
			return dispatchSetRows(ctx, desc);
		}
		// Other ops stay NOT_IMPLEMENTED — C++ side falls back to CPU.
		return STATUS_NOT_IMPLEMENTED;
	};

	module.jsepSync = (): void => {
		counters.sync++;
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
