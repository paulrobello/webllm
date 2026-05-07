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
		// Submit any pending compute dispatches before queuing the
		// writeBuffer. WebGPU queue ops are FIFO and `device.queue.writeBuffer`
		// is enqueued immediately, so without this flush a write would slip
		// ahead of dispatches that were recorded earlier and depend on the
		// pre-write contents — a write meant to feed dispatch N+1 would
		// silently overwrite the data that dispatch N still needed to read.
		encoderBatcher.flush();
		// Stage 4.9 diagnostic — capture host_mirror peek for the
		// distinctive H1-inverse signature of i=3 src0 (handle=26, offset=0,
		// size=6144). Records bytes BEFORE writeBuffer fires so we know what
		// host_mirror holds at H1-inverse time.
		const __h1invDiag = (
			globalThis as unknown as {
				__h1invDiag?: {
					captures: Array<{
						callIdx: number;
						handle: number;
						offset: number;
						size: number;
						first16: number[];
						first8F32: number[];
					}>;
					callIdx: number;
					match: { handle: number; offset: number; size: number };
				};
			}
		).__h1invDiag;
		if (
			__h1invDiag &&
			__h1invDiag.captures.length < 8 &&
			handle === __h1invDiag.match.handle &&
			offset === __h1invDiag.match.offset &&
			size === __h1invDiag.match.size
		) {
			const heap8 = new Uint8Array(module.HEAPU8.buffer, hostPtr, 16);
			const heap32 = new Float32Array(module.HEAPU8.buffer, hostPtr, 8);
			__h1invDiag.captures.push({
				callIdx: __h1invDiag.callIdx++,
				handle,
				offset,
				size,
				first16: Array.from(heap8),
				first8F32: Array.from(heap32),
			});
		}
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
		// Submit any pending compute dispatches before issuing the staging
		// copy. WebGPU queue operations are FIFO; without this flush, a
		// readAsync that follows a recorded-but-unsubmitted compute dispatch
		// reads the pre-dispatch buffer state (zeros). The scheduler's
		// JSEP→CPU split copies use this path, so without the flush every
		// activation propagated to a subsequent CPU split is zero — which
		// collapses the entire forward pass to all-zero logits. Diagnosed
		// via Stage 3 self-test (Q4_K kernel correct in isolation; bug
		// reproduced as zero logits for whole-model decode).
		encoderBatcher.flush();
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
		// Same FIFO ordering concern as jsepWrite — clear is implemented
		// via writeBuffer-of-zeros and would jump ahead of pending compute
		// dispatches without a flush.
		encoderBatcher.flush();
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

	// Stage 4.8 warm-up: dispatch one throwaway SET_ROWS divert before any
	// real graph traffic so the SET_ROWS pipeline + bind-group layout +
	// temp-dst command-encoder path are all hot. Without this the first
	// production divert dispatch (Stage 4.7 D2-tight: i=3, K-cache layer 0
	// at dstO=0) silently no-ops — its dst stays at post-allocation zero.
	// Cost: ~50 µs of GPU time amortised over model load. Bypasses the
	// jsepRunOp callback so it doesn't perturb diagnostic wrappers.
	{
		const F32_BYTES = 4;
		const I64_BYTES = 8;
		const F16_BYTES = 2;
		// Match the production K-cache layer 0 SET_ROWS shape so any
		// shape-specific first-call compile/dispatch race is absorbed by
		// the warm-up rather than i=3.
		const NE0 = 256; // inner dim (head_dim * n_kv_heads)
		const NR = 6; // matches prefill-token-count
		const DST_ROWS = 512; // n_ctx
		const srcAllocSize = NE0 * NR * F32_BYTES + NR * I64_BYTES; // 6192
		const dstAllocSize = NE0 * DST_ROWS * F16_BYTES; // 262144
		const srcHandle = dataManager.alloc(srcAllocSize);
		const dstHandle = dataManager.alloc(dstAllocSize);
		const desc = {
			op: GGML_OP_SET_ROWS,
			nSrc: 3,
			dst: {
				bufHandle: dstHandle,
				offset: 0,
				type: 1, // F16
				ne: [NE0, DST_ROWS, 1, 1] as [number, number, number, number],
				nb: [
					F16_BYTES,
					NE0 * F16_BYTES,
					NE0 * DST_ROWS * F16_BYTES,
					NE0 * DST_ROWS * F16_BYTES,
				] as [number, number, number, number],
			},
			srcs: [
				{
					bufHandle: srcHandle,
					offset: 0,
					type: 0, // F32
					ne: [NE0, NR, 1, 1] as [number, number, number, number],
					nb: [
						F32_BYTES,
						NE0 * F32_BYTES,
						NE0 * NR * F32_BYTES,
						NE0 * NR * F32_BYTES,
					] as [number, number, number, number],
				},
				{
					bufHandle: srcHandle,
					offset: NE0 * F32_BYTES,
					type: 27, // I64
					ne: [NR, 1, 1, 1] as [number, number, number, number],
					nb: [I64_BYTES, NR * I64_BYTES, NR * I64_BYTES, NR * I64_BYTES] as [
						number,
						number,
						number,
						number,
					],
				},
				{
					bufHandle: dstHandle, // forces dispatchSetRows to take the divert path
					offset: 0,
					type: 1, // F16
					ne: [NE0, DST_ROWS, 1, 1] as [number, number, number, number],
					nb: [
						F16_BYTES,
						NE0 * F16_BYTES,
						NE0 * DST_ROWS * F16_BYTES,
						NE0 * DST_ROWS * F16_BYTES,
					] as [number, number, number, number],
				},
			],
		};
		const ctx = {
			device,
			dataManager,
			encoderBatcher,
			pipelineCache,
			bindGroupLayoutCache,
		};
		dispatchSetRows(ctx, desc);
		dataManager.free(srcHandle);
		dataManager.free(dstHandle);
	}

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
