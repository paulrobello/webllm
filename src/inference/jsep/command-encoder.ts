/**
 * Single open command encoder + auto-flush batcher for the JSEP backend.
 *
 * Mirrors ORT-Web's `backend-webgpu.ts:200`-style flush threshold: keep
 * one open compute pass; auto-flush when the pending dispatch count
 * crosses `maxDispatch` (default 16). `flush()` is idempotent.
 */

interface DispatchRecord {
	pipeline: GPUComputePipeline;
	bindGroup: GPUBindGroup;
	dispatchX: number;
	dispatchY: number;
	dispatchZ: number;
}

export interface CommandEncoderBatcherOptions {
	maxDispatch?: number;
}

export class CommandEncoderBatcher {
	private readonly device: GPUDevice;
	private readonly maxDispatch: number;
	private commandEncoder: GPUCommandEncoder | null = null;
	private passEncoder: GPUComputePassEncoder | null = null;
	private pendingDispatchCount = 0;

	constructor(device: GPUDevice, options?: CommandEncoderBatcherOptions) {
		this.device = device;
		this.maxDispatch = options?.maxDispatch ?? 16;
	}

	record(dispatch: DispatchRecord): void {
		if (!this.commandEncoder) {
			this.commandEncoder = this.device.createCommandEncoder();
		}
		if (!this.passEncoder) {
			this.passEncoder = this.commandEncoder.beginComputePass();
		}
		this.passEncoder.setPipeline(dispatch.pipeline);
		this.passEncoder.setBindGroup(0, dispatch.bindGroup);
		this.passEncoder.dispatchWorkgroups(
			dispatch.dispatchX,
			dispatch.dispatchY,
			dispatch.dispatchZ,
		);
		this.pendingDispatchCount++;
		if (this.pendingDispatchCount >= this.maxDispatch) {
			this.flush();
		}
	}

	flush(): void {
		if (!this.commandEncoder) return;
		if (this.passEncoder) {
			this.passEncoder.end();
			this.passEncoder = null;
		}
		const commands = this.commandEncoder.finish();
		this.device.queue.submit([commands]);
		this.commandEncoder = null;
		this.pendingDispatchCount = 0;
	}

	/** Test/debug helper. */
	pendingCount(): number {
		return this.pendingDispatchCount;
	}
}
