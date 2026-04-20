/** Configuration for initializing the GGML WASM bridge. */
export interface GgmlWasmConfig {
	/** URL to the Emscripten-generated JS module (e.g. "webllm-wasm.js"). */
	wasmUrl: string;
}

/**
 * WebAssembly bridge for GGML inference via the ggml-webgpu backend.
 *
 * Loads the Emscripten MODULARIZE output and delegates all GPU compute
 * to the WASM module — buffer management is handled internally by the
 * ggml-webgpu backend, not by this layer.
 */
export class GgmlWasm {
	private mod: any = null;
	private initialized = false;

	/**
	 * Load the Emscripten module and initialize the WebGPU backend.
	 *
	 * @param config - WASM module URL.
	 * @throws If the module fails to load or WebGPU backend init fails.
	 */
	async init(config: GgmlWasmConfig): Promise<void> {
		const factory = (await import(config.wasmUrl)).default;
		this.mod = await factory();
		const result: number = this.mod._webgpu_init();
		if (result !== 0) {
			throw new Error(`WASM WebGPU init failed with code ${result}`);
		}
		this.initialized = true;
	}

	/**
	 * Shut down the WebGPU backend and release the WASM module.
	 *
	 * Safe to call multiple times; no-ops if already shut down.
	 */
	async shutdown(): Promise<void> {
		if (!this.initialized) return;
		this.mod._webgpu_shutdown();
		this.initialized = false;
		this.mod = null;
	}
}
