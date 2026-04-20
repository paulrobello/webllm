import type { Character, CharacterConfig } from "../characters/character.js";
import { CharacterManager } from "../characters/character-manager.js";
import type { GenerationConfig } from "../inference/generation.js";
import {
	LightweightModel,
	type LightweightModelConfig,
} from "../inference/lightweight.js";
import type { Tokenizer } from "../inference/tokenizer.js";
import { KVCache } from "../models/kv-cache.js";
import { MemoryPool } from "./memory-pool.js";
import { ModelManager } from "./model-manager.js";
import { PipelineCache } from "./pipeline-cache.js";
import { Scheduler } from "./scheduler.js";
import type {
	EventHandler,
	ModelEntry,
	ModelHandle,
	ModelHyperparams,
	ModelLoadOptions,
	WebLLMConfig,
} from "./types.js";

/**
 * Main entry point for browser-native LLM inference with multi-model scheduling and character system.
 */
export class WebLLM {
	private _config: WebLLMConfig;
	private memoryPool: MemoryPool;
	private scheduler: Scheduler;
	private _pipelineCache: PipelineCache;
	private modelManager: ModelManager;
	private characterManager: CharacterManager;
	private eventHandlers = new Map<string, Set<EventHandler>>();

	private constructor(config: WebLLMConfig) {
		this._config = config;
		this.memoryPool = new MemoryPool(config.memoryBudget);
		this.characterManager = new CharacterManager();
		this.modelManager = new ModelManager(this.memoryPool);
		this.scheduler = new Scheduler({
			frameBudgetMs: config.frameBudgetMs ?? 8,
		});
		this._pipelineCache = new PipelineCache(config.cacheDir ?? "webllm-cache");
	}

	/**
	 * Create and initialize a new WebLLM instance.
	 *
	 * @param config - Engine configuration including device, memory budget, and frame budget.
	 * @returns Initialized WebLLM instance.
	 */
	static async init(config: WebLLMConfig): Promise<WebLLM> {
		return new WebLLM(config);
	}

	/**
	 * Load a model by name with the given options and register it with the model manager.
	 *
	 * @param name - Human-readable model identifier.
	 * @param options - Load options including priority, context length, and GPU layers.
	 * @returns Handle to the loaded model.
	 */
	async loadModel(
		name: string,
		options: ModelLoadOptions,
	): Promise<ModelHandle> {
		const id = `model-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
		const entry: ModelEntry = {
			id,
			name,
			priority: options.priority,
			lightweight: options.lightweight ?? false,
			hyperparams: {} as ModelHyperparams,
			kvCache: new KVCache({
				nLayers: 0,
				nEmbdHeadK: 0,
				nEmbdHeadV: 0,
				nKvHead: 0,
				maxContextLength: options.contextLength ?? 2048,
				dataType: "f32",
			}),
			tokenizer: null as unknown as Tokenizer,
			memoryAllocations: [],
			loaded: false,
			activeSessions: 0,
		};
		this.modelManager.register(entry);
		return entry;
	}

	/**
	 * Unload a model by ID, freeing its memory allocations and KV cache.
	 *
	 * @param id - The model handle ID to unload.
	 */
	async unloadModel(id: string): Promise<void> {
		await this.modelManager.unregister(id);
	}

	/**
	 * Load a lightweight model for auxiliary tasks such as tokenization or embedding.
	 *
	 * @param config - Model configuration excluding the device field (inherited from engine).
	 * @returns Initialized lightweight model instance.
	 */
	async loadLightweightModel(
		config: Omit<LightweightModelConfig, "device">,
	): Promise<LightweightModel> {
		const model = new LightweightModel({
			device: this._config.device,
			...config,
		});
		await model.init();
		return model;
	}

	/**
	 * Run a chat completion.
	 *
	 * Stub — requires WASM forward pass integration (post-Phase 2).
	 */
	async chat(
		_modelId: string,
		_prompt: string,
		_config?: Partial<GenerationConfig>,
	): Promise<string> {
		throw new Error("chat() requires WASM forward pass integration");
	}

	/**
	 * Load a model from a raw GGUF buffer.
	 *
	 * Stub — requires GPU buffer integration (post-Phase 2).
	 */
	static async loadModelFromBuffer(
		_data: ArrayBuffer,
		_options: ModelLoadOptions,
		_config: WebLLMConfig,
	): Promise<ModelHandle> {
		throw new Error("loadModelFromBuffer() requires GPU buffer integration");
	}
	/** Current engine configuration. */
	get config(): WebLLMConfig {
		return this._config;
	}
	/** Pipeline cache for reusing WebGPU compute pipelines across sessions. */
	get pipelineCache(): PipelineCache {
		return this._pipelineCache;
	}
	/** @returns The shared GPU memory pool. */
	getMemoryPool(): MemoryPool {
		return this.memoryPool;
	}
	/** @returns The priority-based task scheduler. */
	getScheduler(): Scheduler {
		return this.scheduler;
	}
	/** @returns The multi-model lifecycle coordinator. */
	getModelManager(): ModelManager {
		return this.modelManager;
	}

	/**
	 * Create a new character with the given configuration.
	 *
	 * @param config - Character personality and behavior configuration.
	 * @returns The created character instance.
	 */
	createCharacter(config: CharacterConfig): Character {
		return this.characterManager.create(config);
	}

	/**
	 * Retrieve a character by ID.
	 *
	 * @param id - Character identifier.
	 * @returns The character, or undefined if not found.
	 */
	getCharacter(id: string): Character | undefined {
		return this.characterManager.get(id);
	}

	/**
	 * Remove a character by ID.
	 *
	 * @param id - Character identifier to remove.
	 */
	async removeCharacter(id: string): Promise<void> {
		await this.characterManager.remove(id);
	}

	/** @returns The character lifecycle manager. */
	getCharacterManager(): CharacterManager {
		return this.characterManager;
	}

	/**
	 * Subscribe to an engine event.
	 *
	 * @param event - Event name.
	 * @param handler - Callback invoked when the event fires.
	 */
	on(event: string, handler: EventHandler): void {
		if (!this.eventHandlers.has(event))
			this.eventHandlers.set(event, new Set());
		this.eventHandlers.get(event)?.add(handler);
	}

	/**
	 * Unsubscribe from an engine event.
	 *
	 * @param event - Event name.
	 * @param handler - Previously registered callback to remove.
	 */
	off(event: string, handler: EventHandler): void {
		this.eventHandlers.get(event)?.delete(handler);
	}

	/**
	 * Tear down all models, scheduler tasks, and event listeners.
	 */
	async shutdown(): Promise<void> {
		this.modelManager.clear();
		this.scheduler.clear();
		this.eventHandlers.clear();
	}
}
