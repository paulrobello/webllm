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

	static async init(config: WebLLMConfig): Promise<WebLLM> {
		return new WebLLM(config);
	}

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

	async unloadModel(id: string): Promise<void> {
		await this.modelManager.unregister(id);
	}

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
	get config(): WebLLMConfig {
		return this._config;
	}
	get pipelineCache(): PipelineCache {
		return this._pipelineCache;
	}
	getMemoryPool(): MemoryPool {
		return this.memoryPool;
	}
	getScheduler(): Scheduler {
		return this.scheduler;
	}
	getModelManager(): ModelManager {
		return this.modelManager;
	}

	createCharacter(config: CharacterConfig): Character {
		return this.characterManager.create(config);
	}

	getCharacter(id: string): Character | undefined {
		return this.characterManager.get(id);
	}

	async removeCharacter(id: string): Promise<void> {
		await this.characterManager.remove(id);
	}

	getCharacterManager(): CharacterManager {
		return this.characterManager;
	}

	on(event: string, handler: EventHandler): void {
		if (!this.eventHandlers.has(event))
			this.eventHandlers.set(event, new Set());
		this.eventHandlers.get(event)?.add(handler);
	}

	off(event: string, handler: EventHandler): void {
		this.eventHandlers.get(event)?.delete(handler);
	}

	async shutdown(): Promise<void> {
		this.modelManager.clear();
		this.scheduler.clear();
		this.eventHandlers.clear();
	}
}
