import type { Character, CharacterConfig } from "../characters/character.js";
import { CharacterManager } from "../characters/character-manager.js";
import {
	formatChatDelta,
	formatChatPrompt,
} from "../inference/chat-template.js";
import { type GenerationConfig, Generator } from "../inference/generation.js";
import { GgmlWasm } from "../inference/ggml-wasm.js";
import {
	LightweightModel,
	type LightweightModelConfig,
} from "../inference/lightweight.js";
import {
	type DecodeMode,
	type DecodeResult,
	ModelInference,
} from "../inference/model-inference.js";
import { Sampler } from "../inference/sampler.js";
import { StreamingDecoder, Tokenizer } from "../inference/tokenizer.js";
import { GgufParser } from "../models/gguf-parser.js";
import type { GgufContext } from "../models/gguf-types.js";
import { InferenceSession } from "../models/inference-session.js";
import { KVCache } from "../models/kv-cache.js";
import { ModelLoader } from "../models/model-loader.js";
import type {
	ChatMessage,
	CompletionChunk,
	CompletionConfig,
} from "./chat-types.js";
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

interface ConversationSession {
	session: InferenceSession;
	messageCount: number;
}

export class WebLLM {
	private _config: WebLLMConfig;
	private memoryPool: MemoryPool;
	private scheduler: Scheduler;
	private _pipelineCache: PipelineCache;
	private modelManager: ModelManager;
	private characterManager: CharacterManager;
	private eventHandlers = new Map<string, Set<EventHandler>>();
	private wasmModules = new Map<string, GgmlWasm>();
	private inferenceEngines = new Map<string, ModelInference>();
	private sessions = new Map<string, ConversationSession>();

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
		this.sessions.delete(id);
		const inf = this.inferenceEngines.get(id);
		if (inf) {
			await inf.dispose();
			this.inferenceEngines.delete(id);
		}
		const wasm = this.wasmModules.get(id);
		if (wasm) {
			await wasm.shutdown();
			this.wasmModules.delete(id);
		}
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

	async chat(
		modelId: string,
		prompt: string,
		config?: Partial<GenerationConfig>,
	): Promise<string> {
		const entry = this.modelManager.get(modelId);
		if (!entry) throw new Error(`Model "${modelId}" not found`);
		if (!entry.loaded || !entry.tokenizer)
			throw new Error(`Model "${modelId}" not fully loaded`);

		const inf = this.inferenceEngines.get(modelId);
		if (!inf) throw new Error(`No inference engine for model "${modelId}"`);

		const tokenizer = entry.tokenizer;
		const sampler = new Sampler(config ?? {});
		const session = new InferenceSession(
			{
				maxTokens: config?.maxTokens ?? 512,
				temperature: config?.temperature ?? 1.0,
				topK: config?.topK ?? 0,
				topP: config?.topP ?? 1.0,
				repetitionPenalty: config?.repetitionPenalty ?? 1.0,
				contextOverflowPolicy: "truncate",
			},
			0,
		);

		const tokens = tokenizer.encode(prompt);
		const forwardPass = async (
			ids: number[],
			positions: number[],
		): Promise<Float32Array> => {
			return await inf.forward(new Int32Array(ids), new Int32Array(positions));
		};

		const genConfig: GenerationConfig = {
			prompt,
			maxTokens: config?.maxTokens ?? 512,
			temperature: config?.temperature ?? 1.0,
			topK: config?.topK ?? 0,
			topP: config?.topP ?? 1.0,
			repetitionPenalty: config?.repetitionPenalty ?? 1.0,
		};

		const gen = Generator.generate(
			tokens,
			sampler,
			session,
			tokenizer.eosId ?? 2,
			forwardPass,
			genConfig,
		);
		const genTokens: number[] = [];
		for await (const token of gen) {
			genTokens.push(token);
		}

		return tokenizer.decode(genTokens);
	}

	/**
	 * Streaming chat completion with multi-turn KV cache reuse.
	 *
	 * Yields CompletionChunks with incremental text, followed by a final
	 * done=true chunk carrying generation stats. The KV cache is reused
	 * across calls when the message array grows incrementally.
	 */
	async *chatCompletion(
		modelId: string,
		messages: ChatMessage[],
		config?: CompletionConfig,
	): AsyncGenerator<CompletionChunk, void> {
		const entry = this.modelManager.get(modelId);
		if (!entry) throw new Error(`Model "${modelId}" not found`);
		if (!entry.loaded || !entry.tokenizer)
			throw new Error(`Model "${modelId}" not fully loaded`);

		const inf = this.inferenceEngines.get(modelId);
		if (!inf) throw new Error(`No inference engine for model "${modelId}"`);

		const tokenizer = entry.tokenizer;
		const sampler = new Sampler({
			temperature: config?.temperature,
			topK: config?.topK,
			topP: config?.topP,
			repetitionPenalty: config?.repetitionPenalty,
		});

		const sessionInfo = this.getOrCreateSession(modelId);
		const session = sessionInfo.session;
		const prevMsgCount = sessionInfo.messageCount;
		const tmpl = tokenizer.options.chatTemplate;

		let promptTokens: number[];

		if (
			messages.length > prevMsgCount &&
			prevMsgCount > 0 &&
			session.currentPosition === inf.cachedTokenCount
		) {
			const delta = formatChatDelta(messages, prevMsgCount, tmpl);
			promptTokens = tokenizer.encode(delta);
		} else {
			const prompt = formatChatPrompt(messages, tmpl);
			promptTokens = [tokenizer.bosId, ...tokenizer.encode(prompt)];
			session.reset();
			inf.resetKVCache();
		}
		sessionInfo.messageCount = messages.length;

		const eosId = tokenizer.eosId;
		const maxTokens = config?.maxTokens ?? 512;

		const forwardPass = async (
			ids: number[],
			positions: number[],
		): Promise<Float32Array> => {
			return await inf.forward(new Int32Array(ids), new Int32Array(positions));
		};

		const forwardDecode = async (
			ids: number[],
			positions: number[],
			mode: DecodeMode,
			topK?: number,
		): Promise<DecodeResult> => {
			return await inf.forwardDecode(
				new Int32Array(ids),
				new Int32Array(positions),
				mode,
				topK,
			);
		};

		const genConfig: GenerationConfig = {
			prompt: "",
			maxTokens,
			temperature: config?.temperature ?? 1.0,
			topK: config?.topK ?? 0,
			topP: config?.topP ?? 1.0,
			repetitionPenalty: config?.repetitionPenalty ?? 1.0,
			stopTokens: config?.stopTokenIds,
		};

		const startTime = performance.now();
		const decoder = new StreamingDecoder(tokenizer);
		let tokenCount = 0;
		let timeToFirstTokenMs = 0;

		const gen = Generator.generate(
			promptTokens,
			sampler,
			session,
			eosId,
			forwardPass,
			genConfig,
			config?.signal,
			forwardDecode,
		);

		try {
			for await (const tokenId of gen) {
				if (tokenCount === 0)
					timeToFirstTokenMs = performance.now() - startTime;
				tokenCount++;
				const delta = decoder.push(tokenId);
				yield { text: delta, done: false };
			}
		} catch {
			// On error or abort, yield what we have as the final chunk
			const elapsed = performance.now() - startTime;
			yield {
				text: "",
				done: true,
				stats: {
					tokenCount,
					tokensPerSecond: tokenCount / (elapsed / 1000) || 0,
					timeToFirstTokenMs,
					totalMs: elapsed,
					text: decoder.text,
				},
			};
			return;
		}

		// Final yield with stats
		const elapsed = performance.now() - startTime;
		yield {
			text: "",
			done: true,
			stats: {
				tokenCount,
				tokensPerSecond: tokenCount / (elapsed / 1000) || 0,
				timeToFirstTokenMs,
				totalMs: elapsed,
				text: decoder.text,
			},
		};
	}

	/** Clear conversation history and KV cache for a model. */
	resetConversation(modelId: string): void {
		this.sessions.delete(modelId);
		const inf = this.inferenceEngines.get(modelId);
		if (inf) inf.resetKVCache();
	}

	private getOrCreateSession(modelId: string): ConversationSession {
		let entry = this.sessions.get(modelId);
		if (!entry) {
			entry = {
				session: new InferenceSession(
					{
						maxTokens: 2048,
						temperature: 1,
						topK: 0,
						topP: 1,
						repetitionPenalty: 1,
						contextOverflowPolicy: "stop",
					},
					0,
				),
				messageCount: 0,
			};
			this.sessions.set(modelId, entry);
		}
		return entry;
	}

	static async loadModelFromBuffer(
		data: ArrayBuffer,
		name: string,
		config: WebLLMConfig,
		wasmUrl = "webllm-wasm.js",
	): Promise<{
		handle: ModelHandle;
		engine: WebLLM;
		inference: ModelInference;
	}> {
		const engine = await WebLLM.init(config);

		const parsed = ModelLoader.parseModel(data);
		const ggufCtx = GgufParser.parse(data) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({ wasmUrl });

		const inference = new ModelInference(wasm, parsed.hyperparams);
		inference.loadWeights(ggufCtx, data);
		inference.initKVCache(parsed.kvCacheConfig.maxContextLength);

		const handle = await engine.loadModel(name, { priority: 0 });

		const entry = engine.getModelManager().get(handle.id);
		if (entry) {
			entry.hyperparams = parsed.hyperparams;
			entry.tokenizer = new Tokenizer(parsed.tokenizerConfig);
			entry.kvCache = new KVCache(parsed.kvCacheConfig);
			entry.loaded = true;
		}

		engine.wasmModules.set(handle.id, wasm);
		engine.inferenceEngines.set(handle.id, inference);

		return { handle, engine, inference };
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
		this.sessions.clear();
		for (const [, wasm] of this.wasmModules) {
			await wasm.shutdown();
		}
		this.wasmModules.clear();
		for (const [, inf] of this.inferenceEngines) {
			await inf.dispose();
		}
		this.inferenceEngines.clear();
		this.modelManager.clear();
		this.scheduler.clear();
		this.eventHandlers.clear();
	}
}
