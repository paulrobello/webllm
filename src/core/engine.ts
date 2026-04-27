import type { Character, CharacterConfig } from "../characters/character.js";
import { CharacterManager } from "../characters/character-manager.js";
import {
	detectChatTemplate,
	encodeChatPrompt,
	formatChatDelta,
} from "../inference/chat-template.js";
import { EncoderInference } from "../inference/encoder-inference.js";
import {
	type GenerationConfig,
	Generator,
	generateTextStream,
} from "../inference/generation.js";
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
import { SpeculativeGenerator } from "../inference/speculative.js";
import { StreamingDecoder, Tokenizer } from "../inference/tokenizer.js";
import { GgufParser } from "../models/gguf-parser.js";
import type { GgufContext } from "../models/gguf-types.js";
import { InferenceSession } from "../models/inference-session.js";
import { KVCache } from "../models/kv-cache.js";
import type { ParsedModel } from "../models/model-loader.js";
import { ModelLoader } from "../models/model-loader.js";
import type {
	ChatMessage,
	CompletionChunk,
	CompletionConfig,
	StreamChunk,
	StreamConfig,
	StreamInput,
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

const QWEN_THINKING_DEFAULTS = {
	temperature: 0.6,
	topK: 20,
	topP: 0.95,
	repetitionPenalty: 1.05,
} as const;

const QWEN_NON_THINKING_DEFAULTS = {
	temperature: 0.7,
	topK: 20,
	topP: 0.8,
	repetitionPenalty: 1.1,
} as const;

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
	private encoderEngines = new Map<string, EncoderInference>();
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
		const enc = this.encoderEngines.get(id);
		if (enc) {
			await enc.dispose();
			this.encoderEngines.delete(id);
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
	 * Public streaming generation API for prompt or chat inputs.
	 *
	 * Yields incremental decoded chunks followed by a final done=true chunk
	 * carrying completion metadata. Decode-mode selection remains internal.
	 */
	async *generateStream(
		modelId: string,
		input: StreamInput,
		config?: StreamConfig,
	): AsyncGenerator<StreamChunk, void> {
		const entry = this.modelManager.get(modelId);
		if (!entry) throw new Error(`Model "${modelId}" not found`);
		if (!entry.loaded || !entry.tokenizer)
			throw new Error(`Model "${modelId}" not fully loaded`);

		const inf = this.inferenceEngines.get(modelId);
		if (!inf) throw new Error(`No inference engine for model "${modelId}"`);

		const tokenizer = entry.tokenizer;
		const chatTemplate = tokenizer.options.chatTemplate;
		const isQwenChatml =
			Array.isArray(input) &&
			String(entry.hyperparams.architecture).startsWith("qwen") &&
			detectChatTemplate(chatTemplate ?? "") === "chatml";
		const qwenDefaults =
			isQwenChatml && config?.enableThinking === false
				? QWEN_NON_THINKING_DEFAULTS
				: QWEN_THINKING_DEFAULTS;
		const effectiveTemperature = isQwenChatml
			? (config?.temperature ?? qwenDefaults.temperature)
			: (config?.temperature ?? 1.0);
		const effectiveTopK = isQwenChatml
			? (config?.topK ?? qwenDefaults.topK)
			: (config?.topK ?? 0);
		const effectiveTopP = isQwenChatml
			? (config?.topP ?? qwenDefaults.topP)
			: (config?.topP ?? 1.0);
		const effectiveRepetitionPenalty = isQwenChatml
			? (config?.repetitionPenalty ?? qwenDefaults.repetitionPenalty)
			: (config?.repetitionPenalty ?? 1.0);
		const sampler = new Sampler({
			temperature: effectiveTemperature,
			topK: effectiveTopK,
			topP: effectiveTopP,
			repetitionPenalty: effectiveRepetitionPenalty,
			seed: config?.seed,
		});
		const genConfig: GenerationConfig = {
			prompt: typeof input === "string" ? input : "",
			maxTokens: config?.maxTokens ?? 512,
			temperature: effectiveTemperature,
			topK: effectiveTopK,
			topP: effectiveTopP,
			repetitionPenalty: effectiveRepetitionPenalty,
			stopTokens: config?.stopTokenIds,
		};

		// Speculative-decode routing. Runs before Qwen mask injection so the
		// steering-fields probe sees a clean genConfig and can reject any
		// caller-set steering, and so we can refuse Qwen3 thinking-mode
		// requests explicitly (the engine would otherwise inject masks
		// downstream that the spec driver doesn't support in v1).
		if (config?.drafter !== undefined) {
			const drafterId = config.drafter;
			const drafterInf = this.inferenceEngines.get(drafterId);
			if (!drafterInf) {
				throw new Error(
					`Drafter "${drafterId}" is not loaded; load it first via loadModel.`,
				);
			}
			const drafterEntry = this.modelManager.get(drafterId);
			if (!drafterEntry?.tokenizer) {
				throw new Error(`Drafter "${drafterId}" tokenizer is not initialized.`);
			}
			if (drafterEntry.tokenizer.vocabSize !== tokenizer.vocabSize) {
				throw new Error(
					`Drafter vocab (${drafterEntry.tokenizer.vocabSize}) != target vocab (${tokenizer.vocabSize}); cross-tokenizer drafters are not supported in v1.`,
				);
			}
			if (
				drafterEntry.tokenizer.bosId !== tokenizer.bosId ||
				drafterEntry.tokenizer.eosId !== tokenizer.eosId
			) {
				throw new Error("Drafter EOS/BOS mismatch with target.");
			}
			// `isQwenChatml` already requires Array.isArray(input); a future
			// refactor that loosens that condition would silently bypass this
			// gate, so re-evaluate this guard if that happens.
			if (isQwenChatml && config?.enableThinking !== false) {
				throw new Error(
					"Speculative decode does not support Qwen3 thinking-mode steering; pass enableThinking: false or use the non-drafted path.",
				);
			}
			// Belt-and-suspenders: the only writer of these fields lives below
			// in the Qwen mask-injection branch, so this probe is currently
			// unreachable. It's cheap and turns a future regression (e.g.
			// steering writes added above this block) into a loud throw.
			const steeringFields: (keyof GenerationConfig)[] = [
				"maskedTokensWhileThinking",
				"maskedTokensAfterThinkingUntilAnswer",
				"maskedTokensAfterAnswerStarts",
				"requireVisibleAnswerAfterThinking",
				"requireVisibleAnswerBeforeStop",
				"requireLeadingWhitespaceAfterThinking",
				"suppressWhitespaceOnlyAfterThinking",
				"suppressWhitespaceOnlyUntilAnswer",
				"enforceSingleThinkBlock",
				"forbiddenReentryTokens",
			];
			const probe = genConfig as unknown as Record<string, unknown>;
			for (const field of steeringFields) {
				const v = probe[field];
				const set = Array.isArray(v)
					? v.length > 0
					: v !== undefined && v !== false;
				if (set) {
					throw new Error(
						`Speculative decode requires no steering; got ${String(field)} configured. Disable thinking mode or use the non-drafted path.`,
					);
				}
			}
			const draftLength = config.draftLength ?? 4;
			if (draftLength < 2 || draftLength > 32) {
				throw new Error(`draftLength must be in [2, 32], got ${draftLength}.`);
			}

			const promptTokensSpec = Array.isArray(input)
				? this.prepareChatPrompt(modelId, input, tokenizer, inf, config)
				: tokenizer.encode(input);

			const result = SpeculativeGenerator.generate({
				promptTokenIds: promptTokensSpec,
				target: inf,
				drafter: drafterInf,
				tokenizer,
				sampler,
				config: genConfig,
				eosTokenId: tokenizer.eosId,
				draftLength,
				signal: config?.signal,
			});
			const decoder = new StreamingDecoder(tokenizer);
			const startTime = performance.now();
			let res = await result.next();
			while (!res.done) {
				const tokenId = res.value;
				const text = decoder.push(tokenId);
				yield { text, tokenId, done: false };
				res = await result.next();
			}
			const final = res.value;
			yield {
				text: "",
				done: true,
				stats: {
					tokenCount: final.tokenCount,
					tokensPerSecond: final.tokensPerSecond,
					timeToFirstTokenMs: final.timeToFirstTokenMs,
					totalMs: performance.now() - startTime,
					text: decoder.text,
					finishReason: final.finishReason,
				},
			};
			return;
		}

		if (Array.isArray(input)) {
			if (isQwenChatml) {
				const imStartId = tokenizer.getId("<|im_start|>");
				const imEndId = tokenizer.getId("<|im_end|>");
				const endoftextId = tokenizer.getId("<|endoftext|>");
				const thinkOpenId = tokenizer.getId("<think>");
				const thinkCloseId = tokenizer.getId("</think>");
				const toolCallOpenId = tokenizer.getId("<tool_call>");
				const toolCallCloseId = tokenizer.getId("</tool_call>");
				const toolResponseOpenId = tokenizer.getId("<tool_response>");
				const toolResponseCloseId = tokenizer.getId("</tool_response>");
				if (imStartId !== undefined) {
					genConfig.forbiddenReentryTokens = [imStartId];
				}
				if (endoftextId !== undefined) {
					genConfig.stopTokens = [...(config?.stopTokenIds ?? []), endoftextId];
				}
				if (config?.enableThinking === false) {
					genConfig.tokenizer = tokenizer;
				}
				if (thinkOpenId !== undefined && thinkCloseId !== undefined) {
					genConfig.tokenizer = tokenizer;
					genConfig.thinkingOpenTokenId = thinkOpenId;
					genConfig.thinkingCloseTokenId = thinkCloseId;
					genConfig.enforceSingleThinkBlock = true;
					genConfig.maskedTokensWhileThinking = [
						thinkOpenId,
						imStartId,
						imEndId,
						endoftextId,
					].filter((id): id is number => id !== undefined);
					genConfig.maskedTokensAfterThinkingUntilAnswer = [
						thinkOpenId,
						imStartId,
						imEndId,
						endoftextId,
						toolCallOpenId,
						toolCallCloseId,
						toolResponseOpenId,
						toolResponseCloseId,
					].filter((id): id is number => id !== undefined);
					// During the visible answer, the model must be allowed to
					// terminate via `<|im_end|>` (the chat EOS) and
					// `<|endoftext|>` (a secondary stop). Mask only the
					// scaffolding controls so it can't relapse into a new
					// `<think>`, `<|im_start|>`, or tool-call envelope.
					genConfig.maskedTokensAfterAnswerStarts = [
						thinkOpenId,
						imStartId,
						toolCallOpenId,
						toolCallCloseId,
						toolResponseOpenId,
						toolResponseCloseId,
					].filter((id): id is number => id !== undefined);
					genConfig.requireVisibleAnswerAfterThinking = true;
					genConfig.suppressWhitespaceOnlyAfterThinking = true;
					genConfig.requireLeadingWhitespaceAfterThinking = true;
				}
			}
		}

		const promptTokens = Array.isArray(input)
			? this.prepareChatPrompt(modelId, input, tokenizer, inf, config)
			: tokenizer.encode(input);
		const session = Array.isArray(input)
			? this.getOrCreateSession(modelId).session
			: new InferenceSession(
					{
						maxTokens: genConfig.maxTokens,
						temperature: genConfig.temperature,
						topK: genConfig.topK,
						topP: genConfig.topP,
						repetitionPenalty: genConfig.repetitionPenalty,
						contextOverflowPolicy: "truncate",
					},
					0,
				);

		const forwardPass = async (
			ids: number[],
			positions: number[],
		): Promise<Float32Array> => {
			return await inf.forward(new Int32Array(ids), new Int32Array(positions));
		};

		const forwardDecode =
			typeof inf.forwardDecode === "function"
				? async (
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
					}
				: undefined;

		yield* generateTextStream({
			promptTokenIds: promptTokens,
			sampler,
			session,
			eosTokenId: tokenizer.eosId,
			tokenizer,
			forwardPass,
			config: genConfig,
			signal: config?.signal,
			forwardDecode,
		});
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
		yield* this.generateStream(modelId, messages, config);
	}

	/**
	 * Compute an L2-normalized sentence embedding for the given text using
	 * a registered bidirectional-encoder model (e.g. Arctic-Embed). The
	 * model must be loaded with a bert-architecture GGUF; non-encoder
	 * models throw with a descriptive error.
	 */
	async embed(modelId: string, text: string): Promise<Float32Array> {
		const entry = this.modelManager.get(modelId);
		if (!entry) throw new Error(`Model "${modelId}" not found`);
		if (!entry.loaded || !entry.tokenizer) {
			throw new Error(`Model "${modelId}" not fully loaded`);
		}
		const enc = this.encoderEngines.get(modelId);
		if (!enc) {
			throw new Error(
				`embed() requires a bidirectional encoder model; "${modelId}" is architecture "${entry.hyperparams.architecture}"`,
			);
		}
		const ids = entry.tokenizer.encode(text);
		return enc.embed(new Int32Array(ids));
	}

	/** Clear conversation history and KV cache for a model. */
	resetConversation(modelId: string): void {
		this.sessions.delete(modelId);
		const inf = this.inferenceEngines.get(modelId);
		if (inf) inf.resetKVCache();
	}

	private prepareChatPrompt(
		modelId: string,
		messages: ChatMessage[],
		tokenizer: Tokenizer,
		inf: ModelInference,
		config?: StreamConfig,
	): number[] {
		const sessionInfo = this.getOrCreateSession(modelId);
		const session = sessionInfo.session;
		const prevMsgCount = sessionInfo.messageCount;
		const tmpl = tokenizer.options.chatTemplate;
		const promptMessages = messages;
		const templateOptions = {
			enableThinking: config?.enableThinking,
			tools: config?.tools,
		};

		let promptTokens: number[];
		if (
			promptMessages.length > prevMsgCount &&
			prevMsgCount > 0 &&
			session.currentPosition === inf.cachedTokenCount
		) {
			const delta = formatChatDelta(
				promptMessages,
				prevMsgCount,
				tmpl,
				templateOptions,
			);
			promptTokens = tokenizer.encode(delta);
		} else {
			promptTokens = encodeChatPrompt(
				promptMessages,
				tokenizer,
				templateOptions,
			);
			session.reset();
			inf.resetKVCache();
		}
		sessionInfo.messageCount = promptMessages.length;
		return promptTokens;
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

	/**
	 * Promote a manually-assembled inference pipeline into this engine. Useful
	 * for consumers (the smoke-test page, notebooks, custom loaders) that
	 * already built `wasm` + `ModelInference` + `ParsedModel` by hand and want
	 * to drive them through the library primitives (`Character.chat`,
	 * `runTask`, `runTasks`) without re-loading the model.
	 *
	 * After this returns, the returned handle is ready for `engine.chatCompletion`
	 * and everything built on top of it.
	 */
	async adoptPreloadedModel(
		name: string,
		pipeline: {
			wasm: GgmlWasm;
			inference: ModelInference | EncoderInference;
			parsed: ParsedModel;
		},
	): Promise<ModelHandle> {
		const isEncoder = pipeline.inference instanceof EncoderInference;
		// Causal-LM pipelines may have been used before (e.g. the smoke
		// page's [7/8] one-shot writes to the KV cache). Reset the cache
		// here so the engine's session tracker doesn't collide with
		// existing state. Encoder pipelines have no KV cache to reset.
		if (!isEncoder) {
			(pipeline.inference as ModelInference).resetKVCache();
		}

		const handle = await this.loadModel(name, { priority: 0 });
		const entry = this.modelManager.get(handle.id);
		if (!entry) {
			throw new Error(
				`adoptPreloadedModel: model manager entry missing for ${handle.id}`,
			);
		}
		entry.hyperparams = pipeline.parsed.hyperparams;
		entry.tokenizer = new Tokenizer(pipeline.parsed.tokenizerConfig);
		entry.kvCache = new KVCache(pipeline.parsed.kvCacheConfig);
		entry.loaded = true;
		this.wasmModules.set(handle.id, pipeline.wasm);
		if (isEncoder) {
			this.encoderEngines.set(
				handle.id,
				pipeline.inference as EncoderInference,
			);
		} else {
			this.inferenceEngines.set(
				handle.id,
				pipeline.inference as ModelInference,
			);
		}
		return handle;
	}

	static async loadModelFromBuffer(
		data: ArrayBuffer | Uint8Array,
		name: string,
		config: WebLLMConfig,
		wasmUrl = "webllm-wasm.js",
	): Promise<{
		handle: ModelHandle;
		engine: WebLLM;
		inference: ModelInference | EncoderInference;
	}> {
		const engine = await WebLLM.init(config);

		const view = data instanceof Uint8Array ? data : new Uint8Array(data);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const wasm = new GgmlWasm();
		await wasm.init({ wasmUrl });

		const isEncoder = parsed.hyperparams.architecture === "bert";
		let inference: ModelInference | EncoderInference;
		if (isEncoder) {
			const enc = new EncoderInference(wasm, parsed.hyperparams);
			enc.loadWeights(ggufCtx, view);
			inference = enc;
		} else {
			const inf = new ModelInference(wasm, parsed.hyperparams);
			inf.loadWeights(ggufCtx, view);
			inf.initKVCache(parsed.kvCacheConfig.maxContextLength);
			inference = inf;
		}

		const handle = await engine.loadModel(name, { priority: 0 });

		const entry = engine.getModelManager().get(handle.id);
		if (entry) {
			entry.hyperparams = parsed.hyperparams;
			entry.tokenizer = new Tokenizer(parsed.tokenizerConfig);
			entry.kvCache = new KVCache(parsed.kvCacheConfig);
			entry.loaded = true;
		}

		engine.wasmModules.set(handle.id, wasm);
		if (inference instanceof EncoderInference) {
			engine.encoderEngines.set(handle.id, inference);
		} else {
			engine.inferenceEngines.set(handle.id, inference);
		}

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
		// Inject ourselves as the engine unless the caller passed one
		// explicitly (e.g. a mock for testing).
		return this.characterManager.create({ engine: this, ...config });
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
		for (const [, enc] of this.encoderEngines) {
			await enc.dispose();
		}
		this.encoderEngines.clear();
		this.modelManager.clear();
		this.scheduler.clear();
		this.eventHandlers.clear();
	}
}
