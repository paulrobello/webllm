import type { Character, CharacterConfig } from "../characters/character.js";
import { CharacterManager } from "../characters/character-manager.js";
import { CausalLMEmbedder } from "../inference/causal-embedder-inference.js";
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
	type InternalGenerationOptions,
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
import { Tokenizer } from "../inference/tokenizer.js";
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
import {
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
} from "./errors.js";
import { MemoryPool } from "./memory-pool.js";
import { ModelManager } from "./model-manager.js";
import { PipelineCache } from "./pipeline-cache.js";
import {
	QWEN_NON_THINKING_DEFAULTS,
	QWEN_THINKING_DEFAULTS,
} from "./sampling-profiles.js";
import { Scheduler } from "./scheduler.js";
import {
	type EventHandler,
	isCausalEmbedderArchitecture,
	isEncoderArchitecture,
	type ModelEntry,
	type ModelHandle,
	type ModelHyperparams,
	type ModelLoadOptions,
	type WebLLMConfig,
} from "./types.js";

interface ConversationSession {
	session: InferenceSession;
	messageCount: number;
}

/**
 * Heap-margin threshold for routing between the wasm32 and wasm64 binaries.
 *
 * 3.5 GiB = 10% under the wasm32 4 GiB heap cap. Models at or below this
 * size fit comfortably in the wasm32 heap (with headroom for KV cache and
 * scratch buffers); larger models require the wasm64 binary's 16 GiB heap.
 */
const WASM32_HEAP_MARGIN = 3.5 * 1024 * 1024 * 1024;

/**
 * Pick the WASM binary based on model file size.
 *
 * Models ≤ 3.5 GiB (10% under the wasm32 4 GiB heap cap) route through
 * `webllm-wasm.js` (wasm32 — smaller bundle, no BigInt-allocation
 * overhead in hot paths). Larger models route through
 * `webllm-wasm-mem64.js` (wasm64 — 16 GiB heap cap, the only path
 * available for 7B+ Q4_K_S, 13B Q4_K_S, and 30B IQ3_M targets).
 *
 * Pass an explicit `override` to bypass the default (e.g., to force
 * wasm64 for testing or to point at a custom-served binary).
 */
export function pickWasmUrl(
	modelByteLength: number,
	override?: string,
): string {
	if (override) return override;
	return modelByteLength > WASM32_HEAP_MARGIN
		? "webllm-wasm-mem64.js"
		: "webllm-wasm.js";
}

export class WebLLM {
	private _config: WebLLMConfig;
	private _memoryPool: MemoryPool;
	private _scheduler: Scheduler;
	private _pipelineCache: PipelineCache;
	private _modelManager: ModelManager;
	private characterManager: CharacterManager;
	private eventHandlers = new Map<string, Set<EventHandler>>();
	private wasmModules = new Map<string, GgmlWasm>();
	private inferenceEngines = new Map<string, ModelInference>();
	private encoderEngines = new Map<string, EncoderInference>();
	private causalEmbedderEngines = new Map<string, CausalLMEmbedder>();
	private sessions = new Map<string, ConversationSession>();

	private constructor(config: WebLLMConfig) {
		this._config = config;
		this._memoryPool = new MemoryPool(config.memoryBudget);
		this.characterManager = new CharacterManager();
		this._modelManager = new ModelManager(this._memoryPool);
		this._scheduler = new Scheduler({
			frameBudgetMs: config.frameBudgetMs ?? 8,
		});
		this._pipelineCache = new PipelineCache(config.cacheDir ?? "webllm-cache");
	}

	static async init(config: WebLLMConfig): Promise<WebLLM> {
		return new WebLLM(config);
	}

	/**
	 * Register an empty model entry and mint a handle. Internal helper used
	 * by `loadModelFromBuffer` and `adoptPreloadedModel` before the inference
	 * pipeline + tokenizer are wired up. Not a consumer-facing loader — call
	 * `loadModelFromBuffer` (instance or static) to actually load weights.
	 */
	private registerModelHandle(
		name: string,
		options: ModelLoadOptions,
	): ModelHandle {
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
		this._modelManager.register(entry);
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
		const cembed = this.causalEmbedderEngines.get(id);
		if (cembed) {
			await cembed.dispose();
			this.causalEmbedderEngines.delete(id);
		}
		const wasm = this.wasmModules.get(id);
		if (wasm) {
			await wasm.shutdown();
			this.wasmModules.delete(id);
		}
		await this._modelManager.unregister(id);
	}

	async loadLightweightModel(
		config: LightweightModelConfig,
	): Promise<LightweightModel> {
		const model = new LightweightModel(config);
		await model.init();
		return model;
	}

	async chat(
		modelId: string,
		prompt: string,
		config?: Partial<GenerationConfig>,
	): Promise<string> {
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer)
			throw new ModelNotLoadedError(modelId);

		const inf = this.inferenceEngines.get(modelId);
		if (!inf) throw new InferenceEngineMissingError(modelId);

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

		const genConfig: InternalGenerationOptions = {
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
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer)
			throw new ModelNotLoadedError(modelId);

		const inf = this.inferenceEngines.get(modelId);
		if (!inf) throw new InferenceEngineMissingError(modelId);

		const tokenizer = entry.tokenizer;
		const chatTemplate = tokenizer.options.chatTemplate;
		const samplingMode = config?.sampling ?? "auto";
		const isQwenChatml =
			Array.isArray(input) &&
			String(entry.hyperparams.architecture).startsWith("qwen") &&
			detectChatTemplate(chatTemplate ?? "") === "chatml";
		const applyAutoQwen = samplingMode === "auto" && isQwenChatml;
		const forcedProfile =
			samplingMode === "qwen-thinking"
				? QWEN_THINKING_DEFAULTS
				: samplingMode === "qwen-default"
					? QWEN_NON_THINKING_DEFAULTS
					: null;
		const autoProfile = applyAutoQwen
			? config?.enableThinking === false
				? QWEN_NON_THINKING_DEFAULTS
				: QWEN_THINKING_DEFAULTS
			: null;
		const activeProfile = forcedProfile ?? autoProfile;
		// Consumer-provided values override profile defaults; profile defaults
		// override engine fallbacks. samplingMode === "raw" produces a null
		// activeProfile, falling through to the engine fallbacks directly.
		const effectiveTemperature =
			config?.temperature ?? activeProfile?.temperature ?? 1.0;
		const effectiveTopK = config?.topK ?? activeProfile?.topK ?? 0;
		const effectiveTopP = config?.topP ?? activeProfile?.topP ?? 1.0;
		const effectiveRepetitionPenalty =
			config?.repetitionPenalty ?? activeProfile?.repetitionPenalty ?? 1.0;
		const sampler = new Sampler({
			temperature: effectiveTemperature,
			topK: effectiveTopK,
			topP: effectiveTopP,
			repetitionPenalty: effectiveRepetitionPenalty,
			seed: config?.seed,
		});
		const genConfig: InternalGenerationOptions = {
			maxTokens: config?.maxTokens ?? 512,
			temperature: effectiveTemperature,
			topK: effectiveTopK,
			topP: effectiveTopP,
			repetitionPenalty: effectiveRepetitionPenalty,
			stopTokens: config?.stopTokenIds ? [...config.stopTokenIds] : undefined,
			signal: config?.signal,
		};

		// Speculative-decode is reserved in v1: measurement on 2026-04-26
		// against qwen3-8b-iq3m / qwen3-0.6b-q4f16 at K=4 produced 3.0 tok/s
		// vs 15.3 tok/s baseline (0.20×) — verify-readback overhead and
		// per-step drafter forwards dwarf the savings. Driver, sampler
		// helpers, forwardVerify, truncateKVCache, and tests remain in
		// tree (`src/inference/speculative.ts`, etc.) so a v2 lever
		// (dynamic K, multi-tokenizer drafters, or GPU-resident verify
		// reduction) can revisit without re-discovering the design. See
		// TODO.md §19 and docs/superpowers/specs/2026-04-26-speculative-
		// decoding-design.md.
		if (config?.drafter !== undefined) {
			throw new SpeculativeDecodingReservedError();
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
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelId);
		}
		const ids = entry.tokenizer.encode(text);
		const enc = this.encoderEngines.get(modelId);
		if (enc) return enc.embed(new Int32Array(ids));
		const cembed = this.causalEmbedderEngines.get(modelId);
		if (cembed) return cembed.embed(new Int32Array(ids));
		throw new EncoderRequiredError(
			modelId,
			String(entry.hyperparams.architecture),
		);
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
			inference: ModelInference | EncoderInference | CausalLMEmbedder;
			parsed: ParsedModel;
		},
	): Promise<ModelHandle> {
		const isEncoder = pipeline.inference instanceof EncoderInference;
		const isCausalEmbedder = pipeline.inference instanceof CausalLMEmbedder;
		// Causal-LM pipelines may have been used before (e.g. the smoke
		// page's [7/8] one-shot writes to the KV cache). Reset the cache
		// here so the engine's session tracker doesn't collide with
		// existing state. Encoder and causal-embedder pipelines have no
		// KV cache to reset.
		if (!isEncoder && !isCausalEmbedder) {
			(pipeline.inference as ModelInference).resetKVCache();
		}

		const handle = this.registerModelHandle(name, { priority: 0 });
		const entry = this._modelManager.get(handle.id);
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
		} else if (isCausalEmbedder) {
			this.causalEmbedderEngines.set(
				handle.id,
				pipeline.inference as CausalLMEmbedder,
			);
		} else {
			this.inferenceEngines.set(
				handle.id,
				pipeline.inference as ModelInference,
			);
		}
		return handle;
	}

	/**
	 * Load a model directly from an in-memory GGUF buffer into this engine.
	 *
	 * Parses the GGUF, instantiates the WASM backend, uploads weights, wires
	 * up the inference (or encoder) pipeline, and returns a handle ready for
	 * `chatCompletion` / `embed` / `createCharacter`. Call this multiple times
	 * to host several models on the same engine.
	 *
	 * **WASM binary selection.** When `wasmUrl` is omitted, the binary
	 * is chosen by model file size via {@link pickWasmUrl}: models
	 * ≤ 3.5 GiB use `webllm-wasm.js` (wasm32, smaller bundle, no BigInt
	 * dispatch overhead); larger models use `webllm-wasm-mem64.js`
	 * (wasm64, 16 GiB heap — required for 7B+ Q4_K_S, 13B Q4_K_S, and
	 * 30B IQ3_M). Pass an explicit `wasmUrl` to override the default
	 * (e.g., force wasm64 for testing, or point at a custom-served
	 * binary).
	 */
	async loadModelFromBuffer(
		data: ArrayBuffer | Uint8Array,
		name: string,
		wasmUrl?: string,
	): Promise<{
		handle: ModelHandle;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
	}> {
		const view = data instanceof Uint8Array ? data : new Uint8Array(data);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const resolvedWasmUrl = pickWasmUrl(view.byteLength, wasmUrl);
		const wasm = new GgmlWasm();
		await wasm.init({ wasmUrl: resolvedWasmUrl });

		const arch = parsed.hyperparams.architecture;
		const isEncoder = isEncoderArchitecture(arch);
		const isCausalEmbedder = isCausalEmbedderArchitecture(arch);
		let inference: ModelInference | EncoderInference | CausalLMEmbedder;
		if (isEncoder) {
			const enc = new EncoderInference(wasm, parsed.hyperparams);
			enc.loadWeights(ggufCtx, view);
			inference = enc;
		} else if (isCausalEmbedder) {
			const cembed = new CausalLMEmbedder(wasm, parsed.hyperparams);
			cembed.loadWeights(ggufCtx, view);
			inference = cembed;
		} else {
			const inf = new ModelInference(wasm, parsed.hyperparams);
			inf.loadWeights(ggufCtx, view);
			inf.initKVCache(parsed.kvCacheConfig.maxContextLength);
			inference = inf;
		}

		const handle = this.registerModelHandle(name, { priority: 0 });
		const entry = this._modelManager.get(handle.id);
		if (entry) {
			entry.hyperparams = parsed.hyperparams;
			entry.tokenizer = new Tokenizer(parsed.tokenizerConfig);
			entry.kvCache = new KVCache(parsed.kvCacheConfig);
			entry.loaded = true;
		}

		this.wasmModules.set(handle.id, wasm);
		if (inference instanceof EncoderInference) {
			this.encoderEngines.set(handle.id, inference);
		} else if (inference instanceof CausalLMEmbedder) {
			this.causalEmbedderEngines.set(handle.id, inference);
		} else {
			this.inferenceEngines.set(handle.id, inference);
		}

		return { handle, inference };
	}

	/**
	 * Convenience factory: initialize a fresh engine and load a single model
	 * from an in-memory GGUF buffer in one call. For multi-model setups, use
	 * {@link WebLLM.init} + {@link WebLLM.prototype.loadModelFromBuffer}
	 * directly.
	 */
	static async loadModelFromBuffer(
		data: ArrayBuffer | Uint8Array,
		name: string,
		config: WebLLMConfig,
		wasmUrl?: string,
	): Promise<{
		handle: ModelHandle;
		engine: WebLLM;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
	}> {
		const engine = await WebLLM.init(config);
		const { handle, inference } = await engine.loadModelFromBuffer(
			data,
			name,
			wasmUrl,
		);
		return { handle, engine, inference };
	}

	get config(): WebLLMConfig {
		return this._config;
	}
	get pipelineCache(): PipelineCache {
		return this._pipelineCache;
	}
	get memoryPool(): MemoryPool {
		return this._memoryPool;
	}
	get scheduler(): Scheduler {
		return this._scheduler;
	}
	get modelManager(): ModelManager {
		return this._modelManager;
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
		for (const [, cembed] of this.causalEmbedderEngines) {
			await cembed.dispose();
		}
		this.causalEmbedderEngines.clear();
		this._modelManager.clear();
		this._scheduler.clear();
		this.eventHandlers.clear();
	}
}
