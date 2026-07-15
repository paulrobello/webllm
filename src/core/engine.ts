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
	type ConversationHandle,
	type ConversationOptions,
	ConversationPool,
} from "./conversation-pool.js";
import {
	ConversationBusyError,
	ConversationContextOverflowError,
	ConversationNotPopulatedError,
	EncoderRequiredError,
	InferenceEngineMissingError,
	ModelNotFoundError,
	ModelNotLoadedError,
	SpeculativeDecodingReservedError,
} from "./errors.js";
import { MemoryPool } from "./memory-pool.js";
import { ModelManager } from "./model-manager.js";
import {
	computeTokenizerHash,
	decodePersistedConversation,
	encodePersistedConversation,
	KV_PERSISTENCE_SCHEMA_VERSION,
	type ModelFingerprint,
} from "./persistence.js";
import { PipelineCache } from "./pipeline-cache.js";
import { resolveSamplingParams } from "./sampling-profiles.js";
import { Scheduler } from "./scheduler.js";
import {
	type EventHandler,
	isCausalEmbedderArchitecture,
	isEncoderArchitecture,
	type LoadedModelMetadata,
	type ModelEntry,
	type ModelHandle,
	type ModelHyperparams,
	type ModelLoadOptions,
	type WebLLMConfig,
} from "./types.js";

interface ConversationSession {
	session: InferenceSession;
	messageCount: number;
	/**
	 * Snapshot of the leading messages from the previous call's prompt.
	 * The delta-encoding fast-path in `prepareChatPrompt` is only safe when
	 * the new prompt's leading `messageCount` messages match this snapshot
	 * — otherwise the cached KV is from a different conversation and a
	 * delta append would silently produce a corrupt response. Stored as a
	 * shallow array of `{ role, content }` so caller-side mutation of the
	 * passed-in messages can't poison the snapshot.
	 */
	cachedMessages: ChatMessage[];
}

/**
 * Per-model state aggregate (ARC-004). Consolidates the six prior parallel
 * `Map<string, …>` fields (`wasmModules`, `inferenceEngines`,
 * `encoderEngines`, `causalEmbedderEngines`, `sessions`, `modelChatChains`)
 * into one row per loaded model id. Load/unload consistency is now enforced
 * by construction — a single `delete` drops everything.
 *
 * The three engine-kind fields (`inference` / `encoder` / `causalEmbedder`)
 * remain type-partitioned (ARC-006); only the containers are unified, not
 * the types. All fields are optional because load is staged: a fresh record
 * may be created with just `wasm`, gain an engine a moment later, and only
 * acquire a `session` / `chatChain` on first chat use.
 */
interface ModelRecord {
	wasm?: GgmlWasm;
	inference?: ModelInference;
	encoder?: EncoderInference;
	causalEmbedder?: CausalLMEmbedder;
	session?: ConversationSession;
	chatChain?: Promise<void>;
}

/**
 * True when the cached prompt's leading messages still appear at the start
 * of the new prompt. The delta-encoding fast-path in `prepareChatPrompt`
 * relies on this — if cached and new diverge on the leading slice, the
 * cached KV is from a different conversation and reusing it would silently
 * produce a corrupt response. Bug surfaced 2026-05-02 by the interleaved
 * NPC probe (`eval/reports/prefix-cache-interleaved-2026-05-02/SUMMARY.md`).
 */
function leadingMessagesMatch(
	cached: readonly ChatMessage[],
	next: readonly ChatMessage[],
): boolean {
	if (cached.length === 0 || cached.length > next.length) return false;
	for (let i = 0; i < cached.length; i++) {
		if (cached[i].role !== next[i].role) return false;
		if (cached[i].content !== next[i].content) return false;
	}
	return true;
}

/**
 * Append a chat-template-specific turn-end token to `genConfig.stopTokens`
 * when the tokenizer recognises it. Used to fix architectures whose GGUF
 * declares an EOS that the model rarely emits in chat (e.g. Phi-3.5
 * declares `<|endoftext|>` but generates `<|end|>` to terminate turns).
 */
function addChatStopToken(
	genConfig: InternalGenerationOptions,
	tokenizer: Tokenizer,
	config: CompletionConfig | undefined,
	tokenText: string,
): void {
	const id = tokenizer.getId(tokenText);
	if (id === undefined) return;
	const existing = genConfig.stopTokens ?? config?.stopTokenIds ?? [];
	const next = [...existing];
	if (!next.includes(id)) next.push(id);
	genConfig.stopTokens = next;
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
 * Default total byte budget for {@link MemoryPool} when `WebLLMConfig.memoryBudget`
 * is omitted.
 *
 * Per CLAUDE.md's hardware baseline doctrine (16 GB unified-memory floor /
 * 32 GB recommended / 128 GB dev), WebGPU sees ~10–11 GiB on the floor tier;
 * Three.js coexistence takes another 0.5–1 GiB and KV cache + browser overhead
 * 1–2 GiB, leaving ~7–8 GiB for the model. The project caps the chat model at
 * 8B params (Q4_K_M ≈ 5 GiB), so an 8 GiB MemoryPool default fits the load-
 * bearing agent + Three.js use case with headroom for one embedder alongside.
 * Callers may override via `WebLLMConfig.memoryBudget` for dev-tier (128 GB)
 * machines or constrained environments.
 */
export const DEFAULT_MEMORY_BUDGET_BYTES = 8 * 1024 * 1024 * 1024;

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
	backend: "default" | "jsep" = "default",
): string {
	if (override) return override;
	// JSEP variant ships a single artifact (no wasm32/wasm64 split for
	// the prototype — Phase 2 targets sub-3.5 GiB models only).
	if (backend === "jsep") {
		return "./webllm-wasm-jsep.js";
	}
	// Relative-path default: dynamic `import()` inside the bundle
	// resolves this against the bundle's own URL, so consumers who
	// drop `webllm-wasm.js` / `webllm-wasm-mem64.js` next to their
	// bundle "just work" without passing an explicit override. Bare
	// specifiers fail in plain ESM (no import map) and would force
	// every consumer to pass an override. Override remains available
	// for non-co-located deployments.
	return modelByteLength > WASM32_HEAP_MARGIN
		? "./webllm-wasm-mem64.js"
		: "./webllm-wasm.js";
}

function isWorkerContext(): boolean {
	// biome-ignore lint/suspicious/noExplicitAny: globalThis narrowing for WebWorker types not in tsconfig lib
	const g = globalThis as any;
	return (
		typeof g.DedicatedWorkerGlobalScope !== "undefined" &&
		g instanceof g.DedicatedWorkerGlobalScope
	);
}

// Project-side `ModelHyperparams` field names diverge from the persistence
// wire format (vocabularySize -> vocabSize, embeddingLength -> nEmbd, etc.).
// `ModelLoader.extractHyperparams` already defaults `ropeFreqBase` to 10_000
// when the GGUF metadata key is absent, so the typed value is always concrete.
function buildModelFingerprint(
	hp: ModelHyperparams,
	tokenizerHash: string,
): ModelFingerprint {
	return {
		architecture: hp.architecture,
		vocabSize: hp.vocabularySize,
		nEmbd: hp.embeddingLength,
		nLayer: hp.layerCount,
		nHead: hp.headCount,
		nHeadKV: hp.headCountKv,
		ropeBase: hp.ropeFreqBase,
		quantType: hp.quantType,
		tokenizerHash,
	};
}

/**
 * Main WebLLM engine: browser-side LLM inference over WebGPU.
 *
 * Hosts one or more GGUF models (chat, encoder, causal-embedder) and
 * drives streaming chat completion, multi-turn conversation KV-cache
 * reuse, sentence embeddings, and character personas with tool calling.
 * The WASM `ggml-webgpu` backend does the heavy lifting; this class is
 * the TypeScript orchestration layer above it. See the
 * [README](../../README.md) for the load → chat / embed / character
 * workflow and the architecture diagram.
 *
 * **Worker vs inline.** Pass `worker: true` in the config to run inference
 * off the main thread; {@link WebLLM.init} returns a proxy with the same
 * public surface that forwards every call over `postMessage`. Inline mode
 * (the default) runs everything on the main thread.
 */
export class WebLLM {
	private _config: WebLLMConfig;
	private _memoryPool: MemoryPool;
	private _scheduler: Scheduler;
	private _pipelineCache: PipelineCache;
	private _modelManager: ModelManager;
	private characterManager: CharacterManager;
	private eventHandlers = new Map<string, Set<EventHandler>>();
	/**
	 * Per-model state (ARC-004). Single row per loaded model id holding the
	 * wasm module, the type-partitioned engine (chat / encoder /
	 * causal-embedder), the default chat session, and the per-model
	 * chat-serialization chain. Replaces six prior parallel Maps.
	 */
	private models = new Map<string, ModelRecord>();
	private conversationPool: ConversationPool;

	/**
	 * Install JSEP callbacks on a freshly-initialized `GgmlWasm` if the
	 * engine was constructed with `backend: "jsep"`. Must be called AFTER
	 * `wasm.init()` (the module exists) and BEFORE the first model load
	 * (so the C++ side's `EM_ASM_INT(Module.jsepAlloc, ...)` finds a real
	 * callback rather than `undefined`).
	 *
	 * Acquires a JS-side `GPUDevice` via the standard
	 * `navigator.gpu.requestAdapter() → adapter.requestDevice()` path.
	 * The C++ ggml-webgpu backend inside the WASM owns its own (separate)
	 * device — the prototype's two backends partition ops via the
	 * scheduler and do not share GPU buffers.
	 */
	private async maybeInstallJsep(wasm: GgmlWasm): Promise<void> {
		if (this._config.backend !== "jsep") return;
		const gpu = (navigator as Navigator & { gpu?: GPU | undefined }).gpu;
		if (!gpu) {
			throw new Error(
				"backend: 'jsep' requires WebGPU; navigator.gpu is unavailable",
			);
		}
		const adapter = await gpu.requestAdapter();
		if (!adapter) {
			throw new Error("backend: 'jsep' could not acquire a GPUAdapter");
		}
		const device = await adapter.requestDevice();
		await wasm.installJsepCallbacks(device);
	}

	private constructor(config: WebLLMConfig) {
		this._config = config;
		this._memoryPool = new MemoryPool(
			config.memoryBudget ?? DEFAULT_MEMORY_BUDGET_BYTES,
		);
		this.characterManager = new CharacterManager();
		this._modelManager = new ModelManager(this._memoryPool);
		this._scheduler = new Scheduler({
			frameBudgetMs: config.frameBudgetMs ?? 8,
		});
		this._pipelineCache = new PipelineCache(config.cacheDir ?? "webllm-cache");
		this.conversationPool = new ConversationPool({
			maxConversations: config.maxConversations ?? 4,
		});
	}

	/**
	 * Initialize a WebLLM engine and return a ready-to-use instance.
	 *
	 * Factory entry point for the library. In inline mode (the default)
	 * constructs the engine directly; in worker mode (`config.worker: true`)
	 * returns a proxy that forwards every public method to a
	 * `DedicatedWorker` over `postMessage`. The proxy is structurally
	 * compatible with `WebLLM`, so callers can treat both paths identically.
	 *
	 * @example
	 * ```ts
	 * const engine = await WebLLM.init({
	 *   memoryBudget: 8 * 1024 ** 3, // optional; defaults to 8 GiB
	 *   worker: true,                 // off-main-thread inference
	 * });
	 * ```
	 */
	static async init(config: WebLLMConfig): Promise<WebLLM> {
		if (config.worker && !isWorkerContext()) {
			const { WebLLMProxy } = await import("./webllm-proxy.js");
			// The proxy mirrors WebLLM's public surface; its TS type is
			// structurally compatible enough to return as WebLLM.
			return WebLLMProxy.init(config) as unknown as Promise<WebLLM>;
		}
		return new WebLLM({ ...config, worker: false });
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
			embeddingCapable: options.embeddingCapable,
			embeddingPooling: options.embeddingPooling,
		};
		this._modelManager.register(entry);
		return entry;
	}

	async unloadModel(id: string): Promise<void> {
		this.conversationPool.disposeAllForModel(id);
		const record = this.models.get(id);
		if (record) {
			const { inference, encoder, causalEmbedder, wasm } = record;
			if (inference) await inference.dispose();
			if (encoder) await encoder.dispose();
			if (causalEmbedder) await causalEmbedder.dispose();
			if (wasm) await wasm.shutdown();
			this.models.delete(id);
		}
		await this._modelManager.unregister(id);
	}

	/**
	 * Release all engine resources: unload every model, free the WebGPU
	 * device, drop the wasm module references. After dispose(), the engine
	 * is unusable. Worker-mode callers see worker.terminate() too via
	 * WebLLMProxy.dispose().
	 */
	async dispose(): Promise<void> {
		const ids = [...this.models.keys()];
		const seen = new Set<string>();
		for (const id of ids) {
			if (seen.has(id)) continue;
			seen.add(id);
			await this.unloadModel(id);
		}
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

		const inf = this.models.get(modelId)?.inference;
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

		const inf = this.models.get(modelId)?.inference;
		if (!inf) throw new InferenceEngineMissingError(modelId);

		const tokenizer = entry.tokenizer;
		const chatTemplate = tokenizer.options.chatTemplate;
		const isQwenChatml =
			Array.isArray(input) &&
			String(entry.hyperparams.architecture).startsWith("qwen") &&
			detectChatTemplate(chatTemplate ?? "") === "chatml";
		const isPhi3 =
			Array.isArray(input) &&
			(String(entry.hyperparams.architecture) === "phi3" ||
				detectChatTemplate(chatTemplate ?? "") === "phi3");
		const isMistral =
			Array.isArray(input) &&
			detectChatTemplate(chatTemplate ?? "") === "llama2" &&
			!(chatTemplate ?? "").includes("<<SYS>>");
		const isGemma4 =
			Array.isArray(input) &&
			String(entry.hyperparams.architecture) === "gemma4";
		const {
			temperature: effectiveTemperature,
			topK: effectiveTopK,
			topP: effectiveTopP,
			repetitionPenalty: effectiveRepetitionPenalty,
		} = resolveSamplingParams({
			samplingMode: config?.sampling ?? "auto",
			isQwenChatml,
			isPhi3,
			isMistral,
			isGemma4,
			enableThinking: config?.enableThinking,
			consumer: {
				temperature: config?.temperature,
				topK: config?.topK,
				topP: config?.topP,
				repetitionPenalty: config?.repetitionPenalty,
			},
		});
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
			} else if (
				String(entry.hyperparams.architecture) === "phi3" ||
				detectChatTemplate(chatTemplate ?? "") === "phi3"
			) {
				// Phi-3 / Phi-3.5 emit `<|end|>` to terminate every assistant
				// turn. The GGUF's declared EOS is `<|endoftext|>` (rarely
				// produced in chat), so without this the model runs to
				// maxTokens and the response wanders through training-data
				// completions after the real reply ends.
				addChatStopToken(genConfig, tokenizer, config, "<|end|>");
			} else {
				const tmpl = detectChatTemplate(chatTemplate ?? "");
				if (tmpl === "llama2" || tmpl === "mistral-v7") {
					// Mistral / Llama-2 [INST]…[/INST] family terminates each
					// assistant turn with `</s>`. The declared EOS in some
					// uploader-built GGUFs is `<s>` (id 1) instead of `</s>`
					// (id 2), so without this the model continues past
					// end-of-turn into a fabricated multi-turn dialogue with
					// itself. See CLAUDE.md regression notes.
					addChatStopToken(genConfig, tokenizer, config, "</s>");
				} else if (tmpl === "chatml") {
					// Non-Qwen ChatML models (Hermes-3, SmolLM2, etc.) — the
					// full Qwen branch above is gated on `architecture` ===
					// "qwen*" because it carries Qwen-specific think-mode and
					// tool-call masking. Other chatml-trained models still
					// need `<|im_end|>` registered so they stop at turn end
					// rather than running into multi-turn self-dialogue.
					// `<|endoftext|>` is *not* registered here: in some
					// non-Qwen chatml vocabs (e.g. SmolLM2) it resolves to a
					// low id that aliases the `<unk>`/pad slot, and
					// registering id 0 as a stop would terminate generation
					// on any unknown-token emission.
					addChatStopToken(genConfig, tokenizer, config, "<|im_end|>");
				} else if (tmpl === "gemma" || tmpl === "gemma4") {
					// Gemma chat models terminate every turn with
					// `<end_of_turn>`. The GGUF's declared EOS is `<eos>`
					// (id 1), which the model rarely emits in chat — without
					// explicit `<end_of_turn>` registration the model runs
					// to maxTokens. Unsloth's Gemma-4 / Gemma-3N variant
					// renames the literal to `<turn|>` (id 106); pick the
					// literal that the active template actually uses so
					// `addChatStopToken` resolves to a real vocab id.
					const tpl = tokenizer.options.chatTemplate ?? "";
					const stopText = tpl.includes("<turn|>")
						? "<turn|>"
						: "<end_of_turn>";
					addChatStopToken(genConfig, tokenizer, config, stopText);
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
	 * Tokenize `text` using the loaded model's tokenizer and return the
	 * resulting token IDs. Useful for context-window accounting on the
	 * UI side — counting `tokenize(id, fullPromptText).length` against
	 * `model.contextLength` reflects what the engine will actually
	 * prefill more faithfully than a `chars / 4` estimate.
	 *
	 * Throws `ModelNotFoundError` if `modelHandleId` was never registered;
	 * `ModelNotLoadedError` if the model is registered but its tokenizer
	 * isn't ready yet (load in progress).
	 */
	tokenize(modelHandleId: string, text: string): readonly number[] {
		const entry = this._modelManager.get(modelHandleId);
		if (!entry) throw new ModelNotFoundError(modelHandleId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelHandleId);
		}
		return entry.tokenizer.encode(text);
	}

	async createConversation(
		modelHandleId: string,
		options: ConversationOptions = {},
	): Promise<ConversationHandle> {
		const entry = this._modelManager.get(modelHandleId);
		if (!entry) throw new ModelNotFoundError(modelHandleId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelHandleId);
		}
		const inf = this.models.get(modelHandleId)?.inference;
		if (!inf) throw new InferenceEngineMissingError(modelHandleId);
		if (!inf.flashAttn) {
			throw new Error(
				`Conversations require FA mode (flashAttn=true); model "${modelHandleId}" is in manual mode.`,
			);
		}
		// options.maxContextTokens is forwarded unclamped; chatCompletion(conv,…)
		// is where ConversationContextOverflowError fires against the model's max.
		return this.conversationPool.create(modelHandleId, options);
	}

	async disposeConversation(conv: ConversationHandle): Promise<void> {
		this.conversationPool.dispose(conv);
	}

	/**
	 * Spawn a new conversation that inherits a deep copy of `src`'s KV
	 * snapshot. The new conversation's first `chatCompletion` call will
	 * find the inherited snapshot via the longest-shared-token-prefix
	 * walk and prefill only the divergent tail.
	 *
	 * Use case: multiple agents (e.g. NPCs) sharing a long system /
	 * persona prefix. Drive one base conversation through the shared
	 * prefix once, then fork it per-agent — each fork pays only the
	 * agent-specific tail prefill instead of re-prefilling the shared
	 * prefix from scratch.
	 *
	 * Throws `ConversationNotFoundError` if `src` doesn't exist (or was
	 * disposed). Throws `ConversationNotPopulatedError` if `src` has
	 * no snapshot yet — drive at least one `chatCompletion` call on
	 * `src` before forking.
	 *
	 * Spec follow-up #2 (cross-conversation prefix sharing) — copy-from-
	 * prefix-store path; no C++ patch.
	 */
	async forkConversation(src: ConversationHandle): Promise<ConversationHandle> {
		return this.conversationPool.fork(src);
	}

	/**
	 * Serialize a conversation's KV cache and token prefix into a portable
	 * `Uint8Array` blob (the `WLKV` wire format).
	 *
	 * The blob is self-describing: it carries a `ModelFingerprint` so
	 * {@link importConversation} can refuse a mismatched model. Pair with
	 * `IndexedDBConversationStore` (or any custom store against the same
	 * `Uint8Array` contract) for cross-session persistence.
	 *
	 * Throws {@link ConversationBusyError} if another call is in flight on
	 * this handle; {@link ConversationNotPopulatedError} if the conversation
	 * has no saved snapshot yet (drive at least one `chatCompletion` first);
	 * {@link ModelNotFoundError} or {@link InferenceEngineMissingError} if
	 * the underlying model was unloaded.
	 *
	 * @example
	 * ```ts
	 * const blob = await engine.exportConversation(conv);
	 * await store.put("user-42-session", blob);
	 * ```
	 */
	async exportConversation(conv: ConversationHandle): Promise<Uint8Array> {
		this.conversationPool.assertExists(conv);
		const release = this.conversationPool.tryAcquireLock(conv);
		if (!release) throw new ConversationBusyError(conv.id);
		try {
			const snap = this.conversationPool.get(conv);
			if (!snap) throw new ConversationNotPopulatedError(conv.id);
			const entry = this._modelManager.get(snap.modelHandleId);
			if (!entry) throw new ModelNotFoundError(snap.modelHandleId);
			if (!entry.fingerprint) {
				throw new InferenceEngineMissingError(snap.modelHandleId);
			}
			const header = {
				schemaVersion: KV_PERSISTENCE_SCHEMA_VERSION as 1,
				fingerprint: entry.fingerprint,
				conversationOptions: this.conversationPool.options(conv),
				tokenIds: snap.tokenIds,
				byteSize: snap.byteSize,
				savedAtMs: Date.now(),
			};
			return encodePersistedConversation(header, snap.kvBytes);
		} finally {
			release();
		}
	}

	/**
	 * Rehydrate a conversation from a `WLKV` blob produced by
	 * {@link exportConversation}.
	 *
	 * Validates the blob's `ModelFingerprint` against the currently-loaded
	 * model and restores the KV cache and token prefix into a fresh
	 * conversation handle. Requires Flash-Attention mode
	 * (`ModelLoadOptions.flashAttn: true` at load time).
	 *
	 * Throws {@link ModelNotFoundError}, {@link ModelNotLoadedError}, or
	 * {@link InferenceEngineMissingError} for model lifecycle failures;
	 * {@link IncompatibleConversationError} (from the persistence decoder)
	 * when the fingerprint doesn't match; {@link CorruptBlobError} on
	 * malformed bytes. Also throws a generic `Error` if the model is not
	 * in FA mode.
	 *
	 * @example
	 * ```ts
	 * const blob = await store.get("user-42-session");
	 * const conv = blob
	 *   ? await engine.importConversation(model.id, blob)
	 *   : await engine.createConversation(model.id);
	 * ```
	 */
	async importConversation(
		modelHandleId: string,
		blob: Uint8Array,
		options?: ConversationOptions,
	): Promise<ConversationHandle> {
		const entry = this._modelManager.get(modelHandleId);
		if (!entry) throw new ModelNotFoundError(modelHandleId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelHandleId);
		}
		const inf = this.models.get(modelHandleId)?.inference;
		if (!inf) throw new InferenceEngineMissingError(modelHandleId);
		if (!inf.flashAttn) {
			throw new Error(
				`importConversation requires FA mode; "${modelHandleId}" is in manual mode.`,
			);
		}
		if (!entry.fingerprint) {
			throw new InferenceEngineMissingError(modelHandleId);
		}
		const { header, kvBytes } = decodePersistedConversation(
			blob,
			entry.fingerprint,
		);
		const opts = options ?? header.conversationOptions;
		const conv = this.conversationPool.create(modelHandleId, opts);
		this.conversationPool.set(conv, {
			conversationId: conv.id,
			modelHandleId,
			tokenIds: header.tokenIds,
			kvBytes,
			byteSize: header.byteSize,
			lastAccessMs: Date.now(),
		});
		return conv;
	}

	/**
	 * Streaming chat completion with multi-turn KV cache reuse.
	 *
	 * Yields CompletionChunks with incremental text, followed by a final
	 * done=true chunk carrying generation stats. The KV cache is reused
	 * across calls when the message array grows incrementally.
	 *
	 * Two dispatch modes:
	 *   - `chatCompletion(modelId, messages, config)` — model-id-based.
	 *     Drives the engine's per-model session-tracker; the prior turn's
	 *     working KV is reused via the existing delta-prefill heuristic.
	 *   - `chatCompletion(conv, messages, config)` — conversation-handle-
	 *     based. Performs longest-shared-token-prefix detection vs a saved
	 *     KV snapshot, swaps it into the working KV, prefills the divergent
	 *     tail, decodes, and saves an updated snapshot back into the pool.
	 *     Two conversations on the same model are isolated via a per-model
	 *     serialization chain.
	 */
	async *chatCompletion(
		first: string | ConversationHandle,
		messages: ChatMessage[],
		config?: CompletionConfig,
	): AsyncGenerator<CompletionChunk, void> {
		if (typeof first === "string") {
			yield* this.generateStream(first, messages, config);
			return;
		}
		yield* this.chatCompletionWithConversation(first, messages, config);
	}

	private async *chatCompletionWithConversation(
		conv: ConversationHandle,
		messages: ChatMessage[],
		config?: CompletionConfig,
	): AsyncGenerator<CompletionChunk, void> {
		// 1. Validate handle + acquire per-conv lock (single-writer).
		this.conversationPool.assertExists(conv);
		const release = this.conversationPool.tryAcquireLock(conv);
		if (!release) throw new ConversationBusyError(conv.id);

		// 2. Per-model serialization chain — the working KV is shared across
		// conversations on the same model. Wait for any prior conv call on
		// this model to finish before entering our load/prefill/decode/save.
		const chainRecord = this.models.get(conv.modelHandleId);
		const prior = chainRecord?.chatChain;
		let resolveChain!: () => void;
		const chainTail = new Promise<void>((res) => {
			resolveChain = res;
		});
		if (chainRecord) chainRecord.chatChain = chainTail;
		if (prior) await prior;

		try {
			// 3. Resolve model resources.
			const entry = this._modelManager.get(conv.modelHandleId);
			if (!entry) throw new ModelNotFoundError(conv.modelHandleId);
			if (!entry.loaded || !entry.tokenizer) {
				throw new ModelNotLoadedError(conv.modelHandleId);
			}
			const inf = this.models.get(conv.modelHandleId)?.inference;
			if (!inf) throw new InferenceEngineMissingError(conv.modelHandleId);
			const tokenizer = entry.tokenizer;

			if (config?.drafter !== undefined) {
				throw new SpeculativeDecodingReservedError();
			}

			// 4. Tokenize the canonical prompt (chat template applied).
			const newTokens = encodeChatPrompt(messages, tokenizer, {
				enableThinking: config?.enableThinking,
				tools: config?.tools,
			});

			// 5. Context-overflow check.
			const convOpts = this.conversationPool.options(conv);
			const maxCtx = convOpts.maxContextTokens ?? inf.maxContextLength;
			if (newTokens.length > maxCtx) {
				throw new ConversationContextOverflowError(
					conv.id,
					newTokens.length,
					maxCtx,
				);
			}

			// 6. Longest-shared-token-prefix vs prior snapshot.
			const priorSnap = this.conversationPool.get(conv);
			let sharedLen = 0;
			if (priorSnap) {
				const upper = Math.min(priorSnap.tokenIds.length, newTokens.length);
				while (
					sharedLen < upper &&
					priorSnap.tokenIds[sharedLen] === newTokens[sharedLen]
				) {
					sharedLen++;
				}
			}

			// 7. Load phase — swap snapshot's [0, sharedLen) into working KV.
			if (sharedLen > 0 && priorSnap) {
				await inf.loadKVCache(
					priorSnap.kvBytes,
					sharedLen,
					priorSnap.tokenIds.length,
				);
			} else {
				inf.resetKVCache();
			}

			// 8. Prefill all but the last prompt token manually. The last
			// token is handed to generateTextStream as its prompt so its
			// own prefill writes the final KV slot AND returns the logits
			// needed to sample the first generated token. Edge cases:
			//   - newTokens.length === 1: nothing to manually prefill.
			//   - sharedLen === newTokens.length: every prompt token is
			//     already in the loaded KV. The last slot WILL be rewritten
			//     by generateTextStream's prefill — but since the model is
			//     deterministic given conditioning, KV at that position is
			//     identical, and this costs one cheap forward pass.
			const lastTokenId = newTokens[newTokens.length - 1];
			const lastPos = newTokens.length - 1;
			const manualPrefillEnd = lastPos; // exclusive on the last token
			if (sharedLen < manualPrefillEnd) {
				const midIds = newTokens.slice(sharedLen, manualPrefillEnd);
				const midPos = new Int32Array(midIds.length);
				for (let i = 0; i < midIds.length; i++) midPos[i] = sharedLen + i;
				await inf.forward(new Int32Array(midIds), midPos);
			} else if (sharedLen > manualPrefillEnd) {
				// Loaded KV is longer than the prompt minus its last token —
				// truncate it back so generateTextStream's prefill writes the
				// last slot (rather than reading stale data past its
				// expected end).
				inf.truncateKVCache(manualPrefillEnd);
			}

			// 9. Build sampler/genConfig (mirror generateStream's resolution).
			const isQwenChatml =
				String(entry.hyperparams.architecture).startsWith("qwen") &&
				detectChatTemplate(tokenizer.options.chatTemplate ?? "") === "chatml";
			const isPhi3 =
				String(entry.hyperparams.architecture) === "phi3" ||
				detectChatTemplate(tokenizer.options.chatTemplate ?? "") === "phi3";
			const isMistral =
				detectChatTemplate(tokenizer.options.chatTemplate ?? "") === "llama2" &&
				!(tokenizer.options.chatTemplate ?? "").includes("<<SYS>>");
			const isGemma4 = String(entry.hyperparams.architecture) === "gemma4";
			const {
				temperature: effectiveTemperature,
				topK: effectiveTopK,
				topP: effectiveTopP,
				repetitionPenalty: effectiveRepetitionPenalty,
			} = resolveSamplingParams({
				samplingMode: config?.sampling ?? "auto",
				isQwenChatml,
				isPhi3,
				isMistral,
				isGemma4,
				enableThinking: config?.enableThinking,
				consumer: {
					temperature: config?.temperature,
					topK: config?.topK,
					topP: config?.topP,
					repetitionPenalty: config?.repetitionPenalty,
				},
			});
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

			// 9b. Qwen3 chat-ML special handling — mirrors the block in
			// generateStream (engine.ts qwen-chatml block). Without this the
			// conv path regresses qwen3 generation quality (think-mode mask,
			// reentry forbidden tokens, dual EOS stop set). See CLAUDE.md
			// regression note: <|im_end|> (151645) and <|endoftext|> (151643)
			// are both valid end-of-turn tokens for qwen3 chat.
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
			} else if (
				String(entry.hyperparams.architecture) === "phi3" ||
				detectChatTemplate(tokenizer.options.chatTemplate ?? "") === "phi3"
			) {
				// Phi-3 / Phi-3.5: stop on `<|end|>` (see same-named block in
				// chatCompletion). Without this the conv path also wanders
				// through training data after the real reply ends.
				addChatStopToken(genConfig, tokenizer, config, "<|end|>");
			} else {
				const tmpl = detectChatTemplate(tokenizer.options.chatTemplate ?? "");
				if (tmpl === "llama2" || tmpl === "mistral-v7") {
					// Mistral / Llama-2 [INST] family: stop on `</s>` (see
					// same-named block in chatCompletion).
					addChatStopToken(genConfig, tokenizer, config, "</s>");
				} else if (tmpl === "chatml") {
					// Non-Qwen ChatML (Hermes-3, SmolLM2, etc.): stop on
					// `<|im_end|>` (see same-named block in chatCompletion).
					addChatStopToken(genConfig, tokenizer, config, "<|im_end|>");
				} else if (tmpl === "gemma" || tmpl === "gemma4") {
					// Gemma: stop on `<end_of_turn>` / `<turn|>` (see
					// same-named block in chatCompletion for the unsloth
					// Gemma-4 / Gemma-3N variant rationale).
					const tpl = tokenizer.options.chatTemplate ?? "";
					const stopText = tpl.includes("<turn|>")
						? "<turn|>"
						: "<end_of_turn>";
					addChatStopToken(genConfig, tokenizer, config, stopText);
				}
			}

			// 10. Decode loop — drive generateTextStream with the last prompt
			// token as a single-token "prefill". generateTextStream's prefill
			// rewrites slot `lastPos` (deterministic given the same
			// conditioning, so KV is identical) and yields its logits to
			// sample the first generated token. The session starts at
			// `lastPos` so position arithmetic stays consistent.
			const forwardPass = async (
				ids: number[],
				positions: number[],
			): Promise<Float32Array> => {
				return await inf.forward(
					new Int32Array(ids),
					new Int32Array(positions),
				);
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

			// InferenceSession's constructor signature is (config, sequenceId);
			// the second arg is a sequence id, NOT the starting position
			// (the constructor unconditionally sets position=0). Use
			// `advance(lastPos)` to seed the position so generateTextStream's
			// prefill writes the last prompt token at slot `lastPos` rather
			// than overwriting slot 0 of the loaded prefix.
			const seedSession = new InferenceSession(
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
			seedSession.advance(lastPos);

			const generatedIds: number[] = [];
			for await (const chunk of generateTextStream({
				promptTokenIds: [lastTokenId],
				sampler,
				session: seedSession,
				eosTokenId: tokenizer.eosId,
				tokenizer,
				forwardPass,
				config: genConfig,
				forwardDecode,
			})) {
				if (chunk.tokenId !== undefined) {
					generatedIds.push(chunk.tokenId);
				}
				yield chunk;
			}

			// 12. Save phase. The working KV now holds [0, finalLen) where
			// finalLen = newTokens.length + generatedCount. Snapshot and
			// store under the conversation handle. Skip when the caller
			// flagged `skipSave` — the ~1.5 s serialize cost is the
			// dominant per-call overhead, and ticks that won't reuse the
			// prefix don't pay it.
			if (config?.skipSave !== true) {
				const finalLen = inf.cachedTokenCount;
				const fullIds = new Array<number>(finalLen);
				for (let i = 0; i < newTokens.length && i < finalLen; i++) {
					fullIds[i] = newTokens[i];
				}
				// The generated-tail token ids matter only insofar as the next
				// turn's longest-shared-prefix walk reaches them. A new user
				// turn always introduces fresh tokens after the assistant
				// response, so divergence happens at or before the first
				// generated id. We store them faithfully when available; pad
				// with -1 if generatedIds underruns finalLen (e.g., when
				// generateTextStream advanced cachedTokenCount via its prefill
				// without yielding a sampled token — shouldn't happen, but
				// defensive).
				for (let i = newTokens.length, g = 0; i < finalLen; i++, g++) {
					fullIds[i] = g < generatedIds.length ? generatedIds[g] : -1;
				}

				const kvBytes = await inf.serializeKVCache(finalLen);
				this.conversationPool.set(conv, {
					conversationId: conv.id,
					modelHandleId: conv.modelHandleId,
					tokenIds: fullIds,
					kvBytes,
					byteSize: kvBytes.byteLength,
					lastAccessMs: Date.now(),
				});
			}
		} finally {
			release();
			resolveChain();
			const r = this.models.get(conv.modelHandleId);
			if (r?.chatChain === chainTail) {
				// Drop only the chain slot — the model is still loaded, so
				// the record itself stays. `delete` is required because
				// `exactOptionalPropertyTypes: true` rejects `= undefined`.
				delete r.chatChain;
			}
		}
	}

	/**
	 * @internal — for unit tests only. Returns the inference engine for a
	 * model id so tests can spy on its KV-cache primitives. Not a stable
	 * API; the field type is the public {@link ModelInference}.
	 */
	__debugInferenceForModel(modelId: string): ModelInference | undefined {
		return this.models.get(modelId)?.inference;
	}

	/**
	 * Compute an L2-normalized sentence embedding for the given text.
	 *
	 * Dispatch order (first match wins):
	 *   1. **Encoder** — bidirectional BERT/RoBERTa model registered as
	 *      `ModelRecord.encoder` (e.g. `bge-large-en-v1.5`). Highest quality.
	 *   2. **Causal-embedder** — causal-LM fine-tuned for retrieval,
	 *      registered as `ModelRecord.causalEmbedder` (e.g.
	 *      `qwen3-embedding-0.6b-hyb`). MTEB-competitive quality.
	 *   3. **Chat-model / bucket D** — general chat model whose registration
	 *      entry carries `embeddingCapable: true`. Taps the post-`output_norm`
	 *      hidden state at the final EOS token. Carries a ~5-15% MTEB delta
	 *      vs a dedicated embedder; suitable for in-domain agent retrieval
	 *      when a second model is not loaded. Only models that have passed the
	 *      parity gate are registered with `embeddingCapable`.
	 *
	 * Throws {@link EncoderRequiredError} when the model is loaded but falls
	 * into none of the three tiers (i.e. it is a plain chat model without
	 * `embeddingCapable: true`).
	 */
	async embed(modelId: string, text: string): Promise<Float32Array> {
		const entry = this._modelManager.get(modelId);
		if (!entry) throw new ModelNotFoundError(modelId);
		if (!entry.loaded || !entry.tokenizer) {
			throw new ModelNotLoadedError(modelId);
		}
		const ids = entry.tokenizer.encode(text);
		const record = this.models.get(modelId);
		const enc = record?.encoder;
		if (enc) return enc.embed(new Int32Array(ids));
		const cembed = record?.causalEmbedder;
		if (cembed) {
			// Causal-LM-derived embedders require an explicit EOS marker at the
			// end of the input (the model was fine-tuned to pool the hidden
			// state at this position). sentence-transformers appends EOS via
			// `add_special_tokens=True`; our `Tokenizer.encode()` does not, so
			// we append it here to match the reference encoding behavior. Drop
			// any existing trailing EOS (e.g. from a tokenizer that already
			// added one) before re-appending so the count stays at one.
			const eos = entry.tokenizer.eosId;
			const withEos =
				ids.length > 0 && ids[ids.length - 1] === eos ? ids : [...ids, eos];
			return cembed.embed(new Int32Array(withEos));
		}

		// Bucket D — chat-model self-embedding. Gated on the registration
		// flag so we only fall through for chat models that have passed
		// the parity gate (encoder / causalEmbedder remain the high-quality
		// path; bucket D is the simplicity / single-model-load path).
		if (entry.embeddingCapable) {
			const inf = record?.inference;
			if (inf) {
				const eos = entry.tokenizer.eosId;
				const withEos =
					ids.length > 0 && ids[ids.length - 1] === eos ? ids : [...ids, eos];
				const pooling = entry.embeddingPooling ?? "last-token";
				return inf.embed(new Int32Array(withEos), { pooling });
			}
		}

		throw new EncoderRequiredError(
			modelId,
			String(entry.hyperparams.architecture),
			"register the model with `embeddingCapable: true` to use the chat-model self-embedding path",
		);
	}

	/**
	 * Reset the per-model default session and KV cache for a loaded model.
	 *
	 * Takes a model id (not a {@link ConversationHandle} — the neighboring
	 * conversation-pool APIs operate on those), drops the model's default
	 * session-tracker entry, and clears its inference engine's KV cache.
	 */
	resetModelSession(modelId: string): void {
		const record = this.models.get(modelId);
		if (record) delete record.session;
		const inf = record?.inference;
		if (inf) inf.resetKVCache();
	}

	/** @deprecated Use {@link resetModelSession}. */
	resetConversation(modelId: string): void {
		this.resetModelSession(modelId);
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
			session.currentPosition === inf.cachedTokenCount &&
			leadingMessagesMatch(sessionInfo.cachedMessages, promptMessages)
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
		sessionInfo.cachedMessages = promptMessages.map((m) => ({
			role: m.role,
			content: m.content,
		}));
		return promptTokens;
	}

	private getOrCreateSession(modelId: string): ConversationSession {
		let record = this.models.get(modelId);
		let entry = record?.session;
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
				cachedMessages: [],
			};
			if (!record) {
				record = {};
				this.models.set(modelId, record);
			}
			record.session = entry;
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
	 * @param name - Unique model identifier within this engine.
	 * @param pipeline - The manually-assembled WASM, inference, and parsed model.
	 * @param options - Optional metadata (e.g. embeddingCapable to enable bucket D dispatch).
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
		options?: {
			embeddingCapable?: boolean;
			embeddingPooling?: "last-token" | "mean";
			/**
			 * GGUF byteLength of the adopted model, used to record the
			 * weight footprint in the MemoryPool so the budget gate
			 * (`ModelManager.canFit`) and pressure/eviction events remain
			 * truthful on the adopt path. Optional because some callers
			 * hand off a pipeline built from a stream they no longer
			 * reference; omitting it leaves the pool untouched (as before).
			 */
			weightBytes?: number;
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
		entry.tokenizerHash = await computeTokenizerHash(
			pipeline.parsed.tokenizerConfig,
		);
		entry.fingerprint = buildModelFingerprint(
			pipeline.parsed.hyperparams,
			entry.tokenizerHash,
		);
		entry.loaded = true;
		if (options?.embeddingCapable !== undefined) {
			entry.embeddingCapable = options.embeddingCapable;
		}
		if (options?.embeddingPooling !== undefined) {
			entry.embeddingPooling = options.embeddingPooling;
		}
		// Upsert the model record: wasm + exactly one of the three
		// engine-kind fields (preserving the type partition per ARC-006).
		const record = this.models.get(handle.id) ?? {};
		record.wasm = pipeline.wasm;
		if (isEncoder) {
			record.encoder = pipeline.inference as EncoderInference;
		} else if (isCausalEmbedder) {
			record.causalEmbedder = pipeline.inference as CausalLMEmbedder;
		} else {
			record.inference = pipeline.inference as ModelInference;
		}
		this.models.set(handle.id, record);
		// ARC-002: record the adopted pipeline's weight footprint when the
		// caller knows it, so `unloadModel` → `_modelManager.unregister`
		// → `memoryPool.evictModel` keeps the budget honest on this path.
		if (options?.weightBytes !== undefined && options.weightBytes > 0) {
			this._memoryPool.allocate(options.weightBytes, 0, handle.id);
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
		options?: Partial<ModelLoadOptions>,
	): Promise<{
		handle: ModelHandle;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
		metadata: LoadedModelMetadata;
	}> {
		const view = data instanceof Uint8Array ? data : new Uint8Array(data);
		const parsed = ModelLoader.parseModel(view);
		const ggufCtx = GgufParser.parse(view) as GgufContext;

		const resolvedWasmUrl = pickWasmUrl(
			view.byteLength,
			wasmUrl,
			this._config.backend ?? "default",
		);
		const wasm = new GgmlWasm();
		await wasm.init({ wasmUrl: resolvedWasmUrl });
		// JSEP callbacks must land BEFORE _buildInferenceAndRegister kicks
		// off model load — `ggml_backend_jsep_alloc_buffer` is invoked
		// during weight upload and reads `Module.jsepAlloc` synchronously.
		// If JSEP install throws (e.g., `navigator.gpu` undefined,
		// `requestAdapter()` returns null, `requestDevice()` rejects), the
		// freshly-`init`ed `wasm` would otherwise be orphaned, leaking the
		// in-WASM GPUDevice + pipeline state. Mirror the staging-failure
		// shutdown pattern below.
		try {
			await this.maybeInstallJsep(wasm);
		} catch (err) {
			try {
				await wasm.shutdown();
			} catch {
				// Best effort — don't mask the original install error.
			}
			throw err;
		}

		// `view` lives in the JS heap (not the WASM heap), so there is no
		// staging pointer to free — pass `undefined`. `_buildInferenceAndRegister`
		// owns failure-path teardown of the wasm module from here on.
		return this._buildInferenceAndRegister(
			parsed,
			ggufCtx,
			view,
			wasm,
			name,
			options,
			undefined,
			view.byteLength,
		);
	}

	/**
	 * Load a GGUF model by URL, streaming bytes directly into the WASM heap.
	 *
	 * The main-thread / worker-thread mirror of {@link loadModelFromBuffer}
	 * for the case where the caller does not already hold the full GGUF as
	 * an ArrayBuffer. Streaming into the WASM heap (instead of through a
	 * JS-heap intermediary `ArrayBuffer`) avoids V8's per-allocation cap,
	 * which trips for models larger than ~3.5 GB. This is the canonical
	 * worker-mode load path for 7B+ models.
	 *
	 * Steps:
	 *   1. Fetch the URL; require `content-length` so the WASM heap can be
	 *      sized up-front (no realloc on the hot path).
	 *   2. Init `GgmlWasm` with the binary picked from the model size
	 *      (wasm32 ≤ 3.5 GiB; wasm64 above) unless `wasmUrl` overrides.
	 *   3. `wasm.malloc(total)` reserves the GGUF region; chunks are
	 *      streamed via `wasm.heapU8.set(value, ptr + received)`.
	 *   4. Parse + build inference + upload weights using the callback
	 *      form of `loadWeights` so weight uploads survive any heap-grow
	 *      events triggered by per-chunk scratch mallocs inside ggml.
	 *   5. Free the GGUF heap region after weights have been uploaded to
	 *      GPU buffers — same pattern as the smoke-test main-thread path.
	 */
	async loadModelFromUrl(
		url: string,
		name: string,
		wasmUrl?: string,
		options?: Partial<ModelLoadOptions>,
		onProgress?: (received: number, total: number) => void,
	): Promise<{
		handle: ModelHandle;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
		metadata: LoadedModelMetadata;
	}> {
		const resp = await fetch(url);
		if (!resp.ok) {
			throw new Error(`loadModelFromUrl: HTTP ${resp.status} fetching ${url}`);
		}
		const total = Number(resp.headers.get("content-length") || 0);
		if (!total || total <= 0) {
			throw new Error(
				`loadModelFromUrl: missing content-length on ${url}; streaming into the WASM heap requires it`,
			);
		}
		if (!resp.body) {
			throw new Error(`loadModelFromUrl: response body unavailable for ${url}`);
		}

		const resolvedWasmUrl = pickWasmUrl(
			total,
			wasmUrl,
			this._config.backend ?? "default",
		);
		const wasm = new GgmlWasm();
		await wasm.init({ wasmUrl: resolvedWasmUrl });
		// JSEP callbacks must land BEFORE the first weight upload. The
		// `try` block below malloc's the staging region and streams bytes,
		// then hands the staging ptr to `_buildInferenceAndRegister` which
		// invokes `webllm_load_model` (the JSEP `alloc_buffer` trigger).
		// If JSEP install throws BEFORE we enter the staging try/catch
		// below, the freshly-`init`ed `wasm` would be orphaned (no
		// `wasm.shutdown()` runs, leaking the in-WASM GPUDevice + pipeline
		// state). Mirror the staging-failure shutdown pattern below.
		try {
			await this.maybeInstallJsep(wasm);
		} catch (err) {
			try {
				await wasm.shutdown();
			} catch {
				// Best effort — don't mask the original install error.
			}
			throw err;
		}

		// Post-init cleanup: if any of malloc / fetch / parse / build fails,
		// we own the `wasm` (it has constructed a WebGPU device + pipeline
		// state) and must `shutdown()` it before propagating. On the
		// success path the registered model entry takes ownership of `wasm`
		// — do NOT shutdown there. Hence try/catch (not try/finally).
		//
		// Staging-pointer ownership: once the staging ptr is handed to
		// `_buildInferenceAndRegister`, that helper owns freeing it (both
		// success path — between `loadWeights` and `initKVCache`, so the
		// model staging buffer is freed before ~1 GB KV-cache allocation
		// — and failure path inside the helper). The catch here only
		// frees `ptr` if we threw BEFORE handing ownership to the helper
		// (e.g., short read, parse failure, or malloc returning 0).
		let ptr = 0;
		let stagingOwnedByHelper = false;
		try {
			ptr = wasm.malloc(total);
			if (!ptr) {
				throw new Error(
					`loadModelFromUrl: wasm.malloc(${total}) returned null for ${url}`,
				);
			}

			const reader = resp.body.getReader();
			let received = 0;
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				// Re-fetch heapU8 each chunk: malloc above and any future
				// heap growth detach prior buffer references; the getter
				// returns the current view.
				wasm.heapU8.set(value, ptr + received);
				received += value.length;
				onProgress?.(received, total);
			}
			if (received !== total) {
				throw new Error(
					`loadModelFromUrl: short read from ${url}: expected ${total} bytes, got ${received}`,
				);
			}

			// Callback form keeps weight uploads safe across any heap-grow
			// events triggered by per-chunk scratch mallocs inside ggml.
			const dataAt = (off: number, len: number): Uint8Array =>
				new Uint8Array(wasm.heapU8.buffer, ptr + off, len);
			const fullView = dataAt(0, total);
			const parsed = ModelLoader.parseModel(fullView);
			const ggufCtx = GgufParser.parse(fullView) as GgufContext;

			// Hand staging ownership to the helper: it frees `ptr` after
			// `loadWeights` and BEFORE `initKVCache`, which keeps the
			// transient WASM-heap footprint to max(model_bytes, KV_bytes)
			// rather than (model_bytes + KV_bytes). Critical for 7B+ Q4
			// models on the wasm64 16 GiB heap cap.
			stagingOwnedByHelper = true;
			return this._buildInferenceAndRegister(
				parsed,
				ggufCtx,
				dataAt,
				wasm,
				name,
				options,
				ptr,
				total,
			);
		} catch (e) {
			// Partial-failure path: free the staging allocation if it
			// hasn't been handed to the helper yet, then tear down the
			// WebGPU device + pipeline state `wasm.init()` constructed.
			// Without this, every failed load leaks a WebGPU device.
			if (ptr && !stagingOwnedByHelper) {
				try {
					wasm.free(ptr);
				} catch {
					// ignore; shutdown below releases all heap state anyway
				}
			}
			try {
				await wasm.shutdown();
			} catch {
				// ignore shutdown errors; we're already on the failure path
			}
			throw e;
		}
	}

	/**
	 * Shared post-parse path used by both `loadModelFromBuffer` and
	 * `loadModelFromUrl`. Builds the architecture-appropriate inference
	 * pipeline, uploads weights, registers the handle on the engine.
	 *
	 * `dataSrc` may be a Uint8Array (when the GGUF lives in a JS-heap
	 * buffer that is independent of the WASM heap) or a callback
	 * `(offset, len) => Uint8Array` (when the GGUF lives in the WASM
	 * heap and weight uploads must re-derive views post-grow).
	 *
	 * **Staging-pointer ownership.** When `stagingPtr` is supplied (the
	 * `loadModelFromUrl` path, where the GGUF was streamed into a WASM-
	 * heap region via `wasm.malloc(total)`), this helper takes ownership
	 * of freeing it. The free is sequenced AFTER `loadWeights` (weights
	 * are now resident in GPU buffers; the WASM-heap copy is no longer
	 * needed) and BEFORE `initKVCache` (which `ctxCreate`s ~1 GB of KV +
	 * scratch tensors). Without this ordering, the transient WASM-heap
	 * footprint is `model_bytes + KV_bytes` simultaneously, which on
	 * 7B Q4 models exceeds the wasm64 16 GiB cap (minus browser/WebGPU
	 * overhead) and aborts inside `ctx_create` — see regression note in
	 * CLAUDE.md. On any throw inside this helper the staging is freed
	 * via the catch path before re-throwing, so callers never need to
	 * free `stagingPtr` themselves once handed in.
	 */
	private async _buildInferenceAndRegister(
		parsed: ParsedModel,
		ggufCtx: GgufContext,
		dataSrc: Uint8Array | ((offset: number, len: number) => Uint8Array),
		wasm: GgmlWasm,
		name: string,
		options?: Partial<ModelLoadOptions>,
		stagingPtr?: number,
		weightBytes?: number,
	): Promise<{
		handle: ModelHandle;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
		metadata: LoadedModelMetadata;
	}> {
		const freeStaging = (): void => {
			if (!stagingPtr) return;
			try {
				wasm.free(stagingPtr);
			} catch {
				// ignore; on the success path this is best-effort cleanup,
				// on the failure path the caller's `wasm.shutdown()` releases
				// all heap state anyway.
			}
			stagingPtr = 0;
		};

		try {
			const arch = parsed.hyperparams.architecture;
			const isEncoder = isEncoderArchitecture(arch);
			const isCausalEmbedder = isCausalEmbedderArchitecture(arch);
			let inference: ModelInference | EncoderInference | CausalLMEmbedder;
			if (isEncoder) {
				const enc = new EncoderInference(wasm, parsed.hyperparams);
				enc.loadWeights(ggufCtx, dataSrc);
				freeStaging();
				inference = enc;
			} else if (isCausalEmbedder) {
				const cembed = new CausalLMEmbedder(wasm, parsed.hyperparams);
				cembed.loadWeights(ggufCtx, dataSrc);
				freeStaging();
				inference = cembed;
			} else {
				const inf = new ModelInference(wasm, parsed.hyperparams, {
					flashAttn: !!options?.flashAttn,
				});
				inf.loadWeights(ggufCtx, dataSrc);
				// Free staging BEFORE initKVCache so the model-file-sized
				// region doesn't share the WASM heap with the ~1 GB KV cache
				// + scratch buffers ctx_create allocates.
				freeStaging();
				const requestedCtx = options?.contextLength;
				const ctxLen =
					typeof requestedCtx === "number" && requestedCtx > 0
						? Math.min(requestedCtx, parsed.kvCacheConfig.maxContextLength)
						: parsed.kvCacheConfig.maxContextLength;
				inf.initKVCache(ctxLen);
				inference = inf;
			}

			const handle = this.registerModelHandle(name, {
				priority: 0,
				...options,
			});
			const entry = this._modelManager.get(handle.id);
			if (entry) {
				entry.hyperparams = parsed.hyperparams;
				entry.tokenizer = new Tokenizer(parsed.tokenizerConfig);
				entry.kvCache = new KVCache(parsed.kvCacheConfig);
				entry.tokenizerHash = await computeTokenizerHash(
					parsed.tokenizerConfig,
				);
				entry.fingerprint = buildModelFingerprint(
					parsed.hyperparams,
					entry.tokenizerHash,
				);
				entry.loaded = true;
			}

			// Upsert the model record: wasm + exactly one engine-kind field.
			const record = this.models.get(handle.id) ?? {};
			record.wasm = wasm;
			if (inference instanceof EncoderInference) {
				record.encoder = inference;
			} else if (inference instanceof CausalLMEmbedder) {
				record.causalEmbedder = inference;
			} else {
				record.inference = inference;
			}
			this.models.set(handle.id, record);

			// ARC-002: record the model's weight footprint in the MemoryPool
			// so `ModelManager.canFit` enforces `WebLLMConfig.memoryBudget`
			// and the budget-pressure / eviction events can actually fire.
			// The GGUF byteLength is the closest cheap proxy for resident
			// GPU weight memory (dequant happens lazily; KV-cache accounting
			// is tracked separately as a follow-up ENH). Allocation is tagged
			// with `handle.id` so `_modelManager.unregister(id)` →
			// `memoryPool.evictModel(id)` (already invoked by `unloadModel`)
			// releases it without an explicit `free()` here.
			if (weightBytes !== undefined && weightBytes > 0) {
				this._memoryPool.allocate(
					weightBytes,
					options?.priority ?? 0,
					handle.id,
				);
			}

			// `parsed` is the loader-internal `ParsedModel`; its three fields
			// (hyperparams, tokenizerConfig, kvCacheConfig) are exactly the
			// public `LoadedModelMetadata` shape and are pure data — safe to
			// hand to a worker-mode caller across the postMessage boundary.
			const metadata: LoadedModelMetadata = {
				hyperparams: parsed.hyperparams,
				tokenizerConfig: parsed.tokenizerConfig,
				kvCacheConfig: parsed.kvCacheConfig,
			};
			return { handle, inference, metadata };
		} catch (e) {
			// Failure path inside the helper: free staging if we hadn't yet
			// (e.g., a throw during `loadWeights` or earlier — `freeStaging`
			// is idempotent because it nulls `stagingPtr` after freeing).
			// The caller is responsible for tearing down `wasm` itself
			// (shutdown / WebGPU device); we only own the staging ptr.
			freeStaging();
			throw e;
		}
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
		options?: Partial<ModelLoadOptions>,
	): Promise<{
		handle: ModelHandle;
		engine: WebLLM;
		inference: ModelInference | EncoderInference | CausalLMEmbedder;
		metadata: LoadedModelMetadata;
	}> {
		const engine = await WebLLM.init(config);
		const { handle, inference, metadata } = await engine.loadModelFromBuffer(
			data,
			name,
			wasmUrl,
			options,
		);
		return { handle, engine, inference, metadata };
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

	/**
	 * Build a {@link Character} persona bound to this engine.
	 *
	 * Characters carry a persistent system prompt, sampling config, and an
	 * optional tool surface; their `.chat()` method drives streaming
	 * completion through this engine's chat path. The character is
	 * registered with the engine's `CharacterManager` and can be removed
	 * via {@link removeCharacter}.
	 *
	 * @example
	 * ```ts
	 * const npc = engine.createCharacter({
	 *   modelId: handle.id,
	 *   systemPrompt: "You are a friendly shopkeeper.",
	 *   temperature: 0.7,
	 *   maxTokens: 256,
	 * });
	 * for await (const t of npc.chat("What do you sell?")) dialogueBox.addText(t);
	 * ```
	 */
	createCharacter(config: CharacterConfig): Character {
		// Inject ourselves as the engine unless the caller passed one
		// explicitly (e.g. a mock for testing).
		return this.characterManager.create({ engine: this, ...config });
	}

	getCharacter(id: string): Character | undefined {
		return this.characterManager.get(id);
	}

	/**
	 * Unregister and tear down a character previously created via
	 * {@link createCharacter}. Resolves once the character's resources are
	 * released; no-op (resolves without rejection) when `id` does not
	 * designate a live character.
	 */
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

	/**
	 * Release every resource held by this engine: dispose all loaded models
	 * (chat inference, encoders, causal-embedders, WASM modules), then clear
	 * the conversation pool, model manager, scheduler, and event handlers.
	 *
	 * After shutdown resolves the engine is unusable. Worker-mode callers
	 * additionally see `worker.terminate()` via the proxy's shutdown path.
	 * Safe to call from a `beforeunload` handler.
	 */
	async shutdown(): Promise<void> {
		this.conversationPool.clear();
		for (const record of this.models.values()) {
			const { inference, encoder, causalEmbedder, wasm } = record;
			if (inference) await inference.dispose();
			if (encoder) await encoder.dispose();
			if (causalEmbedder) await causalEmbedder.dispose();
			if (wasm) await wasm.shutdown();
		}
		this.models.clear();
		this._modelManager.clear();
		this._scheduler.clear();
		this.eventHandlers.clear();
	}
}
