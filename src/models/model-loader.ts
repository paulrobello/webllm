import {
	isCausalEmbedderArchitecture,
	isEncoderArchitecture,
	type ModelHyperparams,
} from "../core/types.js";
import {
	TokenAttribute,
	type TokenData,
	type TokenizerConfig,
	TokenizerType,
} from "../inference/tokenizer.js";
import { GgufParser } from "./gguf-parser.js";
import type { GgufContext } from "./gguf-types.js";
import type { KVCacheConfig } from "./kv-cache.js";

/** Result of parsing a GGUF model file. */
export interface ParsedModel {
	hyperparams: ModelHyperparams;
	tokenizerConfig: TokenizerConfig;
	kvCacheConfig: KVCacheConfig;
}

/**
 * Loads and parses GGUF model files, extracting hyperparameters, tokenizer
 * configuration, and KV cache configuration from GGUF metadata.
 *
 * The actual GPU weight loading is deferred to the WASM bridge integration.
 */
// biome-ignore lint/complexity/noStaticOnlyClass: instance methods planned for Phase 2
export class ModelLoader {
	/** Parse a GGUF model buffer into hyperparams, tokenizer config, and KV cache config. */
	static parseModel(data: Uint8Array): ParsedModel {
		const ctx = GgufParser.parse(data);
		const hyperparams = ModelLoader.extractHyperparams(ctx);
		const tokenizerConfig = ModelLoader.buildTokenizerConfig(ctx);
		// Fill vocabularySize from tokenizer config now that we know it
		hyperparams.vocabularySize = tokenizerConfig.vocabSize;
		tokenizerConfig.contextLength = hyperparams.contextLength;
		const kvCacheConfig = ModelLoader.buildKvCacheConfig(hyperparams);
		return { hyperparams, tokenizerConfig, kvCacheConfig };
	}

	/** Extract model hyperparameters from GGUF metadata. */
	private static extractHyperparams(ctx: GgufContext): ModelHyperparams {
		const rawArch = getMetaString(
			ctx,
			"general.architecture",
			"llama",
		) as ModelHyperparams["architecture"];

		// Derive qwen3-embedding from the qwen3 base arch when LAST-TOKEN pooling
		// is set in metadata (qwen3.pooling_type=3). The Qwen3-Embedding GGUFs
		// share the qwen3 architecture string but carry a pooling_type that the
		// chat Qwen3 GGUFs do not. See bucket C Phase 0 probe report.
		const qwen3PoolingRaw =
			rawArch === "qwen3"
				? getMetaNumberOptional(ctx, "qwen3.pooling_type")
				: undefined;
		const arch: ModelHyperparams["architecture"] =
			rawArch === "qwen3" && qwen3PoolingRaw === 3
				? "qwen3-embedding"
				: rawArch;

		const embeddingLength = getMetaNumber(
			ctx,
			`${arch}.embedding_length`,
			4096,
		);
		const headCount = getMetaNumber(ctx, `${arch}.attention.head_count`, 32);

		// BERT-family encoders use a plain LayerNorm epsilon under a different
		// metadata key. Fall back to the RMSNorm key for non-encoder archs.
		const normEpsilon = isEncoderArchitecture(arch)
			? getMetaFloat(ctx, `${arch}.attention.layer_norm_epsilon`, 1e-12)
			: getMetaFloat(ctx, `${arch}.attention.layer_norm_rms_epsilon`, 1e-5);

		// Pooling + causal flag live on encoder models; causal-LM embedders
		// (e.g. qwen3-embedding) carry pooling_type=LAST. Causal defaults true elsewhere.
		let poolingType: ModelHyperparams["poolingType"];
		let causalAttention: boolean | undefined;
		let alibiMaxBias: number | undefined;
		if (isEncoderArchitecture(arch)) {
			const pt = getMetaNumberOptional(ctx, `${arch}.pooling_type`) ?? 2;
			// llama.cpp enum: NONE=0, MEAN=1, CLS=2, LAST=3, RANK=4. We only
			// implement CLS and MEAN for encoders; anything else falls back to CLS.
			poolingType = pt === 1 ? "mean" : "cls";
			causalAttention =
				getMetaBooleanOptional(ctx, `${arch}.attention.causal`) ?? false;
			if (arch === "jina-bert-v2") {
				// gaianet GGUF mirror omits this key; 8.0 is the upstream default
				// (jina-bert-v2 reference impl + llama.cpp).
				alibiMaxBias =
					getMetaNumberOptional(ctx, `${arch}.attention.alibi_bias_max`) ?? 8.0;
			}
		} else if (isCausalEmbedderArchitecture(arch)) {
			// llama.cpp enum: LAST=3. Hard-pin "last-token" for the causal-LM-derived
			// embedder family — no other pooling mode is supported for them.
			poolingType = "last-token";
		}

		return {
			architecture: arch,
			contextLength: getMetaNumber(ctx, `${arch}.context_length`, 2048),
			embeddingLength,
			headCount,
			headCountKv: getMetaNumber(
				ctx,
				`${arch}.attention.head_count_kv`,
				headCount,
			),
			layerCount: getMetaNumber(ctx, `${arch}.block_count`, 32),
			vocabularySize: 0, // filled after tokenizer parse
			embeddingHeadLength: getMetaNumber(
				ctx,
				`${arch}.attention.key_length`,
				embeddingLength / headCount,
			),
			feedForwardLength: getMetaNumber(
				ctx,
				`${arch}.feed_forward_length`,
				11008,
			),
			ropeFreqBase:
				getMetaNumberOptional(ctx, `${arch}.rope_freq_base`) ??
				getMetaNumberOptional(ctx, `${arch}.rope.freq_base`) ??
				10000,
			ropeScale: getMetaNumber(ctx, `${arch}.rope_scale`, 1),
			normEpsilon,
			expertCount: getMetaNumber(ctx, `${arch}.expert_count`, 0),
			expertUsedCount: getMetaNumber(ctx, `${arch}.expert_used_count`, 0),
			poolingType,
			causalAttention,
			alibiMaxBias,
		};
	}

	/** Build tokenizer configuration from GGUF metadata. */
	private static buildTokenizerConfig(ctx: GgufContext): TokenizerConfig {
		const modelStr = getMetaString(ctx, "tokenizer.ggml.model", "llama");
		const type =
			modelStr === "gpt2"
				? TokenizerType.BPE
				: modelStr === "bert"
					? TokenizerType.WORDPIECE
					: TokenizerType.SPM;

		const rawTokens = getMetaStringArray(ctx, "tokenizer.ggml.tokens", []);
		const rawScores = getMetaNumberArray(ctx, "tokenizer.ggml.scores", []);
		const rawTokenTypes = getMetaNumberArray(
			ctx,
			"tokenizer.ggml.token_type",
			[],
		);
		const rawMerges = getMetaStringArray(ctx, "tokenizer.ggml.merges", []);

		const tokens: TokenData[] = rawTokens.map((text, i) => ({
			text,
			score: rawScores[i] ?? 0,
			attr: mapGgufTokenTypeToTokenizerAttr(rawTokenTypes[i] ?? 0),
		}));

		const eosTokenId = getMetaNumber(ctx, "tokenizer.ggml.eos_token_id", 2);
		const bosTokenId = getMetaNumber(ctx, "tokenizer.ggml.bos_token_id", 1);
		const padTokenId = getMetaNumberOptional(
			ctx,
			"tokenizer.ggml.padding_token_id",
		);
		const preTokenizer = getMetaStringOptional(ctx, "tokenizer.ggml.pre");
		const addBosToken = getMetaBooleanOptional(
			ctx,
			"tokenizer.ggml.add_bos_token",
		);

		const addedTokens = new Map<string, number>();

		// GGUF tokenizer.ggml.token_type uses llama.cpp vocab types where
		// CONTROL=3 and USER_DEFINED=4. Both should be treated as special tokens
		// that bypass normal subword tokenization.
		for (let i = 0; i < tokens.length && i < rawTokenTypes.length; i++) {
			if (rawTokenTypes[i] === 3 || rawTokenTypes[i] === 4) {
				addedTokens.set(tokens[i].text, i);
			}
		}

		// Always ensure EOS and BOS are in addedTokens regardless of token_type
		if (eosTokenId >= 0 && eosTokenId < tokens.length) {
			addedTokens.set(tokens[eosTokenId].text, eosTokenId);
		}
		if (bosTokenId >= 0 && bosTokenId < tokens.length) {
			addedTokens.set(tokens[bosTokenId].text, bosTokenId);
		}

		const bpeRanks = new Map<string, number>();
		if (type === TokenizerType.BPE) {
			for (let i = 0; i < rawMerges.length; i++) {
				bpeRanks.set(rawMerges[i], i);
			}
		}

		// BERT-family WordPiece convention: [CLS] = bos and [SEP] = eos.
		// Some BERT-family GGUFs (e.g. nomic-embed-text-v1.5) omit
		// `cls_token_id` and rely on the bos/eos fallback. Mirroring this
		// fallback keeps WordPiece initialization working across all
		// BERT-family encoders without per-model special cases.
		const clsTokenId =
			getMetaNumberOptional(ctx, "tokenizer.ggml.cls_token_id") ??
			(type === TokenizerType.WORDPIECE && bosTokenId >= 0
				? bosTokenId
				: undefined);
		// GGUF key is misspelled "seperator" upstream (llama.cpp + Arctic-Embed
		// GGUFs); do NOT correct to "separator" or bert metadata reads will fail.
		const sepTokenId =
			getMetaNumberOptional(ctx, "tokenizer.ggml.seperator_token_id") ??
			(type === TokenizerType.WORDPIECE && eosTokenId >= 0
				? eosTokenId
				: undefined);
		const unkTokenId = getMetaNumberOptional(
			ctx,
			"tokenizer.ggml.unknown_token_id",
		);
		const maskTokenId = getMetaNumberOptional(
			ctx,
			"tokenizer.ggml.mask_token_id",
		);

		return {
			type,
			tokens,
			bpeRanks,
			addedTokens,
			eosTokenId,
			bosTokenId,
			padTokenId: padTokenId ?? -1,
			vocabSize: tokens.length,
			chatTemplate: getMetaString(ctx, "tokenizer.chat_template", ""),
			preTokenizer,
			addBosToken,
			clsTokenId,
			sepTokenId,
			unkTokenId,
			maskTokenId,
		};
	}

	/** Build KV cache configuration from model hyperparameters. */
	private static buildKvCacheConfig(hp: ModelHyperparams): KVCacheConfig {
		return {
			nLayers: hp.layerCount,
			nEmbdHeadK: hp.embeddingHeadLength,
			nEmbdHeadV: hp.embeddingHeadLength,
			nKvHead: hp.headCountKv,
			maxContextLength: hp.contextLength,
			dataType: "f32",
		};
	}
}

function mapGgufTokenTypeToTokenizerAttr(rawType: number): TokenAttribute {
	switch (rawType) {
		case 1:
			return TokenAttribute.NORMAL;
		case 2:
			return TokenAttribute.UNKNOWN;
		case 3:
			return TokenAttribute.CONTROL;
		case 4:
			return TokenAttribute.USER_DEFINED;
		case 6:
			return TokenAttribute.BYTE;
		default:
			return 0 as TokenAttribute;
	}
}

/** Get a string metadata value or default. */
function getMetaString(
	ctx: GgufContext,
	key: string,
	defaultValue: string,
): string {
	const kv = ctx.metadata.get(key);
	if (kv && typeof kv.value === "string") return kv.value;
	return defaultValue;
}

/** Get a numeric metadata value or default. */
function getMetaNumber(
	ctx: GgufContext,
	key: string,
	defaultValue: number,
): number {
	const kv = ctx.metadata.get(key);
	if (kv && typeof kv.value === "number") return kv.value;
	return defaultValue;
}

/** Get a numeric metadata value if present. */
function getMetaNumberOptional(
	ctx: GgufContext,
	key: string,
): number | undefined {
	const kv = ctx.metadata.get(key);
	if (kv && typeof kv.value === "number") return kv.value;
	return undefined;
}

function getMetaStringOptional(
	ctx: GgufContext,
	key: string,
): string | undefined {
	const kv = ctx.metadata.get(key);
	if (kv && typeof kv.value === "string") return kv.value;
	return undefined;
}

function getMetaBooleanOptional(
	ctx: GgufContext,
	key: string,
): boolean | undefined {
	const kv = ctx.metadata.get(key);
	if (kv && typeof kv.value === "boolean") return kv.value;
	return undefined;
}

/** Get a float metadata value (covers both FLOAT32 and FLOAT64 stored values). */
function getMetaFloat(
	ctx: GgufContext,
	key: string,
	defaultValue: number,
): number {
	return getMetaNumber(ctx, key, defaultValue);
}

/** Get a string array metadata value or default. */
function getMetaStringArray(
	ctx: GgufContext,
	key: string,
	defaultValue: string[],
): string[] {
	const kv = ctx.metadata.get(key);
	if (kv && Array.isArray(kv.value)) return kv.value as string[];
	return defaultValue;
}

/** Get a number array metadata value or default. */
function getMetaNumberArray(
	ctx: GgufContext,
	key: string,
	defaultValue: number[],
): number[] {
	const kv = ctx.metadata.get(key);
	if (kv && Array.isArray(kv.value)) return kv.value as number[];
	return defaultValue;
}
