import type { ModelHyperparams } from "../core/types.js";
import {
	type TokenAttribute,
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
	static parseModel(data: ArrayBuffer): ParsedModel {
		const ctx = GgufParser.parse(data);
		const hyperparams = ModelLoader.extractHyperparams(ctx);
		const tokenizerConfig = ModelLoader.buildTokenizerConfig(ctx);
		// Fill vocabularySize from tokenizer config now that we know it
		hyperparams.vocabularySize = tokenizerConfig.vocabSize;
		const kvCacheConfig = ModelLoader.buildKvCacheConfig(hyperparams);
		return { hyperparams, tokenizerConfig, kvCacheConfig };
	}

	/** Extract model hyperparameters from GGUF metadata. */
	private static extractHyperparams(ctx: GgufContext): ModelHyperparams {
		const arch = getMetaString(
			ctx,
			"general.architecture",
			"llama",
		) as ModelHyperparams["architecture"];

		const embeddingLength = getMetaNumber(
			ctx,
			`${arch}.embedding_length`,
			4096,
		);
		const headCount = getMetaNumber(ctx, `${arch}.attention.head_count`, 32);

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
			ropeFreqBase: getMetaNumber(ctx, `${arch}.rope_freq_base`, 10000),
			ropeScale: getMetaNumber(ctx, `${arch}.rope_scale`, 1),
			normEpsilon: getMetaFloat(
				ctx,
				`${arch}.attention.layer_norm_rms_epsilon`,
				1e-5,
			),
			expertCount: getMetaNumber(ctx, `${arch}.expert_count`, 0),
			expertUsedCount: getMetaNumber(ctx, `${arch}.expert_used_count`, 0),
		};
	}

	/** Build tokenizer configuration from GGUF metadata. */
	private static buildTokenizerConfig(ctx: GgufContext): TokenizerConfig {
		const modelStr = getMetaString(ctx, "tokenizer.ggml.model", "llama");
		const type = modelStr === "gpt2" ? TokenizerType.BPE : TokenizerType.SPM;

		const rawTokens = getMetaStringArray(ctx, "tokenizer.ggml.tokens", []);
		const rawScores = getMetaNumberArray(ctx, "tokenizer.ggml.scores", []);
		const rawTokenTypes = getMetaNumberArray(
			ctx,
			"tokenizer.ggml.token_type",
			[],
		);

		const tokens: TokenData[] = rawTokens.map((text, i) => ({
			text,
			score: rawScores[i] ?? 0,
			attr: (rawTokenTypes[i] ?? 0) as TokenAttribute,
		}));

		const eosTokenId = getMetaNumber(ctx, "tokenizer.ggml.eos_token_id", 2);
		const bosTokenId = getMetaNumber(ctx, "tokenizer.ggml.bos_token_id", 1);

		return {
			type,
			tokens,
			bpeRanks: new Map(),
			addedTokens: new Map(),
			eosTokenId,
			bosTokenId,
			padTokenId: -1,
			vocabSize: tokens.length,
			chatTemplate: getMetaString(ctx, "tokenizer.chat_template", ""),
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
