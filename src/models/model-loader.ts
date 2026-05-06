import {
	isCausalEmbedderArchitecture,
	isEncoderArchitecture,
	type ModelHyperparams,
} from "../core/types.js";
import type { LlamaBridge } from "../inference/llama-bridge.js";
import type { KVCacheConfig } from "./kv-cache.js";

/** Result of loading + introspecting a GGUF model via the bridge. */
export interface LoadedModelMetadata {
	/** Bridge handle to the upstream llama_model. Owned by the caller. */
	model: number;
	hyperparams: ModelHyperparams;
	kvCacheConfig: KVCacheConfig;
	/** Chat template string, or "" if missing. */
	chatTemplate: string;
}

/**
 * Load a GGUF buffer through the bridge and pull metadata back into
 * the {@link ModelHyperparams} shape engine.ts already consumes. The
 * caller owns the returned model handle and must call
 * `bridge.freeModel(metadata.model)` on disposal.
 *
 * In the legacy path, `ModelLoader.parseModel` re-implemented every
 * GGUF header field in TS. Upstream's parser is the source of truth
 * for everything llama.cpp supports — this thin wrapper surfaces it.
 */
export async function loadModelMetadata(
	bridge: LlamaBridge,
	data: Uint8Array,
): Promise<LoadedModelMetadata> {
	const model = await bridge.loadModel(data);
	try {
		const archStr =
			bridge.getMetadata(model, "general.architecture") ?? "llama";
		const metaPrefix = archStr;

		// Same Qwen3-Embedding derivation rule as the legacy loader:
		// pooling_type=3 (LAST) on a qwen3 model means it's the embedding
		// variant, not the chat variant. Surfaces via metadata so the
		// rule stays load-bearing post-migration.
		const qwen3PoolingRaw =
			archStr === "qwen3"
				? Number(bridge.getMetadata(model, "qwen3.pooling_type") ?? "")
				: Number.NaN;
		const arch: ModelHyperparams["architecture"] =
			archStr === "qwen3" && qwen3PoolingRaw === 3
				? "qwen3-embedding"
				: (archStr as ModelHyperparams["architecture"]);

		const embeddingLength = bridge.nEmbd(model);
		const headCount = bridge.nHead(model);
		const headCountKv = bridge.nHeadKv(model);
		const layerCount = bridge.nLayer(model);
		const contextLength = bridge.nCtxTrain(model);

		const ftypeRaw = bridge.getMetadata(model, "general.file_type");
		const quantType =
			ftypeRaw !== null ? mapFtypeToQuantName(Number(ftypeRaw)) : "unknown";

		// Norm epsilon: encoders use layer_norm_epsilon, others use
		// the RMSNorm key. Same dispatch as the legacy loader.
		const normEpsilonStr = isEncoderArchitecture(arch)
			? bridge.getMetadata(model, `${metaPrefix}.attention.layer_norm_epsilon`)
			: bridge.getMetadata(
					model,
					`${metaPrefix}.attention.layer_norm_rms_epsilon`,
				);
		const normEpsilon =
			normEpsilonStr !== null
				? Number(normEpsilonStr)
				: isEncoderArchitecture(arch)
					? 1e-12
					: 1e-5;

		// Pooling + causal flag for encoders / causal-LM-derived embedders.
		let poolingType: ModelHyperparams["poolingType"];
		let causalAttention: boolean | undefined;
		let alibiMaxBias: number | undefined;
		if (isEncoderArchitecture(arch)) {
			const ptStr = bridge.getMetadata(model, `${metaPrefix}.pooling_type`);
			const pt = ptStr !== null ? Number(ptStr) : 2;
			poolingType = pt === 1 ? "mean" : "cls";
			const causalStr = bridge.getMetadata(
				model,
				`${metaPrefix}.attention.causal`,
			);
			causalAttention =
				causalStr === null ? false : causalStr.toLowerCase() === "true";
			if (arch === "jina-bert-v2") {
				const alibiStr = bridge.getMetadata(
					model,
					`${metaPrefix}.attention.alibi_bias_max`,
				);
				alibiMaxBias = alibiStr !== null ? Number(alibiStr) : 8.0;
			}
		} else if (isCausalEmbedderArchitecture(arch)) {
			poolingType = "last-token";
		}

		const hyperparams: ModelHyperparams = {
			architecture: arch,
			contextLength,
			embeddingLength,
			headCount,
			headCountKv,
			layerCount,
			vocabularySize: bridge.nVocab(model),
			embeddingHeadLength: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.attention.key_length`,
				Math.floor(embeddingLength / Math.max(1, headCount)),
			),
			feedForwardLength: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.feed_forward_length`,
				11008,
			),
			ropeFreqBase:
				numFromMetaOptional(bridge, model, `${metaPrefix}.rope_freq_base`) ??
				numFromMetaOptional(bridge, model, `${metaPrefix}.rope.freq_base`) ??
				10000,
			ropeScale: numFromMeta(bridge, model, `${metaPrefix}.rope_scale`, 1),
			normEpsilon,
			expertCount: numFromMeta(bridge, model, `${metaPrefix}.expert_count`, 0),
			expertUsedCount: numFromMeta(
				bridge,
				model,
				`${metaPrefix}.expert_used_count`,
				0,
			),
			quantType,
			poolingType,
			causalAttention,
			alibiMaxBias,
		};

		const kvCacheConfig: KVCacheConfig = {
			nLayers: hyperparams.layerCount,
			nEmbdHeadK: hyperparams.embeddingHeadLength,
			nEmbdHeadV: hyperparams.embeddingHeadLength,
			nKvHead: hyperparams.headCountKv,
			maxContextLength: hyperparams.contextLength,
			dataType: "f32",
		};

		const chatTemplate =
			bridge.getMetadata(model, "tokenizer.chat_template") ?? "";

		return { model, hyperparams, kvCacheConfig, chatTemplate };
	} catch (err) {
		bridge.freeModel(model);
		throw err;
	}
}

function numFromMeta(
	bridge: LlamaBridge,
	model: number,
	key: string,
	fallback: number,
): number {
	const v = bridge.getMetadata(model, key);
	if (v === null) return fallback;
	const n = Number(v);
	return Number.isFinite(n) ? n : fallback;
}

function numFromMetaOptional(
	bridge: LlamaBridge,
	model: number,
	key: string,
): number | undefined {
	const v = bridge.getMetadata(model, key);
	if (v === null) return undefined;
	const n = Number(v);
	return Number.isFinite(n) ? n : undefined;
}

// Mirrors llama.cpp `enum llama_ftype` (llama.h). Stable since 2024.
const LLAMA_FTYPE_NAMES: Readonly<Record<number, string>> = {
	0: "F32",
	1: "F16",
	2: "Q4_0",
	3: "Q4_1",
	7: "Q8_0",
	8: "Q5_0",
	9: "Q5_1",
	10: "Q2_K",
	11: "Q3_K_S",
	12: "Q3_K_M",
	13: "Q3_K_L",
	14: "Q4_K_S",
	15: "Q4_K_M",
	16: "Q5_K_S",
	17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS",
	20: "IQ2_XS",
	21: "Q2_K_S",
	22: "IQ3_XS",
	23: "IQ3_XXS",
	24: "IQ1_S",
	25: "IQ4_NL",
	26: "IQ3_S",
	27: "IQ3_M",
	28: "IQ2_S",
	29: "IQ2_M",
	30: "IQ4_XS",
	31: "IQ1_M",
	32: "BF16",
	36: "TQ1_0",
	37: "TQ2_0",
};

function mapFtypeToQuantName(ftype: number): string {
	return LLAMA_FTYPE_NAMES[ftype] ?? "unknown";
}

export type { ParsedModel } from "./model-loader-legacy.js";
/**
 * Legacy compatibility shim. Encoder + causal-embedder loaders still
 * construct `ParsedModel` from a Uint8Array; they're rewritten in
 * P3/P4. Re-exporting the legacy API surface here lets P2 land
 * without touching them. Direct test consumers
 * (tests/model-loader-bpe, kv-snapshot-roundtrip,
 * forward-verify-equivalence, encoder-inference,
 * chat-template-special-tokens) also resolve through this re-export.
 */
export { ModelLoader } from "./model-loader-legacy.js";
