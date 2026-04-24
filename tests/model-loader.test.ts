import { describe, expect, test } from "bun:test";
import type { ModelArchitecture, ModelHyperparams } from "../src/core/types.js";
import { TokenizerType } from "../src/inference/tokenizer.js";
import {
	GGUF_MAGIC,
	GGUF_VERSION,
	GgufValueType,
} from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

function writeString(buf: DataView, offset: number, str: string): number {
	buf.setBigUint64(offset, BigInt(str.length), true);
	offset += 8;
	for (let i = 0; i < str.length; i++) {
		buf.setUint8(offset++, str.charCodeAt(i));
	}
	return offset;
}

function writeUint32(buf: DataView, offset: number, val: number): number {
	buf.setUint32(offset, val, true);
	return offset + 4;
}

function writeKvString(
	buf: DataView,
	offset: number,
	key: string,
	value: string,
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.STRING as number);
	offset = writeString(buf, offset, value);
	return offset;
}

function writeKvUint32(
	buf: DataView,
	offset: number,
	key: string,
	value: number,
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.UINT32 as number);
	offset = writeUint32(buf, offset, value);
	return offset;
}

function writeKvFloat32(
	buf: DataView,
	offset: number,
	key: string,
	value: number,
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.FLOAT32 as number);
	buf.setFloat32(offset, value, true);
	return offset + 4;
}

function writeKvStringArray(
	buf: DataView,
	offset: number,
	key: string,
	values: string[],
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.ARRAY as number);
	offset = writeUint32(buf, offset, GgufValueType.STRING as number);
	buf.setBigUint64(offset, BigInt(values.length), true);
	offset += 8;
	for (const v of values) {
		offset = writeString(buf, offset, v);
	}
	return offset;
}

function writeKvFloatArray(
	buf: DataView,
	offset: number,
	key: string,
	values: number[],
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.ARRAY as number);
	offset = writeUint32(buf, offset, GgufValueType.FLOAT32 as number);
	buf.setBigUint64(offset, BigInt(values.length), true);
	offset += 8;
	for (const v of values) {
		buf.setFloat32(offset, v, true);
		offset += 4;
	}
	return offset;
}

function writeKvIntArray(
	buf: DataView,
	offset: number,
	key: string,
	values: number[],
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.ARRAY as number);
	offset = writeUint32(buf, offset, GgufValueType.INT32 as number);
	buf.setBigUint64(offset, BigInt(values.length), true);
	offset += 8;
	for (const v of values) {
		buf.setInt32(offset, v, true);
		offset += 4;
	}
	return offset;
}

/**
 * Build a minimal GGUF buffer with all metadata keys required by ModelLoader.
 * Includes architecture hyperparams and tokenizer config fields.
 */
function buildModelLoaderGguf(): ArrayBuffer {
	const headerSize = 24;
	const arch = "llama";

	// Tokenizer data
	const tokens = ["<unk>", "<s>", "</s>", "a", "b", "c"];
	const scores = [0.0, 0.0, 0.0, -1.0, -1.0, -1.0];
	const tokenTypes = [2, 3, 3, 1, 1, 1]; // UNKNOWN, CONTROL, CONTROL, NORMAL, ...

	// Build all KV entries
	const kvEntries: Array<{ key: string; type: GgufValueType; value: unknown }> =
		[
			{ key: "general.architecture", type: GgufValueType.STRING, value: arch },
			{
				key: `${arch}.context_length`,
				type: GgufValueType.UINT32,
				value: 4096,
			},
			{
				key: `${arch}.embedding_length`,
				type: GgufValueType.UINT32,
				value: 4096,
			},
			{
				key: `${arch}.attention.head_count`,
				type: GgufValueType.UINT32,
				value: 32,
			},
			{
				key: `${arch}.attention.head_count_kv`,
				type: GgufValueType.UINT32,
				value: 32,
			},
			{ key: `${arch}.block_count`, type: GgufValueType.UINT32, value: 32 },
			{
				key: `${arch}.attention.layer_norm_rms_epsilon`,
				type: GgufValueType.FLOAT32,
				value: 1e-5,
			},
			{
				key: `${arch}.feed_forward_length`,
				type: GgufValueType.UINT32,
				value: 11008,
			},
			{
				key: `${arch}.attention.key_length`,
				type: GgufValueType.UINT32,
				value: 128,
			},
			{
				key: `${arch}.attention.value_length`,
				type: GgufValueType.UINT32,
				value: 128,
			},
			{
				key: "tokenizer.ggml.model",
				type: GgufValueType.STRING,
				value: "llama",
			},
			{
				key: "tokenizer.ggml.tokens",
				type: GgufValueType.ARRAY,
				value: tokens,
			},
			{
				key: "tokenizer.ggml.scores",
				type: GgufValueType.ARRAY,
				value: scores,
			},
			{
				key: "tokenizer.ggml.token_type",
				type: GgufValueType.ARRAY,
				value: tokenTypes,
			},
			{
				key: "tokenizer.ggml.eos_token_id",
				type: GgufValueType.UINT32,
				value: 2,
			},
			{
				key: "tokenizer.ggml.bos_token_id",
				type: GgufValueType.UINT32,
				value: 1,
			},
		];

	// Calculate buffer size
	// Header: 24 bytes
	// Each KV entry: key string (8 + len) + type(4) + value
	let kvTotalSize = 0;
	for (const entry of kvEntries) {
		kvTotalSize += 8 + entry.key.length + 4; // key string + type
		if (entry.type === GgufValueType.STRING) {
			kvTotalSize += 8 + (entry.value as string).length;
		} else if (
			entry.type === GgufValueType.UINT32 ||
			entry.type === GgufValueType.FLOAT32
		) {
			kvTotalSize += 4;
		} else if (entry.type === GgufValueType.ARRAY) {
			const arr = entry.value as unknown[];
			// Determine element type
			const firstVal = arr[0];
			if (typeof firstVal === "string") {
				kvTotalSize += 4 + 8; // elem type + count
				for (const v of arr as string[]) kvTotalSize += 8 + v.length;
			} else if (typeof firstVal === "number") {
				// Use FLOAT32 for score arrays, INT32 for token_type arrays
				if (entry.key.includes("scores")) {
					kvTotalSize += 4 + 8; // elem type + count
					kvTotalSize += arr.length * 4;
				} else {
					kvTotalSize += 4 + 8; // elem type + count
					kvTotalSize += arr.length * 4;
				}
			}
		}
	}

	// One dummy tensor: name + nDims(4) + dim(8) + type(4) + offset(8)
	const tensorName = "output.weight";
	const tensorInfoSize = 8 + tensorName.length + 4 + 8 + 4 + 8;
	const totalSize = headerSize + kvTotalSize + tensorInfoSize + 64;
	const buf = new ArrayBuffer(totalSize);
	const view = new DataView(buf);
	let offset = 0;

	// Write header
	view.setUint32(offset, GGUF_MAGIC, true);
	offset += 4;
	view.setUint32(offset, GGUF_VERSION, true);
	offset += 4;
	view.setBigUint64(offset, BigInt(1), true); // tensorCount
	offset += 8;
	view.setBigUint64(offset, BigInt(kvEntries.length), true);
	offset += 8;

	// Write KV entries
	for (const entry of kvEntries) {
		if (entry.type === GgufValueType.STRING) {
			offset = writeKvString(view, offset, entry.key, entry.value as string);
		} else if (entry.type === GgufValueType.UINT32) {
			offset = writeKvUint32(view, offset, entry.key, entry.value as number);
		} else if (entry.type === GgufValueType.FLOAT32) {
			offset = writeKvFloat32(view, offset, entry.key, entry.value as number);
		} else if (entry.type === GgufValueType.ARRAY) {
			const arr = entry.value as unknown[];
			const firstVal = arr[0];
			if (typeof firstVal === "string") {
				offset = writeKvStringArray(view, offset, entry.key, arr as string[]);
			} else if (typeof firstVal === "number") {
				if (entry.key.includes("scores")) {
					offset = writeKvFloatArray(view, offset, entry.key, arr as number[]);
				} else {
					offset = writeKvIntArray(view, offset, entry.key, arr as number[]);
				}
			}
		}
	}

	// Write dummy tensor info
	offset = writeString(view, offset, tensorName);
	offset = writeUint32(view, offset, 1); // nDimensions
	view.setBigUint64(offset, BigInt(6), true); // dimension
	offset += 8;
	offset = writeUint32(view, offset, 0); // type = f32
	view.setBigUint64(offset, BigInt(0), true); // offset
	offset += 8;

	return buf.slice(0, offset);
}

describe("ModelLoader", () => {
	test("extracts hyperparams from GGUF metadata", () => {
		const buf = buildModelLoaderGguf();
		const parsed = ModelLoader.parseModel(buf);

		expect(parsed.hyperparams.architecture).toBe("llama");
		expect(parsed.hyperparams.contextLength).toBe(4096);
		expect(parsed.hyperparams.embeddingLength).toBe(4096);
		expect(parsed.hyperparams.headCount).toBe(32);
		expect(parsed.hyperparams.headCountKv).toBe(32);
		expect(parsed.hyperparams.layerCount).toBe(32);
		expect(parsed.hyperparams.feedForwardLength).toBe(11008);
		expect(parsed.hyperparams.embeddingHeadLength).toBe(128);
		expect(parsed.hyperparams.vocabularySize).toBe(6);
		expect(parsed.hyperparams.normEpsilon).toBeCloseTo(1e-5);
	});

	test("builds tokenizer config from GGUF metadata", () => {
		const buf = buildModelLoaderGguf();
		const parsed = ModelLoader.parseModel(buf);

		expect(parsed.tokenizerConfig.type).toBe(0); // SPM
		expect(parsed.tokenizerConfig.vocabSize).toBe(6);
		expect(parsed.tokenizerConfig.bosTokenId).toBe(1);
		expect(parsed.tokenizerConfig.eosTokenId).toBe(2);
		expect(parsed.tokenizerConfig.tokens).toHaveLength(6);
		expect(parsed.tokenizerConfig.tokens[0].text).toBe("<unk>");
		expect(parsed.tokenizerConfig.tokens[1].text).toBe("<s>");
	});

	test("builds KV cache config from hyperparams", () => {
		const buf = buildModelLoaderGguf();
		const parsed = ModelLoader.parseModel(buf);

		expect(parsed.kvCacheConfig.nLayers).toBe(32);
		expect(parsed.kvCacheConfig.nKvHead).toBe(32);
		expect(parsed.kvCacheConfig.maxContextLength).toBe(4096);
		expect(parsed.kvCacheConfig.nEmbdHeadK).toBe(128);
		expect(parsed.kvCacheConfig.nEmbdHeadV).toBe(128);
		expect(parsed.kvCacheConfig.dataType).toBe("f32");
	});
});

describe("ModelArchitecture union", () => {
	test("includes bert", () => {
		const arch: ModelArchitecture = "bert";
		expect(arch).toBe("bert");
	});
	test("ModelHyperparams exposes poolingType", () => {
		const hp: ModelHyperparams = {
			architecture: "bert",
			contextLength: 512,
			embeddingLength: 384,
			headCount: 12,
			headCountKv: 12,
			layerCount: 12,
			vocabularySize: 30522,
			embeddingHeadLength: 32,
			feedForwardLength: 1536,
			ropeFreqBase: 10000,
			ropeScale: 1,
			normEpsilon: 1e-12,
			expertCount: 0,
			expertUsedCount: 0,
			poolingType: "cls",
			causalAttention: false,
		};
		expect(hp.poolingType).toBe("cls");
	});
});

function writeKvBool(
	buf: DataView,
	offset: number,
	key: string,
	value: boolean,
): number {
	offset = writeString(buf, offset, key);
	offset = writeUint32(buf, offset, GgufValueType.BOOL as number);
	buf.setUint8(offset, value ? 1 : 0);
	return offset + 1;
}

interface BertGgufOptions {
	/** When provided, written as `bert.pooling_type` (UINT32). Omit to leave key absent. */
	poolingType?: number;
	/** When provided, written as `bert.attention.causal` (BOOL). Omit to leave key absent. */
	causal?: boolean;
	/** When provided, overrides the default `bert.attention.layer_norm_epsilon`. */
	normEpsilon?: number;
	/**
	 * When true, emits a bert-style tokenizer vocab with CLS/SEP/UNK/BOS/EOS
	 * token-id metadata keys. `mask_token_id` is intentionally omitted so the
	 * loader's `undefined` fallback can be exercised.
	 */
	withBertTokenizer?: boolean;
}

/**
 * Build a minimal GGUF buffer with `general.architecture = "bert"` plus the
 * subset of bert.* hyperparam keys needed by ModelLoader.parseModel, with
 * configurable pooling_type / attention.causal / layer_norm_epsilon entries.
 */
function buildBertGguf(options: BertGgufOptions = {}): ArrayBuffer {
	const headerSize = 24;
	const arch = "bert";
	const tokens = options.withBertTokenizer
		? ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"]
		: ["[PAD]", "[CLS]", "[SEP]", "a", "b", "c"];
	const scores = options.withBertTokenizer
		? [0.0, 0.0, 0.0, 0.0, -1.0]
		: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
	const tokenTypes = options.withBertTokenizer
		? [3, 3, 3, 3, 1]
		: [3, 3, 3, 1, 1, 1];
	const bosTokenId = options.withBertTokenizer ? 2 : 1;
	const eosTokenId = options.withBertTokenizer ? 3 : 2;

	type KvEntry = { key: string; type: GgufValueType; value: unknown };
	const kvEntries: KvEntry[] = [
		{ key: "general.architecture", type: GgufValueType.STRING, value: arch },
		{ key: `${arch}.context_length`, type: GgufValueType.UINT32, value: 512 },
		{
			key: `${arch}.embedding_length`,
			type: GgufValueType.UINT32,
			value: 384,
		},
		{
			key: `${arch}.attention.head_count`,
			type: GgufValueType.UINT32,
			value: 12,
		},
		{
			key: `${arch}.attention.head_count_kv`,
			type: GgufValueType.UINT32,
			value: 12,
		},
		{ key: `${arch}.block_count`, type: GgufValueType.UINT32, value: 12 },
		{
			key: `${arch}.attention.layer_norm_epsilon`,
			type: GgufValueType.FLOAT32,
			value: options.normEpsilon ?? 1e-12,
		},
		{
			key: `${arch}.feed_forward_length`,
			type: GgufValueType.UINT32,
			value: 1536,
		},
		{
			key: `${arch}.attention.key_length`,
			type: GgufValueType.UINT32,
			value: 32,
		},
		{
			key: "tokenizer.ggml.model",
			type: GgufValueType.STRING,
			value: "bert",
		},
		{ key: "tokenizer.ggml.tokens", type: GgufValueType.ARRAY, value: tokens },
		{ key: "tokenizer.ggml.scores", type: GgufValueType.ARRAY, value: scores },
		{
			key: "tokenizer.ggml.token_type",
			type: GgufValueType.ARRAY,
			value: tokenTypes,
		},
		{
			key: "tokenizer.ggml.eos_token_id",
			type: GgufValueType.UINT32,
			value: eosTokenId,
		},
		{
			key: "tokenizer.ggml.bos_token_id",
			type: GgufValueType.UINT32,
			value: bosTokenId,
		},
	];

	if (options.withBertTokenizer) {
		kvEntries.push(
			{
				key: "tokenizer.ggml.cls_token_id",
				type: GgufValueType.UINT32,
				value: 2,
			},
			{
				key: "tokenizer.ggml.seperator_token_id",
				type: GgufValueType.UINT32,
				value: 3,
			},
			{
				key: "tokenizer.ggml.unknown_token_id",
				type: GgufValueType.UINT32,
				value: 1,
			},
		);
	}

	if (options.poolingType !== undefined) {
		kvEntries.push({
			key: `${arch}.pooling_type`,
			type: GgufValueType.UINT32,
			value: options.poolingType,
		});
	}
	if (options.causal !== undefined) {
		kvEntries.push({
			key: `${arch}.attention.causal`,
			type: GgufValueType.BOOL,
			value: options.causal,
		});
	}

	let kvTotalSize = 0;
	for (const entry of kvEntries) {
		kvTotalSize += 8 + entry.key.length + 4;
		if (entry.type === GgufValueType.STRING) {
			kvTotalSize += 8 + (entry.value as string).length;
		} else if (
			entry.type === GgufValueType.UINT32 ||
			entry.type === GgufValueType.FLOAT32
		) {
			kvTotalSize += 4;
		} else if (entry.type === GgufValueType.BOOL) {
			kvTotalSize += 1;
		} else if (entry.type === GgufValueType.ARRAY) {
			const arr = entry.value as unknown[];
			const firstVal = arr[0];
			if (typeof firstVal === "string") {
				kvTotalSize += 4 + 8;
				for (const v of arr as string[]) kvTotalSize += 8 + v.length;
			} else if (typeof firstVal === "number") {
				kvTotalSize += 4 + 8;
				kvTotalSize += arr.length * 4;
			}
		}
	}

	const tensorName = "output.weight";
	const tensorInfoSize = 8 + tensorName.length + 4 + 8 + 4 + 8;
	const totalSize = headerSize + kvTotalSize + tensorInfoSize + 64;
	const buf = new ArrayBuffer(totalSize);
	const view = new DataView(buf);
	let offset = 0;

	view.setUint32(offset, GGUF_MAGIC, true);
	offset += 4;
	view.setUint32(offset, GGUF_VERSION, true);
	offset += 4;
	view.setBigUint64(offset, BigInt(1), true);
	offset += 8;
	view.setBigUint64(offset, BigInt(kvEntries.length), true);
	offset += 8;

	for (const entry of kvEntries) {
		if (entry.type === GgufValueType.STRING) {
			offset = writeKvString(view, offset, entry.key, entry.value as string);
		} else if (entry.type === GgufValueType.UINT32) {
			offset = writeKvUint32(view, offset, entry.key, entry.value as number);
		} else if (entry.type === GgufValueType.FLOAT32) {
			offset = writeKvFloat32(view, offset, entry.key, entry.value as number);
		} else if (entry.type === GgufValueType.BOOL) {
			offset = writeKvBool(view, offset, entry.key, entry.value as boolean);
		} else if (entry.type === GgufValueType.ARRAY) {
			const arr = entry.value as unknown[];
			const firstVal = arr[0];
			if (typeof firstVal === "string") {
				offset = writeKvStringArray(view, offset, entry.key, arr as string[]);
			} else if (typeof firstVal === "number") {
				if (entry.key.includes("scores")) {
					offset = writeKvFloatArray(view, offset, entry.key, arr as number[]);
				} else {
					offset = writeKvIntArray(view, offset, entry.key, arr as number[]);
				}
			}
		}
	}

	offset = writeString(view, offset, tensorName);
	offset = writeUint32(view, offset, 1);
	view.setBigUint64(offset, BigInt(6), true);
	offset += 8;
	offset = writeUint32(view, offset, 0);
	view.setBigUint64(offset, BigInt(0), true);
	offset += 8;

	return buf.slice(0, offset);
}

describe("ModelLoader bert metadata extraction", () => {
	test("pooling_type=1 maps to mean", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({ poolingType: 1 }));
		expect(parsed.hyperparams.architecture).toBe("bert");
		expect(parsed.hyperparams.poolingType).toBe("mean");
	});

	test("pooling_type=2 maps to cls", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({ poolingType: 2 }));
		expect(parsed.hyperparams.poolingType).toBe("cls");
	});

	test("pooling_type=0 (NONE) falls back to cls", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({ poolingType: 0 }));
		expect(parsed.hyperparams.poolingType).toBe("cls");
	});

	test("missing pooling_type defaults to cls", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({}));
		expect(parsed.hyperparams.poolingType).toBe("cls");
	});

	test("attention.causal=false yields causalAttention=false", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({ causal: false }));
		expect(parsed.hyperparams.causalAttention).toBe(false);
	});

	test("attention.causal=true yields causalAttention=true", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({ causal: true }));
		expect(parsed.hyperparams.causalAttention).toBe(true);
	});

	test("missing attention.causal defaults to false", () => {
		const parsed = ModelLoader.parseModel(buildBertGguf({}));
		expect(parsed.hyperparams.causalAttention).toBe(false);
	});

	test("attention.layer_norm_epsilon flows into normEpsilon", () => {
		const parsed = ModelLoader.parseModel(
			buildBertGguf({ normEpsilon: 1e-12 }),
		);
		expect(parsed.hyperparams.normEpsilon).toBeCloseTo(1e-12, 14);
	});

	test("non-bert arch leaves poolingType and causalAttention undefined", () => {
		const parsed = ModelLoader.parseModel(buildModelLoaderGguf());
		expect(parsed.hyperparams.architecture).toBe("llama");
		expect(parsed.hyperparams.poolingType).toBeUndefined();
		expect(parsed.hyperparams.causalAttention).toBeUndefined();
	});

	test("bert tokenizer metadata populates WORDPIECE config with cls/sep/unk ids and undefined mask", () => {
		const parsed = ModelLoader.parseModel(
			buildBertGguf({ withBertTokenizer: true }),
		);
		expect(parsed.tokenizerConfig.type).toBe(TokenizerType.WORDPIECE);
		expect(parsed.tokenizerConfig.clsTokenId).toBe(2);
		expect(parsed.tokenizerConfig.sepTokenId).toBe(3);
		expect(parsed.tokenizerConfig.unkTokenId).toBe(1);
		expect(parsed.tokenizerConfig.maskTokenId).toBeUndefined();
	});
});
