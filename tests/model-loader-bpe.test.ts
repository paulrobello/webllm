import { describe, expect, test } from "bun:test";

import { formatChatPrompt } from "../src/inference/chat-template.js";
import { Tokenizer, type TokenizerConfig } from "../src/inference/tokenizer.js";
import {
	GGUF_MAGIC,
	GGUF_VERSION,
	type GgufContext,
	GgufValueType,
} from "../src/models/gguf-types.js";
import { ModelLoader } from "../src/models/model-loader.js";

const buildTokenizerConfig = (
	ModelLoader as unknown as {
		buildTokenizerConfig(ctx: GgufContext): TokenizerConfig;
	}
).buildTokenizerConfig.bind(ModelLoader);

function writeString(buf: DataView, offset: number, str: string): number {
	buf.setBigUint64(offset, BigInt(str.length), true);
	offset += 8;
	for (let i = 0; i < str.length; i++) {
		buf.setUint8(offset++, str.charCodeAt(i));
	}
	return offset;
}

function makeContext(metadataValues: Record<string, unknown>): GgufContext {
	const metadata = new Map(
		Object.entries(metadataValues).map(([key, value]) => [
			key,
			{
				key,
				type: Array.isArray(value)
					? GgufValueType.ARRAY
					: typeof value === "string"
						? GgufValueType.STRING
						: GgufValueType.UINT32,
				isArray: Array.isArray(value),
				value,
			},
		]),
	);

	return {
		header: {
			magic: 0,
			version: 3,
			tensorCount: 0,
			metadataKvCount: metadata.size,
		},
		metadata,
		tensors: [],
		alignment: 32,
		dataOffset: 0,
		totalDataSize: 0,
	};
}

describe("ModelLoader BPE tokenizer config", () => {
	test("loads BPE merge ranks from GGUF tokenizer metadata", () => {
		const ctx = makeContext({
			"tokenizer.ggml.model": "gpt2",
			"tokenizer.ggml.tokens": ["h", "e", "l", "o", "he", "ll", "hello"],
			"tokenizer.ggml.scores": [0, 0, 0, 0, 0, 0, 0],
			"tokenizer.ggml.token_type": [1, 1, 1, 1, 1, 1, 1],
			"tokenizer.ggml.merges": ["h e", "l l", "he ll", "hell o"],
			"tokenizer.ggml.eos_token_id": 6,
			"tokenizer.ggml.bos_token_id": 6,
		});

		const config = buildTokenizerConfig(ctx);
		const tokenizer = new Tokenizer(config);

		expect(config.bpeRanks.get("h e")).toBe(0);
		expect(config.bpeRanks.get("l l")).toBe(1);
		expect(tokenizer.encode("hello")).toEqual([6]);
	});

	test("treats GGUF control tokens as added tokens", () => {
		const ctx = makeContext({
			"tokenizer.ggml.model": "gpt2",
			"tokenizer.ggml.tokens": ["<|im_start|>", "<|im_end|>", "hello"],
			"tokenizer.ggml.scores": [0, 0, 0],
			"tokenizer.ggml.token_type": [3, 4, 1],
			"tokenizer.ggml.eos_token_id": 2,
			"tokenizer.ggml.bos_token_id": 2,
		});

		const config = buildTokenizerConfig(ctx);
		const tokenizer = new Tokenizer(config);
		expect(config.addedTokens.get("<|im_start|>")).toBe(0);
		expect(config.addedTokens.get("<|im_end|>")).toBe(1);
		expect(tokenizer.decode([0, 1, 2])).toBe("hello");
	});

	test("reads qwen tokenizer metadata for pre-tokenizer and BOS policy", () => {
		const ctx = makeContext({
			"tokenizer.ggml.model": "gpt2",
			"tokenizer.ggml.pre": "qwen2",
			"tokenizer.ggml.add_bos_token": false,
			"tokenizer.ggml.tokens": ["<pad>", "<s>", "</s>", "Ċ", "Ġ", "a"],
			"tokenizer.ggml.scores": [0, 0, 0, 0, 0, 0],
			"tokenizer.ggml.token_type": [3, 3, 3, 1, 1, 1],
			"tokenizer.ggml.eos_token_id": 2,
			"tokenizer.ggml.bos_token_id": 1,
			"tokenizer.ggml.padding_token_id": 0,
		});

		const config = buildTokenizerConfig(ctx);
		expect(config.preTokenizer).toBe("qwen2");
		expect(config.addBosToken).toBe(false);
		expect(config.padTokenId).toBe(0);
	});

	test("qwen chat prompts do not close the think block unless thinking is disabled", () => {
		const template = `{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}`;
		const prompt = formatChatPrompt(
			[{ role: "user", content: "Hello!" }],
			template,
		);
		expect(prompt).toBe(
			"<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n",
		);
	});

	test("qwen chat prompts close the think block when thinking is explicitly disabled", () => {
		const template = `{%- for message in messages %}{%- if message.role == "user" %}{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}`;
		const prompt = formatChatPrompt(
			[{ role: "user", content: "Hello!" }],
			template,
			{ enableThinking: false },
		);
		expect(prompt).toContain("<|im_start|>assistant\n<think>\n\n</think>\n\n");
	});
	test("qwen thinking templates do not inject the generic default system prompt", () => {
		const template = `{%- for message in messages %}{%- if message.role == "user" %}{{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}{%- endif %}`;
		const prompt = formatChatPrompt(
			[{ role: "user", content: "Tell one short joke." }],
			template,
		);
		expect(prompt).not.toContain(
			"You are a helpful assistant. Answer questions directly and concisely.",
		);
		expect(prompt).toContain(
			"<|im_start|>user\nTell one short joke.<|im_end|>\n<|im_start|>assistant",
		);
	});

	test("plain smoke prompt for qwen stays free of chat control tokens", () => {
		const prompt = "Tell one short joke.";
		expect(prompt).not.toContain("<|im_start|>");
		expect(prompt).not.toContain("<think>");
	});

	test("detects qwen thinking templates for sampled smoke generation", () => {
		const template = `{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}`;
		expect(template.includes("enable_thinking")).toBe(true);
		expect(template.includes("<think>")).toBe(true);
	});

	test("reads alternate rope freq_base metadata keys used by qwen3", () => {
		const kvEntries: Array<[string, string | number]> = [
			["general.architecture", "qwen3"],
			["qwen3.context_length", 40960],
			["qwen3.embedding_length", 1024],
			["qwen3.attention.head_count", 16],
			["qwen3.attention.head_count_kv", 8],
			["qwen3.block_count", 28],
			["qwen3.attention.key_length", 128],
			["qwen3.feed_forward_length", 3072],
			["qwen3.rope.freq_base", 1_000_000],
			["qwen3.attention.layer_norm_rms_epsilon", 1e-6],
			["tokenizer.ggml.model", "gpt2"],
			["tokenizer.ggml.tokens", ["<unk>", "<s>", "</s>"] as unknown as string],
		];

		const buf = new ArrayBuffer(4096);
		const view = new DataView(buf);
		view.setUint32(0, GGUF_MAGIC, true);
		view.setUint32(4, GGUF_VERSION, true);
		view.setBigUint64(8, 0n, true);
		view.setBigUint64(16, 11n, true);
		let offset = 24;

		offset = writeString(view, offset, "general.architecture");
		view.setUint32(offset, GgufValueType.STRING, true);
		offset += 4;
		offset = writeString(view, offset, "qwen3");

		for (const [key, value] of kvEntries.slice(1, 10)) {
			offset = writeString(view, offset, key);
			view.setUint32(
				offset,
				typeof value === "number" && !Number.isInteger(value)
					? GgufValueType.FLOAT32
					: GgufValueType.UINT32,
				true,
			);
			offset += 4;
			if (typeof value === "number" && !Number.isInteger(value)) {
				view.setFloat32(offset, value, true);
				offset += 4;
			} else {
				view.setUint32(offset, Number(value), true);
				offset += 4;
			}
		}

		offset = writeString(view, offset, "tokenizer.ggml.model");
		view.setUint32(offset, GgufValueType.STRING, true);
		offset += 4;
		offset = writeString(view, offset, "gpt2");

		offset = writeString(view, offset, "tokenizer.ggml.tokens");
		view.setUint32(offset, GgufValueType.ARRAY, true);
		offset += 4;
		view.setUint32(offset, GgufValueType.STRING, true);
		offset += 4;
		view.setBigUint64(offset, 3n, true);
		offset += 8;
		for (const token of ["<unk>", "<s>", "</s>"]) {
			offset = writeString(view, offset, token);
		}

		const parsed = ModelLoader.parseModel(buf.slice(0, offset));
		expect(parsed.hyperparams.ropeFreqBase).toBe(1_000_000);
	});
});
