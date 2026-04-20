import { describe, expect, test } from "bun:test";
import { GgufParser } from "../src/models/gguf-parser.js";
import type { GgufValueType } from "../src/models/gguf-types.js";
import { GGUF_MAGIC, GGUF_VERSION } from "../src/models/gguf-types.js";

function writeString(buf: DataView, offset: number, str: string): number {
	buf.setBigUint64(offset, BigInt(str.length), true);
	offset += 8;
	for (let i = 0; i < str.length; i++) {
		buf.setUint8(offset++, str.charCodeAt(i));
	}
	return offset;
}

function buildMinimalGguf(): ArrayBuffer {
	const headerSize = 24;
	const kv: Array<{ key: string; type: GgufValueType; value: unknown }> = [
		{ key: "general.architecture", type: 8 as GgufValueType, value: "llama" },
		{ key: "llama.context_length", type: 4 as GgufValueType, value: 4096 },
	];
	const tensorCount = 1;
	const tensorName = "token_embd.weight";

	let kvSize = 0;
	for (const entry of kv) {
		kvSize += 8 + entry.key.length + 4;
		if ((entry.type as GgufValueType) === 8)
			kvSize += 8 + (entry.value as string).length;
		else if ((entry.type as GgufValueType) === 4) kvSize += 4;
	}

	const tensorInfoSize = 8 + tensorName.length + 4 + 8 + 4 + 8;
	const totalSize = headerSize + kvSize + tensorInfoSize + 32 + 64;
	const buf = new ArrayBuffer(totalSize);
	const view = new DataView(buf);
	let offset = 0;

	view.setUint32(offset, GGUF_MAGIC, true);
	offset += 4;
	view.setUint32(offset, GGUF_VERSION, true);
	offset += 4;
	view.setBigUint64(offset, BigInt(tensorCount), true);
	offset += 8;
	view.setBigUint64(offset, BigInt(kv.length), true);
	offset += 8;

	for (const entry of kv) {
		offset = writeString(view, offset, entry.key);
		view.setUint32(offset, entry.type as number, true);
		offset += 4;
		if ((entry.type as GgufValueType) === 8)
			offset = writeString(view, offset, entry.value as string);
		else if ((entry.type as GgufValueType) === 4) {
			view.setUint32(offset, entry.value as number, true);
			offset += 4;
		}
	}

	offset = writeString(view, offset, tensorName);
	view.setUint32(offset, 1, true);
	offset += 4;
	view.setBigUint64(offset, BigInt(10), true);
	offset += 8;
	view.setUint32(offset, 0, true);
	offset += 4;
	view.setBigUint64(offset, BigInt(0), true);
	offset += 8;

	return buf.slice(0, offset);
}

describe("GgufParser", () => {
	test("parses header correctly", () => {
		const ctx = GgufParser.parse(buildMinimalGguf());
		expect(ctx.header.magic).toBe(GGUF_MAGIC);
		expect(ctx.header.version).toBe(GGUF_VERSION);
		expect(ctx.header.tensorCount).toBe(1);
		expect(ctx.header.metadataKvCount).toBe(2);
	});

	test("parses metadata KV pairs", () => {
		const ctx = GgufParser.parse(buildMinimalGguf());
		expect(ctx.metadata.size).toBe(2);
		expect(ctx.metadata.get("general.architecture")?.value).toBe("llama");
		expect(ctx.metadata.get("llama.context_length")?.value).toBe(4096);
	});

	test("parses tensor info", () => {
		const ctx = GgufParser.parse(buildMinimalGguf());
		expect(ctx.tensors).toHaveLength(1);
		expect(ctx.tensors[0].name).toBe("token_embd.weight");
		expect(ctx.tensors[0].nDimensions).toBe(1);
		expect(ctx.tensors[0].dimensions[0]).toBe(10);
		expect(ctx.tensors[0].type).toBe(0);
	});

	test("throws on invalid magic", () => {
		const buf = new ArrayBuffer(24);
		new DataView(buf).setUint32(0, 0xdeadbeef, true);
		expect(() => GgufParser.parse(buf)).toThrow("Invalid GGUF magic");
	});

	test("throws on unsupported version", () => {
		const buf = new ArrayBuffer(24);
		const view = new DataView(buf);
		view.setUint32(0, GGUF_MAGIC, true);
		view.setUint32(4, 99, true);
		expect(() => GgufParser.parse(buf)).toThrow("Unsupported GGUF version");
	});

	test("getMetadataString returns string value", () => {
		const ctx = GgufParser.parse(buildMinimalGguf());
		expect(GgufParser.getMetadataString(ctx, "general.architecture")).toBe(
			"llama",
		);
	});

	test("getMetadataNumber returns numeric value", () => {
		const ctx = GgufParser.parse(buildMinimalGguf());
		expect(GgufParser.getMetadataNumber(ctx, "llama.context_length")).toBe(
			4096,
		);
	});
});
