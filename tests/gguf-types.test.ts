import { describe, expect, test } from "bun:test";
import {
	GGUF_DEFAULT_ALIGNMENT,
	GGUF_MAGIC,
	GGUF_VERSION,
	GgufValueType,
} from "../src/models/gguf-types.js";

describe("GGUF Format Constants", () => {
	test('GGUF_MAGIC is "GGUF" as uint32 little-endian', () => {
		const view = new DataView(new ArrayBuffer(4));
		view.setUint32(0, GGUF_MAGIC, true);
		const decoded = new TextDecoder().decode(new Uint8Array(view.buffer));
		expect(decoded).toBe("GGUF");
	});

	test("GGUF_VERSION is 3", () => {
		expect(GGUF_VERSION).toBe(3);
	});

	test("GGUF_DEFAULT_ALIGNMENT is 32", () => {
		expect(GGUF_DEFAULT_ALIGNMENT).toBe(32);
	});
});

describe("GgufValueType enum", () => {
	test("covers all 13 GGUF value types", () => {
		const values = Object.values(GgufValueType).filter(
			(v) => typeof v === "number",
		);
		expect(values).toHaveLength(13);
	});

	test("matches llama.cpp gguf_type enum values", () => {
		expect(GgufValueType.UINT8).toBe(0);
		expect(GgufValueType.INT8).toBe(1);
		expect(GgufValueType.UINT16).toBe(2);
		expect(GgufValueType.INT16).toBe(3);
		expect(GgufValueType.UINT32).toBe(4);
		expect(GgufValueType.INT32).toBe(5);
		expect(GgufValueType.FLOAT32).toBe(6);
		expect(GgufValueType.BOOL).toBe(7);
		expect(GgufValueType.STRING).toBe(8);
		expect(GgufValueType.ARRAY).toBe(9);
		expect(GgufValueType.UINT64).toBe(10);
		expect(GgufValueType.INT64).toBe(11);
		expect(GgufValueType.FLOAT64).toBe(12);
	});
});
