import { describe, expect, it } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { GgufParser } from "../../src/models/gguf-parser.js";

const GEMMA4 = "smoke-test/models/gemma-4-e2b-it-q4km.gguf";

describe.skipIf(!existsSync(GEMMA4))(
	"GGUF array readers on Gemma 4 E2B",
	() => {
		it("reads gemma4.feed_forward_length as a 35-element array", () => {
			const buf = readFileSync(GEMMA4);
			const ctx = GgufParser.parse(buf);
			const ffn = GgufParser.getMetadataNumberArray(
				ctx,
				"gemma4.feed_forward_length",
			);
			expect(ffn).toHaveLength(35);
			expect(ffn[0]).toBe(6144);
			expect(ffn[14]).toBe(6144);
			expect(ffn[15]).toBe(12288);
			expect(ffn[34]).toBe(12288);
		});

		it("reads gemma4.attention.sliding_window_pattern as 35-element bool array", () => {
			const buf = readFileSync(GEMMA4);
			const ctx = GgufParser.parse(buf);
			const pat = GgufParser.getMetadataBooleanArray(
				ctx,
				"gemma4.attention.sliding_window_pattern",
			);
			expect(pat).toHaveLength(35);
			// Pattern (T,T,T,T,F) × 7 — index 4 (0-based) is first global
			expect(pat[0]).toBe(true);
			expect(pat[3]).toBe(true);
			expect(pat[4]).toBe(false);
			expect(pat[9]).toBe(false);
		});

		it("returns fallback for missing key", () => {
			const buf = readFileSync(GEMMA4);
			const ctx = GgufParser.parse(buf);
			expect(GgufParser.getMetadataNumberArray(ctx, "nonexistent.key")).toEqual(
				[],
			);
			expect(
				GgufParser.getMetadataNumberArray(ctx, "nonexistent.key", [42]),
			).toEqual([42]);
			expect(
				GgufParser.getMetadataBooleanArray(ctx, "nonexistent.key"),
			).toEqual([]);
		});
	},
);
