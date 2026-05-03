import { describe, expect, test } from "bun:test";
import {
	CorruptBlobError,
	IncompatibleConversationError,
} from "../src/core/errors.js";
import {
	computeTokenizerHash,
	decodePersistedConversation,
	encodePersistedConversation,
	KV_PERSISTENCE_MAGIC,
	KV_PERSISTENCE_SCHEMA_VERSION,
	type PersistedConversationHeader,
} from "../src/core/persistence.js";

describe("computeTokenizerHash", () => {
	const baseConfig = {
		type: "BPE" as const,
		vocab: { hello: 1, world: 2 },
		merges: ["h e", "l l"],
		specialTokens: { bos: 0, eos: 3 },
	};

	test("identical input yields identical hash", async () => {
		const a = await computeTokenizerHash(baseConfig);
		const b = await computeTokenizerHash(baseConfig);
		expect(a).toBe(b);
		expect(a).toMatch(/^[0-9a-f]{64}$/); // sha256 hex
	});

	test("key-permuted input yields identical hash", async () => {
		const reordered = {
			specialTokens: { eos: 3, bos: 0 },
			merges: ["h e", "l l"],
			vocab: { world: 2, hello: 1 },
			type: "BPE" as const,
		};
		expect(await computeTokenizerHash(baseConfig)).toBe(
			await computeTokenizerHash(reordered),
		);
	});

	test("vocab change yields different hash", async () => {
		const changed = {
			...baseConfig,
			vocab: {
				...baseConfig.vocab,
				foo: 4,
			},
		};
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});

	test("specialTokens change yields different hash", async () => {
		const changed = {
			...baseConfig,
			specialTokens: {
				...baseConfig.specialTokens,
				eos: 99,
			},
		};
		expect(await computeTokenizerHash(baseConfig)).not.toBe(
			await computeTokenizerHash(changed),
		);
	});

	test("Map-valued field: hash differs when Map contents differ", async () => {
		const baseWithMap = {
			type: "BPE" as const,
			bpeRanks: new Map([
				["h e", 1],
				["l l", 2],
			]),
		};
		const changedMap = {
			type: "BPE" as const,
			bpeRanks: new Map([
				["h e", 1],
				["l l", 99], // different rank
			]),
		};
		expect(await computeTokenizerHash(baseWithMap)).not.toBe(
			await computeTokenizerHash(changedMap),
		);
	});

	test("Map-valued field: insertion-order-permuted Map yields identical hash", async () => {
		const a = {
			bpeRanks: new Map([
				["a", 1],
				["b", 2],
				["c", 3],
			]),
		};
		const b = {
			bpeRanks: new Map([
				["c", 3],
				["a", 1],
				["b", 2],
			]),
		};
		expect(await computeTokenizerHash(a)).toBe(await computeTokenizerHash(b));
	});

	test("Uint8Array field: hash differs when bytes differ", async () => {
		const a = { precompiledCharsmap: new Uint8Array([1, 2, 3, 4]) };
		const b = { precompiledCharsmap: new Uint8Array([1, 2, 3, 5]) };
		expect(await computeTokenizerHash(a)).not.toBe(
			await computeTokenizerHash(b),
		);
	});

	test("undefined-valued keys are dropped (matches JSON.stringify semantics)", async () => {
		const withUndef = { a: 1, b: undefined };
		const withoutB = { a: 1 };
		expect(await computeTokenizerHash(withUndef)).toBe(
			await computeTokenizerHash(withoutB),
		);
	});

	test("nested-object key permutation also yields identical hash", async () => {
		const a = { outer: { x: 1, y: 2, nested: { p: "a", q: "b" } } };
		const b = { outer: { nested: { q: "b", p: "a" }, y: 2, x: 1 } };
		expect(await computeTokenizerHash(a)).toBe(await computeTokenizerHash(b));
	});
});

const SAMPLE_FINGERPRINT = {
	architecture: "qwen3",
	vocabSize: 151_936,
	nEmbd: 4096,
	nLayer: 28,
	nHead: 32,
	nHeadKV: 8,
	ropeBase: 10_000,
	quantType: "Q4_K_M",
	tokenizerHash: "a".repeat(64),
};

const SAMPLE_HEADER: PersistedConversationHeader = {
	schemaVersion: 1,
	fingerprint: SAMPLE_FINGERPRINT,
	conversationOptions: { maxContextTokens: 4096 },
	tokenIds: [1, 2, 3, 4, 5],
	byteSize: 16,
	savedAtMs: 1_700_000_000_000,
};

const SAMPLE_KV = new Uint8Array([
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
	0x1d, 0x1e, 0x1f,
]);

describe("encodePersistedConversation", () => {
	test("layout: magic + uint32 LE headerLen + header JSON + kvBytes", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);

		// Magic.
		expect(blob.slice(0, 4)).toEqual(KV_PERSISTENCE_MAGIC);

		// headerLen (uint32 LE).
		const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
		const headerLen = dv.getUint32(4, /* littleEndian */ true);
		expect(headerLen).toBeGreaterThan(0);

		// Header JSON parses.
		const headerJson = new TextDecoder().decode(
			blob.subarray(8, 8 + headerLen),
		);
		const header = JSON.parse(headerJson);
		expect(header.schemaVersion).toBe(KV_PERSISTENCE_SCHEMA_VERSION);
		expect(header.fingerprint).toEqual(SAMPLE_FINGERPRINT);
		expect(header.tokenIds).toEqual([1, 2, 3, 4, 5]);

		// KV bytes.
		expect(blob.subarray(8 + headerLen)).toEqual(SAMPLE_KV);
		expect(blob.byteLength).toBe(8 + headerLen + SAMPLE_KV.byteLength);
	});

	test("two encodes of identical input produce byte-identical blobs", () => {
		const a = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		const b = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		expect(a).toEqual(b);
	});
});

describe("decodePersistedConversation — happy path", () => {
	test("round-trip preserves header + kvBytes byte-equal", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		const { header, kvBytes } = decodePersistedConversation(
			blob,
			SAMPLE_FINGERPRINT,
		);
		expect(header).toEqual(SAMPLE_HEADER);
		expect(kvBytes).toEqual(SAMPLE_KV);
	});
});

describe("decodePersistedConversation — failure modes", () => {
	test("throws CorruptBlobError(bad-magic) on wrong magic bytes", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		blob[0] = 0xff; // corrupt magic
		expect(() => decodePersistedConversation(blob, SAMPLE_FINGERPRINT)).toThrow(
			CorruptBlobError,
		);
		try {
			decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
		} catch (e) {
			expect((e as CorruptBlobError).reason).toBe("bad-magic");
		}
	});

	test("throws CorruptBlobError(bad-header-len) on overflowing headerLen", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		new DataView(blob.buffer).setUint32(4, blob.byteLength * 2, true);
		expect(() => decodePersistedConversation(blob, SAMPLE_FINGERPRINT)).toThrow(
			CorruptBlobError,
		);
	});

	test("throws CorruptBlobError(bad-header-json) on broken JSON", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		const dv = new DataView(blob.buffer);
		const headerLen = dv.getUint32(4, true);
		// Replace mid-header bytes with characters that produce invalid JSON.
		blob[8 + 1] = 0x21; // '!'
		blob[8 + headerLen - 1] = 0x21; // '!' — break trailing brace
		expect(() => decodePersistedConversation(blob, SAMPLE_FINGERPRINT)).toThrow(
			CorruptBlobError,
		);
		try {
			decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
		} catch (e) {
			expect((e as CorruptBlobError).reason).toBe("bad-header-json");
		}
	});

	test("throws IncompatibleConversationError(schema-mismatch) on wrong schemaVersion", () => {
		const wrong = { ...SAMPLE_HEADER, schemaVersion: 99 as 1 };
		const blob = encodePersistedConversation(wrong, SAMPLE_KV);
		expect(() => decodePersistedConversation(blob, SAMPLE_FINGERPRINT)).toThrow(
			IncompatibleConversationError,
		);
		try {
			decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
		} catch (e) {
			const ic = e as IncompatibleConversationError;
			expect(ic.reason).toBe("schema-mismatch");
			expect(ic.details).toEqual({ got: 99, want: 1 });
		}
	});

	test("throws IncompatibleConversationError(fingerprint-mismatch) reporting first differing field", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		const want = { ...SAMPLE_FINGERPRINT, nLayer: 32 };
		try {
			decodePersistedConversation(blob, want);
			throw new Error("should have thrown");
		} catch (e) {
			expect(e).toBeInstanceOf(IncompatibleConversationError);
			const ic = e as IncompatibleConversationError;
			expect(ic.reason).toBe("fingerprint-mismatch");
			expect(ic.details).toEqual({ field: "nLayer", got: 28, want: 32 });
		}
	});

	test("throws IncompatibleConversationError(tokenizer-mismatch) when only tokenizer differs", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		const want = { ...SAMPLE_FINGERPRINT, tokenizerHash: "b".repeat(64) };
		try {
			decodePersistedConversation(blob, want);
			throw new Error("should have thrown");
		} catch (e) {
			expect(e).toBeInstanceOf(IncompatibleConversationError);
			expect((e as IncompatibleConversationError).reason).toBe(
				"tokenizer-mismatch",
			);
		}
	});

	test("throws CorruptBlobError(byte-size-mismatch) when kvBytes length differs from header.byteSize", () => {
		// Build a blob whose header.byteSize disagrees with actual trailing length.
		const truncatedKv = SAMPLE_KV.slice(0, 8); // header still says byteSize=16
		const blob = encodePersistedConversation(SAMPLE_HEADER, truncatedKv);
		expect(() => decodePersistedConversation(blob, SAMPLE_FINGERPRINT)).toThrow(
			CorruptBlobError,
		);
		try {
			decodePersistedConversation(blob, SAMPLE_FINGERPRINT);
		} catch (e) {
			expect((e as CorruptBlobError).reason).toBe("byte-size-mismatch");
		}
	});

	test("first differing field is the one reported (deterministic order: architecture before nLayer)", () => {
		const blob = encodePersistedConversation(SAMPLE_HEADER, SAMPLE_KV);
		// Differ on architecture AND nLayer; expect architecture to win
		// because it's earlier in validateFingerprint's check order.
		const want = { ...SAMPLE_FINGERPRINT, architecture: "llama", nLayer: 32 };
		try {
			decodePersistedConversation(blob, want);
			throw new Error("should have thrown");
		} catch (e) {
			expect((e as IncompatibleConversationError).details).toMatchObject({
				field: "architecture",
			});
		}
	});
});
