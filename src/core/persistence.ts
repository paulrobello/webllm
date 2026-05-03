/**
 * Wire format and helpers for persisted-conversation blobs.
 *
 * Pure module: no I/O, no platform APIs, no proxy concerns. The engine
 * (`engine.ts`) and helper (`indexeddb-store.ts`) compose these
 * primitives. Spec: 2026-05-03-prefix-cache-persistence-design.md.
 */

import { CorruptBlobError, IncompatibleConversationError } from "./errors.js";

export const KV_PERSISTENCE_SCHEMA_VERSION = 1;
// "WLKV" — magic bytes that mark a persisted-conversation blob.
export const KV_PERSISTENCE_MAGIC = new Uint8Array([0x57, 0x4c, 0x4b, 0x56]);

export interface ModelFingerprint {
	architecture: string;
	vocabSize: number;
	nEmbd: number;
	nLayer: number;
	nHead: number;
	nHeadKV: number;
	ropeBase: number;
	quantType: string;
	tokenizerHash: string;
}

export interface PersistedConversationHeader {
	schemaVersion: 1;
	fingerprint: ModelFingerprint;
	conversationOptions: { maxContextTokens?: number };
	tokenIds: number[];
	byteSize: number;
	savedAtMs: number;
}

/**
 * Deterministic hex-sha256 over the canonical-key-sorted JSON of a
 * tokenizer config. Used to fingerprint vocab pinning so blobs from a
 * subtly-different tokenizer (same arch, retrained) refuse to import.
 */
export async function computeTokenizerHash(cfg: unknown): Promise<string> {
	const canonical = stableStringify(cfg);
	const bytes = new TextEncoder().encode(canonical);
	const digest = await crypto.subtle.digest("SHA-256", bytes);
	return Array.from(new Uint8Array(digest))
		.map((b) => b.toString(16).padStart(2, "0"))
		.join("");
}

function stableStringify(v: unknown): string {
	if (v === null || typeof v !== "object") return JSON.stringify(v) ?? "null";
	if (Array.isArray(v)) {
		return `[${v.map(stableStringify).join(",")}]`;
	}
	if (v instanceof Map) {
		// Sort by stringified key for canonical order. Map keys can be any
		// type, but the production caller (TokenizerConfig.bpeRanks /
		// addedTokens) uses string keys. We coerce + sort to handle both.
		const entries = [...v.entries()].sort((a, b) => {
			const ka = String(a[0]);
			const kb = String(b[0]);
			return ka < kb ? -1 : ka > kb ? 1 : 0;
		});
		const inner = entries
			.map(([k, val]) => `${stableStringify(k)}:${stableStringify(val)}`)
			.join(",");
		return `{__map__:[${inner}]}`;
	}
	if (v instanceof Uint8Array) {
		// Length-prefixed CSV of bytes — canonical order is the array's own
		// order (positional, not sorted). Length prefix prevents collisions
		// between a 0-byte array and a missing field.
		return `{__u8__:${v.byteLength},${Array.from(v).join(",")}}`;
	}
	if (ArrayBuffer.isView(v)) {
		// Other typed arrays (Int32Array, Float32Array, etc.) — fall back to
		// the same byte-CSV form against the underlying buffer view.
		const u8 = new Uint8Array((v as ArrayBufferView).buffer);
		return `{__u8__:${u8.byteLength},${Array.from(u8).join(",")}}`;
	}
	const obj = v as Record<string, unknown>;
	const keys = Object.keys(obj).sort();
	const entries = keys
		.filter((k) => obj[k] !== undefined) // match JSON.stringify: drop undefined-valued keys
		.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`)
		.join(",");
	return `{${entries}}`;
}

/**
 * Build the wire-format blob for a persisted conversation.
 *
 * The header is serialized via `JSON.stringify`. Callers should
 * construct `header` with stable key order — the production caller in
 * `engine.ts:exportConversation` constructs an object literal with a
 * fixed key sequence, so V8/JSC preserves it (per ES2015 own-keys
 * ordering). For two different callers to produce byte-identical
 * blobs, both must build their headers identically.
 */
export function encodePersistedConversation(
	header: PersistedConversationHeader,
	kvBytes: Uint8Array,
): Uint8Array {
	const headerJson = JSON.stringify(header);
	const headerBytes = new TextEncoder().encode(headerJson);
	const out = new Uint8Array(
		4 + 4 + headerBytes.byteLength + kvBytes.byteLength,
	);
	out.set(KV_PERSISTENCE_MAGIC, 0);
	new DataView(out.buffer).setUint32(4, headerBytes.byteLength, /* LE */ true);
	out.set(headerBytes, 8);
	out.set(kvBytes, 8 + headerBytes.byteLength);
	return out;
}

/**
 * Parse and validate a persisted-conversation blob. Throws
 * `CorruptBlobError` for malformed bytes (bad magic, header overflow,
 * JSON parse failure, byte-size mismatch) and
 * `IncompatibleConversationError` for valid-but-wrong blobs (schema
 * version, fingerprint, or tokenizer hash diverge from the loaded
 * model). On success returns `{header, kvBytes}` — `kvBytes` is a
 * fresh copy decoupled from the input blob's underlying buffer (so
 * caller-side detach of the input doesn't poison the snapshot).
 */
export function decodePersistedConversation(
	blob: Uint8Array,
	expectedFingerprint: ModelFingerprint,
): { header: PersistedConversationHeader; kvBytes: Uint8Array } {
	// 1. Magic.
	if (blob.byteLength < 8) {
		throw new CorruptBlobError("bad-magic", { byteLength: blob.byteLength });
	}
	for (let i = 0; i < 4; i++) {
		if (blob[i] !== KV_PERSISTENCE_MAGIC[i]) {
			throw new CorruptBlobError("bad-magic", {
				firstFour: Array.from(blob.subarray(0, 4)),
			});
		}
	}
	// 2. headerLen.
	const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
	const headerLen = dv.getUint32(4, /* LE */ true);
	if (8 + headerLen > blob.byteLength) {
		throw new CorruptBlobError("bad-header-len", {
			headerLen,
			blobLength: blob.byteLength,
		});
	}
	// 3. Header JSON.
	let header: PersistedConversationHeader;
	try {
		const json = new TextDecoder().decode(blob.subarray(8, 8 + headerLen));
		header = JSON.parse(json) as PersistedConversationHeader;
	} catch (e) {
		throw new CorruptBlobError("bad-header-json", {
			cause: e instanceof Error ? e.message : String(e),
		});
	}
	// 4. Schema version.
	if (header.schemaVersion !== KV_PERSISTENCE_SCHEMA_VERSION) {
		throw new IncompatibleConversationError("schema-mismatch", {
			got: header.schemaVersion,
			want: KV_PERSISTENCE_SCHEMA_VERSION,
		});
	}
	// 5. Fingerprint (with tokenizer separated to give the more-specific
	//    "tokenizer-mismatch" reason on tokenizer-only divergence).
	validateFingerprint(header.fingerprint, expectedFingerprint);
	// 6. byteSize sanity.
	const kvBytes = blob.subarray(8 + headerLen);
	if (kvBytes.byteLength !== header.byteSize) {
		throw new CorruptBlobError("byte-size-mismatch", {
			got: kvBytes.byteLength,
			want: header.byteSize,
		});
	}
	// Return a fresh copy of kvBytes — decouples from the caller's blob
	// underlying buffer so a transferable detach upstream doesn't poison
	// the snapshot the engine stashes in its conversation pool.
	return { header, kvBytes: new Uint8Array(kvBytes) };
}

function validateFingerprint(
	got: ModelFingerprint,
	want: ModelFingerprint,
): void {
	// Check non-tokenizer fields first in deterministic order; only after
	// those match do we check tokenizerHash, so a tokenizer-only mismatch
	// surfaces as the more-specific "tokenizer-mismatch" reason rather
	// than the generic "fingerprint-mismatch" with field=tokenizerHash.
	const fields: Array<keyof ModelFingerprint> = [
		"architecture",
		"vocabSize",
		"nEmbd",
		"nLayer",
		"nHead",
		"nHeadKV",
		"ropeBase",
		"quantType",
	];
	for (const f of fields) {
		if (got[f] !== want[f]) {
			throw new IncompatibleConversationError("fingerprint-mismatch", {
				field: f,
				got: got[f],
				want: want[f],
			});
		}
	}
	if (got.tokenizerHash !== want.tokenizerHash) {
		throw new IncompatibleConversationError("tokenizer-mismatch", {
			got: got.tokenizerHash,
			want: want.tokenizerHash,
		});
	}
}
