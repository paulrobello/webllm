/**
 * Wire format and helpers for persisted-conversation blobs.
 *
 * Pure module: no I/O, no platform APIs, no proxy concerns. The engine
 * (`engine.ts`) and helper (`indexeddb-store.ts`) compose these
 * primitives. Spec: 2026-05-03-prefix-cache-persistence-design.md.
 */

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
