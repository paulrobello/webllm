/**
 * Wire format and helpers for persisted-conversation blobs.
 *
 * Pure module: no I/O, no platform APIs, no proxy concerns. The engine
 * (`engine.ts`) and helper (`indexeddb-store.ts`) compose these
 * primitives. Spec: 2026-05-03-prefix-cache-persistence-design.md.
 */

import type { TokenizerConfig } from "../inference/tokenizer.js";
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
export async function computeTokenizerHash(
	cfg: TokenizerConfig,
): Promise<string> {
	const canonical = stableStringify(cfg as unknown);
	const bytes = new TextEncoder().encode(canonical);
	const digest = await crypto.subtle.digest("SHA-256", bytes);
	return Array.from(new Uint8Array(digest))
		.map((b) => b.toString(16).padStart(2, "0"))
		.join("");
}

function stableStringify(v: unknown): string {
	if (v === null || typeof v !== "object") return JSON.stringify(v);
	if (Array.isArray(v)) {
		return `[${v.map(stableStringify).join(",")}]`;
	}
	const obj = v as Record<string, unknown>;
	const keys = Object.keys(obj).sort();
	const entries = keys
		.map((k) => `${JSON.stringify(k)}:${stableStringify(obj[k])}`)
		.join(",");
	return `{${entries}}`;
}

// Forward-positioned imports — consumed by encode/decode in subsequent tasks.
void CorruptBlobError;
void IncompatibleConversationError;
