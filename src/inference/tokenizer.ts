/**
 * Tokenizer supporting SPM (SentencePiece) and BPE (Byte Pair Encoding) tokenization.
 *
 * Reads vocabulary data and implements encode/decode for both algorithms using
 * a doubly-linked-list + priority-queue approach for merge operations.
 */

/** Supported tokenizer types. */
export enum TokenizerType {
	SPM = 0,
	BPE = 1,
	WORDPIECE = 2,
}

/** Token attributes describing token semantics. */
export enum TokenAttribute {
	NORMAL = 1,
	UNKNOWN = 2,
	CONTROL = 4,
	USER_DEFINED = 8,
	BYTE = 16,
	NORMALIZED = 32,
	LSTRIP = 64,
	RSTRIP = 128,
	SINGLE_WORD = 256,
}

/** Data for a single vocabulary token. */
export interface TokenData {
	text: string;
	score: number;
	attr: TokenAttribute;
}

/** Configuration required to construct a Tokenizer. */
export interface TokenizerConfig {
	type: TokenizerType;
	tokens: TokenData[];
	/** BPE merge ranks — key is "left right", value is rank (lower = higher priority). BPE only. */
	bpeRanks: Map<string, number>;
	/** Special token text -> token ID. */
	addedTokens: Map<string, number>;
	eosTokenId: number;
	bosTokenId: number;
	padTokenId: number;
	vocabSize: number;
	/**
	 * SPM only. When true, replace ASCII spaces in encode input with U+2581 (▁)
	 * and prepend a ▁ to the text; invert on decode. Matches LLaMA SentencePiece
	 * conventions. Defaults to true at the Tokenizer level.
	 */
	addPrefixSpace?: boolean;
	/** Pre-compiled character map for normalization. SPM only. */
	precompiledCharsmap?: Uint8Array;
	/** Raw chat template string from GGUF metadata (tokenizer.chat_template). */
	chatTemplate?: string;
	/** GGUF pre-tokenizer identifier such as qwen2 or llama-bpe. */
	preTokenizer?: string;
	/** Whether the model requests automatic BOS insertion. */
	addBosToken?: boolean;
	/** BERT WordPiece: id of [CLS]. */
	clsTokenId?: number;
	/** BERT WordPiece: id of [SEP]. */
	sepTokenId?: number;
	/** BERT WordPiece: id of [UNK]. */
	unkTokenId?: number;
	/** BERT WordPiece: id of [MASK] (kept for future; not used in encode/decode). */
	maskTokenId?: number;
	/** Maximum token count for encode truncation (BERT-style). */
	contextLength?: number;
}

export interface DecodeOptions {
	/** When true, preserve control and user-defined tokens in the decoded text. */
	includeSpecialTokens?: boolean;
}

/** SentencePiece whitespace marker (U+2581). */
const SPM_SPACE = "▁";

/** Min-heap priority queue for BPE merge operations (lowest rank first). */
class MinHeap {
	private heap: Array<{ rank: number; left: number; right: number }> = [];

	get length(): number {
		return this.heap.length;
	}

	push(item: { rank: number; left: number; right: number }): void {
		this.heap.push(item);
		this.bubbleUp(this.heap.length - 1);
	}

	pop(): { rank: number; left: number; right: number } | undefined {
		if (this.heap.length === 0) return undefined;
		const top = this.heap[0];
		const last = this.heap.pop();
		if (last === undefined) return top;
		if (this.heap.length > 0) {
			this.heap[0] = last;
			this.sinkDown(0);
		}
		return top;
	}

	private bubbleUp(i: number): void {
		while (i > 0) {
			const parent = (i - 1) >> 1;
			if (this.heap[parent].rank <= this.heap[i].rank) break;
			[this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
			i = parent;
		}
	}

	private sinkDown(i: number): void {
		const n = this.heap.length;
		for (;;) {
			let smallest = i;
			const left = 2 * i + 1;
			const right = 2 * i + 2;
			if (left < n && this.heap[left].rank < this.heap[smallest].rank)
				smallest = left;
			if (right < n && this.heap[right].rank < this.heap[smallest].rank)
				smallest = right;
			if (smallest === i) break;
			[this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
			i = smallest;
		}
	}
}

/** Max-heap priority queue for SPM merge operations (highest score first = most negative -score). */
class MaxHeap {
	private heap: Array<{ negScore: number; left: number; right: number }> = [];

	get length(): number {
		return this.heap.length;
	}

	push(item: { negScore: number; left: number; right: number }): void {
		this.heap.push(item);
		this.bubbleUp(this.heap.length - 1);
	}

	pop(): { negScore: number; left: number; right: number } | undefined {
		if (this.heap.length === 0) return undefined;
		const top = this.heap[0];
		const last = this.heap.pop();
		if (last === undefined) return top;
		if (this.heap.length > 0) {
			this.heap[0] = last;
			this.sinkDown(0);
		}
		return top;
	}

	private bubbleUp(i: number): void {
		while (i > 0) {
			const parent = (i - 1) >> 1;
			if (this.heap[parent].negScore <= this.heap[i].negScore) break;
			[this.heap[parent], this.heap[i]] = [this.heap[i], this.heap[parent]];
			i = parent;
		}
	}

	private sinkDown(i: number): void {
		const n = this.heap.length;
		for (;;) {
			let smallest = i;
			const left = 2 * i + 1;
			const right = 2 * i + 2;
			if (left < n && this.heap[left].negScore < this.heap[smallest].negScore)
				smallest = left;
			if (right < n && this.heap[right].negScore < this.heap[smallest].negScore)
				smallest = right;
			if (smallest === i) break;
			[this.heap[smallest], this.heap[i]] = [this.heap[i], this.heap[smallest]];
			i = smallest;
		}
	}
}

const BPE_REGEX_PATTERNS: Record<string, string[]> = {
	default: [
		"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
	],
	"llama-bpe": [
		"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	],
	qwen2: [
		"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	],
	qwen35: [
		"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
	],
};

function wpNormalize(text: string): string {
	// NFKD decomposition then strip combining marks (U+0300..U+036F) then lowercase.
	const nf = text.normalize("NFKD");
	let out = "";
	for (const ch of nf) {
		const code = ch.codePointAt(0) ?? 0;
		if (code >= 0x0300 && code <= 0x036f) continue;
		out += ch;
	}
	return out.toLowerCase();
}

function wpIsCjk(cp: number): boolean {
	return (
		(cp >= 0x4e00 && cp <= 0x9fff) ||
		(cp >= 0x3400 && cp <= 0x4dbf) ||
		(cp >= 0x20000 && cp <= 0x2a6df) ||
		(cp >= 0x2a700 && cp <= 0x2b73f) ||
		(cp >= 0x2b740 && cp <= 0x2b81f) ||
		(cp >= 0x2b820 && cp <= 0x2ceaf) ||
		(cp >= 0xf900 && cp <= 0xfaff) ||
		(cp >= 0x2f800 && cp <= 0x2fa1f)
	);
}

// ASCII ranges mirror HF BertTokenizer's explicit punctuation set, which is a
// superset of Unicode \p{P} (e.g. `$`, `^`, `` ` `` are \p{Sc}/\p{Sk} but HF
// splits on them). Do not collapse to \p{P} alone — it changes tokenization on
// common text.
function wpIsPunctuation(cp: number): boolean {
	if (
		(cp >= 33 && cp <= 47) ||
		(cp >= 58 && cp <= 64) ||
		(cp >= 91 && cp <= 96) ||
		(cp >= 123 && cp <= 126)
	) {
		return true;
	}
	return /\p{P}/u.test(String.fromCodePoint(cp));
}

function wpIsWhitespace(cp: number): boolean {
	if (cp === 32 || cp === 9 || cp === 10 || cp === 13) return true;
	return /\s/.test(String.fromCodePoint(cp));
}

/** Basic tokenize: return whitespace-separated chunks with punctuation/CJK split. */
export function wpBasicTokenize(text: string): string[] {
	const norm = wpNormalize(text);
	let out = "";
	for (const ch of norm) {
		const cp = ch.codePointAt(0) ?? 0;
		if (wpIsWhitespace(cp)) {
			out += " ";
		} else if (wpIsCjk(cp) || wpIsPunctuation(cp)) {
			out += ` ${ch} `;
		} else {
			out += ch;
		}
	}
	return out.split(/\s+/).filter((s) => s.length > 0);
}

function preTokenize(text: string, preTokenizer?: string): string[] {
	const patternStrings =
		(preTokenizer && BPE_REGEX_PATTERNS[preTokenizer]) ??
		BPE_REGEX_PATTERNS.default;
	const results: string[] = [];
	for (const patternString of patternStrings) {
		const pattern = new RegExp(patternString, "gu");
		for (const match of text.matchAll(pattern)) {
			results.push(match[0]);
		}
	}
	return results;
}

function buildBytesToUnicode(): Map<number, string> {
	const bs: number[] = [];
	for (let i = 33; i <= 126; i++) bs.push(i);
	for (let i = 161; i <= 172; i++) bs.push(i);
	for (let i = 174; i <= 255; i++) bs.push(i);
	const cs = [...bs];
	let n = 0;
	for (let b = 0; b < 256; b++) {
		if (!bs.includes(b)) {
			bs.push(b);
			cs.push(256 + n);
			n++;
		}
	}
	return new Map(bs.map((b, i) => [b, String.fromCodePoint(cs[i])]));
}

const BYTES_TO_UNICODE = buildBytesToUnicode();
const UNICODE_TO_BYTES = new Map(
	Array.from(BYTES_TO_UNICODE.entries(), ([b, ch]) => [ch, b]),
);

function encodeBytesToUnicode(text: string): string {
	const bytes = new TextEncoder().encode(text);
	let out = "";
	for (const b of bytes) {
		out += BYTES_TO_UNICODE.get(b) ?? String.fromCharCode(b);
	}
	return out;
}

function decodeUnicodeToBytes(text: string): string {
	const bytes: number[] = [];
	for (const ch of [...text]) {
		const b = UNICODE_TO_BYTES.get(ch);
		if (b !== undefined) {
			bytes.push(b);
		} else {
			bytes.push(...new TextEncoder().encode(ch));
		}
	}
	return new TextDecoder().decode(new Uint8Array(bytes));
}

export class Tokenizer {
	private config: TokenizerConfig;
	private tokenToId: Map<string, number>;
	private idToToken: TokenData[];

	constructor(config: TokenizerConfig) {
		this.config = config;
		this.tokenToId = new Map();
		this.idToToken = config.tokens;

		for (let i = 0; i < config.tokens.length; i++) {
			this.tokenToId.set(config.tokens[i].text, i);
		}

		// Also add special tokens from addedTokens map
		for (const [text, id] of config.addedTokens) {
			this.tokenToId.set(text, id);
		}
	}

	/** Encode text into token IDs. */
	encode(text: string): number[] {
		// WORDPIECE always frames with [CLS] ... [SEP], even when text is
		// empty (HF parity: tokenizer.encode("") -> [101, 102]). For other
		// tokenizer types empty input remains an empty id list.
		if (text.length === 0) {
			if (this.config.type === TokenizerType.WORDPIECE) {
				return this.encodeWordPiece(text);
			}
			return [];
		}

		// Check for special/added tokens first
		const addedResult = this.encodeWithSpecialTokens(text);
		if (addedResult !== null) return addedResult;

		switch (this.config.type) {
			case TokenizerType.BPE:
				return this.encodeBpe(text);
			case TokenizerType.SPM:
				return this.encodeSpm(text);
			case TokenizerType.WORDPIECE:
				return this.encodeWordPiece(text);
			default:
				return this.encodeBpe(text);
		}
	}

	/** Decode token IDs back to text, skipping CONTROL tokens by default. */
	decode(ids: number[], options?: DecodeOptions): string {
		if (this.config.type === TokenizerType.WORDPIECE) {
			return this.decodeWordPiece(ids, options);
		}
		const includeSpecialTokens = options?.includeSpecialTokens ?? false;
		if (this.config.type === TokenizerType.BPE && this.bpeByteEncodeEnabled()) {
			let encoded = "";
			let decoded = "";
			const flushEncoded = () => {
				if (encoded.length === 0) return;
				decoded += decodeUnicodeToBytes(encoded);
				encoded = "";
			};
			for (const id of ids) {
				const token = this.idToToken[id];
				if (!token) continue;
				if (
					token.attr &
					(TokenAttribute.CONTROL | TokenAttribute.USER_DEFINED)
				) {
					if (!includeSpecialTokens) {
						continue;
					}
					flushEncoded();
					decoded += token.text;
					continue;
				}
				encoded += token.text;
			}
			flushEncoded();
			return decoded;
		}

		// Convert byte-fallback tokens back to their raw bytes and decode as UTF-8.
		// Mixed run of byte tokens (e.g. multi-byte ▁ or non-ASCII) must be buffered
		// so TextDecoder sees a valid UTF-8 sequence.
		const bytes: number[] = [];
		const flushBytes = (acc: string[]) => {
			if (bytes.length === 0) return;
			acc.push(new TextDecoder().decode(new Uint8Array(bytes)));
			bytes.length = 0;
		};

		const parts: string[] = [];
		for (const id of ids) {
			const token = this.idToToken[id];
			if (!token) continue;
			if (token.attr & (TokenAttribute.CONTROL | TokenAttribute.USER_DEFINED)) {
				flushBytes(parts);
				if (!includeSpecialTokens) {
					continue;
				}
				parts.push(token.text);
				continue;
			}
			if (token.attr & TokenAttribute.BYTE) {
				const m = /^<0x([0-9a-fA-F]{2})>$/.exec(token.text);
				if (m) {
					bytes.push(parseInt(m[1], 16));
					continue;
				}
			}
			flushBytes(parts);
			parts.push(token.text);
		}
		flushBytes(parts);

		let text = parts.join("");
		if (this.config.type === TokenizerType.SPM) {
			text = text.replaceAll(SPM_SPACE, " ");
			if (text.startsWith(" ")) text = text.slice(1);
		}
		return text;
	}

	/** Get token data by ID. */
	getToken(id: number): TokenData | undefined {
		return this.idToToken[id];
	}

	/** Get token ID by text. */
	getId(token: string): number | undefined {
		return this.tokenToId.get(token);
	}

	/** End-of-sequence token ID. */
	get eosId(): number {
		return this.config.eosTokenId;
	}

	/** Beginning-of-sequence token ID. */
	get bosId(): number {
		return this.config.bosTokenId;
	}

	/** Padding token ID. */
	get padId(): number {
		return this.config.padTokenId;
	}

	/** Vocabulary size. */
	get vocabSize(): number {
		return this.config.vocabSize;
	}

	/** Raw tokenizer configuration (including chatTemplate). */
	get options(): TokenizerConfig {
		return this.config;
	}

	/**
	 * Attempt to encode text that may contain special/added tokens.
	 * Returns null if no special tokens are found (fall through to normal encode).
	 */
	private encodeWithSpecialTokens(text: string): number[] | null {
		if (this.config.addedTokens.size === 0) return null;

		// Try to find special tokens in the text and split around them
		const result: number[] = [];
		let remaining = text;
		let foundAny = false;

		while (remaining.length > 0) {
			let earliestIdx = remaining.length;
			let earliestToken = "";
			let earliestId = -1;

			for (const [tokenText, tokenId] of this.config.addedTokens) {
				const idx = remaining.indexOf(tokenText);
				if (idx !== -1 && idx < earliestIdx) {
					earliestIdx = idx;
					earliestToken = tokenText;
					earliestId = tokenId;
				}
			}

			if (earliestIdx === remaining.length) {
				// No more special tokens found
				break;
			}

			foundAny = true;

			// Encode text before the special token
			if (earliestIdx > 0) {
				const before = remaining.slice(0, earliestIdx);
				result.push(...this.encodeRaw(before));
			}

			result.push(earliestId);
			remaining = remaining.slice(earliestIdx + earliestToken.length);
		}

		if (!foundAny) return null;

		// Encode remaining text after last special token
		if (remaining.length > 0) {
			result.push(...this.encodeRaw(remaining));
		}

		return result;
	}

	/** Encode without special token handling. */
	private encodeRaw(text: string): number[] {
		switch (this.config.type) {
			case TokenizerType.BPE:
				return this.encodeBpe(text);
			case TokenizerType.SPM:
				return this.encodeSpm(text);
			case TokenizerType.WORDPIECE:
				return this.encodeWordPiece(text);
			default:
				return this.encodeBpe(text);
		}
	}

	private encodeWordPiece(text: string): number[] {
		const cfg = this.config;
		if (
			cfg.clsTokenId === undefined ||
			cfg.sepTokenId === undefined ||
			cfg.unkTokenId === undefined
		) {
			throw new Error(
				"WORDPIECE tokenizer requires clsTokenId, sepTokenId, and unkTokenId in config",
			);
		}
		const clsId = cfg.clsTokenId;
		const sepId = cfg.sepTokenId;
		const unkId = cfg.unkTokenId;

		const chunks = wpBasicTokenize(text);
		const ids: number[] = [clsId];
		for (const chunk of chunks) {
			ids.push(...this.wpSubword(chunk, unkId));
		}
		ids.push(sepId);

		// Truncate to contextLength, keeping [CLS] at front and [SEP] at end.
		const maxLen = cfg.contextLength ?? 512;
		if (ids.length > maxLen) {
			const trimmed = ids.slice(0, maxLen - 1);
			trimmed.push(sepId);
			return trimmed;
		}
		return ids;
	}

	private wpSubword(chunk: string, unkId: number): number[] {
		// llama.cpp's BERT GGUF converter rewrites the vocabulary: the original
		// HF "##xyz" continuation tokens are stored without the "##" prefix,
		// and every other (word-initial) token gains a phantom "▁" (U+2581)
		// prefix. Match that convention here: prepend "▁" to the whole word,
		// then run a positional longest-match scan — the first iteration
		// covers the whole-word shortcut, and subsequent positions naturally
		// pick up continuation tokens (now stored without any prefix).
		const word = `▁${chunk}`;
		const chars = [...word]; // code-point array
		// HF parity: BertWordpieceTokenizer treats any chunk longer than
		// max_input_chars_per_word (default 100) as [UNK] without subword splitting.
		if (chars.length - 1 > 100) return [unkId];

		const out: number[] = [];
		let start = 0;
		while (start < chars.length) {
			let matched: { id: number; end: number } | null = null;
			for (let end = chars.length; end > start; end--) {
				const sub = chars.slice(start, end).join("");
				const id = this.tokenToId.get(sub);
				if (id !== undefined) {
					matched = { id, end };
					break;
				}
			}
			if (!matched) return [unkId];
			out.push(matched.id);
			start = matched.end;
		}
		return out;
	}

	private decodeWordPiece(ids: number[], options?: DecodeOptions): string {
		const cfg = this.config;
		const preserve = options?.includeSpecialTokens === true;
		const specialIds = new Set<number>(
			[cfg.clsTokenId, cfg.sepTokenId, cfg.padTokenId].filter(
				(v): v is number => typeof v === "number",
			),
		);
		const parts: string[] = [];
		for (const id of ids) {
			if (id < 0 || id >= cfg.tokens.length) continue;
			const tok = cfg.tokens[id];
			if (!preserve && specialIds.has(id)) continue;
			parts.push(tok.text);
		}
		let out = "";
		for (const p of parts) {
			// Phantom-space convention (matches llama.cpp's BERT GGUF):
			// "▁foo" marks a word-initial token (emit preceded by a space
			// except at output start); any other token is a continuation
			// and concatenates directly.
			if (p.startsWith("▁")) {
				if (out.length > 0) out += " ";
				out += p.slice(1);
			} else {
				out += p;
			}
		}
		return out;
	}

	/**
	 * BPE encode algorithm:
	 * 1. Pre-tokenize text using GPT-2 regex pattern
	 * 2. For each pre-tokenized word, split into characters
	 * 3. Build a doubly-linked list of symbols
	 * 4. Seed a min-heap with all adjacent pairs, ranked by BPE merge rank
	 * 5. Pop lowest-rank pair; if still valid, merge them
	 * 6. After queue drains, convert each symbol to a token ID
	 */
	private encodeBpe(text: string): number[] {
		const words = preTokenize(text, this.config.preTokenizer);
		const result: number[] = [];

		for (const rawWord of words) {
			if (rawWord.length === 0) continue;
			const word = this.bpeByteEncodeEnabled()
				? encodeBytesToUnicode(rawWord)
				: rawWord;
			const chars = [...word];

			// Initialize symbol table as parallel arrays
			const numSymbols = chars.length;
			const symText: string[] = new Array(numSymbols);
			const symPrev: Int32Array = new Int32Array(numSymbols);
			const symNext: Int32Array = new Int32Array(numSymbols);
			const symMerged: Uint8Array = new Uint8Array(numSymbols);

			for (let i = 0; i < numSymbols; i++) {
				symText[i] = chars[i];
				symPrev[i] = i - 1;
				symNext[i] = i + 1 < numSymbols ? i + 1 : -1;
			}

			// Seed the min-heap with all adjacent pairs
			const heap = new MinHeap();
			for (let i = 0; i < numSymbols - 1; i = symNext[i]) {
				if (symNext[i] === -1) break;
				const key = `${symText[i]} ${symText[symNext[i]]}`;
				const rank = this.config.bpeRanks.get(key);
				if (rank !== undefined) {
					heap.push({ rank, left: i, right: symNext[i] });
				}
			}

			// Process merges
			while (heap.length > 0) {
				const item = heap.pop();
				if (!item) break;

				const { rank, left, right } = item;

				// Validate that the pair is still current. Adjacency + merge
				// flags catch most stale entries, but they miss the case where
				// `symText[left]` was extended by a prior merge — the position
				// is still un-merged and still adjacent to `right`, but the
				// stored rank no longer describes the current symbol pair.
				// Re-derive the rank from current contents and skip if it
				// disagrees; the correct (newer) entry was already pushed when
				// the prior merge fired and will pop later.
				if (symMerged[left] || symMerged[right]) continue;
				if (symNext[left] !== right) continue;
				const currentKey = `${symText[left]} ${symText[right]}`;
				if (this.config.bpeRanks.get(currentKey) !== rank) continue;

				// Merge right into left
				symText[left] += symText[right];
				symMerged[right] = 1;
				symNext[left] = symNext[right];
				if (symNext[right] !== -1) {
					symPrev[symNext[right]] = left;
				}

				// Add new pairs formed by the merged symbol
				if (symPrev[left] !== -1) {
					const prevLeft = symPrev[left];
					if (!symMerged[prevLeft]) {
						const key = `${symText[prevLeft]} ${symText[left]}`;
						const r = this.config.bpeRanks.get(key);
						if (r !== undefined) {
							heap.push({ rank: r, left: prevLeft, right: left });
						}
					}
				}

				if (symNext[left] !== -1) {
					const nextRight = symNext[left];
					if (!symMerged[nextRight]) {
						const key = `${symText[left]} ${symText[nextRight]}`;
						const r = this.config.bpeRanks.get(key);
						if (r !== undefined) {
							heap.push({ rank: r, left, right: nextRight });
						}
					}
				}
			}

			// Convert symbols to token IDs
			// Find the head of the linked list
			let head = 0;
			for (let i = 0; i < numSymbols; i++) {
				if (!symMerged[i]) {
					head = i;
					break;
				}
			}

			let current: number = head;
			while (current !== -1) {
				const id = this.tokenToId.get(symText[current]);
				if (id !== undefined) {
					result.push(id);
				} else {
					// Fallback: encode each character individually
					for (const ch of symText[current]) {
						const charId = this.tokenToId.get(ch);
						if (charId !== undefined) {
							result.push(charId);
						}
					}
				}
				current = symNext[current];
			}
		}

		return result;
	}

	/**
	 * SPM encode algorithm:
	 * 1. Split text into UTF-8 bytes
	 * 2. Build a doubly-linked list of symbols (one per byte initially)
	 * 3. Seed a max-heap with all adjacent pairs, ranked by -score (highest score first)
	 * 4. Pop highest-score pair; merge if valid
	 * 5. After queue drains, convert symbols to token IDs
	 * 6. For unknown sequences, use byte fallback (token ID = 0x0100 + byte_value)
	 */
	private encodeSpm(text: string): number[] {
		// SentencePiece normalization: prepend ▁ and replace spaces with ▁ so
		// vocab entries like "▁Once" can match. Controlled by addPrefixSpace
		// (default true) to match LLaMA conventions.
		const normalized = this.spmPrefixEnabled()
			? SPM_SPACE + text.replaceAll(" ", SPM_SPACE)
			: text;

		// Iterate Unicode code points (not UTF-8 bytes) so multi-byte chars like
		// ▁ form a single symbol that can match vocab entries such as "▁Once".
		const chars: string[] = [...normalized];
		const numSymbols = chars.length;

		if (numSymbols === 0) return [];

		// Initialize symbol table — one symbol per code point
		const symText: string[] = new Array(numSymbols);
		const symPrev: Int32Array = new Int32Array(numSymbols);
		const symNext: Int32Array = new Int32Array(numSymbols);
		const symMerged: Uint8Array = new Uint8Array(numSymbols);

		for (let i = 0; i < numSymbols; i++) {
			symText[i] = chars[i];
			symPrev[i] = i - 1;
			symNext[i] = i + 1 < numSymbols ? i + 1 : -1;
		}

		// Seed the max-heap with all adjacent pairs
		const heap = new MaxHeap();
		for (let i = 0; i < numSymbols - 1; i = symNext[i]) {
			if (symNext[i] === -1) break;
			const pairText = symText[i] + symText[symNext[i]];
			const tokenData = this.tokenToId.get(pairText);
			if (tokenData !== undefined) {
				const token = this.idToToken[tokenData];
				if (token) {
					// Use -score so highest score pops first (max-heap via min of -score)
					heap.push({ negScore: -token.score, left: i, right: symNext[i] });
				}
			}
		}

		// Process merges
		while (heap.length > 0) {
			const item = heap.pop();
			if (!item) break;

			const { left, right } = item;

			// Validate that the pair is still current
			if (symMerged[left] || symMerged[right]) continue;
			if (symNext[left] !== right) continue;

			// Verify the merged text still matches a valid token
			const mergedText = symText[left] + symText[right];
			const mergedId = this.tokenToId.get(mergedText);
			if (mergedId === undefined) continue;

			// Merge right into left
			symText[left] = mergedText;
			symMerged[right] = 1;
			symNext[left] = symNext[right];
			if (symNext[right] !== -1) {
				symPrev[symNext[right]] = left;
			}

			// Add new pairs formed by the merged symbol
			if (symPrev[left] !== -1) {
				const prevLeft = symPrev[left];
				if (!symMerged[prevLeft]) {
					const pairText = symText[prevLeft] + symText[left];
					const tid = this.tokenToId.get(pairText);
					if (tid !== undefined) {
						const token = this.idToToken[tid];
						if (token) {
							heap.push({
								negScore: -token.score,
								left: prevLeft,
								right: left,
							});
						}
					}
				}
			}

			if (symNext[left] !== -1) {
				const nextRight = symNext[left];
				if (!symMerged[nextRight]) {
					const pairText = symText[left] + symText[nextRight];
					const tid = this.tokenToId.get(pairText);
					if (tid !== undefined) {
						const token = this.idToToken[tid];
						if (token) {
							heap.push({ negScore: -token.score, left, right: nextRight });
						}
					}
				}
			}
		}

		// Convert symbols to token IDs
		const result: number[] = [];

		// Find the head of the linked list
		let head = -1;
		for (let i = 0; i < numSymbols; i++) {
			if (!symMerged[i]) {
				head = i;
				break;
			}
		}

		const byteEncoder = new TextEncoder();
		let current = head;
		while (current !== -1) {
			const symbolText = symText[current];
			const id = this.findLongestToken(symbolText);
			if (id !== undefined) {
				result.push(id);
			} else {
				// Byte fallback: emit one <0xHH> token per UTF-8 byte of the symbol.
				// LLaMA stores these by text (e.g. "<0x20>") at low, non-contiguous
				// IDs, so look them up by name rather than a fixed base offset.
				const bytes = byteEncoder.encode(symbolText);
				for (let b = 0; b < bytes.length; b++) {
					const byteTokenText = `<0x${bytes[b].toString(16).toUpperCase().padStart(2, "0")}>`;
					const byteId = this.tokenToId.get(byteTokenText);
					if (byteId !== undefined) {
						result.push(byteId);
					}
				}
			}
			current = symNext[current];
		}

		return result;
	}

	/** Find a token ID for the given text. Returns undefined if not found. */
	private findLongestToken(text: string): number | undefined {
		return this.tokenToId.get(text);
	}

	private spmPrefixEnabled(): boolean {
		return this.config.addPrefixSpace === true;
	}

	private bpeByteEncodeEnabled(): boolean {
		return this.config.type === TokenizerType.BPE;
	}
}

/**
 * Incremental (streaming) text decoder.
 *
 * Call push(tokenId) after each sampled token and read the return value
 * for the new text fragment. Re-decodes the full accumulated array and diffs
 * against the previous result — safe with SPM byte-fallback tokens.
 */
export class StreamingDecoder {
	private tokenizer: Tokenizer;
	private decodeOptions: DecodeOptions;
	private emittedTokenIds: number[] = [];
	private thinkDepth = 0;
	private thinkOpenId: number | undefined;
	private thinkCloseId: number | undefined;
	private tokenIds: number[] = [];
	private prevText = "";

	constructor(tokenizer: Tokenizer, decodeOptions: DecodeOptions = {}) {
		this.tokenizer = tokenizer;
		this.decodeOptions = decodeOptions;
		this.thinkOpenId = tokenizer.getId("<think>");
		this.thinkCloseId = tokenizer.getId("</think>");
	}

	/** Append a token and return the new text fragment. */
	push(tokenId: number): string {
		this.tokenIds.push(tokenId);
		if (!this.decodeOptions.includeSpecialTokens) {
			if (tokenId === this.thinkOpenId) {
				this.thinkDepth++;
				return "";
			}
			if (tokenId === this.thinkCloseId) {
				this.thinkDepth = Math.max(0, this.thinkDepth - 1);
				return "";
			}
			if (this.thinkDepth > 0) {
				return "";
			}
		}

		this.emittedTokenIds.push(tokenId);
		const fullText = this.tokenizer.decode(
			this.emittedTokenIds,
			this.decodeOptions,
		);
		const delta = fullText.slice(this.prevText.length);
		this.prevText = fullText;
		return delta;
	}

	/** Full decoded text so far. */
	get text(): string {
		return this.prevText;
	}

	/** Accumulated token IDs. */
	get tokens(): readonly number[] {
		return this.tokenIds;
	}

	/** Reset for a new generation. */
	reset(): void {
		this.emittedTokenIds = [];
		this.thinkDepth = 0;
		this.tokenIds = [];
		this.prevText = "";
	}
}
