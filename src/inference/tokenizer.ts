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
	WPM = 2,
	UGM = 3,
	RWKV = 4,
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

/**
 * GPT-2 pre-tokenization regex pattern.
 * Splits text into words, contractions, individual letters, and whitespace groups.
 */
const GPT2_PATTERNS = [
	/(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu,
];

/**
 * Pre-tokenize text using GPT-2 regex pattern for BPE tokenization.
 * Returns an array of word-level chunks.
 */
function preTokenize(text: string): string[] {
	const results: string[] = [];
	for (const pattern of GPT2_PATTERNS) {
		let match = pattern.exec(text);
		while (match !== null) {
			results.push(match[0]);
			match = pattern.exec(text);
		}
	}
	return results;
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
		if (text.length === 0) return [];

		// Check for special/added tokens first
		const addedResult = this.encodeWithSpecialTokens(text);
		if (addedResult !== null) return addedResult;

		switch (this.config.type) {
			case TokenizerType.BPE:
				return this.encodeBpe(text);
			case TokenizerType.SPM:
				return this.encodeSpm(text);
			default:
				return this.encodeBpe(text);
		}
	}

	/** Decode token IDs back to text, skipping CONTROL tokens. */
	decode(ids: number[]): string {
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
			if (token.attr & TokenAttribute.CONTROL) {
				flushBytes(parts);
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
		if (this.config.type === TokenizerType.SPM && this.spmPrefixEnabled()) {
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
			default:
				return this.encodeBpe(text);
		}
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
		const words = preTokenize(text);
		const result: number[] = [];

		for (const word of words) {
			if (word.length === 0) continue;

			// Initialize symbol table as parallel arrays
			const numSymbols = word.length;
			const symText: string[] = new Array(numSymbols);
			const symPrev: Int32Array = new Int32Array(numSymbols);
			const symNext: Int32Array = new Int32Array(numSymbols);
			const symMerged: Uint8Array = new Uint8Array(numSymbols);

			for (let i = 0; i < numSymbols; i++) {
				symText[i] = word[i];
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

				const { rank: _rank, left, right } = item;

				// Validate that the pair is still current
				if (symMerged[left] || symMerged[right]) continue;
				if (symNext[left] !== right) continue;

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
		return this.config.addPrefixSpace !== false;
	}
}
