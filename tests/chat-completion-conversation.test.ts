import { describe, expect, test } from "bun:test";
import { ConversationPool } from "../src/core/conversation-pool.js";
import { WebLLM } from "../src/index.js";
import {
	TokenAttribute,
	type TokenData,
	Tokenizer,
	type TokenizerConfig,
	TokenizerType,
} from "../src/inference/tokenizer.js";

// Minimal tokenizer that matches the engine-streaming-api.test.ts pattern.
const TOKENS: TokenData[] = [
	{ text: "<pad>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "<s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "</s>", score: 0, attr: TokenAttribute.CONTROL },
	{ text: "p", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "h", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "i", score: -1, attr: TokenAttribute.NORMAL },
	{ text: "!", score: -1, attr: TokenAttribute.NORMAL },
];

function createTokenizer(): Tokenizer {
	const config: TokenizerConfig = {
		type: TokenizerType.BPE,
		tokens: TOKENS,
		bpeRanks: new Map(),
		addedTokens: new Map(),
		eosTokenId: 2,
		bosTokenId: 1,
		padTokenId: 0,
		vocabSize: TOKENS.length,
	};
	return new Tokenizer(config);
}

function createLogits(tokenId: number, vocabSize: number): Float32Array {
	const logits = new Float32Array(vocabSize);
	logits[tokenId] = 100;
	return logits;
}

type EngineInternals = {
	_modelManager: {
		get(id: string): unknown;
		unregister?: (id: string) => Promise<void>;
	};
	inferenceEngines: Map<string, FakeInference>;
	encoderEngines: Map<string, unknown>;
	causalEmbedderEngines: Map<string, unknown>;
	wasmModules: Map<string, unknown>;
	conversationPool: ConversationPool;
	sessions: Map<string, unknown>;
	modelChatChains: Map<string, Promise<void>>;
};

function asInternals(engine: WebLLM): EngineInternals {
	return engine as unknown as EngineInternals;
}

interface FakeInference {
	flashAttn: boolean;
	maxContextLength: number;
	cachedTokenCount: number;
	forward: (ids: Int32Array, positions: Int32Array) => Promise<Float32Array>;
	resetKVCache: () => void;
	truncateKVCache: (n: number) => void;
	loadKVCache: (
		bytes: Uint8Array,
		nTokens: number,
		snapshotLen?: number,
	) => Promise<void>;
	serializeKVCache: (nTokens: number) => Promise<Uint8Array>;
}

function createFakeEngine(sequence: number[] = [4, 5, 2]): WebLLM {
	const tokenizer = createTokenizer();
	let step = 0;
	const fake: FakeInference = {
		flashAttn: true,
		maxContextLength: 2048,
		cachedTokenCount: 0,
		forward: async (
			ids: Int32Array,
			positions: Int32Array,
		): Promise<Float32Array> => {
			// Mirror the real ModelInference: positions[last]+1 is the new
			// total slot count after the forward pass writes positions
			// [positions[0], positions[last]].
			const lastPos = positions[positions.length - 1];
			fake.cachedTokenCount = lastPos + 1;
			const tokenId = sequence[Math.min(step, sequence.length - 1)];
			step++;
			void ids;
			return createLogits(tokenId, tokenizer.vocabSize);
		},
		resetKVCache: () => {
			fake.cachedTokenCount = 0;
		},
		truncateKVCache: (n: number) => {
			fake.cachedTokenCount = n;
		},
		loadKVCache: async (_bytes, nTokens) => {
			fake.cachedTokenCount = nTokens;
		},
		serializeKVCache: async (nTokens: number) => {
			// One byte per token is enough — engine doesn't introspect.
			return new Uint8Array(nTokens);
		},
	};

	const engine = Object.create(WebLLM.prototype) as WebLLM;
	const internals = asInternals(engine);
	internals._modelManager = {
		get: (id: string) => {
			if (id !== "tl") return undefined;
			return {
				id: "tl",
				loaded: true,
				tokenizer,
				hyperparams: { architecture: "llama" },
			};
		},
		unregister: async () => {},
	};
	internals.inferenceEngines = new Map<string, FakeInference>([["tl", fake]]);
	internals.encoderEngines = new Map();
	internals.causalEmbedderEngines = new Map();
	internals.wasmModules = new Map();
	internals.conversationPool = new ConversationPool({ maxConversations: 4 });
	internals.sessions = new Map();
	internals.modelChatChains = new Map();
	return engine;
}

describe("chatCompletion(conv, ...)", () => {
	test("on disposed handle throws ConversationNotFoundError", async () => {
		const engine = createFakeEngine();
		const conv = engine.createConversation("tl");
		engine.disposeConversation(conv);

		let threw: unknown;
		try {
			for await (const _chunk of engine.chatCompletion(conv, [
				{ role: "user", content: "hi" },
			])) {
				// drain
			}
		} catch (err) {
			threw = err;
		}
		expect(threw).toBeDefined();
		expect((threw as Error).message).toMatch(/conversation/i);
	});

	test("concurrent call on the same handle throws ConversationBusyError", async () => {
		const engine = createFakeEngine();
		const conv = engine.createConversation("tl");
		const internals = asInternals(engine);
		// Pre-acquire the lock to simulate an in-flight call.
		const release = internals.conversationPool.tryAcquireLock(conv);
		expect(release).not.toBeNull();

		let threw: unknown;
		try {
			for await (const _chunk of engine.chatCompletion(conv, [
				{ role: "user", content: "hi" },
			])) {
				// drain
			}
		} catch (err) {
			threw = err;
		}
		expect(threw).toBeDefined();
		expect((threw as Error).message).toMatch(/busy|in-flight/i);
		release?.();
	});

	test("second turn's prefill positions start after sharedLen", async () => {
		const engine = createFakeEngine();
		const conv = engine.createConversation("tl");
		const internals = asInternals(engine);
		const inf = internals.inferenceEngines.get("tl");
		if (!inf) throw new Error("missing fake");

		// Wrap inf.forward to log per-call (ids, positions).
		const positionsLog: Array<{ ids: number[]; positions: number[] }> = [];
		const origForward = inf.forward;
		inf.forward = async (ids: Int32Array, positions: Int32Array) => {
			positionsLog.push({ ids: [...ids], positions: [...positions] });
			return origForward(ids, positions);
		};

		// First turn: cold start. inf.forward will be called with positions
		// [0, 1, ..., n-1].
		for await (const _chunk of engine.chatCompletion(
			conv,
			[{ role: "user", content: "p" }],
			{ maxTokens: 1, temperature: 0 },
		)) {
			// drain
		}
		const turn1Positions = [...positionsLog];
		expect(turn1Positions.length).toBeGreaterThan(0);
		// First call's positions begin at 0.
		expect(turn1Positions[0].positions[0]).toBe(0);

		// Snapshot the first turn's prompt length — this is the lower-bound
		// for sharedLen on turn 2 (the chat template will reproduce these
		// tokens deterministically as the first part of the next prompt).
		const snap1 = internals.conversationPool.get(conv);
		expect(snap1).toBeDefined();
		const turn1Len = snap1?.tokenIds.length ?? 0;
		expect(turn1Len).toBeGreaterThan(0);

		positionsLog.length = 0;

		// Second turn: superset of turn 1 (same first user message + assistant + new user).
		// sharedLen is at least the prefix common to turn 1's prompt and turn 2's
		// prompt. The new prefill positions MUST start at >= sharedLen, NOT at 0.
		// (The buggy code passed lastPos as sequenceId — the constructor ignored
		// it and left position=0, which means every forward call began at 0,
		// overwriting the loaded prefix's KV slot 0.)
		for await (const _chunk of engine.chatCompletion(
			conv,
			[
				{ role: "user", content: "p" },
				{ role: "assistant", content: "i" },
				{ role: "user", content: "h" },
			],
			{ maxTokens: 1, temperature: 0 },
		)) {
			// drain
		}
		const turn2Positions = [...positionsLog];
		expect(turn2Positions.length).toBeGreaterThan(0);
		// Second turn's prefill positions must START at >= sharedLen >= 1
		// (BOS is always shared). With C1 unfixed this asserts as 0.
		expect(turn2Positions[0].positions[0]).toBeGreaterThanOrEqual(1);

		// Restore.
		inf.forward = origForward;
	});

	test("shared-prefix path skips resetKVCache and uses loadKVCache", async () => {
		const engine = createFakeEngine([4, 5, 2]);
		const conv = engine.createConversation("tl");
		const internals = asInternals(engine);
		const inf = internals.inferenceEngines.get("tl");
		if (!inf) throw new Error("missing fake");

		// Seed a snapshot directly so we don't depend on chat-template
		// determinism across calls. Pretend a prior turn populated the KV
		// with prompt tokens [1, 3, 5] (BOS + "p" + "i") and the working
		// KV byte buffer is whatever — engine just slices by snapshotLen.
		const seededTokenIds = [1, 3, 5];
		internals.conversationPool.set(conv, {
			conversationId: conv.id,
			modelHandleId: "tl",
			tokenIds: [...seededTokenIds],
			kvBytes: new Uint8Array(seededTokenIds.length),
			byteSize: seededTokenIds.length,
			lastAccessMs: 0,
		});

		// Spy on resetKVCache and loadKVCache.
		let resetCalls = 0;
		let loadCalls = 0;
		const origReset = inf.resetKVCache;
		const origLoad = inf.loadKVCache;
		inf.resetKVCache = () => {
			resetCalls++;
			origReset();
		};
		inf.loadKVCache = async (bytes, nTokens, snapshotLen) => {
			loadCalls++;
			await origLoad(bytes, nTokens, snapshotLen);
		};

		// `engine.chatCompletion(conv, [user "p"])` will tokenize via the
		// chat template — for the llama/zephyr template that emits a
		// preamble, then user text, etc. The exact share is dependent on
		// the chat template's first tokens (BOS=1 is added automatically
		// since addBosToken !== false). Whatever the share is, the test
		// asserts: load was called (sharedLen > 0 expected because BOS is
		// shared) OR if shared=0 reset was called.
		for await (const _ of engine.chatCompletion(
			conv,
			[{ role: "user", content: "p" }],
			{ maxTokens: 1, temperature: 0 },
		)) {
			// drain
		}

		// At minimum, BOS=1 is shared between seeded and new tokens →
		// load path taken, reset NOT called by chatCompletionWithConversation.
		expect(loadCalls).toBe(1);
		expect(resetCalls).toBe(0);

		// Restore.
		inf.resetKVCache = origReset;
		inf.loadKVCache = origLoad;
	});

	test("skipSave=true skips serializeKVCache and leaves prior snapshot untouched", async () => {
		const engine = createFakeEngine();
		const conv = engine.createConversation("tl");
		const internals = asInternals(engine);
		const inf = internals.inferenceEngines.get("tl");
		if (!inf) throw new Error("missing fake");

		let serializeCalls = 0;
		const origSerialize = inf.serializeKVCache;
		inf.serializeKVCache = async (n: number) => {
			serializeCalls++;
			return origSerialize(n);
		};

		// Seed a sentinel snapshot so we can prove it's untouched after the call.
		const sentinel = new Uint8Array([0xab, 0xcd]);
		internals.conversationPool.set(conv, {
			conversationId: conv.id,
			modelHandleId: "tl",
			tokenIds: [1, 2, 3],
			kvBytes: sentinel,
			byteSize: sentinel.byteLength,
			lastAccessMs: 0,
		});

		for await (const _ of engine.chatCompletion(
			conv,
			[{ role: "user", content: "p" }],
			{ maxTokens: 1, temperature: 0, skipSave: true },
		)) {
			// drain
		}

		expect(serializeCalls).toBe(0);
		const snap = internals.conversationPool.get(conv);
		expect(snap?.kvBytes).toBe(sentinel);

		inf.serializeKVCache = origSerialize;
	});

	test("skipSave omitted (default) still serializes and updates snapshot", async () => {
		const engine = createFakeEngine();
		const conv = engine.createConversation("tl");
		const internals = asInternals(engine);
		const inf = internals.inferenceEngines.get("tl");
		if (!inf) throw new Error("missing fake");

		let serializeCalls = 0;
		const origSerialize = inf.serializeKVCache;
		inf.serializeKVCache = async (n: number) => {
			serializeCalls++;
			return origSerialize(n);
		};

		const sentinel = new Uint8Array([0xab, 0xcd]);
		internals.conversationPool.set(conv, {
			conversationId: conv.id,
			modelHandleId: "tl",
			tokenIds: [1, 2, 3],
			kvBytes: sentinel,
			byteSize: sentinel.byteLength,
			lastAccessMs: 0,
		});

		for await (const _ of engine.chatCompletion(
			conv,
			[{ role: "user", content: "p" }],
			{ maxTokens: 1, temperature: 0 },
		)) {
			// drain
		}

		expect(serializeCalls).toBe(1);
		const snap = internals.conversationPool.get(conv);
		expect(snap?.kvBytes).not.toBe(sentinel);

		inf.serializeKVCache = origSerialize;
	});
});
