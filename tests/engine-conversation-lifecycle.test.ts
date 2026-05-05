import { describe, expect, test } from "bun:test";
import { ConversationPool } from "../src/core/conversation-pool.js";
import {
	ConversationNotFoundError,
	ModelNotFoundError,
	ModelNotLoadedError,
} from "../src/core/errors.js";
import { WebLLM } from "../src/index.js";

type EngineInternals = {
	_modelManager: {
		get(id: string): unknown;
		unregister(id: string): Promise<void>;
		clear(): void;
	};
	inferenceEngines: Map<string, unknown>;
	encoderEngines: Map<string, unknown>;
	causalEmbedderEngines: Map<string, unknown>;
	wasmModules: Map<string, unknown>;
	conversationPool: ConversationPool;
	sessions: Map<string, unknown>;
	_scheduler: { clear(): void };
	eventHandlers: Map<string, Set<unknown>>;
};

function asInternals(engine: WebLLM): EngineInternals {
	return engine as unknown as EngineInternals;
}

function createFakeEngine(opts?: {
	flashAttn?: boolean;
	loaded?: boolean;
	hasInferenceEngine?: boolean;
	maxConversations?: number;
}): WebLLM {
	const flashAttn = opts?.flashAttn ?? true;
	const loaded = opts?.loaded ?? true;
	const hasInf = opts?.hasInferenceEngine ?? true;
	const max = opts?.maxConversations ?? 4;

	const engine = Object.create(WebLLM.prototype) as WebLLM;
	const internals = asInternals(engine);
	internals._modelManager = {
		get: (id: string) => {
			if (id === "tl") {
				return {
					id: "tl",
					loaded,
					tokenizer: loaded ? {} : null,
					hyperparams: { architecture: "llama" },
				};
			}
			return undefined;
		},
		unregister: async () => {},
		clear: () => {},
	};
	internals.inferenceEngines = new Map<string, unknown>(
		hasInf
			? [
					[
						"tl",
						{
							flashAttn,
							maxContextLength: 2048,
							dispose: async () => {},
						},
					],
				]
			: [],
	);
	internals.encoderEngines = new Map();
	internals.causalEmbedderEngines = new Map();
	internals.wasmModules = new Map();
	internals.conversationPool = new ConversationPool({ maxConversations: max });
	internals.sessions = new Map();
	internals._scheduler = { clear: () => {} };
	internals.eventHandlers = new Map();
	return engine;
}

describe("engine conversation lifecycle", () => {
	test("createConversation returns a fresh handle for a loaded model", async () => {
		const engine = createFakeEngine();
		const conv = await engine.createConversation("tl");
		expect(conv.modelHandleId).toBe("tl");
		expect(conv.id).toMatch(/^conv_/);
		await engine.disposeConversation(conv);
	});

	test("createConversation on missing model throws ModelNotFoundError", async () => {
		const engine = createFakeEngine();
		expect(engine.createConversation("nope")).rejects.toThrow(
			ModelNotFoundError,
		);
	});

	test("createConversation on unloaded model throws ModelNotLoadedError", async () => {
		const engine = createFakeEngine({ loaded: false });
		expect(engine.createConversation("tl")).rejects.toThrow(
			ModelNotLoadedError,
		);
	});

	test("createConversation rejects manual-mode model", async () => {
		const engine = createFakeEngine({ flashAttn: false });
		expect(engine.createConversation("tl")).rejects.toThrow(/FA mode/i);
	});

	test("disposeConversation is idempotent", async () => {
		const engine = createFakeEngine();
		const conv = await engine.createConversation("tl");
		await engine.disposeConversation(conv);
		expect(engine.disposeConversation(conv)).resolves.toBeUndefined();
	});

	test("createConversation at cap evicts LRU non-locked entry", async () => {
		const engine = createFakeEngine({ maxConversations: 2 });
		const a = await engine.createConversation("tl");
		const b = await engine.createConversation("tl");
		// `a` is the oldest non-locked entry; create() should evict it.
		const c = await engine.createConversation("tl");
		// `a` is gone, `b` and `c` survive.
		const internals = asInternals(engine);
		expect(() => internals.conversationPool.assertExists(a)).toThrow();
		expect(() => internals.conversationPool.assertExists(b)).not.toThrow();
		expect(() => internals.conversationPool.assertExists(c)).not.toThrow();
		await engine.disposeConversation(b);
		await engine.disposeConversation(c);
	});

	test("unloadModel disposes all conversations attached to that model", async () => {
		const engine = createFakeEngine();
		const conv = await engine.createConversation("tl");
		// Spy on the pool to confirm disposeAllForModel was called.
		const internals = asInternals(engine);
		let disposed = "";
		const orig = internals.conversationPool.disposeAllForModel.bind(
			internals.conversationPool,
		);
		internals.conversationPool.disposeAllForModel = (id: string) => {
			disposed = id;
			orig(id);
		};
		await engine.unloadModel("tl");
		expect(disposed).toBe("tl");
		// After unload, the conv handle's pool entry is gone — assertExists throws.
		expect(() => internals.conversationPool.assertExists(conv)).toThrow(
			ConversationNotFoundError,
		);
	});

	test("shutdown clears conversation snapshots held by the pool", async () => {
		const engine = createFakeEngine();
		const conv = await engine.createConversation("tl");
		const internals = asInternals(engine);
		internals.conversationPool.set(conv, {
			conversationId: conv.id,
			modelHandleId: conv.modelHandleId,
			tokenIds: [1, 2, 3],
			kvBytes: new Uint8Array(1024),
			byteSize: 1024,
			lastAccessMs: 0,
		});

		await engine.shutdown();

		expect(() => internals.conversationPool.assertExists(conv)).toThrow(
			ConversationNotFoundError,
		);
	});
});
