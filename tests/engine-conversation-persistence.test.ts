import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { WebLLM } from "../src/core/engine.js";
import {
	ConversationBusyError,
	ConversationNotPopulatedError,
} from "../src/core/errors.js";
import {
	decodePersistedConversation,
	KV_PERSISTENCE_MAGIC,
} from "../src/core/persistence.js";

const HAS_WEBGPU = typeof navigator !== "undefined" && "gpu" in navigator;
const TINYLLAMA =
	process.env.WEBLLM_TINYLLAMA_GGUF ??
	"fixtures/tinyllama-1.1b-chat-v1.0-q4_0.gguf";

describe.skipIf(!HAS_WEBGPU || !existsSync(TINYLLAMA))(
	"WebLLM.exportConversation",
	() => {
		let webllm: WebLLM;
		let modelId: string;

		beforeAll(async () => {
			webllm = await WebLLM.init({ memoryBudget: 8 * 1024 * 1024 * 1024 });
			const buf = await Bun.file(TINYLLAMA).arrayBuffer();
			const result = await webllm.loadModelFromBuffer(buf, "tinyllama");
			modelId = result.handle.id;
		});

		afterAll(async () => {
			await webllm.dispose();
		});

		test("export after one chatCompletion produces a blob with correct magic + parseable header", async () => {
			const conv = await webllm.createConversation(modelId);
			for await (const _ of webllm.chatCompletion(
				conv,
				[{ role: "user", content: "Hi" }],
				{ temperature: 0, maxTokens: 4 },
			)) {
				/* drain */
			}
			const blob = await webllm.exportConversation(conv);
			expect(blob).toBeInstanceOf(Uint8Array);
			expect(blob.byteLength).toBeGreaterThan(8);
			expect(blob.slice(0, 4)).toEqual(KV_PERSISTENCE_MAGIC);

			const fingerprint = (
				webllm as unknown as {
					_modelManager: { get(id: string): { fingerprint: unknown } };
				}
			)._modelManager.get(modelId).fingerprint;
			const { header, kvBytes } = decodePersistedConversation(
				blob,
				fingerprint as never,
			);
			expect(header.schemaVersion).toBe(1);
			expect(header.tokenIds.length).toBeGreaterThan(0);
			expect(kvBytes.byteLength).toBe(header.byteSize);
			await webllm.disposeConversation(conv);
		});

		test("export of un-populated conv throws ConversationNotPopulatedError", async () => {
			const conv = await webllm.createConversation(modelId);
			await expect(webllm.exportConversation(conv)).rejects.toBeInstanceOf(
				ConversationNotPopulatedError,
			);
			await webllm.disposeConversation(conv);
		});

		test("concurrent export while chatCompletion is in-flight throws ConversationBusyError", async () => {
			const conv = await webllm.createConversation(modelId);
			const stream = webllm.chatCompletion(
				conv,
				[{ role: "user", content: "Test" }],
				{ temperature: 0, maxTokens: 8 },
			);
			const iter = stream[Symbol.asyncIterator]();
			await iter.next();
			await expect(webllm.exportConversation(conv)).rejects.toBeInstanceOf(
				ConversationBusyError,
			);
			for (;;) {
				const r = await iter.next();
				if (r.done) break;
			}
			await webllm.disposeConversation(conv);
		});
	},
);
