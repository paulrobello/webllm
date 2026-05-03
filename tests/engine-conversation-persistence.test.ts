import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { WebLLM } from "../src/core/engine.js";
import {
	ConversationBusyError,
	ConversationNotPopulatedError,
	CorruptBlobError,
	IncompatibleConversationError,
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

describe.skipIf(!HAS_WEBGPU || !existsSync(TINYLLAMA))(
	"WebLLM.importConversation",
	() => {
		let webllm: WebLLM;
		let modelId: string;

		beforeAll(async () => {
			webllm = await WebLLM.init({ memoryBudget: 8 * 1024 * 1024 * 1024 });
			const buf = await Bun.file(TINYLLAMA).arrayBuffer();
			const result = await webllm.loadModelFromBuffer(buf, "tinyllama-imp");
			modelId = result.handle.id;
		});

		afterAll(async () => {
			await webllm.dispose();
		});

		test("export → import round-trip yields a fresh handle whose next turn matches a fresh-prefill control under greedy decoding", async () => {
			const convA = await webllm.createConversation(modelId);
			const messages = [{ role: "user" as const, content: "Hello" }];
			for await (const _ of webllm.chatCompletion(convA, messages, {
				temperature: 0,
				maxTokens: 8,
			})) {
				/* drain */
			}
			const blob = await webllm.exportConversation(convA);
			await webllm.disposeConversation(convA);

			const convB = await webllm.importConversation(modelId, blob);
			expect(convB.id).not.toBe(convA.id);
			expect(convB.modelHandleId).toBe(modelId);

			const followUp = [
				...messages,
				{ role: "assistant" as const, content: "" },
				{ role: "user" as const, content: "And again" },
			];
			const collectedB: number[] = [];
			for await (const chunk of webllm.chatCompletion(convB, followUp, {
				temperature: 0,
				maxTokens: 8,
			})) {
				if (typeof (chunk as { tokenId?: number }).tokenId === "number") {
					collectedB.push((chunk as { tokenId: number }).tokenId);
				}
			}

			const convCtrl = await webllm.createConversation(modelId);
			const collectedCtrl: number[] = [];
			for await (const chunk of webllm.chatCompletion(convCtrl, followUp, {
				temperature: 0,
				maxTokens: 8,
			})) {
				if (typeof (chunk as { tokenId?: number }).tokenId === "number") {
					collectedCtrl.push((chunk as { tokenId: number }).tokenId);
				}
			}
			expect(collectedB).toEqual(collectedCtrl);

			await webllm.disposeConversation(convB);
			await webllm.disposeConversation(convCtrl);
		});

		test("import of fingerprint-mismatched blob throws IncompatibleConversationError", async () => {
			const conv = await webllm.createConversation(modelId);
			for await (const _ of webllm.chatCompletion(
				conv,
				[{ role: "user", content: "Hi" }],
				{ temperature: 0, maxTokens: 4 },
			)) {
				/* drain */
			}
			const blob = await webllm.exportConversation(conv);
			const dv = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
			const headerLen = dv.getUint32(4, true);
			const headerJson = new TextDecoder().decode(
				blob.subarray(8, 8 + headerLen),
			);
			const parsed = JSON.parse(headerJson);
			parsed.fingerprint.nLayer = 999;
			const newJson = new TextEncoder().encode(JSON.stringify(parsed));
			const kvBytes = blob.subarray(8 + headerLen);
			const out = new Uint8Array(8 + newJson.byteLength + kvBytes.byteLength);
			out.set(blob.subarray(0, 4), 0);
			new DataView(out.buffer).setUint32(4, newJson.byteLength, true);
			out.set(newJson, 8);
			out.set(kvBytes, 8 + newJson.byteLength);

			await expect(
				webllm.importConversation(modelId, out),
			).rejects.toBeInstanceOf(IncompatibleConversationError);

			await webllm.disposeConversation(conv);
		});

		test("import of corrupt-magic blob throws CorruptBlobError", async () => {
			const blob = new Uint8Array([0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0]);
			await expect(
				webllm.importConversation(modelId, blob),
			).rejects.toBeInstanceOf(CorruptBlobError);
		});
	},
);
