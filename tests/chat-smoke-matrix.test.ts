import { expect, test } from "bun:test";
import { buildMatrixPlan } from "../eval/chat-smoke-matrix.js";

test("buildMatrixPlan expands default smoke pages across models", () => {
	expect(
		buildMatrixPlan({
			models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
			pages: ["smoke", "debug"],
			prompt: "hello",
		}),
	).toEqual([
		{ model: "qwen3-0.6b-q4f16", page: "smoke", prompt: "hello" },
		{ model: "qwen3-0.6b-q4f16", page: "debug", prompt: "hello" },
		{ model: "llama-3.2-1b-q4f16", page: "smoke", prompt: "hello" },
		{ model: "llama-3.2-1b-q4f16", page: "debug", prompt: "hello" },
	]);
});
