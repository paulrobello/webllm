import { expect, test } from "bun:test";
import {
	buildMatrixPlan,
	getMatrixPreset,
	modelSupportsThinking,
} from "../eval/chat-smoke-matrix.js";

test("buildMatrixPlan expands thinking modes and collapses unsupported models", () => {
	expect(
		buildMatrixPlan({
			models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
			pages: ["smoke", "debug"],
			prompt: "hello",
			thinkingModes: ["off"],
		}),
	).toEqual([
		{
			model: "qwen3-0.6b-q4f16",
			page: "smoke",
			prompt: "hello",
			thinking: "off",
		},
		{
			model: "qwen3-0.6b-q4f16",
			page: "debug",
			prompt: "hello",
			thinking: "off",
		},
		{
			model: "llama-3.2-1b-q4f16",
			page: "smoke",
			prompt: "hello",
			thinking: "off",
		},
		{
			model: "llama-3.2-1b-q4f16",
			page: "debug",
			prompt: "hello",
			thinking: "off",
		},
	]);

	expect(
		buildMatrixPlan({
			models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
			pages: ["smoke"],
			prompt: "hi",
			thinkingModes: ["off", "on"],
		}),
	).toEqual([
		{ model: "qwen3-0.6b-q4f16", page: "smoke", prompt: "hi", thinking: "off" },
		{ model: "qwen3-0.6b-q4f16", page: "smoke", prompt: "hi", thinking: "on" },
		{
			model: "llama-3.2-1b-q4f16",
			page: "smoke",
			prompt: "hi",
			thinking: "off",
		},
	]);
});

test("getMatrixPreset defines named matrix presets with thinking modes", () => {
	expect(getMatrixPreset("fast")).toEqual({
		models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
		pages: ["smoke", "debug"],
		thinkingModes: ["off"],
	});
	expect(getMatrixPreset("full")).toEqual({
		models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
		pages: ["smoke", "debug"],
		thinkingModes: ["off", "on"],
	});
	expect(getMatrixPreset("qwen-only")).toEqual({
		models: ["qwen3-0.6b-q4f16"],
		pages: ["smoke", "debug"],
		thinkingModes: ["off", "on"],
	});
	expect(getMatrixPreset("smoke-only")).toEqual({
		models: ["qwen3-0.6b-q4f16", "llama-3.2-1b-q4f16"],
		pages: ["smoke"],
		thinkingModes: ["off"],
	});
});

test("modelSupportsThinking only matches qwen3 family", () => {
	expect(modelSupportsThinking("qwen3-0.6b-q4f16")).toBe(true);
	expect(modelSupportsThinking("qwen3-4b-q4f16")).toBe(true);
	expect(modelSupportsThinking("qwen2.5-1.5b-q4f16")).toBe(false);
	expect(modelSupportsThinking("llama-3.2-1b-q4f16")).toBe(false);
	expect(modelSupportsThinking("tinyllama-1.1b-chat-q4_0")).toBe(false);
});
