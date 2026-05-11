import { describe, expect, it } from "bun:test";
import {
	detectChatTemplate,
	formatGemma4,
} from "../../src/inference/chat-template.js";

describe("formatGemma4", () => {
	it("emits <start_of_turn>user...<end_of_turn>", () => {
		const out = formatGemma4(
			[{ role: "user", content: "Hello." }],
			/* addGenerationPrompt */ true,
		);
		expect(out).toBe(
			"<start_of_turn>user\nHello.<end_of_turn>\n" + "<start_of_turn>model\n",
		);
	});

	it("maps assistant role to 'model'", () => {
		const out = formatGemma4(
			[
				{ role: "user", content: "Hi" },
				{ role: "assistant", content: "Hey" },
			],
			false,
		);
		expect(out).toContain("<start_of_turn>model\nHey<end_of_turn>");
	});

	it("does not append generation prompt when flag is false", () => {
		const out = formatGemma4([{ role: "user", content: "Q" }], false);
		expect(out).not.toContain("<start_of_turn>model\n");
		expect(out.endsWith("<end_of_turn>\n")).toBe(true);
	});

	// Unsloth's Gemma-4-E2B-it-GGUF (and the broader Gemma-3N family) ships
	// a vocab that stores the turn-boundary special tokens under
	// `<|turn>` / `<turn|>` instead of the classical literals. The template
	// detection still resolves to "gemma4" but the formatter must emit the
	// matching literals or the tokenizer BPE-splits each into ~7 garbage
	// tokens and the model produces degenerate output.
	it("emits <|turn>...<turn|> when template uses the unsloth variant", () => {
		const unslothTemplate =
			"{%- macro format_parameters(...) -%}{{ '<|turn>user\\n' }}...{{ '<turn|>\\n' }}";
		const out = formatGemma4(
			[{ role: "user", content: "Hello." }],
			/* addGenerationPrompt */ true,
			unslothTemplate,
		);
		expect(out).toBe("<|turn>user\nHello.<turn|>\n" + "<|turn>model\n");
	});
});

describe("detectChatTemplate — gemma4", () => {
	// Gemma 4's GGUF chat template starts with the format_parameters macro
	// for tool calls; absent tools the plain message branch uses the same
	// <start_of_turn>...<end_of_turn> markers as Gemma 2.
	it("returns 'gemma4' when template contains '{% macro format_parameters'", () => {
		const tmpl =
			"{%- macro format_parameters(properties, required, filter_keys=false) -%}<start_of_turn>user\n{{ messages[0].content }}<end_of_turn>";
		expect(detectChatTemplate(tmpl)).toBe("gemma4");
	});

	it("still returns 'gemma' for Gemma 2's simpler template", () => {
		const tmpl =
			"<start_of_turn>{{ messages[0].role }}\n{{ messages[0].content }}<end_of_turn>";
		expect(detectChatTemplate(tmpl)).toBe("gemma");
	});
});
