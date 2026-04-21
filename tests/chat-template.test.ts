import { describe, expect, test } from "bun:test";
import type { ChatMessage } from "../src/core/chat-types.js";
import {
	detectChatTemplate,
	formatChatDelta,
	formatChatPrompt,
} from "../src/inference/chat-template.js";

const LLAMA2_TMPL =
	"{% if message['role'] == 'user' %}[INST] {{ content }} [/INST]{% endif %}";
const CHATML_TMPL =
	"{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
const GEMMA_TMPL =
	"{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>user\n' }}{% endfor %}";
const PHI3_TMPL =
	"{{ bos_token }}{% for message in messages %}<|assistant|?\n{{ message.content }}<|end|>\n{% endfor %}";
const LLAMA3_TMPL =
	"{% for message in loop_messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{% endfor %}";
const MISTRAL_TMPL =
	"{%- if messages[0]['role'] == 'system' %}[SYSTEM_PROMPT] {{ system_message }}[/SYSTEM_PROMPT]{%- endif %}";
const ZEPHYR_TMPL =
	"{% for message in messages %}{% if message['role'] == 'assistant' %}{{ '<|assistant|' + message['content'] + eos_token }}{% endif %}{% endfor %}";

describe("detectChatTemplate", () => {
	test("returns llama2 for INST", () => {
		expect(detectChatTemplate(LLAMA2_TMPL)).toBe("llama2");
	});
	test("returns llama2 for SYS", () => {
		expect(detectChatTemplate("<<SYS>>")).toBe("llama2");
	});
	test("returns chatml", () => {
		expect(detectChatTemplate(CHATML_TMPL)).toBe("chatml");
	});
	test("returns gemma", () => {
		expect(detectChatTemplate(GEMMA_TMPL)).toBe("gemma");
	});
	test("returns zephyr for assistant without end", () => {
		expect(detectChatTemplate(ZEPHYR_TMPL)).toBe("zephyr");
	});
	test("returns phi3 for assistant with end", () => {
		expect(detectChatTemplate(PHI3_TMPL)).toBe("phi3");
	});
	test("returns llama3", () => {
		expect(detectChatTemplate(LLAMA3_TMPL)).toBe("llama3");
	});
	test("returns mistral-v7", () => {
		expect(detectChatTemplate(MISTRAL_TMPL)).toBe("mistral-v7");
	});
	test("returns unknown for empty", () => {
		expect(detectChatTemplate("")).toBe("unknown");
	});
	test("returns unknown for garbage", () => {
		expect(detectChatTemplate("xyz")).toBe("unknown");
	});
});

describe("formatChatPrompt llama2", () => {
	test("single user", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatPrompt(m, LLAMA2_TMPL)).toBe("[INST] Hello [/INST] ");
	});
	test("system + user", () => {
		const m: ChatMessage[] = [
			{ role: "system", content: "Sys" },
			{ role: "user", content: "Hi" },
		];
		expect(formatChatPrompt(m, LLAMA2_TMPL)).toBe(
			"[INST] <<SYS>>\nSys\n<</SYS>>\n\nHi [/INST] ",
		);
	});
	test("multi-turn", () => {
		const m: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		expect(formatChatPrompt(m, LLAMA2_TMPL)).toBe(
			"[INST] Q1 [/INST] A1</s><s>[INST] Q2 [/INST] ",
		);
	});
});

describe("formatChatPrompt zephyr", () => {
	test("single user", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatPrompt(m, ZEPHYR_TMPL)).toBe(
			"<|user|>\nHello</s><|assistant|>",
		);
	});
	test("system + user", () => {
		const m: ChatMessage[] = [
			{ role: "system", content: "Be nice" },
			{ role: "user", content: "Hi" },
		];
		expect(formatChatPrompt(m, ZEPHYR_TMPL)).toBe(
			"<|system|>\nBe nice</s><|user|>\nHi</s><|assistant|>",
		);
	});
	test("multi-turn", () => {
		const m: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		expect(formatChatPrompt(m, ZEPHYR_TMPL)).toBe(
			"<|user|>\nQ1</s><|assistant|>\nA1</s><|user|>\nQ2</s><|assistant|>",
		);
	});
});

describe("formatChatPrompt mistral-v7", () => {
	test("system + multi-turn", () => {
		const m: ChatMessage[] = [
			{ role: "system", content: "Be helpful." },
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
			{ role: "user", content: "Bye" },
		];
		expect(formatChatPrompt(m, MISTRAL_TMPL)).toBe(
			"[SYSTEM_PROMPT] Be helpful.[/SYSTEM_PROMPT][INST] Hello[/INST] Hi there</s>[INST] Bye[/INST]",
		);
	});
	test("no system", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hi" }];
		expect(formatChatPrompt(m, MISTRAL_TMPL)).toBe("[INST] Hi[/INST]");
	});
});

describe("formatChatDelta", () => {
	test("prevCount 0 returns full", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatDelta(m, 0, ZEPHYR_TMPL)).toBe(
			formatChatPrompt(m, ZEPHYR_TMPL),
		);
	});
	test("prevCount == length returns empty", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatDelta(m, 1, ZEPHYR_TMPL)).toBe("");
	});
	test("delta with chatml", () => {
		const m: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		expect(formatChatDelta(m, 2, CHATML_TMPL)).toBe(
			"<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n",
		);
	});
	test("prevCount > length returns empty", () => {
		const m: ChatMessage[] = [{ role: "user", content: "Hi" }];
		expect(formatChatDelta(m, 5, ZEPHYR_TMPL)).toBe("");
	});
});
