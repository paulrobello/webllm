import { describe, expect, test } from "bun:test";
import type { ChatMessage } from "../src/core/chat-types.js";
import {
	detectChatTemplate,
	formatChatDelta,
	formatChatPrompt,
} from "../src/inference/chat-template.js";

describe("detectChatTemplate", () => {
	test("returns 'llama2' for [INST] template", () => {
		const tmpl =
			"{% for message in loop_messages %}{% if message['role'] == 'user' %}[INST] {{ content }} [/INST]{% endif %}{% endfor %}";
		expect(detectChatTemplate(tmpl)).toBe("llama2");
	});

	test("returns 'llama2' for <<SYS>> template", () => {
		expect(detectChatTemplate("<<SYS>>")).toBe("llama2");
	});

	test("returns 'chatml' for <|im_start|> template", () => {
		const tmpl =
			"{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
		expect(detectChatTemplate(tmpl)).toBe("chatml");
	});

	test("returns 'gemma' for <start_of_turn> template", () => {
		const tmpl =
			"{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>user\n' }}{% endfor %}";
		expect(detectChatTemplate(tmpl)).toBe("gemma");
	});

	test("returns 'phi3' for <|assistant| + <|end|> template", () => {
		const tmpl =
			"{{ bos_token }}{% for message in messages %}<|assistant|?\n{{ message.content }}<|end|>\n{% endfor %}";
		expect(detectChatTemplate(tmpl)).toBe("phi3");
	});

	test("returns 'llama3' for <|start_header_id|> template", () => {
		const tmpl =
			"{% for message in loop_messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{% endfor %}";
		expect(detectChatTemplate(tmpl)).toBe("llama3");
	});

	test("returns 'mistral-v7' for [SYSTEM_PROMPT] template", () => {
		const tmpl =
			"{%- if messages[0]['role'] == 'system' %}[SYSTEM_PROMPT] {{ system_message }}[/SYSTEM_PROMPT]{%- endif %}";
		expect(detectChatTemplate(tmpl)).toBe("mistral-v7");
	});

	test("returns 'unknown' for empty string", () => {
		expect(detectChatTemplate("")).toBe("unknown");
	});

	test("returns 'unknown' for unrecognized template", () => {
		expect(detectChatTemplate("some random template string")).toBe("unknown");
	});
});

describe("formatChatPrompt (llama2)", () => {
	test("formats single user message", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatPrompt(messages)).toBe("[INST] Hello [/INST] ");
	});

	test("formats system + user message", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "You are helpful." },
			{ role: "user", content: "Hi" },
		];
		expect(formatChatPrompt(messages)).toBe(
			"[INST] <<SYS>>\nYou are helpful.\n<</SYS>>\n\nHi [/INST] ",
		);
	});

	test("formats multi-turn with assistant responses", () => {
		const messages: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		expect(formatChatPrompt(messages)).toBe(
			"[INST] Q1 [/INST] A1</s><s>[INST] Q2 [/INST] ",
		);
	});

	test("ignores duplicate system messages", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "First" },
			{ role: "system", content: "Second" },
			{ role: "user", content: "Hi" },
		];
		expect(formatChatPrompt(messages)).toBe(
			"[INST] <<SYS>>\nFirst\n<</SYS>>\n\nHi [/INST] ",
		);
	});
});

describe("formatChatPrompt (chatml)", () => {
	const tmpl =
		"{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";

	test("formats system + user + generation prompt", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "You are helpful." },
			{ role: "user", content: "Hi" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n",
		);
	});

	test("formats multi-turn conversation", () => {
		const messages: ChatMessage[] = [
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there!" },
			{ role: "user", content: "How are you?" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there!<|im_end|>\n<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n",
		);
	});
});

describe("formatChatPrompt (gemma)", () => {
	const tmpl =
		"{{ bos_token }}{% for message in messages %}{{ '<start_of_turn>user\n' }}{% endfor %}";

	test("formats user message with model generation prompt", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "What is 2+2?" }];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<start_of_turn>user\nWhat is 2+2?<end_of_turn>\n<start_of_turn>model\n",
		);
	});

	test("maps assistant role to model", () => {
		const messages: ChatMessage[] = [
			{ role: "user", content: "Hi" },
			{ role: "assistant", content: "Hello!" },
			{ role: "user", content: "Bye" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n<start_of_turn>user\nBye<end_of_turn>\n<start_of_turn>model\n",
		);
	});
});

describe("formatChatPrompt (phi3)", () => {
	const tmpl =
		"{{ bos_token }}{% for message in messages %}<|assistant|?\n{{ message.content }}<|end|>\n{% endfor %}";

	test("formats system + user messages", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "Be precise." },
			{ role: "user", content: "What is PI?" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<|system|>\nBe precise.<|end|>\n<|user|>\nWhat is PI?<|end|>\n<|assistant|?\n",
		);
	});
});

describe("formatChatPrompt (llama3)", () => {
	const tmpl =
		"{% for message in loop_messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{% endfor %}";

	test("formats multi-role conversation", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "You are a pirate." },
			{ role: "user", content: "Hello" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a pirate.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
		);
	});
});

describe("formatChatPrompt (mistral-v7)", () => {
	const tmpl =
		"{%- if messages[0]['role'] == 'system' %}[SYSTEM_PROMPT] {{ system_message }}[/SYSTEM_PROMPT]{%- endif %}";

	test("formats system + user + assistant", () => {
		const messages: ChatMessage[] = [
			{ role: "system", content: "Be helpful." },
			{ role: "user", content: "Hello" },
			{ role: "assistant", content: "Hi there" },
			{ role: "user", content: "How are you?" },
		];
		expect(formatChatPrompt(messages, tmpl)).toBe(
			"[SYSTEM_PROMPT] Be helpful.[/SYSTEM_PROMPT][INST] Hello[/INST] Hi there</s>[INST] How are you?[/INST]",
		);
	});

	test("formats without system message", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "Hi" }];
		expect(formatChatPrompt(messages, tmpl)).toBe("[INST] Hi[/INST]");
	});
});

describe("formatChatDelta", () => {
	test("returns full prompt when prevCount is 0", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatDelta(messages, 0)).toBe(formatChatPrompt(messages));
	});

	test("returns empty string when prevCount equals messages length", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatDelta(messages, 1)).toBe("");
	});

	test("returns only new portion when messages grow (llama2)", () => {
		const messages: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		const delta = formatChatDelta(messages, 2);
		expect(delta).toBe("<s>[INST] Q2 [/INST] ");
	});

	test("returns empty for prevCount beyond messages length", () => {
		const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
		expect(formatChatDelta(messages, 5)).toBe("");
	});

	test("passes template through to delta formatting", () => {
		const chatmlTmpl =
			"{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}";
		const messages: ChatMessage[] = [
			{ role: "user", content: "Q1" },
			{ role: "assistant", content: "A1" },
			{ role: "user", content: "Q2" },
		];
		const delta = formatChatDelta(messages, 2, chatmlTmpl);
		expect(delta).toBe(
			"<|im_start|>user\nQ2<|im_end|>\n<|im_start|>assistant\n",
		);
	});
});
