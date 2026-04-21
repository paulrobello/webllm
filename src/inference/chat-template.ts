import type { ChatMessage } from "../core/chat-types.js";

export type ChatTemplateType =
	| "llama2"
	| "chatml"
	| "gemma"
	| "phi3"
	| "llama3"
	| "mistral-v7"
	| "zephyr"
	| "unknown";

/**
 * Detect the chat template type from a GGUF template string.
 * Uses the same substring-matching approach as llama.cpp's llm_chat_detect_template().
 */
export function detectChatTemplate(template: string): ChatTemplateType {
	if (!template) return "unknown";
	if (template.includes("<|im_start|>")) return "chatml";
	if (template.includes("[SYSTEM_PROMPT]")) return "mistral-v7";
	if (template.includes("<start_of_turn>")) return "gemma";
	if (template.includes("<|assistant|") && !template.includes("<|end|>"))
		return "zephyr";
	if (template.includes("<|assistant|") && template.includes("<|end|>"))
		return "phi3";
	if (template.includes("<|start_header_id|>")) return "llama3";
	if (template.includes("[INST]") || template.includes("<<SYS>>"))
		return "llama2";
	return "unknown";
}

// --- Per-format formatters ---

function formatLlama2(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let systemContent = "";
	let hasSystem = false;
	const turns: Array<{ role: string; content: string }> = [];

	for (const msg of messages) {
		if (msg.role === "system") {
			if (!hasSystem) {
				systemContent = msg.content;
				hasSystem = true;
			}
		} else {
			turns.push(msg);
		}
	}

	let prompt = "[INST] ";
	if (hasSystem) {
		prompt += `<<SYS>>\n${systemContent}\n<</SYS>>\n\n`;
	}
	if (turns.length === 0) return prompt;

	for (let i = 0; i < turns.length; i++) {
		const t = turns[i];
		if (t.role === "user") {
			if (i > 0) {
				prompt += "<s>[INST] ";
			}
			prompt += `${t.content} [/INST] `;
		} else {
			prompt += `${t.content}</s>`;
		}
	}

	if (addGenerationPrompt && !prompt.endsWith(" [/INST] ")) {
		// already in generation-ready state
	}
	return prompt;
}

function formatChatml(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
	}
	if (addGenerationPrompt) {
		prompt += "<|im_start|>assistant\n";
	}
	return prompt;
}

function formatGemma(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		const role = msg.role === "assistant" ? "model" : msg.role;
		prompt += `<start_of_turn>${role}\n${msg.content}<end_of_turn>\n`;
	}
	if (addGenerationPrompt) {
		prompt += "<start_of_turn>model\n";
	}
	return prompt;
}

function formatPhi3(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		prompt += `<|${msg.role}|>\n${msg.content}<|end|>\n`;
	}
	if (addGenerationPrompt) {
		prompt += "<|assistant|?\n";
	}
	return prompt;
}

function formatLlama3(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "<|begin_of_text|>";
	for (const msg of messages) {
		prompt += `<|start_header_id|>${msg.role}<|end_header_id|>\n\n${msg.content}<|eot_id|>`;
	}
	if (addGenerationPrompt) {
		prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n";
	}
	return prompt;
}

function formatMistralV7(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		if (msg.role === "system") {
			prompt += `[SYSTEM_PROMPT] ${msg.content}[/SYSTEM_PROMPT]`;
		} else if (msg.role === "user") {
			prompt += `[INST] ${msg.content}[/INST]`;
		} else {
			prompt += ` ${msg.content}</s>`;
		}
	}
	if (addGenerationPrompt) {
		// No explicit generation prompt for mistral-v7
	}
	return prompt;
}

/**
 * Zephyr format (used by TinyLlama-Chat, HuggingFaceH4/zephyr, etc.)
 *
 * Uses pipe-delimited role markers:
 *   <|system|>\n{content}</s>
 *   <|user|>\n{content}</s>
 *   <|assistant|{content}</s>
 * Generation prompt: <|assistant|   (no newline)
 */
function formatZephyr(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
): string {
	let prompt = "";
	for (const msg of messages) {
		if (msg.role === "assistant") {
			prompt += `<|assistant|${msg.content}</s>`;
		} else {
			prompt += `<|${msg.role}|>\n${msg.content}</s>`;
		}
	}
	if (addGenerationPrompt) {
		prompt += "<|assistant|";
	}
	return prompt;
}

const FORMATTERS: Record<
	ChatTemplateType,
	(messages: ChatMessage[], addGenerationPrompt: boolean) => string
> = {
	llama2: formatLlama2,
	chatml: formatChatml,
	gemma: formatGemma,
	phi3: formatPhi3,
	llama3: formatLlama3,
	"mistral-v7": formatMistralV7,
	zephyr: formatZephyr,
	unknown: formatZephyr, // fallback — most common for small models
};

/**
 * Format messages into a prompt string using the detected template type.
 * Falls back to zephyr format when the template is unknown or empty.
 */
export function formatChatPrompt(
	messages: ChatMessage[],
	template?: string,
): string {
	const tmpl = detectChatTemplate(template ?? "");
	return FORMATTERS[tmpl](messages, true);
}

/** Return only the new portion given the number of previously formatted messages. */
export function formatChatDelta(
	messages: ChatMessage[],
	prevCount: number,
	template?: string,
): string {
	if (prevCount <= 0) return formatChatPrompt(messages, template);
	if (prevCount >= messages.length) return "";
	const tmpl = detectChatTemplate(template ?? "");
	const formatter = FORMATTERS[tmpl] ?? FORMATTERS.zephyr;
	const full = formatter(messages, true);
	const prev = formatter(messages.slice(0, prevCount), false);
	return full.slice(prev.length);
}
