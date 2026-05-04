import type {
	ChatMessage,
	JsonSchemaParameterType,
} from "../core/chat-types.js";
import type { Tokenizer } from "./tokenizer.js";

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
	template?: string,
): string {
	// Llama-2 wraps system content in `<<SYS>>...<</SYS>>` inside the first
	// `[INST]` block. Mistral-Instruct family (v0.1–v0.3) shares the
	// `[INST]…[/INST]` skeleton but has no native system role — its official
	// Jinja template raises an exception when `role === "system"`. The
	// idiomatic workaround is to merge the system content into the first
	// user turn. Detect by whether the original template references the
	// `<<SYS>>` envelope.
	const useSysEnvelope = (template ?? "").includes("<<SYS>>");

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
	if (hasSystem && useSysEnvelope) {
		prompt += `<<SYS>>\n${systemContent}\n<</SYS>>\n\n`;
	} else if (hasSystem) {
		// Mistral-style: system content prefixes the first user message,
		// separated by a blank line.
		prompt += `${systemContent}\n\n`;
	}
	if (turns.length === 0) return prompt;

	// Llama-2 training keeps a trailing space after `[/INST]`. Mistral's
	// official Jinja emits `'[INST] ' + content + ' [/INST]'` — the space
	// goes *before* `[/INST]` only, with no trailing space. Feeding Mistral
	// the trailing-space variant makes the model emit no-leading-space
	// variants of subsequent BPE tokens (e.g. `here` instead of `▁here`),
	// producing missing spaces in the streamed output (`I'mhere`).
	const closeInst = useSysEnvelope ? " [/INST] " : " [/INST]";

	for (let i = 0; i < turns.length; i++) {
		const t = turns[i];
		if (t.role === "user") {
			if (i > 0) {
				prompt += "<s>[INST] ";
			}
			prompt += `${t.content}${closeInst}`;
		} else {
			prompt += `${t.content}</s>`;
		}
	}

	if (addGenerationPrompt && !prompt.endsWith(closeInst)) {
		// already in generation-ready state
	}
	return prompt;
}

/**
 * Minimal tool schema for prompt injection — mirrors `ChatToolSchema` from
 * `core/chat-types.ts`. The parameter `type` union is sourced from
 * `JsonSchemaParameterType` so drift across the three tool-schema mirrors
 * (this one, `ChatToolSchema`, `ToolParameter`) is a typecheck error.
 */
export interface ChatTemplateToolSchema {
	name: string;
	description: string;
	parameters: Record<
		string,
		{
			type: JsonSchemaParameterType;
			description?: string;
			required?: boolean;
		}
	>;
}

export interface ChatTemplateRenderOptions {
	enableThinking?: boolean;
	/**
	 * Tool schemas to inject into the system message for templates that
	 * support tool calling (ChatML-family: Qwen3, Hermes, etc.). Ignored
	 * by templates without tool support.
	 */
	tools?: readonly ChatTemplateToolSchema[];
}

type ChatPromptTokenizer = Pick<Tokenizer, "bosId" | "encode" | "options">;

function shouldCloseThinkBlock(
	template?: string,
	options?: ChatTemplateRenderOptions,
): boolean {
	return (
		options?.enableThinking === false &&
		(template?.includes("enable_thinking") ?? false) &&
		(template?.includes("<think>") ?? false)
	);
}

function formatChatml(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
	template?: string,
	options?: ChatTemplateRenderOptions,
): string {
	const tools = options?.tools ?? [];
	const messagesToEmit =
		tools.length > 0 ? injectToolsIntoSystem(messages, tools) : messages;

	let prompt = "";
	for (const msg of messagesToEmit) {
		prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
	}
	if (addGenerationPrompt) {
		prompt += "<|im_start|>assistant\n";
		if (shouldCloseThinkBlock(template, options)) {
			prompt += "<think>\n\n</think>\n\n";
		}
	}
	return prompt;
}

/**
 * Qwen3 / Hermes tool-calling convention: tools are described in the
 * system message with a `<tools>…</tools>` JSON-lines block plus
 * tool-call instructions. If the caller already has a system message we
 * append to it; otherwise we prepend a new one.
 */
function injectToolsIntoSystem(
	messages: ChatMessage[],
	tools: readonly ChatTemplateToolSchema[],
): ChatMessage[] {
	const toolBlock = buildToolsBlock(tools);
	const out: ChatMessage[] = [];
	let injected = false;
	for (const m of messages) {
		if (!injected && m.role === "system") {
			out.push({ role: "system", content: `${m.content}\n\n${toolBlock}` });
			injected = true;
		} else {
			out.push(m);
		}
	}
	if (!injected) {
		out.unshift({ role: "system", content: toolBlock });
	}
	return out;
}

function buildToolsBlock(tools: readonly ChatTemplateToolSchema[]): string {
	const lines = tools.map((t) =>
		JSON.stringify({
			type: "function",
			function: {
				name: t.name,
				description: t.description,
				parameters: toJsonSchema(t.parameters),
			},
		}),
	);
	return [
		"# Tools",
		"",
		"You may call one or more functions to assist with the user query.",
		"",
		"You are provided with function signatures within <tools></tools> XML tags:",
		"<tools>",
		...lines,
		"</tools>",
		"",
		"For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:",
		"<tool_call>",
		'{"name": "<function-name>", "arguments": <args-json-object>}',
		"</tool_call>",
	].join("\n");
}

function toJsonSchema(
	params: Record<
		string,
		{ type: string; description?: string; required?: boolean }
	>,
): Record<string, unknown> {
	const properties: Record<string, { type: string; description?: string }> = {};
	const required: string[] = [];
	for (const [key, spec] of Object.entries(params)) {
		properties[key] = { type: spec.type };
		if (spec.description) properties[key].description = spec.description;
		if (spec.required) required.push(key);
	}
	return { type: "object", properties, required };
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
		prompt += "<|assistant|>\n";
	}
	return prompt;
}

function formatLlama3(
	messages: ChatMessage[],
	addGenerationPrompt: boolean,
	_template?: string,
	options?: ChatTemplateRenderOptions,
): string {
	const tools = options?.tools ?? [];
	const messagesToEmit =
		tools.length > 0 ? injectToolsIntoSystem(messages, tools) : messages;

	let prompt = "<|begin_of_text|>";
	for (const msg of messagesToEmit) {
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
			prompt += `<|assistant|>\n${msg.content}</s>`;
		} else {
			prompt += `<|${msg.role}|>\n${msg.content}</s>`;
		}
	}
	if (addGenerationPrompt) {
		prompt += "<|assistant|>";
	}
	return prompt;
}

const FORMATTERS: Record<
	ChatTemplateType,
	(
		messages: ChatMessage[],
		addGenerationPrompt: boolean,
		template?: string,
		options?: ChatTemplateRenderOptions,
	) => string
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

const DEFAULT_SYSTEM =
	"You are a helpful assistant. Answer questions directly and concisely.";

function shouldInjectDefaultSystem(template?: string): boolean {
	return !(
		template?.includes("enable_thinking") && template.includes("<think>")
	);
}

/**
 * Format messages into a prompt string using the detected template type.
 * Falls back to zephyr format when the template is unknown or empty.
 * Prepends a default system message if none is present.
 */
export function formatChatPrompt(
	messages: ChatMessage[],
	template?: string,
	options?: ChatTemplateRenderOptions,
): string {
	const tmpl = detectChatTemplate(template ?? "");
	const hasSystem = messages.length > 0 && messages[0].role === "system";
	const injectDefaultSystem = shouldInjectDefaultSystem(template);
	const msgs = hasSystem
		? messages
		: injectDefaultSystem
			? [{ role: "system" as const, content: DEFAULT_SYSTEM }, ...messages]
			: messages;
	return FORMATTERS[tmpl](msgs, true, template, options);
}

export function encodeChatPrompt(
	messages: ChatMessage[],
	tokenizer: ChatPromptTokenizer,
	options?: ChatTemplateRenderOptions,
): number[] {
	const prompt = formatChatPrompt(
		messages,
		tokenizer.options.chatTemplate,
		options,
	);
	const encoded = tokenizer.encode(prompt);
	return tokenizer.options.addBosToken === false
		? encoded
		: [tokenizer.bosId, ...encoded];
}

/** Return only the new portion given the number of previously formatted messages. */
export function formatChatDelta(
	messages: ChatMessage[],
	prevCount: number,
	template?: string,
	options?: ChatTemplateRenderOptions,
): string {
	if (prevCount <= 0) return formatChatPrompt(messages, template, options);
	if (prevCount >= messages.length) return "";
	const tmpl = detectChatTemplate(template ?? "");
	const formatter = FORMATTERS[tmpl] ?? FORMATTERS.zephyr;
	const hasSystem = messages.length > 0 && messages[0].role === "system";
	const injectDefaultSystem = shouldInjectDefaultSystem(template);
	const msgs = hasSystem
		? messages
		: injectDefaultSystem
			? [{ role: "system" as const, content: DEFAULT_SYSTEM }, ...messages]
			: messages;
	const full = formatter(msgs, true, template, options);
	const prev = formatter(
		hasSystem
			? messages.slice(0, prevCount)
			: injectDefaultSystem
				? [
						{ role: "system" as const, content: DEFAULT_SYSTEM },
						...messages.slice(0, prevCount),
					]
				: messages.slice(0, prevCount),
		false,
		template,
		options,
	);
	return full.slice(prev.length);
}
