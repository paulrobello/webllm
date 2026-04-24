import type {
	ChatMessage,
	CompletionChunk,
	CompletionConfig,
} from "../core/chat-types.js";
import type { ToolDefinition, ToolResult } from "./tool-system.js";
import { ToolSystem } from "./tool-system.js";

/**
 * The slice of the engine that Character needs to drive real inference.
 * Declared as a minimal interface (rather than importing `WebLLM`) so
 * tests can provide a stub and external consumers can plug in any
 * engine that matches this shape.
 */
export interface ChatEngine {
	chatCompletion(
		modelId: string,
		messages: ChatMessage[],
		config?: CompletionConfig,
	): AsyncGenerator<CompletionChunk, void, unknown>;
	/**
	 * Optional: clear conversation state + KV cache for a model. Engines
	 * that maintain per-conversation cache state should implement this so
	 * eval runners can isolate one task from the next.
	 */
	resetConversation?(modelId: string): void;
}

/** Configuration for creating a Character instance. */
export interface CharacterConfig {
	/** Optional custom ID; auto-generated if omitted. */
	id?: string;
	/** Identifier of the model this character uses. */
	modelId: string;
	/** System prompt that defines the character's behavior. */
	systemPrompt: string;
	/** Maximum tokens per generation response. */
	maxTokens?: number;
	/** Sampling temperature. */
	temperature?: number;
	/** Top-P (nucleus) sampling threshold. */
	topP?: number;
	/** Top-K sampling cutoff. */
	topK?: number;
	/** Repetition penalty multiplier. */
	repetitionPenalty?: number;
	/** Custom stop token strings that halt generation. */
	stopTokens?: string[];
	/** Tools the character can invoke from its output. */
	tools?: ToolDefinition[];
	/** Qwen3-style thinking mode; ignored by templates that don't support it. */
	enableThinking?: boolean;
	/**
	 * Engine that fulfills chat() generation. Normally injected by
	 * `WebLLM.createCharacter()`. Omit only in tests — calling `.chat()`
	 * without an engine throws.
	 */
	engine?: ChatEngine;
}

/** A single message in the character's conversation history. */
export interface CharacterMessage {
	/** Role of the message author. */
	role: "system" | "user" | "assistant" | "tool";
	/** Text content of the message. */
	content: string;
	/** Attached tool result for role="tool" messages. */
	toolResult?: ToolResult;
}

/**
 * Stateful chat character with streaming async generator output and optional tool integration.
 *
 * Maintains a conversation history (system, user, assistant, tool messages) and
 * streams character-by-character responses. Supports tool call parsing from output
 * and runtime parameter tuning.
 */
export class Character {
	readonly id: string;
	readonly modelId: string;
	readonly systemPrompt: string;
	readonly config: Required<
		Omit<CharacterConfig, "id" | "tools" | "engine">
	> & {
		tools: ToolDefinition[];
	};
	private engine: ChatEngine | null;

	private messages: CharacterMessage[];
	private toolSystem: ToolSystem | null;
	private active: boolean;
	private stopped: boolean;

	/**
	 * @param config - Character configuration including model, prompt, sampling params, and engine.
	 */
	constructor(config: CharacterConfig) {
		this.id =
			config.id ??
			`character-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
		this.modelId = config.modelId;
		this.systemPrompt = config.systemPrompt;
		this.config = {
			modelId: config.modelId,
			systemPrompt: config.systemPrompt,
			maxTokens: config.maxTokens ?? 256,
			temperature: config.temperature ?? 0.7,
			topP: config.topP ?? 0.9,
			topK: config.topK ?? 40,
			repetitionPenalty: config.repetitionPenalty ?? 1.0,
			stopTokens: config.stopTokens ?? [],
			tools: config.tools ?? [],
			enableThinking: config.enableThinking ?? false,
		};
		this.toolSystem = config.tools?.length
			? new ToolSystem(config.tools)
			: null;
		this.engine = config.engine ?? null;
		this.messages = [{ role: "system", content: this.systemPrompt }];
		this.active = false;
		this.stopped = false;
	}

	/**
	 * Attach (or replace) the engine this character drives. Normally called
	 * once by `WebLLM.createCharacter`; exposed so tests and consumers can
	 * construct a Character first and bind the engine later.
	 */
	attachEngine(engine: ChatEngine): void {
		this.engine = engine;
	}

	/**
	 * Send a user message and stream the character's response.
	 *
	 * @param input - User message text.
	 * @yields Response characters one at a time. If a tool call is detected, yields the formatted tool result instead.
	 */
	async *chat(input: string): AsyncGenerator<string> {
		this.messages.push({ role: "user", content: input });
		if (this.stopped) {
			return;
		}
		if (!this.engine) {
			throw new Error(
				"Character has no engine attached. Use `WebLLM.createCharacter(...)` or pass `engine` in CharacterConfig.",
			);
		}
		this.active = true;

		// ChatMessage doesn't include a "tool" role; tool-result turns live
		// in Character's history only for bookkeeping and are stripped when
		// we replay the conversation to the engine. The preceding assistant
		// turn (which contained the tool call itself) carries the intent.
		const chatMessages: ChatMessage[] = this.messages
			.filter((m) => m.role !== "tool")
			.map((m) => ({
				role: m.role as "system" | "user" | "assistant",
				content: m.content,
			}));
		const completionConfig: CompletionConfig = {
			temperature: this.config.temperature,
			topP: this.config.topP,
			topK: this.config.topK,
			repetitionPenalty: this.config.repetitionPenalty,
			maxTokens: this.config.maxTokens,
			enableThinking: this.config.enableThinking,
			// Surface tools to the chat template so the model is actually told
			// they exist. Handlers/results stay on the Character side; only
			// the schema crosses into the prompt.
			tools:
				this.config.tools.length > 0
					? this.config.tools.map((t) => ({
							name: t.name,
							description: t.description,
							parameters: t.parameters,
						}))
					: undefined,
		};

		let fullResponse = "";
		try {
			for await (const chunk of this.engine.chatCompletion(
				this.modelId,
				chatMessages,
				completionConfig,
			)) {
				if (!this.active) {
					this.stopped = false;
					return;
				}
				if (chunk.text) {
					fullResponse += chunk.text;
					yield chunk.text;
				}
			}
		} finally {
			this.active = false;
		}

		// Check for tool calls in the assembled response. If present, execute
		// and record; otherwise just archive the assistant turn.
		const toolCall = this.toolSystem?.parseToolCall(fullResponse);
		if (toolCall && this.toolSystem) {
			const toolResult = await this.toolSystem.execute(toolCall);
			const formatted = this.toolSystem.formatResult(toolResult);
			this.messages.push({
				role: "assistant",
				content: fullResponse,
			});
			this.messages.push({
				role: "tool",
				content: formatted,
				toolResult,
			});
			return;
		}

		this.messages.push({ role: "assistant", content: fullResponse });
	}

	/**
	 * @returns Read-only snapshot of the full conversation history.
	 */
	getHistory(): readonly CharacterMessage[] {
		return this.messages;
	}

	/** Reset the conversation history, keeping only the system prompt. */
	clearHistory(): void {
		this.messages = [{ role: "system", content: this.systemPrompt }];
	}

	/** Immediately halt any in-progress streaming response. */
	stop(): void {
		this.stopped = true;
		this.active = false;
	}

	/**
	 * Update the sampling temperature at runtime.
	 *
	 * @param t - New temperature value.
	 */
	setTemperature(t: number): void {
		this.config.temperature = t;
	}

	/**
	 * Update the top-P sampling parameter at runtime.
	 *
	 * @param p - New top-P value.
	 */
	setTopP(p: number): void {
		this.config.topP = p;
	}

	/**
	 * Update the maximum token generation limit at runtime.
	 *
	 * @param n - New max token count.
	 */
	setMaxTokens(n: number): void {
		this.config.maxTokens = n;
	}

	get isActive(): boolean {
		return this.active;
	}

	get messageCount(): number {
		return this.messages.length;
	}
}
