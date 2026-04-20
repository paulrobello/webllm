import type { ToolDefinition, ToolResult } from "./tool-system.js";
import { ToolSystem } from "./tool-system.js";

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
	readonly config: Required<Omit<CharacterConfig, "id" | "tools">> & {
		tools: ToolDefinition[];
	};

	private messages: CharacterMessage[];
	private toolSystem: ToolSystem | null;
	private active: boolean;
	private stopped: boolean;

	/**
	 * @param config - Character configuration including model, prompt, and sampling params.
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
		};
		this.toolSystem = config.tools?.length
			? new ToolSystem(config.tools)
			: null;
		this.messages = [{ role: "system", content: this.systemPrompt }];
		this.active = false;
		this.stopped = false;
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
		this.active = true;

		const fullResponse = `[Character ${this.id}]: Response to "${input}"`;

		const toolCall = this.toolSystem?.parseToolCall(fullResponse);
		if (toolCall && this.toolSystem) {
			const toolResult = await this.toolSystem.execute(toolCall);
			const formatted = this.toolSystem.formatResult(toolResult);
			this.messages.push({
				role: "tool",
				content: formatted,
				toolResult,
			});
			this.stopped = false;
			this.active = false;
			yield formatted;
			return;
		}

		for (const char of fullResponse) {
			if (!this.active) {
				this.stopped = false;
				return;
			}
			yield char;
		}

		this.messages.push({ role: "assistant", content: fullResponse });
		this.stopped = false;
		this.active = false;
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
