/**
 * Message envelope types and shared helpers for the WebLLM worker bridge.
 *
 * Both `webllm-proxy.ts` (main thread) and `webllm-worker-host.ts` (worker)
 * import from here. Pure type module — no runtime imports of engine code.
 */

import type { GenerationStreamChunk } from "../inference/generation.js";
import type { WebLLMErrorCode } from "./errors.js";
import type { WebLLMConfig } from "./types.js";

export type RequestId = number;
export type StreamId = number;

export interface SerializedError {
	code: WebLLMErrorCode | "GENERIC" | "DISPOSED";
	message: string;
	stack?: string;
	modelId?: string;
	architecture?: string;
	conversationId?: string;
	liveConversationIds?: string[];
	requestedTokens?: number;
	maxContextTokens?: number;
}

export type ProxyToWorker =
	| { type: "init"; id: RequestId; config: Omit<WebLLMConfig, "worker"> }
	| { type: "method-call"; id: RequestId; name: string; args: unknown[] }
	| {
			type: "stream-start";
			streamId: StreamId;
			name: "chatCompletion" | "generateStream";
			args: unknown[];
	  }
	| { type: "stream-cancel"; streamId: StreamId }
	| { type: "dispose"; id: RequestId };

export type WorkerToProxy =
	| { type: "init-done"; id: RequestId }
	| { type: "method-result"; id: RequestId; value: unknown }
	| { type: "method-error"; id: RequestId; error: SerializedError }
	| {
			// Single-chunk variant — retained for backwards compatibility with
			// any external/test code constructing one-off chunk envelopes.
			// Production worker host now coalesces via `stream-chunks` (plural)
			// to reduce postMessage traffic during decode (one event per
			// 16 ms / 8 tokens instead of one per token). The proxy handles
			// both transparently.
			type: "stream-chunk";
			streamId: StreamId;
			chunk: GenerationStreamChunk;
	  }
	| {
			// Coalesced batch variant — produced by webllm-worker-host's
			// stream loop. The proxy fans `chunks` back out to the consumer
			// one-at-a-time so the public `AsyncIterableIterator` surface is
			// unchanged.
			type: "stream-chunks";
			streamId: StreamId;
			chunks: GenerationStreamChunk[];
	  }
	| { type: "stream-done"; streamId: StreamId; value?: unknown }
	| { type: "stream-error"; streamId: StreamId; error: SerializedError }
	| { type: "log"; level: "info" | "warn" | "error"; message: string };

let _nextRequestId = 0;
let _nextStreamId = 0;

export function makeRequestId(): RequestId {
	_nextRequestId += 1;
	return _nextRequestId;
}

export function makeStreamId(): StreamId {
	_nextStreamId += 1;
	return _nextStreamId;
}
