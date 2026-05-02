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
			type: "stream-chunk";
			streamId: StreamId;
			chunk: GenerationStreamChunk;
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
