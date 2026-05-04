// chat-conversation.js — conversation handle lifecycle + send-turn loop.

/**
 * Create a fresh conversation handle bound to the currently loaded
 * model + system prompt. The system prompt is re-sent as the first
 * `system` message on every send so the engine's longest-shared-token-
 * prefix walk against the prior KV snapshot succeeds (KV reuse depends
 * on identical prompt tokens).
 */
export async function createChatConversation(engine, model, systemPrompt) {
  const handle = await engine.createConversation(model.id, {
    maxContextTokens: model.contextLength,
  });
  return { handle, modelId: model.id, systemPrompt, messages: [] };
}

export async function disposeChatConversation(engine, conv) {
  if (!conv) return;
  try { await engine.disposeConversation(conv.handle); } catch (_e) { /* tolerate */ }
}

/**
 * Send a single user turn and stream the reply via the supplied
 * callbacks. Returns the per-turn metrics on completion / abort.
 *
 * Callbacks:
 *   onFirstChunk(ms)         — fired once when the first token arrives.
 *   onChunk(text, totalText) — fired for every streamed text chunk.
 *   onDone({ ttftMs, decodeMs, totalMs, outputTokens, finishReason, text, stopped })
 *   onError(err)             — fired on a non-abort thrown error.
 */
export async function sendTurn({
  engine,
  conv,
  userText,
  config,
  signal,
  onFirstChunk,
  onChunk,
  onDone,
  onError,
}) {
  const messages = [
    ...(conv.systemPrompt ? [{ role: "system", content: conv.systemPrompt }] : []),
    ...conv.messages,
    { role: "user", content: userText },
  ];
  conv.messages.push({ role: "user", content: userText });

  const sendTime = performance.now();
  let firstChunkTime = 0;
  let totalText = "";
  let outputTokens = 0;
  let finishReason = "unknown";
  let stopped = false;

  try {
    for await (const chunk of engine.chatCompletion(conv.handle, messages, { ...config, signal })) {
      if (chunk.done) {
        finishReason = chunk.stats?.finishReason ?? "stop";
        outputTokens = chunk.stats?.tokenCount ?? outputTokens;
        break;
      }
      if (!firstChunkTime) {
        firstChunkTime = performance.now();
        onFirstChunk?.(firstChunkTime - sendTime);
      }
      totalText += chunk.text;
      outputTokens += 1;
      onChunk?.(chunk.text, totalText);
    }
  } catch (e) {
    if (e?.name === "AbortError" || signal?.aborted) {
      stopped = true;
    } else {
      onError?.(e);
      return null;
    }
  }

  const endTime = performance.now();
  if (stopped && !firstChunkTime) {
    // Drop the just-pushed user message so persisted history stays
    // monotone (every user message has a corresponding assistant reply).
    conv.messages.pop();
  } else {
    conv.messages.push({ role: "assistant", content: totalText });
  }

  const metrics = {
    ttftMs: firstChunkTime ? firstChunkTime - sendTime : 0,
    decodeMs: firstChunkTime ? endTime - firstChunkTime : 0,
    totalMs: endTime - sendTime,
    outputTokens,
    finishReason,
    text: totalText,
    stopped,
  };
  onDone?.(metrics);
  return metrics;
}
