# Chat page (`smoke-test/chat.html`)

A focused, multi-turn chat surface against the registered chat-model
fleet, with live context / TTFT / decode metrics, settings panel, and
single-slot conversation persistence across reloads.

## Run

```bash
make smoke-serve
```
Open `http://localhost:8031/chat.html` in Chrome.

## Manual smoke checklist

1. Pick **TinyLlama 1.1B Chat (Q4_0)**. Verify the load card progresses
   to 100% and ends with `Loaded TinyLlama …`.
2. Set system prompt to "You answer in one sentence." Click **Apply**.
3. Send "Say hi." Verify the user bubble appears immediately, the
   assistant bubble streams text, the **last:** pill populates with
   TTFT / tok/s / wall / output-token count, and the context bar
   advances.
4. Send 2 more turns. Verify the **session:** pill updates and the
   chart toggle reveals a sparkline.
5. Reload the page. Verify the restore card appears with the correct
   turn count and relative timestamp. Click **Resume**. Confirm the
   transcript repopulates.
6. Send turn 4. Confirm TTFT is materially lower than turn 1 with the
   same total token count would have been (KV reuse evidence — visible
   in the console `[chat-page] turn done` payload).
7. Click **Clear conversation**. Verify transcript empties, context
   bar resets to 0%, IndexedDB slot empty (DevTools → Application).
8. Switch model to **Qwen3 0.6B**. Confirm "discard?" dialog. Verify
   thinking toggle appears in Settings; with thinking on, send a
   question and verify the `▸ thinking` block renders as a collapsed
   region above the visible answer.
9. Pick **Mistral 7B Instruct (Q4_K_S)**. Verify the amber VRAM pill
   appears in the load card.
10. Send messages until the context bar turns red, then send one more.
    Verify the overflow error UI offers **Clear conversation** and
    **Export & clear**.

## Known limitations

- Single-slot persistence; opening the page in two tabs at once leads
  to last-writer-wins on the IndexedDB slot.
- No custom-GGUF picker; only registered `BENCHMARK_MODELS` chat
  entries are selectable.
- Model load progress requires a `content-length` header on the GGUF
  response (matches `make smoke-serve` defaults).
