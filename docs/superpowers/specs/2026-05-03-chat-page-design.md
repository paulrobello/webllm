# Dedicated chat page — design spec

> **Date:** 2026-05-03
> **Topic:** A dedicated, user-facing chat page (`smoke-test/chat.html`) that
> lets you select a registered chat model, set a system prompt, hold a multi-
> turn conversation against the local engine, and surface live metrics
> (context-window utilization, time-to-first-token, decode throughput).
> **Scope:** ships in `smoke-test/`, served by the existing `make smoke-serve`
> on port 8031, reusing the existing `webllm-bundle.js` / `webllm-wasm.js`
> assets. No bundler / build-step changes.

## Goals

1. Be a focused, debugging-grade chat surface — distinct from
   `real-model.html` which mixes chat with bench / probe / scenario flows.
2. Exercise the public chat-completion API end-to-end via the
   `ConversationHandle` path (KV-cache reuse turn-over-turn).
3. Surface metrics that make the engine's behavior legible: live
   context-window utilization, per-turn TTFT and decode tok/s, session
   totals, and a 20-turn rolling sparkline.
4. Persist the in-progress conversation across reloads using the shipped
   `exportConversation` / `importConversation` + `IndexedDBConversationStore`
   API (single-slot auto-save).
5. Keep the existing smoke / bench / dashboard surfaces unchanged.

## Non-goals (v1)

- Custom GGUF file picker. Only registered models from `eval/models.ts`.
- Multiple saved conversations / sidebar. Single auto-saved slot only.
- Auto-summarize-and-continue on context overflow. Honest error → user
  resets.
- Concurrent in-flight turns (handled by the engine's busy semantics —
  the UI just disables Send during generation).
- Multi-tab synchronization beyond "last writer wins" on the IndexedDB
  slot.
- Model load progress streaming via service worker / range cache. We rely
  on whatever `loadModelFromUrl` already exposes today.

## Files

All new files live in `smoke-test/`, mirroring the
`real-model.html` / `real-model-page.js` / `real-model.css` triple.

- **`smoke-test/chat.html`** — minimal shell, imports `chat-page.js`,
  links `chat.css`. Cache-busting query suffix is forwarded to the
  imported assets the same way `real-model.html` does.
- **`smoke-test/chat-page.js`** — the page module. Owns DOM, model
  lifecycle, conversation lifecycle, streaming, persistence, metrics,
  rendering. Plain ESM. Imports `WebLLM`, `BENCHMARK_MODELS`, types
  from `./webllm-bundle.js`.
- **`smoke-test/chat.css`** — page styles. Owns layout + theming. The
  frontend-design skill is responsible for the visual treatment (see
  Section 8).
- **`smoke-test/vendor/marked.min.js`** — markdown renderer. ESM.
  Pinned version recorded in a sibling `vendor/README.md` entry.
- **`smoke-test/vendor/highlight.min.js`** + **`highlight-common.min.js`** —
  syntax highlighter + common-language pack (TS / JS / Python / Bash /
  JSON / Markdown). Same vendor pattern.
- **`smoke-test/index.html`** (edit) — add a single link to `chat.html`
  alongside the existing entries.

### Catalog plumbing

`BENCHMARK_MODELS` lives in `eval/models.ts` and is **not** re-exported
through `src/index.ts`, so it does not ship in the existing
`smoke-test/webllm-bundle.js`. The page needs the catalog at runtime.

Approach: build a second tiny bundle next to the main one.

- **New file:** `smoke-test/webllm-models.js`, produced by
  `bun build eval/models.ts --outfile smoke-test/webllm-models.js
  --target browser`.
- **Makefile change:** add the new `bun build` line to the existing
  `smoke-test:` target in `Makefile` (right after the
  `webllm-bundle.js` line). One line, same target, same workflow.
- **Page import:** `import { BENCHMARK_MODELS } from
  "./webllm-models.js"`.

Rationale for the second-bundle approach over re-exporting through
`src/index.ts`: the catalog imports nothing engine-side beyond a
`ModelArchitecture` type alias from `src/core/types.ts`, but
re-exporting from `src/index.ts` would mean the published library
package suddenly ships a `BENCHMARK_MODELS` constant, which is an
eval-harness concern, not a public-library concern. A separate bundle
keeps the public API surface unchanged.

No changes to `webllm-bundle.js`, `webllm-wasm.js`, the engine source,
the eval harness, or the dashboard.

## UI structure

```
┌─ Header bar ──────────────────────────────────────────────┐
│  WebLLM Chat       [Model ▾]  [⚙ Settings]                │
└───────────────────────────────────────────────────────────┘
┌─ System prompt (collapsible, expanded by default) ────────┐
│  System: [textarea]                          [Apply]      │
└───────────────────────────────────────────────────────────┘
┌─ Transcript (scrolls) ────────────────────────────────────┐
│  user: …                                                  │
│  assistant: …  ▸ thinking (collapsed)                     │
│  …                                                        │
└───────────────────────────────────────────────────────────┘
┌─ Composer ────────────────────────────────────────────────┐
│  [textarea — Shift+Enter newline, Enter send]   [Send]    │
│                                                 [Stop]    │  (visible only mid-gen)
└───────────────────────────────────────────────────────────┘
┌─ Status strip ────────────────────────────────────────────┐
│  ▰▰▰▰▱▱▱  context 1,842/4,096 (45%)                       │
│  last: 312ms TTFT · 28.4 tok/s · 1.7s · 48 tok            │
│  session: 6 turns · 312 tok · ⌀ 27.9 tok/s    [chart ▾]   │
└───────────────────────────────────────────────────────────┘
[Clear conversation]   [⤓ Export]
```

### Buttons & state-reset semantics

- **Clear conversation** — disposes the current `ConversationHandle`,
  clears the IndexedDB persistence slot, clears the transcript, keeps
  the same model loaded and the same system prompt. The next Send
  creates a fresh handle under the existing model + system prompt.
- **Export** — downloads `exportConversation(handle)` as JSON. Useful
  for sharing transcripts or seeding tests; not required for any other
  feature.
- **Auto-clear triggers** (no extra button):
  - Switching model in the dropdown → confirm "discard current
    conversation?" → reload model + fresh conversation.
  - Editing the system prompt and clicking **Apply** → confirm
    "discard current conversation?" → fresh conversation under the
    new system prompt.

### Settings panel

Collapsed by default. Contents:

- **Temperature** (slider, 0.0 – 2.0, default from
  `sampling-profiles.ts` for the selected family).
- **Top-k** (number, default from family profile).
- **Top-p** (slider, 0.0 – 1.0, default from family profile).
- **Max output tokens** (number, default 512, capped at remaining
  context).
- **Seed** (number, blank = random).
- **Thinking mode** (toggle, Qwen3 only — gates use of
  `QWEN_THINKING_DEFAULTS` vs `QWEN_NON_THINKING_DEFAULTS`).
- **Reset to defaults** (button — re-pulls defaults for the current
  model family).

Changes apply per-turn, not retroactively. Changing settings does
**not** invalidate the conversation handle — sampling parameters are
pure decode-time knobs and do not touch the KV cache.

## Model lifecycle

### Catalog & filtering

- Source: `BENCHMARK_MODELS` imported from `./webllm-models.js`
  (the second bundle, see "Catalog plumbing" above).
- Filter: `m => !m.capabilities.embedding && m.architecture !== "bert"`.
  This excludes encoders and pure embedders. Chat models flagged
  `embeddingCapable: true` (bucket D) remain in scope as chat models.
- Group by `family`; sort by `paramsB` ascending within family.
- Each option label: `<name> · <paramsB>B · <defaultQuant> · ~<vramMB> MB`.

### Selecting a model

1. If a model is currently loaded:
   - Confirm "discard current conversation?" if there's unsent or
     unfinished transcript content.
   - Dispose the conversation handle (`engine.disposeConversation`).
   - Clear the IndexedDB persistence slot.
   - Drop the model handle (engine reclaims GPU memory).
2. Render a load card with: filename, total bytes, progress bar,
   MB/s, ETA. Populated from `loadModelFromUrl`'s existing byte-progress
   callback (same one the smoke page uses).
3. On load success: persist the chosen `modelId` to `localStorage` key
   `chat:lastModelId`. Do **not** auto-load on next visit — wait for an
   explicit click. Multi-GB downloads on every page open is hostile.
4. On load failure: surface the error verbatim in the load card. No
   model loaded; transcript stays empty; Send disabled.

### VRAM guardrail

If `model.vramMB > 5000`, show a small amber pill near the dropdown
reminding that the 16 GB hardware tier is tight when paired with
Three.js (per `CLAUDE.md` "Hardware baseline doctrine"). Non-blocking.

## Conversation lifecycle

### Handle creation

On the first Send after one of:
- model load,
- Clear conversation,
- Apply system prompt,

the page calls:

```ts
handle = await engine.createConversation(modelId, {
  systemPrompt,
  maxContextTokens: model.contextLength,
});
```

Subsequent turns reuse the same handle.

### Per-turn flow

1. Append the user message to a local
   `messages: ChatMessage[]` array and render it immediately.
2. Render an empty assistant bubble in `streaming` state.
3. Iterate `engine.chatCompletion(handle, [{ role: "user",
   content }], cfg)`:
   - `cfg` carries the sampling parameters from the Settings panel.
   - On first chunk: record TTFT (`now - sendTime`); switch the
     assistant bubble out of `pending` state.
   - On each chunk: append text, increment a per-turn token counter,
     re-render markdown incrementally (debounced to ~30 Hz).
   - On stream end: capture `CompletionStats` (or compute decode
     tok/s locally from `outputTokens / (now - firstChunkTime)`).
4. Append the completed assistant message to `messages`.
5. Auto-save: write `exportConversation(handle)` to IndexedDB.

### Stop / abort

The Stop button calls the conversation-level abort path
(verified at implementation time against
`core/conversation-pool.ts` — likely `handle.abort()` or a passed
`AbortSignal`). On abort:

- The partial assistant message is kept in the transcript with a
  `[stopped]` marker.
- The user message stays.
- Auto-save fires with the partial assistant message included so a
  reload reproduces the visible transcript.

### Persistence

- **Storage:** IndexedDB via the shipped `IndexedDBConversationStore`,
  database `webllm-chat`, store `conversations`, key `chat:current`.
  Single slot.
- **Save trigger:** after each completed assistant response, and after
  abort.
- **Save payload:** `{ modelId, systemPrompt, settings, messages,
  blob: exportConversation(handle), savedAt }`.
- **Restore on load:** if `chat:current` exists and `modelId` is still
  registered, render a restore card above the transcript:
  > Resume conversation with `<model>` (`<N>` turns, last active
  > `<relative-time>`)? **[Resume]** **[Discard]**
  - **Resume** → load model → `importConversation(modelId, blob)` →
    populate transcript from `messages`.
  - **Discard** → delete the slot, proceed to empty state.
- **Invalidation:**
  - Clear conversation, model change, system-prompt Apply →
    `disposeConversation` + clear the slot.
  - `IncompatibleConversationError` on import → toast
    "Saved conversation incompatible with current model registration"
    + Discard.

## Metrics

All values are computed in the page; no telemetry / live-server
plumbing required.

### Context-window utilization

- **Used:** read from the conversation handle's reported KV-occupancy
  field (verified at implementation time; if the handle does not
  expose it directly, derive from cumulative tokenized message length
  via `engine.tokenize` per turn).
- **Max:** `model.contextLength`.
- **Bar thresholds:** ≥80% amber, ≥95% red.
- **Update cadence:** after every completed turn and on Apply /
  Clear / Restore. Not during streaming (would add latency to the
  hot path for negligible UX gain).

### Per-turn

Captured directly from the stream:

- **TTFT:** `firstChunkTime - sendTime`.
- **Decode tok/s:** `outputTokens / (lastChunkTime - firstChunkTime)`.
- **Total wall:** `lastChunkTime - sendTime`.
- **Output tokens:** chunk-level token count from `StreamChunk`.

### Session totals

Plain JS counters:

- Turns (assistant replies completed).
- Cumulative output tokens.
- Average decode tok/s = `sum(outputTokens) / sum(decodeWall)`.

### Live chart

A 20-turn rolling sparkline of decode tok/s. Hidden behind a
`[chart ▾]` toggle in the status strip; collapsed by default.
Implemented as a single `<canvas>` (~120×30) with a hand-rolled
single-`path` render — no chart library dependency. Auto-rescales
y-axis to `0 .. 1.1 * max(window)`.

## Errors & edge cases

- **`ConversationContextOverflowError`** — surface inline in the
  transcript:
  > Context full (`<used>/<max>`). Clear the conversation to continue.
  > **[Clear conversation]** **[Export & clear]**
  No silent truncation, no auto-summarize.
- **Model load failure** — error surfaced verbatim in the load card,
  no model loaded, Send disabled.
- **Abort during prefill** (i.e., before first chunk) — treated as a
  no-op turn: user message stays with a `[stopped, no reply]` marker,
  no assistant bubble.
- **Reload mid-generation** — auto-save fires only on completed turns
  *and* on abort. Refreshing during streaming without an explicit
  abort drops the in-flight turn entirely (history stays monotone:
  every saved user message has a corresponding assistant reply or a
  `[stopped]` marker).
- **Two tabs open** — last writer wins on `chat:current`. Documented
  as a known limitation in `README.md` updates that ship with the
  page.
- **`IncompatibleConversationError` on resume** — see Persistence /
  Invalidation above.

## Testing

### Unit tests (`tests/`, Bun)

- `tests/chat-page-markdown.test.ts` — markdown render of a
  `<think>...</think>` block produces a collapsible region with the
  thinking text isolated and the post-thinking visible answer
  rendered as the bubble body.
- `tests/chat-page-context-bar.test.ts` — bar threshold math:
  `<80%` neutral, `[80, 95)` amber, `>=95%` red; numeric formatting
  ("1,842 / 4,096 — 45%").
- `tests/chat-page-persistence.test.ts` — round-trip a fixture
  conversation through `exportConversation` → JSON → IndexedDB →
  `importConversation` against a stub engine, verifying schema and
  that history + sampler settings survive.

### Manual browser smoke

Documented checklist in `docs/CHAT_PAGE.md`:

1. `make smoke-serve`. Open `http://localhost:8031/chat.html?v=1`.
2. Pick **TinyLlama 1.1B Chat (Q4_0)**. Verify load card progresses
   to 100%.
3. Set system prompt to "You answer in one sentence." Click Apply.
4. Send "Say hi." Verify TTFT, decode tok/s, total wall, output
   tokens populate; context bar advances; transcript shows reply.
5. Send 2 more turns. Verify context bar grows, sparkline appears
   when expanded, session totals update.
6. Reload the page. Verify the restore card appears with the
   correct turn count and timestamp. Click Resume.
7. Send turn 4. Verify TTFT is materially lower than turn 1 with the
   same total token count would have been (KV reuse evidence).
8. Click Clear conversation. Verify transcript empties, context bar
   resets to 0%, IndexedDB slot is empty (DevTools → Application).
9. Switch model to **Qwen3 0.6B**. Confirm dialog. Verify thinking
   toggle appears in Settings; with thinking on, send a question and
   verify the `<think>` block renders as a collapsed region.

### CI

No new CI jobs in v1. The page is exercised manually + via the unit
tests above. An `agentchrome`-driven CI smoke can layer on later.

### Regression guards

- `make smoke-bench` and `make dashboard-serve` paths must remain
  unchanged.
- The existing `real-model.html` / `real-model-page.js` flow must
  continue to pass its smoke checklist.
- `make checkall` (fmt + lint + typecheck + test) must pass.

## Implementation skill handoff

The implementation plan **must invoke the `frontend-design` skill**
when implementing `chat.html` / `chat.css` and the visual layer of
`chat-page.js`. This spec fixes the component anatomy and behavior
(Sections "UI structure" through "Errors & edge cases");
`frontend-design` owns:

- Visual treatment: typography, color, spacing, density.
- Dark mode (light-mode is acceptable as a v1 default if dark-mode is
  cheap to add; not required).
- Message-bubble visual style for user / assistant / system / error /
  stopped variants.
- Thinking-block collapse animation.
- Status-strip pill / progress-bar styling and amber / red color
  ramp.
- Sparkline visual style.
- Load card and restore card visual treatment.

`frontend-design` must not change: which buttons exist, which metrics
are surfaced, the catalog filter, persistence semantics, or the
conversation invalidation rules.

## Out-of-scope follow-ups (filed for later)

- **Named conversation list / sidebar** — would extend persistence
  from single-slot to multi-slot using the same `exportConversation`
  format.
- **Custom GGUF file picker** — non-trivial loader work; revisit only
  if the registered fleet is insufficient for some workflow.
- **Auto-summarize on context overflow** — explicit non-goal in v1
  (silent truncation is hostile to a debugging-grade page); could
  ship later as an opt-in setting.
- **Concurrent multi-conversation tabs** — gated by engine-level
  per-conversation queueing (TODO §11 follow-up #4).
- **agentchrome-driven CI smoke** — the manual checklist becomes a
  recorded script.
