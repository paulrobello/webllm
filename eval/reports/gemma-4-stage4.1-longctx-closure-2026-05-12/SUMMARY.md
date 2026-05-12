# Stage 4.1 long-context closure — chat.html non-crash gate

**Date:** 2026-05-12 (EOS-5)
**Tip:** `770c65f` (post-FA-VEC clamp `9ea3bfc`).
**Status:** ✅ PASS — Gemma 4 E2B + FA chat path survives a 2,238-token
context (4.4× the 512 SWA window) and produces a coherent,
fact-correct one-sentence answer drawn from the prompt.

## Gate

Stage 4.1 final gate per the EOS-4 SUMMARY recommendation:

> Drive `engine.chatCompletion` greedy on a long (~1129-token) prompt
> via `chat.html`; verify non-crash + coherent output.

This is the cheap reachability gate the FA-VEC clamp unblocked. It is
*positive-but-weak* evidence the full Gemma 4 chain (per-layer mixed
GQA + shared-KV + PLE + RoPE + SWA + FA + chat template) holds
together at long context; it does NOT verify numerical parity vs HF —
that gate is still queued as Stage 4.3 (long-context regression
probe).

## Probe configuration

- Browser: agentchrome (headless: false), tab navigated to
  `http://localhost:8031/chat.html?model=gemma-4-e2b-it-q4km`.
- Model: `gemma-4-e2b-it-q4km` Q4_K_M (3.11 GB).
- Engine init: `WebLLM.init({ memoryBudget: 2_000_000_000,
  maxConversations: 4, worker: true })` per chat-page default.
- Load options: `flashAttn: true`, `contextLength: 4096` per
  `chat-models.js:60-80` (FA required for `createConversation`).
- Prompt: `smoke-test/inputs-longctx.json` (~1100 tokens of
  transformer/Gemma background) + a one-sentence direct question
  appended:
  ```
  Based on the passage above, what does the Gemma 4 family do with
  the first fifteen layers? Answer in one sentence.
  ```
- Sampling: chat default (greedy via temp=0 path is not the chat
  default — chat.html uses the registered `GEMMA4_DEFAULTS` profile).

## Observed result

```
context 2,238 / 4,096 — 55%
last: 42337ms TTFT · 35.5 tok/s · 43.3s · 33 tok
session: 1 turns · 33 tok · ⌀ 35.5 tok/s
```

Assistant output (verbatim):

> The first fifteen layers of the Gemma-4 family use sliding-window
> attention with a window size of 512 tokens and 8 query heads per
> layer.

Source span in the prompt:

> in Gemma 4, the first fifteen layers use sliding window attention
> with a window of 512 tokens and 8 query heads per layer

Verbatim factual match (modulo hyphen and "the" article). Coherent
ASCII English, ended on a stop token, exactly one sentence as
requested.

Console: clean (0 errors / 0 asserts / 0 RuntimeError). No
`Error: unreachable`, no `GGML_ASSERT`, no WebGPU device-loss.

## What this proves and does NOT prove

**Proves:**
- The FA-VEC `prefillTileSize=16` clamp shipped in `9ea3bfc` holds
  end-to-end through the chat.html path (the path that
  `createConversation` drives) at a context length that
  *substantially exceeds* the 512-token SWA window.
- Gemma 4's full forward chain (PLE + per-layer mixed-GQA + dual RoPE
  + shared-KV at layers 15–34 + Phase B per-layer SWA mask + FA-VEC
  attention + chat-template tokenization) survives a context where
  SWA layers must use the *banded* mask region, not just the
  full-causal region they exercised in the 46-token "Paris" smoke.
- Cross-boundary retrieval works *in this single case*: the answer is
  pulled from the early section of the prompt (~position 280 of
  ~2200), so the global layers can attend back across the SWA-windowed
  region without falling into nonsense.

**Does NOT prove:**
- Numerical parity vs an HF reference at long context. The Stage 4.3
  probe still owns that gate (1000-token greedy + cosine ≥ 0.95 vs
  HF; the 6-token parity-capture harness from 3.3l-P2 extended to
  long-context inputs).
- That all SWA-banded retrieval is correct — only that *this*
  question's answer survived. A retrieval from the *middle* of the
  passage where it sits past the SWA window of the model's later
  decode positions would be a stronger test (Stage 4.3 territory).
- Greedy-decode quality at scale across the 36-prompt eval. That is
  Stage 4.4 (eval re-gate at ≥68 % baseline).

## Performance footprint

- Prefill of ~2200 tokens via FA-VEC at `prefillTileSize=16`: 42.3 s.
  This is the cost of the clamp — VEC at q_tile=1 trades batch
  throughput for path eligibility. The §C-v2-A / §27 prefill-tiling
  history is the cleanup lever: bumping the upstream
  `ggml-webgpu-shader-lib.hpp:734` VEC `ne[1] < 20` ceiling (or
  TILE/SUBGROUP_MATRIX LDS rework) recovers the throughput. Tracked
  as TODO §EOS-4 item 2 ("Upstream patch follow-up: bump VEC `ne[1]`
  ceiling").
- Decode at 35.5 tok/s with `pastLen ≈ 2200` is consistent with the
  ~10 tok/s `real-model.html` headline at much shorter context being
  decoder-bound, not memory-bound (KV grows; FA reads dominate at long
  context but stay reasonable).

## Stage 4 status after this gate

| Stage | Status |
|---|---|
| 4.0 mask-feasibility probe | CLOSED 2026-05-11 |
| 4.1 per-layer mask construction (Phase A + B) | CLOSED 2026-05-11 |
| 4.1 final gate — chat.html long-context non-crash | **CLOSED 2026-05-12 (this report)** |
| 4.2 Gemma 2 alternating-period SWA derivation | queued |
| 4.3 long-context regression probe (HF cosine ≥ 0.95) | queued |
| 4.4 eval re-gate (≥68 %) | queued |

## Repro

```bash
# Terminal 1
make smoke-serve   # port 8031

# Terminal 2 (headed Chrome — headless gpu less reliable on Apple)
agentchrome connect --launch
agentchrome navigate "http://localhost:8031/chat.html?model=gemma-4-e2b-it-q4km"
# Wait for "Loaded Gemma 4 E2B Instruct (Q4_K_M)" in the page.

# Paste the long-context prompt with the appended question,
# click Send. See /tmp/chat-longctx-driver2.js for the driver
# script used here.
```

## Artifacts

- `chat-output.txt` — raw assistant reply
- `metrics.json` — TTFT / tok-s / context / tokens-out

See sibling files in this directory.
