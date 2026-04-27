# §22 prefill-tiling matrix — 2026-04-27

Phase-0 recommended tile size: **128** (see `00-phase0-diagnostic.txt`).
Branch: `feat/prefill-tiling-22`.

## Matrix

| Cell | Model | Prefill | Tile | Prefill (ms) | Decode (tok/s) | Status |
|------|---|---:|---:|---:|---:|---|
| 1 | tinyllama-1.1b-chat-q4_0       | 256 |   0 |  289 | 101.9 | works (control) |
| 2 | tinyllama-1.1b-chat-q4_0       | 256 | 128 |  524 | 106.9 | works — Δ TTFT = +81.3%, Δ decode = +4.9% (faster) |
| 3 | mistral-7b-instruct-v0.3-q4ks  | 512 |   0 |   —  |   —   | aborts (`ggml_tallocr_alloc: not enough space … node_510`, confirms §20) |
| 4 | mistral-7b-instruct-v0.3-q4ks  | 512 | 128 | 4368 |  33.6 | works (unblock) |
| 5 | qwen3-8b-iq3m                  | 512 | 128 | 4518 |  16.2 | works (unblock) |

All visible answers were coherent, on-topic English. TinyLlama tile=0 vs
tile=128 produced bytewise-identical output (equivalence holds at the
sampling level).

## Decision-rule evaluation

- TinyLlama TTFT regression at tile=128: **+81.3%** — gate (≤3%): **FAIL**.
- TinyLlama decode regression at tile=128: **+4.9% (improvement, not regression)** — gate (≤3%): pass directionally.
- Mistral-7B unblocks at tile=128: **YES**.
- Qwen3-8B unblocks at tile=128: **YES**.

**Decision: ship gated.**

The unblock works on both 7B and 8B causal LMs — Mistral-7B Q4_K_S and
Qwen3-8B IQ3_M complete prefill-512 cleanly at tile=128 where tile=0
aborts with the same compute-graph allocation failure §20 documented.
However, applying tile=128 to TinyLlama at prefill-256 nearly doubles
TTFT (289 ms → 524 ms, +81.3%), violating the ≤3% regression gate for
the small-model fast path. The tiling overhead is the cost of N extra
graph dispatches per prefill — it pays for itself when each chunk
unblocks the model, but it punishes models that already fit a single
graph.

The right ship posture is therefore the gated path: keep the default at
tile=0 so the TinyLlama short-prompt baseline is preserved, and have
callers opt into tile=128 (or auto-engage it per-model) when they
target 7B+ models or long prefill. The smoke page already wires the
`?prefillTile=N` URL knob (Task 3) and `eval/perf.ts` accepts
`--prefill-tile <n>` (Task 4), so opt-in is in place. A future task can
land a per-model auto-default keyed on parameter-count or a probe of
the per-graph buffer cap.
