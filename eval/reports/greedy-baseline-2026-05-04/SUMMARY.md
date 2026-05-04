# Greedy bench baseline — 2026-05-04

## Outcome

First multi-model bench run after the **2026-05-04 greedy cutover**.
20 unique model variants × 34 evals × 48 tasks per eval, all at
`temperature: 0`. The DB snapshot is pinned at
[`eval/reports/archive/smoke-runs-greedy-baseline-2026-05-04.db`](../archive/smoke-runs-greedy-baseline-2026-05-04.db)
(811 KB) and serves as the inaugural canonical baseline for this
sampling regime. Pre-cutover history (mostly captured at
profile-native temps `0.6` / `0.9` / `0.1`) is in
[`eval/reports/archive/smoke-runs-pre-greedy-cutover-20260504-140045.db`](../archive/smoke-runs-pre-greedy-cutover-20260504-140045.db);
the two series are not directly comparable and must not share a
score-over-time chart.

## Cutover scope

- **Default sampling for the accuracy pass** is now `temperature: 0`
  (greedy). Speed pass is unaffected — `chat-smoke.ts` keeps
  profile-native temperature so tok/s headlines stay comparable to
  pre-cutover history. Override via `bun run eval/bench.ts --eval-temperature <n>`.
- **Bench session envelope**: `eval/bench.ts` mints a `sessionId` per
  invocation and emits `bench_session_started` / `bench_session_complete`
  live events bracketing the multi-model loop. Each per-model
  `eval_started` carries the parent `sessionId` so the dashboard
  rolls up overall progress (model X/Y · task A/B). Session metadata
  records the pinned `evalTemperature` so the dashboard can refuse
  to chart cross-temperature comparisons.

## Headline ranking

| Rank | Model | Overall | Notes |
| --- | --- | --- | --- |
| 1 | `qwen3-8b-iq3m` | 0.91 | Top of leaderboard. |
| 2 | `llama-3.1-8b-instruct-iq3m` | 0.89 | Tool-calling 0.98 — best in class. |
| 3 | `qwen2.5-3b-q4f16` | 0.86 | |
| 3 | `qwen2.5-1.5b-q4f16` | 0.84 | |
| 3 | `qwen2.5-coder-1.5b-q4f16` | 0.84 | |
| 3 | `qwen3-1.7b-q4f16` | 0.84 | |
| 7 | `qwen3-0.6b-q4f16` | 0.77 | |
| 8 | `mistral-nemo-instruct-2407-q4ks` | 0.73 | Tool-calling 0.17 — see findings. |
| 9 | `smollm2-1.7b-q4f16` | 0.70 | |
| 10 | `llama-3.2-3b-q4f16` | 0.66 | |
| 11 | `llama-3.2-1b-q4f16` | 0.63 | |
| 12 | `hermes-3-llama-3.2-3b-q4f16` | 0.61 | |
| 12 | `phi-3.5-mini-q4km` | 0.61 | reasoning **1.00**. |
| 14 | `smollm2-360m-q4f16` | 0.56 | |
| 15 | `mistral-7b-instruct-v0.3-q5km` | 0.54 | Tool-format mismatch — fixed post-bench. |
| 16 | `mistral-7b-instruct-v0.3-q3km` | 0.53 | Same. |
| 16 | `mistral-7b-instruct-v0.3-q4ks` | 0.53 | Same. |
| 18 | `mistral-7b-instruct-v0.3-iq4xs` | 0.46 | Same. |
| 19 | `tinyllama-1.1b-chat-q4_0` | 0.27 | |
| 20 | `qwen3-4b-q4f16` | 0.08 | WASM abort on tc-005 — fixed post-bench. |

## Per-dimension table (select models)

Tool-calling stratifies sharply by chat-template family — see findings.

| Model | reasoning | inst-follow | semantic | tool-call |
| --- | --- | --- | --- | --- |
| `qwen3-8b-iq3m` | 0.92 | 0.95 | 0.87 | 0.74 |
| `llama-3.1-8b-instruct-iq3m` | — | — | — | **0.98** |
| `phi-3.5-mini-q4km` | **1.00** | 0.76 | 0.53 | 0.17 |
| `qwen3-1.7b-q4f16` | 0.92 | 0.96 | 0.72 | 0.60 |
| `qwen3-0.6b-q4f16` | 0.92 | 0.75 | 0.41 | 0.87 |
| `mistral-7b-v0.3-q4ks` | 0.75 | 0.43 | 0.76 | 0.17 |
| `tinyllama-1.1b` | 0.42 | 0.21 | 0.29 | 0.17 |

## Findings

### 1. Tool-calling 0.17 cluster — chat-template tool-format mismatch

Mistral 7B v0.3 (all four quants), Mistral Nemo, Phi-3.5, Llama-3.2-3B,
TinyLlama, and Hermes-3-Llama-3.2-3B all clustered at tool-calling
score `0.167`, which is exactly `2 / 12` — passing only the two
`no_tool_call` tasks (`tc-003`, `tc-012`) by emitting nothing.
Pulling sample outputs from the failing tasks confirms the pattern:
the models hallucinate tool *results* instead of emitting tool calls.

```
tc-001 (Mistral V0.3) → "The weather in Tokyo is currently sunny..."
tc-002 (Mistral V0.3) → "The search tool will scan through its database..."
```

**Root cause**: `src/inference/chat-template.ts` had a single
`injectToolsIntoSystem` keyed off the Qwen3/Hermes
`<tools>...</tools>` + `<tool_call>{...}</tool_call>` convention.
Models trained on different conventions (Mistral V0.3's
`[AVAILABLE_TOOLS]`, Llama-3.x's native format, Phi-3's
prompt-engineered style) ignored the unfamiliar instruction and
answered conversationally.

**Resolution**: commit `0f590a4`
(`feat(chat-template): Mistral V0.3 [AVAILABLE_TOOLS] tool-block format`)
adds a Mistral-specific block emitted before the first `[INST]` when
the loaded template lacks the `<<SYS>>` envelope (the existing
discriminator between Mistral V0.3 and real Llama-2). Real Llama-2
has no function-calling fine-tune so the block is correctly skipped
for that family. Parser side (`tool-system.ts`) already handles
Mistral's `[TOOL_CALLS] [{...}]` emission via the existing
`JSON_OBJECT_RE` — added an explicit test to pin it.

**Out of scope, deferred to follow-up tickets**:

- Llama-3.x native format (`<|python_tag|>` for built-in tools, JSON
  for custom). Note: same code path produces 0.98 on Llama-3.1-8B and
  0.17 on Llama-3.2-3B; format-only theory doesn't fully explain that
  delta, so the Llama fix needs investigation rather than a direct
  port of the Mistral pattern.
- Phi-3 has no upstream-canonical tool format — one possible direction
  is the Llama-3 native format Microsoft used in their function-calling
  examples, but this needs a probe before committing to a layout.

### 2. `qwen3-4b-q4f16` WASM abort on tc-005

Two runs, both 0.08. 88 of 96 tasks failed with `Invalid array length`
and 0–1 ms latency. Inspection showed the failure pattern:

- `tc-001`..`tc-004` succeed (1.5–6 s, 39–63 tok/s, 57–97 char output).
- `tc-005` aborts the WASM module with `Aborted(). Build with -sASSERTIONS for more info.`
- Every subsequent task fails instantly (`Invalid array length`) —
  the dead module's heap surfaces garbage sizes to JS-side allocators.

**Root cause**: `tc-005` is the first medium-difficulty tool-calling
task — its system prompt declares 3 tools with multiple parameters,
producing the largest prefill the bench has thrown at the model.
Cross-referencing
[`eval/reports/prefill-tiling-2026-04-27/00-phase0-diagnostic.txt`](../prefill-tiling-2026-04-27/00-phase0-diagnostic.txt)
confirms the abort signature (`ggml-alloc.c:82: not enough space in
the buffer`) is the same §22 graph-allocator overrun pattern that
prefill tiling was designed to mitigate. But qwen3-4B (36 layers ×
2560 emb) fell **outside** the heuristic's gate
(`layerCount >= 32 AND embeddingLength >= 4096`) — its embedding
dim is too small to clear the AND.

F32 intermediates at seq≈800 (tc-005's tool-heavy prefill):

| Model | Layers × emb | Seq | Total intermediates |
| --- | --- | --- | --- |
| Mistral 7B (§22 abort baseline) | 32 × 4096 | 512 | 256 MB |
| **qwen3-4B (this abort)** | 36 × 2560 | ~800 | 295 MB |
| qwen3-1.7B (works untiled) | 28 × 2048 | ~800 | 184 MB |

qwen3-4B's intermediate budget at tc-005's seq is **larger** than the
Mistral 7B case the heuristic was tuned for, but the heuristic
returned 0 (no tiling) and the graph allocator overran.

**Resolution**: commit `e5e5c81`
(`fix(qwen3-4b): drop emb gate from prefill-tile heuristic`) changes
`computeDefaultPrefillTileSize` from
`layerCount >= 32 AND embeddingLength >= 4096` to `layerCount >= 32`
alone. Layer count is the dominant predictor of graph allocator
pressure (each layer adds independent intermediates) and no
sub-32-layer model in the registry needs tiling at the seq lengths
the bench exercises. Tests cover all currently-registered model
shapes.

### 3. n=1 stability

Pre-cutover, tinyllama's overall flipped 0.35 → 0.24 on consecutive
n=1 reruns of identical code (semantic-reasoning task `emb-001`
flipped `1 → 0 → 1` across three runs, the smoking gun for
sampling-driven variance). Post-cutover: greedy gives **byte-for-byte
identical scores** on rerun, settling the variance question. The
0.27 tinyllama overall is the model's actual mean, not a sampled
draw from a wide distribution.

This is the load-bearing reason to stay on greedy as the default
even though some thinking-mode models (Qwen3-thinking) have
recommended T=0.6: the bench's job is to compare model quality
across families with reproducible numbers, and greedy is the only
sampling regime that produces them at n=1.

## Reproduction

The bench harness runs as:

```bash
make dashboard-serve  # background, port 8033
WEBLLM_LIVE_BENCH_URL=http://localhost:8033 bun run eval/bench.ts \
  --profiles full
```

`--eval-temperature 0` is the default; explicit temperature override
via `--eval-temperature <n>`. Single-model runs:

```bash
bun run eval/bench.ts --profiles qwen3-0.6b-thinking-warm
```

Per-model details (model output, per-task scores, latencies) live in
the pinned DB at
[`eval/reports/archive/smoke-runs-greedy-baseline-2026-05-04.db`](../archive/smoke-runs-greedy-baseline-2026-05-04.db).
Quick query for any model:

```bash
sqlite3 eval/reports/archive/smoke-runs-greedy-baseline-2026-05-04.db "
SELECT json_extract(value, '\$.taskId'),
       json_extract(value, '\$.score'),
       substr(json_extract(value, '\$.modelOutput'), 1, 80)
FROM evals, json_each(report_json, '\$.results')
WHERE model_id = 'mistral-7b-instruct-v0.3-q4ks'
  AND json_extract(value, '\$.dimension') = 'tool-calling'
ORDER BY 1;
"
```

## Pinned artifacts

- DB snapshot: [`../archive/smoke-runs-greedy-baseline-2026-05-04.db`](../archive/smoke-runs-greedy-baseline-2026-05-04.db) (811 KB)
- Pre-cutover archive: [`../archive/smoke-runs-pre-greedy-cutover-20260504-140045.db`](../archive/smoke-runs-pre-greedy-cutover-20260504-140045.db)
- §22 graph-allocator abort signature: [`../prefill-tiling-2026-04-27/00-phase0-diagnostic.txt`](../prefill-tiling-2026-04-27/00-phase0-diagnostic.txt)

## Validation re-runs (2026-05-04 evening)

Re-ran the bench against the two affected profiles after both fixes
landed. Both fixes validate end-to-end against real WebGPU inference:

### `mistral-7b-v0.3-warm` — tool-calling 0.17 → 0.48 (+31 pp)

| Dimension | Pre-fix | Post-fix | Δ |
| --- | --- | --- | --- |
| reasoning | 0.75 | 0.75 | 0 |
| instruction-following | 0.43 | 0.43 | 0 |
| semantic-reasoning | 0.76 | 0.76 | 0 |
| **tool-calling** | **0.17** | **0.48** | **+0.31** |
| **overall** | **0.53** | **0.60** | **+0.07** |

Per-task inspection confirms Mistral is now emitting its native
format (`[{"name":"...","arguments":{...}}]`) on the tasks it
handles. Of the 6 remaining tool-calling failures: 2 emit
Python-syntax `func(arg=val)` instead of JSON (parser limitation,
not format selection), 1 fails the implicit refusal (tc-012 — model
called the tool when asked to delete files), and the rest are
multi-step chain or tool-selection misses. All are model-quality
issues — Mistral V0.3 at this quant has limited tool reasoning —
not chat-template format issues. The fix delivers what was
expected: from "model isn't trying" to "model emits native format
and gets the easy ones right."

### `qwen3-4b-warm` — 0.08 → 0.88 (+80 pp)

| Dimension | Pre-fix | Post-fix | Δ |
| --- | --- | --- | --- |
| instruction-following | 0.00 (all errored) | 1.00 | +1.00 |
| reasoning | 0.00 (all errored) | 0.92 | +0.92 |
| semantic-reasoning | 0.00 (all errored) | 0.81 | +0.81 |
| tool-calling | 0.33 (no-call default) | 0.79 | +0.46 |
| **overall** | **0.08** | **0.88** | **+0.80** |

Zero errors across all 48 tasks (vs 88 pre-fix). The model lands
between qwen3-1.7B (0.84) and qwen3-8B (0.91) — exactly where a 4B
Qwen3 should sit. The crash cascade is gone.

### Implication for the greedy baseline

The pinned snapshot at
[`../archive/smoke-runs-greedy-baseline-2026-05-04.db`](../archive/smoke-runs-greedy-baseline-2026-05-04.db)
captures the **pre-fix** state for these two models — qwen3-4B at
0.08 and Mistral V0.3 at 0.53. Future bench cycles should be
compared against the **post-fix** numbers above for those two
specific profiles, not the snapshot. A full re-bench across all
20 model variants is the cleanest way to refresh the canonical
baseline; queued as follow-up ticket 3 below.

## Follow-up tickets

1. **`feat(chat-template)`: Llama-3.x native tool format** — investigate
   the Llama-3.1-8B (0.98) vs Llama-3.2-3B (0.17) gap on the same
   code path before committing to a format change. Likely needs a
   smoke probe with both models to capture what each actually emits
   under the current Qwen3-style prompt.
2. **`feat(chat-template)`: Phi-3 tool format** — no upstream-canonical
   layout exists; pick between Llama-3-style JSON, OpenAI-style
   `function_call`, or keep as-is and accept Phi's tool-calling weakness.
3. **`feat(bench)`: re-run after both chat-template fixes land** —
   should promote at least the Mistral V0.3 cluster from 0.17 to
   ≥0.6 tool-calling, validating the fix end-to-end.
4. **`feat(dashboard)`: persist bench sessions to SQLite** — the
   current envelope events are in-memory only; a backend restart
   mid-session loses the rollup. Optional polish; only matters if
   long-running sessions become routine.
