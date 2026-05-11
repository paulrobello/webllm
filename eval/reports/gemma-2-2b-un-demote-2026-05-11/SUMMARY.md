# Gemma 2 2B — UN-DEMOTED back into canonical fleet (2026-05-11)

## Outcome

`gemma-2-2b-q4f16` restored to `SMOKE_PROFILE_SETS.full` in
`eval/smoke-profiles.ts`. Eval at greedy temp 0 lands **60 %
overall (29/48)**, exactly clearing the Phi-3 closure baseline
gate. 92 % reasoning, 72 % instruction-following, 61 %
semantic-reasoning, 17 % tool-calling (capability=false, expected
low).

Supersedes the demote at
[`eval/reports/gemma2-demote-2026-05-01/SUMMARY.md`](../gemma2-demote-2026-05-01/SUMMARY.md).

## Root cause of demote (now fixed)

The original demote SUMMARY listed five candidate causes
(post-attention LayerNorm, post-FFW LayerNorm, alternating SWA,
attention/output logit soft-capping, tied output↔embedding
weights). Empirically the load-bearing ones turned out to be:

| Cause                                           | Fix                                                                 | Commit       |
| ----------------------------------------------- | ------------------------------------------------------------------- | ------------ |
| NEOX-style RoPE for Gemma family               | `getRopeModeForArchitecture` returns `RopeMode.NEOX` for gemma2.    | `c8c8447`    |
| Attention logit soft-cap (`attn_logit_softcapping=50`) | New `op_tanh` WASM binding; FA shader native + manual softmax path. | `f2735d5` + `5d1aba4` |
| Final logit soft-cap (`final_logit_softcapping=30`)    | `softCap` helper wraps lm_head output at all 4 sites.               | `5d1aba4`    |
| Input embed scaling by sqrt(n_embd)            | Extended gemma4-only branch to whole gemma family.                  | `31d53a5`    |
| GELU-parallel FFN activation (vs SwiGLU)       | Extended gemma4-only branch to whole gemma family.                  | `31d53a5`    |
| Soft-cap applied to *unscaled* qk             | Refactored manual softmax path: scale qk first, then softcap, then softmax with scale=1.0. | `31d53a5`    |

Not load-bearing on the eval gate:

- **Post-attention LayerNorm / post-FFW LayerNorm** — already
  wired generically via `lw.postAttentionNorm` / `lw.postFfwNorm`
  ternaries in Task 3.3d (commit `73f77df`, pre-Q1). Gemma 2's
  GGUF ships the same tensor names so they were loading correctly
  the whole time.
- **Tied output↔embedding weights** — lm_head fallback to
  `weights.tokEmb` when `weights.output` is null has been in place
  since before the demote. Gemma 2's GGUF has no `output.weight`
  (per `gemma2.cpp:39 TENSOR_DUPLICATED`); the fallback fired
  silently.
- **Alternating sliding-window attention** — Gemma 2 2B uses
  a 4096-token window which is wider than the 36-prompt eval's
  longest task. All-global fallback produces correct math at
  these lengths. SWA proper is queued as Campaign Q2 for longer-
  context Gemma fleet (Gemma 4 SWA layers + Gemma 2/3 windowed
  paths).

## Eval matrix (greedy temp 0, 48-task suite)

| Dimension              | Tasks | Passed | Score  | Avg latency |
| ---------------------- | -----:| ------:| ------:| -----------:|
| reasoning              | 12    | 11     | **92 %** | 796 ms      |
| instruction-following  | 12    | 9      | **72 %** | 1294 ms     |
| semantic-reasoning     | 12    | 7      | **61 %** | 498 ms      |
| tool-calling           | 12    | 2      | 17 %   | 1840 ms     |
| **Overall**            | **48**| **29** | **60 %** | —           |

Excluding tool-calling (model registered with
`capabilities.toolCalling: false`): **27/36 = 75 %**.

Eval row: `bench-1778542064055-onf00d` in `eval/reports/smoke-runs.db`.

## Speed (chat-smoke regression, profile gemma-2-2b-warm)

```
oneShot:     58.8 tok/s decode (prefill 298 ms, decode 323 ms, total 600 ms, 19 tokens)
interactive: 55.7 tok/s decode (total 650 ms, 19 tokens, finish=stop-token)
prompt:      "Tell one short joke."
response:    "Why don't scientists trust atoms? Because they make up everything! 😄"
```

Run record:
[`eval/reports/smoke-runs/01778542060669-gemma-2-2b-warm.json`](../smoke-runs/01778542060669-gemma-2-2b-warm.json).

Reference: the demote SUMMARY measured 54.0 tok/s on the broken
math (no soft-cap = saturated tanh = effectively-uniform attention,
which is paradoxically cheap). Post-fix 58.8 tok/s is slightly
faster despite the added soft-cap ops + GELU vs SwiGLU swap —
the GELU path uses fewer ops than SwiGLU's `gate * silu(up)`
fusion, and ggml-webgpu's tanh shader is GPU-native.

## Smoke probe (greedy temp 0 raw completion)

```
prompt:     "The capital of France is"
response:   " Paris."
tokens:     4 (finish=stop-token)
```

The pre-fix smoke produced 20 tokens of id 139 (whitespace) at
greedy temp 0. The cause was the soft-cap saturating: with the
old order `softCap(qk, 50) → softmax(qk * 1/sqrt(256))`, the
softcap saw qk magnitudes ~sqrt(256)=16× larger than reference,
so tanh saturated at ±1 and the cap output was ≈±50 regardless
of input. The corrected order `softCap(qk * 1/sqrt(256), 50) →
softmax(qk, scale=1)` matches `gemma2.cpp:110 +
ggml-cpu/ops.cpp:8232-8305`.

## Regression sanity-check

TinyLlama smoke (`tinyllama-1.1b-chat-q4_0`, greedy temp 0):

```
prompt:   "The capital of France is"
response: " I do not have access to the latest financial"
tokens:   10 (finish=max-tokens)
rate:     169.8 tok/s
```

Unchanged from pre-Q1 behavior — TinyLlama has no
`attnLogitSoftcap` so all softcap branches are skipped, and
`isGemmaFamily("llama")` is false so the embed-scale + GELU
paths are skipped. Non-Gemma fleet bit-identical.

## Commits

- `c8c8447` — fix(gemma): extend NEOX RoPE to gemma2/gemma3 (pre-Q1, 2026-05-11 EOS-12).
- `f2735d5` — feat(wasm): add op_tanh binding for Gemma 2 soft-capping.
- `5d1aba4` — feat(gemma2): wire attn + final-logit soft-capping.
- `bb73d4f` — refactor(gemma2): move softCap helper out of buildQKV JSDoc block.
- `31d53a5` — fix(gemma): extend embed-scale + GELU FFN to whole gemma family + scale-first softcap order.
- (this commit) — feat(eval): un-demote gemma-2-2b-warm + closure docs.

## Doctrine lessons

1. **Lookup-table extension audits pay off.** Three of the six
   gating fixes were "extend the `gemma4`-only branch to the
   whole Gemma family" or "add gemma2/gemma3 to the NEOX list".
   The post-NEOX 2026-05-11 EOS-12 audit (queued as Q1) caught
   the lot in one cycle. Per the Task 3.5 doctrine lesson:
   *cross-architecture lookup tables ossify* — audit them
   against the latest `llama-model.cpp` switch-statement after
   every major rebase.

2. **Soft-cap order is non-trivial.** The naïve order
   (`softcap → scale`) silently corrupts the attention
   distribution because the cap acts on the wrong-magnitude
   input. Reference order (`scale → softcap → softmax`) is
   recorded in `ggml-cpu/ops.cpp:8232-8233` as a one-line
   pre-divide: `scale /= logit_softcap`. The WebGPU host
   (`ggml-webgpu.cpp:1942-1944`) does the same trick. Both
   paths arrive at `softcap * tanh(scaled_qk / softcap)`.

3. **Demote candidates are usually plural.** The 2026-05-01
   demote SUMMARY enumerated five candidate causes and listed
   them with similar weight. The actual un-demote took fixes
   for **six** distinct items, three of which weren't even on
   the original list (NEOX-RoPE, embed-scale, GELU FFN, softcap
   order). Future demotes should default to "expect plural
   root causes" rather than "one of these five".

## Follow-ups

- **Campaign Q2 — Stage 4 real SWA** (queued in TODO.md). Now
  unlocks the Gemma 2 long-context regime + the Gemma 4 long-
  context path. Gemma 2's window (4096) is wider than every
  current eval task, so Q1 closure doesn't gate on it.
- **Campaign Q3 — Stage 5 bench + closure** for Gemma 4 + Gemma 2
  in the canonical fleet. Refresh canonical baselines, fold both
  into the dashboard's accuracy×speed scatter.
- **Tool-calling for Gemma 2.** Currently registered with
  `capabilities.toolCalling: false`. The 17 % tool-calling score
  (2/12) on a model trained for tool use suggests the parser
  leniency + Gemma 2's instruction-following could lift this with
  capability=true + a chat-template audit. Not gating closure;
  filed as a separate audit candidate.
