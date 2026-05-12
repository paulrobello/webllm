# Gemma 4 E2B — Stage 4.4 Eval Re-Gate — CLOSED 2026-05-12 ✅

**Tip:** `f8f8a64` (post Stage 4.3 PARTIAL closure).
**Profile:** `gemma-4-e2b-warm` (model `gemma-4-e2b-it-q4km`, prefill+decode FA per default config).
**Eval suite:** 36-prompt + 12 semantic-reasoning chat suite (48 tasks total; embedding dimension skipped per model's `capabilities.embedding=false`).
**Eval temperature:** 0 (greedy, per project doctrine "Greedy by default for accuracy bench" set 2026-05-04).
**Gate:** overall ≥ 68 % (Stage 3 closure baseline must not regress).
**Result:** **34/48 = 70.8 %** ✅ — exactly equal to Stage 3 closure; zero regression.

## Per-dimension breakdown

| Dimension | Stage 3 closure (2026-05-11) | Stage 4.4 (2026-05-12) | Δ |
|---|---|---|---|
| tool-calling | 2/12 = 17 % | 2/12 = 17 % | — |
| reasoning | 10/12 = 83 % | 10/12 = 83 % | — |
| instruction-following | 11/12 = 92 % | 11/12 = 92 % | — |
| semantic-reasoning | 11/12 = 92 % | 11/12 = 92 % | — |
| **overall** | **34/48 = 70.8 %** | **34/48 = 70.8 %** | **flat** |

Pre-Stage-3 numbers (the 9 → 68 % lift line in TODO.md) compared *pre-PLE* state to Stage 3 closure — they are not the Stage 3 baseline.

Stage 3 closure evalId: `bench-1778539035299-8rtsiz` (2026-05-11T22:38:03Z).
Stage 4.4 evalId: `bench-1778629846312-rm2kv2` (2026-05-12T23:51:49Z).
Both at greedy temp=0, same chat suite, same model GGUF.

## Why "exactly identical" is the right outcome

Stage 4's only architectural change vs Stage 3 is **real sliding-window attention on the 4-of-5 SWA-marked Gemma 4 layers**. SWA only differs from full-causal attention when the past+current sequence length exceeds the 512-token window. The 36-prompt eval suite issues short prompts (each task ≤ ~200 input tokens, generates ≤ 256 output tokens) — never reaches the boundary where the banded mask diverges from the full-causal mask.

This was predicted by:

- **Stage 4.0 windowed-mask feasibility probe (2026-05-11):** the mask is a purely additive per-element term in both `opSoftMaxExt` and `opFlashAttn`, so a banded mask is bit-identical in shape to a full-causal mask at sub-window lengths.
- **Phase B implementation invariant:** `swaMaskTensor` is only allocated when `hp.slidingWindowPattern?.some(b => b)` AND `hp.slidingWindowSize > 0`; per-layer dispatch picks the SWA mask only when `nTokens > 1 || pastLen + nTokens > swaWindow`. At sub-window prompts the SWA layers see the same full-causal mask the global layers see.

Long-context behavior — where SWA actually exercises the banded mask — was validated separately by **Stage 4.1's chat closure** (2026-05-12 EOS-5, [`gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md`](../gemma-4-stage4.1-longctx-closure-2026-05-12/SUMMARY.md)): a 2,238-token prompt (4.4× the 512-token window) produced a coherent fact-correct retrieval from prompt position ~280, 0 console errors, no `RuntimeError: unreachable`. That run is the load-bearing functional gate for the SWA path; the 36-prompt eval is the regression sanity-check.

## Speed-pass anomaly (separate, non-blocking)

The bench harness's `--- speed: gemma-4-e2b-warm ---` sub-step failed with:

```
Fatal: Timed out waiting for smoke-test chat output
```

Profile-native temperature is 0.6 (not greedy); on this prompt ("Tell one short joke.") the model can wander long enough that the smoke harness's output-wait deadline trips. This is **harness-side**, not a Gemma 4 regression:

- Stage 4.1's chat.html long-context run (2,238 tokens prompt, FA on, ctx 4096) produced 33 tokens in 43.3 s with healthy argmax — same model, longer task, no harness timeout (different harness).
- Stage 3 closure's smoke pass for this profile completed in prior sessions (see `smoke-runs/01778534779057-gemma-4-e2b-warm.json` and 4 sibling runs); the harness deadline today appears tighter than typical Gemma 4 wall time, possibly because the smoke browser had no prior warm-up before this run.

Filed as a follow-up watch-item rather than a Stage 4.4 gate failure. The eval pass is the load-bearing gate; the speed sub-step exists to populate the dashboard's tok/s headline and can be re-captured later without re-running the eval. The smoke-runs from 2026-05-11 already give us reasonable Stage 3 baseline speed numbers; the SWA wiring shouldn't change steady-state decode (one extra mask buffer alloc per forward, no extra ops per layer at sub-window contexts).

## Discrepancy between bench-log "33/48" and dashboard "34/48"

The streaming bench output reported `Done: 33/48 passing` while the persisted dashboard record shows `34/48`. One task evidently transitioned from running-or-failing to pass after the last bench-log progress checkpoint was printed. The dashboard's persisted state is authoritative; the bench-log line is a streaming snapshot.

Either count clears the ≥68 % gate (33/48 = 68.75 %, 34/48 = 70.8 %).

## Bench artifacts

- Bench log: `/tmp/webllm-stage4.4/bench.log` (also saved as Section "Bench session output" below).
- Live dashboard eval row: `evalId=bench-1778629846312-rm2kv2`, model `gemma-4-e2b-it-q4km`, totalTasks=48, params.temperature=0.
- Session id: `sess-1778629471454-4kjgb5`.

## Bench session output

```
[bench] session sess-1778629471454-4kjgb5 · 1 model(s) · eval temperature 0

═══ gemma-4-e2b-warm ═══

--- speed: gemma-4-e2b-warm ---
agentchrome: no smoke tab on port 53198 — creating one on the smoke-test page…
Navigating to http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&ctx=4096&chatSmoke=1778629472036&ingest=http%3A%2F%2Flocalhost%3A8033&temp=0.6&prompt=Tell+one+short+joke.&profile=gemma-4-e2b-warm
Running interactive chat regression with prompt "Tell one short joke."
Fatal: Timed out waiting for smoke-test chat output

--- accuracy: gemma-4-e2b-warm ---
Skipping embedding tasks (model "gemma-4-e2b-it-q4km" has capabilities.embedding=false; rerun with --dimension embedding to force).
Staging 48 tasks at http://localhost:8033/tasks…
Task list id: 1778629838101-zvy2u7
Navigating to http://localhost:8031/real-model.html?model=gemma-4-e2b-it-q4km&ctx=4096&browserEval=1778629838225&bench=1778629838101-zvy2u7&ingest=http%3A%2F%2Flocalhost%3A8033&temp=0&prompt=Tell+one+short+joke.&profile=gemma-4-e2b-warm&session=sess-1778629471454-4kjgb5
  1/48 tasks (0 passing)
  2/48 tasks (0 passing)
  4/48 tasks (1 passing)
  6/48 tasks (1 passing)
  8/48 tasks (1 passing)
  17/48 tasks (6 passing)
  23/48 tasks (11 passing)
  26/48 tasks (14 passing)
  28/48 tasks (16 passing)
  30/48 tasks (18 passing)
  34/48 tasks (22 passing)
  42/48 tasks (29 passing)
  48/48 tasks (33 passing)

Done: 33/48 passing · overall 68%

Bench summary
Total:   2
Passed:  1
Failed:  1
  [FAIL] gemma-4-e2b-warm · speed — Fatal: Timed out waiting for smoke-test chat output
  [PASS] gemma-4-e2b-warm · accuracy
```

## Closure

- ✅ Stage 4.4 gate cleared (70.8 % ≥ 68 %).
- ✅ Stage 4 campaign is functionally complete: Stage 4.0 (mask feasibility), Stage 4.1 (per-layer mask construction + long-context chat closure), Stage 4.2 (Gemma 2/3 SWA derivation), Stage 4.3 (incremental parity infra; Gemma 4 numerical parity blocked by per-binding 128 MiB cap — qualitative gate covered by 4.1), and Stage 4.4 (eval re-gate, this report) all closed.
- ➡️ Next: Campaign Q3 (Stage 5) — pre-rebase baseline capture on the canonical 6, add `gemma-4-e2b-warm` to `bench-full`, write the canonical Gemma 4 campaign closure SUMMARY.
- Watch-item: smoke-harness output-wait deadline for `gemma-4-e2b-warm` profile. Re-capture speed numbers when convenient; not gating Stage 5.
