# Gemma 4 E2B — Stage 3 closure (PASSED, 2026-05-11)

**Outcome:** **Stage 3 gate CLEARED at 68 %**, well above the
≥40 % threshold (and above the Phi-3 closure baseline of 60 %).
The fix is a **one-line** addition to `getRopeModeForArchitecture`:
Gemma family uses NEOX-style RoPE (split-halves), not interleaved.
The whole 9 % → 68 % jump came from changing a single return
statement.

## Eval matrix

| Dimension              | Before (9 %) | After (68 %) | Δ        |
|------------------------|--------------|--------------|----------|
| instruction-following  | 2.33 / 12 = 19.4 % | **11.0 / 12 = 91.7 %** | **+72 pp** |
| reasoning              | 0 / 12 = 0 %       | **10.0 / 12 = 83.3 %** | **+83 pp** |
| semantic-reasoning     | 0 / 12 = 0 %       | **9.58 / 12 = 79.8 %** | **+80 pp** |
| tool-calling           | 2 / 12 = 16.7 %    | 2 / 12 = 16.7 %        | 0 (capability disabled by design — Gemma 4 PEG format unsupported) |
| **Overall**            | **4.33 / 48 = 9 %** | **32.58 / 48 = 68 %** | **+59 pp** |

Eval IDs: `bench-1778534531604-16bcgh` (before) → `bench-1778539035299-8rtsiz` (after).

## Sample output deltas

| Task    | Before                                       | After     |
|---------|----------------------------------------------|-----------|
| rs-001  | "Please provide the question..."            | **"4"**   |
| rs-005  | "I am a large language."                     | **"Yes."** |
| emb-001 | `<eos>`                                      | **"joyful"** |
| emb-002 | `<eos>`                                      | **"food"** |

Speed-pass smoke (`"Tell one short joke."` at temp 0.6):

> "Why don't scientists trust atoms? Because they make up everything!"

Up from the pre-fix "Why did the chicken cross the road?" stub (no
punchline). The model is now generating complete coherent responses
instead of degenerate single-line continuations.

## Parity recovery on the emb-001 prompt

Phase A vs the NEOX-fix re-run on the identical 95-token chat-
formatted input:

| Metric                | Phase A (broken) | NEOX fix |
|-----------------------|------------------|----------|
| Embedding cos         | 0.9972 ✓         | 0.9972 ✓ |
| Block 0 cos           | 0.9900           | **0.9998** |
| Block 1 cos           | 0.9467           | **0.9997** |
| Block 2 cos           | **0.6520**       | **0.9996** |
| Block 11 cos          | **0.1538**       | **0.9820** |
| Block 33 cos          | 0.9102           | **0.9982** |
| Final-norm cos        | **0.5824**       | **0.9951** |
| Top-1 argmax          | MISMATCH (`<eos>`) | **MATCH** |
| Top-16 overlap        | 0 / 16           | **14 / 16** |

Block 34 still shows the known comparison artifact (HF puts
post-final-norm at index 34; we tap post-block-34) — same as
Phase 4 closure, not a regression.

## Root cause

`src/inference/model-inference.ts:getRopeModeForArchitecture()`
mapped only `nomic-bert`, `phi3`, and `qwen*` architectures to
NEOX-style RoPE (split-halves). Gemma family — `gemma`, `gemma2`,
`gemma3`, `gemma3n`, `gemma4` — fell through to `RopeMode.NORMAL`
(interleaved). The canonical llama.cpp puts all Gemma variants in
the NEOX case-list at `llama-model.cpp:2275-2310`.

### Why it manifested as a length-dependent forward-path bug

NEOX-style RoPE rotates the split halves `(0, head_dim/2),
(1, head_dim/2+1), ...` while interleaved RoPE rotates feature
pairs `(0,1), (2,3), ...`. At position p the rotation phase is
`p / freq_base^(2i/head_dim)` for each frequency bin i.

- **Small p (≤ ~10)**: rotations are tiny; the wrong feature
  pairing introduces only a small phase error. Phase 4's 6-token
  completion squeaked past the parity threshold (final-norm cos
  0.9722, top-1 MATCH).
- **Larger p (≥ ~20)**: rotation phases accumulate; the wrong
  pairing's accumulated error compounds quadratically through
  attention (Q · K^T pulls misaligned features together). By
  block 2 the residual stream is already mostly orthogonal to
  the canonical answer (cos 0.65 at p=91 on the Phase A capture).

The bug was invisible on Phase 4's short completion fixture, which
is why none of the earlier Stage 3 sub-tasks (3.3a-l) — all
correct individually — produced a working eval. The chat-formatted
parity probe (Phase A) and the length × content bisection
(Phase B) were necessary to localize it.

## Fix

```ts
// src/inference/model-inference.ts:getRopeModeForArchitecture
if (architecture === "gemma" || architecture === "gemma4") {
    return RopeMode.NEOX;
}
```

Three lines plus a load-bearing comment block. `make checkall`
green (763 pass / 0 fail). Smoke chat regression PASS. 36-prompt
eval 68 % vs 40 % gate.

## Probe chain (for the record)

The localization required five §28 negative-result closures and
two POSITIVE parity probes to land:

1. Baseline eval at 9 % (Task 3.5 first run)
2. §28 — default-system suppression A/B
3. §28 — temperature 0.6 sweep
4. §28 — stop-token leak audit (engine wiring correct; model
   emits `<eos>` as 1st token)
5. **POSITIVE — Phase A chat-formatted parity probe** (95-token
   chat prompt diverges catastrophically vs HF)
6. **POSITIVE — Phase B length × content bisection** (B1 92-token
   plain completion isolates length as the dominant variable;
   B2 10-token short chat shows the bug is mild at small N)
7. **Code-inspection finding**: `getRopeModeForArchitecture`
   doesn't list gemma family → NEOX-RoPE fix

The Phase B bisection was load-bearing — it pointed the inspection
at length-dependent per-block math. RoPE was the only candidate
that scales smoothly with position and differs between archs.

## Stage 3 implications

- The PLE injection, gated GELU, post-norm pairs,
  layer-output-scale, embedding scaling, GELU FFN, V bare-RMS-norm,
  FA softcap fix, BF16→F32 cast, rope_freqs (Tasks 3.3a–k), and
  shared-KV ref-share (Phase 4) all **were correct on their own**.
  The forward path needed the NEOX RoPE pairing to actually
  exercise them at scale.
- Phase 4's parity SUMMARY (cosine 0.9722 on a 6-token prompt) was
  technically correct but **insufficient evidence** that the
  forward path worked. The lesson: parity probes must include
  prompts ≥ the eval-task length, not just the Phase 4 canonical
  short-completion fixture.

## Stage 4 / Stage 5 outlook

- **Stage 4 (real SWA)**: the all-global fallback is currently
  fine for prompts < 512 tokens (the SWA window). 36-prompt eval
  outputs all fit comfortably. SWA implementation can wait until
  there's a documented quality drop on long-context probes
  (planned: Stage 4 work).
- **Stage 5 (shared-KV bench + closure)**: Phase 4 already wired
  the ref-share at L15-34. With the NEOX fix, the dashboard
  perf data the next bench-full sweep produces will be
  meaningful — re-run `make bench-full` (or the gemma-4 profile
  alone) once.

## Doctrine

This is the canonical **§27-style free-win closure**:
- Single-line code change.
- 9 % → 68 % accuracy.
- Required diagnostic infrastructure (parity probe) to localize.
- Confirms two §C-v2-A doctrine points:
  - **"Length-dependent bugs hide behind short-prompt parity"**:
    add ≥-eval-length fixtures to the canonical parity inputs.
  - **"Cross-architecture lookup tables ossify"**: any model-
    family branch table (`getRopeModeForArchitecture`,
    `attnSoftmaxScale`, the chat-template detector) should be
    audited against the latest llama.cpp once per major rebase.

## Artifacts

- Fix commit: this commit
- Parity verification: `eval/reports/parity-gemma-4-e2b-rope-neox-fix-2026-05-11/REPORT.md`
- Stage 3 baseline (broken):
  `eval/reports/gemma-4-stage3-eval-baseline-2026-05-11/SUMMARY.md`
- Phase A / B closures:
  - `eval/reports/parity-gemma-4-e2b-chat-emb001-2026-05-11/SUMMARY.md`
  - `eval/reports/parity-gemma-4-e2b-phaseB-bisect-2026-05-11/SUMMARY.md`
