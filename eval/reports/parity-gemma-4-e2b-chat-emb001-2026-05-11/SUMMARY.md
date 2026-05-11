# Gemma 4 E2B — chat-formatted parity probe (POSITIVE, 2026-05-11)

**Outcome:** Real per-block forward-path divergence found. The forward
pass diverges catastrophically starting at block 2 when fed a chat-
formatted eval prompt (95 tokens), while Phase 4's parity on a
6-token completion prompt (`"The capital of France is"`) passes
cleanly. This is the **root cause** of the 9 % Stage 3 eval score:
the model isn't broken at the architecture-spec level — Phase 4
verified prefill on the short prompt — but **something in the
per-block forward path is wrong as a function of either prompt
length, special-token content, or both**.

## Headline

| Metric                                | Phase 4 (6-token completion) | Phase A (95-token chat) |
|---------------------------------------|------------------------------|-------------------------|
| Embedding output cosine               | n/a                          | 0.9972                  |
| Block 0 residual cosine               | ≥0.99                        | 0.9900 ⚠                |
| Block 1 residual cosine               | ≥0.99                        | **0.9467** ⚠            |
| Block 2 residual cosine               | ≥0.99                        | **0.6520** ⚠⚠           |
| Block 9 residual cosine               | ~0.98                        | **0.4888** ⚠⚠           |
| Block 11 residual cosine              | ~0.98                        | **0.1538** ⚠⚠⚠          |
| Block 34 residual cosine              | 0.28 (artifact)              | **0.0753**              |
| Final-norm hidden cosine              | **0.9722** ✓                 | **0.5824** ⚠⚠           |
| Top-1 argmax                          | MATCH (id 9079, "Paris")     | **MISMATCH**            |
| Top-16 logits overlap                 | 13/16                        | **0/16**                |

WebLLM's top-1 sits at **id 1 (`<eos>`) with logit 18.2** — ~9 logits
of margin over the next contender. HF's top-1 is id 3672 (logit
9.13). WebLLM's residual stream by end-of-stack is essentially
**orthogonal** to HF's (cosine 0.075).

## Setup

Phase A probe in the [`forwardSingle` parity tap] hypothesis chain
queued at the close of the stop-token audit:

- **Fixture**: the **exact** token ID sequence WebLLM sends to the
  engine on eval task `emb-001` ("Which word is most similar in
  meaning to 'happy': sad, joyful, tired, heavy?" with the
  semantic-similarity system prompt). Captured via direct
  `engine.chatCompletion` browser probe; 95 tokens including the
  full chat-formatted scaffolding (`<bos><|turn>system\n…<turn|>\n
  <|turn>user\n…<turn|>\n<|turn>model\n`).
- **HF reference**: `transformers` `unsloth/gemma-4-E2B-it` at
  float32, CPU, forward pass with `output_hidden_states=True` and
  `use_cache=False`. Bypasses the HF tokenizer by feeding the raw
  ID sequence (added a `--input_token_ids` path to
  `capture-hf-ref.py`).
- **WebLLM**: `gemma-4-e2b-it-q4km` (unsloth Q4_K_M) via
  `ModelInference.forwardWithLayerTaps`, fresh forward pass (no
  cache write).
- **Tooling**: extended `eval/tools/parity-capture/capture-hf-ref.py`
  to honor an `input_token_ids` field in inputs.json; existing
  `parity-capture.html` already supports `?inputIds=` URL param;
  `compare.py` unchanged.

## Diagnosis

1. **Embedding lookup is correct** (cos 0.9972) → `opGetRows +
   opScale(sqrt(n_embd))` works, the BF16→F32 cast (Task 3.3j)
   isn't suspect.
2. **First measurable divergence appears at block 0 / 1** (cos
   ~0.99 → 0.95), suggesting a small per-block error that
   compounds. Block 0 / 1 are both SWA layers per the iSWA
   `slidingWindowPattern` (`13 SWA → 1 full → 5×(4 SWA + 1
   full)`).
3. **Catastrophic divergence at block 2** (cos 0.95 → 0.65). Same
   SWA layer type as block 1, so the block-2-specific drop is
   compounding from the block-1 residual rather than a new
   pathology unique to block 2.
4. **Block 11 reaches near-orthogonal** (cos 0.15) before any
   full-attention layer or shared-KV layer fires. The bug is
   **within the SWA layer math** at scale.
5. **Block 14 (first full layer) doesn't recover** — final-norm
   cosine 0.58. The shared-KV remap (Phase 4) is consuming a
   already-corrupted residual stream and can't restore parity.
6. **WebLLM's logit 18.2 on `<eos>` (id 1)** is so high because
   the corrupted residual stream resolves at `output_norm + lm_head`
   to a peaked distribution on the `<eos>` token. The model
   emits `<eos>` immediately, decode stops, eval scores 0.

**The bug is in the per-block forward path and triggers as a
function of prompt structure (length OR content OR both).** Phase
4's 6-token completion prompt didn't exercise the failure path;
the 95-token chat-formatted prompt does.

## Hypothesis for the next probe (Phase B)

Bisect on prompt length / content:

- **(B1) Length-only probe**: feed a 95-token *completion* prompt
  (no special tokens). If parity holds → bug is content-dependent,
  likely tied to the special tokens `<|turn>`/`<turn|>`/`\n`
  appearing in the prompt. If parity breaks → bug scales with
  position / RoPE / SWA-window math.
- **(B2) Content-only probe**: feed a 6-token chat-formatted
  prompt (e.g. just `<bos><|turn>user\nHi<turn|>\n<|turn>model\n`).
  If parity holds → bug is length-dependent. If parity breaks →
  bug is in how special tokens / chat boundaries propagate
  through PLE / attention.
- **(B3) Per-block internal taps**: add taps inside block 1 (the
  first block where cosine drops below 0.99) to localize the bug
  to Q-proj / K-proj / V-proj / QK-norm / RoPE / attention /
  post-attn-norm / FFN / post-FFW-norm / PLE injection /
  layer-output-scale.

Recommend running **(B1) and (B2) in parallel** as cheap text
fixtures — both are 30-second captures, reuse all existing tooling
with new inputs.json files. Whichever bisects the bug picks the
direction for (B3).

## Artifacts

- HF reference: `hf-ref.json` (35-layer × 1536-dim float32 dump)
- WebLLM capture: `webllm.json` (matching shape)
- Compare: `REPORT.md` (auto-generated by compare.py)
- Inputs: `inputs.json` (raw token IDs + observation notes)
- Capture-server log: `/tmp/capture-server.log` (transient)

## Doctrine

This is the **first POSITIVE** in the Task 3.5 chain — a real
forward-path bug isolated to per-block math triggered by chat-
formatted (or long, or special-token-heavy) prompts. Phase B
narrows further. This finding alone justifies the heavier probe
spend: previous §28 closures pointed at the model behavior but
left the question "is the engine actually correct on chat
prompts" open. Answer: **no, it isn't.**
