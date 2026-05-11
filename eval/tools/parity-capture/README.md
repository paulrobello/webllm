# Parity Capture & Comparison Tooling

A reusable pipeline for diagnosing where webllm's forward pass diverges
from a HuggingFace `transformers` reference, layer-by-layer. Designed
to localize correctness bugs to the first divergent block instead of
guessing at architectural fixes.

## When to use this

When a webllm-loaded model produces output that's syntactically valid
but semantically wrong (e.g. degenerate repetitions, mixed-script
noise, locked low-entropy logits), and the obvious architectural
pieces have been wired but the symptom persists. The parity capture
narrows the search from "somewhere in the 35-layer stack" to "the
specific block where webllm's residual stream diverges from HF's."

Not the right tool for:
- Tokenization bugs (use a single-prompt encode comparison instead).
- Sampling/temperature issues (greedy + same logits is the cleaner test).
- Performance regressions (use `make smoke-bench` / `make bench-*`).

## Workflow at a glance

1. **Pick a model + prompt.** Write the test fixture to
   `inputs.json` — same prompt that webllm's smoke test uses.
2. **Capture HF reference.** Run `capture-hf-ref.py` to produce
   per-layer residual stream + final logits for the prompt.
3. **Capture webllm output.** Run the webllm side (a tapped
   `forwardSingle`) on the same tokenized prompt; save to the
   same JSON shape.
4. **Compare.** Run `compare.py` to compute cosine + L2 per layer
   and emit a markdown report. The first layer where cosine drops
   below the configured threshold pinpoints the bug.

## Standard format

Both sides write JSON conforming to this schema:

```jsonc
{
  "model": "unsloth/gemma-4-E2B-it",            // canonical HF id
  "captured_with": "transformers" | "webllm",   // source identifier
  "captured_at": "2026-05-11T18:24:00Z",        // ISO 8601 UTC
  "torch_dtype": "float32" | "bfloat16" | null, // null on webllm side
  "prompt": "The capital of France is",         // raw text input
  "input_token_ids": [2, 651, 6037, 576, 6082, 603],
  "n_layer": 35,                                // model layer count
  "n_embd": 1536,                               // residual stream dim
  "per_layer_residual_last_token": [
    [0.123, -0.045, ...],                       // layer 0 output for last token (length n_embd)
    [0.241, -0.072, ...],                       // layer 1
    // ... one entry per layer
  ],
  "final_norm_hidden_last_token": [0.301, ...], // post-`output_norm` for last token
  "logits_top16": {                             // top-16 logits indices + values for last token
    "ids": [12345, 6789, ...],
    "values": [12.4, 11.9, ...]
  }
}
```

**Why last-token only:** the per-prompt-token residual would be O(N · L · E · 4) bytes (e.g. 73 × 35 × 1536 × 4 = 15 MB) — workable but noisy. Last-token suffices because the divergence at the final position carries the cumulative error from every earlier block. If finer localization is needed, expand to all-tokens in a later version.

**Why `logits_top16`:** full-vocab logits are 262144 × 4 = 1 MB per capture. Top-16 is enough to compute precision-at-k overlap with the reference and to verify the greedy argmax matches.

## Files

| File | Purpose |
|---|---|
| `README.md` | This document. |
| `inputs.json` | The prompt fixture(s) — checked in, model-agnostic. |
| `requirements.txt` | Python pinned deps for `uv run --no-project --with-requirements`. |
| `capture-hf-ref.py` | Generic HF reference capture. Reads inputs.json, writes `<run-dir>/hf-ref.json`. |
| `capture-webllm.html` | Browser harness that runs webllm's tapped forward + POSTs to `capture-server.ts`. (Phase 2) |
| `capture-server.ts` | Bun HTTP server that accepts webllm captures + writes `<run-dir>/webllm.json`. (Phase 2) |
| `compare.py` | Loads both JSONs, computes cosine + L2 per layer, writes `<run-dir>/REPORT.md`. (Phase 2) |

## Run directory convention

Each parity run lives under
`eval/reports/parity-<model-slug>-<date>/`. Inside:

- `hf-ref.json` — reference capture
- `webllm.json` — webllm capture
- `REPORT.md` — comparison output
- `PROBE.md` (optional) — design notes for THIS specific run

This mirrors the existing `eval/reports/<probe>/SUMMARY.md` pattern
so reports are discoverable from the standard places.

## Phase 1 status (2026-05-11)

- `inputs.json` — written; canonical Stage 3 prompt "The capital of France is".
- `requirements.txt` — written.
- `capture-hf-ref.py` — written; reusable across causal-LM
  architectures supported by `transformers`.
- `capture-webllm.html` / `capture-server.ts` / `compare.py` —
  deferred to Phase 2.

## Phase 1 usage example (Gemma 4 E2B)

```bash
# 1. Pre-fetch model (per CLAUDE.md: always use hfdownloader)
hfdownloader download unsloth/gemma-4-E2B-it

# 2. Capture reference (writes to RUN_DIR)
RUN_DIR=eval/reports/parity-gemma-4-e2b-2026-05-11
mkdir -p "$RUN_DIR"
uv run --no-project --with-requirements \
  eval/tools/parity-capture/requirements.txt \
  python eval/tools/parity-capture/capture-hf-ref.py \
  --model unsloth/gemma-4-E2B-it \
  --inputs eval/tools/parity-capture/inputs.json \
  --output "$RUN_DIR/hf-ref.json"

# (Phase 2) 3. Capture webllm:
#   open http://localhost:8031/parity-capture.html?model=gemma-4-e2b-it-q4km
#   the harness POSTs to capture-server.ts → $RUN_DIR/webllm.json
#
# (Phase 2) 4. Compare:
#   uv run --no-project --with-requirements requirements.txt \
#     python compare.py --run-dir "$RUN_DIR"
#   open "$RUN_DIR/REPORT.md"
```

## Quantization & thresholds

The HF reference runs at fp32 (or bf16 if VRAM-constrained). WebLLM
loads quantized weights (e.g. Q4_K_M). Per-layer error compounds
across 35 blocks; expected cosine at end-of-stack on an OTHERWISE
CORRECT implementation is around 0.95-0.99 depending on quant level.

**Guidance for picking thresholds:**

- **First-layer divergence:** cosine should be ≥ 0.99. The first
  block's input is the (scaled) embedding lookup — quantization of
  `token_embd` alone shouldn't drop cosine below 0.99 at any token.
- **Mid-stack drift:** cosine should decay roughly monotonically.
  A SUDDEN drop (> 0.05 between consecutive layers) is a bug
  signature, not quantization noise.
- **End-of-stack target:** cosine ≥ 0.95 for FP32-vs-Q4_K_M is the
  "implementation correct" gate for Stage 3 closure.

When the diff jumps at a specific layer, the bug is in either the
op sequence for that block, the inputs to that block (residual carry
from the previous block), or the weights themselves (rare; only if
GGUF tensor names mis-mapped).
