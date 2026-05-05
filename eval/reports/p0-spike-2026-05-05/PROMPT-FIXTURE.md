# P0 Spike — Prompt fixture

**Model:** TinyLlama-1.1B-Chat-v1.0 Q4_0
**Local path:** `./smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf`
**Prompt:** `"The capital of France is"`

## Token IDs

```
PROMPT_TOKEN_IDS = [1, 450, 7483, 310, 3444, 338]
EXPECTED_PARIS_ID = 3681
```

Per-token breakdown (from `llama-cli --verbose-prompt`):

| ID | Decoded |
|----|---------|
| 1 | `<s>` (BOS) |
| 450 | ` The` |
| 7483 | ` capital` |
| 310 | ` of` |
| 3444 | ` France` |
| 338 | ` is` |
| 3681 | ` Paris` (next-token target) |

## Host validation

Host `llama-cli` 2026-05-04 at `~/Repos/llama.cpp/build/bin/Release`
(branch `webllm-browser-patches` tip `fc1f81242`) generates ` Paris`
greedy:

```
$ llama-cli -m ./smoke-test/models/tinyllama-1.1b-chat-q4_0.gguf \
    -p "The capital of France is" -n 1 --temp 0 --no-display-prompt
 Paris
```

" Paris" as a continuation-token (i.e. after " is") tokenizes to a
single ID 3681; this is what the WASM-side bridge must produce as
top-1 logit after a single `webllm_decode` call.

## Notes for the spike harness

- TinyLlama declares `add_bos_token=true`, so the BOS (id 1) MUST be
  the first token in the batch. `webllm_decode(ctx, [1, 450, 7483,
  310, 3444, 338], past_len=0)` is the canonical single-batch call.
- Past-len starts at 0; the decode populates KV positions [0, 6).
- Logits flag is set on the LAST token only (`batch.logits[5] = 1`)
  per the bridge implementation in Task 4.

Recorded 2026-05-05 from upstream `llama-cli` against the canonical
TinyLlama Q4_0 GGUF. Used by `smoke-test/p0-spike.html` as a hard-
coded fixture so P0 does not depend on `llama_tokenize` (P1's job).
