# P1 — Tokenizer Parity Fixture

**Date:** 2026-05-05
**Phase:** [P1 — Tokenizer migration](../../../docs/superpowers/specs/2026-05-05-tier3-llama-decode-migration-design.md#p1--tokenizer-migration)
**Plan:** [`docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md`](../../../docs/superpowers/plans/2026-05-05-tier3-p1-tokenizer.md)

## Purpose

Provide a deterministic, byte-exact diff target for the new
`LlamaTokenizer` (wrapping `llama_tokenize`) against the legacy TS
`Tokenizer.encode()` across every vocab class webllm currently
ships. Parity gate is **D-byte-exact**: each prompt's id list must
match exactly, on every vocab.

## Vocab inventory

webllm currently ships five distinct tokenizer classes, distinguished
by the GGUF metadata field `tokenizer.ggml.model` and (for `gpt2`
BPE) the pre-tokenizer regex selector `tokenizer.ggml.pre`. All five
have a small canonical model already on disk under
`smoke-test/models/`.

| Class | `tokenizer.ggml.model` | `tokenizer.ggml.pre` | Canonical model | Size |
|---|---|---|---|---|
| SPM (llama) | `llama` | — | `tinyllama-1.1b-chat-q4_0.gguf` | 638 MiB |
| GPT-2 BPE / llama-bpe | `gpt2` | `llama-bpe` | `llama-3.2-1b-q4f16.gguf` | 770 MiB |
| GPT-2 BPE / qwen2 | `gpt2` | `qwen2` | `qwen2.5-1.5b-q4f16.gguf` | 1017 MiB |
| GPT-2 BPE / qwen3 + qwen35 | `gpt2` | `qwen3`/`qwen35` | `qwen3-0.6b-q4f16.gguf` | 610 MiB |
| WordPiece (BERT) | `bert` | — | `bge-small-en-v1.5-q0f16.gguf` | 64 MiB |

Five vocabs × 200 prompts = **1000 byte-exact equality assertions**
per smoke run.

## Corpus structure

The 200-prompt corpus is organized into 5 categories of 40 prompts
each. Each category targets a distinct class of tokenizer
disagreement that is most likely to surface a bridge bug:

### Category 1 — Plain ASCII (40 prompts)

Routine text that ought to round-trip identically through any
tokenizer. Includes:
- 15 single-sentence prose snippets ("The capital of France is",
  "Hello, world!", quotes, dialog, …)
- 10 multi-sentence paragraphs (2-4 sentences each)
- 5 code snippets across Python, JavaScript, SQL, shell, JSON
- 5 punctuation-heavy strings (URLs without query, dotted IPs,
  email addresses, file paths)
- 5 numeric / arithmetic strings ("1+1=2", "3.14159", "1,000,000",
  "$100.00", "0xDEADBEEF")

### Category 2 — Unicode (40 prompts)

Text categories where pre-tokenizer regexes most often diverge:
- 10 emoji-bearing strings (mix of basic-plane and supplementary-
  plane: 🌞 U+1F31E, 𝕏 U+1D54F, ZWJ sequences like 👨‍👩‍👧)
- 10 CJK strings (5 Chinese, 3 Japanese with kana mix, 2 Korean
  Hangul)
- 5 RTL strings (Arabic, Hebrew)
- 5 combining-diacritics strings (`é` decomposed to `e + U+0301`,
  Vietnamese tone marks, IPA)
- 5 ligatures and special spacing (em-space, em-dash, ellipsis,
  fi/fl ligatures)
- 5 mixed-script strings (English + 中文, English + emoji + RTL)

### Category 3 — Special / chat-template tokens (40 prompts)

The category most likely to surface `addedTokens` registration
drift between legacy and bridge paths. Each prompt is a real
chat-template fragment a real model would see:
- 8 ChatML fragments (`<|im_start|>system\n...<|im_end|>`,
  user/assistant turns, multi-turn)
- 8 Llama-2 INST fragments (`<s>[INST] ... [/INST]`, with and
  without `<<SYS>>`)
- 6 Phi-3 fragments (`<|user|>...<|end|><|assistant|>`)
- 6 Qwen3 thinking-tag fragments (`<think>...</think>` with and
  without `<|im_end|>` framing)
- 6 BERT special-token fragments (`[CLS] ... [SEP]`, `[CLS] q
  [SEP] a [SEP]`, `[MASK]` infill)
- 6 mixed (tool_call, tool_response, endoftext, im_end framing
  combinations)

### Category 4 — Edge cases (40 prompts)

Inputs the legacy code-path explicitly handles with special-case
branches:
- 5 empty-or-whitespace (`""`, `" "`, `"\n"`, `"\t"`, 50-char
  all-whitespace string)
- 5 whitespace-run strings (`"\n\n\n\n\n\n\n\n"`,
  `"\t\t\t\t"`, `"   "`, mixed `"\n \t\n "`, leading whitespace)
- 5 trailing-whitespace strings (text + trailing `\n` × N)
- 5 single-char strings (1 ASCII letter, 1 digit, 1 punctuation,
  1 emoji, 1 CJK)
- 5 very-long strings (256 chars no spaces; 1000 chars repetitive;
  multiline 100-line input)
- 5 control-char strings (`\0`, `\x01`, `\x7f`, BOM `﻿`,
  ZWJ `‍`)
- 5 byte-fallback edge cases (4-byte codepoints split across
  potential token boundaries; raw UTF-8 sequences)
- 5 newline-only-with-context (`"a\nb"`, `"a\n\nb"`, `"\na"`,
  `"a\n"`, `"\n\n"`)

### Category 5 — Mixed real-world inputs (40 prompts)

Hybrid inputs that combine the above categories — represent typical
agent-prompt content:
- 8 URL strings with query strings, fragments, encoded chars
- 8 JSON snippets (well-formed, with nesting, with escaped strings)
- 8 HTML / Markdown fragments (anchor tags, fenced code blocks,
  Markdown links with Unicode)
- 8 multi-paragraph mixed prose-and-code (typical RAG context
  shape: text + ```code``` + text)
- 8 social-media style (mentions, hashtags, emoji, abbreviations,
  multilingual)

## Generation provenance

The fixture file `parity-fixture.json` is produced by
`generate-fixture.ts` (Bun). For each entry in the vocab inventory
the script:

1. Reads the GGUF buffer from disk.
2. Calls `ModelLoader.parseModel(buffer)` to extract a
   `TokenizerConfig`.
3. Constructs a legacy `Tokenizer(config)`.
4. Runs each of the 200 corpus prompts through `Tokenizer.encode()`.
5. Records `{ prompt, ids: number[] }` per prompt under that vocab
   key.

Output schema:
```json
{
  "prompts": ["...", "...", ...],
  "fixture": [
    {
      "vocab": "llama-bpe",
      "ggufUrl": "/models/llama-3.2-1b-q4f16.gguf",
      "expected": [
        {"prompt": "...", "ids": [1, 2, 3]},
        ...
      ]
    },
    ...
  ]
}
```

The script is idempotent — overwrites `parity-fixture.json` on each
run — and is the source-of-truth regenerator for any future corpus
adjustment.

## Parity gate (D-byte-exact)

A vocab passes only when **all 200 prompts** encode to byte-exact
id lists. The browser smoke harness logs the first 3 mismatches
per vocab for diagnosis if any prompt fails.

If parity is achievable for ≥197 prompts on a vocab but ≤3 prompts
have irreducible drift (e.g., a known legacy bug in WordPiece BERT
tokenization), the failing prompts may be **excluded** from the
gate with explicit per-prompt callout in `SUMMARY.md`. The
escape-valve is allowed for ≤3 prompts total across all vocabs;
more than that means the bridge has a real bug to fix.

## Output

- `parity-fixture.json` — the 200-prompt × 5-vocab baseline.
  Committed (force-add); regenerated only when the corpus changes
  and re-committed.
- `SUMMARY.md` — closure report, written after the smoke run in
  Task 7.
