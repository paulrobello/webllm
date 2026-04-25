"""Generate HF-reference WordPiece encodings for tests/wordpiece-golden.test.ts.

Run:
    uv run --with transformers --with torch scripts/generate-bert-golden.py

Loads `bert-base-uncased` (the tokenizer Snowflake Arctic-Embed-s uses
verbatim, modulo the GGUF phantom-space rewrite which is *bijective* on
ids), encodes a set of probe strings exercising the cases that bit us
during the encoder bring-up, and writes the (text -> [ids]) pairs to
tests/fixtures/bert-wordpiece-golden.json.

The TS-side test (tests/wordpiece-golden.test.ts) loads the GGUF vocab
extracted by scripts/extract-bert-vocab.ts, runs each probe through
the local Tokenizer, and asserts byte-for-byte parity with this JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer  # type: ignore[import-untyped]

OUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "bert-wordpiece-golden.json"

# Probe strings chosen to exercise the bugs we hit + a few sanity cases.
# Each one should round-trip cleanly through both HF and our local Tokenizer.
PROBES: list[str] = [
    # Single common words — the simplest case.
    "happy",
    "joyful",
    "hello",
    # Multi-word + punctuation.
    "the dog runs",
    "Hello, World!",
    # Subword splits — "useful" is in vocab as ▁use + ful (continuation).
    "useful",
    # Casing: bert-base-uncased lowercases "Happy" → "happy".
    "Happy",
    "HAPPY",
    # Accent stripping (NFKD).
    "café",
    "naïve",
    # Numbers and digits.
    "2024",
    "version 3.1",
    # Edge: single character.
    "a",
    # Empty string after CLS/SEP framing — tokenizes to [CLS] [SEP] only.
    "",
    # Unknown / OOV — random unicode that won't be in vocab.
    # Skipping for now: HF bert-base-uncased actually splits these into
    # bytes via its `do_basic_tokenize` rules and the result depends
    # on the exact unicode_normalize / whitespace path. Re-enable once
    # wpBasicTokenize covers more edge cases.
    # Multi-token sentences.
    "the quick brown fox jumps over the lazy dog",
    # Mixed punctuation that exercises wpIsPunctuation's $ ^ ` carve-outs.
    "$100 ^ `code`",
]


def main() -> None:
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    cases = []
    for text in PROBES:
        # add_special_tokens=True wraps with [CLS] ... [SEP], matching our
        # encodeWordPiece(). truncation=False so we don't lose trailing tokens
        # on the longer probes.
        ids = tok.encode(text, add_special_tokens=True, truncation=False)
        cases.append({"text": text, "ids": ids})
    out = {
        "model": "bert-base-uncased",
        "cases": cases,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {len(cases)} cases to {OUT_PATH}")


if __name__ == "__main__":
    main()
