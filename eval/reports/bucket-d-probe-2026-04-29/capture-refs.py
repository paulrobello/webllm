# pyright: reportMissingImports=false
# torch / transformers load from the inline uv env via
# --with-requirements; they are not in any project venv that Pyright
# watches. Suppress at file level since this script is a self-
# contained side-run, not part of the TS project's typing surface.
"""
Bucket D Phase 0 — reference-embedding capture for Qwen3-8B (chat
model self-embedding). Tokenize identically to webllm (add_special_
tokens=False + manual EOS append), forward, take post-final-norm last-
token, L2-normalize. No sentence-transformers wrapper exists for the
chat base, so this uses raw transformers + output_hidden_states.

Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: qwen3-8b-ref.json — {model, captured_with, pooling, eos_id, fixtures}.
"""
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

MODEL_NAME = "Qwen/Qwen3-8B"
POOLING = "last-token"
TOLERANCE = 1e-3

print(f"Loading {MODEL_NAME}…", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # f16 ref is tight enough for cos >= 0.999 against IQ3_M WASM;
    # f32 would be ~32 GB and OOMs on the 16/32 GB hardware tier.
    torch_dtype=torch.float16,
    device_map="cpu",  # deterministic; small fixture batch
)
model.eval()

EOS_ID = tokenizer.eos_token_id
print(f"EOS id: {EOS_ID}", file=sys.stderr)

vectors = []
with torch.no_grad():
    for text in inputs:
        # Match webllm's pipeline exactly: encode without specials, then
        # manually append EOS so the tokenization matches what
        # engine.embed feeds to ModelInference.embed.
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[-1] != EOS_ID:
            ids = ids + [EOS_ID]
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # hidden_states[-1] is the post-final-norm last layer output.
        last = out.hidden_states[-1][0, -1, :]  # last token, [E]
        v = last / last.norm(p=2)
        vectors.append(v.tolist())

EXPECTED_DIM = len(vectors[0])
print(f"Expected output dim: {EXPECTED_DIM}", file=sys.stderr)

for i, v in enumerate(vectors):
    mag = math.sqrt(sum(x * x for x in v))
    if abs(mag - 1.0) > TOLERANCE:
        raise SystemExit(
            f"Magnitude check failed for row {i}: |v|_2 = {mag:.6f}"
        )
    if len(v) != EXPECTED_DIM:
        raise SystemExit(
            f"Dim check failed for row {i}: dim = {len(v)} (expected {EXPECTED_DIM})"
        )

print(f"All {len(vectors)} vectors passed.", file=sys.stderr)

fixtures = [
    {"row": i, "input": inputs[i], "mode": "document", "vec": vectors[i]}
    for i in range(len(inputs))
]
out_path = HERE / "qwen3-8b-ref.json"
out_path.write_text(
    json.dumps(
        {
            "model": MODEL_NAME,
            "captured_with": f"transformers (torch.float16 cpu); add_special_tokens=False + manual EOS append (id {EOS_ID})",
            "pooling": POOLING,
            "eos_id": EOS_ID,
            "fixtures": fixtures,
        },
        indent=2,
    )
)
print(f"Wrote {out_path}", file=sys.stderr)
