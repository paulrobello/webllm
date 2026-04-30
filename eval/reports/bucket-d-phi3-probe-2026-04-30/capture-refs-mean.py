# pyright: reportMissingImports=false
"""
Bucket D Phi-3.5 — mean-pool reference-embedding capture for
Phi-3.5-mini-instruct. Mirrors capture-refs.py but uses mean-pool
across token positions instead of last-token. Required because
last-token pool on Phi-3.5 has high cosine anisotropy (paraphrases
and unrelated text both cluster ~0.99 in PyTorch f16); mean-pool
recovers semantic distinguishability per probe-mean-pool.py.

Tokenize identically to webllm (add_special_tokens=False + manual
EOS append), forward, take post-final-norm hidden state, mean across
all N positions, L2-normalize.

Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs-mean.py
Writes: phi-3.5-mini-ref-mean.json — {model, captured_with, pooling, eos_id, fixtures}.
"""
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"
POOLING = "mean"
TOLERANCE = 1e-3

print(f"Loading {MODEL_NAME}…", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",
)
model.eval()

EOS_ID = tokenizer.eos_token_id
print(f"EOS id: {EOS_ID}", file=sys.stderr)

vectors = []
with torch.no_grad():
    for text in inputs:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or ids[-1] != EOS_ID:
            ids = ids + [EOS_ID]
        input_ids = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # hidden_states[-1] is post-final-norm last layer output, shape [1, N, E].
        # Mean across the N token positions (dim=1), then drop the batch dim.
        pooled = out.hidden_states[-1][0].mean(dim=0)  # [E]
        v = pooled / pooled.norm(p=2)
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
out_path = HERE / "phi-3.5-mini-ref-mean.json"
out_path.write_text(
    json.dumps(
        {
            "model": MODEL_NAME,
            "captured_with": f"transformers (torch.float16 cpu); add_special_tokens=False + manual EOS append (id {EOS_ID}); mean-pool across N token positions",
            "pooling": POOLING,
            "eos_id": EOS_ID,
            "fixtures": fixtures,
        },
        indent=2,
    )
)
print(f"Wrote {out_path}", file=sys.stderr)
