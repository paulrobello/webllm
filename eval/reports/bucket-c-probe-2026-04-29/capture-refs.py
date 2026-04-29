"""
Bucket C Phase 0 — reference-embedding capture for Qwen3-Embedding-0.6B.
Two-mode: document (raw) + query (instruction-prefixed).
Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: qwen3-embedding-0.6b-ref.json — {model, captured_with, pooling, instruction_prefix, fixtures}.
"""
import json
import math
import sys
from pathlib import Path

import sentence_transformers
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
POOLING = "last-token"  # qwen3.pooling_type = 3 per Stage 1 metadata.

# Exact runtime byte sequence pinned from Stage 1 README rendering of
# Python f-string `f'Instruct: {task_description}\nQuery:{query}'`.
# Real LF (U+000A) between the two lines; no space after `Query:`.
INSTRUCTION_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages "
    "that answer the query\n"
    "Query:"
)

print(f"Loading {MODEL_NAME}…", file=sys.stderr)
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

# Document mode: raw fixtures.
print("Encoding document mode…", file=sys.stderr)
doc_vecs = model.encode(inputs, normalize_embeddings=True)

# Query mode: instruction-prefixed fixtures.
print("Encoding query mode…", file=sys.stderr)
query_inputs = [INSTRUCTION_PREFIX + s for s in inputs]
query_vecs = model.encode(query_inputs, normalize_embeddings=True)

# Magnitude + dim sanity check.
EXPECTED_DIM = doc_vecs.shape[1]  # Locked from first document mode encoding.
print(f"Expected output dim (from doc-mode shape): {EXPECTED_DIM}", file=sys.stderr)

TOLERANCE = 1e-3

def _check(label, vec, idx):
    mag = math.sqrt(sum(float(x) * float(x) for x in vec.tolist()))
    if abs(mag - 1.0) > TOLERANCE:
        raise SystemExit(
            f"Magnitude assertion failed for {label} row {idx}: |v|_2 = {mag:.6f} (expected 1.0 ± {TOLERANCE})"
        )
    if len(vec) != EXPECTED_DIM:
        raise SystemExit(
            f"Dim assertion failed for {label} row {idx}: dim = {len(vec)} (expected {EXPECTED_DIM})"
        )

for i, v in enumerate(doc_vecs):
    _check("document", v, i)
for i, v in enumerate(query_vecs):
    _check("query", v, i)
print(
    f"All {len(doc_vecs) + len(query_vecs)} vectors passed magnitude + dim assertions.",
    file=sys.stderr,
)

# Build interleaved-by-row output.
fixtures = []
for i, raw in enumerate(inputs):
    fixtures.append({
        "row": i,
        "input": raw,
        "mode": "document",
        "vec": [float(x) for x in doc_vecs[i].tolist()],
    })
    fixtures.append({
        "row": i,
        "input": raw,
        "mode": "query",
        "vec": [float(x) for x in query_vecs[i].tolist()],
    })

out = {
    "model": MODEL_NAME,
    "captured_with": f"sentence-transformers {sentence_transformers.__version__}",
    "pooling": POOLING,
    "instruction_prefix": INSTRUCTION_PREFIX,
    "fixtures": fixtures,
}

out_path = HERE / "qwen3-embedding-0.6b-ref.json"
out_path.write_text(json.dumps(out))
print(
    f"Wrote {out_path} ({len(fixtures)} vectors, dim={EXPECTED_DIM})",
    file=sys.stderr,
)
