"""
One-shot reference-embedding capture for the encoder-parity probe.
Run: uv run --no-project --with-requirements capture-refs-requirements.txt python capture-refs.py
Writes: nomic-ref.json, jina-ref.json — each a list of {"input": str, "vec": [float, ...]}.
"""
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

HERE = Path(__file__).parent
inputs = json.loads((HERE / "inputs.json").read_text())

models = [
    ("nomic", "nomic-ai/nomic-embed-text-v1.5", {"trust_remote_code": True}),
    ("jina",  "jinaai/jina-embeddings-v2-base-en", {"trust_remote_code": True}),
]

for tag, name, kwargs in models:
    print(f"Loading {name}…", file=sys.stderr)
    m = SentenceTransformer(name, **kwargs)
    vecs = m.encode(inputs, normalize_embeddings=True)
    out = [
        {"input": inputs[i], "vec": [float(x) for x in vecs[i].tolist()]}
        for i in range(len(inputs))
    ]
    out_path = HERE / f"{tag}-ref.json"
    out_path.write_text(json.dumps(out))
    print(f"  wrote {out_path} ({len(out)} vectors, dim={len(out[0]['vec'])})", file=sys.stderr)
